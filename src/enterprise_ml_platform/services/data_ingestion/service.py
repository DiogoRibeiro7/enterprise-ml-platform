"""Enterprise data ingestion service with streaming connectors."""

from __future__ import annotations

import hashlib
import io
import json
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
import structlog
from redis import asyncio as aioredis

from .connectors import (
    AsyncDataConnector,
    KafkaConnector,
    PostgresDataConnector,
    S3DataConnector,
)
from .validators import DataValidator

SourceType = Literal["s3", "postgres", "kafka"]


@dataclass(frozen=True)
class DataSource:
    """Configuration for a named data source."""

    name: str
    type: SourceType
    connection: dict[str, Any]
    quality_rules: list[dict[str, Any]] | None = None


@dataclass
class IngestionMetrics:
    """Metrics collected during one ingestion stream."""

    records_ingested: int = 0
    bytes_processed: int = 0
    batches: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    ended_at: float = 0.0

    @property
    def duration(self) -> float:
        """Return elapsed ingestion time in seconds."""
        return (self.ended_at or time.perf_counter()) - self.started_at

    @property
    def throughput(self) -> float:
        """Return records processed per second."""
        return self.records_ingested / self.duration if self.duration else 0.0


class DataIngestionService:
    """Coordinate typed connectors, validation, metrics, and optional caching."""

    def __init__(
        self,
        cache_config: Mapping[str, Any] | None = None,
        validator: DataValidator | None = None,
    ) -> None:
        self.cache_config = dict(cache_config or {})
        self.validator = validator or DataValidator()
        self._sources: dict[str, DataSource] = {}
        self._connectors: dict[str, AsyncDataConnector] = {}
        self._cache: aioredis.Redis | None = None
        self.metrics = IngestionMetrics()
        self._log = structlog.get_logger().bind(service="data_ingestion")

    async def initialize(self) -> None:
        """Construct the optional Redis client.

        ``redis.asyncio.from_url`` returns a client immediately; commands on
        that client are asynchronous, but the factory itself is not awaitable.
        """
        if self.cache_config.get("enabled"):
            redis_url = self.cache_config.get("redis_url")
            if not redis_url:
                raise ValueError("redis_url is required when caching is enabled")
            self._cache = aioredis.from_url(str(redis_url))
            self._log.info("cache initialised")

    async def shutdown(self) -> None:
        """Disconnect every connector and close the optional cache client."""
        for connector in self._connectors.values():
            await connector.disconnect()
        if self._cache is not None:
            await self._cache.aclose()
            self._cache = None

    def register_source(
        self,
        source: DataSource,
        connector: AsyncDataConnector | None = None,
    ) -> None:
        """Register ``source`` with a supplied or configuration-built connector."""
        if connector is None:
            if source.type == "s3":
                connector = S3DataConnector(**source.connection)
            elif source.type == "postgres":
                connector = PostgresDataConnector(**source.connection)
            elif source.type == "kafka":
                connector = KafkaConnector(**source.connection)
            else:  # pragma: no cover - protected by SourceType for typed callers
                raise ValueError(f"Unsupported source type {source.type}")
        self._sources[source.name] = source
        self._connectors[source.name] = connector

    async def ingest(
        self, name: str, read_config: Mapping[str, Any] | None = None
    ) -> AsyncIterator[pd.DataFrame]:
        """Stream validated batches from the registered source ``name``."""
        if name not in self._sources:
            raise KeyError(f"Unknown data source {name}")

        effective_read_config = dict(read_config or {})
        connector = self._connectors[name]
        source = self._sources[name]
        cache_key = self._cache_key(name, effective_read_config)

        if self._cache is not None:
            cached = await self._get_cache(cache_key)
            if cached is not None:
                self._log.info("cache hit", source=name)
                yield cached
                return

        self.metrics = IngestionMetrics()
        await connector.connect()
        try:
            async for batch in connector.read(**effective_read_config):
                self.metrics.batches += 1
                self.metrics.records_ingested += len(batch)
                self.metrics.bytes_processed += int(batch.memory_usage(deep=True).sum())
                validated = await self.validator.validate(batch, source.quality_rules)
                if self._cache is not None:
                    await self._set_cache(cache_key, validated)
                yield validated
        finally:
            await connector.disconnect()
            self.metrics.ended_at = time.perf_counter()
            self._log.info(
                "ingestion complete",
                source=name,
                records=self.metrics.records_ingested,
                duration=self.metrics.duration,
                throughput=self.metrics.throughput,
            )

    def _cache_key(self, name: str, config: Mapping[str, Any]) -> str:
        """Return a stable cache key for one ingestion configuration."""
        canonical_config = json.dumps(
            config, sort_keys=True, separators=(",", ":"), default=str
        )
        digest = hashlib.sha256(canonical_config.encode("utf-8")).hexdigest()
        return f"ingestion:{name}:{digest}"

    async def _get_cache(self, key: str) -> pd.DataFrame | None:
        try:
            data = await self._cache.get(key) if self._cache is not None else None
            if data:
                payload = data.decode("utf-8") if isinstance(data, bytes) else str(data)
                return pd.read_json(io.StringIO(payload), orient="records")
        except Exception as exc:  # pragma: no cover - cache failure
            self._log.warning("cache retrieval failed", error=str(exc))
        return None

    async def _set_cache(self, key: str, frame: pd.DataFrame) -> None:
        if self._cache is None:
            return
        try:
            ttl_seconds = int(self.cache_config.get("ttl_seconds", 3600))
            if ttl_seconds < 1:
                raise ValueError("ttl_seconds must be positive")
            await self._cache.setex(key, ttl_seconds, frame.to_json(orient="records"))
        except Exception as exc:  # pragma: no cover - cache failure
            self._log.warning("cache store failed", error=str(exc))
