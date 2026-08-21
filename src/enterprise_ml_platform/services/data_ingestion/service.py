"""Enterprise data ingestion service with streaming connectors."""

from __future__ import annotations

import hashlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

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


@dataclass
class DataSource:
    """Configuration for a data source."""

    name: str
    type: str  # ``"s3"``, ``"postgres"`` or ``"kafka"``
    connection: dict[str, Any]
    quality_rules: list[dict[str, Any]] | None = None


@dataclass
class IngestionMetrics:
    """Simple metrics collected during ingestion."""

    records_ingested: int = 0
    bytes_processed: int = 0
    batches: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    ended_at: float = 0.0

    @property
    def duration(self) -> float:
        return (self.ended_at or time.perf_counter()) - self.started_at

    @property
    def throughput(self) -> float:
        return self.records_ingested / self.duration if self.duration else 0.0


class DataIngestionService:
    """Coordinate data ingestion from multiple heterogeneous sources.

    The service wires together specialised connectors, a caching layer and a
    rule based validation engine.  Usage typically follows the pattern::

        service = DataIngestionService(cache_config={"enabled": True, "redis_url": "redis://localhost"})
        service.register_source(DataSource(name="transactions", type="s3", connection={"bucket": "my-bucket"}))
        await service.initialize()
        async for batch in service.ingest("transactions", {"prefix": "2024/"}):
            ...
        await service.shutdown()
    """

    def __init__(
        self,
        cache_config: dict[str, Any] | None = None,
        validator: DataValidator | None = None,
    ) -> None:
        self.cache_config = cache_config or {}
        self.validator = validator or DataValidator()
        self._sources: dict[str, DataSource] = {}
        self._connectors: dict[str, AsyncDataConnector] = {}
        self._cache: aioredis.Redis | None = None
        self.metrics = IngestionMetrics()
        self._log = structlog.get_logger().bind(service="data_ingestion")

    # ------------------------------------------------------------------
    # Lifecycle management
    async def initialize(self) -> None:
        """Initialise optional resources such as Redis cache."""
        if self.cache_config.get("enabled"):
            self._cache = await aioredis.from_url(self.cache_config["redis_url"])
            self._log.info("cache initialised")

    async def shutdown(self) -> None:
        """Tear down connectors and caches."""
        for connector in self._connectors.values():
            await connector.disconnect()
        if self._cache:
            await self._cache.close()
            self._cache = None

    # ------------------------------------------------------------------
    def register_source(self, source: DataSource) -> None:
        """Register a data source and create its connector."""
        self._sources[source.name] = source
        if source.type == "s3":
            connector = S3DataConnector(**source.connection)
        elif source.type == "postgres":
            connector = PostgresDataConnector(**source.connection)
        elif source.type == "kafka":
            connector = KafkaConnector(**source.connection)
        else:
            raise ValueError(f"Unsupported source type {source.type}")
        self._connectors[source.name] = connector

    # ------------------------------------------------------------------
    async def ingest(
        self, name: str, read_config: dict[str, Any]
    ) -> AsyncIterator[pd.DataFrame]:
        """Ingest data for the registered source ``name``.

        This method streams batches of data frames after applying validation
        rules and caching (if enabled).
        """

        if name not in self._sources:
            raise KeyError(f"Unknown data source {name}")
        connector = self._connectors[name]
        source = self._sources[name]
        cache_key = self._cache_key(name, read_config)

        if self._cache:
            cached = await self._get_cache(cache_key)
            if cached is not None:
                self._log.info("cache hit", source=name)
                yield cached
                return

        await connector.connect()
        try:
            async for batch in connector.read(**read_config):
                self.metrics.batches += 1
                self.metrics.records_ingested += len(batch)
                self.metrics.bytes_processed += batch.memory_usage(deep=True).sum()
                batch = await self.validator.validate(batch, source.quality_rules)
                if self._cache:
                    await self._set_cache(cache_key, batch)
                yield batch
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

    # ------------------------------------------------------------------
    def _cache_key(self, name: str, config: dict[str, Any]) -> str:
        """Return the cache key for one ingestion configuration.

        MD5 is used to shorten the configuration into a key, not to protect
        anything, so a collision costs a cache miss and nothing more.
        """
        digest = hashlib.md5(str(config).encode(), usedforsecurity=False).hexdigest()
        return f"ingestion:{name}:{digest}"

    async def _get_cache(self, key: str) -> pd.DataFrame | None:
        try:
            data = await self._cache.get(key) if self._cache else None
            if data:
                return pd.read_json(data, orient="records")
        except Exception as exc:  # pragma: no cover - cache failure
            self._log.warning("cache retrieval failed", error=str(exc))
        return None

    async def _set_cache(self, key: str, frame: pd.DataFrame) -> None:
        if self._cache is None:
            return
        try:
            ttl = self.cache_config.get("ttl_seconds", 3600)
            await self._cache.setex(key, ttl, frame.to_json(orient="records"))
        except Exception as exc:  # pragma: no cover - cache failure
            self._log.warning("cache store failed", error=str(exc))
