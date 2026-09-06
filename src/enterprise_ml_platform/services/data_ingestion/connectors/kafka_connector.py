"""Kafka streaming connector for the ingestion service."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import pandas as pd
import pyarrow as pa
import structlog

try:  # pragma: no cover - optional dependency
    from aiokafka import AIOKafkaConsumer as _AIOKafkaConsumer
except ImportError:  # pragma: no cover - optional dependency
    _AIOKafkaConsumer = None

from .base import AsyncDataConnector


class KafkaConnector(AsyncDataConnector):
    """Consume JSON object records from Kafka topics as data frames."""

    def __init__(
        self,
        topics: list[str],
        bootstrap_servers: str,
        group_id: str = "ml-platform",
        **consumer_kwargs: Any,
    ) -> None:
        self.topics = topics
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.consumer_kwargs = consumer_kwargs
        self._consumer: Any | None = None
        self._log = structlog.get_logger().bind(connector="kafka")

    async def connect(self) -> None:
        """Start the configured Kafka consumer."""
        if _AIOKafkaConsumer is None:  # pragma: no cover - dependency guard
            raise RuntimeError("aiokafka is required for Kafka connector")
        self._consumer = _AIOKafkaConsumer(
            *self.topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            enable_auto_commit=False,
            **self.consumer_kwargs,
        )
        await self._consumer.start()

    async def disconnect(self) -> None:
        """Stop the Kafka consumer when it is connected."""
        if self._consumer is not None:
            await self._consumer.stop()
            self._consumer = None

    async def read(self, **config: Any) -> AsyncIterator[pd.DataFrame]:
        """Consume JSON object messages in batches."""
        if self._consumer is None:
            raise RuntimeError("Connector not connected")
        batch_size = int(config.get("batch_size", 1000))
        timeout_ms = int(config.get("timeout_ms", 1000))
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if timeout_ms < 0:
            raise ValueError("timeout_ms cannot be negative")

        while True:
            messages = await self._consumer.getmany(
                timeout_ms=timeout_ms, max_records=batch_size
            )
            batch: list[dict[str, Any]] = []
            for records in messages.values():
                for message in records:
                    try:
                        payload = json.loads(message.value)
                        if not isinstance(payload, Mapping):
                            raise TypeError("Kafka messages must contain JSON objects")
                        batch.append(dict(payload))
                    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
                        self._log.warning("deserialisation failed", error=str(exc))
            if batch:
                yield pd.DataFrame.from_records(batch)

    async def get_schema(self) -> pa.Schema:
        """Return an empty schema because Kafka streams are unbounded."""
        return pa.schema([])
