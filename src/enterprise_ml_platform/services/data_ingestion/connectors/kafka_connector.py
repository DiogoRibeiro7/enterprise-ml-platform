"""Kafka streaming connector for the ingestion service."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pandas as pd
import pyarrow as pa
import structlog

try:  # pragma: no cover - optional dependency
    from aiokafka import AIOKafkaConsumer
except Exception:  # pragma: no cover - optional dependency
    AIOKafkaConsumer = None  # type: ignore

from .base import AsyncDataConnector


class KafkaConnector(AsyncDataConnector):
    """Consume records from Kafka topics as data frames."""

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
        self._consumer: AIOKafkaConsumer | None = None
        self._log = structlog.get_logger().bind(connector="kafka")

    async def connect(self) -> None:
        if AIOKafkaConsumer is None:  # pragma: no cover - dependency guard
            raise RuntimeError("aiokafka is required for Kafka connector")
        self._consumer = AIOKafkaConsumer(
            *self.topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            enable_auto_commit=False,
            **self.consumer_kwargs,
        )
        await self._consumer.start()

    async def disconnect(self) -> None:
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None

    async def read(
        self,
        batch_size: int = 1000,
        timeout_ms: int = 1000,
    ) -> AsyncIterator[pd.DataFrame]:
        """Consume messages in batches and yield as data frames."""

        if not self._consumer:
            raise RuntimeError("Connector not connected")

        while True:
            msgs = await self._consumer.getmany(
                timeout_ms=timeout_ms, max_records=batch_size
            )
            batch = []
            for _tp, records in msgs.items():
                for msg in records:
                    try:
                        batch.append(json.loads(msg.value))
                    except Exception as exc:  # pragma: no cover
                        self._log.warning("deserialisation failed", error=str(exc))
            if batch:
                yield pd.DataFrame(batch)

    async def get_schema(self) -> pa.Schema:
        # Schema inference is not implemented for Kafka streams.
        return pa.schema([])
