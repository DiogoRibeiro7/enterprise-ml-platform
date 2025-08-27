"""Kafka consumer utilities for streaming pipeline."""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional

try:  # pragma: no cover - optional dependency
    from aiokafka import AIOKafkaConsumer
except Exception:  # pragma: no cover
    AIOKafkaConsumer = None  # type: ignore

import structlog

logger = structlog.get_logger()


class KafkaConsumer:
    """Wrapper around :class:`aiokafka.AIOKafkaConsumer` with async iterator."""

    def __init__(
        self,
        topic: str,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "ml-stream",
        concurrency: int = 1,
        enable_auto_commit: bool = False,
    ) -> None:
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.concurrency = concurrency
        self.enable_auto_commit = enable_auto_commit
        self._consumer: Optional[AIOKafkaConsumer] = None
        self.logger = logger.bind(component="kafka-consumer")

    async def start(self) -> None:
        if AIOKafkaConsumer is None:  # pragma: no cover - environment
            raise RuntimeError("aiokafka is required for KafkaConsumer")
        self._consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            enable_auto_commit=self.enable_auto_commit,
        )
        await self._consumer.start()
        self.logger.info("consumer-started", topic=self.topic)

    async def stop(self) -> None:
        if self._consumer:
            await self._consumer.stop()
            self.logger.info("consumer-stopped")

    async def consume(self) -> AsyncIterator[Dict[str, Any]]:
        if not self._consumer:
            raise RuntimeError("consumer not started")
        async for msg in self._consumer:
            yield {"key": msg.key, "value": msg.value, "partition": msg.partition}
