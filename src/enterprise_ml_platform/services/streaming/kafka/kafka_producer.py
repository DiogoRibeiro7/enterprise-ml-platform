from __future__ import annotations

"""Kafka producer utilities for streaming pipeline."""

from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from aiokafka import AIOKafkaProducer
except Exception:  # pragma: no cover
    AIOKafkaProducer = None  # type: ignore

import structlog

logger = structlog.get_logger()


class KafkaProducer:
    """Wrapper around :class:`aiokafka.AIOKafkaProducer` for sending messages."""

    def __init__(
        self,
        topic: str,
        bootstrap_servers: str = "localhost:9092",
        compression_type: str | None = "gzip",
    ) -> None:
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.compression_type = compression_type
        self._producer: Optional[AIOKafkaProducer] = None
        self.logger = logger.bind(component="kafka-producer")

    async def start(self) -> None:
        if AIOKafkaProducer is None:  # pragma: no cover
            raise RuntimeError("aiokafka is required for KafkaProducer")
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            compression_type=self.compression_type,
        )
        await self._producer.start()
        self.logger.info("producer-started", topic=self.topic)

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()
            self.logger.info("producer-stopped")

    async def send(self, message: Dict[str, Any]) -> None:
        if not self._producer:
            raise RuntimeError("producer not started")
        await self._producer.send_and_wait(self.topic, value=message.get("value"), key=message.get("key"))
