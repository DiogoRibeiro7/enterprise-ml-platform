"""Streaming pipeline orchestrator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import structlog

from .checkpointing.checkpoint_manager import CheckpointManager
from .continuous_learning.incremental_trainer import IncrementalTrainer
from .feature_engineering.stream_feature_engine import StreamFeatureEngine
from .kafka.kafka_consumer import KafkaConsumer
from .kafka.kafka_producer import KafkaProducer
from .monitoring.stream_monitor import StreamMonitor
from .predictors.stream_predictor import StreamPredictor
from .state.state_manager import StateManager
from .transformers.stream_transformer import StreamTransformer
from .windowing.window_manager import WindowManager

logger = structlog.get_logger()


@dataclass
class StreamConfig:
    """Configuration for :class:`StreamProcessor`."""

    input_topic: str
    output_topic: str
    group_id: str = "ml-stream"
    concurrency: int = 1


class StreamProcessor:
    """Coordinate real-time consumption, transformation, prediction, and production.

    The processor wires together Kafka clients, feature transformers, predictors,
    and auxiliary components such as state management and monitoring. It ensures
    exactly-once semantics by committing checkpoints after successful prediction
    and production.
    """

    def __init__(
        self,
        consumer: KafkaConsumer,
        producer: KafkaProducer,
        transformer: StreamTransformer,
        predictor: StreamPredictor,
        window_manager: WindowManager | None = None,
        state_manager: StateManager | None = None,
        monitor: StreamMonitor | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        feature_engine: StreamFeatureEngine | None = None,
        incremental_trainer: IncrementalTrainer | None = None,
    ) -> None:
        self.consumer = consumer
        self.producer = producer
        self.transformer = transformer
        self.predictor = predictor
        self.window_manager = window_manager
        self.state_manager = state_manager
        self.monitor = monitor
        self.checkpoint_manager = checkpoint_manager
        self.feature_engine = feature_engine
        self.incremental_trainer = incremental_trainer
        self._running = False
        self.logger = logger.bind(component="stream-processor")

    async def _handle_message(self, message: dict[str, Any]) -> None:
        try:
            features = await self.transformer.transform(message)
            if self.feature_engine:
                features = await self.feature_engine.compute(features)
            if self.window_manager:
                features = await self.window_manager.apply(features)
            if self.state_manager:
                await self.state_manager.update_state(message)
            prediction = await self.predictor.predict(features)
            await self.producer.send(prediction)
            if self.incremental_trainer and "label" in message:
                await self.incremental_trainer.update(features, message["label"])
            if self.monitor:
                await self.monitor.record_success()
        except Exception as exc:  # pragma: no cover - runtime errors
            self.logger.error("stream-process-error", error=str(exc))
            if self.monitor:
                await self.monitor.record_failure()
            raise
        finally:
            if self.checkpoint_manager:
                await self.checkpoint_manager.mark_checkpoint(message)

    async def _consume(self) -> None:
        async for msg in self.consumer.consume():
            await self._handle_message(msg)

    async def start(self) -> None:
        """Start consuming messages until cancelled."""
        if self._running:
            return
        self._running = True
        await self.consumer.start()
        tasks = [
            asyncio.create_task(self._consume())
            for _ in range(self.consumer.concurrency)
        ]
        self.logger.info("stream-started", tasks=len(tasks))
        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        """Gracefully stop the processor."""
        if not self._running:
            return
        self._running = False
        await self.consumer.stop()
        await self.producer.stop()
        if self.state_manager:
            await self.state_manager.close()
        if self.monitor:
            await self.monitor.close()
        if self.checkpoint_manager:
            await self.checkpoint_manager.close()
        self.logger.info("stream-stopped")
