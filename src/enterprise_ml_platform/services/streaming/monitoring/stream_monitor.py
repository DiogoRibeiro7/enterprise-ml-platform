"""Monitoring utilities for streaming pipeline."""

from __future__ import annotations

import structlog
from prometheus_client import Counter, Summary

logger = structlog.get_logger()


class StreamMonitor:
    """Record streaming metrics and expose helpers for monitoring."""

    def __init__(self) -> None:
        self.success_counter = Counter(
            "stream_success_total", "Number of processed events"
        )
        self.failure_counter = Counter(
            "stream_failure_total", "Number of failed events"
        )
        self.latency_summary = Summary(
            "stream_processing_seconds", "Stream processing latency"
        )
        self.logger = logger.bind(component="stream-monitor")

    async def record_success(self) -> None:
        self.success_counter.inc()

    async def record_failure(self) -> None:
        self.failure_counter.inc()

    async def close(self) -> None:
        self.logger.info("monitor-closed")
