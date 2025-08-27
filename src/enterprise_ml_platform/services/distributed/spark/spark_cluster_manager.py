from __future__ import annotations
"""Spark cluster management utilities."""

from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)

try:  # pragma: no cover - optional dependency
    from pyspark.sql import SparkSession
except Exception:  # pragma: no cover
    SparkSession = None  # type: ignore


class SparkClusterManager:
    """Create a minimal Spark session for local development."""

    def __init__(self) -> None:
        self.session: SparkSession | None = None

    def start_cluster(self) -> None:
        if SparkSession is None:
            logger.warning("pyspark not installed; running in local fallback mode")
            return
        self.session = (
            SparkSession.builder.master("local[*]")
            .appName("enterprise-ml-platform")
            .getOrCreate()
        )
        logger.info("spark session started")

    def stop_cluster(self) -> None:
        if self.session is not None:
            self.session.stop()
            logger.info("spark session stopped")
        self.session = None

    @property
    def running(self) -> bool:
        return self.session is not None

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if not self.running:
            raise RuntimeError("Spark session is not running")
        if self.session is None:
            return fn(*args, **kwargs)
        return fn(self.session, *args, **kwargs)
