"""Dask cluster management utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

try:  # pragma: no cover - optional dependency
    from dask.distributed import Client, LocalCluster
except Exception:  # pragma: no cover
    Client = None  # type: ignore
    LocalCluster = None  # type: ignore


class DaskClusterManager:
    """Manage a Dask ``LocalCluster`` for development and tests."""

    def __init__(self) -> None:
        self.client: Client | None = None

    def start_cluster(self) -> None:
        """Spin up a small in-process cluster."""
        if Client is None or LocalCluster is None:
            logger.warning("dask.distributed not installed; using local fallback")
            return
        cluster = LocalCluster(processes=False)
        self.client = Client(cluster)
        logger.info("dask cluster started", dashboard=self.client.dashboard_link)

    def stop_cluster(self) -> None:
        if self.client:
            self.client.close()
            logger.info("dask cluster stopped")
        self.client = None

    @property
    def running(self) -> bool:
        return self.client is not None

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if not self.running:
            raise RuntimeError("Dask cluster is not running")
        if self.client is None:
            return fn(*args, **kwargs)
        future = self.client.submit(fn, *args, **kwargs)
        return future.result()
