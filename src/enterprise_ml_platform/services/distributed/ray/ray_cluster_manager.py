"""Ray cluster management utilities.

This module provides a very small wrapper around ``ray`` to keep the rest of
this repository independent from the actual library.  The class exposes methods
for starting/stopping a cluster and submitting tasks.  When ``ray`` is not
available the implementation gracefully falls back to local execution so the
package can be used in lightweight environments (like the tests for this kata).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

try:  # pragma: no cover - optional dependency
    import ray
except Exception:  # pragma: no cover
    ray = None  # type: ignore


class RayClusterManager:
    """Light‑weight manager for a Ray cluster."""

    def __init__(self) -> None:
        self._running = False

    def start_cluster(self) -> None:
        """Start a local Ray cluster if possible."""
        if ray is None:
            logger.warning("ray not installed; running in local fallback mode")
            self._running = True
            return
        if not ray.is_initialized():  # pragma: no branch - tiny helper
            ray.init(ignore_reinit_error=True)
        self._running = True
        logger.info("ray cluster started")

    def stop_cluster(self) -> None:
        """Shut the cluster down."""
        if ray and ray.is_initialized():
            ray.shutdown()
        self._running = False
        logger.info("ray cluster stopped")

    @property
    def running(self) -> bool:
        return self._running

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute ``fn`` on the cluster.

        In environments where ``ray`` is not available the function is executed
        synchronously in the local process.
        """
        if not self._running:
            raise RuntimeError("Ray cluster is not running")
        if ray is None:
            return fn(*args, **kwargs)
        remote = ray.remote(fn)
        return ray.get(remote.remote(*args, **kwargs))
