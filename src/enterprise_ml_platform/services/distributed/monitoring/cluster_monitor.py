"""Gather basic metrics from cluster managers."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


class ClusterMonitor:
    """Collects run state information from cluster managers."""

    def __init__(self, managers: dict[str, object]) -> None:
        self.managers = managers

    def metrics(self) -> dict[str, bool]:
        """Return a dictionary with the running state of each framework."""
        status = {
            name: bool(getattr(mgr, "running", False))
            for name, mgr in self.managers.items()
        }
        logger.debug("cluster metrics", status=status)
        return status
