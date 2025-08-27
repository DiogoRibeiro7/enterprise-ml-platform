from __future__ import annotations
"""Gather basic metrics from cluster managers."""

from typing import Dict

import structlog

logger = structlog.get_logger(__name__)


class ClusterMonitor:
    """Collects run state information from cluster managers."""

    def __init__(self, managers: Dict[str, object]) -> None:
        self.managers = managers

    def metrics(self) -> Dict[str, bool]:
        """Return a dictionary with the running state of each framework."""
        status = {name: bool(getattr(mgr, "running", False)) for name, mgr in self.managers.items()}
        logger.debug("cluster metrics", status=status)
        return status
