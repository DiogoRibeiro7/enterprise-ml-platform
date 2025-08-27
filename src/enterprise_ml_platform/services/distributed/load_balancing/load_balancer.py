from __future__ import annotations
"""Minimal load balancer for cluster managers."""

from typing import Dict

import structlog

logger = structlog.get_logger(__name__)


class LoadBalancer:
    """Select a cluster manager based on framework name.

    The balancer holds a mapping of framework identifiers (``"ray"``, ``"dask"``,
    ``"spark"``) to their corresponding managers.  A more advanced implementation
    could account for current resource usage or employ work-stealing algorithms,
    but that would be overkill for the purposes of this repository.
    """

    def __init__(self, managers: Dict[str, object]) -> None:
        self.managers = managers

    def choose_manager(self, framework: str):
        if framework not in self.managers:
            raise KeyError(f"Unknown framework {framework}")
        return self.managers[framework]
