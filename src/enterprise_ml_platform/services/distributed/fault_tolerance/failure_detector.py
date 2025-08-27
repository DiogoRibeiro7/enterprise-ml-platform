from __future__ import annotations
"""Simple failure detection hooks."""

from typing import Callable, List

import structlog

logger = structlog.get_logger(__name__)


class FailureDetector:
    """Dispatch failure notifications to registered callbacks."""

    def __init__(self) -> None:
        self._callbacks: List[Callable[[str, Exception], None]] = []

    def register(self, callback: Callable[[str, Exception], None]) -> None:
        self._callbacks.append(callback)

    def notify(self, node_id: str, exc: Exception) -> None:
        logger.error("node failure detected", node=node_id, error=str(exc))
        for cb in self._callbacks:
            try:
                cb(node_id, exc)
            except Exception:  # pragma: no cover - defensive
                logger.exception("failure callback raised")
