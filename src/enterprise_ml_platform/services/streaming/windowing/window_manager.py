"""Window management for streaming aggregates."""

from __future__ import annotations

import collections
import time
from typing import Any

import structlog

logger = structlog.get_logger()


class WindowManager:
    """Maintain sliding or tumbling windows over streaming features."""

    def __init__(self, window_size: float, step: float | None = None) -> None:
        self.window_size = window_size
        self.step = step or window_size
        self.buffer: collections.deque[tuple[float, dict[str, Any]]] = (
            collections.deque()
        )
        self.logger = logger.bind(component="window-manager")

    async def apply(self, features: dict[str, Any]) -> dict[str, Any]:
        """Add features to window and return aggregated features."""
        now = time.time()
        self.buffer.append((now, features))
        while self.buffer and now - self.buffer[0][0] > self.window_size:
            self.buffer.popleft()
        agg: dict[str, Any] = {}
        for _, feat in self.buffer:
            for key, value in feat.items():
                agg[key] = agg.get(key, 0) + value
        return {f"window_{k}": v / len(self.buffer) for k, v in agg.items()}
