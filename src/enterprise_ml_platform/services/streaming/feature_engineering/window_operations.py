"""Sliding window aggregations for streaming features."""

from __future__ import annotations

import collections
import time
from typing import Any


class TimeWindowAggregator:
    """Aggregate numeric features over a time-based sliding window."""

    def __init__(self, field: str, window_size: float) -> None:
        self.field = field
        self.window_size = window_size
        self.buffer: collections.deque[tuple[float, float]] = collections.deque()

    async def apply(self, features: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        value = features.get(self.field)
        if value is not None:
            self.buffer.append((now, float(value)))
        while self.buffer and now - self.buffer[0][0] > self.window_size:
            self.buffer.popleft()
        if not self.buffer:
            return {}
        avg = sum(v for _, v in self.buffer) / len(self.buffer)
        return {f"{self.field}_time_avg": avg}


class CountWindowAggregator:
    """Aggregate numeric features over a count-based window."""

    def __init__(self, field: str, count: int) -> None:
        self.field = field
        self.count = count
        self.buffer: collections.deque[float] = collections.deque(maxlen=count)

    async def apply(self, features: dict[str, Any]) -> dict[str, Any]:
        value = features.get(self.field)
        if value is not None:
            self.buffer.append(float(value))
        if not self.buffer:
            return {}
        avg = sum(self.buffer) / len(self.buffer)
        return {f"{self.field}_count_avg": avg}
