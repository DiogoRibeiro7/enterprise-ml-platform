from __future__ import annotations

"""Analyze resource usage trends."""

from collections import defaultdict, deque
from typing import Deque, Dict


class UsageAnalyzer:
    """Maintain moving averages for project resource usage."""

    def __init__(self, window: int = 5) -> None:
        self.window = window
        self._usage: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def record_usage(self, project: str, metric: str, value: float) -> None:
        self._usage[f"{project}:{metric}"].append(value)

    def moving_average(self, project: str, metric: str) -> float:
        values = self._usage.get(f"{project}:{metric}")
        if not values:
            return 0.0
        return sum(values) / len(values)
