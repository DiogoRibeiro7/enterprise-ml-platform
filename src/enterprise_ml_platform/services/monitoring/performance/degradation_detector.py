"""Detect performance degradation over a rolling window."""

from __future__ import annotations

from collections import defaultdict, deque


class DegradationDetector:
    """Simple detector using moving average thresholds."""

    def __init__(self, window: int = 5, threshold: float = 0.1) -> None:
        self.window = window
        self.threshold = threshold
        self.history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def add(self, model: str, accuracy: float) -> bool:
        """Add accuracy value and return ``True`` if degradation is detected."""
        hist = self.history[model]
        hist.append(accuracy)
        if len(hist) < self.window:
            return False
        return (sum(hist) / len(hist)) < self.threshold
