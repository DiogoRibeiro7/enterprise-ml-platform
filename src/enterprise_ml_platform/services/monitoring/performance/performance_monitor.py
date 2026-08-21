"""Utility for tracking model performance metrics."""

from __future__ import annotations

from collections import defaultdict


class PerformanceMonitor:
    """Track simple accuracy statistics for models."""

    def __init__(self) -> None:
        self._history: dict[str, list[float]] = defaultdict(list)

    def update(self, model: str, actual: float, predicted: float) -> float:
        """Update accuracy for a model and return current average."""
        correct = 1.0 if actual == predicted else 0.0
        hist = self._history[model]
        hist.append(correct)
        return sum(hist) / len(hist)
