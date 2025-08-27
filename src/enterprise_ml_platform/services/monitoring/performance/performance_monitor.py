from __future__ import annotations

"""Utility for tracking model performance metrics."""

from collections import defaultdict
from typing import Dict, List


class PerformanceMonitor:
    """Track simple accuracy statistics for models."""

    def __init__(self) -> None:
        self._history: Dict[str, List[float]] = defaultdict(list)

    def update(self, model: str, actual: float, predicted: float) -> float:
        """Update accuracy for a model and return current average."""
        correct = 1.0 if actual == predicted else 0.0
        hist = self._history[model]
        hist.append(correct)
        return sum(hist) / len(hist)
