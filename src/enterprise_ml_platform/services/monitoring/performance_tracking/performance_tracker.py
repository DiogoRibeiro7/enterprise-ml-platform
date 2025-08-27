from __future__ import annotations

"""Advanced performance tracking for production monitoring.

The :class:`PerformanceTracker` extends the basic :class:`PerformanceMonitor`
with sliding window analysis, rudimentary A/B test comparison and business KPI
correlation hooks.  The implementation is intentionally lightweight yet
illustrates how a production-ready tracker could be structured.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List

import numpy as np


@dataclass
class ABTestResult:
    variant: str
    metric: float


class PerformanceTracker:
    """Track model performance with additional analysis utilities."""

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self.history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self.variants: Dict[str, List[float]] = defaultdict(list)
        self.kpis: Dict[str, float] = {}

    def update(self, model: str, actual: float, predicted: float) -> float:
        correct = 1.0 if actual == predicted else 0.0
        hist = self.history[model]
        hist.append(correct)
        return float(np.mean(hist))

    def record_variant(self, name: str, accuracy: float) -> None:
        self.variants[name].append(accuracy)

    def best_variant(self) -> ABTestResult | None:
        if not self.variants:
            return None
        winner = max(self.variants.items(), key=lambda item: np.mean(item[1]))
        return ABTestResult(winner[0], float(np.mean(winner[1])))

    def update_kpi(self, name: str, value: float) -> None:
        self.kpis[name] = value

    def correlate_kpi(self, name: str, model: str) -> float | None:
        kpi = self.kpis.get(name)
        hist = self.history.get(model)
        if kpi is None or not hist:
            return None
        return float(kpi * np.mean(hist))
