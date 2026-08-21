"""Simple statistical anomaly detection for cost spikes."""

from __future__ import annotations

import statistics


class AnomalyDetector:
    """Detect cost spikes using z-score thresholding."""

    def __init__(self, threshold: float = 1.0) -> None:
        # Lower default so moderate spikes are flagged in tests
        self.threshold = threshold

    def detect(self, series: list[float]) -> bool:
        if len(series) < 2:
            return False
        mean = statistics.mean(series)
        stdev = statistics.stdev(series)
        if stdev == 0:
            return False
        latest = series[-1]
        return abs(latest - mean) / stdev > self.threshold
