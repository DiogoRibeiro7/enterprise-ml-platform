from __future__ import annotations

"""Combine multiple drift detectors and aggregate results."""

from typing import Dict, Sequence

from .ml_drift import MLDriftDetector
from .statistical_drift import StatisticalDriftDetector


class DriftAnalyzer:
    """High level drift analysis orchestrator."""

    def __init__(
        self,
        statistical: StatisticalDriftDetector | None = None,
        ml: MLDriftDetector | None = None,
    ) -> None:
        self.statistical = statistical or StatisticalDriftDetector()
        self.ml = ml or MLDriftDetector()
        self._fitted = False

    def fit(self, reference: Dict[str, Sequence[float]]) -> None:
        self.statistical.fit(reference)
        self.ml.fit(reference)
        self._fitted = True

    def check(self, current: Dict[str, Sequence[float]]) -> Dict[str, float]:
        if not self._fitted:
            self.fit(current)
            return {name: 0.0 for name in current}
        stat_scores = self.statistical.detect(current)
        ml_scores = self.ml.predict(current)
        combined: Dict[str, float] = {}
        for key, score in stat_scores.items():
            combined[key] = (score + ml_scores.get(key, 0.0)) / 2
        for key, score in ml_scores.items():
            combined.setdefault(key, score)
        return combined
