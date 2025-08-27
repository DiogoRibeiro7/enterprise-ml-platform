from __future__ import annotations

"""Combine multiple drift detectors and aggregate results."""

from typing import Dict, Sequence

from .ml_drift import MLDriftDetector
from .statistical_drift import StatisticalDriftDetector
from .advanced_drift import AdvancedDriftDetector, ConceptDriftDetector


class DriftAnalyzer:
    """High level drift analysis orchestrator."""

    def __init__(
        self,
        statistical: StatisticalDriftDetector | None = None,
        ml: MLDriftDetector | None = None,
        advanced: AdvancedDriftDetector | None = None,
        concept: ConceptDriftDetector | None = None,
    ) -> None:
        self.statistical = statistical or StatisticalDriftDetector()
        self.ml = ml or MLDriftDetector()
        self.advanced = advanced or AdvancedDriftDetector()
        self.concept = concept or ConceptDriftDetector()
        self._fitted = False

    def fit(
        self,
        reference: Dict[str, Sequence[float]],
        confidences: Sequence[float] | None = None,
    ) -> None:
        self.statistical.fit(reference)
        self.ml.fit(reference)
        self.advanced.fit(reference)
        if confidences is not None:
            self.concept.fit(confidences)
        self._fitted = True

    def check(
        self,
        current: Dict[str, Sequence[float]],
        confidences: Sequence[float] | None = None,
    ) -> Dict[str, float]:
        if not self._fitted:
            self.fit(current, confidences)
            return {name: 0.0 for name in current}
        stat_scores = self.statistical.detect(current)
        ml_scores = self.ml.predict(current)
        adv_scores = self.advanced.detect(current)
        combined: Dict[str, float] = {}
        keys = set(stat_scores) | set(ml_scores) | set(adv_scores)
        for key in keys:
            combined[key] = (
                stat_scores.get(key, 0.0)
                + ml_scores.get(key, 0.0)
                + adv_scores.get(key, 0.0)
            ) / 3
        if confidences is not None:
            score, _ = self.concept.detect(confidences)
            combined["concept"] = score
        return combined
