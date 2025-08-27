from __future__ import annotations

"""Simple ML based drift detector.

This detector trains a trivial model on reference data statistics and compares
current data against it.  The implementation is intentionally lightweight.
"""

from typing import Dict, Sequence

import numpy as np


class MLDriftDetector:
    """Approximate drift detection using basic statistical learning."""

    def __init__(self) -> None:
        self.reference_means: Dict[str, float] = {}

    def fit(self, reference: Dict[str, Sequence[float]]) -> None:
        for name, values in reference.items():
            self.reference_means[name] = float(np.mean(values))

    def predict(self, current: Dict[str, Sequence[float]]) -> Dict[str, float]:
        """Return absolute mean differences for each feature."""
        scores: Dict[str, float] = {}
        for name, values in current.items():
            ref = self.reference_means.get(name)
            if ref is None:
                continue
            scores[name] = float(abs(ref - np.mean(values)))
        return scores
