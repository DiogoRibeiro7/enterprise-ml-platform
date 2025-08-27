from __future__ import annotations

"""Lightweight statistical drift detector.

The implementation compares feature means between reference and current data and
returns a normalised difference.  It avoids heavy statistical dependencies so it
can run in constrained environments.
"""

from typing import Dict, Sequence

import numpy as np


class StatisticalDriftDetector:
    """Detect drift using simple statistical measures."""

    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold
        self.reference: Dict[str, np.ndarray] = {}

    def fit(self, reference: Dict[str, Sequence[float]]) -> None:
        """Store reference statistics for features."""
        for name, values in reference.items():
            self.reference[name] = np.asarray(values, dtype=float)

    def detect(self, current: Dict[str, Sequence[float]]) -> Dict[str, float]:
        """Return drift scores for supplied feature values."""
        scores: Dict[str, float] = {}
        for name, values in current.items():
            ref = self.reference.get(name)
            if ref is None:
                continue
            cur = np.asarray(values, dtype=float)
            if ref.std() == 0:
                score = float(abs(cur.mean() - ref.mean()))
            else:
                score = float(abs(cur.mean() - ref.mean()) / ref.std())
            scores[name] = score
        return scores
