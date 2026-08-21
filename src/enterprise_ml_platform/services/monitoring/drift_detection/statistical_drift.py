"""Lightweight statistical drift detector.

The implementation compares feature means between reference and current data and
returns a normalised difference.  It avoids heavy statistical dependencies so it
can run in constrained environments.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class StatisticalDriftDetector:
    """Detect drift using simple statistical measures."""

    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold
        self.reference: dict[str, np.ndarray] = {}

    def fit(self, reference: dict[str, Sequence[float]]) -> None:
        """Store reference statistics for features."""
        for name, values in reference.items():
            arr = np.asarray(values)
            if not np.issubdtype(arr.dtype, np.number):
                continue
            self.reference[name] = arr.astype(float)

    def detect(self, current: dict[str, Sequence[float]]) -> dict[str, float]:
        """Return drift scores for supplied feature values."""
        scores: dict[str, float] = {}
        for name, values in current.items():
            ref = self.reference.get(name)
            if ref is None:
                continue
            cur_arr = np.asarray(values)
            if not np.issubdtype(cur_arr.dtype, np.number):
                continue
            cur = cur_arr.astype(float)
            if ref.std() == 0:
                score = float(abs(cur.mean() - ref.mean()))
            else:
                score = float(abs(cur.mean() - ref.mean()) / ref.std())
            scores[name] = score
        return scores
