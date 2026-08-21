"""Advanced statistical drift detection utilities.

This module implements several common statistical techniques for measuring
feature drift between a reference distribution and the current distribution.
The focus is on lightweight implementations that work in restricted test
environments while still providing meaningful drift scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp

_EPS = 1e-8


def _categorical_freq(values: Sequence[object]) -> dict[object, float]:
    """Return frequency distribution for categorical values."""
    uniques, counts = np.unique(np.asarray(values, dtype=str), return_counts=True)
    total = counts.sum() + _EPS
    return {u: c / total for u, c in zip(uniques, counts, strict=True)}


def ks_statistic(ref: Sequence[float], cur: Sequence[float]) -> float:
    """Kolmogorov-Smirnov statistic for two numeric samples."""
    stat, _ = ks_2samp(ref, cur)
    return float(stat)


def psi(ref: Sequence, cur: Sequence, bins: int = 10) -> float:
    """Population Stability Index supporting numeric and categorical data."""
    ref_arr = np.asarray(ref)
    cur_arr = np.asarray(cur)
    if np.issubdtype(ref_arr.dtype, np.number) and np.issubdtype(
        cur_arr.dtype, np.number
    ):
        ref_hist, bin_edges = np.histogram(ref_arr, bins=bins, density=True)
        cur_hist, _ = np.histogram(cur_arr, bins=bin_edges, density=True)
    else:
        ref_dist = _categorical_freq(ref_arr)
        cur_dist = _categorical_freq(cur_arr)
        categories = sorted(set(ref_dist) | set(cur_dist))
        ref_hist = np.array([ref_dist.get(c, _EPS) for c in categories])
        cur_hist = np.array([cur_dist.get(c, _EPS) for c in categories])
    ref_hist += _EPS
    cur_hist += _EPS
    return float(np.sum((ref_hist - cur_hist) * np.log(ref_hist / cur_hist)))


def js_divergence(ref: Sequence, cur: Sequence, bins: int = 10) -> float:
    """Jensen-Shannon divergence between two distributions."""
    ref_arr = np.asarray(ref)
    cur_arr = np.asarray(cur)
    if np.issubdtype(ref_arr.dtype, np.number) and np.issubdtype(
        cur_arr.dtype, np.number
    ):
        ref_hist, bin_edges = np.histogram(ref_arr, bins=bins, density=True)
        cur_hist, _ = np.histogram(cur_arr, bins=bin_edges, density=True)
    else:
        ref_dist = _categorical_freq(ref_arr)
        cur_dist = _categorical_freq(cur_arr)
        categories = sorted(set(ref_dist) | set(cur_dist))
        ref_hist = np.array([ref_dist.get(c, _EPS) for c in categories])
        cur_hist = np.array([cur_dist.get(c, _EPS) for c in categories])
    ref_hist += _EPS
    cur_hist += _EPS
    m = 0.5 * (ref_hist + cur_hist)
    return float(
        0.5
        * (
            np.sum(ref_hist * np.log(ref_hist / m))
            + np.sum(cur_hist * np.log(cur_hist / m))
        )
    )


@dataclass
class ConceptDriftDetector:
    """Detect concept drift using model confidence scores.

    The detector maintains a rolling window of confidence scores from reference
    predictions.  Current confidences are compared to the reference mean and an
    adaptive threshold is derived from the standard deviation of the reference
    window.
    """

    window: int = 100
    multiplier: float = 3.0

    def __post_init__(self) -> None:
        self.reference: np.ndarray | None = None
        self.history: np.ndarray = np.array([], dtype=float)
        self.threshold: float = 0.0

    def fit(self, confidences: Sequence[float]) -> None:
        arr = np.asarray(confidences, dtype=float)[-self.window :]
        self.reference = arr
        self.threshold = float(np.std(arr) * self.multiplier)

    def detect(self, confidences: Sequence[float]) -> tuple[float, bool]:
        cur = np.asarray(confidences, dtype=float)
        self.history = np.concatenate([self.history, cur])[-self.window :]
        if self.reference is None or len(self.history) == 0:
            return 0.0, False
        score = float(abs(np.mean(self.history) - np.mean(self.reference)))
        return score, score > self.threshold


class AdvancedDriftDetector:
    """Combine multiple statistical tests with adaptive thresholds."""

    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold
        self.reference: dict[str, Sequence] = {}

    def fit(self, reference: dict[str, Sequence]) -> None:
        for name, values in reference.items():
            self.reference[name] = list(values)

    def detect(self, current: dict[str, Sequence]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for name, values in current.items():
            ref = self.reference.get(name)
            if ref is None:
                continue
            ref_arr = np.asarray(ref)
            cur_arr = np.asarray(values)
            if np.issubdtype(ref_arr.dtype, np.number) and np.issubdtype(
                cur_arr.dtype, np.number
            ):
                ks = ks_statistic(ref_arr, cur_arr)
            else:
                ks = 0.0
            psi_score = psi(ref_arr, cur_arr)
            js = js_divergence(ref_arr, cur_arr)
            scores[name] = max(ks, psi_score, js)
        return scores
