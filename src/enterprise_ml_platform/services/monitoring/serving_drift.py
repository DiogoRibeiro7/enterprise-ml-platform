"""Version-scoped drift monitoring for live serving traffic.

The reference artifact contains distribution summaries rather than training
rows. A bounded rolling window is kept independently for every served model
version, so an alias promotion cannot mix observations from two artifacts.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import structlog

from .collectors.metrics_collector import MetricsCollector

DRIFT_REFERENCE_ARTIFACT = "monitoring/drift_reference.json"
DRIFT_REFERENCE_SCHEMA_VERSION = 1
_EPSILON = 1e-6

logger = structlog.get_logger(__name__)


def _bucket_probabilities(
    values: np.ndarray, cut_points: tuple[float, ...]
) -> np.ndarray:
    if values.size == 0:
        raise ValueError("drift scoring requires at least one row")
    buckets = np.searchsorted(np.asarray(cut_points), values, side="right")
    counts = np.bincount(buckets, minlength=len(cut_points) + 1)
    return counts.astype(float) / counts.sum()


@dataclass(frozen=True)
class FeatureDriftReference:
    """Distribution summary for one numeric model input."""

    name: str
    mean: float
    standard_deviation: float
    cut_points: tuple[float, ...]
    bucket_probabilities: tuple[float, ...]

    def score(self, current: np.ndarray) -> float:
        """Return a population-stability score for ``current`` values."""
        if not self.cut_points:
            different = ~np.isclose(current, self.mean, rtol=0.0, atol=_EPSILON)
            return float(np.mean(different))

        expected = np.clip(
            np.asarray(self.bucket_probabilities, dtype=float), _EPSILON, None
        )
        observed = np.clip(
            _bucket_probabilities(current, self.cut_points), _EPSILON, None
        )
        return float(np.sum((observed - expected) * np.log(observed / expected)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "name": self.name,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "cut_points": list(self.cut_points),
            "bucket_probabilities": list(self.bucket_probabilities),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FeatureDriftReference:
        """Validate and load one feature reference."""
        cut_points = tuple(float(value) for value in payload["cut_points"])
        probabilities = tuple(float(value) for value in payload["bucket_probabilities"])
        if len(probabilities) != len(cut_points) + 1:
            raise ValueError("drift reference bucket count is inconsistent")
        if (
            not all(math.isfinite(value) for value in cut_points)
            or tuple(sorted(set(cut_points))) != cut_points
        ):
            raise ValueError("drift reference cut points must be finite and increasing")
        if not all(math.isfinite(value) and value >= 0 for value in probabilities):
            raise ValueError(
                "drift reference probabilities must be finite and non-negative"
            )
        if not np.isclose(sum(probabilities), 1.0):
            raise ValueError("drift reference probabilities must sum to one")
        mean = float(payload["mean"])
        standard_deviation = float(payload["standard_deviation"])
        if not math.isfinite(mean):
            raise ValueError("drift reference mean must be finite")
        if not math.isfinite(standard_deviation) or standard_deviation < 0:
            raise ValueError(
                "drift reference standard deviation must be finite and non-negative"
            )
        return cls(
            name=str(payload["name"]),
            mean=mean,
            standard_deviation=standard_deviation,
            cut_points=cut_points,
            bucket_probabilities=probabilities,
        )


@dataclass(frozen=True)
class DriftReference:
    """Immutable, non-row-level baseline stored with a trained model."""

    sample_count: int
    features: tuple[FeatureDriftReference, ...]

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return feature names in model input order."""
        return tuple(feature.name for feature in self.features)

    @property
    def feature_count(self) -> int:
        """Return the expected width of serving rows."""
        return len(self.features)

    @classmethod
    def from_array(
        cls,
        values: Sequence[Sequence[float]] | np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> DriftReference:
        """Build a privacy-preserving baseline from numeric training inputs."""
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
            raise ValueError(
                "drift reference requires at least two rows and one feature"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("drift reference values must all be finite")

        names = (
            tuple(str(name) for name in feature_names)
            if feature_names is not None
            else tuple(f"feature_{index}" for index in range(matrix.shape[1]))
        )
        if len(names) != matrix.shape[1] or len(set(names)) != len(names):
            raise ValueError("feature names must be unique and match the input width")

        quantiles = np.linspace(0.1, 0.9, 9)
        features: list[FeatureDriftReference] = []
        for index, name in enumerate(names):
            column = matrix[:, index]
            cut_points = tuple(
                float(value) for value in np.unique(np.quantile(column, quantiles))
            )
            if np.allclose(column, column[0], rtol=0.0, atol=_EPSILON):
                cut_points = ()
            probabilities = tuple(
                float(value) for value in _bucket_probabilities(column, cut_points)
            )
            features.append(
                FeatureDriftReference(
                    name=name,
                    mean=float(np.mean(column)),
                    standard_deviation=float(np.std(column)),
                    cut_points=cut_points,
                    bucket_probabilities=probabilities,
                )
            )
        return cls(sample_count=matrix.shape[0], features=tuple(features))

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned artifact representation."""
        return {
            "schema_version": DRIFT_REFERENCE_SCHEMA_VERSION,
            "sample_count": self.sample_count,
            "features": [feature.to_dict() for feature in self.features],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DriftReference:
        """Validate and load a reference artifact."""
        if payload.get("schema_version") != DRIFT_REFERENCE_SCHEMA_VERSION:
            raise ValueError("unsupported drift reference schema version")
        sample_count = int(payload["sample_count"])
        features = tuple(
            FeatureDriftReference.from_dict(feature) for feature in payload["features"]
        )
        if sample_count < 2 or not features:
            raise ValueError("drift reference is empty")
        names = [feature.name for feature in features]
        if len(set(names)) != len(names):
            raise ValueError("drift reference feature names must be unique")
        return cls(sample_count=sample_count, features=features)


DriftState = Literal["unavailable", "collecting", "ready"]


@dataclass(frozen=True)
class DriftReport:
    """Current drift state for one immutable model artifact."""

    model_name: str
    model_version: str
    state: DriftState
    observed_rows: int
    required_rows: int
    window_size: int
    threshold: float
    scores: dict[str, float]
    drifted_features: tuple[str, ...]


@dataclass
class _VersionState:
    reference: DriftReference
    rows: deque[tuple[float, ...]]
    scores: dict[str, float] = field(default_factory=dict)
    drifted_features: tuple[str, ...] = ()


class ServingDriftMonitor:
    """Maintain bounded, independent drift windows per model version."""

    def __init__(
        self,
        metrics: MetricsCollector,
        *,
        window_size: int = 256,
        min_samples: int = 50,
        threshold: float = 0.2,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if min_samples < 2 or min_samples > window_size:
            raise ValueError("min_samples must be between 2 and window_size")
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be positive")
        self.metrics = metrics
        self.window_size = window_size
        self.min_samples = min_samples
        self.threshold = threshold
        self._states: dict[tuple[str, str], _VersionState] = {}
        self._lock = threading.RLock()

    def register(
        self, model_name: str, model_version: str, reference: DriftReference
    ) -> None:
        """Register the immutable baseline for a model version."""
        key = (model_name, model_version)
        with self._lock:
            current = self._states.get(key)
            if current is not None:
                if current.reference != reference:
                    raise ValueError(
                        f"a different drift reference is already registered for {key}"
                    )
                return
            self._states[key] = _VersionState(
                reference=reference,
                rows=deque(maxlen=self.window_size),
            )
            self.metrics.set_drift_ready(model_name, model_version, ready=False)

    def observe(
        self,
        model_name: str,
        model_version: str,
        rows: Sequence[Sequence[float]] | np.ndarray,
        reference: DriftReference | None = None,
    ) -> DriftReport:
        """Add validated serving rows and evaluate the current window."""
        key = (model_name, model_version)
        if reference is not None:
            self.register(model_name, model_version, reference)

        with self._lock:
            state = self._states.get(key)
            if state is None:
                return self._report(model_name, model_version, None)

            matrix = np.asarray(rows, dtype=float)
            if (
                matrix.ndim != 2
                or matrix.shape[0] < 1
                or matrix.shape[1] != state.reference.feature_count
            ):
                raise ValueError("serving rows do not match the drift reference width")
            if not np.isfinite(matrix).all():
                raise ValueError("serving rows must all be finite")
            for row in matrix:
                state.rows.append(tuple(float(value) for value in row))

            if len(state.rows) < self.min_samples:
                return self._report(model_name, model_version, state)

            window = np.asarray(state.rows, dtype=float)
            state.scores = {
                feature.name: feature.score(window[:, index])
                for index, feature in enumerate(state.reference.features)
            }
            previous_drifted = set(state.drifted_features)
            state.drifted_features = tuple(
                name for name, score in state.scores.items() if score >= self.threshold
            )
            newly_drifted = set(state.drifted_features) - previous_drifted
            report = self._report(model_name, model_version, state)
            self.metrics.set_drift_ready(model_name, model_version, ready=True)
            for feature, score in report.scores.items():
                self.metrics.set_drift(
                    feature,
                    score,
                    model=model_name,
                    version=model_version,
                    detected=feature in report.drifted_features,
                )
        if newly_drifted:
            logger.warning(
                "serving_drift_detected",
                model=model_name,
                version=model_version,
                features=tuple(sorted(newly_drifted)),
            )
        return report

    def status(
        self,
        model_name: str,
        model_version: str,
        reference: DriftReference | None = None,
    ) -> DriftReport:
        """Return the current state without modifying the window."""
        if reference is not None:
            self.register(model_name, model_version, reference)
        with self._lock:
            return self._report(
                model_name, model_version, self._states.get((model_name, model_version))
            )

    def remove(self, model_name: str, model_version: str) -> None:
        """Discard state and Prometheus children for an unloaded artifact."""
        with self._lock:
            state = self._states.pop((model_name, model_version), None)
            if state is not None:
                self.metrics.clear_drift(
                    model_name, model_version, state.reference.feature_names
                )

    def _report(
        self,
        model_name: str,
        model_version: str,
        state: _VersionState | None,
    ) -> DriftReport:
        if state is None:
            status: DriftState = "unavailable"
            observed_rows = 0
            scores: dict[str, float] = {}
            drifted_features: tuple[str, ...] = ()
        else:
            observed_rows = len(state.rows)
            status = "ready" if observed_rows >= self.min_samples else "collecting"
            scores = dict(state.scores)
            drifted_features = state.drifted_features
        return DriftReport(
            model_name=model_name,
            model_version=model_version,
            state=status,
            observed_rows=observed_rows,
            required_rows=self.min_samples,
            window_size=self.window_size,
            threshold=self.threshold,
            scores=scores,
            drifted_features=drifted_features,
        )
