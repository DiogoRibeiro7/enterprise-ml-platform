"""Collect experiment metrics and expose Prometheus counters."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from prometheus_client import Counter, Histogram

assignment_counter = Counter(
    "ab_test_assignments", "Experiment variant assignments", ["experiment", "variant"]
)
conversion_counter = Counter(
    "ab_test_conversions", "Experiment conversions", ["experiment", "variant"]
)
metric_histogram = Histogram(
    "ab_test_metric", "Observed metric values", ["experiment", "variant"]
)


class ExperimentTracker:
    """In-memory tracking for experiment metrics."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = defaultdict(
            lambda: defaultdict(lambda: {"values": [], "success_fail": [0, 0]})
        )

    def record_assignment(self, experiment: str, variant: str) -> None:
        assignment_counter.labels(experiment, variant).inc()

    def record_outcome(
        self, experiment: str, variant: str, value: float, success: bool
    ) -> None:
        metric_histogram.labels(experiment, variant).observe(value)
        if success:
            conversion_counter.labels(experiment, variant).inc()
            self._data[experiment][variant]["success_fail"][0] += 1
        else:
            self._data[experiment][variant]["success_fail"][1] += 1
        self._data[experiment][variant]["values"].append(value)

    def get_metrics(self, experiment: str) -> dict[str, Any]:
        return {"variants": self._data[experiment]}
