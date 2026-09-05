"""Tests for version-scoped drift monitoring of live serving inputs."""

from __future__ import annotations

import numpy as np
from prometheus_client import CollectorRegistry, generate_latest

from enterprise_ml_platform.services.monitoring.collectors import MetricsCollector
from enterprise_ml_platform.services.monitoring.serving_drift import (
    DriftReference,
    ServingDriftMonitor,
)


def _reference() -> DriftReference:
    values = np.column_stack((np.arange(10, dtype=float), np.arange(10, 20)))
    return DriftReference.from_array(values, ["amount", "velocity"])


def test_reference_round_trip_contains_summaries_not_training_rows() -> None:
    reference = _reference()

    payload = reference.to_dict()

    assert payload["sample_count"] == 10
    assert "rows" not in payload
    assert "values" not in payload
    assert DriftReference.from_dict(payload) == reference


def test_sparse_reference_values_are_not_mistaken_for_a_constant_feature() -> None:
    values = np.asarray([[0.0]] * 99 + [[100.0]])

    reference = DriftReference.from_array(values, ["amount"])
    report = ServingDriftMonitor(
        MetricsCollector(CollectorRegistry()),
        window_size=100,
        min_samples=100,
    ).observe("fraud", "1", values, reference)

    assert reference.features[0].cut_points
    assert report.scores["amount"] == 0.0


def test_windows_and_scores_are_isolated_by_model_version() -> None:
    metrics = MetricsCollector(CollectorRegistry())
    monitor = ServingDriftMonitor(
        metrics, window_size=10, min_samples=10, threshold=0.2
    )
    reference = _reference()
    baseline = np.column_stack((np.arange(10, dtype=float), np.arange(10, 20)))
    shifted = baseline + 100

    stable = monitor.observe("fraud", "1", baseline, reference)
    drifted = monitor.observe("fraud", "2", shifted, reference)

    assert stable.state == "ready"
    assert stable.drifted_features == ()
    assert drifted.state == "ready"
    assert drifted.drifted_features == ("amount", "velocity")
    assert monitor.status("fraud", "1").scores != monitor.status("fraud", "2").scores


def test_window_is_bounded_and_metrics_identify_the_served_artifact() -> None:
    registry = CollectorRegistry()
    metrics = MetricsCollector(registry)
    monitor = ServingDriftMonitor(metrics, window_size=10, min_samples=5, threshold=0.2)
    reference = _reference()

    report = monitor.observe("fraud", "7", np.full((25, 2), 100.0), reference)
    output = generate_latest(registry).decode("utf-8")

    assert report.observed_rows == 10
    assert (
        'ml_feature_drift_detected{feature="amount",model="fraud",version="7"} 1.0'
        in output
    )
    assert 'ml_drift_monitor_ready{model="fraud",version="7"} 1.0' in output


def test_missing_reference_is_explicitly_unavailable() -> None:
    monitor = ServingDriftMonitor(MetricsCollector(CollectorRegistry()))

    report = monitor.observe("legacy", "3", [[1.0, 2.0]])

    assert report.state == "unavailable"
    assert report.scores == {}


def test_unload_removes_the_version_metric_children() -> None:
    registry = CollectorRegistry()
    monitor = ServingDriftMonitor(
        MetricsCollector(registry), window_size=10, min_samples=5
    )
    monitor.observe("fraud", "7", np.full((10, 2), 100.0), _reference())

    monitor.remove("fraud", "7")

    assert 'model="fraud"' not in generate_latest(registry).decode("utf-8")


def test_unload_before_the_window_is_ready_is_safe() -> None:
    registry = CollectorRegistry()
    monitor = ServingDriftMonitor(
        MetricsCollector(registry), window_size=10, min_samples=5
    )
    monitor.register("fraud", "7", _reference())

    monitor.remove("fraud", "7")

    assert 'model="fraud"' not in generate_latest(registry).decode("utf-8")


def test_empty_observation_is_rejected() -> None:
    monitor = ServingDriftMonitor(MetricsCollector(CollectorRegistry()))

    with np.testing.assert_raises_regex(ValueError, "serving rows"):
        monitor.observe("fraud", "7", np.empty((0, 2)), _reference())


def test_non_finite_observation_is_rejected() -> None:
    monitor = ServingDriftMonitor(MetricsCollector(CollectorRegistry()))

    with np.testing.assert_raises_regex(ValueError, "finite"):
        monitor.observe("fraud", "7", [[np.nan, 1.0]], _reference())
