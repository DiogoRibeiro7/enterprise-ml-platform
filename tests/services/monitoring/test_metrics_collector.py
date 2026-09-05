"""Tests for the bounded, version-aware serving metric contract."""

from __future__ import annotations

import asyncio

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from enterprise_ml_platform.services.monitoring import (
    MonitoringService,
    PredictionEvent,
)
from enterprise_ml_platform.services.monitoring.collectors import MetricsCollector


def _render(registry: CollectorRegistry) -> str:
    """Return a registry in Prometheus' text exposition format."""
    return generate_latest(registry).decode("utf-8")


def test_success_records_request_items_version_and_latency() -> None:
    registry = CollectorRegistry()
    metrics = MetricsCollector(registry)

    metrics.record_prediction(
        "fraud-risk",
        0.025,
        version="7",
        item_count=3,
    )

    output = _render(registry)
    assert (
        'ml_prediction_requests_total{model="fraud-risk",outcome="success",'
        'version="7"} 1.0'
    ) in output
    assert 'ml_predictions_total{model="fraud-risk",version="7"} 3.0' in output
    assert (
        'ml_prediction_latency_seconds_count{model="fraud-risk",'
        'outcome="success",version="7"} 1.0'
    ) in output


def test_error_records_request_and_latency_without_scored_items() -> None:
    registry = CollectorRegistry()
    metrics = MetricsCollector(registry)

    metrics.record_prediction_error("fraud-risk", 0.5, version="8")

    output = _render(registry)
    assert (
        'ml_prediction_requests_total{model="fraud-risk",outcome="error",'
        'version="8"} 1.0'
    ) in output
    assert (
        'ml_prediction_latency_seconds_count{model="fraud-risk",'
        'outcome="error",version="8"} 1.0'
    ) in output
    assert 'ml_predictions_total{model="fraud-risk",version="8"}' not in output


def test_success_rejects_an_empty_item_count() -> None:
    metrics = MetricsCollector(CollectorRegistry())

    with pytest.raises(ValueError, match="item_count must be at least 1"):
        metrics.record_prediction("fraud-risk", 0.01, item_count=0)


def test_monitoring_events_keep_model_version_on_drift_metrics() -> None:
    registry = CollectorRegistry()
    metrics = MetricsCollector(registry)

    class DriftAnalyzer:
        def check(self, *args: object, **kwargs: object) -> dict[str, float]:
            return {"amount": 0.42}

    service = MonitoringService(metrics=metrics, drift_analyzer=DriftAnalyzer())  # type: ignore[arg-type]

    asyncio.run(
        service.handle_event(
            PredictionEvent(
                model_name="fraud-risk",
                model_version="7",
                latency=0.01,
                predicted=1.0,
                features={"amount": 100.0},
            )
        )
    )

    assert (
        'ml_feature_drift_score{feature="amount",model="fraud-risk",version="7"} 0.42'
    ) in _render(registry)
