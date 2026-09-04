"""Regression tests for deployable monitoring assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = (
    REPOSITORY_ROOT / "monitoring" / "grafana" / "dashboards" / "model_monitoring.json"
)
PROMETHEUS_CONFIG = REPOSITORY_ROOT / "monitoring" / "prometheus" / "prometheus.yml"


def _dashboard_queries(document: dict[str, Any]) -> list[str]:
    """Return every PromQL expression from a Grafana dashboard."""
    return [
        target["expr"]
        for panel in document["panels"]
        for target in panel.get("targets", [])
        if "expr" in target
    ]


def test_dashboard_queries_version_aware_serving_metrics() -> None:
    document: dict[str, Any] = json.loads(DASHBOARD.read_text(encoding="utf-8"))

    queries = _dashboard_queries(document)

    assert any("ml_predictions_total" in query for query in queries)
    assert any("ml_prediction_requests_total" in query for query in queries)
    assert any("ml_prediction_latency_seconds_bucket" in query for query in queries)
    assert any('outcome="error"' in query for query in queries)
    assert all(
        "model, version" in query for query in queries if "ml_prediction" in query
    )


def test_prometheus_scrapes_the_exported_metrics_route() -> None:
    document: dict[str, Any] = yaml.safe_load(
        PROMETHEUS_CONFIG.read_text(encoding="utf-8")
    )

    scrape_config = next(
        item for item in document["scrape_configs"] if item["job_name"] == "ml-platform"
    )
    assert scrape_config["metrics_path"] == "/api/v1/metrics"
    assert scrape_config["static_configs"] == [{"targets": ["api:8000"]}]
