"""Regression tests for deployable monitoring assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_DASHBOARD = (
    REPOSITORY_ROOT / "monitoring" / "grafana" / "dashboards" / "model_monitoring.json"
)
SYSTEM_DASHBOARD = (
    REPOSITORY_ROOT / "monitoring" / "grafana" / "dashboards" / "system.json"
)
GRAFANA_DATASOURCE = (
    REPOSITORY_ROOT / "monitoring" / "grafana" / "datasources" / "datasource.yml"
)
GRAFANA_DASHBOARDS = (
    REPOSITORY_ROOT / "monitoring" / "grafana" / "provisioning" / "dashboards.yml"
)
PROMETHEUS_CONFIG = REPOSITORY_ROOT / "monitoring" / "prometheus" / "prometheus.yml"
PROMETHEUS_RULES = (
    REPOSITORY_ROOT / "monitoring" / "prometheus" / "rules" / "alert.rules.yml"
)
MONITORING_DOCKERFILE = REPOSITORY_ROOT / "docker" / "Dockerfile.monitoring"
DOCKER_COMPOSE = REPOSITORY_ROOT / "docker" / "docker-compose.yml"
KUBERNETES_PROMETHEUS = (
    REPOSITORY_ROOT / "kubernetes" / "monitoring" / "prometheus.yaml"
)
CI_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"


def _dashboard_queries(document: dict[str, Any]) -> list[str]:
    """Return every PromQL expression from a Grafana dashboard."""
    return [
        target["expr"]
        for panel in document["panels"]
        for target in panel.get("targets", [])
        if "expr" in target
    ]


def test_dashboard_queries_version_aware_serving_metrics() -> None:
    document: dict[str, Any] = json.loads(MODEL_DASHBOARD.read_text(encoding="utf-8"))

    queries = _dashboard_queries(document)

    assert any("ml_predictions_total" in query for query in queries)
    assert any("ml_prediction_requests_total" in query for query in queries)
    assert any("ml_prediction_latency_seconds_bucket" in query for query in queries)
    assert any("ml_feature_drift_score" in query for query in queries)
    assert any("ml_feature_drift_detected" in query for query in queries)
    assert any('outcome="error"' in query for query in queries)
    assert any(" or 0 * " in query for query in queries)
    assert all(
        "model, version" in query for query in queries if "ml_prediction" in query
    )
    assert all(
        "model, version, feature" in query
        for query in queries
        if "ml_feature_drift" in query
    )


def test_system_dashboard_reports_the_real_scrape_target_and_api_process() -> None:
    document: dict[str, Any] = json.loads(SYSTEM_DASHBOARD.read_text(encoding="utf-8"))

    queries = _dashboard_queries(document)

    assert document["panels"]
    assert 'up{job="ml-platform"}' in queries
    assert 'scrape_duration_seconds{job="ml-platform"}' in queries
    assert 'process_resident_memory_bytes{job="ml-platform"}' in queries
    assert any("process_cpu_seconds_total" in query for query in queries)


def test_prometheus_scrapes_the_exported_metrics_route() -> None:
    document: dict[str, Any] = yaml.safe_load(
        PROMETHEUS_CONFIG.read_text(encoding="utf-8")
    )

    scrape_config = next(
        item for item in document["scrape_configs"] if item["job_name"] == "ml-platform"
    )
    assert scrape_config["metrics_path"] == "/api/v1/metrics"
    assert scrape_config["static_configs"] == [{"targets": ["api:8000"]}]
    assert document["rule_files"] == ["rules/alert.rules.yml"]


def test_prometheus_alerts_use_exported_version_aware_metrics() -> None:
    document: dict[str, Any] = yaml.safe_load(
        PROMETHEUS_RULES.read_text(encoding="utf-8")
    )

    rules = {
        item["alert"]: item for group in document["groups"] for item in group["rules"]
    }
    assert "ml_prediction_requests_total" in rules["HighErrorRate"]["expr"]
    assert "model, version" in rules["HighErrorRate"]["expr"]
    assert rules["ModelFeatureDrift"]["expr"] == "ml_feature_drift_detected == 1"
    assert rules["ModelFeatureDrift"]["for"] == "5m"
    assert "$labels.feature" in rules["ModelFeatureDrift"]["annotations"]["summary"]


def test_monitoring_image_packages_the_prometheus_config_and_rules() -> None:
    dockerfile = MONITORING_DOCKERFILE.read_text(encoding="utf-8")

    assert "FROM prom/prometheus:v3.14.0" in dockerfile
    assert "WORKDIR /etc/prometheus" in dockerfile
    assert "monitoring/prometheus/prometheus.yml" in dockerfile
    assert "monitoring/prometheus/rules/" in dockerfile


def test_prometheus_history_uses_persistent_storage() -> None:
    compose: dict[str, Any] = yaml.safe_load(DOCKER_COMPOSE.read_text(encoding="utf-8"))
    documents = list(
        yaml.safe_load_all(KUBERNETES_PROMETHEUS.read_text(encoding="utf-8"))
    )

    assert "prometheus_data" in compose["volumes"]
    assert "prometheus_data:/prometheus" in compose["services"]["prometheus"]["volumes"]

    claim = next(item for item in documents if item["kind"] == "PersistentVolumeClaim")
    deployment = next(item for item in documents if item["kind"] == "Deployment")
    pod_spec = deployment["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]

    assert claim["metadata"]["name"] == "prometheus-data"
    assert {"name": "data", "mountPath": "/prometheus"} in container["volumeMounts"]
    assert {
        "name": "data",
        "persistentVolumeClaim": {"claimName": "prometheus-data"},
    } in pod_spec["volumes"]


def test_local_stack_wires_and_persists_every_observability_service() -> None:
    compose: dict[str, Any] = yaml.safe_load(DOCKER_COMPOSE.read_text(encoding="utf-8"))
    services = compose["services"]

    assert set(services) == {"api", "redis", "mlflow", "prometheus", "grafana"}
    assert "version" not in compose
    assert services["api"]["environment"] == {
        "MLP_ENVIRONMENT": "development",
        "MLP_API_KEY": "",
        "MLP_FEATURE_STORE_REDIS_URL": "redis://redis:6379/0",
    }
    assert services["api"]["depends_on"]["redis"]["condition"] == "service_healthy"
    assert services["prometheus"]["depends_on"]["api"]["condition"] == (
        "service_healthy"
    )
    assert all(
        port.startswith("127.0.0.1:")
        for service in services.values()
        for port in service.get("ports", [])
    )
    assert set(compose["volumes"]) == {
        "grafana_data",
        "mlflow_data",
        "prometheus_data",
        "redis_data",
    }
    assert "redis_data:/data" in services["redis"]["volumes"]
    assert "mlflow_data:/mlflow" in services["mlflow"]["volumes"]
    assert "grafana_data:/var/lib/grafana" in services["grafana"]["volumes"]
    assert services["mlflow"]["image"] == "ghcr.io/mlflow/mlflow:v3.16.0"
    assert services["grafana"]["image"] == "grafana/grafana:13.2.1"
    assert services["grafana"]["environment"]["GF_AUTH_ANONYMOUS_ORG_ROLE"] == (
        "Viewer"
    )


def test_grafana_is_provisioned_from_the_versioned_assets() -> None:
    compose: dict[str, Any] = yaml.safe_load(DOCKER_COMPOSE.read_text(encoding="utf-8"))
    datasource: dict[str, Any] = yaml.safe_load(
        GRAFANA_DATASOURCE.read_text(encoding="utf-8")
    )
    provider: dict[str, Any] = yaml.safe_load(
        GRAFANA_DASHBOARDS.read_text(encoding="utf-8")
    )
    grafana = compose["services"]["grafana"]

    assert datasource["datasources"][0]["url"] == "http://prometheus:9090"
    assert datasource["datasources"][0]["isDefault"] is True
    assert provider["providers"][0]["options"]["path"] == (
        "/var/lib/grafana/dashboards"
    )
    assert provider["providers"][0]["editable"] is False
    assert (
        "../monitoring/grafana/datasources/datasource.yml:"
        "/etc/grafana/provisioning/datasources/prometheus.yml:ro"
    ) in grafana["volumes"]
    assert (
        "../monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro"
    ) in grafana["volumes"]


def test_ci_validates_compose_and_builds_both_owned_images() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "docker compose -f docker/docker-compose.yml config --quiet" in workflow
    assert "file: docker/Dockerfile.api" in workflow
    assert "file: docker/Dockerfile.monitoring" in workflow
    assert "type=gha,scope=api" in workflow
    assert "type=gha,scope=prometheus" in workflow
