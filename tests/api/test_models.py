"""Regression tests for model lifecycle and drift-monitor coordination."""

from __future__ import annotations

from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, generate_latest

from enterprise_ml_platform.api.config import APISettings
from enterprise_ml_platform.api.dependencies import (
    LoadedModel,
    get_drift_monitor,
    get_registry,
)
from enterprise_ml_platform.api.main import create_app
from enterprise_ml_platform.api.schemas import ModelInfo
from enterprise_ml_platform.services.monitoring.collectors import MetricsCollector
from enterprise_ml_platform.services.monitoring.serving_drift import (
    DriftReference,
    ServingDriftMonitor,
)

HEADERS = {"X-API-Key": "test-key"}


def test_reloading_same_version_without_reference_clears_stale_drift() -> None:
    reference = DriftReference.from_array([[0.0], [1.0]], ["amount"])
    old = LoadedModel(
        name="fraud",
        version="7",
        model=object(),
        source="registry",
        n_features=1,
        drift_reference=reference,
    )

    class ReloadingRegistry:
        current = old

        def get(self, name: str) -> LoadedModel | None:
            return self.current if name == "fraud" else None

        def load(
            self,
            name: str,
            *,
            alias: str | None = None,
            version: str | None = None,
        ) -> ModelInfo:
            self.current = LoadedModel(
                name=name,
                version="7",
                model=object(),
                source="registry",
                n_features=1,
                drift_reference=None,
            )
            return ModelInfo(name=name, version="7")

    metrics_registry = CollectorRegistry()
    monitor = ServingDriftMonitor(
        MetricsCollector(metrics_registry), window_size=4, min_samples=2
    )
    monitor.observe("fraud", "7", [[10.0], [11.0]], reference)
    registry = ReloadingRegistry()
    app = create_app(APISettings(environment="development", api_key="test-key"))
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_drift_monitor] = lambda: monitor

    response = TestClient(app).post("/api/v1/models/fraud/load", headers=HEADERS)

    assert response.status_code == 200
    assert monitor.status("fraud", "7").state == "unavailable"
    assert 'model="fraud"' not in generate_latest(metrics_registry).decode("utf-8")
