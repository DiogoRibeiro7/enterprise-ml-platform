"""Concurrency regressions for API dependency initialization."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from pytest import MonkeyPatch

from enterprise_ml_platform.api import dependencies
from enterprise_ml_platform.services.model_registry import ModelVersionInfo
from enterprise_ml_platform.services.monitoring.serving_drift import DriftReference


def test_metrics_collector_is_initialized_once_under_concurrency(
    monkeypatch: MonkeyPatch,
) -> None:
    """Concurrent first requests must not register duplicate timeseries."""
    constructed: list[object] = []

    class SlowMetricsCollector:
        def __init__(self) -> None:
            time.sleep(0.02)
            constructed.append(self)

    monkeypatch.setattr(dependencies, "_metrics", None)
    monkeypatch.setattr(dependencies, "MetricsCollector", SlowMetricsCollector)

    with ThreadPoolExecutor(max_workers=8) as executor:
        instances: list[Any] = list(
            executor.map(lambda _: dependencies.get_metrics(), range(8))
        )

    assert len(constructed) == 1
    assert len({id(instance) for instance in instances}) == 1


def test_alias_is_resolved_once_before_loading_versioned_artifacts() -> None:
    """An alias move cannot mix a model from one version with another baseline."""
    reference = DriftReference.from_array([[0.0], [1.0]], ["amount"])
    version = ModelVersionInfo(
        name="fraud", version="7", source="models:/fraud/7", run_id="run-7"
    )
    calls: list[tuple[str, str]] = []

    class Backend:
        def get_by_alias(self, name: str, alias: str) -> ModelVersionInfo:
            calls.append(("alias", alias))
            return version

        def load(self, name: str, *, version: str) -> object:
            calls.append(("load", version))
            return object()

        def load_drift_reference(self, resolved: ModelVersionInfo) -> DriftReference:
            calls.append(("reference", resolved.version))
            return reference

    registry = dependencies.ModelRegistry(backend=cast(Any, Backend()))

    info = registry.load("fraud")
    loaded = registry.get("fraud")

    assert info.version == "7"
    assert loaded is not None
    assert loaded.drift_reference == reference
    assert loaded.n_features == 1
    assert calls == [("alias", "champion"), ("load", "7"), ("reference", "7")]


def test_drift_monitor_is_initialized_once_under_concurrency(
    monkeypatch: MonkeyPatch,
) -> None:
    constructed: list[object] = []

    class SlowDriftMonitor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            time.sleep(0.02)
            constructed.append(self)

    monkeypatch.setattr(dependencies, "_drift_monitor", None)
    monkeypatch.setattr(dependencies, "ServingDriftMonitor", SlowDriftMonitor)
    monkeypatch.setattr(dependencies, "get_metrics", lambda: object())
    monkeypatch.setattr(dependencies, "get_settings", dependencies.APISettings)

    with ThreadPoolExecutor(max_workers=8) as executor:
        instances: list[Any] = list(
            executor.map(lambda _: dependencies.get_drift_monitor(), range(8))
        )

    assert len(constructed) == 1
    assert len({id(instance) for instance in instances}) == 1
