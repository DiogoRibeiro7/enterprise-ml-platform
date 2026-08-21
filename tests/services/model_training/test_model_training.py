"""Tests for the model training service, including its MLflow tracking."""

from __future__ import annotations

import asyncio
import pathlib

import mlflow
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from enterprise_ml_platform.services.model_registry import CHAMPION, MLflowModelRegistry
from enterprise_ml_platform.services.model_training import (
    ModelConfig,
    ModelTrainingService,
)


@pytest.fixture
def restore_mlflow_state():
    """MLflow keeps tracking configuration in module-level global state."""
    previous_tracking = mlflow.get_tracking_uri()
    previous_registry = mlflow.get_registry_uri()
    yield
    mlflow.set_tracking_uri(previous_tracking)
    mlflow.set_registry_uri(previous_registry)


def _config() -> ModelConfig:
    return ModelConfig(
        algorithm="ensemble",
        ensemble={
            "estimators": [
                ("lr", LogisticRegression(max_iter=100)),
                ("dt", DecisionTreeClassifier(max_depth=3)),
            ],
            "task": "classification",
            "method": "voting",
        },
    )


def test_training_service_with_voting_ensemble() -> None:
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    service = ModelTrainingService()
    model, metrics = asyncio.run(service.train(X, y, _config()))
    assert metrics["accuracy"] > 0


# ----------------------------------------------------------------------
# Tracking
# ----------------------------------------------------------------------
def test_training_without_a_tracking_uri_writes_nothing(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """Regression: training used to materialise a tracking store in the cwd.

    ``mlflow.log_params`` outside a run makes MLflow open an implicit one,
    which created ``mlflow.db``/``mlruns`` wherever the process happened to be
    running -- the repository root, during a test run.
    """
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.chdir(tmp_path)
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)

    service = ModelTrainingService()
    assert not service.tracking_enabled

    asyncio.run(service.train(X, y, _config()))

    assert list(tmp_path.iterdir()) == [], f"training wrote {list(tmp_path.iterdir())}"


def test_training_logs_into_the_configured_store(
    tmp_path: pathlib.Path, restore_mlflow_state
) -> None:
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)

    service = ModelTrainingService(
        tracking_uri=uri,
        experiment_name="training-tests",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    assert service.tracking_enabled
    _, metrics = asyncio.run(service.train(X, y, _config()))

    assert service.last_run_id, "the run id must be exposed for the registry"
    run = mlflow.get_run(service.last_run_id)
    assert run.data.metrics["accuracy"] == pytest.approx(metrics["accuracy"])
    assert (tmp_path / "mlflow.db").exists()


def test_a_tracked_run_can_be_promoted_through_the_registry(
    tmp_path: pathlib.Path, restore_mlflow_state
) -> None:
    """The lifecycle the platform claims: train -> track -> register -> promote."""
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)

    service = ModelTrainingService(
        tracking_uri=uri,
        experiment_name="lifecycle",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    asyncio.run(service.train(X, y, _config()))
    assert service.last_model_uri, "the model artifact must be logged"

    registry = MLflowModelRegistry(tracking_uri=uri, registry_uri=uri)
    version = registry.register("lifecycle-model", service.last_model_uri)
    registry.promote("lifecycle-model", version.version, CHAMPION)

    served = registry.load("lifecycle-model", alias=CHAMPION)
    assert served.predict(X[:3]).shape == (3,)
    assert (
        registry.get_by_alias("lifecycle-model", CHAMPION).run_id == service.last_run_id
    )
