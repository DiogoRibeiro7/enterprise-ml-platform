"""Tests for the MLflow-backed model registry.

Everything runs against a file-based tracking store under ``tmp_path``, so the
tests exercise the real MLflow client without a server and without writing
into the repository.
"""

from __future__ import annotations

import pathlib

import mlflow
import mlflow.sklearn
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from enterprise_ml_platform.services.model_registry import (
    CHALLENGER,
    CHAMPION,
    MLflowModelRegistry,
    ModelRegistryError,
)
from enterprise_ml_platform.services.monitoring.serving_drift import (
    DRIFT_REFERENCE_ARTIFACT,
    DriftReference,
)


@pytest.fixture
def registry(tmp_path: pathlib.Path):
    """A registry on a throwaway SQLite store, with global MLflow state restored."""
    previous_tracking = mlflow.get_tracking_uri()
    previous_registry = mlflow.get_registry_uri()
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    instance = MLflowModelRegistry(tracking_uri=uri, registry_uri=uri)
    mlflow.set_experiment(
        experiment_id=mlflow.create_experiment(
            "tests", artifact_location=(tmp_path / "artifacts").as_uri()
        )
    )
    yield instance
    mlflow.set_tracking_uri(previous_tracking)
    mlflow.set_registry_uri(previous_registry)


@pytest.fixture
def logged_model(registry):
    """Log a trained model and return its artifact URI."""

    def _log(accuracy: float = 0.9) -> str:
        data = load_iris()
        model = LogisticRegression(max_iter=200).fit(data.data, data.target)
        with mlflow.start_run():
            mlflow.log_metrics({"accuracy": accuracy})
            mlflow.log_dict(
                DriftReference.from_array(
                    data.data, [str(name) for name in data.feature_names]
                ).to_dict(),
                DRIFT_REFERENCE_ARTIFACT,
            )
            # Skip dependency inference: it shells out to pip and dominates
            # the runtime of every test in this module.
            return mlflow.sklearn.log_model(
                model, name="model", pip_requirements=["scikit-learn"]
            ).model_uri

    return _log


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------
def test_registering_a_model_assigns_a_version(registry, logged_model) -> None:
    info = registry.register("iris", logged_model(), tags={"team": "risk"})

    assert info.name == "iris"
    assert info.version == "1"
    assert info.run_id, "the version must point back at the run that produced it"
    assert info.tags["team"] == "risk"
    assert info.created_at is not None


def test_versions_increment_and_are_listed_newest_first(registry, logged_model) -> None:
    registry.register("iris", logged_model(accuracy=0.90))
    registry.register("iris", logged_model(accuracy=0.95))

    versions = registry.list_versions("iris")

    assert [v.version for v in versions] == ["2", "1"]


def test_registering_does_not_overwrite_the_previous_version(
    registry, logged_model
) -> None:
    first = registry.register("iris", logged_model())
    second = registry.register("iris", logged_model())

    assert first.version != second.version
    assert registry.get_version("iris", first.version).version == first.version


def test_registered_models_are_listed(registry, logged_model) -> None:
    registry.register("iris", logged_model())
    registry.register("churn", logged_model())

    assert sorted(registry.list_models()) == ["churn", "iris"]


# ----------------------------------------------------------------------
# Promotion
# ----------------------------------------------------------------------
def test_promotion_moves_the_alias_without_touching_versions(
    registry, logged_model
) -> None:
    v1 = registry.register("iris", logged_model(accuracy=0.90))
    v2 = registry.register("iris", logged_model(accuracy=0.95))

    registry.promote("iris", v1.version, CHAMPION)
    assert registry.get_by_alias("iris", CHAMPION).version == v1.version

    registry.promote("iris", v2.version, CHAMPION)
    assert registry.get_by_alias("iris", CHAMPION).version == v2.version
    # Both versions still exist; only the pointer moved.
    assert [v.version for v in registry.list_versions("iris")] == ["2", "1"]


def test_champion_and_challenger_can_point_at_different_versions(
    registry, logged_model
) -> None:
    v1 = registry.register("iris", logged_model())
    v2 = registry.register("iris", logged_model())

    registry.promote("iris", v1.version, CHAMPION)
    registry.promote("iris", v2.version, CHALLENGER)

    assert registry.get_by_alias("iris", CHAMPION).version == v1.version
    assert registry.get_by_alias("iris", CHALLENGER).version == v2.version


def test_rollback_returns_the_alias_to_the_previous_version(
    registry, logged_model
) -> None:
    v1 = registry.register("iris", logged_model())
    v2 = registry.register("iris", logged_model())
    registry.promote("iris", v2.version, CHAMPION)

    rolled_back = registry.rollback("iris", CHAMPION)

    assert rolled_back.version == v1.version
    assert registry.get_by_alias("iris", CHAMPION).version == v1.version


def test_rollback_without_an_older_version_is_refused(registry, logged_model) -> None:
    v1 = registry.register("iris", logged_model())
    registry.promote("iris", v1.version, CHAMPION)

    with pytest.raises(ModelRegistryError, match="no version older"):
        registry.rollback("iris", CHAMPION)


def test_deleting_an_alias_keeps_the_version(registry, logged_model) -> None:
    v1 = registry.register("iris", logged_model())
    registry.promote("iris", v1.version, CHAMPION)

    registry.delete_alias("iris", CHAMPION)

    with pytest.raises(ModelRegistryError):
        registry.get_by_alias("iris", CHAMPION)
    assert registry.get_version("iris", v1.version).version == v1.version


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def test_loading_by_alias_returns_a_usable_model(registry, logged_model) -> None:
    v1 = registry.register("iris", logged_model())
    registry.promote("iris", v1.version, CHAMPION)

    model = registry.load("iris", alias=CHAMPION)

    assert model.predict(load_iris().data[:3]).shape == (3,)


def test_registered_version_loads_its_drift_reference(registry, logged_model) -> None:
    version = registry.register("iris", logged_model())

    reference = registry.load_drift_reference(version)

    assert reference is not None
    assert reference.sample_count == 150
    assert reference.feature_count == 4


def test_loading_follows_the_alias_after_promotion(registry, logged_model) -> None:
    v1 = registry.register("iris", logged_model())
    v2 = registry.register("iris", logged_model())
    registry.promote("iris", v1.version, CHAMPION)
    registry.promote("iris", v2.version, CHAMPION)

    # Serving asks for the alias, never a version number.
    assert registry.resolve_uri("iris", alias=CHAMPION) == "models:/iris@champion"
    assert registry.get_by_alias("iris", CHAMPION).version == v2.version


def test_loading_an_exact_version_ignores_aliases(registry, logged_model) -> None:
    v1 = registry.register("iris", logged_model())
    registry.register("iris", logged_model())
    registry.promote("iris", "2", CHAMPION)

    assert registry.resolve_uri("iris", version=v1.version) == "models:/iris/1"
    assert registry.load("iris", version=v1.version) is not None


# ----------------------------------------------------------------------
# Failure modes
# ----------------------------------------------------------------------
def test_unknown_model_raises_a_typed_error(registry) -> None:
    with pytest.raises(ModelRegistryError):
        registry.get_by_alias("does-not-exist", CHAMPION)


def test_unknown_version_raises_a_typed_error(registry, logged_model) -> None:
    registry.register("iris", logged_model())

    with pytest.raises(ModelRegistryError, match="no version"):
        registry.get_version("iris", "99")


# ----------------------------------------------------------------------
# Aliases survive a listing
# ----------------------------------------------------------------------
def test_list_versions_reports_which_version_holds_the_alias(
    registry, logged_model
) -> None:
    """Regression: search_model_versions returns every alias list empty.

    A caller asking which version is the champion was told that none of them
    was, so a listing could not be used to answer the one question a registry
    listing exists to answer.
    """
    registry.register("iris", logged_model())
    v2 = registry.register("iris", logged_model())
    registry.promote("iris", v2.version, CHAMPION)

    listed = {v.version: v.aliases for v in registry.list_versions("iris")}

    assert CHAMPION in listed[v2.version]
    assert listed["1"] == ()


def test_list_versions_tracks_the_alias_after_it_moves(registry, logged_model) -> None:
    v1 = registry.register("iris", logged_model())
    v2 = registry.register("iris", logged_model())
    registry.promote("iris", v1.version, CHAMPION)
    registry.promote("iris", v2.version, CHALLENGER)

    listed = {v.version: set(v.aliases) for v in registry.list_versions("iris")}
    assert listed[v1.version] == {CHAMPION}
    assert listed[v2.version] == {CHALLENGER}

    registry.promote("iris", v2.version, CHAMPION)

    listed = {v.version: set(v.aliases) for v in registry.list_versions("iris")}
    assert listed[v1.version] == set()
    assert listed[v2.version] == {CHAMPION, CHALLENGER}
