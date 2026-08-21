"""Shared test configuration.

MLflow keeps its tracking and registry URIs in module-level global state and
defaults them to ``./mlruns`` -- a path relative to the current working
directory, which during a test run is the repository root. Every test session
is therefore pinned to a throwaway store so that no test, and no code touched
by a test, can write tracking data into the working tree.
"""

from __future__ import annotations

import os
import pathlib

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolate_mlflow(tmp_path_factory: pytest.TempPathFactory):
    """Point MLflow at a session-scoped throwaway store."""
    mlflow = pytest.importorskip("mlflow")

    root = tmp_path_factory.mktemp("mlflow-session")
    uri = f"sqlite:///{(root / 'session.db').as_posix()}"
    previous_env = {
        key: os.environ.get(key)
        for key in (
            "MLFLOW_TRACKING_URI",
            "MLFLOW_REGISTRY_URI",
            "MLFLOW_ARTIFACT_ROOT",
        )
    }

    os.environ["MLFLOW_TRACKING_URI"] = uri
    os.environ["MLFLOW_REGISTRY_URI"] = uri
    os.environ["MLFLOW_ARTIFACT_ROOT"] = (root / "artifacts").as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)

    yield uri

    for key, value in previous_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


#: Created unconditionally when an MlflowClient is constructed, whatever URI
#: is configured. Harmless while empty -- git does not track empty directories.
TOLERATED_IF_EMPTY = frozenset({"mlruns", "mlartifacts"})


def _is_empty_directory(path: pathlib.Path) -> bool:
    return path.is_dir() and not any(path.iterdir())


def _repository_root_entries(root: pathlib.Path) -> set[str]:
    """Return everything in the repository root worth noticing."""
    return {
        entry.name
        for entry in root.iterdir()
        if not (entry.name in TOLERATED_IF_EMPTY and _is_empty_directory(entry))
    }


@pytest.fixture(scope="session", autouse=True)
def repository_stays_clean(isolate_mlflow):
    """Fail the session if a test writes anything into the repository root.

    Tracking stores, registry databases and artifact directories all default
    to a path relative to the working directory, which during a test run is
    the repository. Comparing the whole directory rather than a list of known
    names means the next component with that habit is caught the first time.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    before = _repository_root_entries(repo_root)

    yield

    created = _repository_root_entries(repo_root) - before
    assert not created, (
        f"tests created {sorted(created)} in the repository root; "
        "point the component at a tmp_path instead of a relative default"
    )
