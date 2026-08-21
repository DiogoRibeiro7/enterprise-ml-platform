"""Tests for the serving endpoints.

The prediction handlers used to call ``model.predict`` directly inside an
``async`` function. Inference is CPU-bound, so that blocked the event loop for
every other in-flight request; the responses also carried no model version, so
a promotion silently changed the answers with no way to tell.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

from enterprise_ml_platform.api.config import APISettings
from enterprise_ml_platform.api.dependencies import LoadedModel, get_registry
from enterprise_ml_platform.api.main import create_app

HEADERS = {"X-API-Key": "test-key"}
IRIS_ROW = [5.1, 3.5, 1.4, 0.2]


@pytest.fixture
def client() -> TestClient:
    """An authenticated app with the demo model already loaded."""
    app = create_app(
        APISettings(environment="development", api_key="test-key", max_batch_size=5)
    )
    client = TestClient(app)
    assert client.post("/api/v1/models/iris/load", headers=HEADERS).status_code == 200
    return client


# ----------------------------------------------------------------------
# Responses identify the model that produced them
# ----------------------------------------------------------------------
def test_single_prediction_reports_the_serving_version(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        headers=HEADERS,
        json={"model_name": "iris", "features": IRIS_ROW},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 1
    assert body["model_name"] == "iris"
    assert body["model_version"] == "demo"
    assert body["latency_ms"] >= 0


def test_batch_prediction_reports_the_serving_version(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict/batch",
        headers=HEADERS,
        json={"model_name": "iris", "items": [IRIS_ROW, IRIS_ROW]},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert body["model_version"] == "demo"


def test_pinning_a_version_that_is_not_served_is_a_conflict(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        headers=HEADERS,
        json={"model_name": "iris", "features": IRIS_ROW, "model_version": "7"},
    )

    assert response.status_code == 409
    assert "demo" in response.json()["detail"]


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------
def test_unloaded_model_returns_404(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        headers=HEADERS,
        json={"model_name": "not-loaded", "features": IRIS_ROW},
    )

    assert response.status_code == 404


def test_wrong_feature_count_is_rejected(client: TestClient) -> None:
    """A misaligned vector must be refused, not scored into a confident answer."""
    response = client.post(
        "/api/v1/predict",
        headers=HEADERS,
        json={"model_name": "iris", "features": [1.0, 2.0]},
    )

    assert response.status_code == 422
    assert "expects 4 features" in response.json()["detail"]


def test_ragged_batch_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict/batch",
        headers=HEADERS,
        json={"model_name": "iris", "items": [IRIS_ROW, [1.0, 2.0]]},
    )

    assert response.status_code == 422


def test_empty_batch_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict/batch",
        headers=HEADERS,
        json={"model_name": "iris", "items": []},
    )

    assert response.status_code == 422


def test_oversized_batch_is_rejected(client: TestClient) -> None:
    """Without a cap, one request can pin a worker for an unbounded time."""
    response = client.post(
        "/api/v1/predict/batch",
        headers=HEADERS,
        json={"model_name": "iris", "items": [IRIS_ROW] * 6},
    )

    assert response.status_code == 413
    assert "exceeds the limit of 5" in response.json()["detail"]


def test_batch_at_the_limit_is_accepted(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict/batch",
        headers=HEADERS,
        json={"model_name": "iris", "items": [IRIS_ROW] * 5},
    )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 5


# ----------------------------------------------------------------------
# The event loop stays free
# ----------------------------------------------------------------------
async def test_inference_runs_off_the_event_loop() -> None:
    """Regression: a blocking predict stalled every other request in the process."""
    loop_thread = threading.current_thread().name
    observed: dict = {}

    class SlowModel:
        n_features_in_ = 4

        def predict(self, data):
            observed["thread"] = threading.current_thread().name
            time.sleep(0.2)
            return [0.0] * len(data)

    from enterprise_ml_platform.api.routers.predictions import _predict

    loaded = LoadedModel(
        name="slow", version="1", model=SlowModel(), source="registry", n_features=4
    )

    ticks = 0
    inference = asyncio.create_task(_predict(loaded, [IRIS_ROW]))
    while not inference.done():
        await asyncio.sleep(0.01)
        ticks += 1
    await inference

    assert observed["thread"] != loop_thread, "inference ran on the event loop thread"
    assert ticks > 1, "the event loop was blocked while inference ran"


def test_concurrent_requests_are_not_serialised_by_one_slow_model() -> None:
    """A slow prediction must not stop the server answering anything else."""
    app = create_app(APISettings(environment="development"))

    class SlowModel:
        n_features_in_ = 4

        def predict(self, data):
            time.sleep(0.3)
            return [0.0] * len(data)

    registry = get_registry()
    registry._models["slow"] = LoadedModel(
        name="slow", version="1", model=SlowModel(), source="registry", n_features=4
    )

    results: dict = {}

    with TestClient(app) as client:

        def call_predict() -> None:
            started = time.perf_counter()
            client.post(
                "/api/v1/predict",
                json={"model_name": "slow", "features": IRIS_ROW},
            )
            results["predict"] = time.perf_counter() - started

        worker = threading.Thread(target=call_predict)
        worker.start()
        time.sleep(0.05)  # let the slow prediction get going

        started = time.perf_counter()
        health = client.get("/api/v1/health")
        results["health"] = time.perf_counter() - started
        worker.join()

    assert health.status_code == 200
    assert results["health"] < 0.25, (
        f"health took {results['health']:.2f}s while a prediction was in flight; "
        "the event loop was blocked"
    )
