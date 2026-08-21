"""End-to-end tests for the API surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from enterprise_ml_platform.api.config import APISettings
from enterprise_ml_platform.api.main import create_app

API_KEY = {"X-API-Key": "test-key"}


@pytest.fixture
def client() -> TestClient:
    """A client for an authenticated app serving the built-in demo model.

    Authentication is enforced whenever an API key is configured, including in
    development, so this exercises the real auth path without needing a model
    registry to stand behind it.
    """
    app = create_app(APISettings(environment="development", api_key="test-key"))
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/health", headers=API_KEY)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_endpoint_requires_the_api_key(client: TestClient) -> None:
    assert client.get("/api/v1/health").status_code == 401


def test_model_load_and_predict(client: TestClient) -> None:
    load_resp = client.post("/api/v1/models/iris/load", headers=API_KEY)
    assert load_resp.status_code == 200
    predict_resp = client.post(
        "/api/v1/predict",
        headers=API_KEY,
        json={"model_name": "iris", "features": [5.1, 3.5, 1.4, 0.2]},
    )
    assert predict_resp.status_code == 200
    data = predict_resp.json()
    assert "predictions" in data and len(data["predictions"]) == 1
