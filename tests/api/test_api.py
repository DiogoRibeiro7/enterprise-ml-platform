import pathlib
import sys

from fastapi.testclient import TestClient

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from enterprise_ml_platform.api.main import app

client = TestClient(app)
API_KEY = {"X-API-Key": "secret"}


def test_health_endpoint():
    response = client.get("/api/v1/health", headers=API_KEY)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_load_and_predict():
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
