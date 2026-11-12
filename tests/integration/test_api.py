import joblib
import pytest
from fastapi.testclient import TestClient

from mta_abtest_engine.config import get_settings
from mta_abtest_engine.serving.app import app

pytestmark = pytest.mark.integration

RECORD = {"x1": 1.0, "x2": -0.5, "x3": 0.2, "x4": 0.0}


@pytest.fixture
def client_with_model(tmp_path, monkeypatch, trained_model):
    model_path = tmp_path / "model.joblib"
    joblib.dump(trained_model, model_path)
    monkeypatch.setenv("APP_MODEL_PATH", str(model_path))
    get_settings.cache_clear()
    with TestClient(app) as client:
        yield client
    get_settings.cache_clear()


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_MODEL_PATH", str(tmp_path / "missing.joblib"))
    get_settings.cache_clear()
    with TestClient(app) as client:
        yield client
    get_settings.cache_clear()


def test_health_reports_loaded_model(client_with_model):
    body = client_with_model.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_probabilities(client_with_model):
    response = client_with_model.post("/predict", json={"records": [RECORD] * 3})
    assert response.status_code == 200
    probs = response.json()["probabilities"]
    assert len(probs) == 3
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_invalid_payload_is_rejected(client_with_model):
    response = client_with_model.post("/predict", json={"records": [{"x1": "a"}]})
    assert response.status_code == 422


def test_predict_without_model_returns_503(client_without_model):
    assert client_without_model.get("/health").json()["model_loaded"] is False
    response = client_without_model.post("/predict", json={"records": [RECORD]})
    assert response.status_code == 503
