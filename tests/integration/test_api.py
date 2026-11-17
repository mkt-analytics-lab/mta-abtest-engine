"""Session propensity API with an exported (test) model."""

import json

import pytest
from fastapi.testclient import TestClient

from mta_abtest_engine.config import get_settings
from mta_abtest_engine.serving.app import app
from mta_abtest_engine.serving.export import export_test_model

pytestmark = pytest.mark.integration

SESSIONS = [
    {
        "session_start_ts": "2021-01-08T09:00:00Z",
        "channel_group": "Organic Search",
        "funnel_stage": "product",
    },
    {
        "session_start_ts": "2021-01-10T18:30:00Z",
        "channel_group": "Referral",
        "funnel_stage": "cart",
    },
]


@pytest.fixture
def model_path(tmp_path, monkeypatch):
    path = export_test_model(tmp_path / "model.joblib")
    monkeypatch.setenv("APP_MODEL_PATH", str(path))
    get_settings.cache_clear()
    yield path
    get_settings.cache_clear()


@pytest.fixture
def client(model_path):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_MODEL_PATH", str(tmp_path / "missing.joblib"))
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def test_health_reports_model_version(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True, "model_version": "ci-test"}


def test_score_returns_propensity_of_last_session(client):
    body = client.post("/score", json={"sessions": SESSIONS}).json()
    assert 0 <= body["propensity"] <= 1
    assert body["n_sessions"] == 2
    assert body["model_version"] == "ci-test"


def test_session_order_does_not_matter(client):
    a = client.post("/score", json={"sessions": SESSIONS}).json()["propensity"]
    b = client.post("/score", json={"sessions": SESSIONS[::-1]}).json()["propensity"]
    assert a == b  # sorted by time before computing features


@pytest.mark.parametrize(
    "bad_session",
    [
        {**SESSIONS[0], "channel_group": "TikTok"},
        {**SESSIONS[0], "funnel_stage": "purchase"},  # no purchase sessions
        {**SESSIONS[0], "session_start_ts": "2021-01-08T09:00:00"},  # no timezone
        {**SESSIONS[0], "extra": 1},
    ],
)
def test_invalid_sessions_are_rejected(client, bad_session):
    assert client.post("/score", json={"sessions": [bad_session]}).status_code == 422


def test_empty_journey_is_rejected(client):
    assert client.post("/score", json={"sessions": []}).status_code == 422


def test_without_model_api_starts_but_score_is_503(client_without_model):
    assert client_without_model.get("/health").json()["model_loaded"] is False
    response = client_without_model.post("/score", json={"sessions": SESSIONS})
    assert response.status_code == 503


def test_feature_mismatch_refuses_to_start(model_path):
    meta_path = model_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    meta["feature_columns"] = meta["feature_columns"][:-1]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="different features"), TestClient(app):
        pass
