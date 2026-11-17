"""Registry: challenger registration, promotion, audit tags and rollback."""

import numpy as np
import pandas as pd
import pytest

from mta_abtest_engine.evaluation.behavior import CheckResult
from mta_abtest_engine.evaluation.gate import GateResult
from mta_abtest_engine.features.scoring import FEATURE_COLUMNS
from mta_abtest_engine.pipelines.propensity import select_challenger
from mta_abtest_engine.training import registry
from mta_abtest_engine.training.propensity_models import build_gbm

pytestmark = pytest.mark.integration
NAME = "test-propensity"


@pytest.fixture
def tracking(tmp_path):
    import mlflow

    registry.use_experiment(
        f"sqlite:///{tmp_path / 'mlflow.db'}", "exp", str(tmp_path / "art")
    )
    yield mlflow


@pytest.fixture
def fitted():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(300, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS
    )
    y = (X["max_funnel_step"] + rng.normal(size=300) > 1).astype(int)
    return build_gbm({"max_iter": 20}, seed=0).fit(X, y), X


def _register(mlflow, model, X) -> str:
    with mlflow.start_run() as run:
        run_id = run.info.run_id
    return registry.register_challenger(run_id, model, X.head(10), NAME, {"k": "v"})


def test_register_challenger_and_load_it(tracking, fitted):
    model, X = fitted
    version = _register(tracking, model, X)
    loaded = registry.load_alias(NAME, registry.CHALLENGER)
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))
    assert registry.load_alias(NAME, registry.CHAMPION) is None
    assert version == "1"


def test_promote_moves_champion_and_records_audit(tracking, fitted):
    from mlflow import MlflowClient

    model, X = fitted
    _register(tracking, model, X)
    registry.promote(NAME, "beats funnel rule", user="duvan")
    client = MlflowClient()
    champion = client.get_model_version_by_alias(NAME, registry.CHAMPION)
    assert str(champion.version) == "1"
    assert champion.tags["promoted_by"] == "duvan"
    assert champion.tags["previous_champion"] == "none"
    assert registry.load_alias(NAME, registry.CHALLENGER) is None  # consumed


def test_second_promotion_and_rollback(tracking, fitted):
    from mlflow import MlflowClient

    model, X = fitted
    _register(tracking, model, X)
    registry.promote(NAME, "first", user="a")
    _register(tracking, model, X)
    registry.promote(NAME, "second", user="a")
    client = MlflowClient()
    assert (
        client.get_model_version_by_alias(NAME, "champion").tags["previous_champion"]
        == "1"
    )
    registry.promote(NAME, "rollback: v2 misbehaves", version="1", user="a")
    assert str(client.get_model_version_by_alias(NAME, "champion").version) == "1"


def test_promote_requires_challenger_and_reason(tracking):
    with pytest.raises(ValueError, match="reason"):
        registry.promote(NAME, "  ")
    with pytest.raises(RuntimeError, match="nothing to promote"):
        registry.promote(NAME, "why not")


def _gate(passed: bool, name: str = "gbm", ref: str = "ref") -> GateResult:
    diff = (-0.002, -0.001) if passed else (-0.001, 0.001)
    return GateResult(name, ref, diff, (0.0, 0.01), (-0.001, 0.001))


def _checks(passed: bool) -> list[CheckResult]:
    return [CheckResult("c", "DIR", passed, "")]


SUMMARY = pd.DataFrame(
    {"model": ["gbm", "logistic"], "metric": ["log_loss"] * 2, "value": [0.028, 0.030]}
)


@pytest.mark.parametrize(
    ("gates", "behaviour", "champion_gates", "expected"),
    [
        ({"gbm": True, "logistic": True}, {"gbm": True, "logistic": True}, None, "gbm"),
        (
            {"gbm": True, "logistic": True},
            {"gbm": False, "logistic": True},
            None,
            "logistic",
        ),
        (
            {"gbm": False, "logistic": False},
            {"gbm": True, "logistic": True},
            None,
            None,
        ),
        (
            {"gbm": True, "logistic": True},
            {"gbm": True, "logistic": True},
            {"gbm": False, "logistic": False},
            None,
        ),  # nobody beats the champion
    ],
)
def test_select_challenger(gates, behaviour, champion_gates, expected):
    chosen = select_challenger(
        SUMMARY,
        {m: _gate(p, m) for m, p in gates.items()},
        {m: _checks(p) for m, p in behaviour.items()},
        None
        if champion_gates is None
        else {m: _gate(p, m, "champion") for m, p in champion_gates.items()},
    )
    assert chosen == expected


def test_batch_loads_the_champion_with_its_version(tracking, fitted):
    from mta_abtest_engine.serving.batch import load_champion

    with pytest.raises(RuntimeError, match="No champion"):
        load_champion(NAME)
    model, X = fitted
    _register(tracking, model, X)
    registry.promote(NAME, "ok", user="ci")
    champion, version = load_champion(NAME)
    assert version == "1"
    np.testing.assert_allclose(champion.predict_proba(X), model.predict_proba(X))


def test_export_champion_writes_model_and_metadata(tracking, fitted, tmp_path):
    import json

    import joblib

    from mta_abtest_engine.serving.export import export_champion

    model, X = fitted
    _register(tracking, model, X)
    registry.promote(NAME, "ok", user="ci")
    out = export_champion(NAME, tmp_path / "models" / "model.joblib")
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["registered_model"] == NAME and meta["model_version"] == "1"
    np.testing.assert_allclose(
        joblib.load(out).predict_proba(X), model.predict_proba(X)
    )
