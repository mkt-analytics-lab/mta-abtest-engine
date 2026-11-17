"""Prefect flows: order of steps, retries policy and parameters (steps faked)."""

import datetime as dt

import pytest
from prefect.testing.utilities import prefect_test_harness

from mta_abtest_engine.pipelines import flow

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def prefect_backend():
    with prefect_test_harness():
        yield


@pytest.fixture
def calls(monkeypatch):
    """Replace the step functions behind each task and record the call order."""
    log: list[str] = []
    monkeypatch.setattr(flow.dbt_build, "fn", lambda: log.append("dbt"))
    monkeypatch.setattr(flow.extract, "fn", lambda: log.append("extract"))
    monkeypatch.setattr(flow.score, "fn", lambda day: log.append(f"score:{day}") or 7)
    monkeypatch.setattr(flow.train_and_gate, "fn", lambda: log.append("train"))
    monkeypatch.setattr(flow.attribute, "fn", lambda: log.append("attribution"))
    monkeypatch.setattr(flow.publish, "fn", lambda: log.append("publish"))
    return log


def test_daily_scoring_runs_steps_in_order(calls):
    assert flow.daily_scoring(dt.date(2021, 1, 15)) == 7
    assert calls == ["dbt", "extract", "score:2021-01-15"]


def test_daily_scoring_can_skip_dbt_and_defaults_to_yesterday(calls):
    flow.daily_scoring(skip_dbt=True)
    yesterday = dt.date.today() - dt.timedelta(days=1)
    assert calls == ["extract", f"score:{yesterday}"]


def test_weekly_retraining_runs_steps_in_order(calls):
    flow.weekly_retraining()
    assert calls == ["extract", "train", "attribution", "publish"]


def test_warehouse_steps_retry_but_training_does_not():
    for warehouse_task in (flow.dbt_build, flow.extract, flow.score, flow.publish):
        assert warehouse_task.retries == 2
    assert flow.train_and_gate.retries == 0


def test_tasks_call_the_same_functions_as_the_poe_commands(monkeypatch):
    """Each task body delegates to the project function (no duplicated logic)."""
    import mta_abtest_engine.data.extract as extract_mod
    import mta_abtest_engine.pipelines.attribution as attribution_mod
    import mta_abtest_engine.pipelines.propensity as propensity_mod
    import mta_abtest_engine.serving.batch as batch_mod

    seen = []
    monkeypatch.setattr(
        extract_mod, "extract_marts", lambda c, s: seen.append("extract")
    )
    monkeypatch.setattr(
        batch_mod, "run_scoring", lambda c, s, a, b: seen.append(f"score:{a}:{b}") or 3
    )
    monkeypatch.setattr(
        batch_mod, "publish_attribution", lambda c, s: seen.append("publish")
    )
    monkeypatch.setattr(propensity_mod, "run", lambda c, s: seen.append("propensity"))
    monkeypatch.setattr(attribution_mod, "run", lambda c, s: seen.append("attribution"))

    flow.extract.fn()
    assert flow.score.fn(dt.date(2021, 1, 2)) == 3
    flow.publish.fn()
    flow.train_and_gate.fn()
    flow.attribute.fn()
    assert seen == [
        "extract",
        "score:2021-01-02:2021-01-02",
        "publish",
        "propensity",
        "attribution",
    ]


def test_dbt_build_raises_when_dbt_fails(monkeypatch):
    import dbt.cli.main as dbt_main

    class FailingRunner:
        def invoke(self, args):
            assert args[0] == "build"
            return type("Result", (), {"success": False, "exception": "boom"})()

    monkeypatch.setattr(dbt_main, "dbtRunner", FailingRunner)
    with pytest.raises(RuntimeError, match="dbt build failed"):
        flow.dbt_build.fn()
