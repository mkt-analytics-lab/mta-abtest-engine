"""Batch scoring and publishing without BigQuery or MLflow (fakes)."""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mta_abtest_engine.config import Settings, load_config
from mta_abtest_engine.data.contracts import validate_mart
from mta_abtest_engine.serving import batch, publish


class StepModel:
    """Propensity = 0.01 * (1 + current funnel step)."""

    def predict_proba(self, X):
        p = 0.01 * (1 + X["cur_funnel_step"].to_numpy())
        return np.column_stack([1 - p, p])


class FakeJob:
    def result(self):
        return self


class FakeClient:
    def __init__(self):
        self.loads, self.datasets = [], []

    def create_dataset(self, ds, exists_ok):
        self.datasets.append((ds.dataset_id, ds.location, exists_ok))

    def load_table_from_dataframe(self, df, destination, job_config):
        self.loads.append((df.copy(), destination, job_config))
        return FakeJob()


@pytest.fixture
def tp(touchpoints_df):
    return validate_mart("fct_touchpoints", touchpoints_df)


def test_scores_only_sessions_of_the_day_without_purchase(tp):
    # 2020-11-07: u2's converting session only -> nothing to score.
    assert batch.score_day(tp, dt.date(2020, 11, 7), StepModel(), "m", "1").empty
    scores = batch.score_day(tp, dt.date(2020, 11, 5), StepModel(), "m", "1")
    assert scores["session_key"].tolist() == ["u2-2002"]
    assert list(scores.columns) == batch.SCORE_COLUMNS
    assert scores.loc[0, "propensity"] == pytest.approx(0.02)  # product step


def test_scoring_never_uses_later_history(tp):
    day = dt.date(2021, 1, 4)
    full = batch.score_day(tp, day, StepModel(), "m", "1")
    cut = tp[tp["session_start_ts"] < pd.Timestamp("2021-01-05", tz="UTC")]
    alone = batch.score_day(cut, day, StepModel(), "m", "1")
    assert full["propensity"].tolist() == alone["propensity"].tolist()


def test_deciles_rank_the_day():
    rows = [
        {
            "session_key": f"s{i}",
            "user_pseudo_id": f"u{i}",
            "touch_position": 1,
            "session_start_ts": pd.Timestamp("2021-01-10 10:00", tz="UTC"),
            "channel_group": "Referral",
            "funnel_step": i % 4,
            "funnel_stage": "browse",
            "is_conversion_touch": False,
        }
        for i in range(20)
    ]
    scores = batch.score_day(
        pd.DataFrame(rows), dt.date(2021, 1, 10), StepModel(), "m", "1"
    )
    top = scores.loc[scores["score_decile"] == 1, "propensity"]
    assert top.min() >= scores["propensity"].max() - 1e-12
    assert sorted(scores["score_decile"].unique()) == list(range(1, 11))


def test_write_partition_is_idempotent_per_day():
    client = FakeClient()
    df = pd.DataFrame({"score_date": [dt.date(2021, 1, 10)], "x": [1]})
    dest = publish.write_partition(
        client, df, "p.d.t", dt.date(2021, 1, 10), "score_date"
    )
    _, destination, config = client.loads[0]
    assert dest == destination == "p.d.t$20210110"
    assert config.write_disposition == "WRITE_TRUNCATE"
    assert config.time_partitioning.field == "score_date"
    assert "ALLOW_FIELD_ADDITION" in config.schema_update_options


def test_run_scoring_writes_one_partition_per_day(tmp_path, monkeypatch, tp):
    cfg = load_config(Path("configs/config.yaml"))
    data = tmp_path / "tp.parquet"
    tp.to_parquet(data)
    cfg.data.touchpoints_path = data
    monkeypatch.setattr(batch, "load_champion", lambda name: (StepModel(), "7"))
    client = FakeClient()
    settings = Settings(_env_file=None, gcp_project_id="proj")
    total = batch.run_scoring(
        cfg, settings, dt.date(2021, 1, 3), dt.date(2021, 1, 10), client=client
    )
    assert total == 2  # u5's two sessions (Jan 4 and Jan 9)
    assert [d for _, d, _ in client.loads] == [
        "proj.mta_results.propensity_scores$20210104",
        "proj.mta_results.propensity_scores$20210109",
    ]
    assert client.datasets == [("mta_results", "US", True)]
    assert (client.loads[0][0]["model_version"] == "7").all()


def test_publish_attribution_replaces_report_tables(tmp_path):
    cfg = load_config(Path("configs/config.yaml"))
    cfg.attribution.report_dir = tmp_path
    for name in ("attribution_states", "attribution_channels"):
        pd.DataFrame({"a": [1]}).to_csv(tmp_path / f"{name}.csv", index=False)
    client = FakeClient()
    written = batch.publish_attribution(
        cfg, Settings(_env_file=None, gcp_project_id="proj"), client=client
    )
    assert written == [
        "proj.mta_results.attribution_states",
        "proj.mta_results.attribution_channels",
    ]
    assert all(c.write_disposition == "WRITE_TRUNCATE" for _, _, c in client.loads)


def test_missing_project_fails_clearly():
    cfg = load_config(Path("configs/config.yaml"))
    with pytest.raises(RuntimeError, match="APP_GCP_PROJECT_ID"):
        batch.bigquery_client(cfg, Settings(_env_file=None, gcp_project_id=None))
