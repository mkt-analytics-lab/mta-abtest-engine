"""Extraction logic with a fake BigQuery client (no cloud access needed)."""

from pathlib import Path

import pandas as pd
import pytest

from mta_abtest_engine.config import Settings, load_config
from mta_abtest_engine.data.extract import build_query, extract_marts


class FakeJob:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_dataframe(self) -> pd.DataFrame:
        return self._df


class FakeClient:
    """Records each query and its job config; returns the fixture for the table."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.calls: list[tuple[str, object]] = []

    def query(self, sql: str, job_config: object) -> FakeJob:
        self.calls.append((sql, job_config))
        table = next(name for name in self.frames if f".{name}`" in sql)
        return FakeJob(self.frames[table])


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(Path("configs/config.yaml"))
    cfg.data.touchpoints_path = tmp_path / "tp.parquet"
    cfg.data.journeys_path = tmp_path / "uj.parquet"
    return cfg


def test_query_selects_contract_columns_only():
    sql = build_query("proj", "mta_marts", "fct_user_journeys")
    assert "*" not in sql
    assert "`proj.mta_marts.fct_user_journeys`" in sql
    assert "converted" in sql


def test_extract_applies_cost_guard_and_writes_parquet(
    cfg, touchpoints_df, journeys_df
):
    client = FakeClient(
        {"fct_touchpoints": touchpoints_df, "fct_user_journeys": journeys_df}
    )
    settings = Settings(_env_file=None, gcp_project_id="proj")
    paths = extract_marts(cfg, settings, client=client)

    assert len(client.calls) == 2
    for _, job_config in client.calls:
        assert job_config.maximum_bytes_billed == cfg.bigquery.maximum_bytes_billed
    assert len(pd.read_parquet(paths["fct_touchpoints"])) == len(touchpoints_df)
    assert len(pd.read_parquet(paths["fct_user_journeys"])) == len(journeys_df)


def test_invalid_mart_is_not_written(cfg, touchpoints_df, journeys_df):
    bad = touchpoints_df.assign(channel_group="TikTok")
    client = FakeClient({"fct_touchpoints": bad, "fct_user_journeys": journeys_df})
    settings = Settings(_env_file=None, gcp_project_id="proj")
    with pytest.raises(Exception, match="channel_group"):
        extract_marts(cfg, settings, client=client)
    assert not cfg.data.touchpoints_path.exists()


def test_missing_project_id_fails_with_clear_message(cfg):
    with pytest.raises(RuntimeError, match="APP_GCP_PROJECT_ID"):
        extract_marts(cfg, Settings(_env_file=None, gcp_project_id=None))
