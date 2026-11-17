"""Daily monitoring on synthetic journeys (fake champion and BigQuery client)."""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mta_abtest_engine.config import Settings, load_config
from mta_abtest_engine.pipelines import monitor

pytestmark = pytest.mark.integration


class StepModel:
    def __init__(self, scale: float = 0.02):
        self.scale = scale

    def predict_proba(self, X):
        p = self.scale * (1 + X["cur_funnel_step"].to_numpy())
        return np.column_stack([1 - p, p])


class FakeClient:
    def __init__(self):
        self.loads = []

    def create_dataset(self, ds, exists_ok):
        pass

    def load_table_from_dataframe(self, df, destination, job_config):
        self.loads.append((df.copy(), destination))
        return type("Job", (), {"result": lambda self: None})()


@pytest.fixture
def touchpoints():
    rng = np.random.default_rng(0)
    rows = []
    start = pd.Timestamp("2020-11-01", tz="UTC")
    for u in range(3000):
        first = start + pd.Timedelta(hours=int(rng.integers(0, 92 * 24)))
        n = int(rng.integers(1, 4))
        steps = rng.choice(4, size=n, p=[0.6, 0.25, 0.1, 0.05])
        buys = bool(rng.random() < 0.03 + 0.05 * steps.max())
        for i in range(n):
            last = i == n - 1
            rows.append(
                {
                    "session_key": f"{u}-{i}",
                    "user_pseudo_id": str(u),
                    "touch_position": i + 1,
                    "n_touchpoints": n,
                    "session_start_ts": first + pd.Timedelta(hours=30 * i),
                    "channel_group": rng.choice(
                        ["Organic Search", "Referral", "Direct"]
                    ),
                    "funnel_step": 4 if (buys and last) else int(steps[i]),
                    "funnel_stage": "x",
                    "is_conversion_touch": buys and last,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(Path("configs/config.yaml"))
    cfg.monitoring.report_dir = tmp_path / "monitoring"
    cfg.monitoring.reference_sample = 1000
    return cfg


def test_monitor_writes_one_partition_per_day_with_alerts(tmp_path, cfg, touchpoints):
    data = tmp_path / "tp.parquet"
    touchpoints.to_parquet(data)
    cfg.data.touchpoints_path = data
    client = FakeClient()
    table = monitor.run(
        cfg,
        Settings(_env_file=None, gcp_project_id="p"),
        dt.date(2021, 1, 15),
        dt.date(2021, 1, 16),
        client=client,
        champion=(StepModel(), "1"),
    )
    assert list(table.columns) == monitor.MONITOR_COLUMNS
    assert [d for _, d in client.loads] == [
        "p.mta_results.monitoring_daily$20210115",
        "p.mta_results.monitoring_daily$20210116",
    ]
    row = table.iloc[0]
    assert row["matured_date"] == dt.date(2021, 1, 1)
    assert row["n_scored"] > 0 and row["matured_n"] > 0
    assert (cfg.monitoring.report_dir / "drift_20210115.html").exists()


def test_over_predicting_model_triggers_calibration_alert(tmp_path, cfg, touchpoints):
    data = tmp_path / "tp.parquet"
    touchpoints.to_parquet(data)
    cfg.data.touchpoints_path = data
    table = monitor.run(
        cfg,
        Settings(_env_file=None, gcp_project_id="p"),
        dt.date(2021, 1, 20),
        dt.date(2021, 1, 20),
        client=FakeClient(),
        champion=(StepModel(scale=0.2), "1"),  # predicts far above reality
    )
    assert bool(table.iloc[0]["alert_calibration"])
    assert bool(table.iloc[0]["retrain_recommended"])
