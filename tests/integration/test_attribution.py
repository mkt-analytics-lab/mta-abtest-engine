"""Attribution pipeline on the hand-written fixture (no BigQuery)."""

from pathlib import Path

import pandas as pd
import pandera.errors as pa_errors
import pytest

from mta_abtest_engine.config import Settings, load_config
from mta_abtest_engine.data.contracts import ChannelCreditSchema, validate_mart
from mta_abtest_engine.pipelines import attribution

pytestmark = pytest.mark.integration


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(Path("configs/config.yaml"))
    cfg.attribution.report_dir = tmp_path
    cfg.attribution.bootstrap_samples = 20
    return cfg


@pytest.fixture
def tp(touchpoints_df):
    return validate_mart("fct_touchpoints", touchpoints_df)


def test_paths_use_channel_and_stage_with_closing_session(cfg, tp):
    paths = attribution.build_paths(tp, cfg).set_index("user_pseudo_id")
    assert paths.loc["u2", "states"] == (
        "Paid Search:browse",
        "Referral:product",
        "Organic Search:closing",
    )
    assert paths.loc["u2", "converted"]


def test_states_are_split_into_closing_and_assist(cfg, tp):
    paths = attribution.build_paths(tp, cfg)
    states, _, _ = attribution.markov_attribution(paths, 20, 0)
    roles = dict(zip(states["state"], states["role"], strict=True))
    assert roles["Organic Search:closing"] == "closing"
    assert roles["Referral:product"] == "assist"
    assert not any(s.endswith(":checkout") and roles[s] == "closing" for s in roles)


def test_markov_credits_add_up_to_conversions(cfg, tp):
    paths = attribution.build_paths(tp, cfg)
    states, channels, p_conv = attribution.markov_attribution(paths, 20, 0)
    assert channels["conversions_credit"].sum() == pytest.approx(
        paths["converted"].sum()
    )
    assert states["conversions_credit"].sum() == pytest.approx(paths["converted"].sum())
    assert 0 < p_conv < 1


def test_benchmark_follows_the_contract(cfg, tp):
    paths = attribution.build_paths(tp, cfg)
    _, markov, _ = attribution.markov_attribution(paths, 20, 0)
    table = attribution.channel_benchmark(paths, markov)
    assert set(table["model"]) == {
        "markov",
        "first_touch",
        "last_touch",
        "ga4_last_click_proxy",
    }


def test_contract_rejects_unknown_channel_and_duplicates():
    bad = pd.DataFrame(
        {
            "model": ["m", "m"],
            "channel": ["TikTok", "TikTok"],
            "conversions_credit": [1.0, 1.0],
            "revenue_credit": [1.0, 1.0],
        }
    )
    with pytest.raises(pa_errors.SchemaErrors):
        ChannelCreditSchema.validate(bad, lazy=True)


def test_run_writes_reports_and_logs(tmp_path, cfg, tp):
    data = tmp_path / "tp.parquet"
    tp.to_parquet(data)
    cfg.data.touchpoints_path = data
    settings = Settings(
        _env_file=None, mlflow_tracking_uri=f"sqlite:///{tmp_path / 'm.db'}"
    )
    attribution.run(cfg, settings)
    assert (tmp_path / "attribution_states.csv").exists()
    assert (tmp_path / "attribution_channels.csv").exists()
