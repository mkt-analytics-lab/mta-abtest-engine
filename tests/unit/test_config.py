from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mta_abtest_engine.config import Settings, load_config

CONFIG_PATH = Path("configs/config.yaml")


def test_project_config_is_valid():
    cfg = load_config(CONFIG_PATH)
    assert 0 < cfg.data.test_size < 1
    assert 0 <= cfg.evaluation.min_roc_auc <= 1


def test_invalid_config_fails_fast(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["data"]["test_size"] = 1.5  # out of range
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError):
        load_config(bad)


def test_bigquery_cost_guard_is_set():
    cfg = load_config(CONFIG_PATH)
    # The whole GA4 sample is 3.34 GiB; the guard must allow it but stay small.
    assert 4 * 2**30 <= cfg.bigquery.maximum_bytes_billed <= 10 * 2**30
    assert cfg.bigquery.location == "US"


def test_invalid_bytes_billed_fails_fast(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["bigquery"]["maximum_bytes_billed"] = 0
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError):
        load_config(bad)


def test_gcp_project_id_comes_from_environment(monkeypatch):
    monkeypatch.setenv("APP_GCP_PROJECT_ID", "my-project")
    assert Settings(_env_file=None).gcp_project_id == "my-project"


def test_gcp_project_id_is_optional(monkeypatch):
    monkeypatch.delenv("APP_GCP_PROJECT_ID", raising=False)
    assert Settings(_env_file=None).gcp_project_id is None


def test_test_period_shorter_than_horizon_fails(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["journey"]["horizon_days"] = 40  # Jan 1 + 40 days > Feb 1
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError):
        load_config(bad)
