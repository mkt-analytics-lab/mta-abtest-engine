from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mta_abtest_engine.config import load_config

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
