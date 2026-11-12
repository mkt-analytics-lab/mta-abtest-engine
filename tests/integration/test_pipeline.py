"""End-to-end smoke test: the whole pipeline runs on a small dataset."""

import shutil
from pathlib import Path

import pytest
import yaml

from mta_abtest_engine.config import get_settings
from mta_abtest_engine.pipelines.run import run

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_full_pipeline_runs(tmp_path, monkeypatch):
    config = yaml.safe_load((PROJECT_ROOT / "configs/config.yaml").read_text())
    config["data"]["n_samples"] = 1000
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APP_MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    get_settings.cache_clear()

    try:
        run("all")
    finally:
        get_settings.cache_clear()

    assert Path("models/model.joblib").exists()
    assert Path("models/model.meta.json").exists()
    assert Path(config["evaluation"]["metrics_path"]).exists()
    shutil.rmtree(tmp_path / "mlruns", ignore_errors=True)
