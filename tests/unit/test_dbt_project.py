"""Static checks on the dbt project (no BigQuery connection needed)."""

from pathlib import Path

import yaml

from mta_abtest_engine.config import load_config

DBT_DIR = Path("dbt")


def _profile_output() -> dict:
    profiles = yaml.safe_load((DBT_DIR / "profiles.yml").read_text())
    return profiles["mta_abtest_engine"]["outputs"]["dev"]


def test_dbt_cost_guard_matches_project_config():
    # Two places define the per-query cost guard; they must never drift apart.
    cfg = load_config(Path("configs/config.yaml"))
    assert _profile_output()["maximum_bytes_billed"] == (
        cfg.bigquery.maximum_bytes_billed
    )


def test_dbt_location_matches_project_config():
    cfg = load_config(Path("configs/config.yaml"))
    assert _profile_output()["location"] == cfg.bigquery.location


def test_dbt_profile_has_no_hardcoded_project_or_keyfile():
    output = _profile_output()
    assert output["method"] == "oauth"  # Application Default Credentials
    assert "APP_GCP_PROJECT_ID" in output["project"]
    assert "keyfile" not in output
