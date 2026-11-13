"""Quality gate decisions on hand-made bootstrap samples."""

import pandas as pd

from mta_abtest_engine.evaluation.gate import evaluate_gate


def _boot(diff_log_loss: float, diff_pr_auc: float, diff_brier: float) -> pd.DataFrame:
    """Two models on 100 resamples; candidate = reference + diff (+ small noise)."""
    rows = []
    for s in range(100):
        noise = (s - 50) / 50000  # within +-0.001
        rows.append(
            {
                "sample": s,
                "model": "ref",
                "log_loss": 0.03,
                "pr_auc": 0.04,
                "brier": 0.005,
            }
        )
        rows.append(
            {
                "sample": s,
                "model": "cand",
                "log_loss": 0.03 + diff_log_loss + noise,
                "pr_auc": 0.04 + diff_pr_auc + noise,
                "brier": 0.005 + diff_brier + noise / 10,
            }
        )
    return pd.DataFrame(rows)


def test_clear_improvement_passes():
    gate = evaluate_gate(_boot(-0.005, 0.01, 0.0), "cand", "ref")
    assert gate.passed


def test_improvement_within_noise_fails():
    # Interval of the log-loss difference crosses zero: could be noise.
    gate = evaluate_gate(_boot(-0.0002, 0.0, 0.0), "cand", "ref")
    assert not gate.log_loss_improves
    assert not gate.passed


def test_better_log_loss_but_worse_ranking_fails():
    gate = evaluate_gate(_boot(-0.005, -0.03, 0.0), "cand", "ref")
    assert gate.log_loss_improves
    assert not gate.pr_auc_not_worse
    assert not gate.passed


def test_worse_brier_fails():
    gate = evaluate_gate(_boot(-0.005, 0.01, 0.002), "cand", "ref")
    assert not gate.brier_not_worse
    assert not gate.passed
