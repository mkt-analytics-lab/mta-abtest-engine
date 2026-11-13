"""Propensity metrics with user-level bootstrap confidence intervals (ADR 0012).

Rows of the same user are not independent (one row per session), so the
bootstrap resamples USERS and keeps all their rows together. All models are
evaluated on the same resamples, which gives paired differences for the gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

METRICS = ("log_loss", "pr_auc", "brier", "top_decile_lift")
EPS = 1e-6


def top_decile_lift(y: np.ndarray, p: np.ndarray) -> float:
    """Purchase rate in the 10% highest scores divided by the overall rate."""
    n_top = max(1, int(np.ceil(0.1 * len(p))))
    top = np.argsort(-p, kind="stable")[:n_top]
    return float(y[top].mean() / y.mean())


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, EPS, 1 - EPS)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "top_decile_lift": top_decile_lift(y, p),
    }


def bootstrap_metrics(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    groups: np.ndarray,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Metrics per (sample, model) on user-level resamples (long format)."""
    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(codes.max() + 1))
    ends = np.append(starts[1:], len(order))
    n_users = len(starts)

    rows = []
    for sample in range(n_samples):
        users = rng.integers(0, n_users, n_users)
        idx = np.concatenate([order[starts[u] : ends[u]] for u in users])
        if y[idx].sum() == 0:  # no positives: metrics undefined, skip resample
            continue
        for name, p in predictions.items():
            rows.append(
                {"sample": sample, "model": name, **compute_metrics(y[idx], p[idx])}
            )
    return pd.DataFrame(rows)


def summarize(
    y: np.ndarray, predictions: dict[str, np.ndarray], boot: pd.DataFrame
) -> pd.DataFrame:
    """Point estimate and 95% interval per model and metric."""
    rows = []
    for name, p in predictions.items():
        point = compute_metrics(y, p)
        samples = boot[boot["model"] == name]
        for metric in METRICS:
            low, high = np.percentile(samples[metric], [2.5, 97.5])
            rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "value": point[metric],
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def paired_difference(
    boot: pd.DataFrame, model: str, reference: str, metric: str
) -> tuple[float, float]:
    """95% interval of metric(model) - metric(reference) on the same resamples."""
    wide = boot.pivot(index="sample", columns="model", values=metric)
    diff = wide[model] - wide[reference]
    low, high = np.percentile(diff, [2.5, 97.5])
    return float(low), float(high)
