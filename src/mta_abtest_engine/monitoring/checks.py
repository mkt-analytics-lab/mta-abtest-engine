"""Model checks beyond feature drift (ADR 0021). Pure functions.

- PSI of the propensity distribution: prediction drift, available the same day.
- Matured performance: 14 days later the labels exist; compare the mean predicted
  propensity with the observed purchase rate (calibration) and compute log-loss
  and PR-AUC on those sessions.
"""

from __future__ import annotations

import numpy as np

from ..evaluation.metrics import compute_metrics


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index on reference quantile bins (0 = identical).

    Rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-6, None), np.clip(cur, 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def matured_performance(labels: np.ndarray, propensity: np.ndarray) -> dict[str, float]:
    """Calibration ratio (mean predicted / observed rate) and metrics with labels."""
    observed = float(labels.mean())
    out = {
        "observed_rate": observed,
        "mean_propensity": float(propensity.mean()),
        "calibration_ratio": float(propensity.mean() / observed)
        if observed
        else np.inf,
    }
    if 0 < labels.sum() < len(labels):
        out.update(compute_metrics(labels, propensity))
    return out
