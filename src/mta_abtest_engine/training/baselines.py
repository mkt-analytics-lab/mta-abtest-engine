"""Baselines for session propensity (ADR 0012). sklearn-style: fit / predict_proba.

0. ``BaseRateModel``: always predicts the training purchase rate.
1. ``FunnelRuleModel``: the analyst's rule "deeper in the funnel = more likely to
   buy", calibrated: P(purchase | max funnel step) estimated on train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BaseRateModel:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseRateModel:
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])


class FunnelRuleModel:
    """Purchase rate per value of one column, with additive smoothing."""

    def __init__(self, column: str = "max_funnel_step", smoothing: float = 1.0):
        self.column = column
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FunnelRuleModel:
        y = pd.Series(np.asarray(y), index=X.index)
        prior = float(y.mean())
        stats = y.groupby(X[self.column]).agg(["sum", "count"])
        # Shrink small groups towards the overall rate.
        self.rates_ = (stats["sum"] + self.smoothing * prior) / (
            stats["count"] + self.smoothing
        )
        self.prior_ = prior
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X[self.column].map(self.rates_).fillna(self.prior_).to_numpy(float)
        return np.column_stack([1 - p, p])
