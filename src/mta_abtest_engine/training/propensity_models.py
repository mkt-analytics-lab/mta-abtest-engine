"""ML candidates for session propensity (ADR 0012, ladder steps 2 and 3).

No class re-weighting: it would distort probabilities, and the primary metric
(log-loss) and the attribution use need calibrated probabilities.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic(params: dict[str, Any], seed: int) -> Pipeline:
    """Scaled logistic regression: interpretable coefficients per feature."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=seed, **params)),
        ]
    )


def build_gbm(params: dict[str, Any], seed: int) -> HistGradientBoostingClassifier:
    """Gradient boosting: captures interactions and non-linear effects."""
    return HistGradientBoostingClassifier(random_state=seed, **params)
