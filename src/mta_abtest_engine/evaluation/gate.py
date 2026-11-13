"""Quality gate: a candidate vs the reference model on paired resamples (ADR 0012).

Pass only if:
- log-loss improves: the 95% interval of (candidate - reference) is below 0;
- PR-AUC does not get worse: that interval is not entirely below 0;
- Brier does not get worse: that interval is not entirely above 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import paired_difference


@dataclass(frozen=True)
class GateResult:
    candidate: str
    reference: str
    log_loss_diff: tuple[float, float]
    pr_auc_diff: tuple[float, float]
    brier_diff: tuple[float, float]

    @property
    def log_loss_improves(self) -> bool:
        return self.log_loss_diff[1] < 0

    @property
    def pr_auc_not_worse(self) -> bool:
        return self.pr_auc_diff[1] >= 0

    @property
    def brier_not_worse(self) -> bool:
        return self.brier_diff[0] <= 0

    @property
    def passed(self) -> bool:
        return self.log_loss_improves and self.pr_auc_not_worse and self.brier_not_worse


def evaluate_gate(boot: pd.DataFrame, candidate: str, reference: str) -> GateResult:
    return GateResult(
        candidate=candidate,
        reference=reference,
        log_loss_diff=paired_difference(boot, candidate, reference, "log_loss"),
        pr_auc_diff=paired_difference(boot, candidate, reference, "pr_auc"),
        brier_diff=paired_difference(boot, candidate, reference, "brier"),
    )
