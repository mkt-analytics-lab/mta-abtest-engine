# 0012. Model ladder, evaluation with user-level bootstrap, and quality gate

- **Status:** Accepted
- **Date:** 2026-11-13

## Context
Session propensity (ADR 0010-0011). Test set (Jan 1-17): 63,469 scoring points,
350 positives, post-holiday base rate 0.55% vs 1.73% in train (ADR 0009). Rows of
a user are correlated (one row per session). ADR 0002 left the gate provisional.

## Options considered
Gate reference: (a) beat the base rate only - too easy; (b) **beat a calibrated
heuristic rule** - what an analyst would do without ML; (c) absolute thresholds -
no basis to set them under the seasonal shift.
Uncertainty: (a) point estimates; (b) row-level bootstrap - overconfident with
correlated rows; (c) **user-level bootstrap with paired differences**.

## Decision
- **Ladder:** 0 base rate -> 1 funnel rule (P(purchase | max funnel step), smoothed)
  -> 2 logistic regression -> 3 gradient boosting (sklearn HistGradientBoosting).
- **Metrics on test:** log-loss (primary), PR-AUC, Brier, top-decile lift, each
  with a 95% interval from a user-level bootstrap (200 resamples, all models on
  the same resamples).
- **Gate (vs funnel rule, paired):** promote only if the 95% interval of the
  log-loss difference is entirely below 0, the PR-AUC difference interval does
  not lie entirely below 0, and Brier does not get worse beyond its interval.
- Every model is an MLflow run (experiment `session-propensity`) with params,
  metrics + intervals and trace tags (git commit, data hash). Registry and
  champion/challenger aliases come in phase 4.

## Rationale
A relative gate is fair under the seasonal shift (all models face January). Paired,
user-level resampling gives honest uncertainty with a small, correlated test set.

## Consequences
- With 350 positives, small real improvements may not pass: accepted, it avoids
  promoting noise.
- The synthetic template pipeline is retired once the ML models are in place.

## Update 2026-11-13: ML candidates
- `logistic`: StandardScaler + LogisticRegression (C=1). `gbm`:
  HistGradientBoosting (learning_rate 0.05, up to 300 iterations with early
  stopping, 15 leaves, min 100 samples per leaf, L2 1.0). Parameters in
  `configs/config.yaml` (`propensity`), no class re-weighting (keeps probabilities
  calibrated). Hyperparameters are sensible defaults, not tuned: tuning must use a
  validation period inside train, never the test set.
- Gate implemented in `evaluation/gate.py`; results logged as MLflow tags
  (`gate_passed`, `gate_reference`) and metrics (difference intervals).
