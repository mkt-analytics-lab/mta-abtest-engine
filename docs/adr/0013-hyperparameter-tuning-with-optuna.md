# 0013. Hyperparameter tuning with Optuna

- **Status:** Accepted
- **Date:** 2026-11-13

## Context
The default GBM passes the gate (ADR 0012) with untuned hyperparameters. Tuning
must never look at the test set (January). The owner wants the tuning setup to be
reusable with a real client (more data, features and seasonal cycles).

## Options considered
1. **No tuning** - fastest; misses a core MLOps practice.
2. **Random search** (sklearn) - correct protocol, no new dependency; inefficient
   when the search space or training time grows.
3. **Optuna (TPE + pruning)** - adaptive search, early stopping of bad trials,
   resumable studies; one light dependency.

## Decision
Option 3, `training/tuning.py` (`uv run poe tune`):
- **Temporal validation inside train:** sub-train = scoring points whose label
  window ends before `validation_start` (2020-12-01); validation = points from
  that date whose window ends before `test_start`. Validation log-loss is the
  objective.
- **Search space and budget in `configs/config.yaml`** (`tuning`): learning rate,
  iterations, leaves, min samples per leaf, L2; `n_trials`, `timeout_seconds`.
  `early_stopping` fixed off (the temporal split replaces sklearn's random one).
- **Pruning:** the GBM grows with warm start; validation loss is reported every
  `pruning_step` iterations and the MedianPruner stops weak trials.
- **Reproducible and resumable:** TPE seeded; study stored in `optuna.db`.
- **Tracking:** parent MLflow run per study, one nested run per trial.
- **Promotion path:** the best params are refit on the whole train, evaluated once
  on the test set and gated against the current GBM (`gbm_tuned` vs `gbm`).

## Rationale
Same protocol as random search, but scales to a real client's data. Config-driven
search space means a new client changes YAML, not code.

## Consequences
- Validation is December (high season) while the test is January (trough): tuned
  parameters may not transfer; the gate decides.
- With 350 test positives, small gains may not be distinguishable: expected.
- With more data, switch to several temporal validation windows (average loss).
