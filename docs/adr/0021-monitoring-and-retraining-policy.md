# 0021. Monitoring and retraining policy

- **Status:** Accepted
- **Date:** 2026-11-17

## Context
The champion (ADR 0015) scores sessions daily (ADR 0017). In January it predicts
~1.1% on average while the observed rate is ~0.55% (seasonality, ADR 0009): a model
degrades silently unless something watches it. Labels arrive 14 days late (label
horizon).

## Options considered
1. **Evidently + own checks** - Evidently for feature drift (HTML report and a
   share of drifted columns); own checks for prediction drift (PSI), calibration
   and matured performance; retraining triggers in config.
2. **All own code** - PSI / statistical tests by hand; reinvents Evidently.
3. **Matured performance only** - simplest; problems detected two weeks late.

## Decision
Option 1, enabled with `uvx copier update --data use_monitoring=true`.
`uv run poe monitor --start D [--end ...]`, also the last step of the daily flow.
For day D:
- **Feature drift** (immediate): Evidently on D's scored sessions vs 20k training
  scoring points; HTML in `reports/monitoring/`. Alert if > 30% of features drift.
- **Prediction drift** (immediate): PSI of the propensity vs training predictions.
  Alert if > 0.2.
- **Matured performance** (D - 14 days): calibration ratio = mean predicted /
  observed rate, log-loss, PR-AUC, lift. Alert if the ratio leaves [0.67, 1.5].
- **Retraining recommended** if the calibration alert fires, or feature and
  prediction drift alert together. One row per day in
  `mta_results.monitoring_daily` (partitioned) for Looker.
- Retraining = the weekly flow / `poe propensity`; promotion stays human
  (ADR 0015).

## Dependency note
The template pins `pandas<3` for Evidently. Verified: evidently 0.7.23 fails with
pandas 3.1.1 (removed internal import) but works with 3.0.x, so the project pins
`pandas>=3.0,<3.1` instead (stays on pandas 3). Remove when Evidently is fixed.
Template backlog: same pin, and a drift module independent of the skeleton.

## Consequences
- Expected on this data: calibration alerts in January (over-prediction ~2x):
  the trigger works as designed; the fix is retraining on recent data or
  recalibration.
- Thresholds are conventional starting points (PSI 0.2, 30% drifted features);
  tune them with experience.
