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

## Update 2026-11-17: first run, two findings, two changes
First run (Jan 15-31): calibration alerts every day (expected, ratio 1.4-4.7),
plus two findings.

1. **Unstable daily calibration.** Jan 16 gave a ratio of 26.0: it evaluates
   Jan 2 sessions, with almost no purchases. **Change:** pool the last
   `calibration_window_days` (7) matured days and require
   `min_matured_positives` (20) purchases; otherwise no calibration verdict.
2. **Tracking change on Jan 21-22.** Feature drift jumped from 12-24% to 36-44%
   and propensity PSI from ~0.1 to 0.5-0.7. Share of non-purchase sessions by
   channel (local Parquet):

   | Channel | Jan 17-20 | Jan 22-26 |
   |---|---|---|
   | Direct | ~11% | ~0.7% |
   | Obfuscated | ~10% | ~1% |
   | Paid Search | ~2.4% | ~0% |
   | Organic Search | ~27% | ~43% |
   | Affiliates | ~0.2% | ~2% |
   | Referral / Unknown | ~20% / ~30% | ~23% / ~29% |

   A 15x drop of Direct overnight, offset by Organic, is a reclassification of
   source / medium (tracking, consent or processing change in the GA4 property or
   the public sample), not customer behaviour. Cause not verifiable without the
   GA4 property. Channel credit after Jan 21 is not comparable with before.
   **Change:** a data-quality check, total variation distance between the day's
   channel shares and training (`channel_mix_tvd_max` 0.15). When it fires the
   action is `check_tracking` and retraining is **not** recommended (retraining
   on broken data would bake the error into the model). New columns:
   `channel_mix_tvd`, `channel_max_change`, `alert_channel_mix`, `action`
   (`none` / `retrain` / `check_tracking`).
