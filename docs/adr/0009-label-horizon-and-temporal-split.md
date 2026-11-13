# 0009. Label horizon and temporal split

- **Status:** Accepted
- **Date:** 2026-11-13

## Context
Data covers 2020-11-01 to 2021-01-31. A user who arrives near the end has little
time to buy: labelling them "no purchase" creates false negatives (right
censoring). Time from first touch to purchase (4,419 purchasers): median 0.05
days, p75 4.5, p90 19.6, p95 29.8, p99 59 (itself censored).

## Options considered
| Horizon | Purchases captured | Train starts | Test starts | Note |
|---|---|---|---|---|
| 7 days | ~80% | Nov 1 - Dec 24 | Jan 1 - Jan 24 | Loses long, multi-touch journeys |
| **14 days** | ~86% | Nov 1 - Dec 17 | Jan 1 - Jan 17 | Balance |
| 30 days | ~95% | Nov 1 - Dec 1 | none | Infeasible: no test data |

## Decision
- **Label:** purchase within **14 days** of the first touchpoint.
- **Window:** touchpoints at or after `start + 14 days` are dropped (future
  information relative to the label), in `features/journeys.py`.
- **Split by journey start:** train if the label window ends by `test_start`
  (2021-01-01); test if it starts on or after `test_start` and its window ends by
  `data_end` (2021-02-01, exclusive); otherwise excluded (censored).
- Parameters in `configs/config.yaml` (`journey`), validated: the test period must
  be longer than the horizon.

## Rationale
Every label is fully observed and training labels are known before the test period
starts, as they would be in production. 14 days keeps most long journeys, which
matter most for multi-touch attribution.

## Consequences
- ~14% of purchases (slow deciders) count as negatives; revisit with more data.
- Dec 18-31 and Jan 18-31 journey starts are excluded from evaluation.
- Not the same as GA4's lookback window (30 days acquisition / 90 other):
  document it in the GA4 comparison.
- Test period is post-holiday (January) while train includes December: the
  evaluation also measures robustness to seasonality.
