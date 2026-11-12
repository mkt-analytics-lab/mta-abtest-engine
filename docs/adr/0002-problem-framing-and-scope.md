# 0002. Problem framing and MVP scope

- **Status:** Accepted
- **Date:** 2026-11-11

## Context
The project combines two problems of different nature: user-level Multi-Touch
Attribution (observational) and an A/B test of Target CPA vs Target ROAS bidding
(causal). The owner's goal is to optimize the **customer journey** (where users
drop off, which touchpoints contribute to purchase), not primarily to reallocate
budget across channels. GA4 already offers Data-Driven Attribution (DDA) and last
click; the in-house model must allow independent analysis and comparison with GA4.

Users:
1. **Marketing decision maker** - fixes the journey, values touchpoints, targets users.
2. **MLOps engineer / data team** - evaluates, maintains and promotes model versions.

The initial Kaggle dataset (`multi_touch_attribution_data.csv`, 10k touches,
2,847 users) was rejected: no signal (conversion ~50% for every channel,
campaign, hour and position; chi-square p > 0.6), conversions flagged per touch
rather than per journey, a 48 h window, and no revenue, cost or experiment arm.

## Options considered
Scope: (1) MTA first, A/B later; (2) A/B first; (3) both minimal.
Data: (A) GA4 obfuscated sample e-commerce (BigQuery public); (B) Criteo
attribution dataset; (C) keep the Kaggle dataset.
Model: (1) heuristics + Markov only; (2) propensity model + derived attribution;
(3) sequence / next-best-action model.
Marketing front end: Looker Studio, Streamlit, or both.

## Decision
- **Scope:** MTA module first; A/B (tCPA vs tROAS) is the second module.
- **Data:** GA4 obfuscated sample e-commerce (Nov 2020 - Jan 2021), BigQuery.
  The Kaggle file may be reused as a data-contract test fixture (phase 1).
- **Model:** user-level purchase propensity classifier (logistic regression, then
  gradient boosting); touchpoint credit derived from marginal contribution
  (Shapley / ablation, Shao & Li 2011). **Baselines:** last click, first click and
  a Markov chain with removal effect.
- **Unit and validation:** one row per user journey; temporal split (train on
  earlier months, test on the last one).
- **GA4 benchmark (pluggable):** a fixed contract table
  `model | channel | conversions_credit | revenue_credit`. MVP fills GA4 rows with
  last click recomputed from raw events; in production an adapter loads GA4 DDA
  (Model comparison export or Data API). Channel grouping, key event and lookback
  window must be aligned with GA4.

### Success metrics
| Level | Metric | Role |
|---|---|---|
| ML primary | Log-loss on temporal test | Promotion between versions |
| ML secondary | PR-AUC, calibration (Brier, reliability curve) | Must not degrade |
| Stability | Rank correlation of channel credit between versions | No unexplained reordering |
| Business | Funnel drop-off by step and entry channel; attributed conversions/revenue and share by channel; removal effect; top-decile lift and revenue captured; top converting paths, touches and time to purchase | Marketing decisions |
| Benchmark | Credit difference by channel vs GA4 | Diagnostic only, never a gate |

**Quality gate (provisional):** promote if log-loss beats the baseline on the
temporal test and PR-AUC and calibration do not degrade. Numeric thresholds are
set after EDA in phase 1.

### Product vision (end-to-end)
The project is end-to-end and designed as in a real work context: extraction ->
transformation -> training -> attribution -> serving -> monitoring -> retraining.

- **Marketing face:** Looker Studio dashboard reading result tables in BigQuery
  (funnel, credit by channel vs GA4, removal effect, segments, paths). Pipeline
  outputs are therefore written back to BigQuery.
- **MLOps face:** MLflow UI (versions, champion/challenger), per-version
  evaluation report, Evidently drift reports.
- **Execution:** batch pipeline (e.g. weekly). An online scoring API is optional.
- **Optional, last:** LLM-generated natural-language summary of the results.
- **Maintenance realism:** the GA4 sample is static; new data arrival is
  simulated by replaying the daily `events_YYYYMMDD` partitions incrementally.
- Tool choices per stage (ELT stack, orchestration, deployment target) are
  decided in their phase with their own ADR.

## Rationale
Option 2 gives an objective, version-comparable metric (needed by the MLOps
audience) while still producing touchpoint credit and drop-off insight for
marketing. Log-loss is a proper scoring rule that penalizes miscalibration, which
directly distorts derived credit. GA4 sample is real data with funnel events and
revenue, matching the journey-optimization framing. Looker Studio is what a
marketing team would use with GA4/BigQuery in a real job.

## Consequences
- Attribution is **correlational**, not causal. Disagreement with GA4 is a
  diagnostic, not a verdict; causal validation needs incrementality experiments
  (the A/B module).
- GA4 DDA credits are not in the BigQuery export nor available for the sample
  period: the MVP benchmarks against last click only.
- Ingestion needs BigQuery credentials (owner's GCP project); CI must not depend
  on them (snapshot / fixture strategy decided in phase 1).
- BigQuery sandbox limits (table expiration, restricted features) may require
  enabling billing within the free tier; decided in phase 1.
- Looker Studio is configured by clicks, not code: dashboard definition is
  documented, result tables are the tested contract.
- No public dataset randomizes tCPA vs tROAS: the A/B module will need a
  design-only approach or a semi-synthetic simulation (decided later).
