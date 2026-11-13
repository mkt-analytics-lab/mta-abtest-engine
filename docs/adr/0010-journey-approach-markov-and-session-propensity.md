# 0010. Journey approach: Markov over channel-stage states + session propensity

- **Status:** Accepted
- **Date:** 2026-11-13
- **Supersedes:** the model approach and unit of ADR 0002

## Context
ADR 0002 planned a journey-level propensity model with channel credit derived by
ablation (Shao & Li). While designing features two problems appeared:
- Funnel depth (checkout) almost reveals the purchase in the converting session;
  channel-only features predict little.
- This is project 1 of a 3-project portfolio; project 2 is a Marketing Mix Model.
  Channel credit is the MMM's question (budget per channel), and the GA4 sample has
  no spend data, so channel credit cannot become budget decisions here anyway.

The owner's goal from phase 0 is to **optimize the user journey**, not to decide
channel investment.

## Options considered
1. **Channel MTA** - propensity on channel features + channel credit. Textbook MTA,
   direct GA4 comparison; overlaps with the MMM project, weak model signal, little
   operational use.
2. **Journey** - Markov chain over `channel:stage` states (which steps and
   transitions lead to purchase, where users drop) + a session-level propensity
   model (which users to re-engage). Differentiated from MMM, actionable without
   spend data, strong signal, real operational use; more complex.
3. **Both as separate ML models** - complete but double MLOps effort.

## Decision
Option 2.
- **Journey model (attribution):** first-order Markov chain over states
  `channel:funnel_stage_pre_purchase` within the 14-day journey window (ADR 0009).
  Removal effect per state = importance of each journey step; removal of all states
  of a channel = channel credit for the GA4 benchmark (and later for MTA vs MMM
  triangulation). Heuristics (first / last touch) remain as baselines.
- **Propensity model (ML, versioned, served, monitored):** unit = a session that
  ends **without** a purchase (scoring point). Label = purchase within 14 days
  after it. Features use only the journey up to and including that session, so
  funnel stage is a legitimate feature. Business use: daily batch scoring for
  re-engagement (e.g. cart abandoners).
- Split by scoring time with the ADR 0009 rules. Primary metric stays log-loss,
  with PR-AUC, calibration and top-decile lift (ADR 0002 metrics still apply).

## Rationale
Matches the stated goal, keeps MTA and MMM projects complementary, gives the ML
model a real operational job (phases 5-7), and reuses the existing marts, which
already carry channel and stage.

## Consequences
- Users who buy in their first session produce no scoring point: the model cannot
  (and need not) re-engage them.
- Markov states with few observations (rare channels) need smoothing or grouping.
- README must explain "MTA + journey analytics" clearly for the portfolio.
