# 0011. Scoring-point features

- **Status:** Accepted
- **Date:** 2026-11-13

## Context
The propensity model scores a session that ends without a purchase (ADR 0010).
Features must describe the journey up to that session only, be identical at
training and at scoring time, and be cheap to compute for ~350k points.

## Options considered
1. **Running aggregates over the journey so far** (counts, max stage, recency,
   channel mix) plus the current session - simple, interpretable, vectorized.
2. **Sequence encodings** (last-k states, n-grams) - richer order information;
   many sparse columns, harder to explain.
3. **Learned sequence models** (RNN/transformer) - best order modelling; overkill
   for ~3k positives, opaque, heavy to serve.

## Decision
Option 1, in `features/scoring.py` (`FEATURE_COLUMNS`, fixed order):
- History (cumulative, including the current session): `n_sessions`, `n_channels`,
  `days_since_first_touch`, `hours_since_prev_session` (-1 for the first),
  `max_funnel_step`, sessions reaching product+ / cart+ / checkout+, sessions per
  channel group (8).
- Current session: `cur_funnel_step`, current channel one-hot (8).
Funnel stage is legitimate here: scoring sessions contain no purchase.

## Rationale
Behaviour (funnel depth, recency, frequency) carries most of the signal;
cumulative computation guarantees no leakage (tested by truncating later
sessions). Order information is covered by the Markov model (ADR 0010).

## Consequences
- 26 features, all numeric; a logistic baseline needs scaling.
- Revisit option 2 if the model plateaus and Markov shows order effects matter.
