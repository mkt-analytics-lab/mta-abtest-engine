# 0015. Model registry and human-approved promotion (champion / challenger)

- **Status:** Accepted
- **Date:** 2026-11-17

## Context
The GBM passes the metric gate (ADR 0012) and all behavioural checks (ADR 0014).
Serving needs one unambiguous "production model"; the MLOps team must see why a
model is in production and be able to roll back.

## Options considered
1. **Automatic promotion** - the pipeline moves `champion` itself; no human in the
   loop, a data problem the gate misses reaches production.
2. **Semi-automatic** - the pipeline registers the best promotable candidate as
   `challenger`; a person runs `poe promote --reason ...`, which moves
   `champion` and records who, when, why and the previous champion.
3. **Manual in the MLflow UI** - no rules, no audit trail.

## Decision
Option 2. Registered model `session-propensity` (MLflow registry, aliases).
- **Promotable** = passes the gate vs the reference rule, every behavioural check
  and, when a champion exists, the gate vs the champion (re-evaluated on the same
  test resamples, without retraining). Best promotable by test log-loss becomes
  `challenger` (one at a time). Nothing promotable -> registry unchanged.
- The registered artifact is **exactly the evaluated model** (fit on train only),
  with an explicit signature (26 float features -> probability), stored with
  skops (safe loading); its own internal types are declared trusted.
- Version tags: git commit, data hash, gate intervals; promotion adds
  `promoted_by`, `promoted_at`, `promotion_reason`, `previous_champion`.
- Rollback: `poe promote --version N --reason ...`.
- Artifacts go to an explicit location (`APP_MLFLOW_ARTIFACT_LOCATION`, default
  `mlartifacts/`; a bucket when a team shares a tracking server).

## Rationale
Automates the mechanical part, keeps the final decision and its justification
with a person: usual for first deployments. Re-evaluating the champion on the
same data makes "better than production" a measured claim.

## Consequences
- Re-running the pipeline with unchanged data and code never creates a new
  challenger (the GBM cannot beat itself).
- Phase 7 may switch to automatic promotion for scheduled retraining once the
  process is trusted.
- Not refitting on train + test trades a little data for traceability.
