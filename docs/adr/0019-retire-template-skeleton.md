# 0019. Retire the template's synthetic walking skeleton

- **Status:** Accepted
- **Date:** 2026-11-17

## Context
The template ships a synthetic end-to-end pipeline (`x1..x4 -> target`) so a new
project runs on day one. Every piece now has a real replacement:

| Skeleton | Replacement |
|---|---|
| `data/ingest.py`, `data/validation.py` | dbt + `data/extract.py`, `data/contracts.py` |
| `features/build.py` | `features/journeys.py`, `features/scoring.py` |
| `training/train.py`, `evaluation/evaluate.py` | baselines, propensity models, tuning, metrics, gate, behaviour |
| `pipelines/run.py` (`poe pipeline`) | `poe propensity`, `poe attribution`, `poe score` |
| `serving/app.py` (x1..x4) | session propensity API (ADR 0018) |

`poe pipeline` also wrote a synthetic model to `models/model.joblib`, the file baked
into the serving image (ADR 0018).

## Options considered
1. **Keep it** as documentation - confusing (two pipelines), CI time, risk of
   shipping a synthetic model.
2. **Remove it** - git history and the template keep it for reference.

## Decision
Option 2: remove the skeleton modules, their tests, config sections (`model`,
`evaluation`, `tracking`, synthetic `data` paths), the `pipeline` task and the
console script. README, CLAUDE.md and notebooks guide updated.

## Consequences
- `uvx copier update` may try to recreate or modify these files when the template
  changes them: delete them again when reviewing the update diff.
