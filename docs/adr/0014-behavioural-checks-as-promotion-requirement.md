# 0014. Behavioural checks as a promotion requirement

- **Status:** Accepted
- **Date:** 2026-11-13

## Context
The blueprint requires behavioural tests (MFT / INV / DIR) before a model is
done. Metrics on January say how good a model is, not whether it is right for
the right reasons. CI has no real data, so the real model cannot be tested there.

## Options considered
1. **pytest only, on a model trained on synthetic data** - runs in CI; tests the
   check logic, not the real model.
2. **Checks only inside the pipeline, on the real model** - tests what matters;
   nothing protects the check code itself.
3. **Both: checks as library code, unit-tested on synthetic models in CI (a good
   one passes, a backwards one is caught) and run on every real candidate.**

## Decision
Option 3. `evaluation/behavior.py` builds hand-made journeys, converts them with the
production feature function and checks:
- MFT: a single browse-only session scores below the train base rate; an engaged
  journey ending in checkout scores above it.
- DIR: score does not decrease browse -> product -> cart -> checkout; checkout >
  browse; two cart sessions >= one.
- INV: shifting a journey 40 days in calendar time does not change the score.
Results are logged in MLflow (`behavior_passed` tag, `behavior_checks.json`).
**A model is promotable only if it passes the metric gate (ADR 0012) and all
behavioural checks** (enforced in phase 4).

## Rationale
Testing the checks against a deliberately reversed model proves they can fail;
running them on the real candidate makes promotion depend on sensible behaviour.

## Consequences
- Strict monotonicity can fail on a tree model through noise; if so, consider
  monotonic constraints on funnel features rather than relaxing the check.
