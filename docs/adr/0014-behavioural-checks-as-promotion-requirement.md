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
- DIR: score does not decrease browse -> product -> cart; checkout > product
  (see update below); two cart sessions >= one.
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

## Update 2026-11-17: checkout dip is real, check corrected
On the real data the GBM failed "deeper stage not lower": single-session scores
browse 0.0044, product 0.0261, cart 0.0692, checkout 0.0567. Raw train data, first
sessions without purchase:

| Stage | Points | Later purchases | Rate |
|---|---|---|---|
| browse | 109,839 | 344 | 0.31% |
| product | 24,490 | 562 | 2.29% |
| cart | 3,163 | 197 | 6.23% |
| checkout | 3,605 | 160 | 4.44% |

Cart minus checkout = +1.8 pp, 95% CI [+0.7, +2.9], p ~ 0.001: a real pattern
(checkout abandoners saw the final price / shipping / payment and left), learned
correctly by the GBM. The logistic model passes only because linearity forces
monotonicity.

Options: (A) correct the check with this evidence; (B) monotonic constraints on
the GBM (distorts a real pattern, worse calibration); (C) keep the check (blocks
the best model on a false assumption). **Decision: A.** The DIR checks become
browse <= product <= cart and checkout > product. Changing a test after seeing a
model fail is only acceptable with evidence independent of the model, as here.
Business insight for the dashboard: checkout friction.
