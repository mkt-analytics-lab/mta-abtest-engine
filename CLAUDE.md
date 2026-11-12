# CLAUDE.md – MTA AB Test Engine

Context for AI assistants (Claude) working in this repository.

## Project context

**Problem.** Optimize the customer journey of an e-commerce site: find where
users drop off and which touchpoints contribute to purchase, with an in-house,
versioned attribution model that can be compared with GA4.

**Users.**
1. Marketing decision maker - fixes the journey, values touchpoints, targets
   users (Looker Studio dashboard on BigQuery result tables).
2. MLOps engineer / data team - evaluates, maintains and promotes model
   versions (MLflow, evaluation reports, drift reports).

**Data.** GA4 obfuscated sample e-commerce (`bigquery-public-data`,
Nov 2020 - Jan 2021). Real-work ETL: incremental, idempotent loads; results
written back to BigQuery. Tooling decided per phase with an ADR.

**Model.** User-level purchase propensity classifier; touchpoint credit from
marginal contribution (Shapley / ablation). Baselines: last click, first click,
Markov chain with removal effect. Temporal train/test split.

**Metrics.** Primary: log-loss (promotion between versions). Secondary: PR-AUC,
calibration, channel-credit rank stability. Business: funnel drop-off, credit
by channel, removal effect, top-decile lift, converting paths. GA4 comparison
is a diagnostic, never a gate.

**Scope.** MVP = MTA module. A/B test of Target CPA vs Target ROAS is the
second module. Attribution is correlational; causal claims need experiments.

Details and rationale: `docs/adr/0002-problem-framing-and-scope.md`.

## How we work (collaboration rules)

- This is a learning-oriented project: **work iteratively, step by step**.
  Do not generate the whole project or large blocks of code at once.
- For every design decision: present the options, explain the *why*
  (trade-offs, metrics, methodology) and **wait for the owner's decision**
  before writing code.
- Record every significant decision as an ADR in `docs/adr/` (copy
  `0000-adr-template.md`).
- If something cannot be done (missing access, tool, data), **stop and say
  so** instead of working around it.
- Explanations in Spanish; code, comments, commit messages and docs in English.

## Engineering standards

Follow `docs/mlops_blueprint.md`. Key rules:

- Code lives in `src/mta_abtest_engine/`; notebooks are for exploration only.
- Feature logic lives ONLY in `features/` and is used by training and serving.
- Every data source has a contract in `data/validation.py` with tests.
- Models need behavioural tests (MFT / INV / DIR) in `tests/model/`.
- Parameters go in `configs/config.yaml`; environment values and secrets in
  `.env` (never committed).
- Before declaring work done: `uv run poe check` must pass.

## Commands

```bash
uv sync                  # install
uv run poe check         # lint + tests (same as CI)
uv run poe format        # auto-format
uv run poe pipeline      # full pipeline
uv run poe serve         # API on :8000
uv run poe               # list all tasks
```
