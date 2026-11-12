# CLAUDE.md – MTA AB Test Engine

Context for AI assistants (Claude) working in this repository.

## Project context

<!-- TODO: replace this section with the project description:
     business problem, users, data sources, success metrics, scope. -->
User-level multi-touch attribution and journey optimization

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
