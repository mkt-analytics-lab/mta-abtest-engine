# 0001. Scaffold the project from mlops-project-template

- **Status:** Accepted
- **Date:** 2026-11-12

## Context
Every project needs the same foundations (structure, testing, CI, packaging,
serving) and should follow `docs/mlops_blueprint.md`. Rebuilding them by hand
is slow and inconsistent.

## Decision
Generate the project with Copier from `mlops-project-template` (version
recorded in `.copier-answers.yml`), with Python 3.13.

Optional components enabled at creation:
- DVC: no
- dbt + DuckDB: no
- Prefect: no
- docker compose + MLflow server: no
- Evidently monitoring: no

## Rationale
- Consistent structure across projects; template improvements can be pulled
  in with `copier update`.
- A runnable walking skeleton from day one (pipeline, tests, API, CI), so
  the work focuses on the problem, not on plumbing.

## Consequences
- Components are activated only when needed, each with its own ADR.
- Template updates must be reviewed (`git diff`) before committing.
