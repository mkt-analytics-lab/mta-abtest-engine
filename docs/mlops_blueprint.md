# MLOps Blueprint

Engineering standard for every project created from `mlops-project-template`.
It describes **what** must be true; the template provides **how** by default.
Deviations are allowed when justified and recorded in an ADR (`docs/adr/`).

## Phases

### 1. ETL / ELT (Data Engineering)
- Extraction, transformation, enrichment and (AI-assisted) synthetic data generation.
- Code in `src/<package>/data/`; SQL transformations in `dbt/` when enabled.
- Raw data is immutable: never edit `data/raw/`, derive new files instead.

### 2. Quality Control & Validation
- **Software:** `pytest` + `ruff` (lint and format). Pre-commit hooks locally, same checks in CI.
  Coverage gate: 85 % minimum.
- **Data:** schema and contract validation (`pandera`) at every ingestion boundary.
- **Models:** behavioural testing – MFT, INV, DIR (`tests/model/`) plus a quality gate on
  held-out metrics (`evaluation/`).
- **Parity:** training-serving skew is prevented by a single feature module used by both
  training and serving, and by a parity test between the API schema and training features.

### 3. Version Control & Model Registry
- Code: git with conventional commits; `uv.lock` committed.
- Experiments and models: MLflow (params, metrics, model artifacts).
- Data and pipelines: DVC (optional component, activate when data must be versioned).
- The template itself is versioned (semver tags + CHANGELOG); projects update with `copier update`.

### 4. Deployment & APIs
- FastAPI + Pydantic contracts; `/health` and `/predict` at minimum.
- Docker multi-stage image with only serving dependencies, non-root user, healthcheck.

### 5. CI/CD, Orchestration & Monitoring
- GitHub Actions: lint + tests + Docker build on every push / PR.
- Dependabot: monthly update PRs for Python deps, Actions and Docker base images.
- Orchestration: task runner (`poe`) first; Prefect when scheduling/retries are needed.
- Monitoring: Evidently drift reports once a model receives new data.

## Definition of done (per change)
- [ ] `uv run poe check` passes locally and in CI
- [ ] New logic has tests in the matching `tests/` folder
- [ ] Significant decisions recorded as ADRs
- [ ] README / CLAUDE.md updated if usage changed
