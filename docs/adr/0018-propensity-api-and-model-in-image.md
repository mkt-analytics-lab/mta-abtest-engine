# 0018. Propensity API with the model baked into the image

- **Status:** Accepted
- **Date:** 2026-11-17

## Context
Second half of ADR 0017 (batch + API). The template's FastAPI service and Docker
image (verified by CI) served the synthetic skeleton model. The API must serve the
registry champion with the same features as training.

## Options considered
How the model reaches the container:
1. **Export to `models/` and mount it as a volume** - matches the template's
   `docker run -v`; changing model = copy + restart; the same container can serve
   a different model if the file changes.
2. **Container reads the MLflow registry** - always the current champion; needs the
   tracking store and artifacts inside the container (a tracking server in prod).
3. **Model baked into the image** - immutable: image tag = model version; rollback
   = deploy the previous image; requires a rebuild per promotion.

## Decision
Option 3.
- `uv run poe export-champion` writes `models/model.joblib` and
  `models/model.meta.json` (registered name, version, run id, feature columns).
- `uv run poe docker-build` tags the image `mta-abtest-engine:model-v<N>` (+ `latest`);
  `docker-run` runs it without mounts.
- API: `POST /score` receives a user's sessions (aware timestamp, channel group,
  funnel stage), sorts them by time, builds features with `build_scoring_features`
  (same code as training and batch) and returns the last session's propensity and
  the model version. `GET /health` reports the loaded version.
- Skew guard: the API refuses to start if the model's feature columns differ from
  the code's `FEATURE_COLUMNS`.
- CI cannot reach the registry: it exports a tiny synthetic model
  (`poe export-test-model`), builds the image and smoke-tests `/health`
  (`model_loaded: true`) and `/score`.

## Rationale
Immutable, versioned images are the production standard (Cloud Run, Kubernetes)
and remove the risk of a silently replaced model file; rebuilding is cheap here.

## Consequences
- The serving image contains no MLflow, Optuna or BigQuery libraries (core +
  `serving` only); verified by running the API in a serving-only environment.
- Propensities are uncalibrated for January (ADR 0017): clients should use ranks
  or thresholds agreed with marketing, not raw probabilities.
- The template's synthetic skeleton (ingest, train, run pipeline) still writes
  `models/model.joblib`: it is retired in the next increment.
