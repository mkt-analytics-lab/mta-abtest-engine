# 0004. Authentication to Google Cloud: Application Default Credentials

- **Status:** Accepted
- **Date:** 2026-11-12

## Context
Code must read the GA4 public sample and write tables in BigQuery. Credentials
must never live in the repo. The repo is public: anyone can clone it.

## Options considered
1. **ADC with the user's account** (`gcloud auth application-default login`) - no key
   files, Google's recommendation for local dev; code runs with the user's permissions.
2. **Service account JSON key** - simple, but a long-lived secret on disk; discouraged.
3. **ADC impersonating a least-privilege service account** - keyless and least
   privilege; more IAM setup.

## Decision
Option 1. Code uses only Application Default Credentials: no key paths, no
credentials in code. The GCP project id is environment configuration
(`APP_GCP_PROJECT_ID` in `.env`, documented in `.env.example`), never hardcoded.

## Rationale
Keyless and quick for a single developer. Because code only relies on ADC, moving to
option 3 or to Workload Identity Federation (GitHub Actions) needs no code change.

## Consequences
- Someone cloning the repo uses **their own** GCP project (billing, budget, quota),
  runs `gcloud auth application-default login` and sets `APP_GCP_PROJECT_ID`.
  Reading the public GA4 dataset is billed to their project. Alternatively the owner
  grants their Google account BigQuery roles on this project via IAM.
- Without GCP, tests still run: CI and unit tests use local fixtures (ADR 0002).
- Revisit (option 3 or WIF) if CI or a scheduled deployment needs BigQuery.
