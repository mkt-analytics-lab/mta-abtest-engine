# 0008. Mart extraction to Parquet and data contracts

- **Status:** Accepted
- **Date:** 2026-11-13

## Context
The dbt marts (ADR 0007) live in BigQuery; modelling runs in Python. CI has no
cloud access (ADR 0002, 0004), yet the data contract must be tested.

## Options considered
1. **BigQuery client -> local Parquet** - simple, standard; marts are small
   (~350k touchpoints); Parquet keeps types. Adds `google-cloud-bigquery`,
   `db-dtypes`, `pyarrow`.
2. **EXPORT DATA to Cloud Storage** - scales to huge tables; extra infra (bucket,
   IAM) the data size does not need.
3. **BigQuery DataFrames** - no download; couples modelling to BigQuery.

Test data: a real extract committed to the repo vs **hand-written fixtures**.

## Decision
- Option 1: `data/extract.py` (`uv run poe extract`) reads each mart with an
  explicit column list (the contract columns), under `maximum_bytes_billed`,
  validates it and only then writes `data/raw/*.parquet` (git-ignored).
- Pandera contracts in `data/contracts.py` repeat the key dbt rules at the Python
  boundary (channel values, no label leak, journeys end at the first purchase).
- Hand-written fixtures in `tests/fixtures/` (5 users, edge cases on purpose).
- New dependency group `warehouse` (not in the serving image).

## Rationale
Small data does not justify extra infrastructure. Validating before writing means
an invalid mart never reaches training. Hand-written fixtures make each test case
explicit and avoid licensing questions about committing real data.

## Consequences
- The synthetic template pipeline (`ingest.py`, `RawDataSchema`) stays as the
  walking skeleton until features and model are replaced in phases 2-3.
- Contracts and dbt tests must evolve together when a mart changes.
