# Architecture Decision Records (ADR)

An ADR is a short note that captures **one** important decision: the context,
the options considered, what was chosen and why. Months later it answers
"why is it built like this?".

## When to write one
- Choosing a tool, model family, data source or storage.
- Activating an optional template component (DVC, dbt, Prefect, ...).
- Deviating from `docs/mlops_blueprint.md`.

## How
1. Copy `0000-adr-template.md` to `NNNN-short-title.md` (next number).
2. Fill it in; status starts as `Proposed`, becomes `Accepted` once decided.
3. Never delete an ADR: if a decision changes, write a new one that
   *supersedes* the old one.
