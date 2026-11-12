# GA4 exploration (phase 1)

Read-only queries to learn the GA4 export before designing staging and journeys.
dbt compiles analyses but never materializes them; `dbt show` runs one and prints
the result. They use `source()`, so lineage and the 5 GiB cost guard apply.

```bash
uv run poe dbt-show -s explore_01_volume
```

| File | Question |
|---|---|
| `explore_01_volume` | Events, users and days per month (date range, temporal split) |
| `explore_02_event_names` | Event types and volume (funnel steps) |
| `explore_03_event_params` | Keys in `event_params`, events and value types (session id, source/medium) |
| `explore_04_session_vs_user_source` | Session medium vs `traffic_source` (first-user) medium |
| `explore_05_purchasers` | Purchaser share, revenue, sessions per user (class imbalance) |
