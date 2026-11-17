# Looker Studio dashboard - build guide

The dashboard reads only the **`mta_reporting`** views built by dbt (ADR 0022).
Every chart below maps to one view, so the numbers are versioned SQL, not clicks.

## 0. Prerequisites

```bash
uv run poe dbt-build                 # builds mta_reporting views (needs mta_results)
```

Check in BigQuery that `mta_reporting` contains: `rpt_funnel_daily`,
`rpt_channel_credit`, `rpt_journey_states`, `rpt_scores_daily`, `rpt_monitoring`.

## 1. Create the report and data sources

1. Open https://lookerstudio.google.com -> **Blank report**.
2. **Add data -> BigQuery -> My projects -> `mta-abtest-engine` -> `mta_reporting`**,
   pick `rpt_funnel_daily`, **Add**. Repeat **Resource -> Manage added data
   sources -> Add a data source** for the other four views.
3. In each data source, check field types: dates as *Date*, `*_share`,
   `calibration_ratio`, `psi_propensity`, `channel_mix_tvd` as *Number*.
4. Report title: "MTA - Journey & propensity (GA4 sample, Nov 2020 - Jan 2021)".

Queries are billed to `mta-abtest-engine` and are tiny (views over small tables),
well inside the free tier and the cost guards (ADR 0003).

## 2. Page "Journey & funnel" (marketing) - `rpt_funnel_daily`

| Chart | Setup |
|---|---|
| Scorecards | `sessions`, `reached_cart`, `purchases`, `revenue_usd` (SUM) |
| Funnel (bar chart, horizontal) | Metrics in order: `sessions`, `reached_product`, `reached_cart`, `reached_checkout`, `purchases` (SUM). Shows the biggest drop: product -> cart |
| Drop-off by entry channel (table) | Dimension `channel_group`; metrics as above; add calculated field `cart_rate = SUM(reached_cart) / SUM(reached_product)` |
| Sessions over time (time series) | Dimension `session_date`, breakdown `channel_group`, metric `sessions`. Add a text box on Jan 21-22: *"Tracking change: channel mix not comparable after this date (ADR 0021)"* |
| Controls | Date range control; drop-down on `channel_group` |

## 3. Page "Attribution" - `rpt_channel_credit`, `rpt_journey_states`

| Chart | Setup |
|---|---|
| Credit by model (grouped bar) | Dimension `channel`, breakdown `model`, metric `conversions_credit` (SUM). Markov vs first / last touch vs GA4 proxy |
| Markov vs GA4 proxy (table) | Filter `model = markov`; dimension `channel`; metrics `conversions_share`, `diff_vs_ga4_proxy`. Conditional formatting: green > 0, red < 0 |
| Closing vs assist (stacked bar) | `rpt_journey_states`: dimension `channel`, breakdown `role`, metric `conversions_credit` |
| Top assist steps (table) | Filter `role = assist`; dimension `state`; metrics `removal_effect`, `visits`; sort by `removal_effect` desc, 10 rows |
| Note (text box) | *"Attribution is correlational; validate decisions with experiments. Credit after Jan 21 is affected by the tracking change."* |

## 4. Page "Re-engagement" - `rpt_scores_daily`

| Chart | Setup |
|---|---|
| Sessions in top decile per day (time series) | Filter `score_decile = 1`; dimension `score_date`; metric `sessions` |
| Top decile by stage and channel (pivot table) | Filter `score_decile = 1`; rows `funnel_stage`, columns `channel_group`, metric `sessions` |
| Decile profile (bar) | Dimension `score_decile`; metric `avg_propensity` (AVG) |
| Note (text box) | *"Use deciles, not raw probabilities: the model over-predicts in January (calibration ratio 2-3)."* |

## 5. Page "Model health" (MLOps) - `rpt_monitoring`

| Chart | Setup |
|---|---|
| Action (table) | Dimensions `monitor_date`, `action`, `channel_max_change`; metrics `calibration_ratio`, `channel_mix_tvd`, `psi_propensity`. Conditional formatting on `action`: `retrain` amber, `check_tracking` red |
| Calibration (time series) | Metric `calibration_ratio`; add reference lines at 0.67 and 1.5 (Style -> Reference lines) |
| Drift (time series) | Metrics `share_drifted_features` (ref. line 0.3), `psi_propensity` (0.2), `channel_mix_tvd` (0.15) |
| Scorecards | `model_version` (MAX), last `monitor_date`, SUM of `alert_*` columns |

## 6. Share and keep it current

- **Share -> Manage access**: viewers need access to the report, and the data
  sources use the owner's credentials (default), so they do not need BigQuery access.
- The views read the latest tables: after `poe score`, `poe monitor`,
  `poe publish-attribution` (or the Prefect flows), refresh the report
  (**View -> Refresh data**).
