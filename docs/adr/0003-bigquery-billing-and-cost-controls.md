# 0003. BigQuery with billing (free tier) and cost controls

- **Status:** Accepted
- **Date:** 2026-11-12

## Context
Phase 1 needs BigQuery to read the GA4 public sample and to write intermediate and
result tables (read by Looker Studio). The owner's Google Cloud account is old: no
new-customer credits.

## Options considered
1. **Sandbox** (no card) - zero cost risk; tables expire after 60 days and some
   write/load features are restricted (bad for incremental ETL and a lasting dashboard).
2. **Billing account + free tier** - no expiration or restrictions, like a real job;
   requires a card, pay beyond the free tier (1 TiB queries, 10 GiB storage per month).
3. **Sandbox now, billing later** - no initial commitment; risk of redoing steps.

## Decision
Option 2 in project `mta-abtest-engine`, with three layered cost controls:
1. **Budget alert** 5 EUR, thresholds 50/90/100% - notifies, does not stop spend.
2. **Project quota "Query usage per day" = 0.1 TiB** (104,858 MiB) - hard cap
   enforced by BigQuery.
3. **`maximum_bytes_billed` on every query** (`configs/config.yaml`, ADR 0005) -
   a query that would scan more is rejected before running.

All project datasets are created in location **`US`**, the region of
`bigquery-public-data.ga4_obfuscated_sample_ecommerce` (no cross-region queries).

## Rationale
The GA4 sample is 3.34 GiB, far below the free tier. Real-work setup without the
sandbox limits; three independent guards make an accidental bill unlikely
(worst case with the daily quota: well under 1 USD per day).

## Consequences
- Free tier terms are set by Google and may change; review the pricing page.
- Data residency is US; acceptable for public sample data, revisit for real data.
- The console's "spend cap enforcement" budgets were not used: service coverage
  unverified, and the daily quota already gives a hard cap.
