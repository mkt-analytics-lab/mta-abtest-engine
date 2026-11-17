-- Re-engagement page: scored sessions per day, decile, channel and stage.
-- Probabilities are uncalibrated in January (ADR 0017, 0021): show deciles.
select
    score_date,
    score_decile,
    channel_group,
    funnel_stage,
    model_version,
    count(*) as sessions,
    avg(propensity) as avg_propensity
from {{ source('results', 'propensity_scores') }}
group by score_date, score_decile, channel_group, funnel_stage, model_version
