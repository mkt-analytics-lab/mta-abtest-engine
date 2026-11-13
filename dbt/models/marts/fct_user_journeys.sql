-- One row per user: the journey up to the first purchase (or the end of data),
-- its label and readable paths for Markov baselines and Looker (ADR 0007).
with touchpoints as (
    select * from {{ ref('fct_touchpoints') }}
)

select
    user_pseudo_id,
    min(session_start_ts) as journey_start_ts,
    max(session_start_ts) as journey_end_ts,
    count(*) as n_touchpoints,
    count(distinct channel_group) as n_channels,
    logical_or(is_conversion_touch) as converted,
    sum(purchase_revenue_usd) as revenue_usd,
    array_agg(channel_group order by touch_position limit 1)[offset(0)]
        as first_channel,
    array_agg(channel_group order by touch_position desc limit 1)[offset(0)]
        as last_channel,
    string_agg(channel_group, ' > ' order by touch_position) as channel_path,
    string_agg(
        concat(channel_group, ':', funnel_stage_pre_purchase), ' > '
        order by touch_position
    ) as state_path
from touchpoints
group by user_pseudo_id
