-- One row per touchpoint (session) of each user journey, cut at the first
-- purchase: sessions after it would leak the label (ADR 0007).
with sessions as (
    select * from {{ ref('int_ga4__sessions') }}
),

channels as (
    select * from {{ ref('channel_groups') }}
),

first_purchase as (
    select
        user_pseudo_id,
        min(session_start_ts) as first_purchase_ts
    from sessions
    where has_purchase
    group by user_pseudo_id
),

journey as (
    select
        s.*,
        c.channel_group
    from sessions as s
    left join channels as c on s.session_medium = c.medium
    left join first_purchase as fp on s.user_pseudo_id = fp.user_pseudo_id
    where fp.first_purchase_ts is null or s.session_start_ts <= fp.first_purchase_ts
)

select
    session_key,
    user_pseudo_id,
    row_number() over (
        partition by user_pseudo_id order by session_start_ts, ga_session_id
    ) as touch_position,
    count(*) over (partition by user_pseudo_id) as n_touchpoints,
    session_start_ts,
    session_date,
    channel_group,
    session_medium,
    session_source,
    session_campaign,
    funnel_step,
    funnel_stage,
    -- Stage without the label: the converting session shows 'checkout' instead of
    -- 'purchase', so features and Markov states do not reveal the outcome.
    if(funnel_stage = 'purchase', 'checkout', funnel_stage) as funnel_stage_pre_purchase,
    has_purchase as is_conversion_touch,
    purchase_revenue_usd
from journey
