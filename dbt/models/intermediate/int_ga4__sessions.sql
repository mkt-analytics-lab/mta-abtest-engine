-- One row per session = one touchpoint of the user journey (ADR 0006).
-- Session key: (user_pseudo_id, ga_session_id); ga_session_id alone is a
-- timestamp and collides across users.
with events as (
    select * from {{ ref('stg_ga4__events') }}
),

sessions as (
    select
        concat(user_pseudo_id, '-', cast(ga_session_id as string)) as session_key,
        user_pseudo_id,
        ga_session_id,
        max(ga_session_number) as ga_session_number,
        min(event_ts) as session_start_ts,
        max(event_ts) as session_end_ts,
        min(event_date) as session_date,
        -- medium is empty on session_start: take the first non-null value.
        array_agg(medium ignore nulls order by event_ts limit 1)[safe_offset(0)]
            as first_medium,
        array_agg(source ignore nulls order by event_ts limit 1)[safe_offset(0)]
            as first_source,
        array_agg(campaign ignore nulls order by event_ts limit 1)[safe_offset(0)]
            as first_campaign,
        -- Deepest funnel step reached in the session (0 = browse ... 4 = purchase).
        max(
            case
                when event_name = 'purchase' then 4
                when event_name in (
                    'begin_checkout', 'add_shipping_info', 'add_payment_info'
                ) then 3
                when event_name = 'add_to_cart' then 2
                when event_name in ('view_item', 'select_item') then 1
                else 0
            end
        ) as funnel_step,
        logical_or(event_name = 'purchase') as has_purchase,
        coalesce(sum(purchase_revenue_usd), 0) as purchase_revenue_usd,
        count(*) as n_events
    from events
    group by user_pseudo_id, ga_session_id
)

select
    session_key,
    user_pseudo_id,
    ga_session_id,
    ga_session_number,
    session_start_ts,
    session_end_ts,
    session_date,
    -- Missing channel kept explicit (26% of sessions); grouping rules live in marts.
    coalesce(first_medium, '(missing)') as session_medium,
    first_source as session_source,
    first_campaign as session_campaign,
    funnel_step,
    case funnel_step
        when 4 then 'purchase'
        when 3 then 'checkout'
        when 2 then 'cart'
        when 1 then 'product'
        else 'browse'
    end as funnel_stage,
    has_purchase,
    purchase_revenue_usd,
    n_events
from sessions
