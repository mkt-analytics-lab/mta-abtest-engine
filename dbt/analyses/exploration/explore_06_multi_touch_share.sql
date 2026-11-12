-- Q6. How many users have a multi-touch journey? Users bucketed by number of
-- sessions and of distinct session channels, split by purchaser / non-purchaser.
-- Attribution only has something to distribute when there are 2+ touchpoints.
with events as (
    select
        user_pseudo_id,
        (select value.int_value from unnest(event_params) where key = 'ga_session_id')
            as ga_session_id,
        event_timestamp,
        event_name,
        (select value.string_value from unnest(event_params) where key = 'medium')
            as medium
    from {{ source('ga4', 'events') }}
),

sessions as (
    select
        user_pseudo_id,
        ga_session_id,
        coalesce(
            array_agg(medium ignore nulls order by event_timestamp limit 1)[
                safe_offset(0)
            ],
            '(missing)'
        ) as session_medium,
        logical_or(event_name = 'purchase') as has_purchase
    from events
    group by user_pseudo_id, ga_session_id
),

users as (
    select
        user_pseudo_id,
        count(*) as n_sessions,
        count(distinct session_medium) as n_channels,
        logical_or(has_purchase) as is_purchaser
    from sessions
    group by user_pseudo_id
)

select
    is_purchaser,
    case
        when n_sessions = 1 then '1'
        when n_sessions = 2 then '2'
        when n_sessions <= 4 then '3-4'
        else '5+'
    end as sessions_bucket,
    if(n_channels = 1, '1 channel', '2+ channels') as channels_bucket,
    count(*) as n_users,
    round(100 * count(*) / sum(count(*)) over (partition by is_purchaser), 2)
        as pct_within_group
from users
group by is_purchaser, sessions_bucket, channels_bucket
order by is_purchaser, sessions_bucket, channels_bucket
