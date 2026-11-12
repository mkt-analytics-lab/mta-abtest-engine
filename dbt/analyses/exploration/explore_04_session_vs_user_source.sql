-- Q4. Session channel vs user acquisition channel (traffic_source).
-- Finding from the first version: `medium` is empty on session_start events, so
-- the session channel is the first non-null `medium` among the session's events.
-- A session is identified by (user_pseudo_id, ga_session_id).
with events as (
    select
        user_pseudo_id,
        (select value.int_value from unnest(event_params) where key = 'ga_session_id')
            as ga_session_id,
        event_timestamp,
        (select value.string_value from unnest(event_params) where key = 'medium')
            as medium,
        traffic_source.medium as user_first_medium
    from {{ source('ga4', 'events') }}
),

sessions as (
    select
        user_pseudo_id,
        ga_session_id,
        array_agg(medium ignore nulls order by event_timestamp limit 1)[safe_offset(0)]
            as session_medium,
        any_value(user_first_medium) as user_first_medium
    from events
    group by user_pseudo_id, ga_session_id
)

select
    session_medium,
    user_first_medium,
    count(*) as n_sessions,
    round(100 * count(*) / sum(count(*)) over (), 2) as pct_sessions,
    session_medium is distinct from user_first_medium as differs
from sessions
group by session_medium, user_first_medium
order by n_sessions desc
