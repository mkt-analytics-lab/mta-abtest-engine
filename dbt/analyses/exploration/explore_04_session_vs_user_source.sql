-- Q4. Session channel (event_params on session_start) vs user acquisition channel
-- (traffic_source). Attribution needs the session channel, not the first one.
with sessions as (
    select
        (select value.string_value from unnest(event_params) where key = 'medium')
            as session_medium,
        traffic_source.medium as user_first_medium
    from {{ source('ga4', 'events') }}
    where event_name = 'session_start'
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
