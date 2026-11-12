-- Q2. Event types and their volume: candidates for the funnel steps.
select
    event_name,
    count(*) as n_events,
    count(distinct user_pseudo_id) as n_users,
    round(100 * count(*) / sum(count(*)) over (), 2) as pct_events
from {{ source('ga4', 'events') }}
group by event_name
order by n_events desc
