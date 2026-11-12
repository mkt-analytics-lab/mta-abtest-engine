-- Q1. Volume per month: events, users and days covered.
-- Used to confirm the date range and plan the temporal train/test split.
select
    format_date('%Y-%m', parse_date('%Y%m%d', event_date)) as month,
    min(parse_date('%Y%m%d', event_date)) as first_day,
    max(parse_date('%Y%m%d', event_date)) as last_day,
    count(distinct event_date) as n_days,
    count(*) as n_events,
    count(distinct user_pseudo_id) as n_users
from {{ source('ga4', 'events') }}
group by month
order by month
