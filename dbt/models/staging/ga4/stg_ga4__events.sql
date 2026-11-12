-- Minimal staging of GA4 events: typed columns, only what we need so far.
-- Unnesting of event_params comes in a later increment.
select
    parse_date('%Y%m%d', event_date) as event_date,
    timestamp_micros(event_timestamp) as event_ts,
    event_name,
    user_pseudo_id
from {{ source('ga4', 'events') }}
