-- Staging of GA4 events: one row per event, typed columns and the event_params
-- we need, extracted from the nested array. Only needed columns are read
-- (cost and data minimisation).
select
    parse_date('%Y%m%d', event_date) as event_date,
    timestamp_micros(event_timestamp) as event_ts,
    event_name,
    user_pseudo_id,
    (select value.int_value from unnest(event_params) where key = 'ga_session_id')
        as ga_session_id,
    (select value.int_value from unnest(event_params) where key = 'ga_session_number')
        as ga_session_number,
    (select value.string_value from unnest(event_params) where key = 'medium')
        as medium,
    (select value.string_value from unnest(event_params) where key = 'source')
        as source,
    (select value.string_value from unnest(event_params) where key = 'campaign')
        as campaign,
    ecommerce.purchase_revenue_in_usd as purchase_revenue_usd
from {{ source('ga4', 'events') }}
