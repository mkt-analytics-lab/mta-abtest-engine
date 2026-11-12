-- Q3. Keys inside the nested event_params array: how often each key appears,
-- in which events, and which value type it uses.
select
    ep.key,
    count(*) as n_occurrences,
    count(distinct e.event_name) as n_event_types,
    string_agg(distinct e.event_name, ', ' order by e.event_name limit 6) as example_events,
    countif(ep.value.string_value is not null) as n_string,
    countif(ep.value.int_value is not null) as n_int,
    countif(ep.value.double_value is not null) as n_double
from {{ source('ga4', 'events') }} as e,
    unnest(e.event_params) as ep
group by ep.key
order by n_occurrences desc
