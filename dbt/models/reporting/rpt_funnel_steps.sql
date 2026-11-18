-- Funnel in long format (one row per day, channel and step) for the native
-- Looker Studio funnel chart, which needs a step dimension and one metric.
-- Step labels carry their order ("1. Session") so alphabetical sort is correct.
with funnel as (
    select * from {{ ref('rpt_funnel_daily') }}
)

select
    f.session_date,
    f.channel_group,
    s.step_order,
    s.step,
    s.step_sessions
from funnel as f
cross join unnest([
    struct(1 as step_order, '1. Session' as step, f.sessions as step_sessions),
    struct(2, '2. Product view', f.reached_product),
    struct(3, '3. Add to cart', f.reached_cart),
    struct(4, '4. Checkout', f.reached_checkout),
    struct(5, '5. Purchase', f.purchases)
]) as s
