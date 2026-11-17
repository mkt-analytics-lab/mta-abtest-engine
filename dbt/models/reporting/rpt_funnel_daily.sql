-- Funnel per day and entry channel: sessions reaching each stage (marketing page).
-- A session counts in every stage up to the deepest one it reached.
with sessions as (
    select * from {{ ref('int_ga4__sessions') }}
),

channels as (
    select * from {{ ref('channel_groups') }}
)

select
    s.session_date,
    c.channel_group,
    count(*) as sessions,
    countif(s.funnel_step >= 1) as reached_product,
    countif(s.funnel_step >= 2) as reached_cart,
    countif(s.funnel_step >= 3) as reached_checkout,
    countif(s.has_purchase) as purchases,
    sum(s.purchase_revenue_usd) as revenue_usd
from sessions as s
left join channels as c on s.session_medium = c.medium
group by s.session_date, c.channel_group
