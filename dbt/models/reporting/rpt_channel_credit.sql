-- Channel credit per attribution model with its share of the model total, plus
-- the difference vs the GA4 last-click proxy (attribution page).
with credit as (
    select
        model,
        channel,
        conversions_credit,
        revenue_credit,
        safe_divide(
            conversions_credit, sum(conversions_credit) over (partition by model)
        ) as conversions_share
    from {{ source('results', 'attribution_channels') }}
),

ga4 as (
    select channel, conversions_credit as ga4_conversions_credit
    from credit
    where model = 'ga4_last_click_proxy'
)

select
    credit.*,
    credit.conversions_credit - coalesce(ga4.ga4_conversions_credit, 0)
        as diff_vs_ga4_proxy
from credit
left join ga4 on credit.channel = ga4.channel
