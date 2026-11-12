-- Q5. Users, purchasers and sessions: class imbalance and revenue concentration.
with users as (
    select
        user_pseudo_id,
        countif(event_name = 'purchase') as n_purchases,
        sum(ecommerce.purchase_revenue_in_usd) as revenue_usd,
        count(distinct (
            select value.int_value from unnest(event_params) where key = 'ga_session_id'
        )) as n_sessions
    from {{ source('ga4', 'events') }}
    group by user_pseudo_id
)

select
    count(*) as n_users,
    countif(n_purchases > 0) as n_purchasers,
    round(100 * countif(n_purchases > 0) / count(*), 2) as pct_purchasers,
    round(sum(revenue_usd), 0) as total_revenue_usd,
    approx_quantiles(n_sessions, 4) as sessions_per_user_quartiles,
    round(avg(if(n_purchases > 0, n_sessions, null)), 2) as avg_sessions_purchasers,
    round(avg(if(n_purchases = 0, n_sessions, null)), 2) as avg_sessions_non_purchasers
from users
