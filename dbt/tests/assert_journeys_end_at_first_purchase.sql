-- Singular test: a journey has at most one conversion touch, and it is the last
-- touchpoint. Returns offending users; the test passes when it returns nothing.
select user_pseudo_id
from {{ ref('fct_touchpoints') }}
group by user_pseudo_id
having
    countif(is_conversion_touch) > 1
    or countif(is_conversion_touch and touch_position != n_touchpoints) > 0
