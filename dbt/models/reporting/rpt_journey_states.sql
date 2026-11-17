-- Markov states split into channel and stage, with their role (attribution page):
-- closing = the converting session, assist = earlier steps of the journey.
select
    state,
    split(state, ':')[safe_offset(0)] as channel,
    split(state, ':')[safe_offset(1)] as stage,
    role,
    visits,
    removal_effect,
    conversions_credit
from {{ source('results', 'attribution_states') }}
