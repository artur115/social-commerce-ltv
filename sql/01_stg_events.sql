-- One row per real event: drop duplicate deliveries, normalise source casing,
-- and turn the microsecond timestamp into a proper timestamp.
create or replace table stg_events as
select
    to_timestamp(event_timestamp / 1e6)            as event_at,
    event_name,
    user_pseudo_id                                 as user_id,
    user_pseudo_id || ':' || ga_session_id::varchar as session_key,
    lower(source)                                  as source,
    lower(medium)                                  as medium,
    campaign,
    experiment_variant,
    purchase_revenue_usd                           as revenue,
    transaction_id
from raw_events
qualify row_number() over (
    partition by user_pseudo_id, ga_session_id, event_name, event_timestamp
    order by event_timestamp
) = 1;
