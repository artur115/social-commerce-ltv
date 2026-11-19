-- One row per user: acquisition channel (first session), cohort and experiment arm.
create or replace table users as
with first_session as (
    select user_id, channel as acquisition_channel, session_start_at as first_seen_at
    from sessions
    qualify row_number() over (partition by user_id order by session_start_at) = 1
),
first_order as (
    select user_id, min(event_at) as first_order_at
    from stg_events where event_name = 'purchase' group by 1
)
select
    f.user_id,
    f.acquisition_channel,
    f.first_seen_at,
    date_trunc('month', f.first_seen_at)                  as cohort_month,
    o.first_order_at,
    max(e.experiment_variant)                             as experiment_variant
from first_session f
left join first_order o using (user_id)
left join stg_events e using (user_id)
group by all;
