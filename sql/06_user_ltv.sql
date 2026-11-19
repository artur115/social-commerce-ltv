-- Revenue per buyer at 30, 90 and 180 days after the first order, plus the
-- early behaviour (first 30 days) used to predict it. Only buyers with a full
-- 180-day window are included.
create or replace table user_ltv as
with buyers as (
    select * from users
    where first_order_at is not null
      and first_order_at <= (select max(event_at) from stg_events) - interval 180 day
),
early_sessions as (
    select b.user_id,
           count(*)                                              as sessions_30d,
           count(distinct s.channel)                             as channels_30d,
           count(*) filter (where s.channel = 'live_stream')     as live_sessions_30d,
           count(*) filter (where s.channel = 'email')           as email_sessions_30d
    from buyers b join sessions s
      on s.user_id = b.user_id
     and s.session_start_at between b.first_order_at and b.first_order_at + interval 30 day
    group by 1
)
select
    b.user_id,
    b.acquisition_channel,
    b.cohort_month,
    b.experiment_variant,
    any_value(o.revenue) filter (where o.order_number = 1)                   as first_order_value,
    count(*) filter (where o.days_since_first_order <= 30)                  as orders_30d,
    min(o.days_since_first_order) filter (where o.order_number = 2)         as days_to_second_order,
    coalesce(es.sessions_30d, 0)                                             as sessions_30d,
    coalesce(es.channels_30d, 0)                                             as channels_30d,
    coalesce(es.live_sessions_30d, 0)                                        as live_sessions_30d,
    coalesce(es.email_sessions_30d, 0)                                       as email_sessions_30d,
    sum(o.revenue) filter (where o.days_since_first_order <= 30)            as ltv_30d,
    sum(o.revenue) filter (where o.days_since_first_order <= 90)            as ltv_90d,
    sum(o.revenue) filter (where o.days_since_first_order <= 180)           as ltv_180d,
    count(*) filter (where o.days_since_first_order <= 180)                 as orders_180d
from buyers b
join orders o using (user_id)
left join early_sessions es using (user_id)
group by all;
