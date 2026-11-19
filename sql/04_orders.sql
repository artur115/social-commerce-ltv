-- One row per order, numbered per user, with days since the user's first order.
create or replace table orders as
select
    e.transaction_id,
    e.user_id,
    e.event_at                                            as ordered_at,
    e.revenue,
    s.channel                                             as last_click_channel,
    row_number() over (partition by e.user_id order by e.event_at) as order_number,
    date_diff('day', u.first_order_at, e.event_at)        as days_since_first_order
from stg_events e
join sessions s using (session_key)
join users u on u.user_id = e.user_id
where e.event_name = 'purchase';
