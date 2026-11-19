-- For each order: every session in the 30 days before it (since the user's
-- previous order), in order. This is the input to the attribution models.
create or replace table touchpaths as
with o as (
    select *, lag(ordered_at) over (partition by user_id order by ordered_at) as prev_order_at
    from orders
)
select
    o.transaction_id,
    o.revenue,
    o.ordered_at,
    s.channel,
    s.session_start_at,
    row_number() over (partition by o.transaction_id order by s.session_start_at) as touch_number
from o
join sessions s
  on s.user_id = o.user_id
 and s.session_start_at <= o.ordered_at
 and s.session_start_at > greatest(o.ordered_at - interval 30 day, coalesce(o.prev_order_at, timestamp '1900-01-01'));
