-- Average cumulative revenue per buyer by days since first order, per acquisition channel.
create or replace table ltv_curves as
with days as (select unnest(range(0, 181, 5)) as day),
per_user as (
    select u.user_id, u.acquisition_channel, d.day,
           coalesce(sum(o.revenue) filter (where o.days_since_first_order <= d.day), 0) as cum_revenue
    from user_ltv u cross join days d
    join orders o using (user_id)
    group by all
)
select acquisition_channel, day, avg(cum_revenue) as avg_cum_revenue, count(*) as buyers
from per_user group by all order by 1, 2;
