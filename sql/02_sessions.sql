-- One row per session with its traffic channel and outcome.
create or replace table sessions as
select
    session_key,
    user_id,
    min(event_at)                                         as session_start_at,
    any_value(source)                                     as source,
    any_value(medium)                                     as medium,
    case
        when any_value(medium) = 'live_shopping'                       then 'live_stream'
        when any_value(medium) = 'influencer'                          then 'influencer'
        when any_value(medium) = 'paid_social'                         then any_value(source)
        when any_value(medium) = 'cpc'                                 then 'paid_search'
        when any_value(medium) = 'organic'                             then 'organic'
        when any_value(medium) = 'email'                               then 'email'
        when any_value(medium) = 'referral'                            then 'referral'
        else 'direct'
    end                                                   as channel,
    count(*) filter (where event_name = 'purchase')       as orders,
    coalesce(sum(revenue) filter (where event_name = 'purchase'), 0) as revenue
from stg_events
group by all;
