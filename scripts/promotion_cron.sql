-- PV Check, saison 1 : promotion automatique (5 -> 10 votes) chaque nuit.
-- À lancer APRÈS season1_5votes.sql (étapes 1 à 3).

create or replace function public.promote_ambiguous(p_campaign_id text default 'season-1')
returns int
language plpgsql
security definer
set search_path = public
as $$
declare n int;
begin
    with cur as (
        select min(batch_no) as b
        from public.campaign_pool
        where campaign_id = p_campaign_id and votes_received < target_votes
    ),
    first5 as (
        select v.detection_id, v.decision,
               row_number() over (partition by v.detection_id order by v.created_at, v.id) as rk
        from public.verifications v
        where v.campaign_id = p_campaign_id
    ),
    agg as (
        select detection_id,
               count(*) filter (where decision = 'ambiguous') as a5,
               count(*) filter (where decision = 'confirm')   as c5,
               count(*) filter (where decision = 'reject')    as r5
        from first5
        where rk <= 5
        group by detection_id
        having count(*) = 5
    ),
    eligible as (
        select cp.id
        from public.campaign_pool cp
        join agg on agg.detection_id = cp.detection_id
        where cp.campaign_id = p_campaign_id
          and cp.arm = 'std5'
          and cp.promoted_at is null
          and cp.votes_received between 5 and 9
          and (agg.a5 >= 2 or (agg.c5 >= 2 and agg.r5 >= 2))
    ),
    upd as (
        update public.campaign_pool cp
           set target_votes   = 10,
               promoted_at    = now(),
               promotion_rule = 'v1: >=2 NSP, or >=2 confirm and >=2 reject, among first 5 votes',
               batch_no_orig  = coalesce(cp.batch_no_orig, cp.batch_no),
               batch_no       = coalesce((select b from cur), cp.batch_no)
          from eligible e
         where cp.id = e.id
        returning 1
    )
    select count(*) into n from upd;
    return n;
end $$;

revoke all on function public.promote_ambiguous(text) from public, anon, authenticated;

-- Planification : tous les jours à 03:00 UTC (05:00 à Paris en été).
-- Prérequis : activer l'extension (Supabase : Database > Extensions > pg_cron),
-- ou la ligne suivante.
create extension if not exists pg_cron;
select cron.schedule('pvcheck-promote-ambiguous', '0 3 * * *', $$select public.promote_ambiguous('season-1')$$);

-- Pour l'arrêter : select cron.unschedule('pvcheck-promote-ambiguous');
-- Pour la lancer à la main : select public.promote_ambiguous();
