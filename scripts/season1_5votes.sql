-- ─── PV Check — saison 1 : cible de votes par installation (5 au lieu de 10) ──
-- Objectif : continuer la saison 1 sans rien remettre à zéro, avec des lots
-- moins coûteux en votes (même nombre d'installations par lot, ~481), et
-- garder 10 votes pour un sous-ensemble destiné au benchmark de calibration.
--
--   * 5 votes par installation par défaut          (arm = 'std5',     target_votes = 5)
--   * au plus 1 installation sur 10 (~48 par lot, réparties entre départements),
--     tirée au hasard, reste à 10 votes dès le départ               (arm = 'bench10',  target_votes = 10)
--   * les installations déjà à 10 votes gardent leur statut (arm = 'legacy10')
--   * à la demande (étape 5), les installations « ambiguës » parmi les 5
--     premiers votes passent à 10 votes (promoted_at renseigné, target_votes = 10)
--
-- Les votes (public.verifications) et le compteur votes_received ne sont
-- JAMAIS modifiés. Seul change le seuil à partir duquel une installation est
-- considérée comme « complète » : campaign_pool.target_votes, au lieu du 10
-- codé en dur dans les trois fonctions get_verification_batch(),
-- season_completion() et season_progress_by_department().
--
-- À exécuter ÉTAPE PAR ÉTAPE (SQL Editor ou psql), en lisant les résultats des
-- étapes de contrôle. L'étape 3 remplace des fonctions utilisées par le jeu :
-- l'exécuter quand l'étape 1 est faite (sinon la colonne n'existe pas).
-- Pour revenir en arrière : voir l'étape 7.

-- ═══ 0. Contrôle : état actuel (lecture seule) ═══════════════════════════════
-- Répartition du nombre de votes par installation (hors installations à 0 vote)
select votes_received, count(*) as installations
from public.campaign_pool
where campaign_id = 'season-1' and votes_received > 0
group by votes_received
order by votes_received;

-- Lots : installations, complètes à 10, déjà à >= 5 votes
select batch_no,
       count(*)                                    as installations,
       count(*) filter (where votes_received >= 10) as a_10_votes,
       count(*) filter (where votes_received >= 5)  as a_5_votes_ou_plus,
       sum(votes_received)                         as votes
from public.campaign_pool
where campaign_id = 'season-1' and batch_no <= 6
group by batch_no
order by batch_no;

-- ═══ 1. Nouvelles colonnes ═══════════════════════════════════════════════════
-- Ajout d'une colonne avec valeur par défaut constante : pas de réécriture de
-- la table (PostgreSQL >= 11), donc instantané même sur ~1,1 M de lignes.
-- target_votes vaut 5 pour toutes les lignes existantes ET pour toute ligne
-- insérée plus tard par la requête de population (default 5).
alter table public.campaign_pool add column if not exists target_votes   int  not null default 5;
alter table public.campaign_pool add column if not exists arm            text not null default 'std5'
    check (arm in ('std5', 'bench10', 'legacy10'));
alter table public.campaign_pool add column if not exists promoted_at    timestamptz;
alter table public.campaign_pool add column if not exists promotion_rule text;
-- Lot d'origine, conservé quand une installation promue est replacée dans le
-- lot en cours (étape 5). Null tant qu'elle n'a pas été déplacée.
alter table public.campaign_pool add column if not exists batch_no_orig  int;

-- ═══ 2. Statut des installations déjà complètes à 10 votes ═══════════════════
-- Les installations déjà à >= 10 votes (les deux premiers lots) gardent une
-- cible à 10 : ce sont les installations de référence du benchmark.
-- Les installations à 5-9 votes passent à « complètes » (cible 5), sans perdre
-- aucun vote ; elles pourront être promues à 10 par la règle de l'étape 5.
update public.campaign_pool
   set arm = 'legacy10', target_votes = 10
 where campaign_id = 'season-1'
   and votes_received >= 10
   and arm = 'std5';

-- ═══ 2b. Bras « bench10 » : au plus 1 installation sur 10 reste à 10 votes ══
-- Tirage au hasard, stocké une fois pour toutes (arm / target_votes), sur les
-- installations qui n'ont encore aucun vote, pour les 20 prochains lots à partir
-- du lot en cours. Dans chaque lot, on tire floor(10 % × taille du lot)
-- installations (~48 sur ~481), réparties entre départements : on classe d'abord
-- au hasard les installations de chaque (département, lot), puis on prend les
-- premiers rangs, en alternant les départements. Changer la part avec 0.10.
-- Rejouable pour de nouveaux lots (les lots qui ont déjà des bench10 sont ignorés).
with cur as (
    select min(batch_no) as b
    from public.campaign_pool
    where campaign_id = 'season-1' and votes_received < target_votes
),
cand as (
    select cp.id, cp.batch_no,
           row_number() over (partition by cp.dpt, cp.batch_no order by random()) as rk_dpt,
           random() as u
    from public.campaign_pool cp, cur
    where cp.campaign_id = 'season-1'
      and cp.votes_received = 0
      and cp.arm = 'std5'
      and cp.dpt is not null
      and cp.batch_no between cur.b and cur.b + 19
      and not exists (
          select 1 from public.campaign_pool b
          where b.campaign_id = cp.campaign_id and b.batch_no = cp.batch_no
            and b.arm = 'bench10'
      )
),
sized as (
    select c.*,
           floor(0.10 * count(*) over (partition by c.batch_no)) as quota,
           row_number() over (partition by c.batch_no order by c.rk_dpt, c.u) as rk_batch
    from cand c
)
update public.campaign_pool cp
   set arm = 'bench10', target_votes = 10
  from sized s
 where cp.id = s.id and s.rk_batch <= s.quota;

-- Contrôle : bras par lot
select batch_no, arm, count(*) as installations, sum(target_votes) as votes_cibles
from public.campaign_pool
where campaign_id = 'season-1' and batch_no <= 8
group by batch_no, arm
order by batch_no, arm;

-- ═══ 2c. Index pour la condition « votes_received < target_votes » ═══════════
-- Sans lui, les fonctions du jeu relisent toute la table (~655 000 lignes) et
-- dépassent la limite de durée du rôle « authenticated » (le jeu ne charge plus).
-- « concurrently » : ne bloque pas le jeu ; à lancer hors d'une transaction.
create index concurrently if not exists idx_campaign_pool_batch_votes_target
    on public.campaign_pool (campaign_id, batch_no) include (votes_received, target_votes);
vacuum (analyze) public.campaign_pool;

-- ═══ 3. Fonctions du jeu : « complet » = votes_received >= target_votes ══════
-- Même signature, mêmes colonnes de sortie : le front-end n'a rien à changer.
-- Seuls le « 10 » codé en dur et le « * 10 » des objectifs sont remplacés par
-- target_votes. Pour les barres de progression, les votes comptés pour un lot
-- sont plafonnés à la cible de chaque installation (least()).

drop function if exists public.get_verification_batch(text, int, text[]);
create or replace function public.get_verification_batch(
    p_campaign_id text,
    p_limit int default 12,
    p_exclude_ids text[] default '{}'
) returns table (
    detection_id text,
    lat double precision,
    lng double precision,
    gsd numeric,
    geometry jsonb
)
language sql
stable
security definer
set search_path = public
as $$
    with active_batch as (
        select min(batch_no) as batch_no
        from public.campaign_pool
        where campaign_id = p_campaign_id and votes_received < target_votes
    ),
    gold as (
        select cp.detection_id, cp.lat, cp.lng, cp.gsd, cp.geometry
        from public.gold_standard_items g
        join public.campaign_pool cp
          on cp.campaign_id = g.campaign_id and cp.detection_id = g.detection_id
        where g.campaign_id = p_campaign_id
          and g.active
          and not (cp.detection_id = any(p_exclude_ids))
          and not exists (
              select 1 from public.verifications v
              where v.user_id = auth.uid()
                and v.campaign_id = p_campaign_id
                and v.detection_id = cp.detection_id
          )
        order by random()
        limit 1
    ),
    tier0 as (
        select cp.detection_id, cp.lat, cp.lng, cp.gsd, cp.geometry
        from public.campaign_pool cp, active_batch ab
        where cp.campaign_id = p_campaign_id
          and cp.batch_no = ab.batch_no
          and cp.votes_received < cp.target_votes
          and not (cp.detection_id = any(p_exclude_ids))
          and not exists (
              select 1 from public.verifications v
              where v.user_id = auth.uid()
                and v.campaign_id = p_campaign_id
                and v.detection_id = cp.detection_id
          )
        order by cp.votes_received asc, random()
        limit p_limit
    ),
    tier1 as (
        select cp.detection_id, cp.lat, cp.lng, cp.gsd, cp.geometry
        from public.campaign_pool cp, active_batch ab
        where cp.campaign_id = p_campaign_id
          and cp.batch_no <> ab.batch_no
          and cp.votes_received > 0 and cp.votes_received < cp.target_votes
          and not (cp.detection_id = any(p_exclude_ids))
          and not exists (
              select 1 from public.verifications v
              where v.user_id = auth.uid()
                and v.campaign_id = p_campaign_id
                and v.detection_id = cp.detection_id
          )
        order by cp.votes_received asc, random()
        limit p_limit
    ),
    tier2 as (
        select cp.detection_id, cp.lat, cp.lng, cp.gsd, cp.geometry
        from public.campaign_pool cp, active_batch ab
        where cp.campaign_id = p_campaign_id
          and cp.batch_no > ab.batch_no
          and cp.votes_received = 0
          and not (cp.detection_id = any(p_exclude_ids))
          and not exists (
              select 1 from public.verifications v
              where v.user_id = auth.uid()
                and v.campaign_id = p_campaign_id
                and v.detection_id = cp.detection_id
          )
        order by cp.batch_no asc, random()
        limit p_limit
    )
    select * from gold
    union all
    select * from tier0
    union all
    select * from tier1
    union all
    select * from tier2
    limit p_limit;
$$;

grant execute on function public.get_verification_batch(text, int, text[]) to authenticated;

create or replace function public.season_completion(p_campaign_id text)
returns jsonb
language sql
stable
security definer
set search_path = public
as $$
    with totals as (
        select
            coalesce(sum(votes_received), 0)                              as votes_cast_total,
            count(*)                                                       as installations_total,
            count(*) filter (where votes_received >= target_votes)         as installations_done,
            count(distinct batch_no)                                       as batch_count,
            min(batch_no) filter (where votes_received < target_votes)     as first_open_batch,
            max(batch_no)                                                  as last_batch
        from public.campaign_pool
        where campaign_id = p_campaign_id
    ),
    active_batch as (
        select coalesce(t.first_open_batch, t.last_batch) as batch_no
        from totals t
    )
    select jsonb_build_object(
        'votes_cast_total',      t.votes_cast_total,
        'installations_done',    t.installations_done,
        'installations_total',   t.installations_total,
        'batch_no',              ab.batch_no,
        'batch_count',           t.batch_count,
        'batch_installations',   count(cp.*),
        'batch_votes_cast',      coalesce(sum(least(cp.votes_received, cp.target_votes)), 0),
        'batch_votes_target',    coalesce(sum(cp.target_votes), 0),
        'pct',                   least(100.0, round(100.0 * coalesce(sum(least(cp.votes_received, cp.target_votes)), 0)
                                         / greatest(coalesce(sum(cp.target_votes), 0), 1), 1))
    )
    from active_batch ab
    cross join totals t
    left join public.campaign_pool cp
        on cp.campaign_id = p_campaign_id and cp.batch_no = ab.batch_no
    group by ab.batch_no, t.votes_cast_total, t.installations_total, t.installations_done, t.batch_count;
$$;

grant execute on function public.season_completion(text) to authenticated;
grant execute on function public.season_completion(text) to service_role;

drop function if exists public.season_progress_by_department(text);
create or replace function public.season_progress_by_department(p_campaign_id text default 'season-1')
returns table (
    dpt               text,
    n_installations   bigint,
    votes_cast        bigint,
    votes_target      bigint,
    pct               numeric,
    vote_share_pct    numeric
)
language sql
stable
security definer
set search_path = public
as $$
    with active_batch as (
        select coalesce(
            (select min(batch_no) from public.campaign_pool where campaign_id = p_campaign_id and votes_received < target_votes),
            (select max(batch_no) from public.campaign_pool where campaign_id = p_campaign_id)
        ) as batch_no
    ),
    per_dept_active as (
        select
            cp.dpt,
            count(*)                                          as n_installations,
            coalesce(sum(least(cp.votes_received, cp.target_votes)), 0) as votes_cast,
            coalesce(sum(cp.target_votes), 0)                 as votes_target
        from public.campaign_pool cp, active_batch ab
        where cp.campaign_id = p_campaign_id
          and cp.dpt is not null
          and cp.batch_no = ab.batch_no
        group by cp.dpt
    )
    select
        pda.dpt,
        pda.n_installations,
        pda.votes_cast,
        pda.votes_target,
        least(100.0, round(100.0 * pda.votes_cast / greatest(pda.votes_target, 1), 1)) as pct,
        round(100.0 * pda.votes_cast / greatest(sum(pda.votes_cast) over (), 1), 2)    as vote_share_pct
    from per_dept_active pda
    order by pct desc, votes_cast desc;
$$;

grant execute on function public.season_progress_by_department(text) to authenticated;

-- Contrôle : lot actif et avancement vus par le jeu
select public.season_completion('season-1');

-- ═══ 4. (rien à faire) ═══════════════════════════════════════════════════════
-- Les nouvelles installations, ajoutées plus tard par la requête de population,
-- arrivent avec target_votes = 5 / arm = 'std5'. Pour leur attribuer ensuite le
-- bras bench10, rejouer l'étape 2b.

-- ═══ 5. Promotion : passer à 10 votes les installations « ambiguës » ═════════
-- Règle v1, calculée sur les 5 PREMIERS votes (par ordre d'écriture) :
--     au moins 2 « ne sais pas »  OU  (au moins 2 « confirmer » ET au moins 2 « rejeter »).
-- Sur les 971 installations déjà à 10 votes, elle sélectionne ~18 % des
-- installations, dont ~60 % finissent à >= 4 NSP ou polarisées (contre 14,5 % au hasard).
-- À lancer quand vous voulez (par ex. à la fin de chaque lot, ou chaque jour).
-- Seules les installations en bras std5, avec 5 à 9 votes, jamais promues, sont
-- concernées. Les installations promues sont REPLACÉES DANS LE LOT EN COURS
-- (batch_no_orig garde leur lot d'origine) : sans cela, un ancien lot déjà
-- terminé redeviendrait « incomplet » et le jeu y reviendrait en arrière.
--
-- Essai à blanc : remplacer « commit » par « rollback » en fin de bloc, lire la
-- liste renvoyée (returning), puis relancer avec « commit ».
begin;

with cur as (
    select min(batch_no) as b
    from public.campaign_pool
    where campaign_id = 'season-1' and votes_received < target_votes
),
first5 as (
    select v.detection_id, v.decision,
           row_number() over (partition by v.detection_id order by v.created_at, v.id) as rk
    from public.verifications v
    where v.campaign_id = 'season-1'
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
    where cp.campaign_id = 'season-1'
      and cp.arm = 'std5'
      and cp.promoted_at is null
      and cp.votes_received between 5 and 9
      and (agg.a5 >= 2 or (agg.c5 >= 2 and agg.r5 >= 2))
)
update public.campaign_pool cp
   set target_votes   = 10,
       promoted_at    = now(),
       promotion_rule = 'v1: >=2 NSP, or >=2 confirm and >=2 reject, among first 5 votes',
       batch_no_orig  = coalesce(cp.batch_no_orig, cp.batch_no),
       batch_no       = coalesce((select b from cur), cp.batch_no)
  from eligible e
 where cp.id = e.id
returning cp.detection_id, cp.dpt, cp.batch_no_orig, cp.batch_no, cp.votes_received;

commit;

-- ═══ 6. Contrôles après coup ═════════════════════════════════════════════════
-- Bras et statut des installations
select arm, target_votes, (promoted_at is not null) as promue, count(*) as installations,
       count(*) filter (where votes_received >= target_votes) as completes
from public.campaign_pool
where campaign_id = 'season-1' and (arm <> 'std5' or votes_received > 0 or promoted_at is not null)
group by arm, target_votes, (promoted_at is not null)
order by arm, target_votes, promue;

-- Vérifier que le jeu sert bien quelque chose (lot actif avec des installations restantes)
select batch_no, count(*) as restantes
from public.campaign_pool
where campaign_id = 'season-1' and votes_received < target_votes
group by batch_no order by batch_no limit 3;

-- ═══ 7. Retour arrière (à n'utiliser que si nécessaire) ══════════════════════
-- Remet la cible à 10 partout et replace les installations promues dans leur
-- lot d'origine. Puis recréer les trois fonctions avec leurs définitions
-- d'origine (scripts/verifications_setup.sql sur la branche gh-pages).
-- update public.campaign_pool
--    set batch_no = coalesce(batch_no_orig, batch_no), batch_no_orig = null,
--        target_votes = 10
--  where campaign_id = 'season-1';
