-- ─── PV Check — vérification communautaire par swipe (backend) ───────────────
-- Run once in the Supabase SQL Editor, in the SAME project as
-- supabase_setup.sql / supabase_detections_setup.sql (this reuses the
-- `detections` table to build the pool, and follows the same insert-only +
-- RLS + security-definer-RPC posture as annotations/issue_reports).
--
-- Design note (see game/README.md "Distribution du travail"): earlier drafts
-- used a claim+lease job-queue pattern (FOR UPDATE SKIP LOCKED, active/
-- inactive tracking, discrete "passes"). Dropped after review — that pattern
-- protects against a unit of work being done twice, which is the OPPOSITE of
-- what we want here (every installation needs ~10 independent votes). What's
-- below is the standard pattern for redundant crowd-labeling instead: a
-- maintained vote counter + "serve the least-covered items first" ordering,
-- no claim, no lease, no coordination between concurrent users.
--
-- Auth note: this app requires a Supabase Auth session (anonymous sign-in at
-- minimum, see game/js/auth.js) for every request — RLS policies below target
-- `authenticated` (which anonymous-signed-in sessions satisfy) rather than
-- `anon`, unlike annotations/events which accept the bare anon API key with
-- no session at all.
--
-- Rerunnable on purpose: every statement below is written so this whole file
-- can be pasted into the SQL Editor again after a change (new RPC, a tweaked
-- one) WITHOUT erroring out on "relation/policy/trigger already exists" —
-- `create table if not exists`, `drop policy/trigger if exists` before
-- `create`, `create or replace function`, `on conflict do nothing` for seed
-- rows. If a future edit adds a new `create table`/`create policy`/
-- `create trigger`, give it the same treatment.

-- ─── Campaigns ("saisons") ─────────────────────────────────────────────────

create table if not exists public.campaigns (
    id          text primary key,          -- slug, e.g. 'season-1'
    label       text not null,
    active      boolean not null default true,
    created_at  timestamptz not null default now()
);

insert into public.campaigns (id, label) values
    ('season-1', 'Saison 1 — installations mono-source (DPVM ou FRPV seul)')
on conflict (id) do nothing;

-- ─── campaign_pool — the target list for a campaign ───────────────────────
-- One row per targeted installation (NOT duplicated 10x). votes_received is
-- a denormalized counter, bumped by the trigger below on every insert into
-- verifications — never recomputed by aggregation at read time.

create table if not exists public.campaign_pool (
    id              bigint generated always as identity primary key,
    campaign_id     text not null references public.campaigns(id),
    detection_id    text not null,          -- detections.id::text, same convention as annotations.target_id
    lat             double precision not null,
    lng             double precision not null,
    gsd             numeric not null,       -- meters/pixel to request from the WMS GetMap, precomputed here
    geometry        jsonb not null,         -- polygon, EPSG:4326 — client draws it as an SVG overlay on the card
    votes_received  int not null default 0,
    unique (campaign_id, detection_id)
);

create index if not exists idx_campaign_pool_campaign_votes
    on public.campaign_pool (campaign_id, votes_received);

-- dpt (département) is denormalized onto campaign_pool for the same reason
-- lat/lng/gsd/geometry already are: season_progress_by_department() used to
-- join campaign_pool (~655k rows) to detections (~1.1M rows) at RPC-call
-- time just to read this one column, and that join alone was enough to hit
-- Supabase's statement timeout on the `authenticated` role (57014) — even
-- after cutting it down to a single join. Storing dpt here means that RPC
-- never joins detections at all; the cost of the join is paid once below
-- (population + backfill), not on every visitor's every page load.
alter table public.campaign_pool add column if not exists dpt text;

-- Population for season-1: mono-source detections only (sources has no comma
-- and is exactly '0' DPVM or '1' FRPV — see SOURCE_LABELS in the map's
-- config.js; 2=OSM/3=manual/4=recall are already considered reliable and
-- excluded). Image footprint: 400x400px, base GSD 0.20 m/px (80m x 80m); if
-- the polygon's bbox (with a 30% margin) doesn't fit in that footprint, GSD
-- is widened just enough to fit it, same 400x400px output either way.
-- Idempotent (on conflict do nothing) — safe to re-run if it's interrupted.
with sized as (
    select
        d.id,
        d.dpt,
        d.geom,
        ST_Y(ST_Centroid(d.geom)) as lat,
        ST_X(ST_Centroid(d.geom)) as lng,
        greatest(
            ST_XMax(ST_Transform(d.geom, 2154)) - ST_XMin(ST_Transform(d.geom, 2154)),
            ST_YMax(ST_Transform(d.geom, 2154)) - ST_YMin(ST_Transform(d.geom, 2154))
        ) as extent_m
    from public.detections d
    where d.sources is not null
      and d.sources !~ ','                       -- mono-source only (no comma)
      and trim(d.sources) in ('0', '1')           -- DPVM or FRPV only
      and coalesce(d.false_positive, false) = false
)
insert into public.campaign_pool (campaign_id, detection_id, lat, lng, gsd, geometry, dpt)
select
    'season-1',
    s.id::text,
    s.lat,
    s.lng,
    greatest(0.20, (s.extent_m * 1.3) / 400.0),
    ST_AsGeoJSON(s.geom)::jsonb,
    s.dpt
from sized s
on conflict (campaign_id, detection_id) do nothing;

-- Without this, `d.id::text = cp.detection_id` below has nothing to use on
-- the detections side (only the bare, untyped-cast id is indexed), so
-- Postgres falls back to a much more expensive plan — this is what made
-- the backfill below hit "canceling statement due to statement timeout"
-- (57014) even run directly via psql as the postgres role, not just from
-- the authenticated-role RPC. Confirmed via a direct count: 1,135,849 of
-- 1,135,850 detections already have dpt set — the source data was never
-- the problem, the join plan was.
create index if not exists idx_detections_id_text on public.detections ((id::text));

-- One-time backfill for rows inserted before the dpt column existed (i.e.
-- anyone re-running this script rather than starting fresh — the insert
-- above is on-conflict-do-nothing, so it never touches already-present
-- rows). This is the one place the campaign_pool <-> detections join still
-- happens; it runs here, once, rather than inside an RPC every visitor
-- calls.
--
-- Split into 10 separate statements rather than one big UPDATE (or a
-- PL/pgSQL loop in a single DO block — statement_timeout bounds the WHOLE
-- block as one unit in that case, a loop inside it does NOT get a fresh
-- timeout per iteration). Each statement here is its own top-level command,
-- so each gets its own fresh timeout window, and thanks to the `is distinct
-- from` guard, if any one batch still times out, everything before it has
-- already committed — just re-run the script and only the remaining
-- batches do any work.
update public.campaign_pool cp set dpt = d.dpt from public.detections d
where d.id::text = cp.detection_id and cp.dpt is distinct from d.dpt and cp.id % 10 = 0;
update public.campaign_pool cp set dpt = d.dpt from public.detections d
where d.id::text = cp.detection_id and cp.dpt is distinct from d.dpt and cp.id % 10 = 1;
update public.campaign_pool cp set dpt = d.dpt from public.detections d
where d.id::text = cp.detection_id and cp.dpt is distinct from d.dpt and cp.id % 10 = 2;
update public.campaign_pool cp set dpt = d.dpt from public.detections d
where d.id::text = cp.detection_id and cp.dpt is distinct from d.dpt and cp.id % 10 = 3;
update public.campaign_pool cp set dpt = d.dpt from public.detections d
where d.id::text = cp.detection_id and cp.dpt is distinct from d.dpt and cp.id % 10 = 4;
update public.campaign_pool cp set dpt = d.dpt from public.detections d
where d.id::text = cp.detection_id and cp.dpt is distinct from d.dpt and cp.id % 10 = 5;
update public.campaign_pool cp set dpt = d.dpt from public.detections d
where d.id::text = cp.detection_id and cp.dpt is distinct from d.dpt and cp.id % 10 = 6;
update public.campaign_pool cp set dpt = d.dpt from public.detections d
where d.id::text = cp.detection_id and cp.dpt is distinct from d.dpt and cp.id % 10 = 7;
update public.campaign_pool cp set dpt = d.dpt from public.detections d
where d.id::text = cp.detection_id and cp.dpt is distinct from d.dpt and cp.id % 10 = 8;
update public.campaign_pool cp set dpt = d.dpt from public.detections d
where d.id::text = cp.detection_id and cp.dpt is distinct from d.dpt and cp.id % 10 = 9;

-- ─── Batching — waterfall through the pool 1% at a time ──────────────────
-- Serving all ~655k installations at once meant votes spread paper-thin
-- across the whole pool — nothing ever felt close to done, and the
-- national/département progress numbers stayed unreadably close to 0% for
-- weeks even with real activity. Splitting the pool into 100 fixed batches
-- (~6.5k installations each) and having get_verification_batch() only ever
-- serve from the lowest-numbered batch that isn't fully done yet (still
-- targeting the full 10 votes/installation within that batch — nothing
-- about the actual redundancy target changes) means batch 1 visibly closes
-- out before batch 2 opens, and every progress number below is scoped to
-- "how's the active batch doing" instead of "how's all 655k doing" — same
-- work, a denominator ~100x smaller and honest about what's actually being
-- worked on right now. (First cut of this used 10 batches of ~65k —
-- 6.5k moves the counter noticeably faster still, at the cost of a batch
-- closing out sooner and handing off to the next one more often.)
--
-- batch_no is assigned by id order and recomputed every time this script
-- runs (no "already has one" guard) — deliberately, so changing the batch
-- COUNT (as just happened, 10 -> 100) or rerunning after the population
-- query adds more rows both just work. ntile() is deterministic for a
-- given row count, so rerunning with nothing new inserted is a no-op in
-- effect, just some write I/O.
alter table public.campaign_pool add column if not exists batch_no int;

with numbered as (
    select id, ntile(100) over (partition by campaign_id order by id) as batch_no
    from public.campaign_pool
)
update public.campaign_pool cp
set batch_no = numbered.batch_no
from numbered
where cp.id = numbered.id
  and cp.batch_no is distinct from numbered.batch_no;

create index if not exists idx_campaign_pool_campaign_batch_votes
    on public.campaign_pool (campaign_id, batch_no, votes_received);

-- Supports season_progress_by_department()'s group-by-(dpt, batch_no) with
-- no join now that dpt lives on this table directly (see above).
create index if not exists idx_campaign_pool_campaign_batch_dpt
    on public.campaign_pool (campaign_id, batch_no, dpt);

-- ─── profiles — pseudo, keyed on the stable Supabase Auth uuid ───────────
-- Same uuid whether the session is anonymous, email, or (later) Google —
-- linkIdentity() converts in place without ever changing this id, so a
-- claimed account keeps its pseudo/history automatically.

create table if not exists public.profiles (
    id          uuid primary key references auth.users(id) on delete cascade,
    pseudo      text not null unique check (char_length(pseudo) between 2 and 24),
    created_at  timestamptz not null default now()
);

alter table public.profiles enable row level security;

drop policy if exists "authenticated can insert own profile" on public.profiles;
create policy "authenticated can insert own profile"
    on public.profiles for insert
    to authenticated
    with check (auth.uid() = id);

-- A user can read their OWN row only (needed so the app can tell, on
-- reload, whether this session already has a pseudo — without this it
-- would have no way to know except asking again every time). Leaderboard
-- reads of OTHER people's pseudos go through the security-definer RPC
-- below instead, so this stays scoped to "your own row", nothing more.
drop policy if exists "authenticated can read own profile" on public.profiles;
create policy "authenticated can read own profile"
    on public.profiles for select
    to authenticated
    using (auth.uid() = id);

-- A user can rename their OWN row only — the menu's pencil icon next to
-- the name (shown once an account has an email, see editPseudo() in
-- game/js/auth.js). The `pseudo` column's own unique/length check
-- constraints do the actual validation; a rename to an already-taken name
-- fails the same way the original insert would (Postgres unique_violation,
-- 23505).
drop policy if exists "authenticated can update own profile" on public.profiles;
create policy "authenticated can update own profile"
    on public.profiles for update
    to authenticated
    using (auth.uid() = id)
    with check (auth.uid() = id);

-- ─── verifications — the swipes themselves ────────────────────────────────
-- Insert-only, immutable, same family as annotations/issue_reports. The
-- (user_id, campaign_id, detection_id) unique constraint is the real
-- guarantee that matters: nobody can vote the same installation twice,
-- independent of whatever ordering the batch RPC happened to serve.

create table if not exists public.verifications (
    id            uuid primary key default gen_random_uuid(),
    created_at    timestamptz not null default now(),
    user_id       uuid not null references auth.users(id),
    campaign_id   text not null references public.campaigns(id),
    detection_id  text not null,
    decision      text not null check (decision in ('confirm', 'reject', 'ambiguous')),
    -- Note: "passer sans juger" (skip) is NOT a decision value — it's handled
    -- purely client-side (just move to the next card, nothing written). A
    -- skip records no judgement, so it must not consume this user's one-shot
    -- exclusion on this item nor count toward the 10-vote target — see
    -- game/README.md "Interaction".
    comment       text check (char_length(comment) <= 500),
    status        text not null default 'pending' check (status in ('pending', 'reviewed')),
    unique (user_id, campaign_id, detection_id)
);

-- Deleting an account must not delete these rows — campaign_pool.votes_
-- received only has an AFTER INSERT trigger to bump it (see below), no
-- compensating DELETE trigger to decrement it, so removing verification
-- rows would silently leave that counter (and every RPC built on it)
-- overcounted forever. NULL doesn't collide with the unique constraint
-- above either (Postgres treats each NULL as distinct there), so any
-- number of deleted-account rows can pile up on the same installation with
-- no conflict. See delete_own_account() further down.
alter table public.verifications alter column user_id drop not null;
alter table public.verifications drop constraint if exists verifications_user_id_fkey;
alter table public.verifications add constraint verifications_user_id_fkey
    foreign key (user_id) references auth.users(id) on delete set null;

alter table public.verifications enable row level security;

drop policy if exists "authenticated can insert own verification" on public.verifications;
create policy "authenticated can insert own verification"
    on public.verifications for insert
    to authenticated
    with check (
        auth.uid() = user_id
        and status = 'pending'                         -- no self-validation
        and coalesce(char_length(comment), 0) <= 500
    );
-- No select/update/delete for authenticated — moderation happens offline via
-- the verifications_summary view below (Dashboard / service_role only), same
-- pattern as annotations_pending / issue_reports_pending.

-- Bump campaign_pool.votes_received on every new verification. security
-- definer isn't needed here (the trigger runs as the table owner regardless
-- of RLS on the invoking role), but search_path is pinned defensively.
create or replace function public.bump_votes_received()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
    update public.campaign_pool
       set votes_received = votes_received + 1
     where campaign_id = new.campaign_id
       and detection_id = new.detection_id;
    return new;
end;
$$;

drop trigger if exists trg_bump_votes_received on public.verifications;
create trigger trg_bump_votes_received
    after insert on public.verifications
    for each row execute function public.bump_votes_received();

-- ─── get_verification_batch — the only scheduling logic there is ─────────
-- Least-covered items first, random tiebreak, excluding whatever this user
-- has already voted — same as before, but now scoped to the single
-- lowest-numbered batch_no that still has incomplete items (see the
-- "Batching" note above campaign_pool.batch_no). Still no claim, no lease,
-- no per-user session state to maintain server-side beyond what's already
-- in `verifications` — the only new thing is which slice of the pool a
-- request is allowed to draw from.

-- `returns table(...)` functions use implicit OUT parameters, and Postgres
-- refuses `create or replace` if the OUT row type changes (even just adding
-- a column) — "cannot change return type of existing function", asking for
-- an explicit drop first. Rather than remember that only when it bites,
-- every table-returning function here is dropped first, every time.
drop function if exists public.get_verification_batch(text, int);
create or replace function public.get_verification_batch(
    p_campaign_id text,
    p_limit int default 12
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
        where campaign_id = p_campaign_id and votes_received < 10
    )
    select cp.detection_id, cp.lat, cp.lng, cp.gsd, cp.geometry
    from public.campaign_pool cp, active_batch ab
    where cp.campaign_id = p_campaign_id
      and cp.batch_no = ab.batch_no
      and cp.votes_received < 10
      and not exists (
          select 1 from public.verifications v
          where v.user_id = auth.uid()
            and v.campaign_id = p_campaign_id
            and v.detection_id = cp.detection_id
      )
    order by cp.votes_received asc, random()
    limit p_limit;
$$;

grant execute on function public.get_verification_batch(text, int) to authenticated;

-- ─── Menu RPCs — mes stats / leaderboard / % complétion ───────────────────

-- Public counter + last-contribution timestamp, same shape as the map's
-- annotation_stats() — used on the landing screen before anyone signs in
-- with anything more than the anonymous session ensureSession() already
-- creates on load.
create or replace function public.verification_stats()
returns jsonb
language sql
stable
security definer
set search_path = public
as $$
    select jsonb_build_object(
        'count',   (select count(*) from public.verifications),
        'last_at', (select max(created_at) from public.verifications)
    )
$$;

-- Granted to anon too, defensively — game/js/main.js calls this on the
-- landing screen right after ensureSession() resolves (so normally there's
-- already an `authenticated` anonymous session by then), but there's no
-- reason to make this public count depend on that succeeding first.
grant execute on function public.verification_stats() to anon, authenticated;

create or replace function public.my_verification_count()
returns bigint
language sql
stable
security definer
set search_path = public
as $$
    select count(*) from public.verifications where user_id = auth.uid();
$$;

grant execute on function public.my_verification_count() to authenticated;

-- Per-decision breakdown for "My verifications" in the menu (confirm/reject/
-- ambiguous counters) — one round trip instead of three separate counts.
create or replace function public.my_verification_breakdown()
returns jsonb
language sql
stable
security definer
set search_path = public
as $$
    select jsonb_build_object(
        'confirm',   count(*) filter (where decision = 'confirm'),
        'reject',    count(*) filter (where decision = 'reject'),
        'ambiguous', count(*) filter (where decision = 'ambiguous'),
        'total',     count(*)
    )
    from public.verifications
    where user_id = auth.uid();
$$;

grant execute on function public.my_verification_breakdown() to authenticated;

-- Self-service account deletion, called by the menu's "Delete my account".
-- profiles cascades automatically (on delete cascade, see its create table
-- above); verifications keeps its rows (on delete set null, see above) —
-- only the personal link is removed, not the vote itself. security definer
-- so an ordinary authenticated session — which has no grants on the auth
-- schema at all — can still reach auth.users to delete ONLY its own row
-- (auth.uid(), never a parameter, so there's no way to pass someone else's
-- id in). This is the standard community pattern for self-service delete
-- on Supabase (no client-side supabase.auth.deleteUser() exists — that's
-- an admin-API-only operation otherwise). If this errors with a permission
-- issue on your project, the function's owner (whichever role runs this
-- script) doesn't have delete rights on auth.users and it'll need granting
-- separately — flag it and we'll sort it out.
create or replace function public.delete_own_account()
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
    delete from auth.users where id = auth.uid();
end;
$$;

grant execute on function public.delete_own_account() to authenticated;

-- p_window: 'week' (rolling 7 days) | 'month' (rolling 30 days) | 'all'
-- Hard-capped at 100 server-side (least(p_limit, 100)) regardless of what a
-- caller passes — the menu always asks for 100, but the cap lives here too
-- so it can't quietly grow into an unbounded query from some other caller.
drop function if exists public.leaderboard(text, int);
create or replace function public.leaderboard(
    p_window text default 'all',
    p_limit int default 100
) returns table (pseudo text, total bigint)
language sql
stable
security definer
set search_path = public
as $$
    select p.pseudo, count(v.id) as total
    from public.verifications v
    join public.profiles p on p.id = v.user_id
    where case p_window
              when 'week'  then v.created_at > now() - interval '7 days'
              when 'month' then v.created_at > now() - interval '30 days'
              else true
          end
    group by p.pseudo
    order by total desc
    limit least(p_limit, 100);
$$;

grant execute on function public.leaderboard(text, int) to authenticated;

-- leaderboard() only returns the top p_limit rows, so the menu can't sum a
-- window total client-side without fetching every player — this is that
-- total, same window semantics (week/month/all), shown above the ranked
-- list itself.
create or replace function public.leaderboard_total(p_window text default 'all')
returns bigint
language sql
stable
security definer
set search_path = public
as $$
    select count(*)
    from public.verifications v
    where case p_window
              when 'week'  then v.created_at > now() - interval '7 days'
              when 'month' then v.created_at > now() - interval '30 days'
              else true
          end;
$$;

grant execute on function public.leaderboard_total(text) to authenticated;

-- The current user's own rank for a window — needed because leaderboard()
-- only returns the top 100, so someone outside that range would otherwise
-- have no way to see where they stand. Works the same for anonymous and
-- email-linked accounts alike (both are `authenticated`, just distinguished
-- by the is_anonymous JWT claim — irrelevant here, this only cares about
-- auth.uid()). Returns zero rows if this user hasn't voted at all in the
-- window (per_user only includes users with >= 1 vote) — the client shows
-- "—" for that case rather than a rank.
create or replace function public.my_leaderboard_rank(p_window text default 'all')
returns table (rnk bigint, total_players bigint, my_total bigint)
language sql
stable
security definer
set search_path = public
as $$
    with per_user as (
        select v.user_id, count(*) as total
        from public.verifications v
        where v.user_id is not null
          and case p_window
                  when 'week'  then v.created_at > now() - interval '7 days'
                  when 'month' then v.created_at > now() - interval '30 days'
                  else true
              end
        group by v.user_id
    ),
    ranked as (
        select user_id, total, rank() over (order by total desc) as rnk
        from per_user
    )
    select r.rnk, (select count(*) from per_user) as total_players, r.total
    from ranked r
    where r.user_id = auth.uid();
$$;

grant execute on function public.my_leaderboard_rank(text) to authenticated;

-- Two different numbers here, on purpose (this got confusing when they
-- shared a name, so they now have distinct keys):
--   - `votes_cast_total` is the honest, unscoped, all-time count of votes
--     cast across the whole campaign (every batch) — same number as
--     verification_stats().count. This is what the headline "N validations
--     done" figure in the menu shows: the real total, full stop.
--   - `batch_votes_cast` / `batch_votes_target` / `pct` stay scoped to the
--     single active batch (see the "Batching" note above
--     campaign_pool.batch_no) — same real 10-votes-per-installation target
--     the scheduler uses, just measured against the ~6.5k installations
--     currently being worked on instead of all ~655k, so the progress BAR
--     still moves at a readable pace instead of crawling against the full
--     season. Falls back to the LAST batch (as 100%) if every batch is
--     done — an edge case this campaign won't realistically hit for a long
--     time, but shouldn't return nothing if it ever does.
-- NOT gated on verifications.status='reviewed' (moderation is offline/
-- after the fact — see verifications_summary below — and must not hold
-- back either number).
create or replace function public.season_completion(p_campaign_id text)
returns jsonb
language sql
stable
security definer
set search_path = public
as $$
    with active_batch as (
        select coalesce(
            (select min(batch_no) from public.campaign_pool where campaign_id = p_campaign_id and votes_received < 10),
            (select max(batch_no) from public.campaign_pool where campaign_id = p_campaign_id)
        ) as batch_no
    ),
    totals as (
        select coalesce(sum(votes_received), 0) as votes_cast_total
        from public.campaign_pool
        where campaign_id = p_campaign_id
    )
    select jsonb_build_object(
        'votes_cast_total',     t.votes_cast_total,
        'batch_no',             ab.batch_no,
        'batch_count',          (select count(distinct batch_no) from public.campaign_pool where campaign_id = p_campaign_id),
        'batch_installations',  count(cp.*),
        'batch_votes_cast',     coalesce(sum(cp.votes_received), 0),
        'batch_votes_target',   count(cp.*) * 10,
        'pct',                  least(100.0, round(100.0 * coalesce(sum(cp.votes_received), 0)
                                        / greatest(count(cp.*) * 10, 1), 1))
    )
    from active_batch ab
    cross join totals t
    left join public.campaign_pool cp
        on cp.campaign_id = p_campaign_id and cp.batch_no = ab.batch_no
    group by ab.batch_no, t.votes_cast_total;
$$;

grant execute on function public.season_completion(text) to authenticated;

-- Per-département breakdown for the menu's "Progress" tab (rendered as a
-- choropleth map client-side, see game/js/deptmap.js). Two different
-- numbers, two different scopes, on purpose:
--   - `pct`/`votes_cast`/`votes_target` are this département's completion
--     WITHIN THE ACTIVE BATCH ONLY — same scope and same real 10-vote
--     target as season_completion() above, just broken out per département
--     instead of collapsed to one national number.
--   - `vote_share_pct` is this département's share of ALL votes cast so
--     far ACROSS EVERY BATCH (sums to ~100 across départements) —
--     deliberately NOT scoped to the active batch. The map colors by this
--     one: scoping the map to the active batch too would leave most
--     départements looking empty at any given moment (only ~6.5k of 655k
--     installations are even in play), which flattens the map instead of
--     showing where the community's total effort has actually gone.
-- View-only either way: this does NOT feed anything back into
-- get_verification_batch(), which stays a single global least-covered-
-- first-within-the-active-batch queue — see game/README.md "Saison 2+" for
-- why a per-département opt-in/filter is deliberately not built: it would
-- let popular départements get over-voted while remote ones get skipped,
-- exactly the sampling bias the batched-but-still-global queue avoids.
--
-- No join to detections here at all — dpt is denormalized onto
-- campaign_pool (see the note above campaign_pool.dpt), so this is a single
-- grouped scan of campaign_pool alone, backed by
-- idx_campaign_pool_campaign_batch_dpt. An earlier version of this function
-- joined campaign_pool to detections (first two joins, then one) to read
-- dpt at call time, and even the single-join version reliably hit
-- Supabase's `authenticated`-role statement timeout (57014) — the join
-- itself was the cost, not how many times it happened.
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
            (select min(batch_no) from public.campaign_pool where campaign_id = p_campaign_id and votes_received < 10),
            (select max(batch_no) from public.campaign_pool where campaign_id = p_campaign_id)
        ) as batch_no
    ),
    per_dept_batch_raw as (
        select
            cp.dpt,
            cp.batch_no,
            count(*)                            as n_installations,
            coalesce(sum(cp.votes_received), 0) as votes_cast
        from public.campaign_pool cp
        where cp.campaign_id = p_campaign_id
          and cp.dpt is not null
        group by cp.dpt, cp.batch_no
    ),
    per_dept_active as (
        select r.dpt, r.n_installations, r.votes_cast
        from per_dept_batch_raw r, active_batch ab
        where r.batch_no = ab.batch_no
    ),
    per_dept_all as (
        select dpt, sum(votes_cast) as votes_cast_all
        from per_dept_batch_raw
        group by dpt
    ),
    totals_all as (
        select greatest(sum(votes_cast_all), 1) as total_votes from per_dept_all
    )
    select
        pda.dpt,
        pda.n_installations,
        pda.votes_cast,
        pda.n_installations * 10                                          as votes_target,
        least(100.0, round(100.0 * pda.votes_cast / greatest(pda.n_installations * 10, 1), 1)) as pct,
        round(100.0 * coalesce(pall.votes_cast_all, 0) / t.total_votes, 2) as vote_share_pct
    from per_dept_active pda
    left join per_dept_all pall on pall.dpt = pda.dpt
    cross join totals_all t
    order by pct desc, votes_cast desc;
$$;

grant execute on function public.season_progress_by_department(text) to authenticated;

-- ─── verifications_summary — moderation view (service_role / dashboard) ──
-- Aggregated per installation, NOT row-by-row like annotations_pending —
-- the whole point is that Gabriel moderates on totals (8 confirm / 2
-- reject → fine; 5/5 split → look closer), not by reading every raw vote.

create or replace view public.verifications_summary
    with (security_invoker = true)
    as
    select
        campaign_id,
        detection_id,
        count(*) filter (where decision = 'confirm')   as confirms,
        count(*) filter (where decision = 'reject')    as rejects,
        count(*) filter (where decision = 'ambiguous') as ambiguous,
        count(*)                                       as total
    from public.verifications
    group by campaign_id, detection_id
    order by total desc;

revoke all on public.verifications_summary from anon, authenticated;

-- ─── spatial_ref_sys RLS advisory — deliberately NOT fixed here ───────────
-- Supabase's linter flags public.spatial_ref_sys (a table auto-created by
-- the PostGIS extension — the standard list of coordinate systems, e.g.
-- EPSG:4326/2154, no user data at all) as "Critical" for missing RLS. This
-- CANNOT be fixed with an ALTER/CREATE POLICY statement, by any role,
-- including postgres — Supabase's own extension-provisioning owns this
-- table under a system role, so any attempt fails with "must be owner of
-- table spatial_ref_sys" (confirmed against this project — see chat). The
-- only real fix is relocating the whole PostGIS extension out of `public`
-- into its own schema, which is a separate, deliberate migration (it can
-- affect every existing `geometry` column across the project) — not
-- something to fold into this script. Until that's done, safe to leave
-- this one specific advisory unresolved; the table holds nothing sensitive.
