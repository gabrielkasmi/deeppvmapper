-- ─── DeepPVMapper — annotations overlay for zone downloads ────────────────────
-- Run once in the Supabase SQL Editor (Dashboard → SQL Editor → New query),
-- AFTER supabase_detections_setup.sql and supabase_setup.sql have been run.
--
-- Adds a "show unreviewed version" / "download with community edits" path
-- alongside the existing plain get_detections_in_zone RPC (left untouched —
-- this is purely additive, nothing that already calls that RPC changes
-- behavior). get_detections_in_zone_with_annotations returns the same zone,
-- but with public.annotations laid over public.detections:
--   * a 'delete' annotation removes the target detection from the result
--   * a 'modify' annotation replaces the target's geometry/properties
--   * an 'add' annotation contributes a brand-new feature (not in `detections`)
-- merged annotations are always applied; pending ones only when
-- include_unreviewed = true; rejected ones are never applied, regardless.
--
-- This is a REPLACE, not a UNION — a modified detection appears once, with
-- the corrected geometry, not twice (once as the original, once as the
-- edit). That's the whole point: no duplicates between the base layer and
-- the annotation layer.
--
-- Unreviewed (pending) 'add'/'modify' annotations carry only the polygon —
-- DeepPVMapper doesn't compute surface/kwp/year for community-submitted
-- geometry today (see the project notes on pypvroof), so those properties
-- come back null for anything sourced from an unreviewed annotation rather
-- than from `detections` itself. Don't rely on them being populated.
--
-- 2026-08-23 fix: both functions below need `security definer` (matching
-- annotation_count()/annotation_stats() in supabase_setup.sql). Without it,
-- a plain function runs as the CALLING role (anon, via PostgREST) — and
-- public.annotations has an insert-only RLS policy for anon, no select —
-- so every read of `annotations` inside these functions was silently
-- filtered to zero rows, not an error. Both are `create or replace`, so
-- re-running this file is safe and just patches them in place.
--
-- 2026-08-23: get_detections_in_zone_with_annotations now also returns
-- 'edit_action' ('modify' | 'add' | 'delete') on every unreviewed-edit
-- feature, so the map overlay's popup can say what kind of edit it's
-- showing, not just that it's unreviewed. 'delete' targets are still
-- excluded from `base` (correctly absent from the actual detections list)
-- but now come back separately, at their original geometry, purely so the
-- map can show where a deletion was flagged — export.js filters
-- edit_action='delete' back out before writing a CSV/GeoJSON.
--
-- 2026-08-23: also returns 'annotation_id' (the annotations.id uuid behind
-- each unreviewed-edit feature — target_id/id on the feature itself is the
-- DETECTION's id for 'modify'/'delete', not the annotation's own), plus a
-- new public.annotation_votes table + annotation_votes_summary view: lets
-- the popup offer "Confirm" / "Dispute" as an informational signal only —
-- it never changes an annotation's status (see the table's own comment).

create or replace function public.get_detections_in_zone_with_annotations(
    zone_geometry jsonb,
    include_unreviewed boolean default false,
    max_count int default 300000
) returns jsonb
language sql stable
security definer
set search_path = public
as $$
    with zone as (
        select ST_SetSRID(ST_GeomFromGeoJSON(zone_geometry::text), 4326) as z
    ),
    -- Most-recent-wins per target: if a target has more than one qualifying
    -- annotation (e.g. two pending edits on the same installation), only the
    -- latest one is applied — arbitration for merged annotations already
    -- happened by hand at review time, so this only matters for pending vs
    -- pending, which is inherently unarbitrated until reviewed.
    --
    -- 'add' rows have target_id = null (nothing existing to reference) —
    -- `distinct on (target_id)` groups ALL nulls together in Postgres, so
    -- applying it across the whole table was collapsing every pending 'add'
    -- in the zone down to just the single most recent one. Dedup only
    -- applies to delete/modify (real target_id, real conflict to arbitrate);
    -- every 'add' is independent and passes through untouched.
    applicable_annotations as (
        select distinct on (target_id) *
        from public.annotations
        where target_id is not null
          and (status = 'merged' or (include_unreviewed and status = 'pending'))
        order by target_id, created_at desc
    ),
    applicable_additions as (
        select *
        from public.annotations
        where target_id is null
          and action = 'add'
          and (status = 'merged' or (include_unreviewed and status = 'pending'))
    ),
    deleted_ids as (
        select id as annotation_id, target_id from applicable_annotations where action = 'delete'
    ),
    modified as (
        select id as annotation_id, target_id, geometry, properties
        from applicable_annotations
        where action = 'modify'
    ),
    added as (
        select id as annotation_id, geometry, properties
        from applicable_additions
    ),
    base as (
        select
            d.id::text as id,
            coalesce(
                ST_AsGeoJSON(ST_SetSRID(ST_GeomFromGeoJSON(m.geometry::text), 4326))::jsonb,
                ST_AsGeoJSON(d.geom)::jsonb
            ) as geometry,
            coalesce(
                m.properties,
                jsonb_build_object('surface', d.surface, 'kwp', d.kwp, 'year', d.first_seen)
            ) as properties,
            (m.target_id is not null) as is_unreviewed_edit,
            case when m.target_id is not null then 'modify' end as edit_action,
            m.annotation_id as annotation_id
        from public.detections d
        left join modified m on m.target_id = d.id::text
        join zone on d.geom && zone.z
        where d.id::text not in (select target_id from deleted_ids)
    ),
    additions as (
        select
            a.annotation_id::text as id,
            ST_AsGeoJSON(ST_SetSRID(ST_GeomFromGeoJSON(a.geometry::text), 4326))::jsonb as geometry,
            coalesce(a.properties, '{}'::jsonb) as properties,
            true as is_unreviewed_edit,
            'add' as edit_action,
            a.annotation_id as annotation_id
        from added a, zone
        where ST_SetSRID(ST_GeomFromGeoJSON(a.geometry::text), 4326) && zone.z
    ),
    -- Excluded from `base` (a reported false positive is correctly absent
    -- from the detections list), but still surfaced here with the target's
    -- ORIGINAL geometry/properties and edit_action='delete' — so the map
    -- overlay can show WHERE a deletion was flagged. Callers that build an
    -- actual detections export (CSV/GeoJSON — see export.js) filter these
    -- back out by edit_action before writing the file; the on-map overlay
    -- (render.js) keeps and displays them, styled apart from add/modify.
    deletions as (
        select
            d.id::text as id,
            ST_AsGeoJSON(d.geom)::jsonb as geometry,
            jsonb_build_object('surface', d.surface, 'kwp', d.kwp, 'year', d.first_seen) as properties,
            true as is_unreviewed_edit,
            'delete' as edit_action,
            del.annotation_id as annotation_id
        from public.detections d
        join deleted_ids del on del.target_id = d.id::text
        join zone on d.geom && zone.z
    ),
    combined as (
        select * from base
        union all
        select * from additions
        union all
        select * from deletions
    )
    select coalesce(jsonb_agg(
        jsonb_build_object(
            'type', 'Feature',
            'id', id,
            'geometry', geometry,
            'properties', properties || jsonb_build_object(
                'is_unreviewed_edit', is_unreviewed_edit,
                'edit_action', edit_action,
                'annotation_id', annotation_id
            )
        )
    ), '[]'::jsonb)
    from (select * from combined limit max_count) t;
$$;

grant execute on function public.get_detections_in_zone_with_annotations(jsonb, boolean, int) to anon;

-- Lightweight companion RPC for the map UI — "N unreviewed edits in this
-- zone" — so the toggle isn't a mystery checkbox when there's nothing to
-- show. Counts pending annotations whose target/geometry falls in the zone
-- (delete/modify matched by target_id against detections in the zone; add
-- matched by its own geometry).
create or replace function public.count_unreviewed_annotations_in_zone(
    zone_geometry jsonb
) returns int
language sql stable
security definer
set search_path = public
as $$
    with zone as (
        select ST_SetSRID(ST_GeomFromGeoJSON(zone_geometry::text), 4326) as z
    ),
    pending as (
        select * from public.annotations where status = 'pending'
    )
    select
        (select count(*) from pending p
           join public.detections d on d.id::text = p.target_id
           join zone on d.geom && zone.z
           where p.action in ('delete', 'modify'))
        +
        (select count(*) from pending p, zone
           where p.action = 'add'
             and ST_SetSRID(ST_GeomFromGeoJSON(p.geometry::text), 4326) && zone.z);
$$;

grant execute on function public.count_unreviewed_annotations_in_zone(jsonb) to anon;

-- ─── Community votes on unreviewed edits ("Confirm" / "Dispute") ──────────────
-- A lightweight, informational signal for the map's unreviewed-edits overlay
-- popup — NOT a moderation action. It never changes an annotation's status;
-- only you can merge/reject (Dashboard or service_role), same as before.
-- Deliberately kept that way: the map has no accounts/roles, so anyone could
-- click, and the annotations RLS policy already states "no self-validation"
-- on purpose (see supabase_setup.sql) — letting a public click flip an
-- annotation's status directly would undo that. This just gives you an extra
-- data point (how many people who looked at it agreed) to weigh at review
-- time; insert-only for anon, same pattern as `annotations`/`events`.
-- This whole file gets re-run as a unit whenever a function above changes
-- (create or replace is idempotent) — the table/policy/view below aren't, by
-- nature, so they're each guarded to make a re-run safe too.
create table if not exists public.annotation_votes (
    id            bigint generated always as identity primary key,
    created_at    timestamptz not null default now(),
    annotation_id uuid not null references public.annotations(id),
    vote          text not null check (vote in ('confirm', 'dispute'))
);

alter table public.annotation_votes enable row level security;

drop policy if exists "anon can insert votes" on public.annotation_votes;
create policy "anon can insert votes"
    on public.annotation_votes for insert
    to anon
    with check (true);
-- (no select/update/delete for anon — same "insert-only, moderate offline"
--  posture as public.annotations and public.events)

-- Convenience view for your review sessions — vote tallies per annotation,
-- pending ones only (a merged/rejected annotation's votes are moot). Not
-- granted to anon (matches the *_pending fix above: security_invoker so it
-- can't be used to bypass RLS on the underlying insert-only tables even if
-- a future grant slips in by accident).
create or replace view public.annotation_votes_summary
    with (security_invoker = true)
    as
    select
        a.id as annotation_id, a.action, a.target_id, a.created_at,
        count(*) filter (where v.vote = 'confirm') as confirms,
        count(*) filter (where v.vote = 'dispute') as disputes
    from public.annotations a
    join public.annotation_votes v on v.annotation_id = a.id
    where a.status = 'pending'
    group by a.id, a.action, a.target_id, a.created_at
    order by a.created_at;

revoke all on public.annotation_votes_summary from anon, authenticated;
