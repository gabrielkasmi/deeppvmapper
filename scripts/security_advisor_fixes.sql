-- ─── DeepPVMapper — Supabase Security Advisor fixes ───────────────────────────
-- Run each numbered block as its OWN separate query in the Supabase SQL
-- Editor (select the block's text, or open a fresh query per block) — do
-- NOT paste the whole file and run it in one go. Supabase runs a pasted
-- query as a single implicit transaction: if block 2 errors (as it did —
-- "must be owner of table spatial_ref_sys"), Postgres rolls back every
-- statement that ran earlier in that SAME transaction, including block 1's
-- fixes. Keeping them as separate executions means a failure in block 2
-- can never undo block 1.
--
-- Addresses the Security Advisor's "Critical" findings plus one "Warning":
--   1. Security Definer View  — public.annotations_pending
--   1. Security Definer View  — public.issue_reports_pending
--   2. RLS Disabled in Public — public.spatial_ref_sys
--   3. Duplicate Index        — public.detections (Warning, not Critical)


-- ═══ BLOCK 1 — run this one first, on its own ════════════════════════════
-- The fix that actually matters: both views were meant as "service role /
-- dashboard only" per their own setup-script comments (supabase_setup.sql,
-- issue_reports_setup.sql), but neither script ever revoked the default
-- anon/authenticated grants Supabase applies to new public-schema objects,
-- and a plain view runs with its OWNER's privileges, not the querying
-- user's — so it bypasses the base table's RLS entirely. Net effect:
-- *_pending, meant to be internal triage-only, was very likely readable by
-- anyone through the public REST API.

-- Force RLS/privilege enforcement to the QUERYING user (Postgres 15+, the
-- default on current Supabase projects) instead of the view owner...
alter view public.annotations_pending   set (security_invoker = true);
alter view public.issue_reports_pending set (security_invoker = true);

-- ...and, belt-and-suspenders, revoke the default API-role grants outright —
-- these views should only ever be queried from the Dashboard or with the
-- service_role key, never through anon/authenticated PostgREST access.
revoke all on public.annotations_pending   from anon, authenticated;
revoke all on public.issue_reports_pending from anon, authenticated;

-- Sanity check after running (see below for how to run this as anon):
-- it should now return "permission denied" instead of rows.
--   select * from public.annotations_pending;
--   select * from public.issue_reports_pending;


-- ═══ BLOCK 2 — separate query, OK if this one fails ══════════════════════
-- spatial_ref_sys: a PostGIS system table (bundled with the extension),
-- holding only the public catalog of coordinate reference systems — no
-- application or user data, so this finding carries near-zero real risk,
-- unlike block 1. The ALTER below needs table ownership, which the role
-- your SQL Editor connects as often doesn't have here (that's exactly the
-- "must be owner of table spatial_ref_sys" error you hit) — PostGIS's
-- install role owns it, not you. If it keeps failing, it's fine to
-- acknowledge/ignore this one specific finding in the Advisor rather than
-- chase ownership further; it's a well-known PostGIS quirk on existing
-- projects, not a DeepPVMapper-specific exposure like the two views above.
alter table public.spatial_ref_sys enable row level security;
create policy "Public read access (reference data)"
    on public.spatial_ref_sys for select
    to public
    using (true);


-- ═══ BLOCK 3 — separate query, unrelated to blocks 1/2 ═══════════════════
-- Duplicate index on public.detections: detections_geom_idx and
-- detections_geom_geom_idx are identical GIST indexes on the same geom
-- column — pure waste (extra disk space, extra write overhead on every
-- insert), not a security issue, hence "Warning" rather than "Critical".
-- scripts/supabase_detections_setup.sql only ever creates
-- detections_geom_idx (line 15), so detections_geom_geom_idx is the stray
-- duplicate — almost certainly added later some other way (Table Editor
-- UI, a one-off manual statement) without anyone noticing the first one
-- already existed. Keeping the one the repo's own setup script defines
-- keeps that script an accurate record of the schema.
drop index if exists public.detections_geom_geom_idx;


-- ═══ How to verify block 1 actually took effect ══════════════════════════
-- The SQL Editor itself runs as a privileged role, so it will still show
-- rows from *_pending even after the fix — that's expected and not a sign
-- it didn't work. To confirm anon is actually locked out, test through the
-- REST API instead (swap in your project ref and anon key):
--
--   curl "https://<project-ref>.supabase.co/rest/v1/issue_reports_pending?select=*" \
--     -H "apikey: <anon-key>" -H "Authorization: Bearer <anon-key>"
--
-- Before the fix this returns the pending rows; after it, it should return
-- a permission-denied error instead.
