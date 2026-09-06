-- ─── DeepPVMapper — RLS fix: campaigns / campaign_pool ───────────────────────
-- Run once in the Supabase SQL Editor (Dashboard → SQL Editor → New query).
--
-- Both tables were created (see scripts/verifications_setup.sql) without ever
-- enabling Row Level Security or revoking Supabase's default anon/authenticated
-- grants on new public-schema tables — the same class of gap already found and
-- fixed once for annotations_pending/issue_reports_pending (see
-- scripts/security_advisor_fixes.sql). Net effect before this fix: both tables
-- were very likely readable, and possibly writable, by anyone through the
-- public REST API (/rest/v1/campaigns, /rest/v1/campaign_pool), completely
-- bypassing get_verification_batch() and the other security-definer RPCs that
-- were built specifically to control access to this data.
--
-- Fix: enable RLS on both, with NO permissive policies for anon/authenticated.
-- This is deliberate, not a placeholder to fill in later — nothing in the
-- frontend (game/js/*.js) ever queries `campaigns` or `campaign_pool` directly;
-- every read goes through get_verification_batch() and the other RPCs, which
-- are `security definer` and therefore keep working exactly as before (they
-- run with the table owner's privileges, not the caller's, so RLS on the base
-- tables doesn't affect them at all). Deny-by-default here just closes the
-- direct-table path that nothing legitimate was using anyway.

alter table public.campaigns     enable row level security;
alter table public.campaign_pool enable row level security;

-- Belt-and-suspenders, matching the fix already applied to the *_pending
-- views: revoke the default API-role grants outright, in case they were
-- explicitly present rather than just implicit.
revoke all on public.campaigns     from anon, authenticated;
revoke all on public.campaign_pool from anon, authenticated;

-- No create policy statements — that's the point. Zero policies + RLS enabled
-- means every direct table request from anon/authenticated is denied, while
-- get_verification_batch() (security definer, set search_path = public) keeps
-- reading both tables internally without any change needed there.

-- ─── How to verify ────────────────────────────────────────────────────────
-- The SQL Editor runs as a privileged role and will still show rows from
-- both tables after this fix — expected, not a sign it didn't work. Confirm
-- through the REST API instead (swap in your anon key):
--
--   curl "https://zelhliylrlktnasircwp.supabase.co/rest/v1/campaign_pool?select=*&limit=1" \
--     -H "apikey: <anon-key>" -H "Authorization: Bearer <anon-key>"
--
-- Before this fix that likely returns rows; after it, it should return an
-- empty result or a permission-denied error instead. Also re-run the PV
-- Check game end-to-end afterwards (fetch a batch, submit a vote) to confirm
-- get_verification_batch()/insert into `verifications` still work unaffected.
