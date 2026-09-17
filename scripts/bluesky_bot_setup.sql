-- ─── Bluesky bot — automated progress posts ────────────────────────────────
-- Run once in the Supabase SQL Editor (same project as verifications_setup.sql
-- / supabase_setup.sql). Rerunnable, same posture as the other *_setup.sql
-- files in this folder — `create table if not exists`, `on conflict do
-- nothing`, `grant` statements are idempotent by nature.
--
-- The bot (scripts/bluesky-bot/post-updates.mjs, run on a schedule via
-- GitHub Actions — see .github/workflows/bluesky-bot.yml) is a trusted,
-- server-side script that never runs in a browser, so it authenticates to
-- Supabase with the project's SERVICE ROLE key rather than the public anon
-- key used by the website/game. That key bypasses RLS entirely, but
-- Postgres GRANTs are still role-based and checked independently of RLS —
-- the `grant ... to service_role` statements below are what actually let it
-- call these functions; without them, calls from the bot fail with 42501
-- ("permission denied for function ...") even with a valid service_role key.
-- (Supabase's default project setup usually already grants service_role
-- broad EXECUTE via a schema-level default privilege, so these may be
-- no-ops in practice — they're here so the bot doesn't silently depend on
-- that default rather than an explicit, visible grant.)

-- Read-only stats RPCs the bot needs, already granted to anon/authenticated
-- for the website/game (see supabase_setup.sql, verifications_setup.sql) —
-- service_role added alongside; the existing grants are untouched.
grant execute on function public.annotation_stats() to service_role;
grant execute on function public.leaderboard(text, int) to service_role;
grant execute on function public.season_completion(text) to service_role;

-- ─── bluesky_bot_state — what the bot has already announced ───────────────
-- Singleton row (id is always 1) so a scheduled run only posts on a genuine
-- new milestone/leaderboard change, not every single time the workflow
-- fires. RLS is enabled with NO policies for anon/authenticated — only
-- service_role (which bypasses RLS) can read or write this table, which is
-- exactly and only what the bot uses. Nothing here is sensitive (it's just
-- "what did we last post"), but it has no reason to be readable from the
-- browser either.
create table if not exists public.bluesky_bot_state (
    id                          int primary key default 1,
    -- Snapshot of the all-time top 5 as of the last successful "leaderboard
    -- changed" post: [{"pseudo": "...", "total": 123}, ...], ordered by
    -- rank. Compared against the current top 5 on every run; a post only
    -- goes out (and this gets overwritten) when the two differ.
    last_top5                   jsonb,
    -- Same idea, for the rolling-7-day leaderboard (leaderboard(p_window=
    -- 'week') — a rolling window, not a fixed calendar week, so this is
    -- checked/compared the same way as last_top5 above, just against a
    -- different window).
    last_week_top5              jsonb,
    -- Last multiple of 50 announced for installations_done (season_completion()).
    -- A post only goes out when floor(current / 50) > floor(this / 50).
    last_installations_posted   int not null default 0,
    -- Same idea for annotation_stats().count, in steps of 1000.
    last_annotations_posted     int not null default 0,
    updated_at                  timestamptz not null default now(),
    constraint bluesky_bot_state_singleton check (id = 1)
);

-- For a project that already ran this script before last_week_top5 existed
-- (create table if not exists is a no-op on an existing table, so the new
-- column needs adding explicitly here).
alter table public.bluesky_bot_state
    add column if not exists last_week_top5 jsonb;

insert into public.bluesky_bot_state (id) values (1)
on conflict (id) do nothing;

alter table public.bluesky_bot_state enable row level security;

grant select, update on public.bluesky_bot_state to service_role;
