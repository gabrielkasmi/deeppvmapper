-- ─── DeepPVMapper — annotations backend ──────────────────────────────────────
-- Run once in the Supabase SQL Editor (Dashboard → SQL Editor → New query).
--
-- Design:
--   * one insert-only table; anonymous visitors can submit, never read/edit
--   * rows are never deleted — status moves pending → merged | rejected,
--     so count(*) is a lifetime community counter
--   * annotation_count() is the only public read, callable from any page

create table public.annotations (
    id          uuid primary key default gen_random_uuid(),
    created_at  timestamptz not null default now(),
    action      text not null check (action in ('delete', 'modify', 'add')),
    target_id   text,             -- WFS feature id (null for 'add')
    geometry    jsonb,            -- new/modified GeoJSON geometry (null for 'delete')
    properties  jsonb,            -- optional: surface, kwp… as submitted
    original    jsonb,            -- snapshot of the original feature (for 'modify'/'delete'),
                                  -- lets you merge even if the dataset moved meanwhile
    comment     text check (char_length(comment) <= 500),
    status      text not null default 'pending'
                check (status in ('pending', 'merged', 'rejected'))
);

-- Row Level Security: anonymous = insert only, immutable once submitted.
alter table public.annotations enable row level security;

create policy "anon can insert"
    on public.annotations for insert
    to anon
    with check (
        status = 'pending'                                       -- no self-validation
        and coalesce(pg_column_size(geometry),   0) < 50000      -- crude payload caps
        and coalesce(pg_column_size(original),   0) < 50000      -- (coalesce: NULL columns
        and coalesce(pg_column_size(properties), 0) < 5000       --  must pass the check)
    );
-- (no select / update / delete policies for anon — you moderate via the
--  Dashboard or the service_role key)

-- Lifetime counter, readable by anyone (the table itself stays unreadable).
create or replace function public.annotation_count()
returns bigint
language sql
security definer
set search_path = public
as $$ select count(*) from public.annotations $$;

grant execute on function public.annotation_count() to anon;

-- Convenience view for your moderation sessions (service role / dashboard only).
create view public.annotations_pending as
    select id, created_at, action, target_id, comment, geometry, original
    from public.annotations
    where status = 'pending'
    order by created_at;

-- ─── Lightweight usage events (e.g. CSV downloads) ───────────────────────────
-- Insert-only for anon, unreadable publicly; analyse via the Dashboard.

create table public.events (
    id          bigint generated always as identity primary key,
    created_at  timestamptz not null default now(),
    event       text not null check (char_length(event) <= 50),
    detail      text check (char_length(detail) <= 200)
);

alter table public.events enable row level security;

create policy "anon can insert events"
    on public.events for insert
    to anon
    with check (true);
