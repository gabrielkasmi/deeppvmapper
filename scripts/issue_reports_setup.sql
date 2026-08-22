-- ─── DeepPVMapper — "Report an issue" backend ────────────────────────────────
-- Run once in the Supabase SQL Editor (Dashboard → SQL Editor → New query).
-- Same design as public.annotations (see supabase_setup.sql): one insert-only
-- table, anonymous visitors can submit, never read/edit. This one is for
-- reports that don't fit annotate.js's delete/modify/add actions (which
-- already cover false positives, wrong shapes, and missing installations
-- with instant, geometry-based edits) — attribute errors, a utility-scale
-- plant mistagged as rooftop, or anything else worth flagging.
--
-- Reports are moderated offline (Dashboard, or the service_role key) and,
-- once confirmed, promoted by hand to a GitHub issue on
-- gabrielkasmi/openpvmapper-issues — this table is just the intake inbox,
-- not the public tracker.

create table public.issue_reports (
    id           uuid primary key default gen_random_uuid(),
    created_at   timestamptz not null default now(),
    category     text not null
                 check (category in ('missing_attributes', 'utility_scale', 'other')),
    target_type  text not null check (target_type in ('installation', 'zone')),
    target_id    text,             -- WFS feature id, when target_type = 'installation'
    target_label text,             -- human label shown back to the reporter (commune, "custom area"…)
    admin        jsonb,            -- { inseeCodes: [...] } | { deptCodes: [...] } | null
    map_url      text,             -- link back to the exact view being reported
    comment      text not null check (char_length(comment) between 1 and 1000),
    status       text not null default 'pending'
                 check (status in ('pending', 'resolved', 'dismissed'))
);

-- Row Level Security: anonymous = insert only, immutable once submitted.
alter table public.issue_reports enable row level security;

create policy "anon can insert issue reports"
    on public.issue_reports for insert
    to anon
    with check (
        status = 'pending'                                  -- no self-validation
        and coalesce(pg_column_size(admin), 0) < 2000        -- crude payload caps
        and coalesce(length(map_url), 0) < 2000
    );
-- (no select / update / delete policies for anon — you moderate via the
--  Dashboard or the service_role key)

-- Convenience view for your triage sessions (service role / dashboard only).
create view public.issue_reports_pending as
    select id, created_at, category, target_type, target_label, admin, map_url, comment
    from public.issue_reports
    where status = 'pending'
    order by created_at;
