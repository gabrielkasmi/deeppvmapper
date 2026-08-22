-- ─── Aggregate RPCs for the per-département stats pages ──────────────────────
--
-- These return small, pre-aggregated result sets (roughly 94-96 rows each,
-- or a handful more for the yearly/source breakdowns) so the build script
-- never has to pull individual detection rows just to sum/count them.
--
-- Run this once in the Supabase SQL editor. Safe to re-run (CREATE OR REPLACE).

-- 1) Headline numbers per département: count, total capacity, rank by capacity.
create or replace function dept_capacity_stats()
returns table (
    dpt text,
    n_systems bigint,
    total_kwp numeric,
    rank_by_capacity bigint
)
language sql
stable
as $$
    select
        dpt,
        count(*)                                          as n_systems,
        sum(kwp)                                           as total_kwp,
        rank() over (order by sum(kwp) desc)               as rank_by_capacity
    from detections
    where dpt is not null
      and coalesce(false_positive, false) = false
    group by dpt;
$$;

-- 2) Yearly evolution per département, based on first_seen (when a system
--    first appeared in the imagery) — the more natural "growth over time"
--    story than last_seen (which just reflects the most recent observation).
--    ASSUMPTION: first_seen is already a year (int or text like "2020"), as
--    filters.js's parseInt(f.properties.last_seen) implies for its sibling
--    column. If first_seen is actually a full date in your schema, swap the
--    cast below for extract(year from first_seen)::int instead.
create or replace function dept_yearly_stats()
returns table (
    dpt text,
    year int,
    n_systems bigint,
    total_kwp numeric
)
language sql
stable
as $$
    select
        dpt,
        first_seen::int                                    as year,
        count(*)                                           as n_systems,
        sum(kwp)                                           as total_kwp
    from detections
    where dpt is not null
      and first_seen is not null
      and coalesce(false_positive, false) = false
    group by dpt, first_seen::int;
$$;

-- 3) Source breakdown per département. `sources` is a comma-separated list
--    of source indices (see SOURCE_LABELS in config.js: 0=DPVM, 1=FRPV,
--    2=OSM, 3=Manual correction, 4=Recall sample) — a single detection can
--    carry more than one, so this unnests rather than treating it as one
--    categorical value.
-- The unnest cross-join below is meaningfully heavier than the two plain
-- aggregates above (one extra row per source per detection before the
-- group by), and hit the default statement_timeout on a full-table run —
-- the `set` clause raises the timeout for calls to this one function only,
-- everything else on the role/database keeps the default.
create or replace function dept_source_stats()
returns table (
    dpt text,
    source_id int,
    n_systems bigint
)
language sql
stable
set statement_timeout to '120000'
as $$
    select
        dpt,
        trim(s)::int                                       as source_id,
        count(*)                                           as n_systems
    from detections, unnest(string_to_array(sources, ',')) as s
    where dpt is not null
      and sources is not null
      and coalesce(false_positive, false) = false
    group by dpt, trim(s)::int;
$$;

-- Grant execute to the anon role (matches the existing get_detections_* RPCs).
grant execute on function dept_capacity_stats() to anon;
grant execute on function dept_yearly_stats()   to anon;
grant execute on function dept_source_stats()   to anon;
