-- ─── DeepPVMapper — detections backend (self-hosted, replaces the IGN WFS) ───
-- Run once in the Supabase SQL Editor (Dashboard → SQL Editor → New query),
-- AFTER the `detections` table has been imported via ogr2ogr:
--
--   ogr2ogr -f PostgreSQL \
--     "PG:host=db.nvtjkzxoothrilrnlkym.supabase.co user=postgres dbname=postgres password=<DB_PASSWORD> port=5432" \
--     france_detections.geojson \
--     -nln detections -lco GEOMETRY_NAME=geom -lco FID=id \
--     -nlt PROMOTE_TO_MULTI -t_srs EPSG:4326 -progress
--
-- Requires the postgis extension (create extension if not exists postgis;)
-- enabled BEFORE the ogr2ogr import, otherwise geometry lands in a plain
-- wkb_geometry column instead of a real PostGIS `geometry` type.
--
-- 2026-08-23 fix: get_detections_bbox/get_detections_in_zone referenced a
-- `year` column that doesn't exist on `detections` (real column is
-- `first_seen`, the detection's first-seen year — same convention as
-- build_department_pages.py). Both functions are `create or replace`, so
-- re-running this whole file is safe and just patches them in place.

create index if not exists detections_geom_idx on public.detections using gist (geom);

-- Row Level Security: public read-only (the frontend uses the anon key).
alter table public.detections enable row level security;
drop policy if exists "public read" on public.detections;
create policy "public read" on public.detections for select using (true);
grant select on public.detections to anon;

-- Bbox fetch — mirrors the old WFS BBOX GetFeature. `&&` is the GIST-indexed
-- bbox-overlap operator, same semantics as the old CQL_FILTER BBOX(...).
create or replace function public.get_detections_bbox(
    min_lon double precision, min_lat double precision,
    max_lon double precision, max_lat double precision,
    max_count int default 2000
) returns jsonb
language sql stable
as $$
    select coalesce(jsonb_agg(
        jsonb_build_object(
            'type', 'Feature',
            'id', id,
            'geometry', ST_AsGeoJSON(geom)::jsonb,
            'properties', jsonb_build_object(
                'surface', surface,
                'kwp', kwp,
                'year', first_seen
            )
        )
    ), '[]'::jsonb)
    from (
        select * from public.detections
        where geom && ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)
        limit max_count
    ) t;
$$;

grant execute on function public.get_detections_bbox(double precision, double precision, double precision, double precision, int) to anon;

-- Full-zone fetch — used by the CSV export only. Filters by exact polygon
-- containment (centroid inside the zone), not by bbox, and has no low cap
-- like the viewport fetch — the download must never silently truncate.
-- `&&` first narrows via the GIST index, ST_Contains refines exactly.
create or replace function public.get_detections_in_zone(
    zone_geometry jsonb,
    max_count int default 300000
) returns jsonb
language sql stable
as $$
    select coalesce(jsonb_agg(
        jsonb_build_object(
            'type', 'Feature',
            'id', id,
            'geometry', ST_AsGeoJSON(geom)::jsonb,
            'properties', jsonb_build_object(
                'surface', surface,
                'kwp', kwp,
                'year', first_seen
            )
        )
    ), '[]'::jsonb)
    from (
        select d.*
        from public.detections d,
             (select ST_SetSRID(ST_GeomFromGeoJSON(zone_geometry::text), 4326) as z) zg
        where d.geom && zg.z
          and ST_Contains(zg.z, ST_Centroid(d.geom))
        limit max_count
    ) t;
$$;

grant execute on function public.get_detections_in_zone(jsonb, int) to anon;
