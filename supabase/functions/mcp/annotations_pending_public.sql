-- Public, privacy-safe view over annotations_pending for the MCP server.
-- Deliberately excludes `comment` (free text) and `original` (raw snapshot).
-- Re-running this (create or replace) is safe at any time: nothing else
-- depends on this view yet.

create or replace view public.annotations_pending_public as
select
  ap.created_at,
  ap.action,
  ap.target_id,
  ap.geometry,                -- GeoJSON Polygon, only populated for action='add'
  d.dpt as target_dpt,        -- département of the existing detection being modified/deleted
  d.insee as target_insee
from public.annotations_pending ap
left join public.detections d
  on ap.target_id is not null
  and ap.target_id ~ '^[0-9]+$'
  and d.id = ap.target_id::integer
order by ap.created_at desc;

grant select on public.annotations_pending_public to anon;
