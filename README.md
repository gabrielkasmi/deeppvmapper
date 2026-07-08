# DeepPVMapper — Project Site & Interactive Registry

Project page and interactive map for **DeepPVMapper**, an open-source deep learning pipeline
for remote sensing of rooftop PV systems at national scale. The registry covers **580,000+ raw
detections (~3.1 GWp)** across metropolitan France, produced from IGN BD ORTHO® aerial imagery.

Live site: [gabrielkasmi.github.io/deeppvmapper](https://gabrielkasmi.github.io/deeppvmapper)

## Site map

| Page | Purpose |
|---|---|
| `index.html` | Project landing page: results, validation, registry audit study, citation |
| `content/map.html` | **Interactive registry** (the main application, see below) |
| `content/pipeline.html` | The DeepPVMapper detection pipeline |
| `content/main-results.html` | Detailed results |
| `content/in-press.html`, `content/outlook.html` | Press & outlook |

## The interactive map

### Features

- **Explore mode** (default): outlines staged by zoom — régions (z < 9), départements (9–11),
  communes (≥ 11, lazy-loaded per département). The zone under the cursor is highlighted
  (outline + dim mask) with a tooltip showing its exact system count and capacity.
- **Target mode**: clicking a zone (or picking a search result) locks onto it — boundary,
  mask, heatmap scoped to the zone, and a stats card (exact counts, capacity/surface
  distributions). Département/commune zones get an exhaustive CSV export (zipped); région
  and national bulk downloads point to the static dump on [Zenodo](https://zenodo.org/records/19188878)
  instead (linked from the intro popup) rather than through the live API.
- **Heatmap regimes**: below z11, precomputed commune-centroid heat (hotspot-truncated,
  `HEAT_TOPSHARE`); z11–15, live heat from the detections backend (full commune set when a
  commune is locked); above z15, individual detection polygons take over (colored by kWp,
  clickable).
- **Community annotations** (**Contribute** button): report a false detection, redraw an
  outline (leaflet-geoman), or add a missed installation. Edits apply visually for the
  session and are submitted to Supabase for offline moderation. Capped at 50 submissions
  per session — a popup then prompts a reload to continue. Lifetime contribution count +
  last-contribution date shown bottom-left.
- **Search** (Nominatim) jumps straight into target mode on the matched commune/département/région.
- Map position (center + zoom) persists across a page reload (`sessionStorage`).

### Architecture

```
                       ┌────────────────────────────────────────┐
   GitHub Pages        │  Supabase (Postgres + PostGIS)          │
   (static site)  ───► │  • detections — 580k+ polygons, RLS     │   source of truth:
                       │    read-only. RPCs:                     │   detection polygons
                       │    get_detections_bbox (viewport/       │
                       │    commune streaming), get_detections_  │
                       │    in_zone (exact, uncapped — CSV export)│
                       │  • annotations — insert-only, RLS        │
                       │  • events — usage tracking               │
                       └────────────────────────────────────────┘
```

No build step, no server: plain ES modules served by GitHub Pages. Self-hosted since the
IGN Géoplateforme "bac à sable" (sandbox) WFS this originally ran on turned out to be exactly
that — a sandbox, not a production service, and got wiped.

- **Frontend**: Leaflet 1.9 + leaflet.heat + leaflet-geoman + Chart.js + JSZip (CDN), vanilla JS.
- **Detections**: never stored in the repo — self-hosted in Supabase Postgres/PostGIS
  (`detections` table, imported via `ogr2ogr`, see `scripts/supabase_detections_setup.sql`).
  Viewport streaming above z11 via `get_detections_bbox` (GIST-indexed bbox filter);
  commune-anchored fetch (bbox + point-in-polygon) when a commune is locked, so dense cities
  display *all* their systems. CSV export uses `get_detections_in_zone` (exact polygon
  containment, no cap) so it always includes every system in the zone, not a sample.
- **Geometries** (`static/data/geo/`): régions/départements/communes from
  [france-geojson](https://github.com/gregoiredavid/france-geojson) (simplified), communes
  split per département for lazy loading. ~21 MB committed.
- **Precomputed stats** (`static/data/stats/`): exact per-commune/département/région counts
  and kWp, built offline by `scripts/build_geo_stats.py --local <geojson>` (spatial join of
  all detections on commune polygons). Powers the tooltips, the stats cards and the national
  heatmap — a single SQL query can't cheaply produce this aggregation, hence the precomputation.
- **Backend** (Supabase, same project as the detections table): two more insert-only tables
  behind RLS — `annotations` (community edits, never deleted, moderated via status
  `pending → merged/rejected`) and `events` (lightweight usage tracking: visits, zone locks,
  searches, CSV downloads, fetch errors). The anon "publishable" key in `config.js` is public
  by design; RLS does the protecting. Setup: `scripts/supabase_setup.sql` (annotations/events)
  and `scripts/supabase_detections_setup.sql` (detections RPCs).

### Code layout

```
content/map.html              page shell, UI elements, CSS
static/js/map/
  config.js                   all tunables: zoom bands, heat params, detections/zone RPC
                               names, Supabase config
  store.js                    shared state, geometry helpers (PIP, centroid), logEvent,
                               Supabase client + detection fetch helpers
  layers.js                   base tiles, heatmap regimes, detection fetches (Supabase
                               RPC), mask, session edits
  nav.js                      explore/target state machine, zone resolution
  stats.js                    stats card (baked KPIs + sampled distributions), exhaustive
                               CSV export (zipped)
  search.js                   Nominatim search → target mode
  annotate.js                 annotation flows (delete/redraw/add via Contribute button),
                               session cap + reload prompt, Supabase submission
  main.js                     boot + intro popup + view persistence (sessionStorage)
scripts/
  build_geo_stats.py          offline stats build (--local dump of the detections GeoJSON)
  supabase_setup.sql          annotations/events tables, RLS policies, annotation_stats() RPC
  supabase_detections_setup.sql   detections table indexing/RLS, get_detections_bbox and
                                   get_detections_in_zone RPCs
```

### Data release cycle

1. Collect validated annotations from Supabase (`annotations_pending` view → `merged`).
2. Apply them to the master GeoJSON (match by feature id, fallback on the `original`
   geometry snapshot — feature ids can shift across publications).
3. Re-import into the `detections` table (`ogr2ogr`, see
   `scripts/supabase_detections_setup.sql`) and re-publish the updated file on Zenodo.
4. Re-run `python3 scripts/build_geo_stats.py --local <geojson>` → commit the refreshed stats.

### Local development

```bash
python3 -m http.server          # ES modules require a server (file:// won't work)
# → http://localhost:8000/content/map.html
```

All tunables live in `static/js/map/config.js`: zoom bands (`BANDS`, `HEAT_FADE`),
hotspot truncation (`HEAT_TOPSHARE`), plan overlay opacity, annotation rate limits.

## Contact

Gabriel Kasmi — [gabriel.kasmi.services@gmail.com](mailto:gabriel.kasmi.services@gmail.com)
