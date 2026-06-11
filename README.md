# DeepPVMapper — Project Site & Interactive Registry

Project page and interactive map for **DeepPVMapper**, an open-source deep learning pipeline
for remote sensing of rooftop PV systems at national scale. The registry covers **460,000+ raw
detections (~2.3 GWp)** across metropolitan France, produced from IGN BD ORTHO® aerial imagery.

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
  distributions, CSV export). Exit by clicking outside the zone.
- **Heatmap regimes**: below z11, precomputed commune-centroid heat (hotspot-truncated,
  `HEAT_TOPSHARE`); z11–15, live heat from WFS detections (full commune set when a commune
  is locked); above z15, individual detection polygons take over (colored by kWp, clickable).
- **Community annotations**: report a false detection, redraw an outline (leaflet-geoman),
  or add a missed installation. Edits apply visually for the session and are submitted to
  Supabase for offline moderation. Lifetime contribution counter, also surfaced on the
  landing page above a threshold (`index.html?contrib` to preview).
- **Search** (Nominatim) jumps straight into target mode on the matched commune/département/région.

### Architecture

```
                       ┌──────────────────────────────┐
   GitHub Pages        │  IGN Géoplateforme (WFS)     │   source of truth:
   (static site)  ───► │  cartes.gouv.fr              │   460k detection polygons
                       └──────────────────────────────┘
        │                       ▲ live queries (viewport / commune bbox / stats samples)
        │              ┌──────────────────────────────┐
        └────────────► │  Supabase (free tier)        │   annotations (insert-only, RLS)
                       │  + events (usage tracking)   │   + public annotation_count() RPC
                       └──────────────────────────────┘
```

No build step, no server: plain ES modules served by GitHub Pages.

- **Frontend**: Leaflet 1.9 + leaflet.heat + leaflet-geoman + Chart.js (CDN), vanilla JS.
- **Detections**: never stored in the repo — fetched live from the IGN WFS
  (`data.geopf.fr`). Viewport streaming above z11; commune-anchored fetch (bbox + point-in-
  polygon filter) when a commune is locked, so dense cities display *all* their systems.
- **Geometries** (`static/data/geo/`): régions/départements/communes from
  [france-geojson](https://github.com/gregoiredavid/france-geojson) (simplified), communes
  split per département for lazy loading. ~21 MB committed.
- **Precomputed stats** (`static/data/stats/`): exact per-commune/département/région counts
  and kWp, built offline by `scripts/build_geo_stats.py` (spatial join of all detections on
  commune polygons). Powers the tooltips, the stats cards and the national heatmap.
  WFS can't aggregate server-side, hence the precomputation.
- **Backend** (Supabase): two insert-only tables behind RLS — `annotations` (community
  edits, never deleted, moderated via status `pending → merged/rejected`) and `events`
  (lightweight usage tracking: visits, zone locks, searches, CSV downloads, WFS errors).
  The anon "publishable" key in `config.js` is public by design; RLS does the protecting.
  Setup: `scripts/supabase_setup.sql`.

### Code layout

```
content/map.html              page shell, UI elements, CSS
static/js/map/
  config.js                   all tunables: zoom bands, heat params, WFS + Supabase config
  store.js                    shared state, geometry helpers (PIP, centroid), logEvent
  layers.js                   base tiles, heatmap regimes, WFS fetches, mask, session edits
  nav.js                      explore/target state machine, zone resolution, breadcrumb
  stats.js                    stats card (baked KPIs + sampled distributions), CSV export
  search.js                   Nominatim search → target mode
  annotate.js                 annotation flows (delete/redraw/add), Supabase submission
  main.js                     boot + intro popup
scripts/
  build_geo_stats.py          offline stats build (WFS paging or --local dump)
  supabase_setup.sql          tables, RLS policies, annotation_count() RPC
```

### Data release cycle

1. Collect validated annotations from Supabase (`annotations_pending` view → `merged`).
2. Apply them to the master GeoJSON (match by feature id, fallback on the `original`
   geometry snapshot — WFS feature ids change across publications).
3. Re-publish on cartes.gouv.fr → update `WFS_LAYER` in `config.js`.
4. Re-run `python3 scripts/build_geo_stats.py` → commit the refreshed stats.

### Local development

```bash
python3 -m http.server          # ES modules require a server (file:// won't work)
# → http://localhost:8000/content/map.html
```

All tunables live in `static/js/map/config.js`: zoom bands (`BANDS`, `HEAT_FADE`),
hotspot truncation (`HEAT_TOPSHARE`), plan overlay opacity, annotation rate limits.

## Contact

Gabriel Kasmi — [gabriel.kasmi.services@gmail.com](mailto:gabriel.kasmi.services@gmail.com)
