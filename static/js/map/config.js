// ─── Shared configuration ─────────────────────────────────────────────────────

// Two-tier fetch, so a selection can scale up to région size:
//   POINTS_RPC     — light: centroid lat/lng + slim properties, no geometry.
//                    Fetched once for the whole selection, whatever its size —
//                    this is what feeds the marker cluster, which is the ONLY
//                    thing shown until a single installation is clicked.
//   DETECTIONS_RPC — full: real polygon outlines. Only ever fetched for a tiny
//                    box around a clicked marker, to pull that one
//                    installation's exact shape — never for a whole viewport.
//   ZONE_RPC       — exact, uncapped, full geometry — GeoJSON export only,
//                    called once against the selection's real boundary.
//
// Fast path for a named-place selection (région/département/commune): when
// the selection resolves to one of those (see geo.js), skip the spatial
// bbox/intersection test entirely and match detections.insee / detections.dpt
// directly — an indexed equality lookup beats a GiST scan over a large bbox,
// which is why département/commune searches load noticeably faster than a
// hand-drawn box of similar size. ADMIN_POINTS_RPC/ADMIN_ZONE_RPC take that
// path; a hand-drawn rectangle (no admin code to match) still uses the bbox
// RPCs above, with the polygon/point-in-polygon intersection as a fallback
// for a named place we couldn't resolve to a clean admin code.
export const POINTS_RPC      = 'get_detections_bbox_points';
export const DETECTIONS_RPC  = 'get_detections_bbox';
export const ZONE_RPC        = 'get_detections_in_zone';
export const ADMIN_POINTS_RPC = 'get_detections_admin_points';
export const ADMIN_ZONE_RPC   = 'get_detections_admin';

export const POINTS_MAX      = 200000;  // whole-selection light fetch
export const SINGLE_FETCH_MAX = 50;     // tiny box around a click — just needs the one polygon
export const ZONE_FETCH_MAX  = 300000;  // exhaustive GeoJSON export

export const GEO_BASE   = '../static/data/geo';

// Half-width (in degrees) of the little box fetched around a clicked marker
// to pull its real polygon — generous enough to contain a large rooftop
// without pulling in a meaningful chunk of the neighbourhood.
export const SINGLE_FETCH_RADIUS_DEG = 0.0015;

// Marker cluster stops clustering (shows individual markers) at/above this zoom.
export const POLYGON_ZOOM = 15;

export const FRANCE_BOUNDS = [[41.2, -5.3], [51.2, 9.7]];

// sources: comma-separated indices baked into `detections.sources` by the
// detection pipeline (see scripts/ — not derived, just documented here).
export const SOURCE_LABELS = {
    0: 'DPVM', 1: 'FRPV', 2: 'OSM', 3: 'Manual correction', 4: 'Recall sample'
};

// IGN's ORTHOIMAGERY.ORTHOPHOTOS tile matrix is PM_0_19 (native tiles stop at
// z19 — IGN dropped PM_0_21 support in 2025). Past this, the WMTS has nothing
// to serve and tiles go blank. Locked here as the hard ceiling for the map.
export const MAX_ZOOM = 19;

// ─── Annotation backend (Supabase) ────────────────────────────────────────────
// Paste your project values (Dashboard → Settings → API). The anon key is
// designed to be public — RLS does the protecting. Leave empty to run the
// annotation UI in local-only mode (visual edits, no submission).
export const SUPABASE_URL      = 'https://zelhliylrlktnasircwp.supabase.co';
export const SUPABASE_ANON_KEY = 'sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi';   // Dashboard → Settings → API Keys → "anon public"

export const ANNOT_MAX_SESSION = 50;    // submissions per session
export const ANNOT_MIN_INTERVAL_MS = 2000;

// IGN orthophoto — the "Satellite" alt base layer (default base layer is a
// plain OSM map, which always renders; this stays an option in the layer
// control since it's occasionally more useful than the map for confirming a
// rooftop, but it shouldn't be the first thing people see).
export const IGN_ORTHO = 'https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0' +
    '&LAYER=ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&FORMAT=image/jpeg' +
    '&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}';
