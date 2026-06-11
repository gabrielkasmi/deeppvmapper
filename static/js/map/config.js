// ─── Shared configuration ─────────────────────────────────────────────────────

export const WFS_URL   = 'https://data.geopf.fr/sandbox/wfs';
export const WFS_LAYER = 'SANDBOX.france_detections_wgs84_geojson_09_06_2026_wfs:france_detections_wgs84';
export const WFS_GEOM  = 'wkb_geometry';

export const GEO_BASE   = '../static/data/geo';
export const STATS_BASE = '../static/data/stats';

// Zoom bands. In EXPLORE mode they stage the hoverable outlines:
//   z <  dept     régions     |   dept <= z < commune   départements
//   z >= commune  communes (of the dept under the cursor)
// `detail` is where WFS detection polygons start being fetched, and HEAT_FADE
// is where the heatmap finally gives way to them (polygons clearly visible).
export const BANDS = { region: 7, dept: 9, commune: 11, detail: 13 };
export const HEAT_FADE = 15;

export const STATS_SAMPLE = 5000;  // WFS sample cap for distribution charts
export const MIN_STATS_N  = 30;    // below this, KPIs only — no distributions

// Precomputed heatmap: keep only the top `topShare` of communes by system
// count, per displayed level. 1.0 = show everything.
export const HEAT_TOPSHARE = { france: 0.10, region: 0.35, dept: 1.0 };

export const FRANCE_BOUNDS = [[41.2, -5.3], [51.2, 9.7]];

// ─── Annotation backend (Supabase) ────────────────────────────────────────────
// Paste your project values (Dashboard → Settings → API). The anon key is
// designed to be public — RLS does the protecting. Leave empty to run the
// annotation UI in local-only mode (visual edits, no submission).
export const SUPABASE_URL      = 'https://nvtjkzxoothrilrnlkym.supabase.co';
export const SUPABASE_ANON_KEY = 'sb_publishable_9CSocTyHTZkVsZAPXDQwIw_Eo0CsBAM';   // Dashboard → Settings → API Keys → "anon public"

export const ANNOT_MAX_SESSION = 30;    // submissions per session
export const ANNOT_MIN_INTERVAL_MS = 2000;

export const IGN_ORTHO = 'https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0' +
    '&LAYER=ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&FORMAT=image/jpeg' +
    '&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}';

// Plan IGN, drawn semi-transparent over the ortho for the hybrid base layer
export const IGN_PLAN = 'https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0' +
    '&LAYER=GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2&STYLE=normal&FORMAT=image/png' +
    '&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}';

export const PLAN_OVERLAY_OPACITY = 0.45;
