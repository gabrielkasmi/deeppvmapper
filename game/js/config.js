// ─── PV Check — configuration ──────────────────────────────────────────────
// Same Supabase project as the map (static/js/map/config.js) — new tables,
// same backend.

export const SUPABASE_URL      = 'https://zelhliylrlktnasircwp.supabase.co';
export const SUPABASE_ANON_KEY = 'sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi';

export const CAMPAIGN_ID = 'season-1';

export const BATCH_SIZE    = 12;   // cards fetched per RPC call
export const PREFETCH_AT   = 5;    // fetch the next batch when this many cards remain

// Géoplateforme IGN — WMS GetMap (arbitrary bbox + pixel size, exact crop,
// unlike the WMTS tiles used for the map's satellite layer — see
// game/README.md "Format des images" for why WMS is the right tool here).
// NOT validated live from this session (sandboxed network, no route out to
// data.geopf.fr) — first real test happens the first time this page is
// opened in a real browser. If the image comes back broken, check this URL/
// param set against the current Géoplateforme docs before anything else.
export const IGN_WMS_URL   = 'https://data.geopf.fr/wms-r/wms';
export const IGN_WMS_LAYER = 'ORTHOIMAGERY.ORTHOPHOTOS';

export const CARD_PX = 400; // output image size, both dimensions
