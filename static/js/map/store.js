// ─── Shared state + small utilities ──────────────────────────────────────────

import { SUPABASE_URL, SUPABASE_ANON_KEY, DETECTIONS_RPC, ZONE_RPC, ZONE_FETCH_MAX } from './config.js';

/** Fire-and-forget usage event (plain REST: no client lib, no failure surface). */
export function logEvent(event, detail) {
    if (!SUPABASE_URL || !SUPABASE_ANON_KEY) return;
    fetch(`${SUPABASE_URL}/rest/v1/events`, {
        method: 'POST',
        headers: { apikey: SUPABASE_ANON_KEY, Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
                   'Content-Type': 'application/json', Prefer: 'return=minimal' },
        body: JSON.stringify({ event, detail: detail?.slice(0, 200) || null })
    }).catch(() => {});
}

// ─── Supabase client (single shared instance) ─────────────────────────────────

let sbClient = null;

/** Lazily-created, shared across annotate.js / layers.js / stats.js. */
export function getSupabase() {
    if (!sbClient && SUPABASE_URL && SUPABASE_ANON_KEY && window.supabase)
        sbClient = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
    return sbClient;
}

/**
 * Detection polygons in a bbox — replaces the old WFS BBOX GetFeature call.
 * Returns a plain array of GeoJSON Features (id, geometry, properties), same
 * shape callers used to get from `data.features`. Pass an AbortSignal and
 * check `signal.aborted` on return (same pattern as the old fetch abort).
 */
export async function fetchDetectionsBBox(west, south, east, north, limit, signal) {
    const sb = getSupabase();
    if (!sb) return [];
    let q = sb.rpc(DETECTIONS_RPC, {
        min_lon: west, min_lat: south, max_lon: east, max_lat: north, max_count: limit
    });
    if (signal) q = q.abortSignal(signal);
    const { data, error } = await q;
    if (error) throw error;
    return data || [];
}

/**
 * All detections in a zone (exact polygon containment, no bbox proxy, no low
 * cap) — CSV export only. Charts/KPIs stay on the cheap bbox sample.
 */
export async function fetchDetectionsInZone(feature, limit = ZONE_FETCH_MAX) {
    const sb = getSupabase();
    if (!sb || !feature?.geometry) return [];
    const { data, error } = await sb.rpc(ZONE_RPC, {
        zone_geometry: feature.geometry, max_count: limit
    });
    if (error) throw error;
    return data || [];
}

export const S = {
    map: null,

    // Navigation
    mode: 'explore',            // 'explore' (hover/wander) | 'target' (locked on a zone)
    target: null,               // {level, code, nom, feature} when mode === 'target'
    level: 'france',            // 'france' | 'region' | 'dept' | 'commune'
    path: [],                   // [{level, code, nom, feature}, ...]

    // Baked statistics (built by scripts/build_geo_stats.py)
    statsRegions: {},           // {code: [n, kwp]}
    statsDepts: {},             // {code: [n, kwp]}
    statsCommunes: {},          // {insee: [n, kwp, lat, lng]}
    regionDepts: {},            // {regionCode: [deptCodes]}

    // Geometry
    regionsGeo: null,
    deptsGeo: null,
    communeCache: new Map(),    // deptCode -> FeatureCollection

    // Heatmap source (derived from statsCommunes)
    heatPoints: [],             // [{lat, lng, n, dept, insee}]

    // Current selection (drives mask + PIP filtering of WFS samples)
    boundaryGeoJSON: null,
    areaLabel: '',
    sampleFeatures: [],         // last WFS sample (charts + CSV export)

    // Annotation session state
    edits: {
        deleted: new Set(),     // featureKeys reported as false positives
        modified: new Map(),    // featureKey -> new GeoJSON geometry
        added: [],              // GeoJSON features drawn by the user
    },
    drawing: false,             // true while geoman draw/edit is active
    lastClickedDetection: null, // {feature, layer, latlng} of the clicked polygon
};

/** Stable identity for a WFS detection feature. */
export function featureKey(f) {
    if (f.id != null) return String(f.id);
    const c = centroid(f.geometry);
    return c ? `${c[0].toFixed(6)},${c[1].toFixed(6)}` : 'unknown';
}

export const $ = id => document.getElementById(id);
export const show = id => { const el = $(id); if (el) el.style.display = 'block'; };
export const hide = id => { const el = $(id); if (el) el.style.display = 'none'; };

export const fmtInt = v => (v ?? 0).toLocaleString('fr-FR');

// ─── Geometry helpers ─────────────────────────────────────────────────────────

export function centroid(geometry) {
    if (!geometry) return null;
    const coords = geometry.type === 'MultiPolygon'
        ? geometry.coordinates[0][0]
        : geometry.coordinates[0];
    if (!coords?.length) return null;
    const lat = coords.reduce((s, c) => s + c[1], 0) / coords.length;
    const lng = coords.reduce((s, c) => s + c[0], 0) / coords.length;
    return [lat, lng];
}

function pointInRing(lng, lat, ring) {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const [xi, yi] = ring[i], [xj, yj] = ring[j];
        if ((yi > lat) !== (yj > lat) && lng < (xj - xi) * (lat - yi) / (yj - yi) + xi)
            inside = !inside;
    }
    return inside;
}

export function pointInGeoJSON(lng, lat, geoJSON) {
    if (!geoJSON?.features?.length) return true;   // no boundary → pass through
    return geoJSON.features.some(f => {
        const geom = f.geometry;
        if (!geom) return false;
        const polys = geom.type === 'Polygon'      ? [geom.coordinates]
                    : geom.type === 'MultiPolygon' ? geom.coordinates : [];
        return polys.some(([exterior]) => pointInRing(lng, lat, exterior));
    });
}

/** WFS sample features filtered to the currently selected boundary. */
export function filteredSample() {
    if (!S.boundaryGeoJSON) return S.sampleFeatures;
    return S.sampleFeatures.filter(f => {
        const c = centroid(f.geometry);
        return c && pointInGeoJSON(c[1], c[0], S.boundaryGeoJSON);
    });
}
