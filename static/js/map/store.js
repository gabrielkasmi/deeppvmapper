// ─── Shared state + small utilities ──────────────────────────────────────────

import { SUPABASE_URL, SUPABASE_ANON_KEY, POINTS_RPC, DETECTIONS_RPC, ZONE_RPC, ZONE_FETCH_MAX,
         ADMIN_POINTS_RPC, ADMIN_ZONE_RPC, ANNOTATIONS_ZONE_RPC, UNREVIEWED_COUNT_RPC,
         DEPT_STATS_RPC } from './config.js';

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
 * Light detections in a bbox: centroid + slim properties, NO geometry — this
 * is what makes a région-sized selection viable (a full polygon fetch of that
 * many rows is what used to hang). Returns pseudo-GeoJSON Point Features so
 * the rest of the app (centroid(), applyFilters(), export) doesn't need to
 * know the difference. Pass an AbortSignal and check `signal.aborted` on return.
 */
export async function fetchDetectionsPoints(west, south, east, north, limit, signal) {
    const sb = getSupabase();
    if (!sb) return [];
    let q = sb.rpc(POINTS_RPC, {
        min_lon: west, min_lat: south, max_lon: east, max_lat: north, max_count: limit
    });
    if (signal) q = q.abortSignal(signal);
    const { data, error } = await q;
    if (error) throw error;
    return (data || []).map(row => ({
        id: row.id,
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [row.lng, row.lat] },
        properties: row,
    }));
}

/**
 * Real detection polygons in a bbox — used only for the tiny box fetched
 * around a clicked marker, to pull that one installation's exact shape.
 * Returns a plain array of GeoJSON Features (id, geometry, properties).
 * Pass an AbortSignal and check `signal.aborted` on return.
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
 * cap), WITH full geometry — GeoJSON export only. `feature` is a GeoJSON
 * Feature whose geometry is the zone to query (a rectangle built from the
 * current selection's bounds works fine — ST_Contains against a rectangle is
 * just a bbox test with extra rigor).
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

/**
 * Same zone as fetchDetectionsInZone, but with public.annotations layered on
 * top of public.detections (merged edits always applied; pending ones only
 * when includeUnreviewed is true; rejected ones never). A 'delete' removes
 * the target, a 'modify' replaces its geometry/properties in place (no
 * duplicate between the base layer and the edit), an 'add' contributes a
 * brand-new feature. Each returned feature carries `is_unreviewed_edit` so
 * callers (export.js) can flag which rows came from an edit rather than the
 * official dataset. Community-submitted geometry has no guarantee of
 * surface/kwp/year — DeepPVMapper doesn't compute those for community-
 * submitted polygons today.
 */
export async function fetchDetectionsInZoneWithAnnotations(feature, includeUnreviewed = false, limit = ZONE_FETCH_MAX) {
    const sb = getSupabase();
    if (!sb || !feature?.geometry) return [];
    const { data, error } = await sb.rpc(ANNOTATIONS_ZONE_RPC, {
        zone_geometry: feature.geometry, include_unreviewed: includeUnreviewed, max_count: limit
    });
    if (error) throw error;
    return data || [];
}

/** Count of pending (not yet reviewed) annotations touching a zone — used to
 *  label the "include unreviewed edits" export option instead of leaving it
 *  a mystery checkbox when there's nothing to include. */
export async function countUnreviewedAnnotationsInZone(feature) {
    const sb = getSupabase();
    if (!sb || !feature?.geometry) return 0;
    const { data, error } = await sb.rpc(UNREVIEWED_COUNT_RPC, { zone_geometry: feature.geometry });
    if (error) throw error;
    return data ?? 0;
}

/**
 * Light points by admin code (région/département/commune) — the fast path for
 * a named-place selection (see geo.js): an indexed insee/dpt equality match,
 * no spatial test at all. Pass insee_codes for a commune, or dept_codes for a
 * département (one code) or région (its member département codes).
 */
export async function fetchDetectionsByAdminPoints(inseeCodes, deptCodes, limit, signal) {
    const sb = getSupabase();
    if (!sb) return [];
    let q = sb.rpc(ADMIN_POINTS_RPC, {
        insee_codes: inseeCodes || null, dept_codes: deptCodes || null, max_count: limit
    });
    if (signal) q = q.abortSignal(signal);
    const { data, error } = await q;
    if (error) throw error;
    return (data || []).map(row => ({
        id: row.id,
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [row.lng, row.lat] },
        properties: row,
    }));
}

/** Same admin-code fast path, WITH full geometry, uncapped — GeoJSON export
 *  for a named-place selection. */
export async function fetchDetectionsByAdmin(inseeCodes, deptCodes, limit = ZONE_FETCH_MAX) {
    const sb = getSupabase();
    if (!sb) return [];
    const { data, error } = await sb.rpc(ADMIN_ZONE_RPC, {
        insee_codes: inseeCodes || null, dept_codes: deptCodes || null, max_count: limit
    });
    if (error) throw error;
    return data || [];
}

/**
 * Département-level counts (n_systems per dept) — the national overview
 * shown before any zone is selected (overview.js). Same pre-aggregated RPC
 * the per-département static stats pages already use (dept_stats_rpcs.sql):
 * ~96 rows, cheap enough to fetch on every load of this page.
 */
export async function fetchDeptStats() {
    const sb = getSupabase();
    if (!sb) return [];
    const { data, error } = await sb.rpc(DEPT_STATS_RPC);
    if (error) throw error;
    return data || [];
}

export const S = {
    map: null,

    // Départements outline, lazy-loaded only to resolve a ?dept=CODE deep link
    // into a bbox (see selection.js) — no other admin-boundary geometry is
    // loaded anymore.
    deptsGeo: null,

    // Current selection (selection.js / search.js)
    selectionBounds: null,      // Leaflet LatLngBounds, or null before any selection
    selectionGeometry: null,    // exact admin-boundary GeoJSON geometry for a named
                                 // place (search.js, via geo.js) — null for a hand-drawn
                                 // rectangle, where the bbox IS the exact selection
    selectionAdmin: null,       // { inseeCodes } | { deptCodes } | null — when set, fetches
                                 // use the fast indexed insee/dpt match (store.js) instead of
                                 // the spatial bbox/intersection path
    selectionLabel: '',         // human label for export filenames / events
    rawFeatures: [],            // last fetch for the current selection, unfiltered
    filters: {
        kwpMin: null, kwpMax: null,
        yearMin: null, yearMax: null,
        sources: null,           // null = all; else a Set of source indices (see SOURCE_LABELS)
        hideFalsePositive: true,
    },

    // Annotation session state (annotate.js — edit UX itself is a separate pass)
    edits: {
        deleted: new Set(),     // featureKeys reported as false positives
        modified: new Map(),    // featureKey -> new GeoJSON geometry
        added: [],              // GeoJSON features drawn by the user
    },
    drawing: false,             // true while geoman draw/edit is active
    lastClickedDetection: null, // {feature, layer, latlng} of the clicked polygon

    // "Show unreviewed community edits" (filters.js checkbox, render.js
    // overlay, export.js download path) — off by default and reset on every
    // new selection (see render.js's loadSelection/clearSelectionRender).
    showUnreviewedEdits: false,
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
    if (geometry.type === 'Point') return [geometry.coordinates[1], geometry.coordinates[0]];
    const coords = geometry.type === 'MultiPolygon'
        ? geometry.coordinates[0][0]
        : geometry.coordinates[0];
    if (!coords?.length) return null;
    const lat = coords.reduce((s, c) => s + c[1], 0) / coords.length;
    const lng = coords.reduce((s, c) => s + c[0], 0) / coords.length;
    return [lat, lng];
}

/** A Leaflet LatLngBounds as a GeoJSON rectangle Feature — for get_detections_in_zone. */
export function boundsToRectFeature(bounds) {
    const w = bounds.getWest(), s = bounds.getSouth(), e = bounds.getEast(), n = bounds.getNorth();
    return { type: 'Feature', properties: {}, geometry: { type: 'Polygon',
        coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] } };
}

/** The current selection as a GeoJSON Feature — exact admin/drawn boundary if
 *  we have one, else the selection's bbox rectangle. Shared by export.js
 *  (download) and render.js (the "show unreviewed edits" overlay), which
 *  both need to hand a zone_geometry to the annotations-overlay RPCs — those
 *  RPCs have no admin-code fast path, so this is always the spatial one. */
export function selectionZoneFeature() {
    if (!S.selectionBounds) return null;
    return S.selectionGeometry
        ? { type: 'Feature', properties: {}, geometry: S.selectionGeometry }
        : boundsToRectFeature(S.selectionBounds);
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

