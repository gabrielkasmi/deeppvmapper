// ─── Zone selection: draw a bbox by hand, or from a search result ────────────
//
// A hand-drawn selection is a plain bbox (Leaflet LatLngBounds, editable
// rectangle via geoman) — the rectangle IS the exact selection there. A
// named-place selection (search.js) instead carries the real admin boundary
// (région/département/commune contour, resolved locally by geo.js) whenever
// one is found, drawn as that exact polygon instead of a rectangle; the
// ?dept=CODE deep link (used by the static per-département Data pages) always
// has one, straight from departements.geojson.

import { S, $, show, hide, logEvent } from './store.js';
import { loadSelection, clearSelectionRender } from './render.js';
import { resolveDepartementByCode } from './geo.js';

let rectLayer = null;

export function initSelection() {
    $('draw-zone-btn')?.addEventListener('click', startDraw);
    $('clear-zone-btn')?.addEventListener('click', clearSelection);
    $('cancel-zone-btn')?.addEventListener('click', cancelDraw);

    resolveDeepLink();
}

// ─── Drawing a rectangle by hand ───────────────────────────────────────────────

function startDraw() {
    if (!S.map.pm) return;
    if (S.drawing) return;
    clearRect();
    S.drawing = true;
    hide('draw-zone-btn');
    show('zone-editbar');   // just a "Cancel" while the rectangle is being drawn
    S.map.pm.enableDraw('Rectangle', { snappable: false });
    S.map.once('pm:create', onRectDrawn);
}

/** Drawing the rectangle IS the confirmation — search fires the moment the
 *  shape is finished, no separate "Search this area" step. */
function onRectDrawn(e) {
    S.map.pm.disableDraw();
    rectLayer = e.layer;
    const bounds = rectLayer.getBounds();
    S.drawing = false;
    hide('zone-editbar');
    show('clear-zone-btn');
    setSelectionFromBounds(bounds, 'custom area');
}

function cancelDraw() {
    clearRect();
    S.map.pm.disableDraw();
    S.drawing = false;
    hide('zone-editbar');
    show('draw-zone-btn');
}

function clearRect() {
    if (rectLayer) { S.map.removeLayer(rectLayer); rectLayer = null; }
}

// ─── Public API ────────────────────────────────────────────────────────────────

/** Draw a bbox rectangle (hand-drawn, or a fallback when no exact boundary
 *  was found for a search result) and load it. */
export function setSelectionFromBounds(bounds, label) {
    clearRect();
    rectLayer = L.rectangle(bounds, {
        color: '#5bc8f5', weight: 2, dashArray: '6 4', fillColor: '#5bc8f5', fillOpacity: 0.05
    }).addTo(S.map);
    show('draw-zone-btn');
    show('clear-zone-btn');
    S.map.flyToBounds(bounds, { padding: [30, 30], duration: 1.0 });
    logEvent('browse_zone', label);
    S.selectionBounds = bounds;
    S.selectionGeometry = null;   // a rectangle IS the exact selection — nothing to clip against
    S.selectionAdmin = null;      // no admin code for an arbitrary rectangle — spatial path only
    loadSelection(bounds, label);
}

/** Draw the exact admin-boundary polygon for a named place (search.js /
 *  ?dept=CODE) and load it. `result` is { feature, admin } (geo.js): feature
 *  draws the real contour and backstops a client-side intersection filter;
 *  admin (insee/dept codes), when non-empty, lets render.js/export.js skip
 *  the spatial query entirely via the fast indexed match — the geometry
 *  intersection stays the fallback for whenever admin resolves to nothing. */
export function setSelectionFromFeature({ feature, admin }, label) {
    clearRect();
    rectLayer = L.geoJSON(feature, {
        style: { color: '#5bc8f5', weight: 2, dashArray: '6 4', fillColor: '#5bc8f5', fillOpacity: 0.05 }
    }).addTo(S.map);
    const bounds = rectLayer.getBounds();
    show('draw-zone-btn');
    show('clear-zone-btn');
    S.map.flyToBounds(bounds, { padding: [30, 30], duration: 1.0 });
    logEvent('browse_zone', label);
    S.selectionBounds = bounds;
    S.selectionGeometry = feature.geometry;
    const hasCodes = admin && ((admin.inseeCodes?.length) || (admin.deptCodes?.length));
    S.selectionAdmin = hasCodes ? admin : null;
    loadSelection(bounds, label);
}

export function clearSelection() {
    clearRect();
    hide('clear-zone-btn');
    S.selectionBounds = null;
    S.selectionGeometry = null;
    S.selectionAdmin = null;
    clearSelectionRender();
}

/** Called by annotate.js while an annotation is being drawn — the zone-draw
 *  button must not compete for map clicks with the polygon draw tool. */
export function suspendSelection() {
    hide('draw-zone-btn'); hide('clear-zone-btn');
}

export function resumeSelection() {
    show('draw-zone-btn');
    if (rectLayer) show('clear-zone-btn');
}

// ─── Deep link: ?dept=CODE (used by the static per-département Data pages) ───

async function resolveDeepLink() {
    const code = new URLSearchParams(location.search).get('dept');
    if (!code) return;
    try {
        const result = await resolveDepartementByCode(code);
        if (!result) return;
        setSelectionFromFeature(result, result.feature.properties.nom || `dept ${code}`);
    } catch (e) {
        console.error('Dept deep-link failed:', e);
    }
}
