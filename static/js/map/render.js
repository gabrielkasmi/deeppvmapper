// ─── Map layers: base tiles + selection render (cluster, click-to-reveal polygon) ─
//
// A selection always renders as a marker cluster, built from ONE light fetch
// (centroids only, no geometry) covering the WHOLE selection — this is what
// makes a large selection viable at all. Real polygon geometry is only ever
// fetched for a tiny box around a CLICKED marker, to show that one
// installation's exact shape — clicking blank map reverts to the cluster.
// Filter changes never re-fetch anything — they just re-run applyFilters()
// over the light points already in memory.

import { IGN_ORTHO, MAX_ZOOM, POLYGON_ZOOM,
         POINTS_MAX, SINGLE_FETCH_MAX, SINGLE_FETCH_RADIUS_DEG } from './config.js';
import { S, show, hide, centroid, featureKey, logEvent, pointInGeoJSON, getSupabase,
         fetchDetectionsPoints, fetchDetectionsBBox, fetchDetectionsByAdminPoints,
         fetchDetectionsInZoneWithAnnotations, selectionZoneFeature } from './store.js';
import { applyFilters, updateFilterBounds, resetUnreviewedToggle, setUnreviewedCount } from './filters.js';
import { refreshReportButtonState } from './report.js';
import { showDeptOverview, hideDeptOverview } from './overview.js';

let clusterLayer, singleLayer, unreviewedLayer;
let singleFeatureId = null;   // id of the one installation currently shown as a polygon, or null
let singleAbort = null;
let unreviewedAbort = null;

// Dashed amber for add/modify — visually distinct both from the normal
// kWp-colored polygons (stylePolygon) and from a user's own not-yet-submitted
// local sketch (annotate.js's ADDED_STYLE, solid green). Pending deletions
// (false-positive reports) get their own sparse red dash instead — "flagged
// for removal" reads differently from "flagged as an addition/correction".
const UNREVIEWED_STYLE = { color: '#f59e0b', weight: 2, dashArray: '4 3', fillColor: '#f59e0b', fillOpacity: 0.25 };
const UNREVIEWED_DELETE_STYLE = { color: '#dc2626', weight: 2, dashArray: '2 5', fillColor: '#dc2626', fillOpacity: 0.12 };
function unreviewedStyle(f) {
    return f.properties?.edit_action === 'delete' ? UNREVIEWED_DELETE_STYLE : UNREVIEWED_STYLE;
}

// ─── Init: base tiles, called once at boot. Detection clusters/polygons only
// render once a zone is selected; before that, a small département-level
// overview (overview.js) is shown instead — see loadSelection/
// clearSelectionRender below for where it's hidden/re-shown. ─────────────────

export function initMap() {
    const map = S.map;

    map.createPane('detectionsPane').style.zIndex = '400';
    // Above detectionsPane so the "unreviewed edits" overlay always reads
    // clearly on top of the normal cluster/polygon layers underneath it.
    map.createPane('unreviewedPane').style.zIndex = '410';

    const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: MAX_ZOOM, minZoom: 2
    }).addTo(map);

    const satellite = L.tileLayer(IGN_ORTHO, {
        attribution: '© <a href="https://www.geoportail.gouv.fr">IGN-F/Géoportail</a>',
        maxZoom: MAX_ZOOM, minZoom: 2
    });

    buildLayerToggle(map, osm, satellite);

    clusterLayer = L.markerClusterGroup({
        pane: 'detectionsPane', maxClusterRadius: 60, disableClusteringAtZoom: POLYGON_ZOOM,
        spiderfyOnMaxZoom: false, showCoverageOnHover: false, chunkedLoading: true
    });
    singleLayer = L.geoJSON(null, { pane: 'detectionsPane', style: stylePolygon, onEachFeature: bindSingleEvents });
    unreviewedLayer = L.geoJSON(null, { pane: 'unreviewedPane', style: unreviewedStyle, onEachFeature: bindUnreviewedEvents });

    // A click on blank map (not a marker, not the currently-shown polygon —
    // both stop propagation): inside the current selection, drops back to its
    // cluster view; outside it (or with nothing selected), clears the whole
    // selection instead — same as pressing the eraser — so wandering off the
    // selected zone always lands back on the département overview.
    map.on('click', handleMapClick);
}

function handleMapClick(e) {
    if (isOutsideSelection(e.latlng)) { clearSelectionOutside(); return; }
    revertToClusters();
}

/** Bbox test first (cheap, always applicable), then the exact admin/drawn
 *  boundary when we have one (S.selectionGeometry) — a hand-drawn rectangle
 *  has none because its bbox already IS the exact selection. */
function isOutsideSelection(latlng) {
    if (!S.selectionBounds) return false;
    if (!S.selectionBounds.contains(latlng)) return true;
    if (S.selectionGeometry) {
        const fc = { type: 'FeatureCollection',
            features: [{ type: 'Feature', properties: {}, geometry: S.selectionGeometry }] };
        return !pointInGeoJSON(latlng.lng, latlng.lat, fc);
    }
    return false;
}

/** Dynamic import: selection.js imports loadSelection/clearSelectionRender
 *  from this module — a static import back the other way would be circular.
 *  Deferred to call time, same pattern as overview.js. */
async function clearSelectionOutside() {
    const { clearSelection } = await import('./selection.js');
    clearSelection();
}

/** Plain segmented Map/Satellite toggle, mounted into our own corner stack
 *  (#layer-toggle-mount) rather than added as a Leaflet control — sidesteps
 *  Leaflet's own layer-control CSS entirely, which is what kept rendering as
 *  an oversized/misaligned box in some browsers no matter how it was
 *  restyled. This is just a plain DOM toggle with full CSS control. */
function buildLayerToggle(map, osm, satellite) {
    const mount = document.getElementById('layer-toggle-mount');
    if (!mount) return;
    mount.innerHTML = `
        <div class="layer-toggle">
            <button type="button" class="is-active" data-layer="map">Map</button>
            <button type="button" data-layer="sat">Satellite</button>
        </div>`;
    mount.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.classList.contains('is-active')) return;
            mount.querySelectorAll('button').forEach(b => b.classList.remove('is-active'));
            btn.classList.add('is-active');
            if (btn.dataset.layer === 'sat') {
                if (!map.hasLayer(satellite)) map.addLayer(satellite);
                if (map.hasLayer(osm)) map.removeLayer(osm);
            } else {
                if (!map.hasLayer(osm)) map.addLayer(osm);
                if (map.hasLayer(satellite)) map.removeLayer(satellite);
            }
        });
    });
}

/** Scroll the page to the map, landing at the same spot as the "Browse
 *  detections" header link — called whenever a selection (drawn or
 *  searched) starts loading, so the result is visible without the user
 *  having to scroll down manually. */
function scrollToMapSection() {
    document.getElementById('map-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Selection fetch + render ──────────────────────────────────────────────────

/** Fetch a bbox's light points once (whole selection, any size) and render it.
 *  Called by selection.js on a confirmed draw/search. */
export async function loadSelection(bounds, label) {
    hideDeptOverview();
    scrollToMapSection();
    // Centered, not the small corner #wfs-spinner (that one's for the quick
    // per-marker click fetch) — this one can take a moment, easy to miss in
    // a corner while the camera is also mid-flight (flyToBounds).
    show('center-spinner');
    try {
        let features;
        if (S.selectionAdmin) {
            // Fast path: named place resolved to an admin code — indexed
            // insee/dpt equality match, no spatial test at all.
            features = await fetchDetectionsByAdminPoints(
                S.selectionAdmin.inseeCodes, S.selectionAdmin.deptCodes, POINTS_MAX
            );
        } else {
            features = await fetchDetectionsPoints(
                bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth(), POINTS_MAX
            );
            if (S.selectionGeometry) features = clipToSelectionGeometry(features);
        }
        S.rawFeatures = features;
        S.selectionLabel = label;
        logEvent('browse_selection', `${label} (${features.length} points)`);
        revertToClusters();             // drop any single-feature view left from a previous selection
        clearUnreviewedOverlay();       // new zone — "show unreviewed edits" starts off, re-enable per selection
        updateFilterBounds(features);   // rebind power/year sliders to this selection's range
        drawClusters();
        if (!S.map.hasLayer(clusterLayer)) S.map.addLayer(clusterLayer);
        show('filter-panel');
        show('filter-toggle-btn');
        show('export-controls');
    } catch (e) {
        console.error('Selection fetch error:', e);
        logEvent('detections_error', `selection: ${e.message || e}`);
    } finally {
        hide('center-spinner');
    }
}

/** A named-place selection (search.js) may carry the exact admin boundary —
 *  keep only points that actually fall inside it, not just its bbox. */
function clipToSelectionGeometry(features) {
    if (!S.selectionGeometry) return features;
    const geoFC = { type: 'FeatureCollection', features: [{ type: 'Feature', properties: {}, geometry: S.selectionGeometry }] };
    return features.filter(f => {
        const c = centroid(f.geometry);
        return c && pointInGeoJSON(c[1], c[0], geoFC);
    });
}

/** Clear the current selection — the map goes back to being empty. */
export function clearSelectionRender() {
    S.rawFeatures = [];
    S.selectionLabel = '';
    revertToClusters();
    clusterLayer.clearLayers();
    if (S.map.hasLayer(clusterLayer)) S.map.removeLayer(clusterLayer);
    clearUnreviewedOverlay();
    hide('filter-panel');
    hide('filter-toggle-btn');
    hide('export-controls');
    showDeptOverview();
}

// ─── "Show unreviewed community edits" overlay ─────────────────────────────────
// A distinct dashed-amber layer, separate from the normal cluster/polygon
// rendering above — only ever populated with features the annotations-
// overlay RPC flagged is_unreviewed_edit (pending 'add'/'modify'; a pending
// 'delete' has nothing to draw, the target is just omitted). Toggled from
// the filter panel (filters.js); export.js reads the same S.showUnreviewedEdits
// flag and always re-fetches fresh at download time, so a download never
// depends on this overlay having been refreshed first.

function clearUnreviewedOverlay() {
    if (unreviewedAbort) { unreviewedAbort.abort(); unreviewedAbort = null; }
    resetUnreviewedToggle();
    unreviewedLayer.clearLayers();
    if (S.map.hasLayer(unreviewedLayer)) S.map.removeLayer(unreviewedLayer);
}

/** Re-fetch and redraw the overlay for the current selection — called on
 *  toggle (filters.js) and safe to call any time (no-ops with nothing
 *  selected or the toggle off). */
export async function refreshUnreviewedOverlay() {
    if (unreviewedAbort) { unreviewedAbort.abort(); unreviewedAbort = null; }

    if (!S.showUnreviewedEdits) {
        unreviewedLayer.clearLayers();
        if (S.map.hasLayer(unreviewedLayer)) S.map.removeLayer(unreviewedLayer);
        setUnreviewedCount(null);
        return;
    }

    const zone = selectionZoneFeature();
    if (!zone) return;

    const abort = unreviewedAbort = new AbortController();
    setUnreviewedCount(null);   // "…" isn't worth a dedicated state — blank while loading is fine, brief fetch
    try {
        const features = await fetchDetectionsInZoneWithAnnotations(zone, true);
        if (abort.signal.aborted) return;
        const edits = features.filter(f => f.properties?.is_unreviewed_edit);
        unreviewedLayer.clearLayers();
        if (edits.length) unreviewedLayer.addData({ type: 'FeatureCollection', features: edits });
        if (!S.map.hasLayer(unreviewedLayer)) S.map.addLayer(unreviewedLayer);
        setUnreviewedCount(edits.length);
    } catch (e) {
        if (abort.signal.aborted) return;
        console.error('Unreviewed-overlay fetch error:', e);
        logEvent('detections_error', `unreviewed_overlay: ${e.message || e}`);
        setUnreviewedCount(null);
    } finally {
        if (unreviewedAbort === abort) unreviewedAbort = null;
    }
}

function bindUnreviewedEvents(f, layer) {
    layer.on('mouseover', function () { this.setStyle({ weight: 3, fillOpacity: 0.4 }); });
    layer.on('mouseout',  function () { unreviewedLayer.resetStyle(this); });
    layer.on('click', (e) => {
        L.DomEvent.stopPropagation(e);
        const p = f.properties || {};
        const what = p.edit_action === 'add' ? 'New installation reported by the community'
                   : p.edit_action === 'modify' ? 'Shape correction submitted by the community'
                   : p.edit_action === 'delete' ? 'Reported as not a PV system (false positive)'
                   : 'Unreviewed community edit';
        const kwp = (p.kwp != null && !isNaN(p.kwp)) ? `${parseFloat(p.kwp).toFixed(2)} kWp` : 'not available yet';
        // "Confirm"/"Dispute" are informational only — see annotation_votes
        // in supabase_annotations_overlay_setup.sql: it never changes the
        // annotation's status, just gives the maintainer an extra data point
        // at review time. No accounts on this map, so a public button can't
        // safely resolve moderation itself (see that file's own comment).
        const votes = p.annotation_id && getSupabase() ? `
            <div class="di-vote-actions">
                <button type="button" class="di-action-btn" onclick="window.voteUnreviewed('${p.annotation_id}','confirm',this)">Looks right</button>
                <button type="button" class="di-action-btn" onclick="window.voteUnreviewed('${p.annotation_id}','dispute',this)">Looks wrong</button>
            </div>` : '';
        L.popup({ className: 'di-popup-wrap', maxWidth: 220 })
            .setLatLng(e.latlng)
            .setContent(`<div class="di-popup"><h4>${what}</h4><p style="margin:0;font-size:12px;color:#6c757d;">Not yet checked by a maintainer. Capacity: ${kwp}.</p>${votes}</div>`)
            .openOn(S.map);
    });
}

/** Record an informational "confirm"/"dispute" vote on an unreviewed edit —
 *  wired via inline onclick from the popup above (same pattern as
 *  window.annotDelete/window.annotModify in annotate.js). Never touches the
 *  annotation's own status; purely a signal for review time. */
window.voteUnreviewed = async function (annotationId, vote, btnEl) {
    const sb = getSupabase();
    const group = btnEl?.closest('.di-vote-actions');
    if (!sb || !annotationId) return;
    if (group) group.querySelectorAll('button').forEach(b => { b.disabled = true; });
    try {
        const { error } = await sb.from('annotation_votes').insert({ annotation_id: annotationId, vote });
        if (error) throw error;
        if (group) group.innerHTML = '<span class="di-vote-thanks">Thanks — noted for review.</span>';
    } catch (e) {
        console.error('Vote insert failed:', e);
        if (group) group.querySelectorAll('button').forEach(b => { b.disabled = false; });
    }
};

/** Re-render with the current filters (filters.js on any filter change). */
export function renderFiltered() {
    drawClusters();
    // If the installation currently shown as a polygon no longer passes the
    // filters (e.g. just got marked a false positive), drop back to clusters.
    if (singleFeatureId != null && !applyFilters(S.rawFeatures).some(f => f.id === singleFeatureId)) {
        revertToClusters();
    }
}

function drawClusters() {
    const feats = applySessionEdits(applyFilters(S.rawFeatures));
    clusterLayer.clearLayers();
    const markers = feats.map(f => {
        const c = centroid(f.geometry);
        if (!c) return null;
        const m = L.marker([c[0], c[1]]);
        m.on('click', (e) => {
            L.DomEvent.stopPropagation(e);
            selectSingleFeature(f.id, L.latLng(c[0], c[1]));
        });
        return m;
    }).filter(Boolean);
    clusterLayer.addLayers(markers);
}

/** Fetch one installation's real polygon (tiny box around the click) and show
 *  it in place of the cluster, centering the view on it at max zoom. */
async function selectSingleFeature(id, latlng) {
    S.map.flyTo(latlng, MAX_ZOOM, { duration: 0.6 });

    if (singleAbort) singleAbort.abort();
    const abort = singleAbort = new AbortController();
    show('wfs-spinner');
    try {
        const feats = await fetchDetectionsBBox(
            latlng.lng - SINGLE_FETCH_RADIUS_DEG, latlng.lat - SINGLE_FETCH_RADIUS_DEG,
            latlng.lng + SINGLE_FETCH_RADIUS_DEG, latlng.lat + SINGLE_FETCH_RADIUS_DEG,
            SINGLE_FETCH_MAX, abort.signal
        );
        if (abort.signal.aborted) return;
        const feature = feats.find(f => f.id === id) || feats[0];
        if (!feature) return;
        showSingleFeature(feature, latlng);
    } catch (e) {
        if (!abort.signal.aborted) {
            console.error('Single feature fetch error:', e);
            logEvent('detections_error', `single: ${e.message || e}`);
        }
    } finally {
        if (singleAbort === abort) { singleAbort = null; hide('wfs-spinner'); }
    }
}

function showSingleFeature(feature, latlng) {
    singleFeatureId = feature.id;
    const [edited] = applySessionEdits([feature]);
    singleLayer.clearLayers();
    if (edited) singleLayer.addData({ type: 'FeatureCollection', features: [edited] });
    if (!S.map.hasLayer(singleLayer))  S.map.addLayer(singleLayer);
    if (S.map.hasLayer(clusterLayer))  S.map.removeLayer(clusterLayer);
    S.lastClickedDetection = { feature, layer: singleLayer, latlng };
    showDetectionInfo(feature, latlng);
    refreshReportButtonState();
}

/** Back to the marker cluster — a blank-map click, a fresh selection landing,
 *  or the shown installation no longer matching the current filters. */
function revertToClusters() {
    if (singleAbort) { singleAbort.abort(); singleAbort = null; }
    singleFeatureId = null;
    singleLayer.clearLayers();
    if (S.map.hasLayer(singleLayer)) S.map.removeLayer(singleLayer);
    if (S.rawFeatures.length && !S.map.hasLayer(clusterLayer)) S.map.addLayer(clusterLayer);
    hideDetectionInfo();
    S.lastClickedDetection = null;   // no installation shown anymore — report.js reads this
    refreshReportButtonState();
}

/** Re-render detections after an annotation edit (annotate.js) — same data,
 *  session edits (delete/modify) just get re-applied on top, no re-fetch. */
export function rerenderDetections() {
    drawClusters();
}

// ─── Session edits (annotations) ──────────────────────────────────────────────
// User edits live in S.edits for the session and are applied at render time:
// reported deletions disappear, modified geometries replace the originals.

function applySessionEdits(features) {
    const { deleted, modified } = S.edits;
    if (!deleted.size && !modified.size) return features;
    return features
        .filter(f => !deleted.has(featureKey(f)))
        .map(f => {
            const geom = modified.get(featureKey(f));
            return geom ? { ...f, geometry: geom } : f;
        });
}

// ─── Polygon styling / popup ───────────────────────────────────────────────────

function kwpColor(kwp) {
    const k = parseFloat(kwp) || 0;
    if (k <= 3)  return '#c9b3e8';
    if (k <= 6)  return '#a07cd4';
    if (k <= 10) return '#7248b8';
    if (k <= 20) return '#4e1d96';
    return '#2c0870';
}

function stylePolygon(f) {
    return { fillColor: kwpColor(f.properties.kwp), fillOpacity: 0.65, color: '#333', weight: 0.8, opacity: 0.9 };
}

function bindSingleEvents(f, layer) {
    layer.on('mouseover', function () { this.setStyle({ weight: 2, fillOpacity: 0.85 }); });
    layer.on('mouseout',  function () { singleLayer.resetStyle(this); });
    layer.on('click', (e) => {
        L.DomEvent.stopPropagation(e);   // stay open — don't let this bubble to the map's revert handler
        if (S.drawing) return;
        S.lastClickedDetection = { feature: f, layer, latlng: e.latlng };
        showDetectionInfo(f, e.latlng);
        refreshReportButtonState();
    });
}

let detectionPopup = null;   // the one open "baseline" popup, if any

/** Installation detail: a real Leaflet popup anchored right at the clicked
 *  installation — a "baseline" popup, the same way any map popup sits next
 *  to the thing it describes, rather than a fixed widget elsewhere on the
 *  map. Closing it (✕, or clicking blank map) reverts to clusters. */
function showDetectionInfo(f, latlng) {
    const p = f.properties;
    const fmt = (v, d) => (v != null && !isNaN(v)) ? parseFloat(v).toFixed(d) : '—';
    const content = `
        <div class="di-popup">
            <h4>Rooftop PV System</h4>
            <table>
                <tr><td>Surface</td><td><strong>${fmt(p.surface, 1)} m²</strong></td></tr>
                <tr><td>Capacity</td><td><strong>${fmt(p.kwp, 2)} kWp</strong></td></tr>
                <tr><td>First seen</td><td><strong>${p.first_seen || '—'}</strong></td></tr>
            </table>
            <div class="di-actions">
                <button type="button" class="di-action-btn" onclick="window.annotDelete && window.annotDelete()">False positive</button>
                <button type="button" class="di-action-btn" onclick="window.annotModify && window.annotModify()">Redraw</button>
            </div>
        </div>`;
    if (detectionPopup) S.map.closePopup(detectionPopup);
    detectionPopup = L.popup({ className: 'di-popup-wrap', maxWidth: 240, autoClose: false, closeOnClick: false })
        .setLatLng(latlng)
        .setContent(content)
        .openOn(S.map);
    // "False positive" / "Redraw" call into annotate.js (window.annotDelete /
    // window.annotModify), which reads S.lastClickedDetection — already set
    // by the caller (selectSingleFeature / bindSingleEvents) before this runs.
}

function hideDetectionInfo() {
    if (detectionPopup) { S.map.closePopup(detectionPopup); detectionPopup = null; }
}
