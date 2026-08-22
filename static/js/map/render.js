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
import { S, show, hide, centroid, featureKey, logEvent, pointInGeoJSON,
         fetchDetectionsPoints, fetchDetectionsBBox, fetchDetectionsByAdminPoints } from './store.js';
import { applyFilters, updateFilterBounds } from './filters.js';
import { refreshReportButtonState } from './report.js';

let clusterLayer, singleLayer;
let singleFeatureId = null;   // id of the one installation currently shown as a polygon, or null
let singleAbort = null;

// ─── Init: base tiles, called once at boot. Nothing is rendered on the map
// until a zone is selected — no ambient/national overview layer. ──────────────

export function initMap() {
    const map = S.map;

    map.createPane('detectionsPane').style.zIndex = '400';

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

    // A click on blank map (not a marker, not the currently-shown polygon —
    // both stop propagation) drops back to the cluster view.
    map.on('click', revertToClusters);
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
    scrollToMapSection();
    show('wfs-spinner');
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
        hide('wfs-spinner');
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
    hide('filter-panel');
    hide('filter-toggle-btn');
    hide('export-controls');
}

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
