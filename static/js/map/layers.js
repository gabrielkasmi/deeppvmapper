// ─── Map layers: base tiles, WFS detections, heatmap regimes, mask ───────────
//
// Visual language: the heatmap is the only density visualisation.
//   z <  BANDS.commune        precomputed heat (commune centroids, hotspot-truncated)
//   z in [commune, HEAT_FADE) live heat — full commune set when one is locked,
//                             WFS viewport sample otherwise
//   z >= HEAT_FADE            heat off — individual detection polygons take over
// Detection polygons are fetched from BANDS.commune up (commune-anchored when
// a commune is locked). Nav polygons (nav.js) live below detections.

import { WFS_URL, WFS_LAYER, WFS_GEOM, BANDS, HEAT_FADE, IGN_ORTHO, IGN_PLAN,
         PLAN_OVERLAY_OPACITY, HEAT_TOPSHARE } from './config.js';
import { S, show, hide, centroid, pointInGeoJSON, featureKey, logEvent } from './store.js';

let polyLayer, heatLayer, maskLayer, boundaryLayer;
let wfsDebounce, wfsAbort = null;
let lastViewportFeatures = [];
let currentRegime = null;            // 'precomputed' | 'live' | 'detail'
let communeCode = null, communeAbort = null;   // commune-anchored detection fetch

// ─── Init ─────────────────────────────────────────────────────────────────────

export function initLayers() {
    const map = S.map;

    // Pane z-order: nav(350) < detections(400, default) < mask(420) < heat(450) < boundary(460)
    map.createPane('navPane').style.zIndex   = '350';
    map.createPane('maskPane').style.zIndex  = '420';
    const hp = map.createPane('heatPane');
    hp.style.zIndex = '450';
    hp.style.pointerEvents = 'none';           // heat must never block clicks
    map.createPane('boundPane').style.zIndex  = '460';
    map.getPane('maskPane').style.pointerEvents = 'none';
    map.getPane('boundPane').style.pointerEvents = 'none';

    const ignAttrib = '© <a href="https://www.geoportail.gouv.fr">IGN-F/Géoportail</a>';

    // Default base: ortho with the Plan IGN drawn semi-transparent on top
    // (roads + labels stay readable over the imagery). Alternative: ortho only.
    const hybrid = L.layerGroup([
        L.tileLayer(IGN_ORTHO, { attribution: ignAttrib, maxZoom: 20, minZoom: 2 }),
        L.tileLayer(IGN_PLAN,  { maxZoom: 19, minZoom: 2, opacity: PLAN_OVERLAY_OPACITY }),
    ]).addTo(map);

    const orthoOnly = L.tileLayer(IGN_ORTHO, {
        attribution: ignAttrib, maxZoom: 20, minZoom: 2
    });

    L.control.layers(
        { 'Orthophoto + map': hybrid, 'Orthophoto only': orthoOnly },
        {},
        { position: 'topright', collapsed: true }
    ).addTo(map);

    polyLayer = L.geoJSON(null, { style: stylePolygon, onEachFeature: bindPolyEvents }).addTo(map);

    boundaryLayer = L.geoJSON(null, {
        pane: 'boundPane',
        interactive: false,
        style: { color: '#5bc8f5', weight: 2.5, opacity: 0.9, dashArray: '7 4',
                 fillColor: '#5bc8f5', fillOpacity: 0.04 }
    }).addTo(map);

    map.on('zoomend moveend', onViewChange);
    onViewChange();
}

// ─── Zoom regime ──────────────────────────────────────────────────────────────

function onViewChange() {
    const z = S.map.getZoom();
    const regime = z >= HEAT_FADE ? 'off' : z >= BANDS.commune ? 'live' : 'precomputed';
    const changed = regime !== currentRegime;
    currentRegime = regime;

    // Capacity legend once detection polygons are readable, heat legend before
    if (z >= BANDS.detail) { hide('legend-heat'); show('legend'); }
    else { hide('legend'); show('legend-heat'); }

    if (regime === 'off') {
        if (changed) removeHeat();
    } else if (regime === 'live') {
        // Commune-anchored data is already complete; otherwise the viewport
        // fetch below renders the heat on arrival.
        if (changed && communeCode) renderLiveHeat(applySessionEdits(lastViewportFeatures));
    } else if (changed) {
        if (!communeCode) { polyLayer.clearLayers(); lastViewportFeatures = []; }
        renderPrecomputedHeat();                // heat points are latlng — pan needs no re-render
    }

    // Detection polygons: viewport streaming unless a commune anchors the data
    if (z >= BANDS.commune && !communeCode) scheduleWfs();
}

/** Re-render the heat for the current regime (called by nav on zone/mode change). */
export function refreshHeat() {
    const z = S.map.getZoom();
    if (z >= HEAT_FADE) removeHeat();
    else if (z >= BANDS.commune) renderLiveHeat(lastViewportFeatures);
    else renderPrecomputedHeat();
}

// ─── Precomputed heat (commune centroids) ─────────────────────────────────────

function activeHeatPoints() {
    // Target mode: heat is scoped (and masked) to the locked zone.
    // Explore mode: unscoped — the viewport does the cropping.
    if (S.mode === 'target' && S.target) {
        const t = S.target;
        if (t.level === 'region') {
            const depts = new Set(S.regionDepts[t.code] || []);
            return S.heatPoints.filter(p => depts.has(p.dept));
        }
        const prefix = t.level === 'dept' ? t.code : t.code.slice(0, 2);
        return S.heatPoints.filter(p => p.dept === prefix);
    }
    return S.heatPoints;
}

function currentTopShare() {
    if (S.mode === 'target' && S.target)
        return S.target.level === 'region' ? HEAT_TOPSHARE.region : HEAT_TOPSHARE.dept;
    const z = S.map.getZoom();
    if (z < BANDS.dept)    return HEAT_TOPSHARE.france;
    if (z < BANDS.commune) return HEAT_TOPSHARE.region;
    return HEAT_TOPSHARE.dept;
}

function renderPrecomputedHeat() {
    let pts = activeHeatPoints();
    if (!pts.length) { removeHeat(); return; }

    // Truncate to the hotspots: top `topShare` of the displayed subset by
    // system count. National view is aggressive; lower levels show more.
    const topShare = currentTopShare();
    if (topShare < 1 && pts.length > 20) {
        const sorted = pts.map(p => p.n).sort((a, b) => b - a);
        const cutoff = sorted[Math.min(sorted.length - 1, Math.floor(topShare * sorted.length))];
        pts = pts.filter(p => p.n >= cutoff);
    }

    const norm = Math.log1p(Math.max(...pts.map(p => p.n)));
    setHeat(
        pts.map(p => [p.lat, p.lng, Math.log1p(p.n) / norm]),
        { radius: 16, blur: 14, minOpacity: 0.25, max: 1.0 }
    );
}

// ─── Live heat (WFS viewport sample) ──────────────────────────────────────────

function renderLiveHeat(features) {
    const pts = features
        .map(f => { const c = centroid(f.geometry); return c ? [c[0], c[1], 1] : null; })
        .filter(Boolean);
    if (!pts.length) { removeHeat(); return; }
    const z = S.map.getZoom();
    const radius = z >= 12 ? 14 : 18;
    setHeat(pts, {
        radius, blur: Math.round(radius * 0.75),
        minOpacity: 0.3, max: z >= 12 ? 1.0 : 0.7
    });
    // (live regime: z in [BANDS.commune, BANDS.detail))
}

function setHeat(pts, opts) {
    removeHeat();
    heatLayer = L.heatLayer(pts, {
        pane: 'heatPane',
        gradient: { 0.25: '#fed976', 0.50: '#fd8d3c', 0.70: '#e31a1c', 0.88: '#a50f15', 1.0: '#67000d' },
        ...opts
    }).addTo(S.map);
    if (heatLayer._canvas) heatLayer._canvas.style.pointerEvents = 'none';
}

function removeHeat() {
    if (heatLayer) { S.map.removeLayer(heatLayer); heatLayer = null; }
}

// ─── Session edits (annotations) ──────────────────────────────────────────────
//
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

/** Re-render detection polygons after an edit (delete/modify). */
export function rerenderDetections() {
    const feats = applySessionEdits(lastViewportFeatures);
    polyLayer.clearLayers();
    if (feats.length) polyLayer.addData({ type: 'FeatureCollection', features: feats });
}

// ─── Commune-anchored detection fetch ─────────────────────────────────────────
//
// At commune scale, viewport streaming caps at N features in storage order —
// in dense cities the sample lands in one corner and the rest shows nothing.
// Instead we anchor on the active commune: one large fetch over its bbox
// (the densest commune holds ~1 400 systems), PIP-filtered to the polygon.

const COMMUNE_COUNT = 6000;

export async function setActiveCommune(feature) {
    if (!feature) {
        communeCode = null;
        if (communeAbort) { communeAbort.abort(); communeAbort = null; }
        return;
    }
    const code = feature.properties.code;
    if (code === communeCode) return;
    communeCode = code;
    if (communeAbort) communeAbort.abort();
    if (wfsAbort) { wfsAbort.abort(); wfsAbort = null; }   // viewport fetch is obsolete
    const abort = communeAbort = new AbortController();
    show('wfs-spinner');

    const b = L.geoJSON(feature).getBounds();
    const bbox = `${b.getWest().toFixed(5)},${b.getSouth().toFixed(5)},${b.getEast().toFixed(5)},${b.getNorth().toFixed(5)}`;
    try {
        const params = new URLSearchParams({
            SERVICE: 'WFS', VERSION: '2.0.0', request: 'GetFeature',
            TYPENAMES: WFS_LAYER, COUNT: COMMUNE_COUNT,
            outputFormat: 'application/json',
            CQL_FILTER: `BBOX(${WFS_GEOM},${bbox},'EPSG:4326')`
        });
        const resp = await fetch(`${WFS_URL}?${params}`, { signal: abort.signal });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        if (abort.signal.aborted || communeCode !== code) return;

        const zone = { features: [feature] };
        const inCommune = (data.features || []).filter(f => {
            const c = centroid(f.geometry);
            return c && pointInGeoJSON(c[1], c[0], zone);
        });
        lastViewportFeatures = inCommune;
        rerenderDetections();
        const z = S.map.getZoom();
        if (z >= BANDS.commune && z < HEAT_FADE) renderLiveHeat(applySessionEdits(inCommune));
    } catch (e) {
        if (e.name !== 'AbortError') {
            console.error('WFS commune fetch:', e);
            logEvent('wfs_error', `commune: ${e.message}`);
            if (communeCode === code) communeCode = null;   // allow retry
        }
    } finally {
        if (communeAbort === abort) { communeAbort = null; hide('wfs-spinner'); }
    }
}

// ─── WFS viewport fetch (detection polygons + live heat) ─────────────────────

function scheduleWfs() {
    if (wfsDebounce) clearTimeout(wfsDebounce);
    wfsDebounce = setTimeout(fetchViewport, 400);
}

async function fetchViewport() {
    const z = S.map.getZoom();
    if (z < BANDS.commune) return;
    // A newer viewport always wins: abort any in-flight request instead of
    // skipping (skipping silently dropped the fetch after fast drill-downs).
    if (wfsAbort) wfsAbort.abort();
    const abort = wfsAbort = new AbortController();
    show('wfs-spinner');

    const b = S.map.getBounds();
    const count = z >= BANDS.detail ? 2000 : 1000;
    const params = new URLSearchParams({
        SERVICE: 'WFS', VERSION: '2.0.0', request: 'GetFeature',
        TYPENAMES: WFS_LAYER, COUNT: count, outputFormat: 'application/json',
        CQL_FILTER: `BBOX(${WFS_GEOM},${b.getWest().toFixed(5)},${b.getSouth().toFixed(5)},${b.getEast().toFixed(5)},${b.getNorth().toFixed(5)},'EPSG:4326')`
    });

    try {
        const resp = await fetch(`${WFS_URL}?${params}`, { signal: abort.signal });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        lastViewportFeatures = data.features || [];
        rerenderDetections();
        if (S.map.getZoom() < BANDS.detail && S.map.getZoom() >= BANDS.commune)
            renderLiveHeat(applySessionEdits(lastViewportFeatures));
    } catch (e) {
        if (e.name !== 'AbortError') {
            console.error('WFS error:', e);
            logEvent('wfs_error', `viewport: ${e.message}`);
        }
    } finally {
        if (wfsAbort === abort) { wfsAbort = null; hide('wfs-spinner'); }
    }
}

// ─── Detection polygon styling / popup ────────────────────────────────────────

function kwpColor(kwp) {
    const k = parseFloat(kwp) || 0;
    if (k <= 3)  return '#c9b3e8';
    if (k <= 6)  return '#a07cd4';
    if (k <= 10) return '#7248b8';
    if (k <= 20) return '#4e1d96';
    return '#2c0870';
}

function stylePolygon(f) {
    return { fillColor: kwpColor(f.properties.kwp_approx), fillOpacity: 0.65,
             color: '#333', weight: 0.8, opacity: 0.9 };
}

function bindPolyEvents(f, layer) {
    layer.on('mouseover', function () { this.setStyle({ weight: 2, fillOpacity: 0.85 }); });
    layer.on('mouseout',  function () { polyLayer.resetStyle(this); });
    layer.on('click', (e) => {
        L.DomEvent.stopPropagation(e);
        if (S.drawing) return;
        S.lastClickedDetection = { feature: f, layer, latlng: e.latlng };
        showPolyPopup(f.properties, e.latlng);
    });
}

function showPolyPopup(p, latlng) {
    const fmt = (v, d) => (v != null && !isNaN(v)) ? parseFloat(v).toFixed(d) : '—';
    const yld = p.yield_kwh ? Math.round(p.yield_kwh).toLocaleString('fr-FR') : '—';
    L.popup({ maxWidth: 240 }).setLatLng(latlng).setContent(`
        <div class="wfs-popup">
            <h4>Rooftop PV System</h4>
            <table>
                <tr><td>Surface</td><td><strong>${fmt(p.surface, 1)} m²</strong></td></tr>
                <tr><td>Capacity</td><td><strong>${fmt(p.kwp_approx, 2)} kWp</strong></td></tr>
                <tr><td>Est. yield</td><td><strong>${yld} kWh/yr</strong></td></tr>
                <tr><td>Detection year</td><td><strong>${p.detection_year || '—'}</strong></td></tr>
            </table>
            <div class="wfs-actions">
                <button onclick="annotModify()">✎ Fix shape</button>
                <button onclick="annotDelete()" class="danger">✕ Not a PV system</button>
            </div>
        </div>`).openOn(S.map);
}

// ─── Boundary + mask ──────────────────────────────────────────────────────────

export function setBoundary(geoJSON, withMask = true) {
    S.boundaryGeoJSON = geoJSON;
    boundaryLayer.clearLayers();
    if (maskLayer) { S.map.removeLayer(maskLayer); maskLayer = null; }
    if (!geoJSON) return;

    boundaryLayer.addData(geoJSON);
    if (!withMask) return;   // explore: highlight only, no dimming

    const exteriorRings = [];
    for (const f of geoJSON.features) {
        const geom = f.geometry;
        if (!geom) continue;
        if (geom.type === 'Polygon') exteriorRings.push(geom.coordinates[0]);
        else if (geom.type === 'MultiPolygon') geom.coordinates.forEach(p => exteriorRings.push(p[0]));
    }
    if (!exteriorRings.length) return;

    const world = [[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]];
    maskLayer = L.geoJSON(
        { type: 'Feature', geometry: { type: 'Polygon', coordinates: [world, ...exteriorRings] } },
        { pane: 'maskPane', interactive: false,
          style: { fillColor: '#0f141e', fillOpacity: 0.55, color: 'none', weight: 0, fillRule: 'evenodd' } }
    ).addTo(S.map);
}
