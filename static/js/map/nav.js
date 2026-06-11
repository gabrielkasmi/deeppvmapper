// ─── Navigation: EXPLORE / TARGET modes ───────────────────────────────────────
//
// EXPLORE (default): wander freely. Outlines staged by zoom (régions →
// départements → communes of the dept under the cursor), hover tooltips with
// the zone's stats, no stats card. Clicking a zone locks it.
//
// TARGET: locked on one zone — boundary + mask + zone heatmap + stats card,
// no tooltips, free zooming down to the detection polygons. Exit by clicking
// anywhere outside the zone (or closing the stats card).

import { GEO_BASE, STATS_BASE, FRANCE_BOUNDS, BANDS } from './config.js';
import { S, $, fmtInt, pointInGeoJSON, logEvent } from './store.js';
import { setBoundary, refreshHeat, setActiveCommune } from './layers.js';
import { showZoneStats, hideStatsCard } from './stats.js';

let navLayer;
let navKey = null;        // what the nav layer currently shows
let zoneSeq = 0;          // guards async commune loads against stale updates
let debounceT;
let cursor = null;        // last mouse position (null on touch devices)

const NAV_STYLE = {
    color: 'rgba(255,255,255,0.55)', weight: 1.2, opacity: 0.9,
    fillColor: '#ffffff', fillOpacity: 0.02
};
const NAV_STYLE_NODATA = { ...NAV_STYLE, color: 'rgba(255,255,255,0.22)', dashArray: '4 4' };
const NAV_HOVER = { weight: 2.2, color: '#5bc8f5', fillOpacity: 0.08 };

// ─── Init ─────────────────────────────────────────────────────────────────────

export async function initNav() {
    const [regions, depts, statsR, statsD, statsC, regionDepts] = await Promise.all([
        fetch(`${GEO_BASE}/regions.geojson`).then(r => r.json()),
        fetch(`${GEO_BASE}/departements.geojson`).then(r => r.json()),
        fetch(`${STATS_BASE}/stats_regions.json`).then(r => r.json()),
        fetch(`${STATS_BASE}/stats_departements.json`).then(r => r.json()),
        fetch(`${STATS_BASE}/stats_communes.json`).then(r => r.json()),
        fetch(`${GEO_BASE}/region_depts.json`).then(r => r.json()),
    ]);

    Object.assign(S, {
        regionsGeo: regions, deptsGeo: depts,
        statsRegions: statsR, statsDepts: statsD, statsCommunes: statsC,
        regionDepts,
    });

    S.heatPoints = Object.entries(statsC).map(([insee, [n, , lat, lng]]) =>
        ({ insee, dept: insee.slice(0, 2), n, lat, lng }));

    navLayer = L.geoJSON(null, {
        pane: 'navPane',
        style: f => zoneStats(f) ? NAV_STYLE : NAV_STYLE_NODATA,
        onEachFeature: bindNavEvents
    }).addTo(S.map);

    S.map.on('moveend zoomend', scheduleZoneRefresh);
    S.map.on('mousemove', (e) => { cursor = e.latlng; scheduleZoneRefresh(); });
    S.map.on('mouseout', () => { cursor = null; });
    S.map.on('click', onMapClick);
    await refreshZone(true);
}

function scheduleZoneRefresh() {
    clearTimeout(debounceT);
    debounceT = setTimeout(() => refreshZone(), 180);
}

// ─── EXPLORE: staged outlines following zoom band + cursor ───────────────────

function findContaining(features, lng, lat) {
    return features.find(f => f.geometry && pointInGeoJSON(lng, lat, { features: [f] }));
}

export async function refreshZone(force = false) {
    if (S.mode === 'target') return;   // locked: nothing follows the cursor
    if (S.drawing) return;             // drawing in progress: outlines stay hidden
    const seq = ++zoneSeq;
    const z = S.map.getZoom();
    const c = cursor || S.map.getCenter();

    const region = findContaining(S.regionsGeo.features, c.lng, c.lat);

    if (z < BANDS.dept) {
        S.level = 'france';
        S.path = [];
        setNavFeatures(S.regionsGeo.features, 'regions');
        highlight(region);                       // cyan outline on the hovered région
        renderBreadcrumb();
        if (force) refreshHeat();
        return;
    }

    const dept = findContaining(S.deptsGeo.features, c.lng, c.lat);

    if (z < BANDS.commune) {
        S.level = 'region';
        S.path = region ? [pathEntry('region', region)] : [];
        setNavFeatures(S.deptsGeo.features, 'depts');
        highlight(dept);                         // cyan outline on the hovered dept
        renderBreadcrumb();
        if (force) refreshHeat();
        return;
    }

    // commune band: show the communes of the dept under the cursor
    if (!dept) return;
    let fc;
    try { fc = await fetchCommunes(dept.properties.code); } catch { return; }
    // Mode may have changed while the commune file was loading
    if (seq !== zoneSeq || S.mode === 'target' || S.drawing) return;

    S.level = 'dept';
    S.path = [pathEntry('region', regionOf(dept)), pathEntry('dept', dept)];
    setNavFeatures(fc.features, `communes-${dept.properties.code}`);
    highlight(findContaining(fc.features, c.lng, c.lat) || dept);   // hovered commune
    renderBreadcrumb();
    if (force) refreshHeat();
}

/** Explore-mode highlight: same contrasted look as target (outline + dim mask). */
function highlight(feature) {
    setBoundary(feature ? { type: 'FeatureCollection', features: [feature] } : null);
}

// ─── TARGET: lock on a zone ───────────────────────────────────────────────────

export async function enterTarget(feature) {
    const level = zoneLevel(feature);
    logEvent('target_zone', `${level}:${feature.properties.nom}`);
    S.mode = 'target';
    S.level = level;
    S.target = pathEntry(level, feature);
    S.path = ancestorsOf(feature);

    setNavFeatures([], 'none');                 // outlines + tooltips off
    setBoundary({ type: 'FeatureCollection', features: [feature] });
    setActiveCommune(level === 'commune' ? feature : null);
    refreshHeat();
    renderBreadcrumb();
    showZoneStats(level, feature.properties.code, feature.properties.nom, feature);

    const bounds = L.geoJSON(feature).getBounds();
    S.map.flyToBounds(bounds, {
        padding: [30, 30],
        maxZoom: level === 'commune' ? 14 : undefined,
        duration: 1.1
    });
}

export function exitTarget() {
    if (S.mode !== 'target') return;
    S.mode = 'explore';
    S.target = null;
    setActiveCommune(null);
    setBoundary(null);
    hideStatsCard();
    navKey = null;                              // force outline rebuild
    refreshZone(true).then(() => refreshHeat());
}

function onMapClick(e) {
    if (S.drawing) return;                      // geoman draw/edit in progress
    if (S.mode !== 'target') return;            // explore: zone clicks handled on features
    const inside = S.boundaryGeoJSON &&
        pointInGeoJSON(e.latlng.lng, e.latlng.lat, S.boundaryGeoJSON);
    if (!inside) exitTarget();
    // Clicks inside the locked zone are reserved for annotation (phase 4).
}

/** Lock onto the administrative zone containing (lat, lng). Used by search. */
export async function targetAt(lat, lng, levelHint = 'commune') {
    const region = findContaining(S.regionsGeo.features, lng, lat);
    if (!region) return false;
    if (levelHint === 'region') { enterTarget(region); return true; }

    const dept = findContaining(S.deptsGeo.features, lng, lat);
    if (!dept) { enterTarget(region); return true; }
    if (levelHint === 'dept') { enterTarget(dept); return true; }

    try {
        const fc = await fetchCommunes(dept.properties.code);
        const commune = findContaining(fc.features, lng, lat);
        enterTarget(commune || dept);
    } catch {
        enterTarget(dept);
    }
    return true;
}

// ─── Shared helpers ───────────────────────────────────────────────────────────

function regionOf(dept) {
    return S.regionsGeo.features.find(r => r.properties.code === dept.properties.region);
}

function ancestorsOf(feature) {
    const level = zoneLevel(feature);
    if (level === 'region') return [pathEntry('region', feature)];
    if (level === 'dept')
        return [pathEntry('region', regionOf(feature)), pathEntry('dept', feature)];
    const dept = S.deptsGeo.features.find(d => d.properties.code === feature.properties.dept);
    return [pathEntry('region', dept && regionOf(dept)), pathEntry('dept', dept),
            pathEntry('commune', feature)];
}

function pathEntry(level, f) {
    return { level, code: f?.properties.code, nom: f?.properties.nom || '?', feature: f };
}

function setNavFeatures(features, key) {
    if (key === navKey) return;
    navKey = key;
    navLayer.clearLayers();
    if (features.length) navLayer.addData({ type: 'FeatureCollection', features });
}

function zoneLevel(f) {
    const p = f.properties;
    if (p.dept)   return 'commune';
    if (p.region) return 'dept';
    return 'region';
}

function zoneStats(f) {
    const dict = { region: S.statsRegions, dept: S.statsDepts, commune: S.statsCommunes }[zoneLevel(f)];
    return dict[f.properties.code] || null;
}

function tooltipHtml(f) {
    const st = zoneStats(f);
    const head = `<strong>${f.properties.nom}</strong><br>`;
    if (!st) {
        const msg = zoneLevel(f) === 'commune' ? 'No detections' : 'Not covered';
        return head + `<span style="opacity:.6">${msg}</span>`;
    }
    return head + `${fmtInt(st[0])} systems · ${fmtInt(Math.round(st[1]))} kWp`;
}

function bindNavEvents(f, layer) {
    layer.bindTooltip(tooltipHtml(f), { sticky: true });
    layer.on('mouseover', function () {
        this.setStyle(NAV_HOVER);
        if (zoneLevel(f) === 'dept') prefetchCommunes(f.properties.code);
    });
    layer.on('mouseout', function () { navLayer.resetStyle(this); });
    layer.on('click', (e) => {
        L.DomEvent.stopPropagation(e);
        if (S.drawing) return;                   // drawing: zone clicks disabled
        enterTarget(f);                          // explore → lock on the clicked zone
    });
}

// ─── Drawing-mode suspension (used by annotate.js) ───────────────────────────

/** Hide nav outlines so they can't swallow drawing clicks. */
export function suspendNav() {
    setNavFeatures([], 'none');
}

/** Restore outlines after drawing ends (explore mode only; target stays bare). */
export function resumeNav() {
    if (S.mode === 'target') return;
    navKey = null;
    refreshZone(true);
}

// ─── Breadcrumb ───────────────────────────────────────────────────────────────

function renderBreadcrumb() {
    const el = $('breadcrumb');
    const locked = S.mode === 'target';
    const parts = [`<span class="bc-item" data-idx="-1">France</span>`];
    S.path.forEach((p, i) => {
        const last = i === S.path.length - 1;
        parts.push('<span class="bc-sep">›</span>');
        parts.push(last && locked
            ? `<span class="bc-item bc-current">${p.nom}</span>`
            : `<span class="bc-item" data-idx="${i}">${p.nom}</span>`);
    });
    if (locked) parts.push('<span class="bc-sep" style="opacity:.6">●</span>');
    el.innerHTML = parts.join('');
    el.style.display = 'flex';

    el.querySelectorAll('.bc-item[data-idx]').forEach(item => {
        item.addEventListener('click', () => {
            const idx = parseInt(item.dataset.idx, 10);
            if (idx === -1) {
                exitTarget();
                S.map.flyToBounds(FRANCE_BOUNDS, { duration: 1.0 });
                return;
            }
            const p = S.path[idx];
            if (p?.feature) enterTarget(p.feature);
        });
    });
}

// ─── Commune geometry loading / prefetch ─────────────────────────────────────

async function fetchCommunes(deptCode) {
    if (S.communeCache.has(deptCode)) return S.communeCache.get(deptCode);
    const promise = fetch(`${GEO_BASE}/communes/communes-${deptCode}.geojson`)
        .then(r => {
            if (!r.ok) throw new Error(`communes-${deptCode}: HTTP ${r.status}`);
            return r.json();
        })
        // 37 communes ship without geometry in france-geojson — Leaflet chokes on them
        .then(fc => ({ type: 'FeatureCollection', features: fc.features.filter(f => f.geometry) }))
        .catch(e => { S.communeCache.delete(deptCode); throw e; });
    S.communeCache.set(deptCode, promise);
    return promise;
}

function prefetchCommunes(deptCode) {
    fetchCommunes(deptCode).catch(() => {});
}
