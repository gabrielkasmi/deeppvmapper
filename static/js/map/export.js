// ─── Export: CSV (centroid) / GeoJSON (polygons) — current selection, filtered ─
//
// CSV reuses S.rawFeatures (the light points already in memory — centroids are
// all a CSV needs). GeoJSON export promises real polygons, but S.rawFeatures
// is now the light, geometry-less fetch (see render.js's two-tier model), so
// exportGeoJSON() does its own one-shot exact-zone fetch (get_detections_in_zone
// or the admin fast path, uncapped) to get full geometry back — which can take
// a while over a large selection. It runs as a plain async fetch (nothing on
// the page blocks on it — panning, filtering, drawing a new selection all
// keep working) and reports progress through #export-toast, a small
// notification separate from #wfs-spinner (shared by selection/single-feature
// loads) so the two don't get confused with each other.

import { S, centroid, logEvent, fetchDetectionsInZone, fetchDetectionsByAdmin, boundsToRectFeature } from './store.js';
import { applyFilters } from './filters.js';

// Lives in #export-controls, inside the #map-bottom-toolbar row (floating
// over the bottom of the map, beside the always-present "Add a missing
// installation" button) — a compact, regular-sized button that opens a
// small dropdown to pick the format. Only shown while a selection exists
// (render.js toggles #export-controls); the Add button next to it is not
// selection-scoped, so it stays visible on its own.
export function initExport() {
    const bar = document.getElementById('export-controls');
    if (!bar) return;

    const row = document.createElement('div');
    row.className = 'fp-dropdown';
    row.innerHTML = `
        <button id="export-toggle" class="me-action-btn fp-export-btn">Download detections ▾</button>
        <div class="fp-dropdown-menu" id="export-menu" style="display:none">
            <button id="export-geojson" class="fp-dropdown-item">GeoJSON (polygons)</button>
            <button id="export-csv" class="fp-dropdown-item">CSV (centroids)</button>
        </div>
    `;
    bar.appendChild(row);

    const toggle = document.getElementById('export-toggle');
    const menu = document.getElementById('export-menu');
    toggle.addEventListener('click', (e) => {
        e.stopPropagation();
        menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
    });
    document.addEventListener('click', () => { menu.style.display = 'none'; });
    document.getElementById('export-csv').addEventListener('click', (e) => {
        e.stopPropagation(); menu.style.display = 'none'; exportCSV();
    });
    document.getElementById('export-geojson').addEventListener('click', (e) => {
        e.stopPropagation(); menu.style.display = 'none'; exportGeoJSON();
    });
}

function slug() {
    return (S.selectionLabel || 'selection').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
}

// ─── Export notification (#export-toast) ──────────────────────────────────────

let toastHideTimer;

function showToast(message, { spinner = true, cls = '' } = {}) {
    const el = document.getElementById('export-toast');
    if (!el) return;
    clearTimeout(toastHideTimer);
    el.className = `me-glass${cls ? ' ' + cls : ''}`;
    el.style.display = 'flex';
    el.innerHTML = `${spinner ? '<span class="et-spinner"></span>' : ''}<span>${message}</span>`;
}

function hideToastAfter(ms) {
    clearTimeout(toastHideTimer);
    toastHideTimer = setTimeout(() => {
        const el = document.getElementById('export-toast');
        if (el) el.style.display = 'none';
    }, ms);
}

function download(content, filename, mime) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([content], { type: mime }));
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
}

export function exportCSV() {
    const features = applyFilters(S.rawFeatures);
    if (!features.length) {
        showToast('No detections match the current filters.', { spinner: false, cls: 'et-error' });
        hideToastAfter(4000);
        return;
    }

    // Union of all property keys across the selection (schema-agnostic — new
    // columns from the backend show up automatically, nothing to update here).
    const keys = Array.from(features.reduce((set, f) => {
        Object.keys(f.properties || {}).forEach(k => set.add(k));
        return set;
    }, new Set()));

    const header = ['lat', 'lng', ...keys].join(',');
    const rows = features.map(f => {
        const c = centroid(f.geometry);
        const vals = keys.map(k => csvCell(f.properties[k]));
        return [c ? c[0].toFixed(6) : '', c ? c[1].toFixed(6) : '', ...vals].join(',');
    });

    logEvent('download_csv', `${S.selectionLabel} (${features.length} systems)`);
    download([header, ...rows].join('\n'), `deeppvmapper_${slug()}.csv`, 'text/csv');
    showToast(`CSV ready — ${features.length.toLocaleString('fr-FR')} systems downloaded.`, { spinner: false, cls: 'et-done' });
    hideToastAfter(4000);
}

function csvCell(v) {
    if (v == null) return '';
    const s = String(v);
    return /[,"\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export async function exportGeoJSON() {
    if (!S.selectionBounds) return;
    showToast('Preparing GeoJSON export… this can take a moment for a large area.');
    try {
        // Admin-code fast path when we have one (named-place selection) — more
        // accurate too, since insee/dpt was assigned by the pipeline, not by
        // our simplified boundary file. Else the exact polygon if we have one
        // (the zone RPC takes arbitrary geometry, not just rectangles), else the bbox.
        const rawPolygons = S.selectionAdmin
            ? await fetchDetectionsByAdmin(S.selectionAdmin.inseeCodes, S.selectionAdmin.deptCodes)
            : await fetchDetectionsInZone(
                  S.selectionGeometry
                      ? { type: 'Feature', properties: {}, geometry: S.selectionGeometry }
                      : boundsToRectFeature(S.selectionBounds)
              );
        const features = applyFilters(rawPolygons);
        if (!features.length) {
            showToast('No detections match the current filters.', { spinner: false, cls: 'et-error' });
            hideToastAfter(4000);
            return;
        }
        logEvent('download_geojson', `${S.selectionLabel} (${features.length} systems)`);
        const fc = { type: 'FeatureCollection', features };
        download(JSON.stringify(fc), `deeppvmapper_${slug()}.geojson`, 'application/geo+json');
        showToast(`GeoJSON ready — ${features.length.toLocaleString('fr-FR')} systems downloaded.`, { spinner: false, cls: 'et-done' });
        hideToastAfter(5000);
    } catch (e) {
        console.error('GeoJSON export fetch error:', e);
        logEvent('detections_error', `export: ${e.message || e}`);
        showToast('Export failed — see console for details.', { spinner: false, cls: 'et-error' });
        hideToastAfter(5000);
    }
}
