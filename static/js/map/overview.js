// ─── National overview: département-level counts, shown before any zone is
// selected ──────────────────────────────────────────────────────────────────
//
// The full dataset (1.1M+ detections) can't be rendered point-by-point on
// load — even the "light" centroid-only fetch is capped well below that
// (POINTS_MAX in config.js). Instead, the very first thing a visitor sees is
// one small aggregate: a circle per département (~96 of them), sized by
// system count, from dept_capacity_stats() — the same pre-aggregated RPC the
// per-département static stats pages already use (dept_stats_rpcs.sql).
// Département centroids come from the boundary file already loaded for
// search/deep-links (geo.js's loadDepartements) — nothing new to ship.
//
// Rendered through the same marker-cluster plugin as real detections
// (leaflet.markercluster, already loaded) rather than a plain layer group —
// several départements (Île-de-France in particular) are small enough that
// their circles overlap at the national zoom; clustering folds those into a
// single "N départements here" bubble instead of an illegible pile of discs,
// and un-clusters automatically on zoom in, for free.
//
// Clicking an individual (un-clustered) département circle opens a small
// popup with its headline numbers (count + total capacity) and a "Show
// detections" button; that button is what actually loads it — the same way
// typing its name in the search bar does (setSelectionFromFeature), same
// fast indexed dept-code fetch, same cluster rendering. That call is also
// what makes this overview disappear (render.js's loadSelection calls
// hideDeptOverview()).

import { DEPT_OVERVIEW_MIN_RADIUS, DEPT_OVERVIEW_MAX_RADIUS } from './config.js';
import { S, fmtInt, centroid, logEvent, fetchDeptStats } from './store.js';
import { loadDepartements, resolveDepartementByCode } from './geo.js';

let overviewLayer = null;
let ready = false;   // true once the layer has data worth showing

export async function initDeptOverview() {
    overviewLayer = L.markerClusterGroup({
        maxClusterRadius: 60, spiderfyOnMaxZoom: false, showCoverageOnHover: false, chunkedLoading: true
    });
    try {
        const [depts, stats] = await Promise.all([loadDepartements(), fetchDeptStats()]);
        const statsByDept = new Map(stats.map(r => [r.dpt, {
            n: Number(r.n_systems) || 0, kwp: Number(r.total_kwp) || 0,
        }]));
        const maxN = Math.max(1, ...Array.from(statsByDept.values(), v => v.n));

        const markers = [];
        for (const feature of depts.features) {
            const code = feature.properties.code;
            const stat = statsByDept.get(code);
            if (!stat || !stat.n) continue;   // nothing to show for a département with no data
            const c = centroid(feature.geometry);
            if (!c) continue;

            const radius = DEPT_OVERVIEW_MIN_RADIUS +
                (DEPT_OVERVIEW_MAX_RADIUS - DEPT_OVERVIEW_MIN_RADIUS) * Math.sqrt(stat.n / maxN);
            const name = feature.properties.nom || code;
            const marker = L.marker([c[0], c[1]], { icon: deptIcon(radius) });
            marker.bindTooltip(`${name} — ${fmtInt(stat.n)}`, { direction: 'top', offset: [0, -radius] });
            marker.on('click', (e) => {
                L.DomEvent.stopPropagation(e);
                openDeptPopup(marker, code, name, stat);
            });
            markers.push(marker);
        }
        overviewLayer.addLayers(markers);
        ready = true;
    } catch (e) {
        console.error('Département overview failed to load:', e);
        return;
    }

    // Only show it if nothing has claimed a selection in the meantime — a
    // ?dept=/?insee=/?region= deep link (selection.js) resolves asynchronously
    // right after this and calls hideDeptOverview() itself when it lands, but
    // it could in principle land first.
    if (!S.selectionBounds) showDeptOverview();
}

/** A plain colored disc, sized to the département's system count — same look
 *  as a circleMarker, just as a divIcon so it can live inside a
 *  MarkerClusterGroup (which expects L.Marker-derived layers, not vector
 *  layers like L.CircleMarker). */
function deptIcon(radius) {
    const size = Math.round(radius * 2);
    return L.divIcon({
        className: 'dept-overview-icon',
        html: `<span style="display:block;width:${size}px;height:${size}px;border-radius:50%;
                     background:#5bc8f5;border:1.5px solid #2c5f8a;opacity:.75;"></span>`,
        iconSize: [size, size],
        iconAnchor: [size / 2, size / 2],
    });
}

function formatCapacity(kwp) {
    if (!kwp) return '—';
    return kwp >= 1000 ? `${(kwp / 1000).toLocaleString('fr-FR', { maximumFractionDigits: 1 })} MWp`
                        : `${fmtInt(Math.round(kwp))} kWp`;
}

/** Headline numbers for one département + a "Show detections" button, which
 *  is the only thing that actually triggers the real fetch — clicking the
 *  circle itself just previews the numbers already in hand. Built as a DOM
 *  node (not an HTML string) so the button can carry a plain closure instead
 *  of a window-global inline handler. */
function openDeptPopup(marker, code, name, stat) {
    const el = document.createElement('div');
    el.className = 'di-popup';
    el.innerHTML = `
        <h4>${name}</h4>
        <table>
            <tr><td>Detections</td><td><strong>${fmtInt(stat.n)}</strong></td></tr>
            <tr><td>Total capacity</td><td><strong>${formatCapacity(stat.kwp)}</strong></td></tr>
        </table>
        <div class="di-actions"></div>`;
    const actions = el.querySelector('.di-actions');
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'di-action-btn';
    btn.textContent = 'Show detections';
    btn.addEventListener('click', () => {
        S.map.closePopup();
        selectDepartement(code, name);
    });
    actions.appendChild(btn);

    L.popup({ className: 'di-popup-wrap', maxWidth: 220 })
        .setLatLng(marker.getLatLng())
        .setContent(el)
        .openOn(S.map);
}

async function selectDepartement(code, name) {
    // Dynamic import: selection.js imports render.js, which imports this
    // module (to hide/show the overview) — a static import here the other
    // way would be circular. Deferring it to call time (everything is already
    // loaded by the time a user can click a marker) sidesteps that cleanly.
    const { setSelectionFromFeature } = await import('./selection.js');
    const result = await resolveDepartementByCode(code);
    if (!result) return;
    logEvent('overview_dept_click', name);
    setSelectionFromFeature(result, name);
}

export function showDeptOverview() {
    if (overviewLayer && ready && !S.map.hasLayer(overviewLayer)) S.map.addLayer(overviewLayer);
}

export function hideDeptOverview() {
    if (overviewLayer && S.map.hasLayer(overviewLayer)) S.map.removeLayer(overviewLayer);
}
