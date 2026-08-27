// ─── "Progress" tab map: a small choropleth of France, colored by each
// département's share of the votes cast so far WITHIN THE ACTIVE BATCH
// (not by completion % — see the comment on season_progress_by_department()
// in scripts/verifications_setup.sql for why those are two different
// numbers). This share re-normalizes as the batch fills in, so a
// département can visibly gain and lose the "leading" color as other
// départements catch up — that's intentional, it's relative progress
// within the current batch, not a fixed target. Hand-rolled SVG on
// purpose: 101 static polygons is small enough
// that pulling in a mapping library (tiles, panning, a JS dependency) would
// be solving a problem this doesn't have — same "right-sized tool" call as
// using plain WMS GetMap instead of a tile layer for the swipe cards.
//
// game/data/departements.geojson is a simplified (mapshaper -simplify 1.5%)
// copy of gregoiredavid/france-geojson's departements-avec-outre-mer.geojson
// — original is ~3.7MB, this is ~62KB. Re-simplify from that source if the
// borders ever need to look sharper.

const GEOJSON_URL = 'data/departements.geojson';
const SCALE = 60; // arbitrary — just keeps the SVG's viewBox in a friendly ~600-unit range

const COLOR_EMPTY = [38, 47, 58];   // --surface-2 — no/near-zero activity
const COLOR_MAX   = [52, 211, 153]; // --confirm — this dept has the most activity right now

let cachedFeatures = null;

async function loadFeatures() {
    if (cachedFeatures) return cachedFeatures;
    const res = await fetch(GEOJSON_URL);
    const geo = await res.json();
    // Metropolitan France + Corsica only (2-character codes: '01'..'95',
    // '2A'/'2B'). Overseas départements (971-976) are geographically far
    // from mainland France and would need their own inset to plot sensibly
    // on the same projection — left off this map rather than distorting it.
    cachedFeatures = geo.features.filter(f => f.properties.code.length === 2);
    return cachedFeatures;
}

function computeBounds(features) {
    let minLng = Infinity, maxLng = -Infinity, minLat = Infinity, maxLat = -Infinity;
    features.forEach(f => {
        const polys = f.geometry.type === 'MultiPolygon' ? f.geometry.coordinates : [f.geometry.coordinates];
        polys.forEach(poly => poly.forEach(ring => ring.forEach(([lng, lat]) => {
            if (lng < minLng) minLng = lng;
            if (lng > maxLng) maxLng = lng;
            if (lat < minLat) minLat = lat;
            if (lat > maxLat) maxLat = lat;
        })));
    });
    const cosLat = Math.cos((minLat + maxLat) / 2 * Math.PI / 180); // rough lng/lat aspect correction
    return {
        minLng, maxLat, cosLat,
        width: (maxLng - minLng) * cosLat * SCALE,
        height: (maxLat - minLat) * SCALE,
    };
}

function project(lng, lat, bounds) {
    const x = (lng - bounds.minLng) * bounds.cosLat * SCALE;
    const y = (bounds.maxLat - lat) * SCALE; // SVG y grows downward, latitude grows upward
    return [x, y];
}

function ringToPath(ring, bounds) {
    return ring.map(([lng, lat], i) => {
        const [x, y] = project(lng, lat, bounds);
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join('') + 'Z';
}

function geometryToPath(geometry, bounds) {
    const polys = geometry.type === 'MultiPolygon' ? geometry.coordinates : [geometry.coordinates];
    // evenodd so holes (rare, but some départements have them) render
    // correctly without having to track outer/inner ring winding order.
    return polys.map(poly => poly.map(ring => ringToPath(ring, bounds)).join('')).join('');
}

function lerpColor(t, from, to) {
    const r = Math.round(from[0] + (to[0] - from[0]) * t);
    const g = Math.round(from[1] + (to[1] - from[1]) * t);
    const b = Math.round(from[2] + (to[2] - from[2]) * t);
    return `rgb(${r},${g},${b})`;
}

/** rows: the array returned by the season_progress_by_department() RPC. */
export async function renderDeptMap(container, rows) {
    let features;
    try {
        features = await loadFeatures();
    } catch (err) {
        console.error('departements.geojson failed to load:', err);
        container.innerHTML = '<p class="menu-dept-error">Map outline failed to load — try reopening this tab.</p>';
        return;
    }

    const bounds = computeBounds(features);
    const byDept = {};
    (rows || []).forEach(r => { byDept[r.dpt] = r; });
    // rows can come back non-empty but with every vote_share_pct at 0 (e.g.
    // if the voted-on installations' detections.dpt is unset) — that's
    // visually identical to "no data" (every path lands on COLOR_EMPTY), so
    // it needs its own flag rather than just checking rows.length.
    const hasShare = (rows || []).some(r => (Number(r.vote_share_pct) || 0) > 0);
    const maxShare = Math.max(0.01, ...(rows || []).map(r => Number(r.vote_share_pct) || 0));

    if (!hasShare) {
        console.warn('season_progress_by_department returned no usable vote_share_pct', { rowCount: (rows || []).length, rows });
    }

    const paths = features.map(f => {
        const row = byDept[f.properties.code];
        const share = row ? Number(row.vote_share_pct) || 0 : 0;
        const t = Math.min(1, share / maxShare);
        const fill = lerpColor(t, COLOR_EMPTY, COLOR_MAX);
        const d = geometryToPath(f.geometry, bounds);
        const title = row
            ? `${f.properties.code} — ${row.pct}% of the active batch (${Number(row.votes_cast).toLocaleString()} / ${Number(row.votes_target).toLocaleString()} votes) — ${row.vote_share_pct}% of the active batch's votes so far`
            : `${f.properties.code} — no votes yet`;
        return `<path d="${d}" fill="${fill}" stroke="#12171d" stroke-width="0.6"><title>${title}</title></path>`;
    }).join('');

    const note = hasShare ? '' :
        '<p class="menu-dept-error">No département data yet for the votes cast so far — see the browser console for details.</p>';

    container.innerHTML =
        `<svg viewBox="0 0 ${bounds.width.toFixed(1)} ${bounds.height.toFixed(1)}" role="img" ` +
        `aria-label="Map of metropolitan France, départements colored by share of community verification votes cast so far">` +
        paths + `</svg>` + note;
}
