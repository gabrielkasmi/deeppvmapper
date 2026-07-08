// ─── Boot ─────────────────────────────────────────────────────────────────────

import { FRANCE_BOUNDS, MAX_ZOOM } from './config.js';
import { S, logEvent } from './store.js';
import { initLayers } from './layers.js';
import { initNav, exitTarget } from './nav.js';
import { initSearch } from './search.js';
import { initAnnotate } from './annotate.js';
import { downloadAreaData } from './stats.js';

// Inline onclick handlers in map.html
window.closeStatsCard   = exitTarget;   // closing the card unlocks target mode
window.downloadAreaData = downloadAreaData;

// ─── View persistence (survives a reload — sessionStorage clears when the tab
// closes, not on refresh; the browser gives JS no way to tell a hard refresh
// apart from a normal one, so both restore the same way) ─────────────────────

const VIEW_KEY = 'dpvm_view';

function restoreView(map) {
    try {
        const saved = JSON.parse(sessionStorage.getItem(VIEW_KEY));
        if (saved && Number.isFinite(saved.lat) && Number.isFinite(saved.lng) && Number.isFinite(saved.zoom)) {
            map.setView([saved.lat, saved.lng], saved.zoom);
            return true;
        }
    } catch { /* corrupted/blocked storage — fall back to the default view */ }
    return false;
}

function persistView(map) {
    const c = map.getCenter();
    try {
        sessionStorage.setItem(VIEW_KEY, JSON.stringify({ lat: c.lat, lng: c.lng, zoom: map.getZoom() }));
    } catch { /* storage full/blocked — non-critical, just skip persistence */ }
}

// ─── Intro popup ──────────────────────────────────────────────────────────────

function initIntro() {
    const overlay = document.getElementById('intro-overlay');
    const open  = () => { overlay.style.display = 'flex'; };
    const close = () => { overlay.style.display = 'none'; };

    document.getElementById('intro-start').addEventListener('click', close);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
    document.getElementById('help-btn').addEventListener('click', open);

    open();   // shown on every page load (the "?" button reopens it anytime)
}

document.addEventListener('DOMContentLoaded', async () => {
    S.map = L.map('map', { zoomControl: true, maxZoom: MAX_ZOOM });
    if (!restoreView(S.map)) S.map.fitBounds(FRANCE_BOUNDS);
    S.map.on('moveend zoomend', () => persistView(S.map));

    logEvent('visit_map');
    initIntro();
    initLayers();
    initSearch();
    initAnnotate();

    try {
        await initNav();
    } catch (e) {
        console.error('Nav init failed:', e);
    }
});
