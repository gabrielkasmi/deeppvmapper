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
    S.map = L.map('map', { zoomControl: true, maxZoom: MAX_ZOOM }).fitBounds(FRANCE_BOUNDS);

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
