// ─── Boot ─────────────────────────────────────────────────────────────────────

import { FRANCE_BOUNDS, MAX_ZOOM } from './config.js';
import { S, logEvent } from './store.js';
import { initMap } from './render.js';
import { initDeptOverview } from './overview.js';
import { initSelection } from './selection.js';
import { initSearch } from './search.js';
import { initFilterPanel } from './filters.js';
import { initExport } from './export.js';
import { initAnnotate } from './annotate.js';
import { initReport } from './report.js';
import { initVersionInfo } from './version.js';

// ─── Intro popup ──────────────────────────────────────────────────────────────

function initIntro() {
    const overlay = document.getElementById('intro-overlay');
    if (!overlay) return;
    const open  = () => { overlay.style.display = 'flex'; };
    const close = () => { overlay.style.display = 'none'; };

    document.getElementById('intro-start')?.addEventListener('click', close);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
    document.getElementById('help-btn')?.addEventListener('click', open);
}

document.addEventListener('DOMContentLoaded', async () => {
    S.map = L.map('map', { zoomControl: true, maxZoom: MAX_ZOOM });
    S.map.fitBounds(FRANCE_BOUNDS);   // always start from the national view — no reload persistence

    logEvent('visit_map');
    initIntro();
    initVersionInfo();   // independent of the map itself — fine to kick off in parallel

    initMap();
    initDeptOverview();   // fire-and-forget: resolves async, hidden again instantly if a deep link lands first
    initSearch();
    initFilterPanel();
    initExport();
    initAnnotate();
    initReport();
    initSelection();          // last: may immediately fire a ?dept= deep-link selection
});
