// ─── Free search (Nominatim) — pure camera move ───────────────────────────────
//
// Navigation is zoom-driven: flying to the result is enough, the zone engine
// (nav.js) resolves the level and stats from the landing viewport.

import { FRANCE_BOUNDS } from './config.js';
import { S, $, logEvent } from './store.js';
import { setSelectionFromBounds, setSelectionFromFeature, clearSelection } from './selection.js';
import { resolveExactFeature } from './geo.js';

let resultsCache = [];

export function initSearch() {
    const input    = $('city-search');
    const clearBtn = $('clear-btn');
    const results  = $('search-results');
    let debounce;

    input.addEventListener('input', () => {
        clearBtn.style.display = input.value.length ? 'flex' : 'none';
        clearTimeout(debounce);
        if (input.value.length >= 2) debounce = setTimeout(() => runSearch(input.value), 320);
        else hideResults();
    });
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') { clearSearchUI(); input.blur(); }
        if (e.key === 'Enter')  runSearch(input.value);
    });
    $('search-btn').addEventListener('click', () => runSearch(input.value));
    clearBtn.addEventListener('click', () => {
        clearSearchUI();
        clearSelection();
        S.map.flyToBounds(FRANCE_BOUNDS, { duration: 1.0 });
    });

    results.addEventListener('click', (e) => {
        const row = e.target.closest('[data-idx]');
        if (!row) return;
        const item = resultsCache[parseInt(row.dataset.idx, 10)];
        if (item) pickResult(item);
    });

    S.map.on('click', hideResults);
}

async function runSearch(q) {
    if (!q || q.length < 2) return;
    const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(q + ' France')}&format=json&limit=6&addressdetails=1`;
    try {
        const data = await fetch(url, { headers: { 'Accept-Language': 'fr' } }).then(r => r.json());
        resultsCache = data;
        renderResults(data);
    } catch (e) {
        console.error('Search error:', e);
    }
}

function renderResults(items) {
    const container = $('search-results');
    if (!items.length) {
        container.innerHTML = '<div class="sr-item" style="opacity:.4">No results</div>';
        container.style.display = 'block';
        return;
    }
    container.innerHTML = items.map((item, i) => {
        const label = item.display_name.split(',').slice(0, 2).join(',');
        const type  = item.addresstype || item.type || '';
        return `<div class="sr-item" data-idx="${i}"><span>${label}</span><span class="sr-type">${type}</span></div>`;
    }).join('');
    container.style.display = 'block';
}

async function pickResult(item) {
    const label = item.display_name.split(',').slice(0, 2).join(',').trim();
    logEvent('search', label);
    $('city-search').value = label;
    $('clear-btn').style.display = 'flex';
    hideResults();

    // Try to resolve the exact région/département/commune contour from our
    // own boundary files first — Nominatim's bbox is only a fallback for
    // whatever it doesn't match (arrondissements, hamlets, overseas communes).
    const result = await resolveExactFeature(item);
    if (result) {
        setSelectionFromFeature(result, label);
        return;
    }
    const [minLat, maxLat, minLon, maxLon] = item.boundingbox.map(Number);
    setSelectionFromBounds(L.latLngBounds([minLat, minLon], [maxLat, maxLon]), label);
}

function clearSearchUI() {
    $('city-search').value = '';
    $('clear-btn').style.display = 'none';
    hideResults();
}

function hideResults() {
    const el = $('search-results');
    el.innerHTML = '';
    el.style.display = 'none';
}
