// ─── Filter panel: power / year (value-snapped sliders) / source (AND) / false-positive ─
//
// Pure client-side filtering over S.rawFeatures — no filter change ever
// re-queries Supabase. renderFiltered() (from render.js) just re-runs
// applyFilters() and redraws the current marker cluster.
//
// Power/year sliders don't sweep a continuous min-max range — they step
// through the sorted list of VALUES ACTUALLY PRESENT in the current selection
// (rebuilt on every new selection via updateFilterBounds()), so you can't land
// on, say, 2019 if the data only has 2017/2020/2023. That indexing also gives
// the kWp slider a "log-like" feel for free: real installed-power data is
// dense at the low end and sparse in the long tail, so stepping through
// sorted actual values naturally spends more slider travel on the crowded low
// end and less on the rare high end — closer to the real distribution than a
// fixed log curve would be, without extra math.
//
// A range only takes effect once the user actually moves a handle off the
// full extent ("validates" it) — from that point on, features with no value
// for that attribute are excluded, since there's no "unknown" bucket to fall
// back into. Source checkboxes are ANDed: checking two sources keeps only
// installations that carry both, not either.

import { SOURCE_LABELS } from './config.js';
import { S, $ } from './store.js';
import { renderFiltered } from './render.js';

// Sorted lists of actual values in the current selection — index-based
// sliders read/write positions into these, never raw numbers directly.
const domains = { kwp: [0, 1], year: [2015, new Date().getFullYear()] };

/** Pure filter — usable from render.js and export.js alike. */
export function applyFilters(features) {
    const { kwpMin, kwpMax, yearMin, yearMax, sources, hideFalsePositive } = S.filters;
    return features.filter(f => {
        const p = f.properties;
        if (hideFalsePositive && p.false_positive === true) return false;

        if (kwpMin != null || kwpMax != null) {
            const kwp = parseFloat(p.kwp);
            if (!Number.isFinite(kwp)) return false;   // no value -> excluded once this filter is active
            if (kwpMin != null && kwp < kwpMin) return false;
            if (kwpMax != null && kwp > kwpMax) return false;
        }

        if (yearMin != null || yearMax != null) {
            const y = parseInt(p.last_seen, 10);
            if (!Number.isFinite(y)) return false;
            if (yearMin != null && y < yearMin) return false;
            if (yearMax != null && y > yearMax) return false;
        }

        // AND, not OR: an installation must carry every checked source.
        if (sources && sources.size) {
            const own = String(p.sources || '').split(',').filter(Boolean);
            if (![...sources].every(s => own.includes(s))) return false;
        }

        return true;
    });
}

export function initFilterPanel() {
    const panel = $('filter-panel');
    if (!panel) return;

    panel.innerHTML = `
        <div class="fp-title">
            Filter this selection
            <button type="button" id="fp-close-btn" class="fp-close-btn" aria-label="Close filters" title="Close">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>
            </button>
        </div>
        <div class="fp-row">
            <label>Power (kWp) <span class="fp-vals" id="fp-kwp-vals"></span></label>
            <div class="fp-slider">
                <div class="fp-track-line"></div>
                <input id="fp-kwp-min" type="range" step="1">
                <input id="fp-kwp-max" type="range" step="1">
            </div>
        </div>
        <div class="fp-row">
            <label>Last seen (year) <span class="fp-vals" id="fp-year-vals"></span></label>
            <div class="fp-slider">
                <div class="fp-track-line"></div>
                <input id="fp-year-min" type="range" step="1">
                <input id="fp-year-max" type="range" step="1">
            </div>
        </div>
        <div class="fp-row">
            <label>Source (must match all checked)</label>
            <div class="fp-sources">
                ${Object.entries(SOURCE_LABELS).map(([idx, label]) => `
                    <label class="fp-check"><input type="checkbox" data-source="${idx}"> ${label}</label>
                `).join('')}
            </div>
        </div>
        <div class="fp-row">
            <label class="fp-check"><input id="fp-hide-fp" type="checkbox" checked> Remove false positives</label>
        </div>
    `;

    $('fp-kwp-min').addEventListener('input', () => updateSliderLabel('kwp'));
    $('fp-kwp-max').addEventListener('input', () => updateSliderLabel('kwp'));
    $('fp-kwp-min').addEventListener('change', () => applyFromSlider('kwp'));
    $('fp-kwp-max').addEventListener('change', () => applyFromSlider('kwp'));

    $('fp-year-min').addEventListener('input', () => updateSliderLabel('year'));
    $('fp-year-max').addEventListener('input', () => updateSliderLabel('year'));
    $('fp-year-min').addEventListener('change', () => applyFromSlider('year'));
    $('fp-year-max').addEventListener('change', () => applyFromSlider('year'));

    panel.querySelectorAll('[data-source]').forEach(el => el.addEventListener('change', applySourcesAndFP));
    $('fp-hide-fp').addEventListener('change', applySourcesAndFP);

    resetSlider('kwp',  domains.kwp);
    resetSlider('year', domains.year);

    // Gear button (map corner stack) and the panel's own close (✕) both
    // toggle/close the same panel. Wired once here at boot; render.js only
    // toggles the gear button's own visibility (shown once a selection
    // exists, hidden when cleared).
    $('filter-toggle-btn')?.addEventListener('click', () => {
        panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    });
    $('fp-close-btn')?.addEventListener('click', () => { panel.style.display = 'none'; });
}

/** Rebind the sliders to the values actually present in this selection, and
 *  clear any power/year filter left over from the previous one — source and
 *  false-positive filters carry over, since they don't depend on the data.
 *  Called by render.js right after a new selection's light points land. */
export function updateFilterBounds(features) {
    const kwpSet = new Set(), yearSet = new Set();
    for (const f of features) {
        const kwp = parseFloat(f.properties.kwp);
        if (Number.isFinite(kwp)) kwpSet.add(Math.round(kwp * 100) / 100);   // 2-decimal dedup keeps step count sane
        const y = parseInt(f.properties.last_seen, 10);
        if (Number.isFinite(y)) yearSet.add(y);
    }
    domains.kwp  = kwpSet.size  ? Array.from(kwpSet).sort((a, b) => a - b)  : [0, 1];
    domains.year = yearSet.size ? Array.from(yearSet).sort((a, b) => a - b) : [2015, new Date().getFullYear()];

    resetSlider('kwp',  domains.kwp);
    resetSlider('year', domains.year);
    S.filters.kwpMin = null; S.filters.kwpMax = null;
    S.filters.yearMin = null; S.filters.yearMax = null;
}

function resetSlider(kind, values) {
    const minEl = $(`fp-${kind}-min`), maxEl = $(`fp-${kind}-max`);
    if (!minEl || !maxEl) return;
    const lastIdx = values.length - 1;
    minEl.min = maxEl.min = 0;
    minEl.max = maxEl.max = lastIdx;
    minEl.value = 0;
    maxEl.value = lastIdx;
    updateSliderLabel(kind);
}

function updateSliderLabel(kind) {
    const minEl = $(`fp-${kind}-min`), maxEl = $(`fp-${kind}-max`);
    if (!minEl || !maxEl) return;
    if (+minEl.value > +maxEl.value) minEl.value = maxEl.value;   // keep handles from crossing
    const values = domains[kind];
    const minVal = values[+minEl.value], maxVal = values[+maxEl.value];
    const label = $(`fp-${kind}-vals`);
    if (!label) return;
    const fmt = kind === 'kwp' ? (v => `${v.toFixed(1)} kWp`) : (v => v);
    label.textContent = `${fmt(minVal)} – ${fmt(maxVal)}`;
}

/** Fires only on release (range 'change'), not while dragging — a range only
 *  starts filtering once it no longer covers the whole selection's extent. */
function applyFromSlider(kind) {
    updateSliderLabel(kind);
    const minEl = $(`fp-${kind}-min`), maxEl = $(`fp-${kind}-max`);
    const values = domains[kind];
    const minIdx = +minEl.value, maxIdx = +maxEl.value;
    const full = minIdx === 0 && maxIdx === values.length - 1;
    const minVal = values[minIdx], maxVal = values[maxIdx];
    if (kind === 'kwp') {
        S.filters.kwpMin = full ? null : minVal;
        S.filters.kwpMax = full ? null : maxVal;
    } else {
        S.filters.yearMin = full ? null : minVal;
        S.filters.yearMax = full ? null : maxVal;
    }
    renderFiltered();
}

function applySourcesAndFP() {
    const checkedSources = new Set(
        Array.from(document.querySelectorAll('[data-source]:checked')).map(el => el.dataset.source)
    );
    S.filters.sources = checkedSources.size ? checkedSources : null;
    S.filters.hideFalsePositive = $('fp-hide-fp').checked;
    renderFiltered();
}
