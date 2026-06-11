// ─── Stats card ───────────────────────────────────────────────────────────────
//
// showZoneStats() — exact baked KPIs + distributions from a WFS sample.
// Thresholds: n = 0 → explicit "no detections"; n < MIN_STATS_N → KPIs only.

import { WFS_URL, WFS_LAYER, WFS_GEOM, STATS_SAMPLE, MIN_STATS_N } from './config.js';
import { S, $, show, hide, fmtInt, centroid, filteredSample, logEvent } from './store.js';

let chartKwp = null, chartSurface = null;
let fetchSeq = 0;   // guards against out-of-order WFS responses

// ─── Public API ───────────────────────────────────────────────────────────────

export function showZoneStats(level, code, nom, feature) {
    const dict = { region: S.statsRegions, dept: S.statsDepts, commune: S.statsCommunes }[level];
    const st = dict[code];
    const n = st ? st[0] : 0;
    const kwp = st ? st[1] : 0;

    openCard(nom);
    S.areaLabel = nom;
    S.sampleFeatures = [];

    if (!n) {
        $('sc-kpis').innerHTML = level === 'commune'
            ? '<span style="opacity:.5;font-size:12px">No detections in this area.</span>'
            : '<span style="opacity:.5;font-size:12px">Area not covered by DeepPVMapper.</span>';
        setNote('');
        hide('sc-charts'); hide('sc-download');
        return;
    }

    renderKpis(n, kwp);

    if (n < MIN_STATS_N) {
        setNote(`Only ${fmtInt(n)} systems — too few for reliable distributions.`);
        hide('sc-charts'); hide('sc-download');
        return;
    }

    // Distributions from a WFS sample inside the zone (PIP-filtered on arrival)
    const b = L.geoJSON(feature).getBounds();
    fetchSample(b.getWest(), b.getSouth(), b.getEast(), b.getNorth(), (features) => {
        const inZone = filteredSample();
        renderCharts(inZone);
        const exhaustive = features.length < STATS_SAMPLE;
        setNote(exhaustive
            ? `Distributions computed on ${fmtInt(inZone.length)} systems.`
            : `Distributions computed on a sample of ${fmtInt(inZone.length)} systems (out of ${fmtInt(n)}).`);
    });
}

export function hideStatsCard() {
    hide('stats-card');
    destroyCharts();
}

// ─── Internals ────────────────────────────────────────────────────────────────

function openCard(title) {
    $('sc-title').textContent = title;
    $('sc-kpis').innerHTML = '';
    setNote('');
    destroyCharts();
    show('sc-charts');
    $('sc-download').style.display = 'flex';   // not show(): the button is a flexbox
    show('stats-card');
}

function setNote(html) { $('sc-note').innerHTML = html; }

function renderKpis(n, totalKwp) {
    const avg = n ? totalKwp / n : 0;
    $('sc-kpis').innerHTML = `
        <div class="sc-kpi">
            <span class="sc-kpi-val">${fmtInt(n)}</span>
            <span class="sc-kpi-lbl">Systems</span>
        </div>
        <div class="sc-kpi">
            <span class="sc-kpi-val">${fmtInt(Math.round(totalKwp))} kWp</span>
            <span class="sc-kpi-lbl">Total</span>
        </div>
        <div class="sc-kpi">
            <span class="sc-kpi-val">${avg.toFixed(1)}</span>
            <span class="sc-kpi-lbl">Avg kWp</span>
        </div>`;
}

async function fetchSample(minLon, minLat, maxLon, maxLat, onDone) {
    const seq = ++fetchSeq;
    const params = new URLSearchParams({
        SERVICE: 'WFS', VERSION: '2.0.0', request: 'GetFeature',
        TYPENAMES: WFS_LAYER, COUNT: STATS_SAMPLE, outputFormat: 'application/json',
        CQL_FILTER: `BBOX(${WFS_GEOM},${minLon.toFixed(5)},${minLat.toFixed(5)},${maxLon.toFixed(5)},${maxLat.toFixed(5)},'EPSG:4326')`
    });
    try {
        const resp = await fetch(`${WFS_URL}?${params}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        if (seq !== fetchSeq) return;   // a newer request superseded this one
        S.sampleFeatures = data.features || [];
        onDone(S.sampleFeatures);
    } catch (e) {
        if (seq !== fetchSeq) return;
        console.error('WFS stats error:', e);
        logEvent('wfs_error', `stats: ${e.message}`);
        setNote('<span style="color:#fc8181">Distributions unavailable (WFS service error).</span>');
        hide('sc-charts');
    }
}

// ─── Charts ───────────────────────────────────────────────────────────────────

function destroyCharts() {
    if (chartKwp)     { chartKwp.destroy();     chartKwp = null; }
    if (chartSurface) { chartSurface.destroy(); chartSurface = null; }
}

function renderCharts(features) {
    destroyCharts();
    show('sc-charts');

    const kwps = features.map(f => parseFloat(f.properties.kwp_approx) || 0).filter(v => v > 0);

    const KWP_BINS   = [0, 3, 6, 10, 20, 50, Infinity];
    const KWP_LABELS = ['0–3', '3–6', '6–10', '10–20', '20–50', '>50'];
    const KWP_COLORS = ['#c9b3e8', '#a07cd4', '#7248b8', '#4e1d96', '#2c0870', '#160038'];
    const kwpCounts  = new Array(KWP_LABELS.length).fill(0);
    kwps.forEach(k => {
        for (let i = 0; i < KWP_BINS.length - 1; i++)
            if (k >= KWP_BINS[i] && k < KWP_BINS[i + 1]) { kwpCounts[i]++; break; }
    });
    chartKwp = new Chart($('sc-chart-kwp').getContext('2d'), {
        type: 'bar',
        data: { labels: KWP_LABELS, datasets: [{ data: kwpCounts, backgroundColor: KWP_COLORS, borderWidth: 0, borderRadius: 3 }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { callbacks: { title: i => i[0].label + ' kWp' } } },
            scales: {
                x: { ticks: { color: 'rgba(255,255,255,0.5)', font: { size: 10 }, maxRotation: 0 }, grid: { display: false } },
                y: { ticks: { color: 'rgba(255,255,255,0.4)', font: { size: 10 }, maxTicksLimit: 4 }, grid: { color: 'rgba(255,255,255,0.06)' } }
            }
        }
    });

    const S_BINS   = [0, 5, 10, 20, 40, 80, 160, 320, Infinity];
    const S_LABELS = ['<5', '5–10', '10–20', '20–40', '40–80', '80–160', '160–320', '>320'];
    const S_COLORS = ['rgba(254,217,118,.75)', 'rgba(253,141,60,.75)', 'rgba(227,26,28,.75)', 'rgba(165,15,21,.75)',
                      'rgba(103,0,13,.75)', 'rgba(70,0,10,.75)', 'rgba(40,0,8,.75)', 'rgba(20,0,5,.75)'];
    const sCounts = new Array(S_LABELS.length).fill(0);
    features.map(f => parseFloat(f.properties.surface) || 0).filter(v => v > 0).forEach(s => {
        for (let i = 0; i < S_BINS.length - 1; i++)
            if (s >= S_BINS[i] && s < S_BINS[i + 1]) { sCounts[i]++; break; }
    });
    chartSurface = new Chart($('sc-chart-year').getContext('2d'), {
        type: 'polarArea',
        data: { labels: S_LABELS.map(l => l + ' m²'), datasets: [{ data: sCounts, backgroundColor: S_COLORS, borderWidth: 0 }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { r: { ticks: { display: false }, grid: { color: 'rgba(255,255,255,0.12)' }, pointLabels: { display: false } } }
        }
    });
}

// ─── CSV export (current zone sample) ─────────────────────────────────────────

export function downloadAreaData() {
    const features = filteredSample();
    if (!features.length) return;

    const header = 'lat,lng,surface_m2,kwp_approx,yield_kwh,detection_year';
    const rows = features.map(f => {
        const c = centroid(f.geometry);
        const p = f.properties;
        return [
            c ? c[0].toFixed(6) : '',
            c ? c[1].toFixed(6) : '',
            p.surface     != null ? parseFloat(p.surface).toFixed(2)    : '',
            p.kwp_approx  != null ? parseFloat(p.kwp_approx).toFixed(3) : '',
            p.yield_kwh   != null ? Math.round(p.yield_kwh)             : '',
            p.detection_year || ''
        ].join(',');
    });

    logEvent('download_csv', `${S.areaLabel} (${features.length} systems)`);

    const csv  = [header, ...rows].join('\n');
    const slug = S.areaLabel.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = `deeppvmapper_${slug || 'area'}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
}
