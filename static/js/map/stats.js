// ─── Stats card ───────────────────────────────────────────────────────────────
//
// showZoneStats() — exact baked KPIs + distributions from a WFS sample.
// Thresholds: n = 0 → explicit "no detections"; n < MIN_STATS_N → KPIs only.

import { STATS_SAMPLE, MIN_STATS_N } from './config.js';
import { S, $, show, hide, fmtInt, centroid, filteredSample, logEvent,
         fetchDetectionsBBox, fetchDetectionsInZone } from './store.js';

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

    // Region-wide zones are too large for a single exhaustive fetch — no export.
    if (level === 'region') hide('sc-download');

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
    try {
        const features = await fetchDetectionsBBox(minLon, minLat, maxLon, maxLat, STATS_SAMPLE);
        if (seq !== fetchSeq) return;   // a newer request superseded this one
        S.sampleFeatures = features;
        onDone(S.sampleFeatures);
    } catch (e) {
        if (seq !== fetchSeq) return;
        console.error('Detections stats error:', e);
        logEvent('detections_error', `stats: ${e.message || e}`);
        setNote('<span style="color:#fc8181">Distributions unavailable (data service error).</span>');
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

    const kwps = features.map(f => parseFloat(f.properties.kwp) || 0).filter(v => v > 0);

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

// ─── CSV export (exhaustive — all systems in the zone, not the chart sample) ──

export async function downloadAreaData() {
    const zoneFeature = S.boundaryGeoJSON?.features?.[0];
    if (!zoneFeature) return;

    const btn = $('sc-download');
    const label = $('sc-download-label');
    if (btn.disabled) return;   // already running
    btn.disabled = true;

    const flash = (msg, delay) => {
        label.textContent = msg;
        setTimeout(() => { label.textContent = 'Download CSV'; }, delay);
    };

    try {
        label.textContent = 'Fetching…';   // no measurable progress on a single RPC call
        const features = await fetchDetectionsInZone(zoneFeature);
        if (!features.length) { flash('No data', 1500); return; }

        const header = 'lat,lng,surface_m2,kwp,detection_year';
        const rows = features.map(f => {
            const c = centroid(f.geometry);
            const p = f.properties;
            return [
                c ? c[0].toFixed(6) : '',
                c ? c[1].toFixed(6) : '',
                p.surface != null ? parseFloat(p.surface).toFixed(2) : '',
                p.kwp     != null ? parseFloat(p.kwp).toFixed(3)     : '',
                p.year || ''
            ].join(',');
        });

        logEvent('download_csv', `${S.areaLabel} (${features.length} systems)`);

        const csv  = [header, ...rows].join('\n');
        const slug = S.areaLabel.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');

        await triggerDownload(csv, slug || 'area', pct => {
            label.textContent = `Compressing… ${Math.round(pct)}%`;
        });
        label.textContent = 'Download CSV';
    } catch (e) {
        console.error('CSV export error:', e);
        logEvent('detections_error', `csv_export: ${e.message || e}`);
        flash('Error — retry', 2000);
    } finally {
        btn.disabled = false;
    }
}

/** Always zips — JSZip's onUpdate gives real compression progress, unlike the fetch step. */
async function triggerDownload(csv, slug, onProgress) {
    if (!window.JSZip) {   // graceful fallback if the CDN script failed to load
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
        a.download = `deeppvmapper_${slug}.csv`;
        a.click();
        URL.revokeObjectURL(a.href);
        return;
    }
    const zip = new window.JSZip();
    zip.file(`deeppvmapper_${slug}.csv`, csv);
    const blob = await zip.generateAsync(
        { type: 'blob', compression: 'DEFLATE' },
        meta => onProgress?.(meta.percent)
    );
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `deeppvmapper_${slug}.zip`;
    a.click();
    URL.revokeObjectURL(a.href);
}
