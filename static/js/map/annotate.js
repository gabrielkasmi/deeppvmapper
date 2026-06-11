// ─── Annotation: report / fix / add PV systems ───────────────────────────────
//
// Edits apply visually for the session (S.edits, rendered by layers.js) and
// are submitted to Supabase (insert-only, moderated offline). On reload the
// map starts clean from the official data — by design.
//
// Flows:
//   delete  — detection popup → "Not a PV system" → form → submit
//   modify  — detection popup → "Fix shape" → original dimmed, REDRAW the
//             outline from scratch (segmentation polygons have far too many
//             vertices to drag one by one) → form → submit
//   add     — side button → draw polygon → form → submit
// While drawing, nav outlines are hidden and their clicks disabled (nav.js).
//
// Requires leaflet-geoman (drawing) and @supabase/supabase-js (submission);
// both optional — the UI degrades to local-only/disabled if missing.

import { SUPABASE_URL, SUPABASE_ANON_KEY, ANNOT_MAX_SESSION, ANNOT_MIN_INTERVAL_MS } from './config.js';
import { S, $, show, hide, fmtInt, featureKey } from './store.js';
import { rerenderDetections } from './layers.js';
import { suspendNav, resumeNav } from './nav.js';

let sb = null;
let totalCount = null;          // lifetime counter from annotation_count()
let sessionCount = 0;
let lastSubmitAt = 0;
let pending = null;             // {action, feature?, layer?, geometry?}
let addedLayer;

const ADDED_STYLE = { color: '#34d399', weight: 2, dashArray: '5 4', fillColor: '#34d399', fillOpacity: 0.35 };

// ─── Init ─────────────────────────────────────────────────────────────────────

export function initAnnotate() {
    if (SUPABASE_URL && SUPABASE_ANON_KEY && window.supabase)
        sb = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

    addedLayer = L.geoJSON(null, { style: ADDED_STYLE }).addTo(S.map);

    $('annot-add-btn').addEventListener('click', startAdd);
    $('annot-submit').addEventListener('click', submitForm);
    $('annot-cancel').addEventListener('click', cancelForm);
    $('annot-cancel-edit').addEventListener('click', cancelDrawing);

    // Called from the detection popup buttons (inline onclick)
    window.annotDelete = startDelete;
    window.annotModify = startModify;

    renderCounter();
    if (sb) {
        sb.rpc('annotation_count')
          .then(({ data, error }) => { if (!error) { totalCount = data; renderCounter(); } });
    }
}

// ─── Delete (false positive) ──────────────────────────────────────────────────

function startDelete() {
    const ctx = S.lastClickedDetection;
    if (!ctx) return;
    S.map.closePopup();
    openForm({ action: 'delete', feature: ctx.feature },
        'Report: not a PV system',
        'This polygon will disappear from your view and be submitted for review.');
}

// ─── Modify (redraw the outline) ──────────────────────────────────────────────

function startModify() {
    const ctx = S.lastClickedDetection;
    if (!ctx) return;
    S.map.closePopup();
    if (!S.map.pm) { toast('Drawing unavailable (geoman not loaded)'); return; }
    if (S.drawing) return;

    pending = { action: 'modify', feature: ctx.feature };
    // Dim the original so the user traces its replacement over it
    if (ctx.layer.setStyle)
        ctx.layer.setStyle({ opacity: 0.35, fillOpacity: 0.1, dashArray: '4 3' });
    startDrawing('Draw the corrected outline over the dimmed shape — click the first point to close.');
}

// ─── Add (draw a missed installation) ─────────────────────────────────────────

function startAdd() {
    if (!S.map.pm) { toast('Drawing unavailable (geoman not loaded)'); return; }
    if (S.drawing) return;
    pending = { action: 'add' };
    startDrawing('Draw the installation outline — click to place points, click the first point to close.');
}

// ─── Shared drawing machinery ─────────────────────────────────────────────────

function startDrawing(instruction) {
    S.drawing = true;
    suspendNav();                  // nav polygons must not swallow drawing clicks
    $('annot-editbar-msg').textContent = instruction;
    show('annot-editbar');
    S.map.pm.enableDraw('Polygon', { snappable: false, continueDrawing: false });
    S.map.once('pm:create', onDrawn);
}

function onDrawn(e) {
    pending.geometry = e.layer.toGeoJSON().geometry;
    S.map.removeLayer(e.layer);
    S.map.pm.disableDraw();
    hide('annot-editbar');
    if (pending.action === 'add') {
        openForm(pending, 'Add a missed installation',
            'The new polygon will appear on your view and be submitted for review.');
    } else {
        openForm(pending, 'Fix shape',
            'The redrawn outline will replace the original in your view and be submitted for review.');
    }
}

function cancelDrawing() {
    if (S.map.off) S.map.off('pm:create', onDrawn);
    if (S.map.pm) S.map.pm.disableDraw();
    hide('annot-editbar');
    pending = null;
    S.drawing = false;
    rerenderDetections();          // restore the dimmed original, if any
    resumeNav();
}

// ─── Form ─────────────────────────────────────────────────────────────────────

function openForm(p, title, note) {
    pending = p;
    S.drawing = true;              // keep map clicks (target exit, zone select) disabled
    $('annot-form-title').textContent = title;
    $('annot-form-note').textContent = note;
    $('annot-comment').value = '';
    $('annot-surface').value = p.feature?.properties?.surface ?? '';
    $('annot-kwp').value = p.feature?.properties?.kwp_approx ?? '';
    show('annot-form');
}

function closeForm() {
    hide('annot-form');
    pending = null;
    S.drawing = false;
    resumeNav();
}

function cancelForm() {
    const wasModify = pending?.action === 'modify';
    closeForm();
    if (wasModify) rerenderDetections();   // discard: restore the original shape
}

async function submitForm() {
    if (!pending) return;
    const now = Date.now();
    if (sessionCount >= ANNOT_MAX_SESSION) { toast('Session submission limit reached.'); return; }
    if (now - lastSubmitAt < ANNOT_MIN_INTERVAL_MS) { toast('Easy — one submission every few seconds.'); return; }

    const { action, feature, geometry } = pending;
    const surface = parseFloat($('annot-surface').value) || null;
    const kwp     = parseFloat($('annot-kwp').value) || null;

    const record = {
        action,
        target_id:  feature ? featureKey(feature) : null,
        geometry:   geometry || null,
        properties: (surface || kwp) ? { surface, kwp } : null,
        original:   feature ? { id: featureKey(feature), geometry: feature.geometry,
                                properties: feature.properties } : null,
        comment:    $('annot-comment').value.trim().slice(0, 500) || null,
    };

    if (sb) {
        $('annot-submit').disabled = true;
        const { error } = await sb.from('annotations').insert(record);
        $('annot-submit').disabled = false;
        if (error) {
            console.error('Annotation insert failed:', error);
            toast('Submission failed — please retry.');
            return;
        }
        if (totalCount != null) totalCount++;
    }

    applyLocally(record, feature, geometry);
    lastSubmitAt = now;
    sessionCount++;
    renderCounter();
    closeForm();
    toast(sb ? 'Thanks — submitted for review.' : 'Applied locally (backend not configured).');
}

function applyLocally(record, feature, geometry) {
    if (record.action === 'delete') {
        S.edits.deleted.add(featureKey(feature));
        rerenderDetections();
    } else if (record.action === 'modify') {
        S.edits.modified.set(featureKey(feature), geometry);
        rerenderDetections();
    } else {
        const f = { type: 'Feature', geometry, properties: record.properties || {} };
        S.edits.added.push(f);
        addedLayer.addData(f);
    }
}

// ─── Counter + toast ──────────────────────────────────────────────────────────

function renderCounter() {
    const el = $('annot-counter');
    const parts = [`✎ ${fmtInt(sessionCount)} this session`];
    if (totalCount != null) parts.push(`${fmtInt(totalCount)} total`);
    el.textContent = parts.join(' · ');
    el.style.display = 'block';
}

let toastT;
function toast(msg) {
    const el = $('annot-toast');
    el.textContent = msg;
    el.style.display = 'block';
    clearTimeout(toastT);
    toastT = setTimeout(() => { el.style.display = 'none'; }, 3500);
}
