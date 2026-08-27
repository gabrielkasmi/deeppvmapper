// ─── "Report an issue" — a free-form report about whatever is currently
// selected (a zone) or shown (a clicked installation).
//
// This is deliberately NOT for false positives, missing installations, or
// wrong shapes — annotate.js already covers those with instant, geometry-
// based edits (delete / redraw / add), reviewed the same way. This form is
// for the rest: incorrect or missing attributes on an installation, a
// utility-scale plant mistagged as rooftop, or anything else worth
// flagging that doesn't fit those three actions.
//
// The button stays visible at all times (so the feature is discoverable)
// but is disabled until there's something to report against — an empty
// report with no context isn't useful to anyone. Submissions go to a
// dedicated, insert-only Supabase table (issue_reports) — reviewed offline,
// then promoted to a GitHub issue on gabrielkasmi/openpvmapper-issues when
// confirmed.

import { S, $, show, hide, getSupabase, logEvent } from './store.js';

const CATEGORIES = [
    { value: 'missing_attributes', label: 'Missing or incorrect attributes (surface, tilt, year…)' },
    { value: 'utility_scale',      label: 'Looks like a utility-scale plant, not a rooftop system' },
    { value: 'other',              label: 'Something else' },
];

const MIN_INTERVAL_MS = 2000;

let sb = null;
let lastSubmitAt = 0;

export function initReport() {
    sb = getSupabase();

    const select = $('report-category');
    if (select) select.innerHTML = CATEGORIES.map(c => `<option value="${c.value}">${c.label}</option>`).join('');

    $('report-issue-btn')?.addEventListener('click', openReportForm);
    $('report-cancel')?.addEventListener('click', closeReportForm);
    $('report-submit')?.addEventListener('click', submitReport);

    refreshReportButtonState();
}

/** Call after any change to the current selection or clicked installation
 *  (selection.js / render.js) — keeps the always-visible button's enabled
 *  state in sync with whether there's actually something to report on. */
export function refreshReportButtonState() {
    const btn = $('report-issue-btn');
    if (!btn) return;
    const hasContext = !!(S.lastClickedDetection || S.selectionBounds);
    btn.disabled = !hasContext;
    // data-tip, not title — the topbar buttons use a custom-styled hover
    // tooltip (see data.html's #me-topbar CSS) instead of the native one.
    btn.dataset.tip = hasContext ? 'Report an issue' : 'Select a zone or an installation on the map first';
}

/** Prefers the installation currently shown (if any) over the broader zone
 *  selection underneath it — a clicked installation is the more precise,
 *  more useful thing to report against. */
function currentContext() {
    const installation = S.lastClickedDetection?.feature;
    if (installation) {
        const p = installation.properties || {};
        const id = installation.id ?? p.id;
        return {
            target_type: 'installation',
            target_id: id != null ? String(id) : null,
            target_label: p.insee ? `Installation in ${p.insee}` : 'A specific installation',
            admin: p.insee ? { inseeCodes: [p.insee] } : (p.dpt ? { deptCodes: [p.dpt] } : null),
        };
    }
    if (S.selectionBounds) {
        return {
            target_type: 'zone',
            target_id: null,
            target_label: S.selectionLabel || 'Selected area',
            admin: S.selectionAdmin,
        };
    }
    return null;
}

function buildMapUrl(ctx) {
    const base = `${location.origin}${location.pathname}`;
    if (ctx.admin?.inseeCodes?.length === 1) return `${base}?insee=${ctx.admin.inseeCodes[0]}`;
    if (ctx.admin?.deptCodes?.length === 1)  return `${base}?dept=${ctx.admin.deptCodes[0]}`;
    return location.href;
}

function openReportForm() {
    const ctx = currentContext();
    if (!ctx) return;   // button should already be disabled in this case
    $('report-form-target').textContent = `Reporting: ${ctx.target_label}`;
    $('report-comment').value = '';
    show('report-form');
}

function closeReportForm() {
    hide('report-form');
}

async function submitReport() {
    const ctx = currentContext();
    if (!ctx) { closeReportForm(); return; }

    const now = Date.now();
    if (now - lastSubmitAt < MIN_INTERVAL_MS) { toast('Easy — one report every few seconds.'); return; }

    const comment = $('report-comment').value.trim().slice(0, 1000);
    if (!comment) { toast('Add a short description first.'); return; }
    const category = $('report-category').value;

    const record = {
        category,
        target_type: ctx.target_type,
        target_id: ctx.target_id,
        target_label: ctx.target_label,
        admin: ctx.admin || null,
        map_url: buildMapUrl(ctx),
        comment,
    };

    if (sb) {
        $('report-submit').disabled = true;
        const { error } = await sb.from('issue_reports').insert(record);
        $('report-submit').disabled = false;
        if (error) {
            console.error('Issue report insert failed:', error);
            toast('Submission failed — please retry.');
            return;
        }
        logEvent('report_issue', category);
    }

    lastSubmitAt = now;
    closeReportForm();
    toast(sb ? 'Thanks — noted for review.' : 'Backend not configured — nothing was sent.');
}

let toastT;
function toast(msg) {
    const el = $('report-toast');
    if (!el) return;
    el.textContent = msg;
    el.style.display = 'block';
    clearTimeout(toastT);
    toastT = setTimeout(() => { el.style.display = 'none'; }, 3500);
}
