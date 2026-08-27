// ─── The swipe screen: card rendering + the four input modes ─────────────
// touch swipe / mouse drag / keyboard / buttons, same convention everywhere:
//   → / swipe right = confirme (c'est un PV)
//   ← / swipe left  = infirme
//   ↑ / swipe up    = incertain (compte quand même comme un vote)
//   ⌫ / bouton dédié = annule le dernier geste, tant que rien d'autre ne
//     s'est passé depuis (pas de fenêtre fixe — voir campaign.js)
// Le skip a disparu : un souci de contenu se règle via "Unsure", un souci
// d'image (cassée, obstruée, illisible) se règle via "Comment" (report).

import { S, $, show, hide, toast, getSupabase } from './store.js';
import { cardImageUrl, preloadImage } from './image.js';
import { fetchNextBatch, maybePrefetch, recordDecision, undoLastDecision, flushPendingNow } from './campaign.js';
import { shouldShowInstallNudge, markInstallNudgeShown, triggerInstall, isIOS } from './pwa.js';
import { checkMilestone, checkRankMaybe } from './hooks.js';

const REPORT_MIN_INTERVAL_MS = 2000;

let dragStartX = null, dragStartY = null, dragging = false;
let lastDecision = null; // 'confirm'|'reject'|'ambiguous', so doUndo() can un-bump the counter
let lastReportAt = 0;

export async function initSwipe() {
    $('#btn-confirm').addEventListener('click', () => decide('confirm'));
    $('#btn-reject').addEventListener('click', () => decide('reject'));
    $('#btn-ambiguous').addEventListener('click', () => decide('ambiguous'));
    $('#btn-comment').addEventListener('click', openReportForm);
    $('#report-send-skip').addEventListener('click', () => submitReport('skip'));
    $('#report-send-stay').addEventListener('click', () => submitReport('stay'));
    $('#report-cancel').addEventListener('click', closeReportForm);
    $('#btn-undo').addEventListener('click', doUndo);
    $('#install-nudge-accept').addEventListener('click', async () => {
        hide($('#install-nudge'));
        const shown = await triggerInstall();
        if (!shown && isIOS()) toast('Tap the Share icon, then "Add to Home Screen".', 4500);
    });
    $('#install-nudge-dismiss').addEventListener('click', () => hide($('#install-nudge')));
    $('#notify-submit').addEventListener('click', () =>
        toast('Coming soon — in the meantime, secure your account from the menu (☰).'));

    document.addEventListener('keydown', onKeydown);
    wireDrag($('#card-current'));
    window.addEventListener('pagehide', flushPendingNow);
    window.addEventListener('visibilitychange', () => { if (document.hidden) flushPendingNow(); });

    if (S.queue.length === 0) await fetchNextBatch();
    hide($('#card-loading'));
    showNextCard();
}

function onKeydown(e) {
    if (!$('#report-form').hidden) return; // typing a report — don't hijack keys
    if (e.key === 'ArrowRight') decide('confirm');
    else if (e.key === 'ArrowLeft') decide('reject');
    else if (e.key === 'ArrowUp') { e.preventDefault(); decide('ambiguous'); }
    else if (e.key === 'Backspace') doUndo();
}

function currentCard() { return S.currentCard || null; }

function showNextCard() {
    maybePrefetch();
    const card = S.queue.shift();
    S.currentCard = card || null;

    if (!card) {
        hide($('#card-current'));
        show($('#card-empty'));
        return;
    }
    hide($('#card-empty'));
    show($('#card-current'));

    const url = cardImageUrl(card.lat, card.lng, card.gsd);
    const img = $('#card-image');
    img.src = url;
    img.alt = '';
    // Card is always framed on the centroid — the marker (see index.html)
    // is a fixed, static square at dead center, no per-card geometry
    // needed. Deliberately NOT drawing the detection polygon here: an
    // outline invites "is this shape right" (a geometry question, already
    // covered by the map's own redraw flow) instead of the actual question
    // this pool exists to answer — "is there a PV installation here at all."
    $('#card-current').style.transform = '';
    $('#card-current').classList.remove('fly-left', 'fly-right', 'fly-up', 'fly-down');
}

function decide(decision) {
    const card = currentCard();
    if (!card) return;
    const flyClass = decision === 'confirm' ? 'fly-right' : decision === 'reject' ? 'fly-left' : 'fly-up';
    animateAway(flyClass);
    recordDecision(card, decision);
    bumpCount(decision);
    afterDecision();
}

function bumpCount(decision) {
    S.counts[decision] = (S.counts[decision] || 0) + 1;
    const el = $(`#sc-${decision}`);
    if (el) el.textContent = S.counts[decision];
    lastDecision = decision;

    const total = S.counts.confirm + S.counts.reject + S.counts.ambiguous;
    if (shouldShowInstallNudge(total)) {
        markInstallNudgeShown();
        show($('#install-nudge'));
    }

    S.lifetimeTotal++;
    checkMilestone(S.lifetimeTotal);
    checkRankMaybe(S.lifetimeTotal);
}

function unbumpLastCount() {
    if (!lastDecision) return;
    S.counts[lastDecision] = Math.max(0, (S.counts[lastDecision] || 0) - 1);
    const el = $(`#sc-${lastDecision}`);
    if (el) el.textContent = S.counts[lastDecision];
    S.lifetimeTotal = Math.max(0, S.lifetimeTotal - 1); // keep it matching the DB after an undo
    lastDecision = null;
}

// ─── Report ("Comment") — a free-form note about the image itself, not a
// vote. Goes to the same insert-only issue_reports table the map's report
// feature uses (scripts/issue_reports_setup.sql), tagged target_type =
// 'card' — needs the matching check-constraint/RLS migration run once in
// Supabase (see that file's latest addition) before this will insert
// successfully. Two ways out once a comment is written: "Send & skip"
// dismisses the card with no vote (the old skip's role for an unusable
// image), "Send & stay" keeps the same card up so the user can still vote
// on it; "Cancel" sends nothing. ─────────────────────────────────────────

function openReportForm() {
    $('#report-comment').value = '';
    show($('#report-form'));
}

function closeReportForm() {
    hide($('#report-form'));
}

async function submitReport(mode) {
    const card = currentCard();
    if (!card) { closeReportForm(); return; }

    const comment = $('#report-comment').value.trim().slice(0, 500);
    if (!comment) { toast('Add a short description first.'); return; }

    const now = Date.now();
    if (now - lastReportAt < REPORT_MIN_INTERVAL_MS) { toast('Easy — one report every few seconds.'); return; }
    lastReportAt = now;

    const sb = getSupabase();
    if (sb) {
        const { error } = await sb.from('issue_reports').insert({
            category: 'image_issue',
            target_type: 'card',
            target_id: card.detection_id,
            target_label: `Verification card ${card.detection_id}`,
            admin: null,
            map_url: location.href,
            comment,
        });
        if (error) console.error('issue report insert failed:', error);
    }

    closeReportForm();
    if (mode === 'skip') {
        animateAway('fly-down');
        afterDecision({ skipUndo: true });
    } else {
        toast('Thanks — noted for review.');
    }
}

function afterDecision({ skipUndo = false } = {}) {
    if (!skipUndo) enableUndoButton();
    setTimeout(showNextCard, 180); // let the fly-away animation read before the next card pops in
}

function animateAway(cls) {
    $('#card-current').classList.add(cls);
}

// ─── Undo ("Go back") — always visible, just enabled/disabled. No fixed
// window: it stays enabled until either the next decision (which commits
// this one — see campaign.js) or an explicit undo clears it. ─────────────

function enableUndoButton() {
    $('#btn-undo').disabled = false;
}

function doUndo() {
    const card = undoLastDecision();
    $('#btn-undo').disabled = true;
    if (!card) return; // already flushed (another decision was made since), nothing to undo
    unbumpLastCount();
    S.queue.unshift(card);
    // put whatever's currently showing back in front of it, so the undone
    // card is next rather than replacing what the user is already looking at
    if (S.currentCard) S.queue.splice(1, 0, S.currentCard);
    showNextCard();
}

// ─── Drag (mouse + touch, via Pointer Events) ─────────────────────────────

function wireDrag(el) {
    el.addEventListener('pointerdown', e => {
        dragging = true;
        dragStartX = e.clientX; dragStartY = e.clientY;
        el.setPointerCapture(e.pointerId);
    });
    el.addEventListener('pointermove', e => {
        if (!dragging) return;
        const dx = e.clientX - dragStartX, dy = e.clientY - dragStartY;
        el.style.transform = `translate(${dx}px, ${dy}px) rotate(${dx / 20}deg)`;
    });
    el.addEventListener('pointerup', e => {
        if (!dragging) return;
        dragging = false;
        const dx = e.clientX - dragStartX, dy = e.clientY - dragStartY;
        const THRESHOLD = 80;
        if (Math.abs(dx) > THRESHOLD && Math.abs(dx) > Math.abs(dy)) decide(dx > 0 ? 'confirm' : 'reject');
        else if (-dy > THRESHOLD && Math.abs(dy) > Math.abs(dx)) decide('ambiguous');
        else el.style.transform = ''; // snap back — not a decisive swipe
    });
}
