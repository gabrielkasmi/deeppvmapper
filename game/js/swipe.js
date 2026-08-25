// ─── The swipe screen: card rendering + the three input modes ────────────
// touch swipe / mouse drag / keyboard, same convention everywhere:
//   → / swipe right  = confirme (c'est un PV)
//   ← / swipe left   = infirme
//   ↓ / bouton dédié = passe sans juger (pas un vote — voir campaign.js)
//   ⌫ / bouton dédié = annule le dernier geste (fenêtre courte, voir campaign.js)

import { UNDO_WINDOW_MS } from './config.js';
import { S, $, show, hide, toast } from './store.js';
import { cardImageUrl, preloadImage } from './image.js';
import { fetchNextBatch, maybePrefetch, recordDecision, undoLastDecision, flushPendingNow } from './campaign.js';

let dragStartX = null, dragStartY = null, dragging = false;
let undoTimer = null;
let lastDecision = null; // 'confirm'|'reject'|'ambiguous', so doUndo() can un-bump the counter

export async function initSwipe() {
    $('#btn-confirm').addEventListener('click', () => decide('confirm'));
    $('#btn-reject').addEventListener('click', () => decide('reject'));
    $('#btn-skip').addEventListener('click', skip);
    $('#btn-ambiguous').addEventListener('click', () => show($('#ambiguous-form')));
    $('#ambiguous-submit').addEventListener('click', submitAmbiguous);
    $('#ambiguous-cancel').addEventListener('click', () => hide($('#ambiguous-form')));
    $('#btn-undo').addEventListener('click', doUndo);
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
    if (!$('#ambiguous-form').hidden) return; // typing a comment — don't hijack keys
    if (e.key === 'ArrowRight') decide('confirm');
    else if (e.key === 'ArrowLeft') decide('reject');
    else if (e.key === 'ArrowDown' || e.key === ' ') { e.preventDefault(); skip(); }
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
    // is a fixed, static crosshair at dead center, no per-card geometry
    // needed. Deliberately NOT drawing the detection polygon here: an
    // outline invites "is this shape right" (a geometry question, already
    // covered by the map's own redraw flow) instead of the actual question
    // this pool exists to answer — "is there a PV installation here at all."
    $('#card-current').style.transform = '';
    $('#card-current').classList.remove('fly-left', 'fly-right', 'fly-down');
}

function decide(decision) {
    const card = currentCard();
    if (!card) return;
    animateAway(decision === 'confirm' ? 'fly-right' : 'fly-left');
    recordDecision(card, decision);
    bumpCount(decision);
    afterDecision();
}

function bumpCount(decision) {
    S.counts[decision] = (S.counts[decision] || 0) + 1;
    const el = $(`#sc-${decision}`);
    if (el) el.textContent = S.counts[decision];
    lastDecision = decision;
}

function unbumpLastCount() {
    if (!lastDecision) return;
    S.counts[lastDecision] = Math.max(0, (S.counts[lastDecision] || 0) - 1);
    const el = $(`#sc-${lastDecision}`);
    if (el) el.textContent = S.counts[lastDecision];
    lastDecision = null;
}

function skip() {
    const card = currentCard();
    if (!card) return;
    animateAway('fly-down');
    // No insert — a skip is not a judgement (see campaign.js / README).
    afterDecision();
}

function submitAmbiguous() {
    const card = currentCard();
    if (!card) return;
    const comment = $('#ambiguous-comment').value.trim().slice(0, 500) || null;
    $('#ambiguous-comment').value = '';
    hide($('#ambiguous-form'));
    animateAway('fly-left');
    recordDecision(card, 'ambiguous', comment);
    bumpCount('ambiguous');
    afterDecision();
}

function afterDecision() {
    showUndoButton();
    setTimeout(showNextCard, 180); // let the fly-away animation read before the next card pops in
}

function animateAway(cls) {
    $('#card-current').classList.add(cls);
}

function showUndoButton() {
    clearTimeout(undoTimer);
    show($('#btn-undo'));
    undoTimer = setTimeout(() => hide($('#btn-undo')), UNDO_WINDOW_MS);
}

function doUndo() {
    clearTimeout(undoTimer);
    hide($('#btn-undo'));
    const card = undoLastDecision();
    if (!card) return; // window already closed, nothing to undo
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
        else if (dy > THRESHOLD && Math.abs(dy) > Math.abs(dx)) skip();
        else el.style.transform = ''; // snap back — not a decisive swipe
    });
}
