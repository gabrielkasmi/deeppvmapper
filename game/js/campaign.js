// ─── Batch fetching + vote submission (with the short undo window) ────────

import { CAMPAIGN_ID, BATCH_SIZE, PREFETCH_AT, UNDO_WINDOW_MS } from './config.js';
import { S, getSupabase } from './store.js';
import { cardImageUrl, preloadImage } from './image.js';

/** Fetches the next batch (least-covered items first, server-side) and
 *  appends it to S.queue. Fires the image preloads immediately so they're
 *  usually already cached by the time each card is actually shown. */
export async function fetchNextBatch() {
    if (S.fetching) return [];
    S.fetching = true;
    try {
        const sb = getSupabase();
        const { data, error } = await sb.rpc('get_verification_batch', {
            p_campaign_id: CAMPAIGN_ID, p_limit: BATCH_SIZE,
        });
        if (error) { console.error('fetchNextBatch failed:', error); return []; }
        const cards = data || [];
        cards.forEach(c => preloadImage(cardImageUrl(c.lat, c.lng, c.gsd)));
        S.queue.push(...cards);
        return cards;
    } finally {
        S.fetching = false;
    }
}

/** Call after showing a card — tops the queue back up before the user runs out. */
export function maybePrefetch() {
    if (S.queue.length <= PREFETCH_AT && !S.fetching) fetchNextBatch();
}

// ─── Submission, with a short optimistic delay before it's actually written ─
// Gmail-"undo send" style: the decision isn't inserted until UNDO_WINDOW_MS
// has passed with nothing else happening. Only one decision is ever
// "pending" at a time — making a new one flushes (commits) the previous
// one first, so the undo button only ever affects the most recent swipe.
// Chosen over client-side batching of several votes: batching risks losing
// everything if the tab closes before a flush, and breaks the insert-only/
// immutable posture the rest of the schema already uses (annotations,
// issue_reports — see game/README.md).

let pending = null; // { detection_id, decision, comment, card, timer }

export function recordDecision(card, decision, comment = null) {
    flushPending();
    pending = {
        detection_id: card.detection_id, decision, comment, card,
        timer: setTimeout(flushPending, UNDO_WINDOW_MS),
    };
}

/** Cancels the pending (not-yet-written) decision and hands the card back
 *  to be shown again. Returns the card, or null if the window already closed. */
export function undoLastDecision() {
    if (!pending) return null;
    clearTimeout(pending.timer);
    const { card } = pending;
    pending = null;
    S.queue.unshift(card);
    return card;
}

function flushPending() {
    if (!pending) return;
    clearTimeout(pending.timer);
    const { detection_id, decision, comment } = pending;
    pending = null;
    insertVerification(detection_id, decision, comment);
}

/** Call on page hide/unload so a decision made just before closing isn't lost. */
export function flushPendingNow() { flushPending(); }

async function insertVerification(detection_id, decision, comment) {
    const sb = getSupabase();
    const { error } = await sb.from('verifications').insert({
        user_id: S.session.user.id,
        campaign_id: CAMPAIGN_ID,
        detection_id, decision, comment,
    });
    if (error) console.error('verification insert failed:', error);
}
