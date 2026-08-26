// ─── Batch fetching + vote submission (with the short undo window) ────────

import { CAMPAIGN_ID, BATCH_SIZE, PREFETCH_AT } from './config.js';
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

// ─── Submission, undoable until the next decision ─────────────────────────
// Gmail-"undo send" style, but no fixed timer anymore: the decision isn't
// inserted until something else happens — either the next decision (which
// flushes/commits this one first, so undo only ever affects the single
// most recent swipe) or the page being hidden/closed (flushPendingNow, see
// swipe.js). That's deliberately unbounded in time — "go back" stays live
// for as long as you like, right up until you swipe again — while keeping
// the same insert-only/immutable posture as the rest of the schema
// (annotations, issue_reports — see game/README.md): once flushed, there's
// no update/delete path, by design.

let pending = null; // { detection_id, decision, comment, card }

export function recordDecision(card, decision, comment = null) {
    flushPending();
    pending = { detection_id: card.detection_id, decision, comment, card };
}

/** Cancels the pending (not-yet-written) decision and hands the card back
 *  to be shown again. Returns the card, or null if it was already flushed
 *  (i.e. another decision has been made since). */
export function undoLastDecision() {
    if (!pending) return null;
    const { card } = pending;
    pending = null;
    S.queue.unshift(card);
    return card;
}

function flushPending() {
    if (!pending) return;
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
