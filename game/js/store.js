// ─── Shared state + Supabase client ───────────────────────────────────────

import { SUPABASE_URL, SUPABASE_ANON_KEY } from './config.js';

let sbClient = null;

/** Lazily-created, shared across every module — same pattern as the map's
 *  store.js, EXCEPT for one deliberate difference: a custom auth.storageKey.
 *
 *  The main site (static/js/map/store.js) creates its Supabase client with
 *  the SAME URL + anon key, no storageKey override — so supabase-js falls
 *  back to its default, which is derived from the project ref alone, not
 *  the page/path. localStorage is scoped per ORIGIN, not per path, so the
 *  main site's client and this one were reading/writing the exact same key.
 *  Any session this game creates (anonymous sign-in, or a real email
 *  login) was leaking into the main map's client too — which then sent
 *  every request as `authenticated` instead of `anon`, and broke the main
 *  site's annotation submissions (`annotations`' insert policy is `to
 *  anon` only, so an authenticated request has no matching policy and gets
 *  rejected with 42501). A distinct storageKey here keeps this game's
 *  sessions in their own bucket, so the main site's client goes back to
 *  never seeing one. */
export function getSupabase() {
    if (!sbClient && window.supabase)
        sbClient = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
            auth: { storageKey: 'sb-pvcheck-auth-token' },
        });
    return sbClient;
}

export const S = {
    session: null,      // Supabase auth session, once signed in (anon or real)
    pseudo: null,       // this user's chosen pseudo, once set
    queue: [],           // prefetched cards not yet shown
    currentCard: null,   // the card on screen right now
    fetching: false,     // guards against a duplicate prefetch in flight
    // detection_ids already queued/shown this tab session, most-recent
    // last — sent back to get_verification_batch as p_exclude_ids so a
    // card doesn't resurface in the very next prefetch just because its
    // vote is still sitting in campaign.js's `pending` slot and hasn't
    // been written to `verifications` yet. See markSeen() below for the
    // insertion/eviction policy. Purely a client-side top-up: the server
    // is still the source of truth via `verifications`, this only closes
    // the timing gap before a vote lands there.
    seenIds: new Set(),
    // This session's running tally, shown live in the game header — resets
    // on reload (it's just "how am I doing right now", not the source of
    // truth; the menu's lifetime breakdown comes from the DB instead via
    // my_verification_breakdown()). Skip is deliberately NOT tracked here:
    // it isn't a vote, so it shouldn't look like one of the three counters.
    counts: { confirm: 0, reject: 0, ambiguous: 0 },
    // Gamification state (see js/hooks.js) — seeded once at boot from the
    // DB (my_verification_breakdown / my_streak / my_leaderboard_rank),
    // then lifetimeTotal is bumped locally per vote so milestone checks
    // don't need a DB round trip on every single swipe.
    lifetimeTotal: 0,
    streak: 0,
    rankCache: { week: null, month: null, all: null },
};

// Cap on S.seenIds — bounds the p_exclude_ids payload sent on every fetch
// instead of letting it grow for the whole length of a long session. 500
// is comfortably above BATCH_SIZE/PREFETCH_AT (see config.js), so nothing
// still "in flight" (queued or just decided) ever ages out.
const MAX_SEEN_IDS = 500;

/** Records a detection_id as already served this session (queued or shown)
 *  so it isn't handed out again — see the comment on S.seenIds. Evicts the
 *  oldest entry first (Set preserves insertion order) once over the cap. */
export function markSeen(detectionId) {
    S.seenIds.delete(detectionId); // re-insert at the end == mark as freshest
    S.seenIds.add(detectionId);
    while (S.seenIds.size > MAX_SEEN_IDS) {
        S.seenIds.delete(S.seenIds.values().next().value);
    }
}

export const $ = sel => document.querySelector(sel);
export const $$ = sel => Array.from(document.querySelectorAll(sel));
export const show = el => { if (el) el.hidden = false; };
export const hide = el => { if (el) el.hidden = true; };

let toastTimer;
export function toast(msg, ms = 3000) {
    const el = $('#toast');
    if (!el) return;
    el.textContent = msg;
    show(el);
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => hide(el), ms);
}
