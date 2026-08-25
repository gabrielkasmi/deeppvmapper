// ─── Shared state + Supabase client ───────────────────────────────────────

import { SUPABASE_URL, SUPABASE_ANON_KEY } from './config.js';

let sbClient = null;

/** Lazily-created, shared across every module — same pattern as the map's store.js. */
export function getSupabase() {
    if (!sbClient && window.supabase)
        sbClient = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
    return sbClient;
}

export const S = {
    session: null,      // Supabase auth session, once signed in (anon or real)
    pseudo: null,       // this user's chosen pseudo, once set
    queue: [],           // prefetched cards not yet shown
    currentCard: null,   // the card on screen right now
    fetching: false,     // guards against a duplicate prefetch in flight
    // This session's running tally, shown live in the game header — resets
    // on reload (it's just "how am I doing right now", not the source of
    // truth; the menu's lifetime breakdown comes from the DB instead via
    // my_verification_breakdown()). Skip is deliberately NOT tracked here:
    // it isn't a vote, so it shouldn't look like one of the three counters.
    counts: { confirm: 0, reject: 0, ambiguous: 0 },
};

export const $ = sel => document.querySelector(sel);
export const $$ = sel => Array.from(document.querySelectorAll(sel));
export const show = el => { if (el) el.hidden = false; };
export const hide = el => { if (el) el.hidden = true; };

let toastTimer;
export function toast(msg) {
    const el = $('#toast');
    if (!el) return;
    el.textContent = msg;
    show(el);
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => hide(el), 3000);
}
