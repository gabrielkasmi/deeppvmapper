// ─── Gamification hooks: milestones, daily streaks, leaderboard-rank jumps ─
// Deliberately NOT modal popups — the brief was "don't overload the UI with
// popups": milestones and streaks are a small badge that pops up over the
// card and fades itself out (no dismiss, no interaction, pointer-events
// disabled so it never blocks a swipe); the leaderboard-rank banner is the
// same idea, just a wider strip with more text and a longer auto-hide.

import { S, $, getSupabase } from './store.js';

const MILESTONES_FIXED = [5, 10, 30, 50, 100, 200, 500, 1000];
const MILESTONE_STEP = 500; // every +500 past the last fixed one (1000)

function isMilestone(total) {
    return MILESTONES_FIXED.includes(total) || (total > 1000 && total % MILESTONE_STEP === 0);
}

// Checked every 5th lifetime vote, not every single one — a rank only
// moves by whole positions when someone else also votes, so checking on
// every swipe would just be extra RPC calls for (almost always) the same
// answer. Every 5 is frequent enough to feel responsive without hammering
// the DB.
const RANK_CHECK_EVERY = 5;
const RANK_WINDOWS = ['week', 'month', 'all'];
const WINDOW_LABEL = { week: 'weekly', month: 'monthly', all: 'all-time' };

function tierOf(rnk) {
    if (rnk == null) return 0;
    if (rnk === 1) return 3;
    if (rnk <= 3) return 2;
    if (rnk <= 10) return 1;
    return 0;
}

let flashTimer;
function showFlash(text, kind) {
    const el = $('#hook-flash');
    if (!el) return;
    clearTimeout(flashTimer);
    el.textContent = text;
    el.className = kind; // resets any previous kind class too
    el.hidden = false;
    // Reflow so re-triggering the animation on back-to-back flashes restarts it.
    void el.offsetWidth;
    el.classList.add('show');
    flashTimer = setTimeout(() => {
        el.classList.remove('show');
        setTimeout(() => { el.hidden = true; }, 250);
    }, 1800);
}

let bannerTimer;
function showRankBanner(html) {
    const el = $('#rank-banner');
    if (!el) return;
    clearTimeout(bannerTimer);
    el.innerHTML = html;
    el.hidden = false;
    void el.offsetWidth;
    el.classList.add('show');
    bannerTimer = setTimeout(() => {
        el.classList.remove('show');
        setTimeout(() => { el.hidden = true; }, 250);
    }, 4500);
}

/** Called once at boot (after a session exists): seeds the lifetime total,
 *  the streak, and a rank baseline for all three windows — without a
 *  baseline the very first periodic rank check would have nothing to
 *  compare against and could fire a false "you just entered the top 10". */
export async function primeHooks() {
    const sb = getSupabase();
    if (!sb) return;

    const [breakdown, streak, ranks] = await Promise.all([
        sb.rpc('my_verification_breakdown').then(r => r.data).catch(() => null),
        sb.rpc('my_streak').then(r => r.data).catch(() => null),
        Promise.all(RANK_WINDOWS.map(w =>
            sb.rpc('my_leaderboard_rank', { p_window: w }).then(r => r.data?.[0]?.rnk ?? null).catch(() => null)
        )),
    ]);

    S.lifetimeTotal = breakdown?.total ?? 0;
    S.streak = streak ?? 0;
    RANK_WINDOWS.forEach((w, i) => { S.rankCache[w] = ranks[i]; });
}

const STREAK_KEY = 'pv-streak-shown-on';

/** Called once the game screen actually becomes visible (not from
 *  primeHooks() itself, which runs earlier while still on the landing
 *  screen — the flash would fire invisibly there, on a hidden section). */
export function announceStreak() {
    if (S.streak < 2) return; // a 1-day "streak" isn't a return visit yet
    const today = new Date().toISOString().slice(0, 10);
    try {
        if (localStorage.getItem(STREAK_KEY) === today) return;
        localStorage.setItem(STREAK_KEY, today);
    } catch { /* private mode etc — just skip the dedupe, no big deal */ }
    showFlash(`🔥 ${S.streak}-day streak!`, 'streak');
}

/** Call after S.lifetimeTotal has just been incremented for a counted vote. */
export function checkMilestone(total) {
    if (isMilestone(total)) {
        showFlash(`🎉 ${total} verifications!`, 'milestone');
    }
}

/** Call after S.lifetimeTotal has just been incremented for a counted vote.
 *  Throttled — only actually hits the DB every RANK_CHECK_EVERY votes. */
export async function checkRankMaybe(total) {
    if (total % RANK_CHECK_EVERY !== 0) return;
    const sb = getSupabase();
    if (!sb) return;

    const results = await Promise.all(RANK_WINDOWS.map(async w => {
        const { data, error } = await sb.rpc('my_leaderboard_rank', { p_window: w });
        const rnk = error ? null : (data?.[0]?.rnk ?? null);
        return { window: w, rnk };
    }));

    // Pick the single best newly-crossed tier across windows, if any — one
    // banner at a time, never stack three. Ties prefer all-time, then
    // month, then week, as the more prestigious one — hence iterating
    // `results` reversed (RANK_WINDOWS is week/month/all) before falling
    // back to strict `>` so an equal tier never displaces an earlier,
    // higher-priority pick.
    let best = null;
    for (const { window, rnk } of [...results].reverse()) {
        const before = tierOf(S.rankCache[window]);
        const after = tierOf(rnk);
        if (after > before && (!best || after > best.tier)) {
            best = { window, rnk, tier: after };
        }
    }
    results.forEach(({ window, rnk }) => { S.rankCache[window] = rnk; });

    if (!best) return;
    const label = WINDOW_LABEL[best.window];
    const html = best.rnk === 1
        ? `You're <strong>#1</strong> on the ${label} leaderboard! 🏆`
        : `Keep going — you're now <strong>#${best.rnk}</strong> on the ${label} leaderboard.`;
    showRankBanner(html);
}
