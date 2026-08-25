// ─── Boot ──────────────────────────────────────────────────────────────────

import { S, $, show, hide, getSupabase } from './store.js';
import { ensureSession, claimPseudo, sendEmailCode, verifyEmailCode } from './auth.js';
import { initSwipe } from './swipe.js';
import { initMenu, openMenu } from './menu.js';
import { initHelp } from './help.js';

const ADJECTIVES = ['Solar', 'Sunny', 'Bright', 'Rooftop', 'Golden', 'Amber', 'Swift', 'Keen', 'Sharp', 'Curious'];
const NOUNS = ['Panel', 'Scanner', 'Falcon', 'Otter', 'Fox', 'Hawk', 'Badger', 'Pixel', 'Ranger', 'Scout'];

async function boot() {
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('service-worker.js').catch(() => {});
    }

    wireAuthScreen();
    // Wired here, not inside startGame(): the landing screen's own
    // Leaderboard/Progress links open this same #menu before a pseudo is
    // ever chosen (see wireAuthScreen below), so its internal buttons
    // (close, the two tab switchers, share, claim, signout) need to be live
    // from the start — otherwise the menu would open but its own close
    // button wouldn't work yet. Safe to call this early: it only attaches
    // listeners, nothing in it depends on a game being in progress.
    initMenu();

    try {
        await ensureSession();
    } catch (err) {
        console.error('ensureSession failed:', err);
        $('#auth-error') && ($('#auth-error').textContent = 'Could not connect — try again in a moment.');
        return;
    }

    // Fire-and-forget, now that a session exists (verification_stats() is
    // granted to `authenticated`, which the anonymous session already is).
    refreshLandingStats();

    if (S.pseudo) startGame();
}

function startGame() {
    hide($('#auth-screen'));
    show($('#game-screen'));
    $('#header-pseudo').textContent = S.pseudo;
    initHelp();
    initSwipe();
}

async function refreshLandingStats() {
    const sb = getSupabase();
    if (!sb) return;
    const { data, error } = await sb.rpc('verification_stats');
    if (error) {
        // Left as "—" in the UI, but logged so it's diagnosable from the
        // browser console — most likely causes: scripts/verifications_setup.sql
        // hasn't been (re-)run in Supabase yet (function missing entirely —
        // PostgREST 404 / PGRST202), or the `grant execute ... to anon,
        // authenticated` line is missing/stale (permission denied — 42501).
        console.error('verification_stats RPC failed:', error);
        return;
    }
    if (!data) return;
    $('#landing-count').textContent = (data.count ?? 0).toLocaleString();
    $('#landing-last').textContent = data.last_at ? timeAgo(new Date(data.last_at)) : 'never yet — be the first';
}

function timeAgo(date) {
    const s = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
    if (s < 60) return 'just now';
    const m = Math.round(s / 60);
    if (m < 60) return `${m} min ago`;
    const h = Math.round(m / 60);
    if (h < 24) return `${h}h ago`;
    const d = Math.round(h / 24);
    return `${d}d ago`;
}

function randomPseudo() {
    const a = ADJECTIVES[Math.floor(Math.random() * ADJECTIVES.length)];
    const n = NOUNS[Math.floor(Math.random() * NOUNS.length)];
    const num = Math.floor(Math.random() * 90 + 10);
    return `${a}${n}${num}`;
}

function validPseudo(v) {
    const clean = v.trim();
    return clean.length >= 2 && clean.length <= 24;
}

function updatePlayButtonState() {
    $('#play-btn').disabled = !validPseudo($('#pseudo-input').value);
}

function wireAuthScreen() {
    const input = $('#pseudo-input');
    input.addEventListener('input', updatePlayButtonState);
    input.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !$('#play-btn').disabled) attemptPlay();
    });

    $('#pseudo-random').addEventListener('click', () => {
        input.value = randomPseudo();
        updatePlayButtonState();
        input.focus();
    });

    $('#play-btn').addEventListener('click', attemptPlay);

    // Both work even before a pseudo is chosen — ensureSession() (called
    // from boot()) has already given this tab an anonymous auth session,
    // which is all the leaderboard/progress RPCs need.
    $('#landing-leaderboard-link').addEventListener('click', async e => {
        e.preventDefault();
        await openMenu('leaderboard');
    });
    $('#landing-progress-link').addEventListener('click', async e => {
        e.preventDefault();
        await openMenu('progress');
    });

    $('#auth-email-link').addEventListener('click', () => show($('#email-form')));

    $('#email-send').addEventListener('click', async () => {
        const email = $('#email-input').value.trim();
        if (!email) return;
        const res = await sendEmailCode(email);
        const err = $('#email-error');
        if (!res.ok) { err.textContent = 'Could not send — check the address.'; show(err); return; }
        hide(err);
        show($('#email-otp-step'));
    });

    $('#email-verify').addEventListener('click', async () => {
        const email = $('#email-input').value.trim();
        const token = $('#email-otp-input').value.trim();
        const res = await verifyEmailCode(email, token);
        const err = $('#email-error');
        if (!res.ok) { err.textContent = 'Invalid or expired code.'; show(err); return; }
        // Session is now permanent (email-linked) — still needs a pseudo if
        // this wasn't already an anonymous session that had one.
        if (S.pseudo) startGame();
    });
}

async function attemptPlay() {
    if ($('#play-btn').disabled) return;
    const res = await claimPseudo($('#pseudo-input').value);
    if (res.ok) { startGame(); return; }
    const msg = { taken: 'That name is already taken — try another.',
                  invalid: 'Use 2 to 24 characters.',
                  error: 'Something went wrong — try again.' }[res.reason];
    const err = $('#pseudo-error');
    err.textContent = msg; show(err);
}

boot();
