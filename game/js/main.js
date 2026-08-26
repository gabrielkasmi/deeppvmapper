// ─── Boot ──────────────────────────────────────────────────────────────────

import { S, $, show, hide, getSupabase, toast } from './store.js';
import { ensureSession, claimPseudo, login, refreshProfile } from './auth.js';
import { initSwipe } from './swipe.js';
import { initMenu, openMenu, shareApp } from './menu.js';
import { initHelp } from './help.js';
import { wireInstallButton } from './pwa.js';
import { primeHooks, announceStreak } from './hooks.js';

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

    // Awaited, not fire-and-forget like refreshLandingStats() below: it
    // seeds S.lifetimeTotal/S.streak/S.rankCache (see js/hooks.js), and
    // milestone/rank checks on the very first swipe need that baseline to
    // already be correct — starting from 0 and racing a fetch would risk a
    // false "10 verifications!" firing on an early vote.
    await primeHooks().catch(err => console.error('primeHooks failed:', err));

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
    announceStreak();
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
    $('#landing-share-card').addEventListener('click', shareApp);
    wireInstallButton($('#landing-install-btn'), () => toast('Tap the Share icon, then "Add to Home Screen".', 4500));

    // "What is the purpose of this game?" — same panel, two entry points
    // (this landing link, and the header button wired in help.js), both
    // work regardless of which screen is showing since #purpose-panel is
    // a top-level element, not nested inside #game-screen (see its HTML
    // comment for why that matters — same issue #toast had).
    $('#landing-purpose-link').addEventListener('click', e => { e.preventDefault(); show($('#purpose-panel')); });
    $('#purpose-close').addEventListener('click', () => hide($('#purpose-panel')));

    // Login fully REPLACES the name/Play block (not shown alongside it) —
    // toggle both ways so cancelling out of login goes back to a clean
    // sign-up state instead of showing both at once.
    $('#auth-login-link').addEventListener('click', () => {
        hide($('#pseudo-form'));
        hide($('#auth-login-link'));
        show($('#login-panel'));
    });
    $('#login-cancel').addEventListener('click', () => {
        hide($('#login-panel'));
        show($('#pseudo-form'));
        show($('#auth-login-link'));
    });

    const attemptLogin = async () => {
        const email = $('#email-input').value.trim();
        const password = $('#password-input').value;
        const err = $('#email-error');
        if (!email || !password) { err.textContent = 'Enter your email and password.'; show(err); return; }
        const res = await login(email, password);
        if (!res.ok) { err.textContent = 'Incorrect email or password.'; show(err); return; }
        hide(err);
        // login() SWITCHES to whatever account that email belongs to —
        // S.pseudo from the session we had a moment ago (usually anonymous)
        // is for a different account now and has to be re-fetched, or a
        // real returning-user login would silently do nothing here.
        await refreshProfile();
        hide($('#login-panel'));
        if (S.pseudo) {
            startGame();
        } else {
            // Existing account with no pseudo yet (edge case) — let them
            // pick one instead of leaving the screen looking stuck.
            $('#pseudo-input').focus();
        }
    };
    $('#login-submit').addEventListener('click', attemptLogin);
    $('#password-input').addEventListener('keydown', e => { if (e.key === 'Enter') attemptLogin(); });
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
