// ─── Menu: my stats (with a confirm/reject/unsure breakdown) / leaderboard / % progress ─

import { CAMPAIGN_ID } from './config.js';
import { $, $$, show, hide, getSupabase, S, toast } from './store.js';
import { sendEmailCode, verifyEmailCode, signOut } from './auth.js';
import { renderDeptMap } from './deptmap.js';

let currentWindow = 'all';
let currentPanel = 'leaderboard';

export function initMenu() {
    $('#menu-open').addEventListener('click', openMenu);
    $('#menu-close').addEventListener('click', () => hide($('#menu')));
    $$('#menu-main-tabs button').forEach(btn =>
        btn.addEventListener('click', () => switchPanel(btn.dataset.panel)));
    $$('#menu-tabs button').forEach(btn =>
        btn.addEventListener('click', () => { currentWindow = btn.dataset.window; refreshLeaderboard(); }));
    $('#menu-share').addEventListener('click', shareApp);
    $('#menu-claim').addEventListener('click', claimByEmail);
    $('#menu-signout').addEventListener('click', async () => {
        if (!window.confirm('Starting over clears the local session (name, client-side history) — votes you already submitted stay saved. Continue?')) return;
        await signOut();
        location.reload();
    });
}

// Minimal (prompt-based) claim flow — functional, not polished. Fine for a
// v1 tucked in the menu; worth a real inline form once the core loop is
// validated. Reuses the exact same sendEmailCode/verifyEmailCode as the
// auth screen, so linkIdentity() applies the same way either place.
async function claimByEmail() {
    const email = window.prompt('Your email, to find your account elsewhere:');
    if (!email) return;
    const sent = await sendEmailCode(email);
    if (!sent.ok) { toast('Could not send — check the address.'); return; }
    const code = window.prompt('Code received by email:');
    if (!code) return;
    const res = await verifyEmailCode(email, code);
    toast(res.ok ? 'Account secured — you can find it again with this email.' : 'Invalid or expired code.');
}

// Web Share API opens the device's native share sheet (Messages, WhatsApp,
// X/Twitter, etc. — whatever the person has) on mobile and most modern
// desktop browsers; where it isn't available (some older/desktop browsers)
// this falls back to copying the link to the clipboard instead.
async function shareApp() {
    const shareData = {
        title: 'PV Check',
        text: 'Help verify solar panel installations across France, one swipe at a time.',
        url: location.origin + location.pathname.replace(/index\.html$/, ''),
    };
    if (navigator.share) {
        try { await navigator.share(shareData); } catch { /* user cancelled the share sheet — not an error */ }
        return;
    }
    try {
        await navigator.clipboard.writeText(shareData.url);
        toast('Link copied — share it however you like!');
    } catch {
        toast(shareData.url);
    }
}

function switchPanel(panel) {
    currentPanel = panel;
    $$('#menu-main-tabs button').forEach(b => b.classList.toggle('active', b.dataset.panel === panel));
    $('#menu-panel-leaderboard').hidden = panel !== 'leaderboard';
    $('#menu-panel-progress').hidden = panel !== 'progress';
}

/** panel: optional — 'leaderboard' | 'progress'. Lets the landing screen's
 *  two links (reachable before a pseudo is chosen — the anonymous session
 *  from ensureSession() already satisfies `to authenticated`, which is all
 *  these RPCs require) open the menu straight to one or the other. */
export async function openMenu(panel) {
    if (panel) currentPanel = panel;
    show($('#menu'));
    switchPanel(currentPanel);
    refreshStats();
    refreshLeaderboard();
    refreshCompletion();
    refreshDeptProgress();
}

async function refreshStats() {
    const sb = getSupabase();
    const { data, error } = await sb.rpc('my_verification_breakdown');
    $('#menu-pseudo').textContent = S.pseudo ?? '';
    if (error || !data) {
        $('#menu-mycount').textContent = '—';
        return;
    }
    $('#menu-mycount').textContent = data.total ?? 0;
    $('#menu-bd-confirm').textContent = data.confirm ?? 0;
    $('#menu-bd-reject').textContent = data.reject ?? 0;
    $('#menu-bd-ambiguous').textContent = data.ambiguous ?? 0;
}

async function refreshLeaderboard() {
    $$('#menu-tabs button').forEach(b => b.classList.toggle('active', b.dataset.window === currentWindow));
    const sb = getSupabase();
    const { data, error } = await sb.rpc('leaderboard', { p_window: currentWindow, p_limit: 20 });
    const list = $('#menu-leaderboard');
    list.innerHTML = '';
    if (error || !data?.length) {
        list.innerHTML = '<li class="empty">No one yet — be the first.</li>';
        return;
    }
    data.forEach((row, i) => {
        const li = document.createElement('li');
        li.className = row.pseudo === S.pseudo ? 'me' : '';
        li.innerHTML = `<span class="rank">${i + 1}</span><span class="pseudo">${escapeHtml(row.pseudo)}</span><span class="total">${row.total}</span>`;
        list.appendChild(li);
    });
}

// season_completion() reports real progress scoped to the single active
// batch (see the "Batching" note above campaign_pool.batch_no in
// scripts/verifications_setup.sql) — same real 10-votes-per-installation
// target the scheduler uses, just measured against the ~65k installations
// currently in play rather than all ~655k, so this actually moves instead
// of reading ~0% for weeks. Leads with the raw count (data.votes_cast) —
// more concrete than a percentage of a number nobody has an intuition for.
async function refreshCompletion() {
    const sb = getSupabase();
    const { data, error } = await sb.rpc('season_completion', { p_campaign_id: CAMPAIGN_ID });
    if (error || !data) return;
    $('#menu-votes-cast').textContent = data.votes_cast.toLocaleString();
    $('#menu-pct').textContent = `${data.pct}%`;
    $('#menu-pct-bar').style.width = `${data.pct}%`;
    $('#menu-batch-no').textContent = data.batch_no;
    $('#menu-batch-count').textContent = data.batch_count;
    $('#menu-pct-detail').textContent = `${data.votes_cast.toLocaleString()} / ${data.votes_target.toLocaleString()} votes`;
}

// Feeds the Progress tab's choropleth (game/js/deptmap.js) — colored by
// each département's share of all votes cast so far, not by its own
// completion %. See the comment on season_progress_by_department() in
// scripts/verifications_setup.sql for why those are two different numbers,
// and for why this stays view-only (no click-to-filter-by-département).
async function refreshDeptProgress() {
    const sb = getSupabase();
    const { data, error } = await sb.rpc('season_progress_by_department', { p_campaign_id: CAMPAIGN_ID });
    const container = $('#menu-dept-map');
    if (error) {
        console.error('season_progress_by_department RPC failed:', error);
        container.innerHTML = '<p class="menu-dept-error">Map data unavailable right now — try again in a moment.</p>';
        return;
    }
    renderDeptMap(container, data || []);
}

function escapeHtml(s) {
    return s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
