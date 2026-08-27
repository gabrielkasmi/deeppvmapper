// ─── Menu: my stats (with a confirm/reject/unsure breakdown) / leaderboard / % progress ─

import { CAMPAIGN_ID } from './config.js';
import { $, $$, show, hide, getSupabase, S, toast } from './store.js';
import { editPseudo, linkEmailPassword, changePassword, signOut, deleteAccount } from './auth.js';
import { renderDeptMap } from './deptmap.js';
import { wireInstallButton } from './pwa.js';

let currentWindow = 'all';
let currentPanel = 'leaderboard';

export function initMenu() {
    // Not `openMenu` directly — that would pass the click Event as `panel`,
    // which fails both === 'leaderboard'/'progress' checks in switchPanel()
    // and leaves BOTH panels hidden until a tab is tapped manually.
    $('#menu-open').addEventListener('click', () => openMenu());
    $('#menu-close').addEventListener('click', () => hide($('#menu')));
    $$('#menu-main-tabs button').forEach(btn =>
        btn.addEventListener('click', () => switchPanel(btn.dataset.panel)));
    $$('#menu-tabs button').forEach(btn =>
        btn.addEventListener('click', () => { currentWindow = btn.dataset.window; refreshLeaderboard(); }));
    $('#menu-share').addEventListener('click', shareApp);
    wireInstallButton($('#menu-install-btn'), () => toast('Tap the Share icon, then "Add to Home Screen".', 4500));

    // Rename — only reachable once an account has an email (the pencil
    // icon stays hidden for anonymous players, see refreshAccountState()).
    $('#menu-edit-pseudo').addEventListener('click', () => {
        $('#menu-pseudo-input').value = S.pseudo ?? '';
        show($('#menu-pseudo-form'));
        $('#menu-pseudo-input').focus();
    });
    $('#menu-pseudo-cancel').addEventListener('click', () => hide($('#menu-pseudo-form')));
    $('#menu-pseudo-submit').addEventListener('click', submitPseudoEdit);
    $('#menu-pseudo-input').addEventListener('keydown', e => { if (e.key === 'Enter') submitPseudoEdit(); });

    $('#menu-claim').addEventListener('click', () => {
        hide($('#menu-password-form'));
        resetClaimForm();
        show($('#menu-claim-form'));
        $('#menu-claim-email').focus();
    });
    $('#menu-claim-cancel').addEventListener('click', () => hide($('#menu-claim-form')));
    $('#menu-claim-submit').addEventListener('click', claimByEmail);
    $('#menu-claim-password').addEventListener('keydown', e => { if (e.key === 'Enter') claimByEmail(); });
    // Gates "Create account" on the privacy-notice checkbox (see index.html)
    // — the button starts disabled and only the checkbox flips it, so
    // someone tabbing straight to Enter on the password field still has to
    // have ticked it first (claimByEmail() re-checks it too, since Enter
    // calls it directly and doesn't go through the button's disabled state).
    $('#menu-claim-consent').addEventListener('change', e => {
        $('#menu-claim-submit').disabled = !e.target.checked;
    });

    $('#menu-change-password').addEventListener('click', () => { hide($('#menu-claim-form')); show($('#menu-password-form')); $('#menu-new-password').focus(); });
    $('#menu-password-cancel').addEventListener('click', () => hide($('#menu-password-form')));
    $('#menu-password-submit').addEventListener('click', submitPasswordChange);
    $('#menu-new-password').addEventListener('keydown', e => { if (e.key === 'Enter') submitPasswordChange(); });

    $('#menu-signout').addEventListener('click', async () => {
        if (!window.confirm('Logging out clears the local session (name, client-side history) — votes you already submitted stay saved. Continue?')) return;
        await signOut();
        location.reload();
    });
    $('#menu-delete-account').addEventListener('click', async () => {
        if (!window.confirm('Delete your account permanently? Your name and email are removed — this cannot be undone. Verifications you already submitted stay counted toward the season, just no longer linked to you.')) return;
        const res = await deleteAccount();
        if (!res.ok) { toast('Could not delete — try again.'); return; }
        location.reload();
    });
}

// Rename — the menu's pencil icon next to the name. Gated to accounts with
// an email (see refreshAccountState()): an anonymous player's pseudo is
// already editable once by claiming a different one at signup, and gating
// this keeps the "one name, tied to an email you can find again" identity
// story simple for v1.
async function submitPseudoEdit() {
    const pseudo = $('#menu-pseudo-input').value;
    const err = $('#menu-pseudo-error');
    const res = await editPseudo(pseudo);
    if (res.ok) {
        hide(err);
        hide($('#menu-pseudo-form'));
        $('#menu-pseudo').textContent = S.pseudo;
        toast('Name updated.');
    } else {
        const msg = { taken: 'That name is already taken — try another.',
                      invalid: 'Use 2 to 24 characters.',
                      error: 'Something went wrong — try again.' }[res.reason];
        err.textContent = msg;
        show(err);
    }
}

// Inline (not prompt-based) claim flow — one email + one password field
// shown right in the account section. Uses linkEmailPassword() (updateUser
// with email+password), NOT login() — login() signs into whichever account
// the email already belongs to, replacing the current session; this one
// attaches the email+password to the CURRENT session instead, so an
// anonymous player's existing votes/pseudo stay theirs. See the auth.js
// module comment for the full LINK vs LOG IN distinction, and for the
// Supabase "Confirm email" dashboard setting this depends on to apply
// instantly instead of waiting on a confirmation email.
function resetClaimForm() {
    $('#menu-claim-email').value = '';
    $('#menu-claim-password').value = '';
    $('#menu-claim-consent').checked = false;
    $('#menu-claim-submit').disabled = true;
    hide($('#menu-claim-error'));
}

async function claimByEmail() {
    const email = $('#menu-claim-email').value.trim();
    const password = $('#menu-claim-password').value;
    const err = $('#menu-claim-error');
    if (!email || password.length < 6) {
        err.textContent = 'Enter an email and a password of at least 6 characters.';
        show(err);
        return;
    }
    // Belt-and-braces: the button is disabled until this is checked, but
    // Enter on the password field calls claimByEmail() directly, bypassing
    // that disabled state.
    if (!$('#menu-claim-consent').checked) {
        err.textContent = 'Please confirm you have read the privacy notice.';
        show(err);
        return;
    }
    const res = await linkEmailPassword(email, password);
    if (res.ok) {
        hide(err);
        hide($('#menu-claim-form'));
        toast('Account created — log in with this email+password on any device.');
        refreshAccountState();
    } else {
        err.textContent = res.error?.message || 'Could not secure the account — try again.';
        show(err);
    }
}

// "Edit account" (password side) — changes the password on the already-
// linked CURRENT session, no re-entry of the old one needed (the session
// itself is the authorization, same as deleteAccount()).
async function submitPasswordChange() {
    const password = $('#menu-new-password').value;
    const err = $('#menu-password-error');
    if (password.length < 6) {
        err.textContent = 'Use at least 6 characters.';
        show(err);
        return;
    }
    const res = await changePassword(password);
    if (res.ok) {
        hide(err);
        $('#menu-new-password').value = '';
        hide($('#menu-password-form'));
        toast('Password updated.');
    } else {
        err.textContent = res.error?.message || 'Could not update — try again.';
        show(err);
    }
}

// Toggles the account section between "anonymous" (Connect-email visible,
// no email/member-since shown, no delete) and "has an email" (email +
// member-since shown, Connect-email hidden — already done, Change-password
// + Delete visible instead). Called on menu open and right after a
// successful claimByEmail(), since S.session just changed.
function refreshAccountState() {
    const email = S.session?.user?.email || '';
    const isAnonymous = S.session?.user?.is_anonymous !== false;
    $('#menu-email-line').hidden = !email;
    if (email) $('#menu-email').textContent = email;
    $('#menu-member-since').hidden = !email;
    if (email && S.session?.user?.created_at) {
        $('#menu-since-date').textContent = new Date(S.session.user.created_at)
            .toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
    }
    $('#menu-edit-pseudo').hidden = !email;
    $('#menu-claim').hidden = !!email;
    $('#menu-change-password').hidden = !email;
    $('#menu-delete-account').hidden = isAnonymous;
    if (!email) { hide($('#menu-password-form')); hide($('#menu-pseudo-form')); }
    if (email) hide($('#menu-claim-form'));
}

// Web Share API opens the device's native share sheet (Messages, WhatsApp,
// X/Twitter, etc. — whatever the person has) on mobile and most modern
// desktop browsers; where it isn't available (some older/desktop browsers)
// this falls back to copying the link to the clipboard instead.
export async function shareApp() {
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
    refreshAccountState();
    refreshStats();
    refreshLeaderboard();
    refreshMyRank();
    refreshCompletion();
    refreshDeptProgress();
}

// One line per window (day/week/all) — my_leaderboard_rank() sees every
// player, not just the top 100 leaderboard() returns, so this stays
// accurate even for someone outside that range. Works identically whether
// the account is anonymous or has an email (both are just `authenticated`
// sessions distinguished by auth.uid()). Fetches all three windows
// up front, independent of the leaderboard tab currently showing, so
// switching tabs doesn't need a re-fetch.
async function refreshMyRank() {
    const sb = getSupabase();
    const targets = { day: '#menu-rank-day', week: '#menu-rank-week', all: '#menu-rank-all' };
    await Promise.all(Object.entries(targets).map(async ([window, sel]) => {
        const { data, error } = await sb.rpc('my_leaderboard_rank', { p_window: window });
        const row = data?.[0];
        $(sel).textContent = (error || !row) ? '—' : `#${row.rnk}`;
    }));
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

const WINDOW_LABELS = { day: 'today', week: 'this week', all: 'all time' };

async function refreshLeaderboard() {
    $$('#menu-tabs button').forEach(b => b.classList.toggle('active', b.dataset.window === currentWindow));
    const sb = getSupabase();
    const [{ data, error }, { data: total, error: totalError }] = await Promise.all([
        sb.rpc('leaderboard', { p_window: currentWindow, p_limit: 100 }),
        sb.rpc('leaderboard_total', { p_window: currentWindow }),
    ]);

    const totalEl = $('#menu-leaderboard-total');
    if (totalError || total == null) {
        totalEl.textContent = '';
    } else {
        totalEl.textContent = `${Number(total).toLocaleString()} validations ${WINDOW_LABELS[currentWindow]}`;
    }

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

// season_completion() splits into two deliberately different numbers (see
// the comment above it in scripts/verifications_setup.sql):
//   - data.votes_cast_total is the headline — the honest, unscoped, all-time
//     count of every vote cast this season. That's the number people
//     actually want to see ("N validations done"), not a fragment of it.
//   - data.pct stays scoped to the single active batch so the progress BAR
//     still moves at a readable pace instead of crawling against the full
//     ~655k season — the batching only changes serving order/display
//     pacing, never the real per-installation vote target. The batch
//     bookkeeping itself (batch_no/batch_count/batch_votes_cast/
//     batch_votes_target) isn't shown here anymore — it's plumbing, not
//     something a player needs to see to understand "how close are we."
async function refreshCompletion() {
    const sb = getSupabase();
    const { data, error } = await sb.rpc('season_completion', { p_campaign_id: CAMPAIGN_ID });
    if (error || !data) return;
    $('#menu-votes-cast').textContent = data.votes_cast_total.toLocaleString();
    $('#menu-pct').textContent = `${data.pct}%`;
    $('#menu-pct-bar').style.width = `${data.pct}%`;
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
