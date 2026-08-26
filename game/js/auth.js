// ─── Identity: anonymous by default, standard email+password auth, Google stub ─
//
// Anonymous is the big/default button — signInAnonymously() gives a real,
// stable auth.uid() with zero fields to fill beyond a pseudo. Google is a
// visible, disabled button for now — see game/README.md, this isn't blocked
// by Google's review process, just sequenced after email.
//
// STANDARD email+password auth — switched from the earlier email-OTP (typed
// 6-digit code) flow after repeated Supabase email-template/deliverability
// problems (wrong template shown, {{ .Token }} missing, codes not arriving
// at all after a user reset). A password login never depends on an email
// arriving, so it can't be broken by a stale template, a rate limit, or a
// spam filter — this is the fix for "I can't receive the token anymore".
//
// Two DIFFERENT operations, on purpose — conflating them was the bug that
// made the old flow flaky, and the same distinction still matters here:
//   - LINK (linkEmailPassword): attaches email+password to the CURRENT
//     session via updateUser(), preserving auth.uid() — every row already
//     written (verifications, profiles) stays attached. Used by the menu's
//     "Connect email" (an anonymous player securing their existing
//     progress).
//   - LOG IN (login): signInWithPassword() signs into whatever account that
//     email+password already belongs to, WHICH REPLACES THE CURRENT
//     SESSION (different auth.uid(), current anonymous progress is not
//     carried over). Used by the landing screen's "Log in" panel — for
//     someone returning to an account they already claimed on another
//     device, not for securing a new one.
//
// IMPORTANT — one-time Supabase dashboard check: Authentication → Providers
// → Email has a "Confirm email" toggle. If it's ON, updateUser({ email })
// inside linkEmailPassword() still sends a confirmation link to the new
// address and the change only takes effect once that's clicked — the exact
// email-deliverability dependency this rewrite is meant to remove. Turn it
// OFF so linking is instant, matching "they can log in right away" from the
// brief. login()/signInWithPassword() never depended on that toggle either
// way — it has no email step at all.

import { S, getSupabase } from './store.js';

/** Re-reads the pseudo (if any) for whatever session is currently active —
 *  shared by ensureSession() (initial load) and login() (switches to a
 *  different, already-claimed account; its pseudo needs re-fetching too —
 *  S.pseudo from the previous session is stale). */
export async function refreshProfile() {
    const sb = getSupabase();
    const { data: profile } = await sb
        .from('profiles')
        .select('pseudo')
        .eq('id', S.session.user.id)
        .maybeSingle();
    S.pseudo = profile?.pseudo ?? null;
    return S.pseudo;
}

/** Ensures a Supabase Auth session exists (anonymous if nothing better yet)
 *  and loads this user's pseudo, if they already picked one. */
export async function ensureSession() {
    const sb = getSupabase();
    if (!sb) throw new Error('Supabase client unavailable (check config.js / CDN script tag)');

    const { data: { session } } = await sb.auth.getSession();
    if (session) {
        S.session = session;
    } else {
        const { data, error } = await sb.auth.signInAnonymously();
        if (error) throw error;
        S.session = data.session;
    }

    await refreshProfile();
    return S.session;
}

/** Attempts to claim a pseudo for the current session. Returns
 *  {ok:true} or {ok:false, reason:'taken'|'invalid'|'error'}. */
export async function claimPseudo(pseudo) {
    const clean = pseudo.trim();
    if (clean.length < 2 || clean.length > 24) return { ok: false, reason: 'invalid' };

    const sb = getSupabase();
    const { error } = await sb.from('profiles').insert({ id: S.session.user.id, pseudo: clean });
    if (error) {
        // Postgres unique_violation
        if (error.code === '23505') return { ok: false, reason: 'taken' };
        console.error('claimPseudo failed:', error);
        return { ok: false, reason: 'error' };
    }
    S.pseudo = clean;
    return { ok: true };
}

/** Renames the current session's pseudo — the menu's pencil icon next to
 *  the name, shown only once an account has an email (see refreshAccountState()
 *  in menu.js). Same validation/uniqueness rules as claimPseudo(), just an
 *  update instead of an insert — see the "authenticated can update own
 *  profile" RLS policy in scripts/verifications_setup.sql. Returns
 *  {ok:true} or {ok:false, reason:'taken'|'invalid'|'error'}. */
export async function editPseudo(pseudo) {
    const clean = pseudo.trim();
    if (clean.length < 2 || clean.length > 24) return { ok: false, reason: 'invalid' };

    const sb = getSupabase();
    const { error } = await sb.from('profiles').update({ pseudo: clean }).eq('id', S.session.user.id);
    if (error) {
        // Postgres unique_violation
        if (error.code === '23505') return { ok: false, reason: 'taken' };
        console.error('editPseudo failed:', error);
        return { ok: false, reason: 'error' };
    }
    S.pseudo = clean;
    return { ok: true };
}

/** LOG IN — signs into whichever account this email+password belongs to.
 *  This REPLACES the current session — see the module comment above. No
 *  email round-trip at all, so this can never be blocked by mail delivery.
 *  Used by the landing screen's "Log in" panel. */
export async function login(email, password) {
    const sb = getSupabase();
    const { data, error } = await sb.auth.signInWithPassword({ email, password });
    if (error) return { ok: false, error };
    S.session = data.session;
    return { ok: true };
}

/** Signs out and clears the local Supabase session — for the debug
 *  "recommencer" button in the menu (and useful later for a real "switch
 *  account" flow too). The next ensureSession() call will create a brand
 *  new anonymous session, so this is how to actually restart the onboarding
 *  from scratch rather than always resuming the same anon account. */
export async function signOut() {
    const sb = getSupabase();
    await sb.auth.signOut();
}

/** LINK — attaches email+password to the CURRENT session via updateUser(),
 *  which preserves auth.uid(): every row already written (verifications,
 *  profiles) stays attached, unlike login() above. This is what the menu's
 *  "Connect email" uses to secure an anonymous player's existing
 *  progress. See the module comment for the "Confirm email" dashboard
 *  setting this depends on for taking effect immediately.
 *  updateUser() returns the updated user, not a new session — S.session.user
 *  is patched in place so refreshAccountState() immediately sees the new
 *  email and is_anonymous:false (this was the bug behind "Delete account"
 *  not appearing right after creating one: S.session was left stale, still
 *  pointing at the old anonymous user). */
export async function linkEmailPassword(email, password) {
    const sb = getSupabase();
    const { data, error } = await sb.auth.updateUser({ email, password });
    if (error) return { ok: false, error };
    if (S.session && data.user) S.session.user = data.user;
    return { ok: true };
}

/** Changes the password on the CURRENT (already-linked) session — the
 *  menu's "Change password", i.e. "edit account" from the brief. Doesn't
 *  need the current password: Supabase authorizes this by the session
 *  itself, the same way it authorizes deleteAccount() below. Same
 *  S.session.user patch as linkEmailPassword(), for consistency. */
export async function changePassword(newPassword) {
    const sb = getSupabase();
    const { data, error } = await sb.auth.updateUser({ password: newPassword });
    if (error) return { ok: false, error };
    if (S.session && data.user) S.session.user = data.user;
    return { ok: true };
}

/** Permanently deletes the current account (see delete_own_account() in
 *  scripts/verifications_setup.sql). Profile/pseudo is removed via its own
 *  `on delete cascade`; verifications already cast stay — their vote
 *  counts still matter for the season and campaign_pool.votes_received has
 *  no compensating decrement for a deleted row — but lose their link to
 *  this user (user_id set null). Signs out locally afterwards regardless,
 *  since the session is invalid either way once the user is gone. */
export async function deleteAccount() {
    const sb = getSupabase();
    const { error } = await sb.rpc('delete_own_account');
    if (error) return { ok: false, error };
    await sb.auth.signOut();
    return { ok: true };
}
