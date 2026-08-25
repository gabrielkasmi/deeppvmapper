// ─── Identity: anonymous by default, email OTP for cross-device, Google stub ─
//
// Anonymous is the big/default button — signInAnonymously() gives a real,
// stable auth.uid() with zero fields to fill beyond a pseudo. Email is the
// smaller "keep my score on other devices" option, using a 6-digit code
// typed back into the app (not a clicked link — a link risks reopening a
// browser tab instead of this installed PWA and losing the session). Google
// is a visible, disabled button for now — see game/README.md, this isn't
// blocked by Google's review process, just sequenced after email.
//
// Claiming: linkIdentity() converts an anonymous session into a permanent
// one (email now, Google later) WITHOUT changing auth.uid() — every row
// already written (verifications, profiles) stays attached automatically.

import { S, getSupabase } from './store.js';

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

    const { data: profile } = await sb
        .from('profiles')
        .select('pseudo')
        .eq('id', S.session.user.id)
        .maybeSingle();
    S.pseudo = profile?.pseudo ?? null;
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

/** Step 1 of the email claim/login: send a 6-digit code. */
export async function sendEmailCode(email) {
    const sb = getSupabase();
    const { error } = await sb.auth.signInWithOtp({ email, options: { shouldCreateUser: true } });
    if (error) return { ok: false, error };
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

/** Step 2: verify the code. If the current session was anonymous, this
 *  links the email identity in place (same auth.uid(), history preserved)
 *  rather than starting a fresh account — Supabase does this automatically
 *  when verifyOtp is called from an existing anonymous session. */
export async function verifyEmailCode(email, token) {
    const sb = getSupabase();
    const { data, error } = await sb.auth.verifyOtp({ email, token, type: 'email' });
    if (error) return { ok: false, error };
    S.session = data.session;
    return { ok: true };
}
