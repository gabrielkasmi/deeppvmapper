// ─── PWA install prompt ("Add to Home Screen") ────────────────────────────
// Chrome/Edge/Android fire `beforeinstallprompt` — we capture it so our own
// buttons can trigger the native install prompt instead of relying on the
// browser's own (easy-to-miss) mini-infobar. iOS Safari never fires that
// event — there's no programmatic install API there — so on iOS we treat
// the platform as "installable" too, but triggerInstall() can't show a
// native prompt there; callers get `false` back and should follow up with
// their own "tap Share, then Add to Home Screen" instructions.

let deferredPrompt = null;
const listeners = [];

export function isStandalone() {
    return window.matchMedia('(display-mode: standalone)').matches
        || window.navigator.standalone === true; // iOS Safari's own flag
}

export function isIOS() {
    return /iphone|ipad|ipod/i.test(navigator.userAgent) && !window.MSStream;
}

export function canInstall() {
    return !isStandalone() && (deferredPrompt != null || isIOS());
}

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    listeners.forEach(fn => fn());
});
window.addEventListener('appinstalled', () => {
    deferredPrompt = null;
    listeners.forEach(fn => fn());
});

/** Triggers the native install prompt when one is available (Chrome, Edge,
 *  Android, desktop Chrome). Returns true if a native prompt was actually
 *  shown, false otherwise (iOS, or nothing captured yet) — callers on iOS
 *  should follow up with their own manual instructions. */
export async function triggerInstall() {
    if (!deferredPrompt) return false;
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    if (outcome === 'accepted') deferredPrompt = null;
    listeners.forEach(fn => fn());
    return true;
}

/** Wires a persistent button: visible only while installable, hidden once
 *  already installed (or on browsers that never offer it). Click triggers
 *  the native prompt, or falls back to onIOSInstructions() when there's no
 *  native prompt to show. */
export function wireInstallButton(btn, onIOSInstructions) {
    if (!btn) return;
    const refresh = () => { btn.hidden = !canInstall(); };
    listeners.push(refresh);
    refresh();
    btn.addEventListener('click', async () => {
        const shown = await triggerInstall();
        if (!shown && isIOS() && onIOSInstructions) onIOSInstructions();
    });
}

// ─── One-time "enjoying it?" nudge, after a few verifications ────────────
const NUDGE_THRESHOLD = 12;
const NUDGE_KEY = 'pv-install-nudge-shown';

export function shouldShowInstallNudge(totalVotes) {
    if (totalVotes < NUDGE_THRESHOLD || isStandalone() || !canInstall()) return false;
    try { if (localStorage.getItem(NUDGE_KEY)) return false; } catch { /* private mode etc — just don't nudge */ }
    return true;
}
export function markInstallNudgeShown() {
    try { localStorage.setItem(NUDGE_KEY, '1'); } catch { /* private mode etc */ }
}
