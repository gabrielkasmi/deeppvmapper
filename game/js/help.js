// ─── Aide / tuto premier lancement ─────────────────────────────────────────
// Même panneau utilisé deux fois : affiché automatiquement au tout premier
// passage sur l'écran de jeu (marqué dans localStorage — pure commodité
// d'affichage, rien qui doit être fiable ou partagé), et réouvrable à tout
// moment via le bouton (?) de l'en-tête.
//
// The tutorial itself is a short auto-advancing sequence of illustrated
// frames (see the #tutorial-frames markup in index.html), not a static
// bullet list — each frame either replays one of the three vote gestures
// on a real (tiny, pre-picked) example card, or pulses a tap-ring around
// a self-contained copy of a utility icon/header glyph. It's deliberately
// NOT spotlighting the live buttons behind a dimmed #game-screen: at this
// panel's size that would mean pixel-precise positioning over elements
// that move between screen widths, for a one-time-per-player payoff that
// doesn't justify the fragility.

import { $, $$, show, hide, toast } from './store.js';
import { wireInstallButton } from './pwa.js';

const SEEN_KEY = 'pvscroll_tutorial_seen';
const FRAME_MS = 2800; // auto-advance interval, paused once the last frame is reached

let frames = [];
let dots = [];
let frameIdx = 0;
let timer = null;

export function initHelp() {
    frames = $$('#tutorial-frames .tutorial-frame');
    dots = $$('#tutorial-dots .tutorial-dot');

    $('#help-open').addEventListener('click', openTutorial);
    wireInstallButton($('#header-install-btn'), () => toast('Tap the Share icon, then "Add to Home Screen".', 4500));
    $('#header-purpose-btn').addEventListener('click', () => show($('#purpose-panel')));
    $('#tutorial-next').addEventListener('click', () => { advance(); restartAutoplay(); });
    $('#tutorial-close').addEventListener('click', dismiss);

    let seen = false;
    try { seen = localStorage.getItem(SEEN_KEY) === '1'; } catch { /* private mode, etc. — just show it every time */ }
    if (!seen) openTutorial();
}

function openTutorial() {
    show($('#tutorial'));
    goToFrame(0);
    restartAutoplay();
}

function goToFrame(i) {
    frameIdx = Math.max(0, Math.min(frames.length - 1, i));
    frames.forEach((f, idx) => { f.hidden = idx !== frameIdx; });
    dots.forEach((d, idx) => d.classList.toggle('active', idx === frameIdx));
    const onLastFrame = frameIdx === frames.length - 1;
    // "Next" and "Got it" are two different buttons (not one relabelled)
    // so the very last tap — the one that dismisses the whole panel and
    // sets the seen flag — is never accidentally the same target a player
    // has already been tapping through five prior frames on autopilot.
    if (onLastFrame) { hide($('#tutorial-next')); show($('#tutorial-close')); }
    else { show($('#tutorial-next')); hide($('#tutorial-close')); }
}

function advance() {
    if (frameIdx >= frames.length - 1) { stopAutoplay(); return; }
    goToFrame(frameIdx + 1);
    if (frameIdx === frames.length - 1) stopAutoplay();
}

function restartAutoplay() {
    stopAutoplay();
    if (frameIdx < frames.length - 1) timer = setInterval(advance, FRAME_MS);
}

function stopAutoplay() {
    clearInterval(timer);
    timer = null;
}

function dismiss() {
    stopAutoplay();
    hide($('#tutorial'));
    try { localStorage.setItem(SEEN_KEY, '1'); } catch { /* not critical if it can't persist */ }
}
