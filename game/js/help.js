// ─── Aide / tuto premier lancement ─────────────────────────────────────────
// Même panneau utilisé deux fois : affiché automatiquement au tout premier
// passage sur l'écran de jeu (marqué dans localStorage — pure commodité
// d'affichage, rien qui doit être fiable ou partagé), et réouvrable à tout
// moment via le bouton (?) de l'en-tête.

import { $, show, hide } from './store.js';

const SEEN_KEY = 'pvscroll_tutorial_seen';

export function initHelp() {
    $('#help-open').addEventListener('click', () => show($('#tutorial')));
    $('#tutorial-close').addEventListener('click', dismiss);

    let seen = false;
    try { seen = localStorage.getItem(SEEN_KEY) === '1'; } catch { /* private mode, etc. — just show it every time */ }
    if (!seen) show($('#tutorial'));
}

function dismiss() {
    hide($('#tutorial'));
    try { localStorage.setItem(SEEN_KEY, '1'); } catch { /* not critical if it can't persist */ }
}
