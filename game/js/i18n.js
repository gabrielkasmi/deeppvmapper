// ─── Language: English (source) + French (js/lang/fr.js) ──────────────────
// English is the source language and stays where it always lived — in
// index.html and as the default argument of t() calls — so English players get
// exactly what they got before this module existed, even if a French string is
// missing or lang/fr.js were to fail. French is looked up by id:
//
//   HTML   data-i18n="id"                        element's innerHTML  <- FR[id]
//          data-i18n-attr="attr:id, attr:id"     attribute            <- FR[id]
//   JS     t('id', 'English default {x}', { x })
//
// An element translated through data-i18n must keep, in its French text, the
// same child elements/ids as its English markup (JS fills <strong id="...">
// inside them later). `python3 scripts/check_game_i18n.py` verifies that, and
// that every id used in the code exists in fr.js (and vice-versa).
//
// Which language: ?lang=xx in the URL (remembered) > remembered choice (the
// EN/FR switch) > browser language (first supported one) > English.

import FR from './lang/fr.js';

const STORAGE_KEY = 'pvcheck-lang';
const SUPPORTED = ['en', 'fr'];

function remember(code) {
    try { localStorage.setItem(STORAGE_KEY, code); } catch { /* private mode etc — just not remembered */ }
}

function detect() {
    try {
        const q = new URLSearchParams(location.search).get('lang');
        if (SUPPORTED.includes(q)) { remember(q); return q; }
    } catch { /* no URL API — ignore */ }
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (SUPPORTED.includes(saved)) return saved;
    } catch { /* private mode etc */ }
    const prefs = navigator.languages && navigator.languages.length ? navigator.languages : [navigator.language];
    for (const l of prefs) {
        const code = String(l || '').toLowerCase().slice(0, 2);
        if (SUPPORTED.includes(code)) return code;
    }
    return 'en';
}

export const lang = detect();
const TABLE = lang === 'fr' ? FR : null;

// undefined = "the browser's own locale", i.e. exactly what a bare
// toLocaleString() / toLocaleDateString() did before — only French is forced.
export const locale = lang === 'fr' ? 'fr-FR' : undefined;
export const fmt = n => Number(n).toLocaleString(locale);

/** id: key in fr.js · en: English text (also the fallback) · vars: {name} placeholders */
export function t(id, en, vars) {
    let s = (TABLE && TABLE[id]) ?? en;
    if (vars) s = s.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? vars[k] : m));
    return s;
}

function applyDom() {
    document.documentElement.lang = lang;
    if (!TABLE) return;
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const v = TABLE[el.dataset.i18n];
        if (v != null) el.innerHTML = v;
    });
    document.querySelectorAll('[data-i18n-attr]').forEach(el => {
        el.dataset.i18nAttr.split(',').forEach(pair => {
            const [attr, id] = pair.split(':').map(s => s.trim());
            const v = TABLE[id];
            if (attr && v != null) el.setAttribute(attr, v);
        });
    });
}

// The EN | FR switch (landing screen + menu). Reloading is the simplest
// reliable way to re-render every dynamically built string; a vote in flight
// is flushed by the pagehide handler in swipe.js, same as any reload.
function wireSwitch() {
    document.querySelectorAll('.lang-switch button[data-lang]').forEach(btn => {
        btn.setAttribute('aria-pressed', btn.dataset.lang === lang ? 'true' : 'false');
        btn.addEventListener('click', () => {
            if (btn.dataset.lang === lang) return;
            remember(btn.dataset.lang);
            const url = new URL(location.href);
            url.searchParams.delete('lang'); // else the URL would override the choice just made
            location.replace(url.toString());
        });
    });
}

applyDom();
wireSwitch();
