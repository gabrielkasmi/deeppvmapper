/*
 * lang-pref.js - remembers the visitor's language choice and suggests the other language.
 *
 *   dpm_pref  explicit choice, written only when the visitor clicks the EN/FR switcher or answers
 *             the banner. On the hand-written pages (Jekyll) the redirect to the preferred language
 *             is done by a tiny inline script in <head> (see _layouts/default.html), before first paint.
 *   dpm_lang  language of the page being viewed (used by the generated region/department/city pages).
 *
 * This file adds two things, on every page that has a language switcher:
 *   1. a click on the switcher stores dpm_pref;
 *   2. with no stored choice, if the first language of the browser (navigator.languages) is French or
 *      English and differs from the page's language, a discreet banner offers the other version.
 *      Nothing is redirected automatically on a first visit, and crawlers (no localStorage, no
 *      interaction) always see the page as shipped.
 *
 * Loaded by the Jekyll layout, and by page-lang.js on the generated pages. Only functional
 * localStorage entries are used (no cookie, no network call).
 */
(function (win) {
  'use strict';
  var doc = win.document;
  if (!doc || win.__DPVM_PREF_LOADED) return;
  win.__DPVM_PREF_LOADED = true;

  function store(k, v) { try { win.localStorage.setItem(k, v); } catch (e) {} }
  function load(k) { try { return win.localStorage.getItem(k); } catch (e) { return null; } }

  var box = doc.querySelector('.topnav-lang');
  if (!box) return;

  // 1. an explicit click on the switcher is a stated preference
  var links = box.querySelectorAll('a[hreflang]');
  for (var i = 0; i < links.length; i++) {
    links[i].addEventListener('click', (function (a) {
      return function () {
        var l = a.getAttribute('hreflang');
        if (l === 'fr' || l === 'en') { store('dpm_pref', l); store('dpm_lang', l); }
      };
    })(links[i]));
  }

  // 2. suggestion banner
  var cfg = win.DPVM_LANG || {};
  if (cfg.noAlt) return;                                   // page without a translation
  var pref = load('dpm_pref');
  if (pref === 'fr' || pref === 'en') return;              // already decided
  var page = (doc.documentElement.getAttribute('lang') || 'en') === 'fr' ? 'fr' : 'en';

  var wanted = null, langs = [];
  try { langs = (win.navigator.languages && win.navigator.languages.length) ? win.navigator.languages : [win.navigator.language]; } catch (e) {}
  for (var j = 0; j < langs.length && !wanted; j++) {
    var p = String(langs[j] || '').toLowerCase().split('-')[0];
    if (p === 'fr' || p === 'en') wanted = p;
  }
  if (!wanted || wanted === page) return;

  var target = box.querySelector('a[hreflang="' + wanted + '"]');
  if (!target) return;

  var T = {
    fr: { msg: page === 'en' ? 'Cette page existe aussi en français.' : '', go: 'Voir en français',
          stay: page === 'en' ? 'Rester en anglais' : '', label: 'Choix de la langue' },
    en: { msg: 'This page is also available in English.', go: 'View in English',
          stay: 'Keep French', label: 'Language choice' }
  }[wanted];

  var bar = doc.createElement('div');
  bar.className = 'lang-banner';
  bar.setAttribute('role', 'region');
  bar.setAttribute('aria-label', T.label);
  bar.setAttribute('lang', wanted);
  var msg = doc.createElement('span'); msg.className = 'lang-banner-msg'; msg.textContent = T.msg;
  var go = doc.createElement('button'); go.type = 'button'; go.className = 'btn lang-banner-go'; go.textContent = T.go;
  var stay = doc.createElement('button'); stay.type = 'button'; stay.className = 'lang-banner-stay'; stay.textContent = T.stay;
  bar.appendChild(msg); bar.appendChild(go); bar.appendChild(stay);

  go.addEventListener('click', function () { target.click(); });
  stay.addEventListener('click', function () {
    store('dpm_pref', page); store('dpm_lang', page);
    if (bar.parentNode) bar.parentNode.removeChild(bar);
  });
  doc.body.appendChild(bar);
})(window);
