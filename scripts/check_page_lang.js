#!/usr/bin/env node
/* Checks for the French generated pages (static/js/page-lang.js, scripts/frenchify_pages.js).
 *
 *   NODE_PATH=... node scripts/check_page_lang.js --en DIR_WITH_ENGLISH_PAGES --fr DIR_WITH_FRENCH_PAGES [--show N]
 *     (each DIR holds cities/ data/ regions/)
 *   1. coverage: strings of the French pages that still look English;
 *   2. round trip: restoreEnglish(French page) must give back exactly the English page
 *      (text nodes, tooltips/attributes, title, meta tags).
 *   Default dirs: --en = content/ (only valid before frenchify has run), --fr = content/.
 * Needs jsdom (`npm i jsdom` somewhere; nothing is installed in the repo). */
const fs = require('fs'), path = require('path');
const { JSDOM } = require('jsdom');
const root = path.join(__dirname, '..');
const args = process.argv.slice(2);
const opt = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const enDir = opt('--en', path.join(root, 'content')), frDir = opt('--fr', path.join(root, 'content'));
const show = +opt('--show', 40);
const win = new JSDOM('<!DOCTYPE html><html><body></body></html>', { url: 'https://deeppvmapper.fr/', runScripts: 'outside-only' }).window;
win.__DPVM_LANG_NO_AUTORUN = true;
win.eval(fs.readFileSync(path.join(root, 'static/js/page-lang.js'), 'utf8'));
const parser = new win.DOMParser();
const ENGLISH = /\b(the|of|in|and|with|by|to|is|are|was|were|rooftop|capacity|detections?|click|see|back|report|installed|estimated|ranking|census|inhabitants|nationally|largest|span|identified|latest|release|odd|here)\b/i;
function strings(doc) {
  const out = [];
  const w = doc.createTreeWalker(doc.body, 4);
  for (let n; (n = w.nextNode());) { const p = n.parentNode.nodeName; if (p !== 'SCRIPT' && p !== 'STYLE') { const t = n.nodeValue.trim(); if (t) out.push(t); } }
  doc.body.querySelectorAll('*').forEach(e => { if (e.closest('.topnav-lang')) return; ['data-tip', 'aria-label', 'title', 'placeholder', 'href'].forEach(a => { if (e.hasAttribute(a)) out.push(a + '=' + e.getAttribute(a)); }); });
  ['meta[name="description"]', 'meta[property="og:title"]', 'meta[property="og:description"]'].forEach(s => { const e = doc.querySelector(s); if (e) out.push(s + '=' + e.getAttribute('content')); });
  out.push('title=' + doc.title);
  return out;
}
let pages = 0, total = 0, eng = 0, rtBad = 0, rtShown = 0, bytesJson = 0;
const left = new Map();
for (const kind of ['cities', 'data', 'regions']) {
  const fd = path.join(frDir, kind), ed = path.join(enDir, kind);
  for (const f of fs.readdirSync(fd).filter(x => x.endsWith('.html'))) {
    const frHtml = fs.readFileSync(path.join(fd, f), 'utf8');
    const fr = parser.parseFromString(frHtml, 'text/html');
    pages++;
    // 1. coverage (links excluded: they are not text)
    for (const s of strings(fr).filter(s => !/^href=/.test(s))) {
      total++;
      if (ENGLISH.test(s)) { eng++; const key = s.replace(/\d[\d  .,]*/g, '#').slice(0, 150); const e = left.get(key) || { n: 0, ex: kind + '/' + f }; e.n++; left.set(key, e); }
    }
    // 2. round trip
    if (fs.existsSync(path.join(ed, f)) && path.resolve(ed) !== path.resolve(fd)) {
      const en = parser.parseFromString(fs.readFileSync(path.join(ed, f), 'utf8'), 'text/html');
      const j = fr.getElementById('dpm-en'); if (j) bytesJson += j.textContent.length;
      win.DPVMLang.restoreEnglish(fr);
      const a = strings(en), b = strings(fr);
      const bad = a.length !== b.length ? ['length ' + a.length + ' vs ' + b.length] : a.map((x, i) => x === b[i] ? null : x + '  !=  ' + b[i]).filter(Boolean);
      if (bad.length) { rtBad++; if (rtShown++ < 5) console.log('ROUNDTRIP MISMATCH', kind + '/' + f, bad.slice(0, 3)); }
    }
  }
}
console.log(`pages: ${pages}  strings: ${total}  still-English-looking: ${eng} (${(100 * eng / total).toFixed(2)}%)  distinct: ${left.size}`);
console.log(`round trip: ${rtBad} page(s) differ from the English original; embedded JSON ${(bytesJson / 1e6).toFixed(1)} MB`);
[...left.entries()].sort((a, b) => b[1].n - a[1].n).slice(0, show).forEach(([k, v]) => console.log(String(v.n).padStart(5), JSON.stringify(k), '  e.g.', v.ex));
