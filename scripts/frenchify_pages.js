#!/usr/bin/env node
/* Rewrite the generated pages (content/cities, content/data, content/regions) in French, in place.
 *
 * scripts/build_department_pages.py writes these pages in English. French is the indexed version,
 * so this step runs right after it: it translates each page with the rules in static/js/page-lang.js
 * and embeds the English originals as JSON (<script id="dpm-en">) so the browser can switch back
 * to English without a second URL. Pages that already have lang="fr" are skipped (idempotent).
 *
 *   NODE_PATH=/path/to/node_modules node scripts/frenchify_pages.js [--out DIR] [--only cities|data|regions]
 *
 * Needs jsdom (`npm i jsdom` anywhere; nothing is installed in the repo). */
const fs = require('fs'), path = require('path');
const { JSDOM } = require('jsdom');
const root = path.join(__dirname, '..');
const src = fs.readFileSync(path.join(root, 'static/js/page-lang.js'), 'utf8');
const args = process.argv.slice(2);
const opt = (k) => (args.includes(k) ? args[args.indexOf(k) + 1] : null);
const outDir = opt('--out'), only = opt('--only');
// one window for everything (a window per page runs out of memory); pages are parsed with DOMParser
const win = new JSDOM('<!DOCTYPE html><html><body></body></html>', { url: 'https://deeppvmapper.fr/', runScripts: 'outside-only' }).window;
win.__DPVM_LANG_NO_AUTORUN = true;
win.eval(src);
const parser = new win.DOMParser();
let done = 0, skipped = 0, bytes0 = 0, bytes1 = 0;
for (const kind of ['cities', 'data', 'regions']) {
  if (only && only !== kind) continue;
  const dir = path.join(root, 'content', kind);
  for (const f of fs.readdirSync(dir).filter((x) => x.endsWith('.html'))) {
    const html = fs.readFileSync(path.join(dir, f), 'utf8');
    if (/<html lang="fr"/.test(html)) { skipped++; continue; }
    const doc = parser.parseFromString(html, 'text/html');
    win.DPVMLang.frenchify(doc, kind);
    const out = '<!DOCTYPE html>\n' + doc.documentElement.outerHTML + '\n';
    const target = outDir ? path.join(outDir, kind, f) : path.join(dir, f);
    if (outDir) fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, out);
    done++; bytes0 += html.length; bytes1 += out.length;
  }
}
console.log(`frenchified ${done} page(s), skipped ${skipped} already French` + (done ? `; size ${(bytes0 / 1e6).toFixed(1)} -> ${(bytes1 / 1e6).toFixed(1)} MB` : ''));
