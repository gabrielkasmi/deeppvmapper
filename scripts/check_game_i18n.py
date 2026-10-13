#!/usr/bin/env python3
"""Consistency check for PV Check's translations (game/). Standard library only.

    python3 scripts/check_game_i18n.py

English lives in game/index.html (data-i18n / data-i18n-attr) and in the t('id', 'English')
calls of game/js/*.js; French lives in game/js/lang/fr.js, keyed by id. This checks that:
  1. every id used in the HTML / JS exists in fr.js (else that string silently stays English),
  2. every id in fr.js is still used somewhere (leftovers after an edit),
  3. a French t() string uses the same {placeholders} as its English default,
  4. a French data-i18n string keeps the same child elements and ids as the English markup
     (JS fills things like <strong id="landing-count"> inside them — losing one breaks the app).
Exit status 0 = consistent, 1 = problems (printed).
"""
import json, os, re, sys
from html.parser import HTMLParser

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'game')
read = lambda *p: open(os.path.join(ROOT, *p), encoding='utf-8').read()
problems = []

# ── fr.js
FR = {}
for line in read('js', 'lang', 'fr.js').splitlines():
    m = re.match(r'^\s*("(?:[^"\\]|\\.)*"):\s*("(?:[^"\\]|\\.)*"),\s*$', line)
    if m: FR[json.loads(m.group(1))] = json.loads(m.group(2))

# ── HTML
class Collector(HTMLParser):
    """Collects, for each data-i18n element, the (tag, id) of its descendants."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.items, self.attr_ids = [], [], []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        for open_item in self.stack: open_item['children'].append((tag, a.get('id')))
        if 'data-i18n-attr' in a:
            self.attr_ids += [p.split(':', 1)[1].strip() for p in a['data-i18n-attr'].split(',') if ':' in p]
        if tag not in ('input', 'img', 'br', 'hr', 'meta', 'link', 'source', 'path', 'line', 'polyline', 'circle', 'rect', 'polygon', 'stop'):
            item = {'id': a.get('data-i18n'), 'tag': tag, 'children': []}
            self.stack.append(item)
        if 'data-i18n' in a and tag in ('input', 'img', 'br', 'hr', 'meta', 'link'):
            problems.append(f'data-i18n on a void element <{tag}> ({a["data-i18n"]}) — use data-i18n-attr')
    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i]['tag'] == tag:
                item = self.stack.pop(i)
                if item['id']: self.items.append(item)
                del self.stack[i:]   # drop any unclosed children
                return

c = Collector(); c.feed(read('index.html'))
html_ids = {i['id'] for i in c.items} | set(c.attr_ids)
for item in c.items:
    fr = FR.get(item['id'])
    if fr is None: continue
    w = Collector(); w.stack = [{'id': None, 'tag': 'div', 'children': []}]; wrapper = w.stack[0]; w.feed(fr)
    got = sorted(map(str, wrapper['children']))
    want = sorted(map(str, item['children']))
    if got != want:
        problems.append(f'{item["id"]}: French markup differs from the English one — English has {want}, French has {got}')

# ── JS
T_CALL = re.compile(r"(?<![\w$.])t\(\s*(['\"])([\w.]+)\1\s*,\s*(?:'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\")")
PLACEHOLDER = re.compile(r'\{(\w+)\}')
js_ids = {}
for fn in sorted(os.listdir(os.path.join(ROOT, 'js'))):
    if not fn.endswith('.js') or fn == 'i18n.js': continue
    for m in T_CALL.finditer(read('js', fn)):
        en = m.group(3) if m.group(3) is not None else m.group(4)
        js_ids.setdefault(m.group(2), []).append((fn, en.replace("\\'", "'").replace('\\"', '"')))

for i, uses in js_ids.items():
    fr = FR.get(i)
    if fr is None: continue
    for fn, en in uses:
        want, got = sorted(set(PLACEHOLDER.findall(en))), sorted(set(PLACEHOLDER.findall(fr)))
        if want != got:
            problems.append(f'{i} ({fn}): placeholders differ — English {want}, French {got}')

used = html_ids | set(js_ids)
for i in sorted(used - set(FR)): problems.append(f'missing in fr.js (stays English): {i}')
for i in sorted(set(FR) - used): problems.append(f'unused in fr.js: {i}')

print(f'{len(used)} ids used ({len(html_ids)} in index.html, {len(js_ids)} in js/), {len(FR)} in fr.js')
for p in problems: print(' -', p)
print('OK' if not problems else f'{len(problems)} problem(s)')
sys.exit(1 if problems else 0)
