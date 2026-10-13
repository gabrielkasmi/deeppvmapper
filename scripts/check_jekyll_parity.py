#!/usr/bin/env python3
"""Check that a Jekyll build of the site (`_site/`) is equivalent to the pre-Jekyll site.

Usage (from the repo root, after `bundle exec jekyll build`):

    python3 scripts/check_jekyll_parity.py                 # compare _site/ with git ref gh-pages
    python3 scripts/check_jekyll_parity.py --base origin/gh-pages --site _site

What it checks, using only the Python standard library:
  1. every file tracked at the base ref is present in _site/ (dot-files such as .github/ are never
     published by Jekyll and are ignored), and Jekyll did not generate pages that have no source file,
  2. files that did not go through Jekyll are byte-identical,
  3. pages that did go through Jekyll (those starting with a `---` front matter) are the same
     document once whitespace, comments, attribute order, entities and relative-URL spelling are
     normalised: same <head> elements (any order), same <body> tree (same order).

Exit status 0 = equivalent, 1 = differences found (they are printed).
"""
import argparse, os, posixpath, re, subprocess, sys
from html.parser import HTMLParser

VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}
URL_ATTRS = {'href', 'src', 'action', 'poster', 'data-src'}
RAW_TEXT = {'pre', 'textarea'}


class Node:
    def __init__(self, tag, attrs): self.tag, self.attrs, self.children = tag, attrs, []


class Builder(HTMLParser):
    def __init__(self, page_path):
        super().__init__(convert_charrefs=True)
        self.page_dir = posixpath.dirname(page_path)
        self.root = Node('#root', ()); self.stack = [self.root]

    def _resolve(self, v):
        if re.match(r'^([a-zA-Z][a-zA-Z0-9+.-]*:|//|#|/)', v) or not v: return v
        m = re.match(r'^([^?#]*)(.*)$', v)
        return posixpath.normpath(posixpath.join(self.page_dir, m.group(1))) + m.group(2) if m.group(1) else v

    def _node(self, tag, attrs):
        out = []
        for k, v in attrs:
            v = '' if v is None else v
            out.append((k, self._resolve(v) if k in URL_ATTRS else v))
        return Node(tag, tuple(sorted(out)))

    def handle_starttag(self, tag, attrs):
        n = self._node(tag, attrs); self.stack[-1].children.append(n)
        if tag not in VOID: self.stack.append(n)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(self._node(tag, attrs))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]; return

    def handle_data(self, data):
        ch = self.stack[-1].children
        if ch and isinstance(ch[-1], str): ch[-1] += data
        else: ch.append(data)


def canon(node, in_raw=False):
    if isinstance(node, str):
        t = node if in_raw else re.sub(r'\s+', ' ', node).strip()
        return t
    raw = in_raw or node.tag in RAW_TEXT
    kids = [canon(c, raw) for c in node.children]
    kids = [k for k in kids if k != '']
    return '<%s %s>%s</%s>' % (node.tag, ' '.join('%s=%r' % kv for kv in node.attrs), ''.join(kids), node.tag)


def parse(html, path):
    b = Builder(path); b.feed(html); b.close()
    def find(n, tag):
        for c in n.children:
            if isinstance(c, Node):
                if c.tag == tag: return c
                r = find(c, tag)
                if r: return r
    return b.root, find(b.root, 'html'), find(b.root, 'head'), find(b.root, 'body')


def compare_pages(old, new, path):
    problems = []
    _, h1, hd1, bd1 = parse(old, path); _, h2, hd2, bd2 = parse(new, path)
    if (h1.attrs if h1 else None) != (h2.attrs if h2 else None):
        problems.append(f'<html> attributes: {h1.attrs if h1 else None} -> {h2.attrs if h2 else None}')
    e1 = sorted(canon(c) for c in hd1.children if canon(c)); e2 = sorted(canon(c) for c in hd2.children if canon(c))
    for x in e1:
        if x not in e2: problems.append('<head> only before: ' + x[:230])
    for x in e2:
        if x not in e1: problems.append('<head> only after:  ' + x[:230])
    b1 = [canon(c) for c in bd1.children]; b1 = [x for x in b1 if x]
    b2 = [canon(c) for c in bd2.children]; b2 = [x for x in b2 if x]
    if (bd1.attrs if bd1 else None) != (bd2.attrs if bd2 else None):
        problems.append(f'<body> attributes: {bd1.attrs} -> {bd2.attrs}')
    if b1 != b2:
        i = next((k for k in range(min(len(b1), len(b2))) if b1[k] != b2[k]), min(len(b1), len(b2)))
        problems.append(f'<body> differs at top-level child #{i} (before has {len(b1)}, after has {len(b2)})')
        if i < len(b1) and i < len(b2):
            x, y = b1[i], b2[i]
            j = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
            problems.append('   first difference, before: ...' + x[max(0, j - 60):j + 160])
            problems.append('   first difference, after:  ...' + y[max(0, j - 60):j + 160])
        elif i < len(b1): problems.append('   only before: ' + b1[i][:300])
        elif i < len(b2): problems.append('   only after:  ' + b2[i][:300])
    return problems


def git(*a, root='.'):
    return subprocess.run(['git', '-C', root, *a], capture_output=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='gh-pages', help='git ref holding the pre-Jekyll site')
    ap.add_argument('--site', default='_site', help='Jekyll output directory')
    ap.add_argument('--root', default='.', help='repository root')
    a = ap.parse_args()
    os.chdir(a.root)
    base_files = git('ls-tree', '-r', '--name-only', '-z', a.base).decode().split('\0')
    base_files = [f for f in base_files if f]
    built = set()
    for d, _, fs in os.walk(a.site):
        for f in fs: built.add(os.path.relpath(os.path.join(d, f), a.site).replace(os.sep, '/'))
    # Jekyll never publishes dot-files / dot-directories (.github/, .gitignore, .nojekyll, ...)
    missing = [f for f in base_files if f not in built and not any(part.startswith('.') for part in f.split('/'))]
    # new files that exist in the repo on purpose (Gemfile is skipped by Jekyll; scripts, ...) are fine;
    # what we want to catch is pages Jekyll *generated* that have no source file (e.g. README.html)
    extra = sorted(f for f in built - set(base_files) if not os.path.exists(f))
    n_same = n_page_ok = 0; bad = []
    for f in base_files:
        if f not in built: continue
        old = git('show', f'{a.base}:{f}'); new = open(os.path.join(a.site, f), 'rb').read()
        if old == new: n_same += 1; continue
        if f.endswith('.html'):
            pr = compare_pages(old.decode('utf-8'), new.decode('utf-8'), f)
            if pr: bad.append((f, pr))
            else: n_page_ok += 1
        else:
            bad.append((f, ['file content differs (not an HTML page processed by Jekyll)']))
    print(f'{n_same} files byte-identical, {n_page_ok} Jekyll pages equivalent after normalisation')
    if missing: print(f'\nMISSING from {a.site} ({len(missing)}):', *missing[:30], sep='\n  ')
    if extra: print(f'\nEXTRA in {a.site} ({len(extra)}):', *extra[:30], sep='\n  ')
    for f, pr in bad:
        print(f'\nDIFFERENT: {f}'); [print('  ' + p) for p in pr]
    ok = not (missing or extra or bad)
    print('\nOK: the build matches the pre-Jekyll site.' if ok else '\nDifferences found (see above).')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
