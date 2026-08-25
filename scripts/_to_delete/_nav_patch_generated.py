#!/usr/bin/env python3
"""One-off: patch the generated department/city/region pages (content/data,
content/cities, content/regions) with the same nav simplification and
footer wording already applied to the hand-authored pages + the
build_department_pages.py generator template. Run once directly on the
device, then delete."""
import glob
import re
import sys

NAV_RE = re.compile(
    r'([ \t]*)<div class="topnav-links">.*?In Press</a>\s*\n[ \t]*</div>',
    re.DOTALL,
)

OLD_FOOTER = '&copy; 2021-2026 Gabriel Kasmi. This work is licensed under'
NEW_FOOTER = 'Maintained by Gabriel Kasmi. This work is licensed under'


def build_block(indent):
    i0 = indent
    i1 = indent + '    '
    i2 = indent + '        '
    i3 = indent + '            '
    lines = [
        f'{i0}<div class="topnav-links">',
        f'{i1}<a href="../about.html" class="topnav-link">About</a>',
        f'{i1}<a href="../data.html" class="topnav-link is-active">Data</a>',
        f'{i1}<a href="../software.html" class="topnav-link">Software</a>',
        f'{i1}<a href="../contribute.html" class="topnav-link">Contribute</a>',
        f'{i1}<div class="dropdown">',
        f'{i2}<span class="topnav-link dropdown-btn">Research &#9662;</span>',
        f'{i2}<div class="dropdown-content">',
        f'{i3}<a href="../openpvmapper.html" class="dropdown-item">OpenPVMapper<span class="dropdown-item-sub">Multi-source database methodology</span></a>',
        f'{i3}<a href="../main-results.html" class="dropdown-item">Registry Audit<span class="dropdown-item-sub">Auditing France\'s PV registries</span></a>',
        f'{i3}<a href="../pipeline.html" class="dropdown-item">Pipeline<span class="dropdown-item-sub">Detection architecture &amp; deployment</span></a>',
        f'{i3}<a href="../resources.html" class="dropdown-item">All resources<span class="dropdown-item-sub">Related work, publications &amp; press</span></a>',
        f'{i2}</div>',
        f'{i1}</div>',
        f'{i0}</div>',
    ]
    return '\n'.join(lines)


def patch_file(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()
    orig = text

    def repl(m):
        return build_block(m.group(1))

    text, n_nav = NAV_RE.subn(repl, text)
    n_footer = text.count(OLD_FOOTER)
    text = text.replace(OLD_FOOTER, NEW_FOOTER)

    if text != orig:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
    return n_nav, n_footer


def main():
    patterns = [
        'content/data/*.html',
        'content/cities/*.html',
        'content/regions/*.html',
    ]
    total_files = 0
    nav_missing = []
    footer_missing = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            total_files += 1
            n_nav, n_footer = patch_file(path)
            if n_nav != 1:
                nav_missing.append((path, n_nav))
            if n_footer != 1:
                footer_missing.append((path, n_footer))
    print(f'Patched {total_files} files.')
    if nav_missing:
        print(f'NAV mismatch in {len(nav_missing)} files (expected exactly 1 match):')
        for p, n in nav_missing[:20]:
            print(f'  {p}: {n}')
    if footer_missing:
        print(f'FOOTER mismatch in {len(footer_missing)} files (expected exactly 1 match):')
        for p, n in footer_missing[:20]:
            print(f'  {p}: {n}')
    if not nav_missing and not footer_missing:
        print('All files matched exactly once for both nav and footer.')


if __name__ == '__main__':
    main()
