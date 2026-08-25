import re, glob, sys, shutil

PATTERN = re.compile(
    r'(<a href="[^"]*outlook\.html" class="topnav-link(?: is-active)?">Use Cases</a>\n)([ \t]*)(<div class="dropdown">)'
)

def build_replacement(m):
    indent = m.group(2)
    return f'{m.group(1)}{indent}<span class="topnav-break" aria-hidden="true"></span>\n{indent}{m.group(3)}'

def patch_file(path, dry=False):
    with open(path, encoding='utf-8') as f:
        content = f.read()
    new_content, n = PATTERN.subn(build_replacement, content, count=1)
    if n != 1:
        return n
    if not dry:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    return n

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'dry'
    files = sorted(glob.glob('content/data/*.html') + glob.glob('content/cities/*.html') + glob.glob('content/regions/*.html'))
    print(f"Found {len(files)} files")
    if mode == 'dry':
        shutil.copy('content/data/02.html', '/tmp/test_break_patch.html')
        n = patch_file('/tmp/test_break_patch.html')
        print(f"Dry-run match count on test copy: {n}")
    else:
        ok, bad = 0, []
        for f in files:
            n = patch_file(f)
            if n == 1:
                ok += 1
            else:
                bad.append((f, n))
        print(f"Patched {ok} files.")
        if bad:
            print(f"MISMATCHES ({len(bad)}):")
            for f, n in bad[:20]:
                print(f"  {f}: {n} matches")
