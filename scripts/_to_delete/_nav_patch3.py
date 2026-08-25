import re, glob, sys

OLD_PATTERN = re.compile(
    r'<div class="topnav-links">.*?All resources<span class="dropdown-item-sub">Related work, publications &amp; press</span></a>\s*</div>\s*</div>\s*</div>',
    re.DOTALL
)

NEW_BLOCK = '''<div class="topnav-links">
                    <a href="../about.html" class="topnav-link">About</a>
                    <a href="../data.html" class="topnav-link is-active">Data</a>
                    <a href="../software.html" class="topnav-link">Software</a>
                    <a href="../contribute.html" class="topnav-link">Contribute</a>
                    <a href="../outlook.html" class="topnav-link">Use Cases</a>
                    <div class="dropdown">
                        <span class="topnav-link dropdown-btn">Docs &#9662;</span>
                        <div class="dropdown-content">
                            <a href="../pipeline.html" class="dropdown-item">DeepPVMapper<span class="dropdown-item-sub">Detection architecture &amp; deployment</span></a>
                            <a href="../openpvmapper.html" class="dropdown-item">OpenPVMapper<span class="dropdown-item-sub">Multi-source database methodology</span></a>
                        </div>
                    </div>
                    <div class="dropdown">
                        <span class="topnav-link dropdown-btn">Publications &#9662;</span>
                        <div class="dropdown-content">
                            <a href="../publications.html" class="dropdown-item">Papers &amp; Preprints<span class="dropdown-item-sub">Peer-reviewed papers, preprints &amp; posters</span></a>
                            <a href="../in-press.html" class="dropdown-item">Press Coverage<span class="dropdown-item-sub">Popular-science &amp; media coverage</span></a>
                        </div>
                    </div>
                </div>'''

def patch_file(path, dry=False):
    with open(path, encoding='utf-8') as f:
        content = f.read()
    new_content, n = OLD_PATTERN.subn(lambda m: NEW_BLOCK, content, count=1)
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
        import shutil
        shutil.copy('content/data/02.html', '/tmp/test_nav_patch.html')
        n = patch_file('/tmp/test_nav_patch.html')
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
