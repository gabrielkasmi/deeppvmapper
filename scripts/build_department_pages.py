#!/usr/bin/env python3
"""
Build static, indexable per-département landing pages for the Data hub.

Offline build step (no server-side templating on the deployed site) — run
locally, commit the generated HTML. Mirrors the spirit of build_geo_stats.py.

Inputs (repo-relative, already committed):
  static/data/stats/stats_departements.json   {code: [n_systems, kWp]}
  static/data/geo/departements.geojson        features[].properties = {code, nom, region}
  static/data/geo/regions.geojson             features[].properties = {code, nom}

Output:
  content/data/{code}.html   one page per département (96 pages)

Usage:
  python3 scripts/build_department_pages.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "content", "data")

with open(os.path.join(ROOT, "static", "data", "stats", "stats_departements.json"), encoding="utf-8") as f:
    STATS = json.load(f)

with open(os.path.join(ROOT, "static", "data", "geo", "departements.geojson"), encoding="utf-8") as f:
    DEPTS = json.load(f)["features"]

with open(os.path.join(ROOT, "static", "data", "geo", "regions.geojson"), encoding="utf-8") as f:
    REGIONS = {f["properties"]["code"]: f["properties"]["nom"] for f in json.load(f)["features"]}

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{nom} ({code}) &middot; DeepPVMapper Data</title>

    <meta name="description" content="Rooftop PV systems detected by DeepPVMapper in {nom} ({code}): {n_fmt} systems, {mwp_fmt} MWp estimated installed capacity. Explore on the map or download the data.">
    <meta name="author" content="Gabriel Kasmi">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://gabrielkasmi.github.io/deeppvmapper/content/data/{code}.html">
    <meta property="og:title" content="{nom} ({code}) &middot; DeepPVMapper Data">
    <meta property="og:description" content="{n_fmt} rooftop PV systems detected in {nom}, {mwp_fmt} MWp estimated installed capacity.">
    <meta property="og:image" content="https://gabrielkasmi.github.io/deeppvmapper/static/images/teaser.webp">
    <link rel="canonical" href="https://gabrielkasmi.github.io/deeppvmapper/content/data/{code}.html">

    <link rel="icon" type="image/x-icon" href="../../static/images/favicon.ico">
    <link rel="stylesheet" href="../../static/css/style.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>

    <header class="header header--compact">
        <div class="header-background"></div>
        <nav class="topnav topnav--overlay">
            <div class="container topnav-inner">
                <a href="../../index.html" class="topnav-logo">DeepPVMapper</a>
                <div class="topnav-links">
                    <a href="../data.html" class="topnav-link is-active">Data</a>
                    <a href="../software.html" class="topnav-link">Software</a>
                    <a href="../contribute.html" class="topnav-link">Contribute</a>
                    <a href="../resources.html" class="topnav-link">Resources</a>
                </div>
            </div>
        </nav>
        <div class="container">
            <p class="page-eyebrow">Data &middot; D&eacute;partement {code}</p>
            <h1 class="title">{nom}</h1>
            <p class="subtitle">{region_nom}</p>
        </div>
    </header>

    <section>
        <div class="container">
            <div class="validation-stats">
                <div class="stat-item">
                    <span class="stat-number">{n_fmt}</span>
                    <span class="stat-label">PV systems detected</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{mwp_fmt} MWp</span>
                    <span class="stat-label">Estimated installed capacity</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">#{rank}</span>
                    <span class="stat-label">by installed capacity, out of {total} d&eacute;partements</span>
                </div>
            </div>
            <div class="card-grid" style="max-width: 700px; margin-left: auto; margin-right: auto;">
                <div class="hub-card">
                    <h3>Explore on the map</h3>
                    <p>Open the interactive registry zoomed straight into {nom}, with the exact detections, stats and boundary.</p>
                    <a href="../map.html?dept={code}" class="btn">Explore {nom} &rarr;</a>
                </div>
                <div class="hub-card">
                    <h3>Download this d&eacute;partement's data</h3>
                    <p>Once on the map, the locked-zone stats card includes an exhaustive CSV export for {nom} &mdash; every detection, not a sample.</p>
                    <a href="../map.html?dept={code}" class="btn">Get the data &rarr;</a>
                </div>
            </div>
            <p style="text-align: center; margin-top: 40px;">
                <a href="../data.html">&larr; Back to all d&eacute;partements</a>
            </p>
        </div>
    </section>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2021-2026 Gabriel Kasmi. This work is licensed under <a href="https://github.com/gabrielkasmi/deeppvmapper/blob/main/LICENSE" target="_blank">MIT</a>.</p>
        </div>
    </footer>

    <script src="../../static/js/script.js"></script>
</body>
</html>
"""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Rank by installed capacity (kWp), descending.
    ranked = sorted(STATS.items(), key=lambda kv: kv[1][1], reverse=True)
    rank_by_code = {code: i + 1 for i, (code, _) in enumerate(ranked)}
    total = len(STATS)

    dept_index = []  # for the Data hub's "browse by département" grid

    for feature in DEPTS:
        props = feature["properties"]
        code, nom, region_code = props["code"], props["nom"], props["region"]
        if code not in STATS:
            continue  # no detections/stats for this département — skip page
        n, kwp = STATS[code]
        region_nom = REGIONS.get(region_code, "France")

        html = TEMPLATE.format(
            code=code,
            nom=nom,
            region_nom=region_nom,
            n_fmt=f"{n:,}".replace(",", " "),
            mwp_fmt=f"{kwp / 1000:.1f}",
            rank=rank_by_code[code],
            total=total,
        )
        with open(os.path.join(OUT_DIR, f"{code}.html"), "w", encoding="utf-8") as f:
            f.write(html)

        dept_index.append({
            "code": code, "nom": nom, "region": region_nom,
            "region_code": region_code, "n": n, "mwp": round(kwp / 1000, 1),
        })

    dept_index.sort(key=lambda d: (d["region"], d["nom"]))

    inject_browse_grid(dept_index)

    print(f"Wrote {len(dept_index)} département pages to {OUT_DIR}")


def inject_browse_grid(dept_index):
    """Render the grouped 'browse by département' links as static HTML and
    splice them into content/data.html between the DEPARTMENTS_GRID markers.
    Static (not JS-fetched) on purpose — every département link is crawlable
    straight from the Data hub, which is the whole point of pre-generating
    these pages in the first place (SEO)."""
    by_region = {}
    for d in dept_index:
        by_region.setdefault(d["region"], []).append(d)

    blocks = []
    for region_nom in sorted(by_region):
        links = "\n".join(
            f'                    <a class="dept-link" href="data/{d["code"]}.html">'
            f'{d["nom"]} <span class="dept-link-code">{d["code"]}</span></a>'
            for d in by_region[region_nom]
        )
        blocks.append(
            f'                <div class="dept-region">\n'
            f'                    <h3>{region_nom}</h3>\n'
            f'                    <div class="dept-list">\n{links}\n'
            f'                    </div>\n'
            f'                </div>'
        )
    grid_html = "\n".join(blocks)

    data_html_path = os.path.join(ROOT, "content", "data.html")
    with open(data_html_path, encoding="utf-8") as f:
        page = f.read()

    start_marker = "<!-- DEPARTMENTS_GRID_START -->"
    end_marker = "<!-- DEPARTMENTS_GRID_END -->"
    start = page.index(start_marker) + len(start_marker)
    end = page.index(end_marker)
    page = page[:start] + "\n" + grid_html + "\n                " + page[end:]

    with open(data_html_path, "w", encoding="utf-8") as f:
        f.write(page)


if __name__ == "__main__":
    main()
