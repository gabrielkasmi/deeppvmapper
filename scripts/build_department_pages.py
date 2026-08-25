#!/usr/bin/env python3
"""
Build static, indexable per-département landing pages for the Data hub.

Offline build step (no server-side templating on the deployed site) — run
locally (or from the Cowork sandbox), commit the generated HTML.

v2: stats now come straight from Supabase (three small aggregate RPCs —
see dept_stats_rpcs.sql) instead of the old local-GeoJSON + IGN-WFS spatial
join pipeline. The `detections` table already carries a `dpt` column
natively (used by the map's admin-code fast path), so a GROUP BY in SQL
gets the same numbers with no separate spatial computation, no need to ship
a 357 MB local GeoJSON dump, and no risk of drifting from what the map
itself serves — same database, same query family.

build_geo_stats.py (commune/région stats, WFS-based) is unrelated to this
script and untouched by this change.

Inputs:
  Supabase RPCs: dept_capacity_stats(), dept_yearly_stats(), dept_source_stats()
  static/data/geo/departements.geojson   features[].properties = {code, nom, region}
  static/data/geo/regions.geojson        features[].properties = {code, nom}

This version (v3) also enriches each page for SEO, and adds two more page
types on top of the per-département ones:
  - the département's region, plus its ~25 largest cities (name + population,
    each deep-linking straight into the map via ?insee=CODE) pulled at build
    time from the (free, keyless) geo.api.gouv.fr commune API — no local
    population dataset is vendored for this.
  - a French + English <meta name="keywords"> tag per page.
  - one aggregate page per région (content/regions/{slug}.html), rolled up
    from its member départements — same charts, plus the région's own
    largest-cities list and a "départements in {région}" grid.
  - one page per city that clears CITY_MIN_SYSTEMS detections
    (content/cities/{insee}.html) — real, distinct content per page (own
    yearly/source charts, own rank within its département) rather than a
    thin reskin, so the ~2,400 largest-by-population candidates only get a
    page where there's something real to show. Needs --local (see below):
    no Supabase RPC exposes the commune-level breakdown yet.
  - a regenerated repo-root sitemap.xml (every content/*.html page, plus
    every département/région/city page) and a robots.txt that points at it.

Output:
  content/data/{code}.html        one page per département with detections
  content/regions/{slug}.html     one aggregate rollup page per région
  content/cities/{insee}.html     one page per city clearing CITY_MIN_SYSTEMS
  content/local-statistics.html   "Leaderboard" rankings (régions/départements/cities) re-injected
  sitemap.xml                     regenerated from the actual content/ tree
  robots.txt                      created, or given a Sitemap: line, if needed

Usage:
  python3 scripts/build_department_pages.py
  python3 scripts/build_department_pages.py --local /path/to/dpvm_enriched.geojson

Note: this script needs outbound internet access (Supabase, or none at all
in --local mode, + geo.api.gouv.fr either way) — run it from your own
machine/terminal, not from a network-sandboxed tool.
"""
import argparse
import datetime
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "content", "data")
REGION_OUT_DIR = os.path.join(ROOT, "content", "regions")
CITY_OUT_DIR = os.path.join(ROOT, "content", "cities")
SITE_BASE = "https://deeppvmapper.fr"

# Keep in sync with static/js/map/config.js — same public anon key used by
# the map itself; RLS does the protecting, not secrecy of this value.
SUPABASE_URL = "https://zelhliylrlktnasircwp.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi"

# Keyless, no-auth commune reference API (INSEE-backed) — used only to rank
# communes by population for the "largest cities" section below.
GEO_API_BASE = "https://geo.api.gouv.fr"
CITIES_PER_DEPT = 25
CITIES_PER_REGION = 25

# A dedicated page is only worth indexing if there's real, distinct content
# to show — a commune with a handful of detections would just be a thin
# reskin of its département page. Below this many detections, the city
# stays as a plain link (map deep-link) inside its département/région page
# instead of getting its own page. Only available in --local mode (no
# Supabase RPC exposes the commune-level breakdown yet).
CITY_MIN_SYSTEMS = 10

# Keep in sync with SOURCE_LABELS in static/js/map/config.js.
SOURCE_LABELS = {0: "DPVM", 1: "FRPV", 2: "OSM", 3: "Manual correction", 4: "Recall sample"}

CHART_PURPLE = "#7248b8"
CHART_PURPLE_LIGHT = "#c9b3e8"
CHART_NAVY = "#34495e"
CHART_GRID = "#e9ecef"
CHART_TEXT = "#6c757d"
CHART_TEXT_DARK = "#495057"

# Palette for the source-combination charts (donut + stacked bar), one color
# per combination of {DPVM, FRPV, OSM} — kept distinct from the navy theme
# used elsewhere so the 6 slices/segments stay readable at a glance.
COMBO_COLORS = {
    "0": "#34495e",       # DPVM alone
    "1": "#c0392b",        # FRPV alone
    "2": "#e0a94e",        # OSM alone
    "0+1": "#7f8fa6",      # DPVM + FRPV
    "0+2": "#8ba888",      # DPVM + OSM
    "1+2": "#b98fc9",      # FRPV + OSM
    "0+1+2": "#5c6b7a",    # all three
}
COMBO_LABELS = {
    "0": "DPVM only", "1": "FRPV only", "2": "OSM only",
    "0+1": "DPVM + FRPV", "0+2": "DPVM + OSM", "1+2": "FRPV + OSM",
    "0+1+2": "DPVM + FRPV + OSM",
}
POWER_CLASSES = ["P1", "P2", "P3", "P4", "P5"]
POWER_CLASS_LABELS = {
    "P1": "&lt; 9 kWp", "P2": "9&ndash;36 kWp", "P3": "36&ndash;100 kWp",
    "P4": "100&ndash;250 kWp", "P5": "&gt; 250 kWp",
}


def power_class(kwp):
    """Power-class bucket per the OpenPVMapper paper's own thresholds
    (Figure 5): P1 <9 kWp, P2 9-36, P3 36-100, P4 100-250, P5 >250."""
    if kwp < 9:
        return "P1"
    if kwp < 36:
        return "P2"
    if kwp < 100:
        return "P3"
    if kwp < 250:
        return "P4"
    return "P5"


# Région code -> name is already derived at build time from regions.geojson
# (see `regions` dict in main()) — no need to hardcode it here. The three
# tables below are genuinely external facts with no pipeline source, each
# pinned to a single, internally-consistent, officially-sourced vintage:
# population is INSEE's official "populations légales 2021" (municipal
# population, authenticated Dec 2023); superficie is each région's total
# land area in km²; GDP is INSEE's provisional 2024 régional-GDP table
# (comptes régionaux annuels, base 2020 — a single consistent figure for
# all 13 métropole régions, supplied directly by Gabriel from insee.fr).
REGION_POPULATION_2021 = {
    "11": 12317279, "24": 2573303, "27": 2800194, "28": 3327966, "32": 5995292,
    "44": 5561287, "52": 3853999, "53": 3394567, "75": 6069352, "76": 6022176,
    "84": 8114361, "93": 5127840, "94": 347597,
}
REGION_SURFACE_KM2 = {
    "11": 12060, "24": 39507, "27": 48036, "28": 30064, "32": 31914,
    "44": 57676, "52": 32388, "53": 27455, "75": 85185, "76": 73385,
    "84": 70895, "93": 31676, "94": 8725,
}
REGION_GDP_2024_MEUR = {
    "11": 865652, "24": 91467, "27": 95775, "28": 117713, "32": 197331,
    "44": 197832, "52": 147791, "53": 126549, "75": 223433, "76": 221752,
    "84": 346405, "93": 217942, "94": 13358,
}
# Short, hand-authored geographic locator for the intro sentence — no
# reliable structured source for "where in France is this région", so this
# is authored prose rather than a lookup.
REGION_ZONE = {
    "11": "in the Paris region, at the heart of northern France",
    "24": "in the center of France",
    "27": "in eastern-central France",
    "28": "in northwestern France, along the Channel coast",
    "32": "in the far north of France",
    "44": "in northeastern France, along the German and Swiss borders",
    "52": "on the Atlantic coast, in western France",
    "53": "on the northwestern peninsula of France",
    "75": "in southwestern France, along the Atlantic coast",
    "76": "in southern France, along the Mediterranean and the Pyrenees",
    "84": "in east-central France, along the Alps",
    "93": "in southeastern France, along the Mediterranean coast",
    "94": "the Mediterranean island region of France",
}

# Official préfecture (chef-lieu) of each métropole département — a fixed,
# stable list (no ambiguity, unlike "largest city", which isn't always the
# préfecture — e.g. Hauts-de-Seine's préfecture is Nanterre, not
# Boulogne-Billancourt). Keyed on the same "code" string used throughout
# this script (departements.geojson properties.code).
DEPT_PREFECTURE = {
    "01": "Bourg-en-Bresse", "02": "Laon", "03": "Moulins", "04": "Digne-les-Bains",
    "05": "Gap", "06": "Nice", "07": "Privas", "08": "Charleville-Mézières",
    "09": "Foix", "10": "Troyes", "11": "Carcassonne", "12": "Rodez",
    "13": "Marseille", "14": "Caen", "15": "Aurillac", "16": "Angoulême",
    "17": "La Rochelle", "18": "Bourges", "19": "Tulle", "2A": "Ajaccio",
    "2B": "Bastia", "21": "Dijon", "22": "Saint-Brieuc", "23": "Guéret",
    "24": "Périgueux", "25": "Besançon", "26": "Valence", "27": "Évreux",
    "28": "Chartres", "29": "Quimper", "30": "Nîmes", "31": "Toulouse",
    "32": "Auch", "33": "Bordeaux", "34": "Montpellier", "35": "Rennes",
    "36": "Châteauroux", "37": "Tours", "38": "Grenoble", "39": "Lons-le-Saunier",
    "40": "Mont-de-Marsan", "41": "Blois", "42": "Saint-Étienne", "43": "Le Puy-en-Velay",
    "44": "Nantes", "45": "Orléans", "46": "Cahors", "47": "Agen",
    "48": "Mende", "49": "Angers", "50": "Saint-Lô", "51": "Châlons-en-Champagne",
    "52": "Chaumont", "53": "Laval", "54": "Nancy", "55": "Bar-le-Duc",
    "56": "Vannes", "57": "Metz", "58": "Nevers", "59": "Lille",
    "60": "Beauvais", "61": "Alençon", "62": "Arras", "63": "Clermont-Ferrand",
    "64": "Pau", "65": "Tarbes", "66": "Perpignan", "67": "Strasbourg",
    "68": "Colmar", "69": "Lyon", "70": "Vesoul", "71": "Mâcon",
    "72": "Le Mans", "73": "Chambéry", "74": "Annecy", "75": "Paris",
    "76": "Rouen", "77": "Melun", "78": "Versailles", "79": "Niort",
    "80": "Amiens", "81": "Albi", "82": "Montauban", "83": "Toulon",
    "84": "Avignon", "85": "La Roche-sur-Yon", "86": "Poitiers", "87": "Limoges",
    "88": "Épinal", "89": "Auxerre", "90": "Belfort", "91": "Évry-Courcouronnes",
    "92": "Nanterre", "93": "Bobigny", "94": "Créteil", "95": "Cergy",
}

# Best-effort, hand-authored "known for" landmark per département — no
# structured source for this, so only départements with a well-established,
# uncontroversial landmark are included; the intro sentence adapts (drops
# the clause) when a département's code isn't in this dict rather than
# guessing. Worth spot-checking after a real build — see build notes.
DEPT_LANDMARK = {
    "06": "the Promenade des Anglais in Nice",
    "11": "the medieval citadel of Carcassonne",
    "12": "the Millau Viaduct",
    "13": "the Calanques and Marseille's Vieux-Port",
    "14": "the D-Day landing beaches",
    "21": "the Burgundy vineyards",
    "28": "Chartres Cathedral",
    "29": "the Pointe du Raz",
    "30": "the Pont du Gard",
    "33": "the Bordeaux vineyards and the Dune du Pilat",
    "34": "the Place de la Comédie in Montpellier",
    "35": "the walled city of Saint-Malo",
    "37": "the Château de Chenonceau",
    "38": "the Vercors and Chartreuse massifs",
    "41": "the Château de Chambord",
    "50": "Mont-Saint-Michel",
    "55": "the Verdun battlefields",
    "56": "the Carnac megaliths",
    "63": "the Puy de Dôme volcano",
    "64": "the Basque coast and Béarn Pyrenees",
    "66": "the Cathar castles of the Pyrénées-Orientales",
    "67": "Strasbourg Cathedral",
    "69": "the Basilica of Notre-Dame de Fourvière in Lyon",
    "74": "Mont Blanc",
    "75": "the Eiffel Tower and Notre-Dame Cathedral",
    "76": "Rouen's Gothic cathedral and the Etretat cliffs",
    "83": "Saint-Tropez",
    "84": "the Palais des Papes in Avignon",
    "86": "the Futuroscope park",
    "2A": "the Corsican coastline near Ajaccio",
    "2B": "the Cap Corse coastline",
}


def rpc(name, timeout=150):
    req = urllib.request.Request(
        f"{SUPABASE_URL}/rest/v1/rpc/{name}",
        data=b"{}",
        headers={
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        print(f"  ! rpc({name!r}) failed: HTTP {e.code}\n    {body}", file=sys.stderr)
        raise


def rank_by_value_desc(items):
    """items: {key: value}. Returns {key: rank}, SQL RANK()-style — ties
    share a rank, the next distinct value skips ahead by the tie count."""
    ranked = sorted(items.items(), key=lambda kv: -kv[1])
    ranks, prev_v, prev_rank = {}, None, 0
    for i, (k, v) in enumerate(ranked, 1):
        if v != prev_v:
            prev_rank = i
        ranks[k] = prev_rank
        prev_v = v
    return ranks


def local_aggregate(path, depts_geo=()):
    """Compute the same three département-level aggregates as the Supabase
    RPCs (dept_capacity_stats / dept_yearly_stats / dept_source_stats), PLUS
    the same three broken down per commune (insee code) — by streaming a
    local copy of the release dump (dpvm_enriched.geojson — one Feature
    object per line) instead of calling Supabase. Same source data, same
    numbers, no network round-trip, no statement-timeout risk. The commune
    breakdown has no Supabase RPC equivalent (yet) — it's what makes the
    per-city pages possible.

    depts_geo (départements' GeoJSON features) is used to precompute each
    département's hex grid up front, so installation points can be assigned
    to a hex cell in the same single streaming pass instead of buffering
    every point in memory.

    Returns (capacity_rows, yearly_rows, source_rows, commune_capacity_rows,
    commune_yearly_rows, commune_source_rows, combo_by_dept, hex_by_dept,
    hex_grids, multi_source_by_dept, combo_by_commune, power_class_by_commune,
    azimuth_by_commune). The first three match the Supabase RPC row shape;
    the commune_* ones are keyed by insee instead of dpt, with
    rank_by_capacity computed within the commune's own département (its
    most locally-relevant comparison), not nationally. combo_by_dept (dpt ->
    {power_class: {combo_key: n}}), hex_by_dept (dpt -> {hex_id: {year:
    kwp_sum}}), hex_grids (dpt -> grid dict from build_dept_hex_grid),
    multi_source_by_dept (dpt -> [n_with_sources, n_with_more_than_2_sources]),
    combo_by_commune / power_class_by_commune (same shape as their _by_dept
    counterparts, but keyed by insee — power_class_by_commune is NOT
    restricted to the DPVM/FRPV/OSM combo taxonomy, unlike combo_by_dept and
    combo_by_commune) and azimuth_by_commune (insee -> {sector 0-15: n}, 16
    22.5°-wide compass sectors starting at true north) all have no Supabase
    equivalent — région/département-level charts built from them, like the
    per-city pages, are only available in --local mode for now."""
    capacity = {}      # dpt -> [n, kwp_sum]
    yearly = {}        # dpt -> {year: [n, kwp_sum]}
    source = {}        # dpt -> {source_id: n}
    c_capacity = {}    # insee -> [dpt, n, kwp_sum]
    c_yearly = {}      # insee -> {year: [n, kwp_sum]}
    c_source = {}      # insee -> {source_id: n}
    combo = {}         # dpt -> {power_class: {combo_key: n}}
    hex_by_dept = {}   # dpt -> {hex_id: {year: kwp_sum}}
    multi_source = {}  # dpt -> [n_with_sources, n_with_more_than_2_sources]
    c_combo = {}         # insee -> {power_class: {combo_key: n}} (commune-level donut)
    c_power_class = {}   # insee -> {power_class: n} (commune-level capacity histogram,
                         # NOT restricted to the DPVM/FRPV/OSM combo taxonomy — every
                         # installation with a usable kWp counts, regardless of source)
    c_azimuth = {}       # insee -> {sector_idx (16 x 22.5°, 0 = North): n}

    hex_grids = {}
    for feat in depts_geo:
        grid = build_dept_hex_grid(feat)
        if grid:
            hex_grids[feat["properties"]["code"]] = grid
    if depts_geo:
        print(f"  Hex grids precomputed for {len(hex_grids)}/{len(depts_geo)} départements.")

    n_lines = n_features = 0
    with open(path, encoding="utf-8") as f:
        for raw in f:
            n_lines += 1
            line = raw.strip().rstrip(",")
            if not line.startswith('{'):
                continue  # header/footer lines ("{", "features": [", "]", "}")
            try:
                feat = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = feat.get("properties")
            if not p:
                continue
            n_features += 1

            dpt = p.get("dpt")
            if not dpt or p.get("false_positive"):
                continue

            kwp = p.get("kWp")
            if kwp is None:
                kwp = p.get("kwp") or 0

            c = capacity.setdefault(dpt, [0, 0.0])
            c[0] += 1
            c[1] += kwp

            insee = p.get("insee")
            if insee:
                cc = c_capacity.setdefault(insee, [dpt, 0, 0.0])
                cc[1] += 1
                cc[2] += kwp

            first_seen = p.get("first_seen")
            year = None
            if first_seen is not None:
                try:
                    year = int(first_seen)
                except (TypeError, ValueError):
                    year = None
            if year is not None:
                yc = yearly.setdefault(dpt, {}).setdefault(year, [0, 0.0])
                yc[0] += 1
                yc[1] += kwp
                if insee:
                    cyc = c_yearly.setdefault(insee, {}).setdefault(year, [0, 0.0])
                    cyc[0] += 1
                    cyc[1] += kwp

            sources = p.get("sources")
            present_012 = set()  # DPVM/FRPV/OSM ids actually seen on this feature
            n_sources_this_feature = 0  # any source id, used for the multi-source metric
            if sources:
                sc = source.setdefault(dpt, {})
                csc = c_source.setdefault(insee, {}) if insee else None
                distinct_sids = set()
                for s in str(sources).split(","):
                    s = s.strip()
                    if not s:
                        continue
                    try:
                        sid = int(s)
                    except ValueError:
                        continue
                    sc[sid] = sc.get(sid, 0) + 1
                    if csc is not None:
                        csc[sid] = csc.get(sid, 0) + 1
                    if sid in (0, 1, 2):
                        present_012.add(sid)
                    distinct_sids.add(sid)
                n_sources_this_feature = len(distinct_sids)

            # Power class is used both by the source-combination breakdown
            # below (région/département donuts, restricted to DPVM/FRPV/OSM)
            # and, unrestricted, by the commune-level capacity histogram —
            # computed once per feature either way.
            pc = power_class(kwp)
            if insee:
                cpc = c_power_class.setdefault(insee, {})
                cpc[pc] = cpc.get(pc, 0) + 1

            # Panel orientation, for the commune-level azimuth rose — degrees
            # clockwise from true north (0-360, per data-documentation.html),
            # bucketed into 16 22.5°-wide compass sectors.
            azimuth = p.get("azimuth")
            if azimuth is not None and insee:
                try:
                    az = float(azimuth) % 360
                except (TypeError, ValueError):
                    az = None
                if az is not None:
                    sector = round(az / 22.5) % 16
                    caz = c_azimuth.setdefault(insee, {})
                    caz[sector] = caz.get(sector, 0) + 1

            # Source-combination × power-class breakdown (région-level chart
            # pair) — restricted to the DPVM/FRPV/OSM taxonomy used by the
            # paper's own Figure 5; manual corrections (3), recall samples
            # (4) and third-party projects (6+) are excluded from this
            # specific combination breakdown, same as in the paper.
            if present_012:
                combo_key = "+".join(str(x) for x in sorted(present_012))
                cd = combo.setdefault(dpt, {}).setdefault(pc, {})
                cd[combo_key] = cd.get(combo_key, 0) + 1
                if insee:
                    ccd = c_combo.setdefault(insee, {}).setdefault(pc, {})
                    ccd[combo_key] = ccd.get(combo_key, 0) + 1

            # Hexbin assignment — only for installations with both a usable
            # coordinate and a known year (the temporal slider has nothing
            # to show an undated point). Geometry may be a Point (centroid)
            # or a Polygon (building footprint, averaged to its centroid) —
            # handled defensively since this hasn't been run against the
            # real production dump yet (see build notes).
            grid = hex_grids.get(dpt)
            if grid and year is not None:
                lonlat = _feature_centroid(feat.get("geometry"))
                if lonlat:
                    hex_id = assign_point_to_hex(grid, lonlat[0], lonlat[1])
                    if hex_id:
                        hd = hex_by_dept.setdefault(dpt, {}).setdefault(hex_id, {})
                        hd[year] = hd.get(year, 0.0) + kwp

            if n_sources_this_feature:
                ms = multi_source.setdefault(dpt, [0, 0])
                ms[0] += 1
                if n_sources_this_feature > 2:
                    ms[1] += 1

    print(f"  {n_lines} lines, {n_features} features, {len(capacity)} départements, "
          f"{len(c_capacity)} communes with detections")

    ranks = rank_by_value_desc({dpt: kwp for dpt, (_, kwp) in capacity.items()})

    capacity_rows = [
        {"dpt": dpt, "n_systems": n, "total_kwp": kwp, "rank_by_capacity": ranks[dpt]}
        for dpt, (n, kwp) in capacity.items()
    ]
    yearly_rows = [
        {"dpt": dpt, "year": year, "n_systems": n, "total_kwp": kwp}
        for dpt, years in yearly.items() for year, (n, kwp) in years.items()
    ]
    source_rows = [
        {"dpt": dpt, "source_id": sid, "n_systems": n}
        for dpt, srcs in source.items() for sid, n in srcs.items()
    ]

    # Rank each commune within its own département (not nationally — "3rd
    # biggest in your département" is the meaningful, locally-relevant stat
    # for a small commune, unlike a national rank out of 34,000).
    by_dept_kwp = {}
    for insee, (dpt, n, kwp) in c_capacity.items():
        by_dept_kwp.setdefault(dpt, {})[insee] = kwp
    commune_ranks = {}
    for dpt, kwp_by_insee in by_dept_kwp.items():
        commune_ranks[dpt] = rank_by_value_desc(kwp_by_insee)

    commune_capacity_rows = [
        {"insee": insee, "dpt": dpt, "n_systems": n, "total_kwp": kwp,
         "rank_in_dept": commune_ranks[dpt][insee], "n_in_dept": len(by_dept_kwp[dpt])}
        for insee, (dpt, n, kwp) in c_capacity.items()
    ]
    commune_yearly_rows = [
        {"insee": insee, "year": year, "n_systems": n, "total_kwp": kwp}
        for insee, years in c_yearly.items() for year, (n, kwp) in years.items()
    ]
    commune_source_rows = [
        {"insee": insee, "source_id": sid, "n_systems": n}
        for insee, srcs in c_source.items() for sid, n in srcs.items()
    ]

    return (capacity_rows, yearly_rows, source_rows,
            commune_capacity_rows, commune_yearly_rows, commune_source_rows, combo,
            hex_by_dept, hex_grids, multi_source, c_combo, c_power_class, c_azimuth)


def fetch_top_cities(dept_code, limit=CITIES_PER_DEPT):
    """The `limit` largest communes (by population) in a département, via the
    free geo.api.gouv.fr reference API. Returns ([], 0) on any failure
    (missing dept, network hiccup, rate limit) so a single bad lookup can't
    abort the whole build — the page just gets no cities section that run.
    Also returns the département's total population — geo.api.gouv.fr has
    no région/département-level population field (checked directly), but
    this endpoint already returns every commune before truncation, so
    summing them gives an accurate total for free, no extra request."""
    qs = urllib.parse.urlencode({"fields": "nom,code,population,centre", "format": "json"})
    url = f"{GEO_API_BASE}/departements/{dept_code}/communes?{qs}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            communes = json.load(r)
    except Exception as e:
        print(f"  ! city lookup failed for {dept_code}: {e}")
        return [], 0
    communes = [c for c in communes if c.get("population")]
    total_population = sum(c["population"] for c in communes)
    communes.sort(key=lambda c: -c["population"])
    return communes[:limit], total_population


def slugify(s):
    """ASCII, hyphenated, lowercase — used for each région's page filename
    (content/regions/{slug}.html), linked from every département page's
    'back to région' link and from the homepage's régions ranking."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


# ─── Inline SVG charts (hand-generated, no matplotlib/JS chart lib — see
# the dept-pages planning discussion: build-time, self-contained, themed to
# match the site exactly via plain fills rather than a client-side library
# or committed raster images). ────────────────────────────────────────────

def svg_yearly_chart(yearly, width=280, height=100):
    """Bar chart of detection count by first-seen year."""
    if not yearly:
        return ""
    pad_l, pad_b, pad_t = 4, 18, 6
    plot_w, plot_h = width - pad_l * 2, height - pad_b - pad_t
    years = sorted(yearly.keys())
    max_n = max(yearly.values()) or 1
    n = len(years)
    bar_w = plot_w / n * 0.62
    gap = plot_w / n
    bars, labels = [], []
    for i, y in enumerate(years):
        v = yearly[y]
        bh = (v / max_n) * plot_h
        x = pad_l + i * gap + (gap - bar_w) / 2
        by = pad_t + (plot_h - bh)
        bars.append(
            f'<rect x="{x:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
            f'rx="1.5" fill="{CHART_PURPLE}"><title>{y}: {v:,} systems</title></rect>'
        )
        if n <= 8 or i % max(1, n // 6) == 0 or i == n - 1:
            labels.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{height - 4}" font-size="8" '
                f'fill="{CHART_TEXT}" text-anchor="middle">{y}</text>'
            )
    baseline = f'<line x1="{pad_l}" y1="{pad_t + plot_h:.1f}" x2="{width - pad_l}" y2="{pad_t + plot_h:.1f}" stroke="{CHART_GRID}"/>'
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="Detections by year">{baseline}{"".join(bars)}{"".join(labels)}</svg>'
    )


def svg_source_chart(sources, width=280):
    """Horizontal bars, one per source actually present, sorted descending."""
    present = sorted(sources.items(), key=lambda kv: -kv[1])
    if not present:
        return ""
    total = sum(v for _, v in present) or 1
    row_h, label_w, gap = 18, 92, 6
    height = row_h * len(present) + 4
    bar_max = width - label_w - 46
    rows = []
    for i, (sid, n) in enumerate(present):
        label = SOURCE_LABELS.get(sid, f"Source {sid}")
        pct = n / total
        bw = max(2, pct * bar_max)
        y = i * row_h + 2
        rows.append(
            f'<text x="0" y="{y + 10}" font-size="9" fill="{CHART_TEXT_DARK}">{label}</text>'
            f'<rect x="{label_w}" y="{y + 2}" width="{bar_max}" height="9" rx="2" fill="{CHART_GRID}"/>'
            f'<rect x="{label_w}" y="{y + 2}" width="{bw:.1f}" height="9" rx="2" fill="{CHART_NAVY}">'
            f'<title>{label}: {n:,} ({pct * 100:.0f}%)</title></rect>'
            f'<text x="{label_w + bar_max + 6}" y="{y + 10}" font-size="9" fill="{CHART_TEXT}">{pct * 100:.0f}%</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="Detections by source">{"".join(rows)}</svg>'
    )


def svg_comparison_chart(this_kwp, all_kwp, width=280, height=54):
    """This département's capacity against the national range, with an
    average marker — not a full 94-bar distribution (too noisy this small),
    just this value positioned on the min-max range with mean called out."""
    if not all_kwp:
        return ""
    lo, hi = min(all_kwp), max(all_kwp)
    avg = sum(all_kwp) / len(all_kwp)
    span = (hi - lo) or 1
    pad = 4
    track_y = 22
    track_w = width - pad * 2

    def pos(v):
        return pad + (v - lo) / span * track_w

    this_x = pos(this_kwp)
    avg_x = pos(avg)
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="Capacity compared to other départements">'
        f'<rect x="{pad}" y="{track_y}" width="{track_w:.1f}" height="6" rx="3" fill="{CHART_GRID}"/>'
        f'<rect x="{pad}" y="{track_y}" width="{max(2, this_x - pad):.1f}" height="6" rx="3" fill="{CHART_PURPLE_LIGHT}"/>'
        f'<line x1="{avg_x:.1f}" y1="{track_y - 5}" x2="{avg_x:.1f}" y2="{track_y + 11}" stroke="{CHART_TEXT}" stroke-width="1"/>'
        f'<text x="{avg_x:.1f}" y="{track_y - 8}" font-size="8" fill="{CHART_TEXT}" text-anchor="middle">avg</text>'
        f'<circle cx="{this_x:.1f}" cy="{track_y + 3}" r="5" fill="{CHART_NAVY}" stroke="white" stroke-width="1.5">'
        f'<title>{this_kwp:,.0f} kWp</title></circle>'
        f'<text x="{pad}" y="{height - 2}" font-size="8" fill="{CHART_TEXT}">{lo / 1000:.1f} MWp</text>'
        f'<text x="{width - pad}" y="{height - 2}" font-size="8" fill="{CHART_TEXT}" text-anchor="end">{hi / 1000:.1f} MWp</text>'
        f'</svg>'
    )


def _polar(cx, cy, r, angle_deg):
    a = math.radians(angle_deg - 90)
    return cx + r * math.cos(a), cy + r * math.sin(a)


def _donut_slice_path(cx, cy, r_outer, r_inner, start_deg, end_deg):
    large = 1 if (end_deg - start_deg) > 180 else 0
    x1, y1 = _polar(cx, cy, r_outer, start_deg)
    x2, y2 = _polar(cx, cy, r_outer, end_deg)
    x3, y3 = _polar(cx, cy, r_inner, end_deg)
    x4, y4 = _polar(cx, cy, r_inner, start_deg)
    return (
        f"M {x1:.1f} {y1:.1f} A {r_outer:.1f} {r_outer:.1f} 0 {large} 1 {x2:.1f} {y2:.1f} "
        f"L {x3:.1f} {y3:.1f} A {r_inner:.1f} {r_inner:.1f} 0 {large} 0 {x4:.1f} {y4:.1f} Z"
    )


SECTOR_LABELS_16 = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]


def svg_azimuth_polar(azimuth_counts, size=200):
    """Wind-rose-style polar histogram of panel orientation (azimuth,
    degrees clockwise from true north) — 16 compass sectors, each wedge's
    radius proportional to how many installations fall in it. Reuses the
    same _polar/_donut_slice_path wedge machinery built for the source-
    composition donut, just driven by direction instead of source
    combination — a pie wedge (r_inner=0) per sector rather than a ring."""
    total = sum(azimuth_counts.values())
    if not total:
        return ""
    max_n = max(azimuth_counts.values())
    cx = cy = size / 2
    r_max = size / 2 - 26  # leave room for the N/E/S/W labels
    sector_deg = 360 / 16
    wedges = []
    for i in range(16):
        n = azimuth_counts.get(i, 0)
        if not n:
            continue
        r = (n / max_n) * r_max
        start = i * sector_deg - sector_deg / 2
        end = start + sector_deg
        d = _donut_slice_path(cx, cy, r, 0, start, end)
        pct = n / total * 100
        wedges.append(
            f'<path class="chart-hover-target" d="{d}" fill="{CHART_NAVY}" fill-opacity="0.85" '
            f'stroke="#fff" stroke-width="1" '
            f'data-tip="{SECTOR_LABELS_16[i]}: {n:,} installations ({pct:.0f}%)"></path>'
        )
    rings = [
        f'<circle cx="{cx}" cy="{cy}" r="{frac * r_max:.1f}" fill="none" stroke="{CHART_GRID}" stroke-width="1"/>'
        for frac in (0.33, 0.66, 1.0)
    ]
    compass = []
    for label, ang in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
        lx, ly = _polar(cx, cy, r_max + 12, ang)
        compass.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="9" fill="{CHART_TEXT}" '
            f'text-anchor="middle" dominant-baseline="middle">{label}</text>'
        )
    return (
        f'<svg viewBox="0 0 {size} {size}" width="100%" height="{size}" '
        f'role="img" aria-label="Panel orientation (azimuth) distribution">'
        f'{"".join(rings)}{"".join(wedges)}{"".join(compass)}</svg>'
    )


def combo_legend_html(combo_counts):
    """Small color-key list shared by the donut and stacked-bar charts —
    only combinations actually present in this région, sorted descending."""
    present = sorted(combo_counts.items(), key=lambda kv: -kv[1])
    total = sum(v for _, v in present) or 1
    rows = []
    for ck, n in present:
        color = COMBO_COLORS.get(ck, "#999")
        label = COMBO_LABELS.get(ck, ck)
        pct = n / total * 100
        rows.append(
            f'<span class="combo-legend-item"><span class="combo-swatch" '
            f'style="background:{color}"></span>{label} &middot; {pct:.0f}%</span>'
        )
    return f'<div class="combo-legend">{"".join(rows)}</div>'


def svg_donut_chart(combo_counts, size=180):
    """Pie/donut of source-combination shares at région scale — the left
    panel of the paper's Figure 5, reproduced per région instead of
    nationally. combo_counts: {combo_key: n}, already rolled up across all
    power classes."""
    total = sum(combo_counts.values())
    if not total:
        return ""
    cx = cy = size / 2
    r_outer, r_inner = size / 2 - 4, size / 2 - 32
    angle = 0.0
    slices = []
    for ck, n in sorted(combo_counts.items(), key=lambda kv: -kv[1]):
        share = n / total
        end = angle + share * 360
        color = COMBO_COLORS.get(ck, "#999")
        label = COMBO_LABELS.get(ck, ck)
        d = _donut_slice_path(cx, cy, r_outer, r_inner, angle, end)
        slices.append(
            f'<path class="chart-hover-target" d="{d}" fill="{color}" '
            f'data-tip="{label}: {share * 100:.0f}%"></path>'
        )
        angle = end
    return (
        f'<svg viewBox="0 0 {size} {size}" width="100%" height="{size}" '
        f'role="img" aria-label="Source-combination shares">{"".join(slices)}</svg>'
    )


def svg_stacked_bar_by_class(combo_by_class, width=280, height=170):
    """Stacked bar (normalized to 100% per bar) of source-combination share
    within each power class P1-P5 — the right panel of Figure 5. Only power
    classes with at least one installation are drawn."""
    classes_present = [pc for pc in POWER_CLASSES if combo_by_class.get(pc)]
    if not classes_present:
        return ""
    pad_l, pad_b, pad_t = 4, 20, 6
    plot_w, plot_h = width - pad_l * 2, height - pad_b - pad_t
    n = len(classes_present)
    bar_w = plot_w / n * 0.55
    gap = plot_w / n
    combo_order = list(COMBO_COLORS.keys())
    bars, labels = [], []
    for i, pc in enumerate(classes_present):
        combos = combo_by_class[pc]
        total = sum(combos.values()) or 1
        x = pad_l + i * gap + (gap - bar_w) / 2
        y_cursor = pad_t + plot_h
        for ck in sorted(combos.keys(), key=lambda k: combo_order.index(k) if k in combo_order else 99):
            share = combos[ck] / total
            seg_h = share * plot_h
            y_cursor -= seg_h
            color = COMBO_COLORS.get(ck, "#999")
            label = COMBO_LABELS.get(ck, ck)
            bars.append(
                f'<rect class="chart-hover-target" x="{x:.1f}" y="{y_cursor:.1f}" width="{bar_w:.1f}" '
                f'height="{seg_h:.1f}" fill="{color}" '
                f'data-tip="{POWER_CLASS_LABELS[pc]} &middot; {label}: {share * 100:.0f}%"></rect>'
            )
        labels.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{height - 4}" font-size="8" '
            f'fill="{CHART_TEXT}" text-anchor="middle">{pc}</text>'
        )
    baseline = f'<line x1="{pad_l}" y1="{pad_t + plot_h:.1f}" x2="{width - pad_l}" y2="{pad_t + plot_h:.1f}" stroke="{CHART_GRID}"/>'
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="Source-combination share by power class">'
        f'{baseline}{"".join(bars)}{"".join(labels)}</svg>'
    )


def _lerp_color(t, c1, c2):
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r, g, b = round(r1 + (r2 - r1) * t), round(g1 + (g2 - g1) * t), round(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _project_lonlat(lon, lat, bbox, width, height, pad=6):
    lon_min, lon_max, lat_min, lat_max = bbox
    lat_mid = (lat_min + lat_max) / 2
    cos_lat = math.cos(math.radians(lat_mid)) or 1e-9
    dx = (lon_max - lon_min) * cos_lat or 1e-9
    dy = (lat_max - lat_min) or 1e-9
    avail_w, avail_h = width - 2 * pad, height - 2 * pad
    scale = min(avail_w / dx, avail_h / dy)
    x = pad + (lon - lon_min) * cos_lat * scale + (avail_w - dx * scale) / 2
    y = pad + (lat_max - lat) * scale + (avail_h - dy * scale) / 2
    return x, y


def _ring_path(ring, bbox, width, height, max_pts=140):
    step = max(1, len(ring) // max_pts)
    pts = ring[::step]
    coords = [_project_lonlat(lon, lat, bbox, width, height) for lon, lat in pts]
    return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in coords) + " Z"


def _geometry_path(geom, bbox, width, height):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    subpaths = []
    for poly in polys:
        for ring in poly:
            subpaths.append(_ring_path(ring, bbox, width, height))
    return " ".join(subpaths)


def svg_region_choropleth(dept_entries, width=360, height=360):
    """dept_entries: [{feature, kwp, rank, nom}] for one région's
    départements. Colors each polygon by its share of the région's total
    installed capacity (lighter = smaller share, navy = largest), with a
    <title> hover showing name/capacity/rank — same zero-JS hover pattern as
    the other build-time SVG charts on this page, rather than an embedded
    tile-based map (no external tiles/JS needed for a small, static
    thumbnail of ~5-12 polygons)."""
    if not dept_entries:
        return ""
    all_lons, all_lats = [], []
    for e in dept_entries:
        geom = e["feature"]["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            for ring in poly:
                for lon, lat in ring:
                    all_lons.append(lon)
                    all_lats.append(lat)
    bbox = (min(all_lons), max(all_lons), min(all_lats), max(all_lats))
    total_kwp = sum(e["kwp"] for e in dept_entries) or 1
    max_share = max((e["kwp"] / total_kwp for e in dept_entries), default=1) or 1
    paths = []
    for e in sorted(dept_entries, key=lambda e: -e["kwp"]):
        share = e["kwp"] / total_kwp
        color = _lerp_color(share / max_share, "#dbe4ec", "#1b2733")
        d = _geometry_path(e["feature"]["geometry"], bbox, width, height)
        cap_fmt = f"{e['kwp'] / 1000:.1f} MWp" if e["kwp"] >= 1000 else f"{e['kwp']:,.0f} kWp"
        paths.append(
            f'<path d="{d}" fill="{color}" stroke="#fff" stroke-width="1.2" fill-rule="evenodd">'
            f'<title>{e["nom"]}: {cap_fmt} &middot; #{e["rank"]} by installed capacity in the r&eacute;gion</title></path>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="Installed capacity share by d&eacute;partement">{"".join(paths)}</svg>'
    )


# ─── Département hexbin engine ──────────────────────────────────────────
# A build-time-only, no-server, no-tiles temporal choropleth: each
# département gets its own flat-top hex grid (clipped to its real boundary,
# not just its bounding box), each installation is assigned to the nearest
# hex, and per-hex cumulative capacity is tracked per year. The page embeds
# the (small) precomputed per-year lookup as JSON and draws the hex grid
# once as static SVG — a year slider only recolors already-drawn hexagons,
# no recomputation or network request at view time.
HEX_TARGET_COLS = 32  # ~32 hexagons across a département's bounding box


def _feature_centroid(geometry):
    """(lon, lat) centroid of a Feature's geometry — Point returns its own
    coordinate, Polygon/MultiPolygon average their exterior ring. Returns
    None for anything unrecognized rather than raising, since the real
    production geometry type hasn't been confirmed against this code yet."""
    if not geometry:
        return None
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        return None
    try:
        if gtype == "Point":
            return coords[0], coords[1]
        if gtype == "MultiPoint" or gtype == "LineString":
            pts = coords
        elif gtype == "Polygon":
            pts = coords[0]
        elif gtype == "MultiPolygon":
            pts = coords[0][0]
        else:
            return None
        lons = [p[0] for p in pts]
        lats = [p[1] for p in pts]
        return sum(lons) / len(lons), sum(lats) / len(lats)
    except (TypeError, IndexError, ZeroDivisionError):
        return None


def _point_in_rings(x, y, rings):
    """Ray-casting point-in-polygon, XOR'd across every ring (exterior +
    holes) — correct for the union of an arbitrary ring set without needing
    to know which rings are holes."""
    inside = False
    for ring in rings:
        n = len(ring)
        j = n - 1
        for i in range(n):
            xi, yi = ring[i]
            xj, yj = ring[j]
            if (yi > y) != (yj > y):
                x_at_y = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
                if x < x_at_y:
                    inside = not inside
            j = i
    return inside


def _hex_corners(cx, cy, size):
    """Flat-top hexagon corners (angles 0/60/120/180/240/300°)."""
    return [
        (cx + size * math.cos(math.radians(60 * i)), cy + size * math.sin(math.radians(60 * i)))
        for i in range(6)
    ]


def _hex_round(q, r):
    x, z = q, r
    y = -x - z
    rx, ry, rz = round(x), round(y), round(z)
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return int(rx), int(rz)


def build_dept_hex_grid(feature, target_cols=HEX_TARGET_COLS):
    """Precompute one département's hex grid in a projected (x, y) plane
    (x = lon * cos(mean_lat), y = lat — same equirectangular approximation
    as the région choropleth). Returns None if the département is too small
    or degenerate to grid usefully. Returned dict: cos_lat, hex_size,
    x_min/y_max (grid origin), rings (projected boundary, for point tests),
    hexes: {hex_id: (cx, cy)} restricted to cells whose center falls inside
    the département's real boundary."""
    geom = feature["geometry"]
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    lats = [lat for poly in polys for ring in poly for lon, lat in ring]
    lons = [lon for poly in polys for ring in poly for lon, lat in ring]
    if not lats:
        return None
    lat_mid = (min(lats) + max(lats)) / 2
    cos_lat = math.cos(math.radians(lat_mid)) or 1e-9

    rings_xy = [[(lon * cos_lat, lat) for lon, lat in ring] for poly in polys for ring in poly]
    xs = [x for ring in rings_xy for x, y in ring]
    ys = [y for ring in rings_xy for x, y in ring]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    span = max(x_max - x_min, 1e-9)
    hex_size = span / target_cols
    if hex_size <= 0:
        return None

    hexes = {}
    # Cover the bbox generously (extra ring of margin) then keep only
    # centers that actually fall inside the département's boundary. q comes
    # straight from inverting cx0(q) = x_min + 1.5*size*q; r depends on q
    # too (axial rows are offset by q/2), so its range is recomputed inside
    # the q loop rather than reused as a single fixed range.
    margin = 1  # extra hex-widths of margin on every side
    q_min = math.floor((-margin * hex_size) / (1.5 * hex_size)) - 1
    q_max = math.ceil(((x_max - x_min) + margin * hex_size) / (1.5 * hex_size)) + 1
    for q in range(q_min, q_max + 1):
        cx0 = x_min + hex_size * 1.5 * q
        if cx0 - hex_size > x_max or cx0 + hex_size < x_min:
            continue
        r_lo = (-margin * hex_size) / (hex_size * math.sqrt(3)) - q / 2
        r_hi = ((y_max - y_min) + margin * hex_size) / (hex_size * math.sqrt(3)) - q / 2
        r_min, r_max = math.floor(r_lo) - 1, math.ceil(r_hi) + 1
        for r in range(r_min, r_max + 1):
            cy0 = hex_size * math.sqrt(3) * (r + q / 2)
            cx, cy = cx0, y_min + cy0
            if cy < y_min - hex_size or cy > y_max + hex_size:
                continue
            if _point_in_rings(cx, cy, rings_xy):
                hexes[f"{q}_{r}"] = (cx, cy)

    if not hexes:
        return None
    return {
        "cos_lat": cos_lat, "hex_size": hex_size,
        "x_min": x_min, "y_min": y_min,
        "hexes": hexes,
        "rings_xy": rings_xy,
        "bbox": (x_min, x_max, y_min, y_max),
    }


def assign_point_to_hex(grid, lon, lat):
    """Nearest valid hex id for a (lon, lat), or None if the point falls
    outside every hex this département actually has (edge of the boundary,
    or a coordinate slightly outside the département polygon)."""
    x = lon * grid["cos_lat"]
    y = lat
    size = grid["hex_size"]
    q = (2 / 3 * (x - grid["x_min"])) / size
    r = (-1 / 3 * (x - grid["x_min"]) + math.sqrt(3) / 3 * (y - grid["y_min"])) / size
    qi, ri = _hex_round(q, r)
    hex_id = f"{qi}_{ri}"
    return hex_id if hex_id in grid["hexes"] else None


def svg_dept_hexbin(grid, hex_year_kwp, years, cities=None, width=340, height=340):
    """Static hex grid SVG, one <path> per hex with data-values (a comma
    list of that hex's cumulative-capacity SHARE at each year in `years`,
    already normalized so every year's shares sum to ~1 across the grid) —
    the year slider just re-reads this attribute client-side, no rebuild.
    Also draws a light basemap underneath the hexes (the département's own
    boundary outline, from `grid["rings_xy"]`) and, on top of the hexes, up
    to 5 labeled city markers so the map reads as a real place rather than
    an abstract grid."""
    if not grid or not grid["hexes"]:
        return "", "[]"
    hexes = grid["hexes"]
    size = grid["hex_size"]
    # Project against the full département boundary's bbox (not just the
    # hexes' own extent) so the boundary outline and city markers share the
    # exact same coordinate frame as the hex grid.
    bx_min, bx_max, by_min, by_max = grid.get("bbox", (None, None, None, None))
    if bx_min is None:
        xs = [c[0] for c in hexes.values()]
        ys = [c[1] for c in hexes.values()]
        bx_min, bx_max, by_min, by_max = min(xs), max(xs), min(ys), max(ys)
    pad = size * 0.6
    x_min, x_max = bx_min - pad, bx_max + pad
    y_min, y_max = by_min - pad, by_max + pad
    span_x, span_y = (x_max - x_min) or 1, (y_max - y_min) or 1
    scale = min((width - 12) / span_x, (height - 12) / span_y)
    off_x = (width - span_x * scale) / 2 - x_min * scale
    off_y = (height - span_y * scale) / 2
    cos_lat = grid.get("cos_lat") or 1e-9

    def project(x, y):
        # y is latitude, which increases northward, but SVG's y-axis
        # increases downward — flip it (y_max - y) so north ends up at the
        # top of the image instead of the bottom.
        return x * scale + off_x, (y_max - y) * scale + off_y

    # Cumulative kWp per hex per year, then converted to a per-year share
    # (the number shown in the tooltip). The fill color, though, is driven
    # by log1p(cumulative kWp) rather than the raw share — most cells stay
    # small relative to a handful of hotspots, so a linear share scale barely
    # moves; log1p keeps the small/medium cells visually distinguishable
    # while still saturating toward the true hotspots.
    cum = {hid: 0.0 for hid in hexes}
    per_year_totals = []
    year_hex_values = {hid: [] for hid in hexes}
    year_hex_kwp = {hid: [] for hid in hexes}
    for y in years:
        for hid in hexes:
            cum[hid] += hex_year_kwp.get(hid, {}).get(y, 0.0)
        total = sum(cum.values()) or 1.0
        per_year_totals.append(total)
        for hid in hexes:
            year_hex_values[hid].append(cum[hid] / total)
            year_hex_kwp[hid].append(cum[hid])

    # abs() as a safety net: a handful of installations in the real dump
    # carry a negative kWp (a data-quality artifact upstream, not a real
    # negative capacity), which would otherwise send a cumulative sum below
    # -1 here and make log1p raise a domain error.
    max_log_kwp = max(
        (math.log1p(abs(v)) for vals in year_hex_kwp.values() for v in vals), default=0
    ) or 1

    # Boundary outline, drawn first so it sits under the hex layer as a
    # light basemap (just the département's own silhouette).
    boundary_paths = []
    for ring in grid.get("rings_xy", []):
        if len(ring) < 3:
            continue
        pts = [project(x, y) for x, y in ring]
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts) + " Z"
        boundary_paths.append(f'<path d="{d}" fill="#f4f6f8" stroke="#c7d0d8" stroke-width="1"></path>')

    paths = []
    for hid, (cx, cy) in hexes.items():
        px, py = project(cx, cy)
        corners = _hex_corners(px, py, size * scale * 0.94)
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in corners) + " Z"
        vals = year_hex_values[hid]
        # Colors precomputed server-side per year (not just the initial
        # share) so the slider only has to swap a fill attribute — no color
        # interpolation math needed client-side. Log1p'd against the
        # cumulative kWp (not the share) so the color scale isn't dominated
        # by one or two hotspot cells.
        colors = [
            _lerp_color(math.log1p(abs(kwp)) / max_log_kwp, "#e4ebf1", "#1b2733")
            for kwp in year_hex_kwp[hid]
        ]
        last_color = colors[-1] if colors else "#e4ebf1"
        values_attr = ",".join(f"{v * 100:.1f}" for v in vals)
        colors_attr = ",".join(colors)
        paths.append(
            f'<path class="hexbin-cell chart-hover-target" data-values="{values_attr}" '
            f'data-colors="{colors_attr}" '
            f'd="{d}" fill="{last_color}" fill-opacity="0.92" stroke="#fff" stroke-width="1"></path>'
        )

    # Up to 5 largest cities, drawn on top of the hexes as small labeled
    # landmarks so the map reads as a real place, not an abstract grid.
    city_markers = []
    if cities:
        ranked = sorted(
            (c for c in cities if isinstance(c.get("centre"), dict)),
            key=lambda c: c.get("population") or 0,
            reverse=True,
        )[:5]
        for c in ranked:
            coords = (c.get("centre") or {}).get("coordinates")
            if not coords or len(coords) < 2:
                continue
            lon, lat = coords[0], coords[1]
            cx, cy = lon * cos_lat, lat
            if not (x_min <= cx <= x_max and y_min <= cy <= y_max):
                continue
            px, py = project(cx, cy)
            nom = c.get("nom", "")
            city_markers.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.2" fill="#2c3e50" stroke="#fff" stroke-width="1"></circle>'
                f'<text x="{px + 5:.1f}" y="{py + 3.5:.1f}" font-size="9" fill="#2c3e50">{nom}</text>'
            )

    svg = (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="Installed capacity share by grid cell over time">'
        f'{"".join(boundary_paths)}{"".join(paths)}{"".join(city_markers)}</svg>'
    )
    return svg, json.dumps(years)


# ─── Région intro prose — several sentence-structure variants per
# paragraph, picked deterministically from the région code, so the 13
# région pages read as distinct write-ups rather than one template with
# swapped-in numbers (each variant still only exists because it's filled
# with genuinely different data — same underlying facts, different framing).
def _fmt_cap(kwp):
    return f"{kwp / 1000:.1f} MWp" if kwp >= 1000 else f"{kwp:,.0f} kWp"


def build_region_intro(region):
    code, nom = region["code"], region["nom"]
    zone = REGION_ZONE.get(code, "in France")
    population = REGION_POPULATION_2021.get(code)
    surface = REGION_SURFACE_KM2.get(code)
    gdp = REGION_GDP_2024_MEUR.get(code)
    pop_rank = rank_by_value_desc(REGION_POPULATION_2021).get(code)
    surf_rank = rank_by_value_desc(REGION_SURFACE_KM2).get(code)
    gdp_rank = rank_by_value_desc(REGION_GDP_2024_MEUR).get(code)

    top_cities = sorted(region["cities"], key=lambda c: -c["population"])[:3]
    city_names = [c["nom"] for c in top_cities]

    top_depts = sorted(region["depts"], key=lambda d: -d["kwp"])[:3]
    dept_names = [d["nom"] for d in top_depts]

    years = sorted(region["yearly"].keys()) if region["yearly"] else []
    year_span = f"{years[0]}&ndash;{years[-1]}" if years else "the observation period"

    n_fmt = f"{region['n']:,}".replace(",", " ")
    mwp = region["kwp"] / 1000
    rank_ord = _ordinal(region["rank"])

    def join_list(items):
        if len(items) <= 1:
            return items[0] if items else ""
        return ", ".join(items[:-1]) + " and " + items[-1]

    variant = int(code) % 3

    if variant == 0:
        pres = (
            f"<p>{nom} is located {zone}. As of 2021 it counted "
            f"{population:,} inhabitants (the {_ordinal(pop_rank)} most populous French r&eacute;gion), "
            f"across {surface:,} km&sup2; of land (the {_ordinal(surf_rank)} largest by area). "
            f"Its GDP stood at &euro;{gdp:,} million in 2024 (provisional), ranking it {_ordinal(gdp_rank)} nationally. "
            f"Its largest cities are {join_list(city_names)}.</p>"
        ) if city_names else (
            f"<p>{nom} is located {zone}. As of 2021 it counted "
            f"{population:,} inhabitants (the {_ordinal(pop_rank)} most populous French r&eacute;gion), "
            f"across {surface:,} km&sup2; of land (the {_ordinal(surf_rank)} largest by area), "
            f"with a GDP of &euro;{gdp:,} million in 2024 ({_ordinal(gdp_rank)} nationally).</p>"
        )
    elif variant == 1:
        pres = (
            f"<p>Situated {zone}, {nom} spans {surface:,} km&sup2; &mdash; the {_ordinal(surf_rank)} "
            f"largest of France's 13 métropole r&eacute;gions. It was home to {population:,} people "
            f"as of 2021 ({_ordinal(pop_rank)} most populous), and generated an estimated "
            f"&euro;{gdp:,} million in GDP in 2024 (ranked {_ordinal(gdp_rank)})."
            + (f" {join_list(city_names)} are among its largest cities." if city_names else "")
            + "</p>"
        )
    else:
        pres = (
            f"<p>With {population:,} inhabitants as of 2021 ({_ordinal(pop_rank)} nationally) spread "
            f"across {surface:,} km&sup2; ({_ordinal(surf_rank)} by area), {nom} lies {zone}. "
            f"Its economy was estimated at &euro;{gdp:,} million in GDP in 2024 ({_ordinal(gdp_rank)} "
            f"nationally)."
            + (f" {join_list(city_names)} rank among its largest cities." if city_names else "")
            + "</p>"
        )

    pv_variant = (int(code) + 1) % 3
    map_link = f'<a href="../map.html?region={code}">click here to see it on the map</a>'
    map_link_cap = f'<a href="../map.html?region={code}">Click here to see it on the map</a>'

    lead_verb = "lead" if len(dept_names) != 1 else "leads"
    account_verb = "account" if len(dept_names) != 1 else "accounts"

    if pv_variant == 0:
        pv = (
            f"<p>The latest OpenPVMapper release has identified {mwp:.1f} MWp of rooftop PV across "
            f"{n_fmt} installations in {nom}, making it the {rank_ord} r&eacute;gion in France by "
            f"installed capacity."
            + (f" The most equipped d&eacute;partement{'s' if len(dept_names) != 1 else ''} "
               f"{'were' if len(dept_names) != 1 else 'was'} {join_list(dept_names)}." if dept_names else "")
            + f" Detections span {year_span}. To explore every detection in {nom}, {map_link}.</p>"
        )
    elif pv_variant == 1:
        pv = (
            f"<p>{n_fmt} rooftop PV installations totalling {mwp:.1f} MWp have been identified across "
            f"{nom} by OpenPVMapper &mdash; the {rank_ord} highest installed capacity among French r&eacute;gions."
            + (f" {join_list(dept_names)} {lead_verb} the r&eacute;gion's d&eacute;partements by capacity." if dept_names else "")
            + f" These detections span {year_span}. {map_link_cap}.</p>"
        )
    else:
        pv = (
            f"<p>OpenPVMapper's latest release puts {nom} at {mwp:.1f} MWp of estimated rooftop PV "
            f"capacity over {n_fmt} installations &mdash; the {rank_ord} r&eacute;gion in France on that measure."
            + (f" {join_list(dept_names)} {account_verb} for the largest share{'s' if len(dept_names) != 1 else ''} "
               f"within the r&eacute;gion." if dept_names else "")
            + f" The detections span {year_span}; {map_link}.</p>"
        )

    return pres + "\n            " + pv


def _ordinal(n):
    if n is None:
        return "?"
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _stable_variant(code, n):
    """Deterministic 0..n-1 bucket from a département/région code — a plain
    int(code) % n for numeric codes, but Corse's "2A"/"2B" aren't numeric,
    so this falls back to a stable character-sum hash (NOT Python's built-in
    hash(), which is randomized per-process and would pick a different
    prose variant on every rebuild)."""
    try:
        return int(code) % n
    except ValueError:
        return sum(ord(c) for c in code) % n


REGION_RANK_PLACEHOLDER = "{{REGION_RANK_ORDINAL}}"


def svg_capacity_histogram(power_class_counts, width=280, height=130):
    """Plain (non-stacked) bar chart of installation counts per power-class
    bucket P1-P5 — the "sober" distribution-of-installed-capacity view
    requested for département pages, reusing the same power-class buckets
    as the région composition-by-class chart rather than a new binning.
    Drawn on a log scale (counts skew heavily toward the smallest P1/P2
    buckets, which a linear scale would flatten to invisibility)."""
    counts = {pc: power_class_counts.get(pc, 0) for pc in POWER_CLASSES}
    if not sum(counts.values()):
        return ""
    log_vals = {pc: math.log10(v + 1) for pc, v in counts.items()}
    max_log = max(log_vals.values()) or 1
    pad_l, pad_b, pad_t = 4, 20, 16
    plot_w, plot_h = width - pad_l * 2, height - pad_b - pad_t
    n = len(POWER_CLASSES)
    bar_w = plot_w / n * 0.55
    gap = plot_w / n
    bars, labels = [], []
    for i, pc in enumerate(POWER_CLASSES):
        v = counts[pc]
        bh = (log_vals[pc] / max_log) * plot_h
        x = pad_l + i * gap + (gap - bar_w) / 2
        y = pad_t + (plot_h - bh)
        bars.append(
            f'<rect class="chart-hover-target" x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{max(bh, 1):.1f}" rx="1.5" fill="{CHART_NAVY}" '
            f'data-tip="{POWER_CLASS_LABELS[pc]}: {v:,} installations"></rect>'
        )
        labels.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{height - 4}" font-size="8" '
            f'fill="{CHART_TEXT}" text-anchor="middle">{pc}</text>'
        )
    baseline = f'<line x1="{pad_l}" y1="{pad_t + plot_h:.1f}" x2="{width - pad_l}" y2="{pad_t + plot_h:.1f}" stroke="{CHART_GRID}"/>'
    scale_label = f'<text x="{pad_l}" y="10" font-size="8" fill="{CHART_TEXT}">log scale</text>'
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="Installed capacity distribution (log scale)">'
        f'{scale_label}{baseline}{"".join(bars)}{"".join(labels)}</svg>'
    )


def build_dept_intro(dept):
    """dept: a dict assembled in main()'s per-département loop — see the
    call site for its exact keys. The régional-rank ordinal isn't known yet
    at this point in the build (sibling départements in the same région
    haven't all been processed), so it's left as a literal placeholder
    string and patched in afterwards, once every région is fully
    aggregated — same technique inject_rankings already uses for the
    homepage Leaderboard."""
    code, nom = dept["code"], dept["nom"]
    variant = _stable_variant(code, 3)

    prefecture = DEPT_PREFECTURE.get(code)
    landmark = DEPT_LANDMARK.get(code)
    population = dept.get("population")

    pop_clause = f"counted {population:,} inhabitants" if population else "has an undocumented population"
    prefecture_clause = f", with {prefecture} as its pr&eacute;fecture" if prefecture else ""
    landmark_clause = f" It is known for {landmark}." if landmark else ""

    if variant == 0:
        pres = (
            f"<p>{nom} is a d&eacute;partement of {dept['region_nom']}{prefecture_clause}. "
            f"As of 2021 it {pop_clause}.{landmark_clause}</p>"
        )
    elif variant == 1:
        pres = (
            f"<p>Part of {dept['region_nom']}, {nom}{prefecture_clause} {pop_clause} as of 2021."
            f"{landmark_clause}</p>"
        )
    else:
        pres = (
            f"<p>{nom}, in {dept['region_nom']}{prefecture_clause}, {pop_clause} as of 2021."
            f"{landmark_clause}</p>"
        )

    years = dept.get("years") or []
    year_span = f"{years[0]}&ndash;{years[-1]}" if years else "the observation period"
    n_fmt = f"{dept['n']:,}".replace(",", " ")
    mwp = dept["kwp"] / 1000
    rank_ord = _ordinal(dept["rank"])
    pct_multi_source = dept.get("pct_multi_source")
    multi_source_clause = (
        f" {pct_multi_source:.0f}% of installations in {nom} are confirmed by more than "
        f"two independent sources."
        if pct_multi_source else ""
    )
    map_link = f'<a href="../map.html?dept={code}">click here to see it on the map</a>'
    map_link_cap = f'<a href="../map.html?dept={code}">Click here to see it on the map</a>'

    pv_variant = (variant + 1) % 3
    if pv_variant == 0:
        pv = (
            f"<p>The latest OpenPVMapper release has identified {mwp:.1f} MWp of rooftop PV across "
            f"{n_fmt} installations in {nom} &mdash; the {rank_ord} d&eacute;partement in France "
            f"and {REGION_RANK_PLACEHOLDER} within {dept['region_nom']} by installed capacity. "
            f"Detections span {year_span}.{multi_source_clause} To explore every detection in {nom}, {map_link}.</p>"
        )
    elif pv_variant == 1:
        pv = (
            f"<p>{n_fmt} rooftop PV installations totalling {mwp:.1f} MWp have been identified in "
            f"{nom} by OpenPVMapper, ranking it {rank_ord} nationally and {REGION_RANK_PLACEHOLDER} "
            f"in {dept['region_nom']}. These detections span {year_span}.{multi_source_clause} {map_link_cap}.</p>"
        )
    else:
        pv = (
            f"<p>OpenPVMapper's latest release puts {nom} at {mwp:.1f} MWp of estimated rooftop PV "
            f"capacity over {n_fmt} installations &mdash; {rank_ord} in France, {REGION_RANK_PLACEHOLDER} "
            f"in {dept['region_nom']}. The detections span {year_span}.{multi_source_clause} {map_link}.</p>"
        )

    return pres + "\n            " + pv


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{nom} ({code}) &middot; DeepPVMapper Data</title>

    <meta name="description" content="Rooftop PV systems detected by DeepPVMapper in {nom} ({code}): {n_fmt} systems, {mwp_fmt} MWp estimated installed capacity. Explore on the map or download the data.">
    <meta name="keywords" content="{keywords}">
    <meta name="author" content="Gabriel Kasmi">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://deeppvmapper.fr/content/data/{code}.html">
    <meta property="og:title" content="{nom} ({code}) &middot; DeepPVMapper Data">
    <meta property="og:description" content="{n_fmt} rooftop PV systems detected in {nom}, {mwp_fmt} MWp estimated installed capacity.">
    <meta property="og:image" content="https://deeppvmapper.fr/static/images/teaser.webp">
    <link rel="canonical" href="https://deeppvmapper.fr/content/data/{code}.html">

    <link rel="icon" type="image/x-icon" href="../../static/images/favicon.ico">
    <link rel="stylesheet" href="../../static/css/style.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

    <style>
        .dept-chart-card {{ text-align: left; }}
        .dept-chart-card h3 {{ text-align: left; font-size: 0.98rem; }}
        .dept-chart-card svg {{ display: block; margin-top: 10px; }}
        .dept-chart-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 22px; max-width: 900px; margin: 34px auto 0;
        }}
        .dept-cities {{ max-width: 720px; margin: 40px auto 0; }}
        .dept-cities h3 {{ text-align: center; margin-bottom: 16px; }}
        .ranking-list {{
            max-width: 720px; margin: 0 auto; max-height: 480px;
            overflow-y: auto; border: 1px solid #e9ecef; border-radius: 8px;
        }}
        .ranking-item {{
            display: flex; align-items: center; gap: 14px; padding: 10px 18px;
            text-decoration: none; color: inherit; border-bottom: 1px solid #f1f3f5;
        }}
        .ranking-item:last-child {{ border-bottom: none; }}
        .ranking-item:hover {{ background: #f8f9fa; }}
        .ranking-rank {{
            flex: none; width: 28px; height: 28px; border-radius: 50%;
            background: #e9ecef; color: #34495e; font-weight: 700; font-size: 0.85rem;
            display: flex; align-items: center; justify-content: center;
        }}
        .ranking-item.rank-1 .ranking-rank {{ background: #f4c95d; color: #5c4400; }}
        .ranking-item.rank-2 .ranking-rank {{ background: #d8dce3; color: #3a3f47; }}
        .ranking-item.rank-3 .ranking-rank {{ background: #e3b088; color: #5a3313; }}
        .ranking-main {{ flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; }}
        .ranking-name {{ font-weight: 600; font-size: 0.95rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .ranking-sub {{ font-size: 0.8rem; color: #6c757d; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .ranking-value {{
            flex: none; font-weight: 700; color: #34495e; font-variant-numeric: tabular-nums;
            font-size: 0.92rem; text-align: right;
        }}
        .dept-back-links {{ text-align: center; margin-top: 40px; display: flex; justify-content: center; gap: 18px; flex-wrap: wrap; }}
        .region-intro {{ max-width: 720px; margin: 0 auto 8px; text-align: left; }}
        .region-intro p {{ margin: 0 0 14px; line-height: 1.65; color: #495057; }}
        .region-intro p:last-child {{ margin-bottom: 0; }}
        .region-intro a {{ color: #34495e; font-weight: 600; }}
        .combo-legend {{
            display: flex; flex-wrap: wrap; gap: 8px 16px; margin-top: 12px;
            font-size: 0.78rem; color: #495057;
        }}
        .combo-legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .combo-swatch {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; flex: none; }}
        .power-class-note {{ font-size: 0.75rem; color: #6c757d; margin-top: 10px; line-height: 1.5; }}
        .chart-hover-target {{ cursor: pointer; }}
        .chart-tooltip {{
            position: fixed; pointer-events: none; background: #2c3e50; color: #fff;
            font-size: 0.78rem; padding: 6px 10px; border-radius: 6px; opacity: 0;
            transition: opacity 0.12s ease; z-index: 9999; white-space: nowrap;
        }}
        .region-map-card {{
            max-width: 460px; margin: 22px auto 0; text-align: center;
            border: 1px solid #e9ecef; border-radius: 8px; padding: 20px;
        }}
        .region-map-card h3 {{ font-size: 0.98rem; margin: 0 0 4px; }}
        .region-map-card svg {{ display: block; margin: 10px auto 0; max-width: 100%; height: auto; }}
        .region-map-note {{ font-size: 0.78rem; color: #6c757d; margin-top: 8px; }}
        .hexbin-slider-row {{
            display: flex; align-items: center; gap: 10px; margin-top: 10px;
            max-width: 280px; margin-left: auto; margin-right: auto;
        }}
        .hexbin-slider-row input[type="range"] {{ flex: 1 1 auto; }}
        .hexbin-year-label {{ font-weight: 700; color: #34495e; font-variant-numeric: tabular-nums; min-width: 3.2em; }}
    </style>
</head>
<body>

    <header class="header header--compact">
        <div class="header-background"></div>
        <nav class="topnav topnav--overlay">
            <div class="container topnav-inner">
                <a href="../../index.html" class="topnav-logo">DeepPVMapper</a>
                <div class="topnav-links">
                    <a href="../about.html" class="topnav-link">About</a>
                    <a href="../data.html" class="topnav-link is-active">Data</a>
                    <a href="../software.html" class="topnav-link">Software</a>
                    <a href="../contribute.html" class="topnav-link">Contribute</a>
                    <a href="../outlook.html" class="topnav-link">Use Cases</a>
                    <span class="topnav-break" aria-hidden="true"></span>
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
                </div>
            </div>
        </nav>
        <div class="container">
            <a href="../../index.html" class="back-home"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9.5 12 3l9 6.5"/><path d="M5 9.5V21h14V9.5"/></svg>Home</a>
            <h1 class="title">{nom}</h1>
            <p class="title-region">{region_nom}</p>
        </div>
    </header>

    <section>
        <div class="container">
            <div class="region-intro">
                {intro_html}
            </div>

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

            <div class="dept-chart-grid">
                {composition_card}
                {histogram_card}
            </div>

            {hexbin_section}

            {cities_section}

            {city_request_note}

            {report_cta}

            <div class="dept-back-links">
                <a href="../regions/{region_slug}.html" class="btn">&larr; Back to {region_nom}</a>
                <a href="../local-statistics.html" class="btn">&larr; Back to all d&eacute;partements</a>
            </div>
        </div>
    </section>

    <div class="chart-tooltip" id="chart-tooltip"></div>

    <footer class="footer">
        <div class="container">
            <p>Maintained by Gabriel Kasmi. This work is licensed under <a href="https://github.com/gabrielkasmi/deeppvmapper/blob/main/LICENSE" target="_blank">MIT</a>.</p>
        </div>
    </footer>

    <script src="../../static/js/script.js"></script>
    <script>
        (function () {{
            var tip = document.getElementById('chart-tooltip');
            if (!tip) return;
            var slider = document.getElementById('hexbin-year-slider');
            document.querySelectorAll('.chart-hover-target').forEach(function (el) {{
                el.addEventListener('mousemove', function (e) {{
                    var text;
                    if (el.classList.contains('hexbin-cell') && slider) {{
                        var vals = el.getAttribute('data-values').split(',');
                        text = vals[parseInt(slider.value, 10)] + '% of capacity that year';
                    }} else {{
                        text = el.getAttribute('data-tip') || '';
                    }}
                    tip.textContent = text;
                    tip.style.left = (e.clientX + 14) + 'px';
                    tip.style.top = (e.clientY + 14) + 'px';
                    tip.style.opacity = '1';
                }});
                el.addEventListener('mouseleave', function () {{ tip.style.opacity = '0'; }});
            }});

            if (slider) {{
                var label = document.getElementById('hexbin-year-label');
                var years = JSON.parse(document.getElementById('hexbin-years-data').textContent);
                var cells = document.querySelectorAll('.hexbin-cell');
                slider.addEventListener('input', function () {{
                    var idx = parseInt(slider.value, 10);
                    label.textContent = years[idx];
                    cells.forEach(function (c) {{
                        var colors = c.getAttribute('data-colors').split(',');
                        c.setAttribute('fill', colors[idx]);
                    }});
                }});
            }}
        }})();
    </script>
</body>
</html>
"""

CHART_CARD = """<div class="hub-card dept-chart-card">
                    <h3>{title}</h3>
                    {svg}
                </div>"""

def build_report_cta(target_label):
    """Lightweight "report an issue" CTA for département/city pages — no
    backend of its own (unlike the interactive map's Supabase-backed report
    form, static/js/map/report.js), so the button just hands the visitor's
    comment off to a mailto: link (wired up in static/js/script.js)."""
    return (
        '<div class="report-issue-card">\n'
        '                <h3>See something odd here?</h3>\n'
        "                <p>Report it &mdash; you'll help improve data quality "
        "for {label} and across France.</p>\n"
        '                <textarea id="report-comment" rows="3" '
        'placeholder="What looks wrong? A count that seems off, a system that looks misclassified..."></textarea>\n'
        '                <button type="button" id="report-issue-btn" class="btn" '
        'data-target-label="{label}">Report an issue</button>\n'
        "            </div>"
    ).format(label=target_label)


def build_city_request_note(dept_nom, code):
    """A small, low-key link under a département page's city ranking, for a
    visitor whose commune isn't listed there — either it hasn't crossed
    CITY_MIN_SYSTEMS yet, or it wasn't among the largest communes checked
    for this département (CITIES_PER_DEPT). Points to a prefilled GitHub
    issue on the same public tracker used elsewhere (openpvmapper-issues,
    see known-issues.html/contribute.html) rather than a new channel — a
    signal of demand, not an instant request: opening the issue doesn't
    make a page appear, it just tells Gabriel where to look next."""
    title = urllib.parse.quote("City request: ")
    body = urllib.parse.quote(
        f"I'd like to see a page for a specific commune in {dept_nom} ({code}) "
        "that doesn't appear in its city ranking.\n\n"
        "Commune: \n\n"
        "(It may not have enough confirmed detections yet, or wasn't among the "
        "largest communes checked for this département — this isn't an "
        "instant request, just a signal of interest.)"
    )
    url = (
        "https://github.com/gabrielkasmi/openpvmapper-issues/issues/new"
        f"?title={title}&body={body}&labels=city-request"
    )
    return (
        '<p class="city-request-note">Don&rsquo;t see your commune above? It may not have enough '
        f'confirmed detections yet &mdash; <a href="{url}" target="_blank">let us know which one you had in mind</a>.</p>'
    )


def fmt_capacity(kwp):
    """City-scale capacities are often well under 1 MWp — show kWp below
    that threshold instead of "0.3 MWp". Returns (value_str, unit)."""
    if kwp >= 1000:
        return f"{kwp / 1000:.1f}", "MWp"
    return f"{kwp:,.0f}".replace(",", " "), "kWp"


def build_city_ranking_list(cities, heading, sub_fn, limit=None):
    """A ranked-by-installed-capacity list of cities, in the same visual
    language as the homepage Rankings tabs (RANKING_ITEM rows) — used for
    both a département page's own cities ranking and a région page's cities
    tab. Only cities that cleared CITY_MIN_SYSTEMS (has_page) are rankable
    at all — there's no reliable capacity number for the rest, so unlike
    the pre-ranking version of this section, ones without a page are simply
    left out rather than shown as an unranked map-link fallback. sub_fn(c)
    builds each row's subtext (e.g. "{n} systems" on a département page, the
    département's name on a région page). Returns "" if nothing qualifies."""
    qualifying = [c for c in cities if c.get("has_page")]
    if not qualifying:
        return ""
    qualifying.sort(key=lambda c: -c["kwp"])
    if limit:
        qualifying = qualifying[:limit]
    rows = []
    for i, c in enumerate(qualifying, 1):
        cap_fmt, cap_unit = fmt_capacity(c["kwp"])
        rows.append(RANKING_ITEM.format(
            rank_class=f" rank-{i}" if i <= 3 else "",
            href=f"../cities/{c['code']}.html", rank=i, nom=c["nom"],
            sub=sub_fn(c), value=f"{cap_fmt} {cap_unit}",
        ))
    rows_html = "\n".join(rows)
    if not heading:
        # Caller already provides the scrollable wrapper (e.g. a région
        # page's tabbed .ranking-panel) — just the rows.
        return rows_html
    return (
        f'<div class="dept-cities">\n'
        f'                <h3>{heading}</h3>\n'
        f'                <div class="ranking-list">\n{rows_html}\n'
        f'                </div>\n'
        f'            </div>'
    )


def build_dept_ranking_list(depts):
    """A région page's départements, ranked by installed capacity — same
    RANKING_ITEM row style as everywhere else, just without the wrapping
    heading (the tab label already says "Départements")."""
    ranked = sorted(depts, key=lambda d: -d["mwp"])
    rows = [
        RANKING_ITEM.format(
            rank_class=f" rank-{i}" if i <= 3 else "",
            href=f"../data/{d['code']}.html", rank=i, nom=d["nom"],
            sub=f"{d['n']:,}".replace(",", " ") + " systems",
            value=f"{d['mwp']:,.1f} MWp".replace(",", " "),
        )
        for i, d in enumerate(ranked, 1)
    ]
    return "\n".join(rows)


def build_keywords(nom, code, region_nom, cities):
    """French + English keyword mix for the per-département SEO meta tag —
    the département/région name in both languages' common phrasing, plus a
    handful of its largest cities as long-tail terms."""
    city_names = ", ".join(c["nom"] for c in cities[:8])
    terms = [
        f"panneaux photovoltaïques {nom}", f"panneaux solaires {nom}",
        f"installations solaires {nom}", f"cadastre solaire {nom}",
        f"énergie solaire {code}", f"photovoltaïque {region_nom}",
        f"rooftop solar {nom}", f"solar panels {nom} France",
        f"PV installations {code}", f"photovoltaic map {nom}",
        f"solar energy {region_nom}", "DeepPVMapper",
    ]
    if city_names:
        terms.append(city_names)
    return ", ".join(terms)


def build_region_keywords(nom, dept_names, cities):
    """Same French + English mix as build_keywords, scaled up to a région:
    its member départements' names plus its largest cities as long-tail
    terms — this is the page most likely to catch a broad "solar in
    {region}" style query."""
    city_names = ", ".join(c["nom"] for c in cities[:10])
    dept_list = ", ".join(dept_names[:15])
    terms = [
        f"panneaux photovoltaïques {nom}", f"panneaux solaires {nom}",
        f"installations solaires {nom}", f"énergie solaire {nom}",
        f"rooftop solar {nom}", f"solar panels {nom} France",
        f"PV installations {nom}", f"photovoltaic map {nom}",
        f"solar energy {nom}", "DeepPVMapper",
    ]
    if dept_list:
        terms.append(dept_list)
    if city_names:
        terms.append(city_names)
    return ", ".join(terms)


def build_city_keywords(nom, dept_nom, region_nom):
    """French + English keyword mix for a per-city SEO meta tag — the exact
    shape of query this whole feature exists for ("panneaux solaires
    bordeaux"), plus its département/région for broader long-tail terms."""
    terms = [
        f"panneaux solaires {nom}", f"panneaux photovoltaïques {nom}",
        f"installations solaires {nom}", f"cadastre solaire {nom}",
        f"rooftop solar {nom}", f"solar panels {nom} France",
        f"photovoltaic {nom}", f"solar energy {nom}",
        f"PV installations {dept_nom}", f"panneaux solaires {dept_nom}",
        f"{region_nom}", "DeepPVMapper",
    ]
    return ", ".join(terms)


CITY_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{nom} ({dept_nom}) &middot; DeepPVMapper Data</title>

    <meta name="description" content="Rooftop PV systems detected by DeepPVMapper in {nom} ({dept_nom}): {n_fmt} systems, {cap_fmt} {cap_unit} estimated installed capacity. Explore on the map or download the data.">
    <meta name="keywords" content="{keywords}">
    <meta name="author" content="Gabriel Kasmi">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://deeppvmapper.fr/content/cities/{insee}.html">
    <meta property="og:title" content="{nom} ({dept_nom}) &middot; DeepPVMapper Data">
    <meta property="og:description" content="{n_fmt} rooftop PV systems detected in {nom}, {cap_fmt} {cap_unit} estimated installed capacity.">
    <meta property="og:image" content="https://deeppvmapper.fr/static/images/teaser.webp">
    <link rel="canonical" href="https://deeppvmapper.fr/content/cities/{insee}.html">

    <link rel="icon" type="image/x-icon" href="../../static/images/favicon.ico">
    <link rel="stylesheet" href="../../static/css/style.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

    <style>
        .dept-chart-card {{ text-align: left; }}
        .dept-chart-card h3 {{ text-align: left; font-size: 0.98rem; }}
        .dept-chart-card svg {{ display: block; margin-top: 10px; }}
        .dept-chart-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 22px; max-width: 900px; margin: 34px auto 0;
        }}
        .dept-back-links {{ text-align: center; margin-top: 40px; display: flex; justify-content: center; gap: 18px; flex-wrap: wrap; }}
        .region-intro {{ max-width: 720px; margin: 0 auto 8px; text-align: left; }}
        .region-intro p {{ margin: 0 0 14px; line-height: 1.65; color: #495057; }}
        .region-intro p:last-child {{ margin-bottom: 0; }}
        .region-intro a {{ color: #34495e; font-weight: 600; }}
        .combo-legend {{
            display: flex; flex-wrap: wrap; gap: 8px 16px; margin-top: 12px;
            font-size: 0.78rem; color: #495057;
        }}
        .combo-legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .combo-swatch {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; flex: none; }}
        .power-class-note {{ font-size: 0.75rem; color: #6c757d; margin-top: 10px; line-height: 1.5; }}
        .chart-hover-target {{ cursor: pointer; }}
        .chart-tooltip {{
            position: fixed; pointer-events: none; background: #2c3e50; color: #fff;
            font-size: 0.78rem; padding: 6px 10px; border-radius: 6px; opacity: 0;
            transition: opacity 0.12s ease; z-index: 9999; white-space: nowrap;
        }}
    </style>
</head>
<body>

    <header class="header header--compact">
        <div class="header-background"></div>
        <nav class="topnav topnav--overlay">
            <div class="container topnav-inner">
                <a href="../../index.html" class="topnav-logo">DeepPVMapper</a>
                <div class="topnav-links">
                    <a href="../about.html" class="topnav-link">About</a>
                    <a href="../data.html" class="topnav-link is-active">Data</a>
                    <a href="../software.html" class="topnav-link">Software</a>
                    <a href="../contribute.html" class="topnav-link">Contribute</a>
                    <a href="../outlook.html" class="topnav-link">Use Cases</a>
                    <span class="topnav-break" aria-hidden="true"></span>
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
                </div>
            </div>
        </nav>
        <div class="container">
            <a href="../../index.html" class="back-home"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9.5 12 3l9 6.5"/><path d="M5 9.5V21h14V9.5"/></svg>Home</a>
            <h1 class="title">{nom}</h1>
            <p class="title-region">{dept_nom} &middot; {region_nom}</p>
        </div>
    </header>

    <section>
        <div class="container">
            <div class="region-intro">
                {intro_html}
            </div>

            <div class="validation-stats">
                <div class="stat-item">
                    <span class="stat-number">{n_fmt}</span>
                    <span class="stat-label">PV systems detected</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{cap_fmt} {cap_unit}</span>
                    <span class="stat-label">Estimated installed capacity</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">#{rank}</span>
                    <span class="stat-label">by installed capacity, out of {n_in_dept} communes in {dept_nom}</span>
                </div>
            </div>

            <div class="dept-chart-grid">
                {composition_card}
                {histogram_card}
                {azimuth_card}
            </div>

            {report_cta}

            <div class="dept-back-links">
                <a href="../data/{dept_code}.html" class="btn">&larr; Back to {dept_nom}</a>
                <a href="../regions/{region_slug}.html" class="btn">&larr; Back to {region_nom}</a>
            </div>
        </div>
    </section>

    <div class="chart-tooltip" id="chart-tooltip"></div>

    <footer class="footer">
        <div class="container">
            <p>Maintained by Gabriel Kasmi. This work is licensed under <a href="https://github.com/gabrielkasmi/deeppvmapper/blob/main/LICENSE" target="_blank">MIT</a>.</p>
        </div>
    </footer>

    <script src="../../static/js/script.js"></script>
    <script>
        (function () {{
            var tip = document.getElementById('chart-tooltip');
            if (!tip) return;
            document.querySelectorAll('.chart-hover-target').forEach(function (el) {{
                el.addEventListener('mousemove', function (e) {{
                    tip.textContent = el.getAttribute('data-tip') || '';
                    tip.style.left = (e.clientX + 14) + 'px';
                    tip.style.top = (e.clientY + 14) + 'px';
                    tip.style.opacity = '1';
                }});
                el.addEventListener('mouseleave', function () {{ tip.style.opacity = '0'; }});
            }});
        }})();
    </script>
</body>
</html>
"""


def build_city_intro(city):
    """city: a dict assembled at build_city_page's call site — nom, insee,
    dept_nom, dept_code, region_nom, population, n, kwp, rank_in_dept,
    n_in_dept, years. Same deterministic-variant technique as
    build_dept_intro/build_region_intro (_stable_variant keyed off the
    insee code), just without a préfecture/landmark clause — a commune has
    no équivalent of those, and no régional rank of its own (only a rank
    within its own département, already known at this point in the build,
    so no placeholder/patch-up pass is needed here)."""
    nom, insee = city["nom"], city["insee"]
    dept_nom, region_nom = city["dept_nom"], city["region_nom"]
    variant = _stable_variant(insee, 3)
    population = city.get("population")
    pop_clause = f"counted {population:,} inhabitants" if population else "has an undocumented population"

    if variant == 0:
        pres = f"<p>{nom} is a commune of {dept_nom}, in {region_nom}. As of the latest census it {pop_clause}.</p>"
    elif variant == 1:
        pres = f"<p>Part of {dept_nom} ({region_nom}), {nom} {pop_clause} as of the latest census.</p>"
    else:
        pres = f"<p>{nom}, in {dept_nom} ({region_nom}), {pop_clause} as of the latest census.</p>"

    years = city.get("years") or []
    year_span = f"{years[0]}&ndash;{years[-1]}" if years else "the observation period"
    n_fmt = f"{city['n']:,}".replace(",", " ")
    cap_fmt, cap_unit = fmt_capacity(city["kwp"])
    rank_ord = _ordinal(city["rank_in_dept"])
    map_link = f'<a href="../map.html?insee={insee}&dept={city["dept_code"]}">click here to see it on the map</a>'
    map_link_cap = f'<a href="../map.html?insee={insee}&dept={city["dept_code"]}">Click here to see it on the map</a>'

    pv_variant = (variant + 1) % 3
    if pv_variant == 0:
        pv = (
            f"<p>OpenPVMapper has identified {n_fmt} rooftop PV installations in {nom}, totalling "
            f"{cap_fmt} {cap_unit} &mdash; the {rank_ord} commune in {dept_nom} by installed capacity, "
            f"out of {city['n_in_dept']} communes with detections. Detections span {year_span}. "
            f"To explore every detection in {nom}, {map_link}.</p>"
        )
    elif pv_variant == 1:
        pv = (
            f"<p>{n_fmt} rooftop PV installations totalling {cap_fmt} {cap_unit} have been identified "
            f"in {nom} by OpenPVMapper, ranking it {rank_ord} in {dept_nom} out of {city['n_in_dept']} "
            f"communes with detections. These detections span {year_span}. {map_link_cap}.</p>"
        )
    else:
        pv = (
            f"<p>OpenPVMapper's latest release puts {nom} at {cap_fmt} {cap_unit} of estimated rooftop "
            f"PV capacity over {n_fmt} installations &mdash; {rank_ord} in {dept_nom} out of "
            f"{city['n_in_dept']} communes with detections. The detections span {year_span}. {map_link}.</p>"
        )

    return pres + "\n            " + pv


def build_city_page(city, cstats, dept_nom, dept_code, region_nom, region_slug,
                     yearly, combo, power_class_counts, azimuth_counts):
    """city: {nom, code(insee), population, dept_code}. cstats: the
    commune_capacity_rows entry (n_systems/total_kwp/rank_in_dept/n_in_dept).
    yearly: this commune's own year->[n,kwp] breakdown — no chart of its own
    anymore, just used for the intro's year span. combo/power_class_counts/
    azimuth_counts: this commune's own combo_by_commune / power_class_by_commune
    / azimuth_by_commune entries, feeding the source-composition donut, the
    capacity histogram, and the orientation rose respectively."""
    nom, insee = city["nom"], city["code"]
    kwp = cstats["total_kwp"]
    cap_fmt, cap_unit = fmt_capacity(kwp)

    combo_totals = {}
    for combos in combo.values():
        for ck, cnt in combos.items():
            combo_totals[ck] = combo_totals.get(ck, 0) + cnt
    composition_card = (
        CHART_CARD.format(
            title="Source composition",
            svg=svg_donut_chart(combo_totals) + combo_legend_html(combo_totals),
        ) if combo_totals else ""
    )
    histogram_card = (
        CHART_CARD.format(
            title="Installed capacity distribution",
            svg=svg_capacity_histogram(power_class_counts),
        ) if power_class_counts else ""
    )
    azimuth_card = (
        CHART_CARD.format(
            title="Panel orientation",
            svg=svg_azimuth_polar(azimuth_counts),
        ) if azimuth_counts else ""
    )

    city_ctx = {
        "nom": nom, "insee": insee, "dept_nom": dept_nom, "dept_code": dept_code,
        "region_nom": region_nom, "population": city.get("population"),
        "n": cstats["n_systems"], "kwp": kwp,
        "rank_in_dept": cstats["rank_in_dept"], "n_in_dept": cstats["n_in_dept"],
        "years": sorted(yearly.keys()) if yearly else [],
    }

    return CITY_TEMPLATE.format(
        nom=nom, insee=insee, dept_nom=dept_nom, dept_code=dept_code,
        region_nom=region_nom, region_slug=region_slug,
        keywords=build_city_keywords(nom, dept_nom, region_nom),
        n_fmt=f"{cstats['n_systems']:,}".replace(",", " "),
        cap_fmt=cap_fmt, cap_unit=cap_unit,
        rank=cstats["rank_in_dept"], n_in_dept=cstats["n_in_dept"],
        intro_html=build_city_intro(city_ctx),
        composition_card=composition_card, histogram_card=histogram_card, azimuth_card=azimuth_card,
        report_cta=build_report_cta(f"{nom} ({insee})"),
    )


REGION_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{nom} &middot; DeepPVMapper Data</title>

    <meta name="description" content="Rooftop PV systems detected by DeepPVMapper across {nom}: {n_fmt} systems, {mwp_fmt} MWp estimated installed capacity over {n_depts} d&eacute;partements. Explore on the map or download the data.">
    <meta name="keywords" content="{keywords}">
    <meta name="author" content="Gabriel Kasmi">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://deeppvmapper.fr/content/regions/{slug}.html">
    <meta property="og:title" content="{nom} &middot; DeepPVMapper Data">
    <meta property="og:description" content="{n_fmt} rooftop PV systems detected across {nom}, {mwp_fmt} MWp estimated installed capacity.">
    <meta property="og:image" content="https://deeppvmapper.fr/static/images/teaser.webp">
    <link rel="canonical" href="https://deeppvmapper.fr/content/regions/{slug}.html">

    <link rel="icon" type="image/x-icon" href="../../static/images/favicon.ico">
    <link rel="stylesheet" href="../../static/css/style.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

    <style>
        .dept-chart-card {{ text-align: left; }}
        .dept-chart-card h3 {{ text-align: left; font-size: 0.98rem; }}
        .dept-chart-card svg {{ display: block; margin-top: 10px; }}
        .dept-chart-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 22px; max-width: 900px; margin: 34px auto 0;
        }}
        .dept-cities {{ max-width: 720px; margin: 40px auto 0; }}
        .dept-cities h3 {{ text-align: center; margin-bottom: 16px; }}
        .dept-back-links {{ text-align: center; margin-top: 40px; display: flex; justify-content: center; gap: 18px; flex-wrap: wrap; }}
        .rankings-tabs {{
            display: flex; justify-content: center; gap: 8px; margin: 8px 0 20px; flex-wrap: wrap;
        }}
        .rankings-tab {{
            font: inherit; font-size: 0.92rem; font-weight: 600; cursor: pointer;
            padding: 8px 20px; border-radius: 6px; border: 1px solid #dee2e6;
            background: #fff; color: #34495e;
        }}
        .rankings-tab.is-active {{ background: #34495e; border-color: #34495e; color: #fff; }}
        .ranking-panel, .ranking-list {{
            max-width: 720px; margin: 0 auto; max-height: 480px;
            overflow-y: auto; border: 1px solid #e9ecef; border-radius: 8px;
        }}
        .ranking-panel {{ display: none; }}
        .ranking-panel.is-active {{ display: block; }}
        .ranking-item {{
            display: flex; align-items: center; gap: 14px; padding: 10px 18px;
            text-decoration: none; color: inherit; border-bottom: 1px solid #f1f3f5;
        }}
        .ranking-item:last-child {{ border-bottom: none; }}
        .ranking-item:hover {{ background: #f8f9fa; }}
        .ranking-rank {{
            flex: none; width: 28px; height: 28px; border-radius: 50%;
            background: #e9ecef; color: #34495e; font-weight: 700; font-size: 0.85rem;
            display: flex; align-items: center; justify-content: center;
        }}
        .ranking-item.rank-1 .ranking-rank {{ background: #f4c95d; color: #5c4400; }}
        .ranking-item.rank-2 .ranking-rank {{ background: #d8dce3; color: #3a3f47; }}
        .ranking-item.rank-3 .ranking-rank {{ background: #e3b088; color: #5a3313; }}
        .ranking-main {{ flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; }}
        .ranking-name {{ font-weight: 600; font-size: 0.95rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .ranking-sub {{ font-size: 0.8rem; color: #6c757d; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .ranking-value {{
            flex: none; font-weight: 700; color: #34495e; font-variant-numeric: tabular-nums;
            font-size: 0.92rem; text-align: right;
        }}
        .region-intro {{ max-width: 720px; margin: 0 auto 8px; text-align: left; }}
        .region-intro p {{ margin: 0 0 14px; line-height: 1.65; color: #495057; }}
        .region-intro p:last-child {{ margin-bottom: 0; }}
        .region-intro a {{ color: #34495e; font-weight: 600; }}
        .combo-legend {{
            display: flex; flex-wrap: wrap; gap: 8px 16px; margin-top: 12px;
            font-size: 0.78rem; color: #495057;
        }}
        .combo-legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .combo-swatch {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; flex: none; }}
        .power-class-note {{ font-size: 0.75rem; color: #6c757d; margin-top: 10px; line-height: 1.5; }}
        .chart-hover-target {{ cursor: pointer; }}
        .chart-tooltip {{
            position: fixed; pointer-events: none; background: #2c3e50; color: #fff;
            font-size: 0.78rem; padding: 6px 10px; border-radius: 6px; opacity: 0;
            transition: opacity 0.12s ease; z-index: 9999; white-space: nowrap;
        }}
        .region-map-card {{
            max-width: 460px; margin: 22px auto 0; text-align: center;
            border: 1px solid #e9ecef; border-radius: 8px; padding: 20px;
        }}
        .region-map-card h3 {{ font-size: 0.98rem; margin: 0 0 4px; }}
        .region-map-card svg {{ display: block; margin: 10px auto 0; max-width: 100%; height: auto; }}
        .region-map-note {{ font-size: 0.78rem; color: #6c757d; margin-top: 8px; }}
        .reveal-section {{
            opacity: 0; transform: translateY(16px);
            transition: opacity 0.5s ease, transform 0.5s ease;
        }}
        .reveal-section.is-visible {{ opacity: 1; transform: none; }}
        @media (prefers-reduced-motion: reduce) {{
            .reveal-section {{ transition: none; opacity: 1; transform: none; }}
        }}
    </style>
</head>
<body>

    <header class="header header--compact">
        <div class="header-background"></div>
        <nav class="topnav topnav--overlay">
            <div class="container topnav-inner">
                <a href="../../index.html" class="topnav-logo">DeepPVMapper</a>
                <div class="topnav-links">
                    <a href="../about.html" class="topnav-link">About</a>
                    <a href="../data.html" class="topnav-link is-active">Data</a>
                    <a href="../software.html" class="topnav-link">Software</a>
                    <a href="../contribute.html" class="topnav-link">Contribute</a>
                    <a href="../outlook.html" class="topnav-link">Use Cases</a>
                    <span class="topnav-break" aria-hidden="true"></span>
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
                </div>
            </div>
        </nav>
        <div class="container">
            <a href="../../index.html" class="back-home"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9.5 12 3l9 6.5"/><path d="M5 9.5V21h14V9.5"/></svg>Home</a>
            <h1 class="title">{nom}</h1>
            <p class="title-region">{n_depts} d&eacute;partements</p>
        </div>
    </header>

    <section>
        <div class="container">
            <div class="region-intro">
                {intro_html}
            </div>

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
                    <span class="stat-label">by installed capacity, out of {total} r&eacute;gions</span>
                </div>
            </div>

            <div class="reveal-section">
                <div class="dept-chart-grid">
                    {composition_card}
                    {stacked_card}
                </div>

                {map_section}
            </div>

            <div class="reveal-section" id="region-rankings">
                <div class="rankings-tabs">
                    <button type="button" class="rankings-tab is-active" data-panel="region-departements">D&eacute;partements</button>
                    <button type="button" class="rankings-tab" data-panel="region-cities">Cities</button>
                </div>
                <div class="ranking-panel is-active" id="ranking-region-departements">
                    {dept_ranking}
                </div>
                <div class="ranking-panel" id="ranking-region-cities">
                    {cities_ranking}
                </div>
            </div>

            {report_cta}

            <p style="text-align: center; margin-top: 40px;">
                <a href="../local-statistics.html" class="btn">&larr; Back to all r&eacute;gions</a>
            </p>
        </div>
    </section>

    <div class="chart-tooltip" id="chart-tooltip"></div>

    <footer class="footer">
        <div class="container">
            <p>Maintained by Gabriel Kasmi. This work is licensed under <a href="https://github.com/gabrielkasmi/deeppvmapper/blob/main/LICENSE" target="_blank">MIT</a>.</p>
        </div>
    </footer>

    <script src="../../static/js/script.js"></script>
    <script>
        (function () {{
            var tip = document.getElementById('chart-tooltip');
            if (!tip) return;
            document.querySelectorAll('.chart-hover-target').forEach(function (el) {{
                el.addEventListener('mousemove', function (e) {{
                    tip.textContent = el.getAttribute('data-tip') || '';
                    tip.style.left = (e.clientX + 14) + 'px';
                    tip.style.top = (e.clientY + 14) + 'px';
                    tip.style.opacity = '1';
                }});
                el.addEventListener('mouseleave', function () {{ tip.style.opacity = '0'; }});
            }});
        }})();

        document.querySelectorAll('#region-rankings .rankings-tab').forEach(function (btn) {{
            btn.addEventListener('click', function () {{
                document.querySelectorAll('#region-rankings .rankings-tab').forEach(function (b) {{ b.classList.remove('is-active'); }});
                document.querySelectorAll('#region-rankings .ranking-panel').forEach(function (p) {{ p.classList.remove('is-active'); }});
                btn.classList.add('is-active');
                document.getElementById('ranking-' + btn.dataset.panel).classList.add('is-active');
            }});
        }});

        (function () {{
            var sections = document.querySelectorAll('.reveal-section');
            if (!('IntersectionObserver' in window) || !sections.length) {{
                sections.forEach(function (s) {{ s.classList.add('is-visible'); }});
                return;
            }}
            var observer = new IntersectionObserver(function (entries) {{
                entries.forEach(function (entry) {{
                    if (entry.isIntersecting) {{
                        entry.target.classList.add('is-visible');
                        observer.unobserve(entry.target);
                    }}
                }});
            }}, {{ threshold: 0.15 }});
            sections.forEach(function (s) {{ observer.observe(s); }});
        }})();
    </script>
</body>
</html>
"""


def build_region_page(region, depts_geo=()):
    """region: accumulated dict — see the region_agg building block in
    main() for its exact shape. depts_geo: all départements' GeoJSON
    features (static/data/geo/departements.geojson) — filtered here to this
    région's own départements, for the choropleth map."""
    nom, code, slug = region["nom"], region["code"], region["slug"]
    n, kwp = region["n"], region["kwp"]

    combo_totals = {}
    for combos in region["combo"].values():
        for ck, cnt in combos.items():
            combo_totals[ck] = combo_totals.get(ck, 0) + cnt

    composition_card = (
        CHART_CARD.format(
            title="Source composition",
            svg=svg_donut_chart(combo_totals) + combo_legend_html(combo_totals),
        ) if combo_totals else ""
    )
    power_class_note = (
        '<p class="power-class-note">'
        + " &middot; ".join(f"{pc} {POWER_CLASS_LABELS[pc]}" for pc in POWER_CLASSES)
        + "</p>"
    )
    stacked_card = (
        CHART_CARD.format(
            title="Composition by power class",
            svg=svg_stacked_bar_by_class(region["combo"]) + power_class_note,
        ) if region["combo"] else ""
    )

    dept_by_code = {d["code"]: d for d in region["depts"]}
    dept_ranks = rank_by_value_desc({d["code"]: d["kwp"] for d in region["depts"]})
    dept_entries = []
    for feat in depts_geo:
        p = feat["properties"]
        d = dept_by_code.get(p["code"])
        if not d:
            continue
        dept_entries.append({
            "feature": feat, "kwp": d["kwp"], "nom": d["nom"],
            "rank": dept_ranks[p["code"]],
        })
    map_section = (
        f'<div class="region-map-card"><h3>Installed capacity by d&eacute;partement</h3>'
        f'{svg_region_choropleth(dept_entries)}'
        f'<p class="region-map-note">Darker = larger share of {nom}&rsquo;s installed capacity.</p></div>'
    ) if dept_entries else ""

    # For keywords, "recognizable" (population) is the right sort — for the
    # ranking panel below, installed capacity is (build_city_ranking_list
    # does that sort itself, filtering to cities with a page).
    named_cities = sorted(region["cities"], key=lambda c: -c["population"])[:CITIES_PER_REGION]

    cities_ranking = build_city_ranking_list(
        region["cities"], "", sub_fn=lambda c: c["dept_nom"], limit=CITIES_PER_REGION,
    )
    dept_ranking = build_dept_ranking_list(region["depts"])

    return REGION_TEMPLATE.format(
        nom=nom, code=code, slug=slug,
        keywords=build_region_keywords(nom, [d["nom"] for d in region["depts"]], named_cities),
        n_fmt=f"{n:,}".replace(",", " "),
        mwp_fmt=f"{kwp / 1000:.1f}",
        rank=region["rank"], total=region["total_regions"],
        n_depts=len(region["depts"]),
        intro_html=build_region_intro(region),
        composition_card=composition_card, stacked_card=stacked_card,
        map_section=map_section,
        dept_ranking=dept_ranking, cities_ranking=cities_ranking,
        report_cta=build_report_cta(nom),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--local", metavar="PATH",
        help="aggregate capacity/yearly/source stats from a local copy of "
             "dpvm_enriched.geojson instead of the Supabase RPCs (same "
             "underlying data — sidesteps network + statement-timeout risk "
             "on the heavier source-breakdown query). The 'largest cities' "
             "section still needs geo.api.gouv.fr regardless.",
    )
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(ROOT, "static", "data", "geo", "departements.geojson"), encoding="utf-8") as f:
        depts_geo = json.load(f)["features"]
    with open(os.path.join(ROOT, "static", "data", "geo", "regions.geojson"), encoding="utf-8") as f:
        regions = {f["properties"]["code"]: f["properties"]["nom"] for f in json.load(f)["features"]}

    if args.local:
        print(f"Aggregating locally from {args.local}…")
        (capacity_rows, yearly_rows, source_rows,
         commune_capacity_rows, commune_yearly_rows, commune_source_rows,
         combo_by_dept, hex_by_dept, hex_grids, multi_source_by_dept,
         combo_by_commune, power_class_by_commune, azimuth_by_commune) = local_aggregate(args.local, depts_geo)
    else:
        print("Fetching aggregate stats from Supabase…")
        capacity_rows = rpc("dept_capacity_stats")
        yearly_rows = rpc("dept_yearly_stats")
        source_rows = rpc("dept_source_stats")
        # No commune-level RPC yet — per-city pages are only built in --local
        # mode for now. The département/région pages work the same either way.
        commune_capacity_rows, commune_yearly_rows, commune_source_rows = [], [], []
        # Ditto for the source-combination × power-class breakdown, the
        # hexbin temporal map, the multi-source figure, and the commune-level
        # donut/histogram/azimuth-rose inputs — all --local only, for now (no
        # per-point/per-array Supabase RPC yet).
        combo_by_dept, hex_by_dept, hex_grids, multi_source_by_dept = {}, {}, {}, {}
        combo_by_commune, power_class_by_commune, azimuth_by_commune = {}, {}, {}
        print("  (Supabase mode: skipping per-city pages — commune-level "
              "breakdown needs --local for now.)")

    capacity_by_dept = {r["dpt"]: r for r in capacity_rows}
    total = len(capacity_by_dept)
    all_kwp = [r["total_kwp"] for r in capacity_rows]

    yearly_by_dept = {}
    for r in yearly_rows:
        yearly_by_dept.setdefault(r["dpt"], {})[int(r["year"])] = r["n_systems"]

    source_by_dept = {}
    for r in source_rows:
        source_by_dept.setdefault(r["dpt"], {})[int(r["source_id"])] = r["n_systems"]

    capacity_by_commune = {r["insee"]: r for r in commune_capacity_rows}

    yearly_by_commune = {}
    for r in commune_yearly_rows:
        yearly_by_commune.setdefault(r["insee"], {})[int(r["year"])] = r["n_systems"]

    source_by_commune = {}
    for r in commune_source_rows:
        source_by_commune.setdefault(r["insee"], {})[int(r["source_id"])] = r["n_systems"]

    # All communes' kwp within each département — the comparison chart's
    # range on each city page ("vs. other communes in {département}").
    commune_kwp_by_dept = {}
    for r in commune_capacity_rows:
        commune_kwp_by_dept.setdefault(r["dpt"], []).append(r["total_kwp"])

    os.makedirs(REGION_OUT_DIR, exist_ok=True)
    if commune_capacity_rows:
        os.makedirs(CITY_OUT_DIR, exist_ok=True)

    dept_index = []
    city_index = []
    region_agg = {}   # region_code -> accumulated stats, see below
    n_depts = len(depts_geo)
    n_city_candidates = n_city_pages = 0

    for i, feature in enumerate(depts_geo, 1):
        props = feature["properties"]
        code, nom, region_code = props["code"], props["nom"], props["region"]
        stats = capacity_by_dept.get(code)
        if not stats:
            continue  # no detections for this département — skip page

        n = stats["n_systems"]
        kwp = stats["total_kwp"]
        rank = stats["rank_by_capacity"]
        region_nom = regions.get(region_code, "France")
        region_slug = slugify(region_nom)

        # Source composition, as a donut (matching the région page) instead
        # of the old "detections by source" bar — same combo_by_dept data,
        # just flattened across power classes and without the P1-P5 split
        # (not worth it at département scale, per the build notes).
        dept_combo = combo_by_dept.get(code, {})
        combo_totals_dept = {}
        power_class_counts = {}
        for pc, combos in dept_combo.items():
            power_class_counts[pc] = sum(combos.values())
            for ck, cnt in combos.items():
                combo_totals_dept[ck] = combo_totals_dept.get(ck, 0) + cnt
        composition_card = (
            CHART_CARD.format(
                title="Source composition",
                svg=svg_donut_chart(combo_totals_dept) + combo_legend_html(combo_totals_dept),
            ) if combo_totals_dept else ""
        )
        histogram_card = (
            CHART_CARD.format(
                title="Installed capacity distribution",
                svg=svg_capacity_histogram(power_class_counts),
            ) if power_class_counts else ""
        )

        print(f"  [{i}/{n_depts}] {nom} ({code}): fetching largest cities…")
        cities, dept_population = fetch_top_cities(code)
        time.sleep(0.1)  # be polite to the free API

        # Hexbin temporal map — only if this département got a usable hex
        # grid (build_dept_hex_grid) and has at least one dated, gridded
        # installation. Cities are drawn on top of the hex layer as small
        # labeled landmarks so the map reads as a real place.
        hexbin_section = ""
        grid = hex_grids.get(code)
        dept_hex_years = sorted({y for hv in hex_by_dept.get(code, {}).values() for y in hv})
        if grid and dept_hex_years:
            hex_svg, years_json = svg_dept_hexbin(grid, hex_by_dept.get(code, {}), dept_hex_years, cities)
            if hex_svg:
                last_idx = len(dept_hex_years) - 1
                hexbin_section = (
                    '<div class="region-map-card">'
                    '<h3>Installed capacity share over time</h3>'
                    '<div class="hexbin-slider-row">'
                    f'<input type="range" id="hexbin-year-slider" min="0" max="{last_idx}" value="{last_idx}" step="1">'
                    f'<span class="hexbin-year-label" id="hexbin-year-label">{dept_hex_years[-1]}</span>'
                    '</div>'
                    f'{hex_svg}'
                    f'<script type="application/json" id="hexbin-years-data">{years_json}</script>'
                    f'<p class="region-map-note">Cumulative share of installed capacity by grid cell, '
                    f'{dept_hex_years[0]}&ndash;{dept_hex_years[-1]}. Drag the slider to see it build up over time.</p>'
                    '</div>'
                )

        pct_multi_source = None
        ms = multi_source_by_dept.get(code)
        if ms and ms[0]:
            pct_multi_source = ms[1] / ms[0] * 100

        # Decide which of this département's largest-by-population cities
        # get their own page: only those with enough real detection content
        # (CITY_MIN_SYSTEMS) get ranked at all — there's no reliable capacity
        # number for the rest, so they're simply left out of the ranking
        # (build_city_ranking_list) rather than shown unranked.
        for c in cities:
            c["dept_code"] = code
            c["dept_nom"] = nom
            cstats = capacity_by_commune.get(c["code"])
            has_page = bool(cstats and cstats["n_systems"] >= CITY_MIN_SYSTEMS)
            c["has_page"] = has_page
            if has_page:
                c["kwp"] = cstats["total_kwp"]
                c["n_systems"] = cstats["n_systems"]
                city_html = build_city_page(
                    c, cstats, nom, code, region_nom, region_slug,
                    yearly_by_commune.get(c["code"], {}),
                    combo_by_commune.get(c["code"], {}),
                    power_class_by_commune.get(c["code"], {}),
                    azimuth_by_commune.get(c["code"], {}),
                )
                with open(os.path.join(CITY_OUT_DIR, f"{c['code']}.html"), "w", encoding="utf-8") as f:
                    f.write(city_html)
                city_index.append({
                    "insee": c["code"], "nom": c["nom"], "dept_code": code,
                    "dept_nom": nom, "region_nom": region_nom,
                    "n_systems": cstats["n_systems"], "kwp": cstats["total_kwp"],
                })

        n_city_candidates += len(cities)
        n_city_pages += sum(1 for c in cities if c["has_page"])

        cities_section = build_city_ranking_list(
            cities, f"Cities in {nom} ranked by installed capacity",
            sub_fn=lambda c: f"{c['n_systems']:,}".replace(",", " ") + " systems",
        )
        keywords = build_keywords(nom, code, region_nom, cities)

        intro_html = build_dept_intro({
            "code": code, "nom": nom, "region_nom": region_nom, "region_code": region_code,
            "n": n, "kwp": kwp, "rank": rank, "total": total,
            "population": dept_population, "years": sorted(yearly_by_dept.get(code, {}).keys()),
            "pct_multi_source": pct_multi_source,
        })

        html = TEMPLATE.format(
            code=code,
            nom=nom,
            region_nom=region_nom,
            region_slug=region_slug,
            keywords=keywords,
            n_fmt=f"{n:,}".replace(",", " "),
            mwp_fmt=f"{kwp / 1000:.1f}",
            rank=rank,
            total=total,
            intro_html=intro_html,
            composition_card=composition_card,
            histogram_card=histogram_card,
            hexbin_section=hexbin_section,
            cities_section=cities_section,
            city_request_note=build_city_request_note(nom, code),
            report_cta=build_report_cta(f"{nom} ({code})"),
        )
        dept_path = os.path.join(OUT_DIR, f"{code}.html")
        with open(dept_path, "w", encoding="utf-8") as f:
            f.write(html)

        dept_index.append({
            "code": code, "nom": nom, "region": region_nom,
            "region_code": region_code, "region_slug": region_slug,
            "n": n, "kwp": kwp, "mwp": round(kwp / 1000, 1), "rank": rank,
            "path": dept_path,
        })

        ra = region_agg.setdefault(region_code, {
            "nom": region_nom, "code": region_code, "slug": region_slug,
            "n": 0, "kwp": 0.0, "yearly": {}, "source": {}, "combo": {}, "depts": [], "cities": [],
        })
        ra["n"] += n
        ra["kwp"] += kwp
        ra["depts"].append({"code": code, "nom": nom, "n": n, "kwp": kwp, "mwp": round(kwp / 1000, 1)})
        ra["cities"].extend(cities)
        for year, cnt in yearly_by_dept.get(code, {}).items():
            ra["yearly"][year] = ra["yearly"].get(year, 0) + cnt
        for sid, cnt in source_by_dept.get(code, {}).items():
            ra["source"][sid] = ra["source"].get(sid, 0) + cnt
        for pc, combos in combo_by_dept.get(code, {}).items():
            rpc_pc = ra["combo"].setdefault(pc, {})
            for ck, cnt in combos.items():
                rpc_pc[ck] = rpc_pc.get(ck, 0) + cnt

    dept_index.sort(key=lambda d: (d["region"], d["nom"]))

    print(f"Cities: {n_city_pages}/{n_city_candidates} candidates cleared "
          f"the >= {CITY_MIN_SYSTEMS}-detection bar and got their own page.")

    # ─── Région pages — one aggregate rollup per région, from the same
    # per-département numbers just computed above. ─────────────────────────
    region_ranks = rank_by_value_desc({code: r["kwp"] for code, r in region_agg.items()})
    all_region_kwp = [r["kwp"] for r in region_agg.values()]
    total_regions = len(region_agg)
    region_index = []

    for region_code, ra in region_agg.items():
        ra["rank"] = region_ranks[region_code]
        ra["total_regions"] = total_regions
        ra["all_region_kwp"] = all_region_kwp
        region_depts_geo = [f for f in depts_geo if f["properties"]["region"] == region_code]
        html = build_region_page(ra, region_depts_geo)
        with open(os.path.join(REGION_OUT_DIR, f"{ra['slug']}.html"), "w", encoding="utf-8") as f:
            f.write(html)
        region_index.append({
            "slug": ra["slug"], "nom": ra["nom"],
            "n": ra["n"], "kwp": ra["kwp"], "rank": ra["rank"],
        })

    # Now that every région is fully aggregated, patch each département
    # page's régional-rank placeholder with the real ordinal — a rank
    # within {n_depts_in_region} only makes sense once every sibling
    # département in that région has been processed, which isn't true yet
    # partway through the loop above (see build_dept_intro's docstring).
    for region_code, ra in region_agg.items():
        dept_ranks_in_region = rank_by_value_desc({d["code"]: d["kwp"] for d in ra["depts"]})
        n_in_region = len(ra["depts"])
        for d in dept_index:
            if d["region_code"] != region_code:
                continue
            ordinal = _ordinal(dept_ranks_in_region[d["code"]])
            rank_text = f"{ordinal} out of {n_in_region}" if n_in_region > 1 else ordinal
            with open(d["path"], encoding="utf-8") as f:
                content = f.read()
            content = content.replace(REGION_RANK_PLACEHOLDER, rank_text)
            with open(d["path"], "w", encoding="utf-8") as f:
                f.write(content)

    inject_rankings(dept_index, region_index, city_index)
    build_sitemap(dept_index, region_index, city_index)
    ensure_robots_txt()

    print(f"Wrote {len(dept_index)} département pages to {OUT_DIR}")
    print(f"Wrote {len(region_index)} région pages to {REGION_OUT_DIR}")
    print(f"Wrote {n_city_pages} city pages to {CITY_OUT_DIR}")
    print("Regenerated sitemap.xml and checked robots.txt")


RANKING_ITEM = """                <a class="ranking-item{rank_class}" href="{href}">
                    <span class="ranking-rank">{rank}</span>
                    <span class="ranking-main">
                        <span class="ranking-name">{nom}</span>
                        <span class="ranking-sub">{sub}</span>
                    </span>
                    <span class="ranking-value">{value}</span>
                </a>"""

TOP_CITIES_FOR_RANKING = 50


def _ranking_rows(items):
    """items already sorted best-first. Returns the joined HTML, numbering
    1..N and flagging the top 3 for the gold/silver/bronze badge style."""
    rows = []
    for i, it in enumerate(items, 1):
        rows.append(RANKING_ITEM.format(
            rank_class=f" rank-{i}" if i <= 3 else "",
            href=it["href"], rank=i, nom=it["nom"], sub=it["sub"], value=it["value"],
        ))
    return "\n".join(rows)


def inject_rankings(dept_index, region_index, city_index):
    """Splice three ranked-by-capacity lists (régions, départements, cities)
    into content/local-statistics.html's Rankings section — the static tab/
    panel markup and CSS/JS already live there permanently; this just
    refills what's between each pair of RANKING_*_START/END markers, same
    splice pattern as inject_browse_grid."""
    regions_sorted = sorted(region_index, key=lambda r: -r["kwp"])
    region_rows = _ranking_rows([
        {"href": f"regions/{r['slug']}.html", "nom": r["nom"],
         "sub": f"{len([d for d in dept_index if d['region_slug'] == r['slug']])} départements",
         "value": f"{r['kwp'] / 1000:,.1f} MWp".replace(",", " ")}
        for r in regions_sorted
    ])

    depts_sorted = sorted(dept_index, key=lambda d: -d["kwp"])
    dept_rows = _ranking_rows([
        {"href": f"data/{d['code']}.html", "nom": d["nom"], "sub": d["region"],
         "value": f"{d['mwp']:,.1f} MWp".replace(",", " ")}
        for d in depts_sorted
    ])

    cities_sorted = sorted(city_index, key=lambda c: -c["kwp"])[:TOP_CITIES_FOR_RANKING]
    city_rows = []
    for c in cities_sorted:
        cap_fmt, cap_unit = fmt_capacity(c["kwp"])
        city_rows.append({
            "href": f"cities/{c['insee']}.html", "nom": c["nom"],
            "sub": f"{c['dept_nom']} · {c['region_nom']}",
            "value": f"{cap_fmt} {cap_unit}",
        })
    city_rows_html = _ranking_rows(city_rows)

    target_path = os.path.join(ROOT, "content", "local-statistics.html")
    with open(target_path, encoding="utf-8") as f:
        page = f.read()

    for label, rows_html in (
        ("RANKING_REGIONS", region_rows),
        ("RANKING_DEPARTEMENTS", dept_rows),
        ("RANKING_CITIES", city_rows_html),
    ):
        start_marker = f"<!-- {label}_START -->"
        end_marker = f"<!-- {label}_END -->"
        start = page.index(start_marker) + len(start_marker)
        end = page.index(end_marker)
        page = page[:start] + "\n" + rows_html + "\n                " + page[end:]

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(page)


# ─── Sitemap / robots.txt (SEO) ────────────────────────────────────────────

def build_sitemap(dept_index, region_index=(), city_index=()):
    """Regenerate sitemap.xml from the actual content/ tree — every top-level
    content/*.html page (excluding the data/regions/cities subfolders
    themselves) plus every content/data/{code}.html, content/regions/{slug}
    .html and content/cities/{insee}.html page just written. Walking the
    real filesystem (rather than a hardcoded list) means pages added/removed
    later stay in sync automatically next time this script runs."""
    today = datetime.date.today().isoformat()
    urls = []

    if os.path.exists(os.path.join(ROOT, "index.html")):
        urls.append((f"{SITE_BASE}/index.html", "weekly", "1.0"))

    content_dir = os.path.join(ROOT, "content")
    for fn in sorted(os.listdir(content_dir)):
        path = os.path.join(content_dir, fn)
        if not (os.path.isfile(path) and fn.endswith(".html")):
            continue
        if fn in ("data.html", "local-statistics.html"):
            urls.append((f"{SITE_BASE}/content/{fn}", "weekly", "0.9"))
        else:
            urls.append((f"{SITE_BASE}/content/{fn}", "monthly", "0.7"))

    for d in dept_index:
        urls.append((f"{SITE_BASE}/content/data/{d['code']}.html", "monthly", "0.6"))

    for r in region_index:
        urls.append((f"{SITE_BASE}/content/regions/{r['slug']}.html", "monthly", "0.7"))

    for c in city_index:
        urls.append((f"{SITE_BASE}/content/cities/{c['insee']}.html", "monthly", "0.5"))

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for loc, freq, prio in urls:
        lines += [
            "  <url>",
            f"    <loc>{loc}</loc>",
            f"    <lastmod>{today}</lastmod>",
            f"    <changefreq>{freq}</changefreq>",
            f"    <priority>{prio}</priority>",
            "  </url>",
        ]
    lines.append("</urlset>")

    with open(os.path.join(ROOT, "sitemap.xml"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def ensure_robots_txt():
    """Create robots.txt if missing, or just append a Sitemap: line to an
    existing one that doesn't have one yet — never otherwise touch a
    hand-maintained robots.txt."""
    path = os.path.join(ROOT, "robots.txt")
    sitemap_line = f"Sitemap: {SITE_BASE}/sitemap.xml"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            content = f.read()
        if "Sitemap:" not in content:
            with open(path, "a", encoding="utf-8") as f:
                f.write(("\n" if not content.endswith("\n") else "") + f"\n{sitemap_line}\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"User-agent: *\nAllow: /\n\n{sitemap_line}\n")


if __name__ == "__main__":
    main()
