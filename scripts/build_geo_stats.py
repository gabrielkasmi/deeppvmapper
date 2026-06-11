#!/usr/bin/env python3
"""
build_geo_stats.py — Aggregate PV detection statistics at commune / departement /
region level for the interactive map.

Source of truth: the IGN WFS layer (default), or a local GeoJSON dump of the
detections (--local) for faster re-runs when you already have the file.

Outputs (committed to the repo, all small):
    static/data/stats/stats_communes.json      {insee: [n, kwp, lat, lng]}
    static/data/stats/stats_departements.json  {code:  [n, kwp]}
    static/data/stats/stats_regions.json       {code:  [n, kwp]}
    static/data/stats/stats_meta.json          build date, totals, source

The commune [lat, lng] is the mean position of the detections inside the
commune — it doubles as the weighted point set for the national heatmap.

Usage:
    python3 scripts/build_geo_stats.py                  # page the WFS (slow, ~100 requests)
    python3 scripts/build_geo_stats.py --local path.geojson
"""

import argparse
import glob
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date

import numpy as np
from pyproj import Transformer
from shapely import STRtree, points
from shapely.geometry import shape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEO_DIR = os.path.join(ROOT, 'static', 'data', 'geo')
OUT_DIR = os.path.join(ROOT, 'static', 'data', 'stats')

WFS_URL = 'https://data.geopf.fr/sandbox/wfs'
WFS_LAYER = ('SANDBOX.france_detections_wgs84_geojson_09_06_2026_wfs:'
             'france_detections_wgs84')
WFS_PAGE = 5000


# ─── Detection loading ────────────────────────────────────────────────────────

def ring_centroid(geometry):
    """Mean of the exterior ring — cheap and good enough for small rooftops."""
    if geometry['type'] == 'MultiPolygon':
        ring = geometry['coordinates'][0][0]
    else:
        ring = geometry['coordinates'][0]
    n = len(ring)
    return (sum(c[0] for c in ring) / n, sum(c[1] for c in ring) / n)


def prop(p, *keys, default=0):
    for k in keys:
        if k in p and p[k] is not None:
            return p[k]
    return default


def load_local(path):
    """Stream a (possibly huge) local GeoJSON dump. Returns (xy array, kwp array, crs)."""
    import ijson
    crs = 'EPSG:4326'
    xs, ys, kwps = [], [], []
    with open(path, 'rb') as fh:
        # CRS is declared near the top; sniff the first 2 kB.
        head = fh.read(2048).decode('utf-8', 'replace')
        if 'EPSG::3035' in head or 'EPSG:3035' in head:
            crs = 'EPSG:3035'
        fh.seek(0)
        for feat in ijson.items(fh, 'features.item', use_float=True):
            try:
                x, y = ring_centroid(feat['geometry'])
            except (KeyError, IndexError, TypeError):
                continue
            xs.append(x)
            ys.append(y)
            kwps.append(float(prop(feat['properties'], 'kwp_approx', 'kWp_approx')))
    return np.array(xs), np.array(ys), np.array(kwps), crs


def load_wfs():
    """Page the full WFS layer. Returns (xy array, kwp array, crs)."""
    xs, ys, kwps = [], [], []
    start = 0
    while True:
        params = urllib.parse.urlencode({
            'SERVICE': 'WFS', 'VERSION': '2.0.0', 'request': 'GetFeature',
            'TYPENAMES': WFS_LAYER, 'COUNT': WFS_PAGE, 'startIndex': start,
            'outputFormat': 'application/json',
        })
        with urllib.request.urlopen(f'{WFS_URL}?{params}', timeout=120) as r:
            data = json.load(r)
        feats = data.get('features', [])
        for feat in feats:
            try:
                x, y = ring_centroid(feat['geometry'])
            except (KeyError, IndexError, TypeError):
                continue
            xs.append(x)
            ys.append(y)
            kwps.append(float(prop(feat['properties'], 'kwp_approx', 'kWp_approx')))
        print(f'  WFS page startIndex={start}: {len(feats)} features '
              f'({len(xs)} total)', file=sys.stderr)
        if len(feats) < WFS_PAGE:
            break
        start += WFS_PAGE
        time.sleep(0.2)  # be polite
    return np.array(xs), np.array(ys), np.array(kwps), 'EPSG:4326'


# ─── Spatial join ─────────────────────────────────────────────────────────────

def load_communes():
    """All commune geometries + codes from the vendored per-dept files."""
    geoms, codes = [], []
    for path in sorted(glob.glob(os.path.join(GEO_DIR, 'communes', 'communes-*.geojson'))):
        with open(path) as fh:
            fc = json.load(fh)
        for f in fc['features']:
            if not f.get('geometry'):
                continue  # a handful of communes ship without geometry
            geoms.append(shape(f['geometry']))
            codes.append(f['properties']['code'])
    return geoms, codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--local', help='local GeoJSON dump of the detections '
                                    '(default: page the WFS)')
    args = ap.parse_args()

    print('Loading detections…', file=sys.stderr)
    if args.local:
        xs, ys, kwps, crs = load_local(args.local)
        source = os.path.basename(args.local)
    else:
        xs, ys, kwps, crs = load_wfs()
        source = WFS_URL
    print(f'  {len(xs)} detections (crs {crs})', file=sys.stderr)

    if crs != 'EPSG:4326':
        tr = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
        xs, ys = tr.transform(xs, ys)

    print('Loading commune geometries…', file=sys.stderr)
    geoms, codes = load_communes()
    tree = STRtree(geoms)

    print('Spatial join…', file=sys.stderr)
    pts = points(np.column_stack([xs, ys]))
    pt_idx, geom_idx = tree.query(pts, predicate='intersects')

    # A point on a shared (simplified) boundary can match twice — keep the first.
    seen = np.zeros(len(pts), dtype=bool)
    communes = defaultdict(lambda: [0, 0.0, 0.0, 0.0])  # n, kwp, sum_lat, sum_lng
    matched = 0
    for pi, gi in zip(pt_idx, geom_idx):
        if seen[pi]:
            continue
        seen[pi] = True
        matched += 1
        c = communes[codes[gi]]
        c[0] += 1
        c[1] += kwps[pi]
        c[2] += ys[pi]
        c[3] += xs[pi]

    unmatched = len(pts) - matched
    print(f'  matched {matched}, unmatched {unmatched} '
          f'({100 * unmatched / max(len(pts), 1):.2f}%)', file=sys.stderr)

    # ─── Aggregate upwards ────────────────────────────────────────────────────
    with open(os.path.join(GEO_DIR, 'region_depts.json')) as fh:
        region_depts = json.load(fh)
    dept_region = {d: r for r, ds in region_depts.items() for d in ds}

    stats_c, stats_d, stats_r = {}, defaultdict(lambda: [0, 0.0]), defaultdict(lambda: [0, 0.0])
    for code, (n, kwp, slat, slng) in sorted(communes.items()):
        stats_c[code] = [n, round(kwp, 1), round(slat / n, 5), round(slng / n, 5)]
        dept = code[:2]
        stats_d[dept][0] += n
        stats_d[dept][1] += kwp
        reg = dept_region.get(dept)
        if reg:
            stats_r[reg][0] += n
            stats_r[reg][1] += kwp

    os.makedirs(OUT_DIR, exist_ok=True)
    dump = lambda o, name: json.dump(
        o, open(os.path.join(OUT_DIR, name), 'w'), separators=(',', ':'))
    dump(stats_c, 'stats_communes.json')
    dump({k: [v[0], round(v[1], 1)] for k, v in sorted(stats_d.items())},
         'stats_departements.json')
    dump({k: [v[0], round(v[1], 1)] for k, v in sorted(stats_r.items())},
         'stats_regions.json')
    dump({
        'built': date.today().isoformat(),
        'source': source,
        'detections': int(len(pts)),
        'matched': int(matched),
        'total_kwp': round(float(kwps.sum()), 1),
        'communes_with_detections': len(stats_c),
    }, 'stats_meta.json')

    print(f'Done. {len(stats_c)} communes, {len(stats_d)} depts, '
          f'{len(stats_r)} regions → {OUT_DIR}', file=sys.stderr)


if __name__ == '__main__':
    main()
