#!/usr/bin/env python3
"""
build_agg.py - Data for the aggregated viewer (static/js/aggmap, _includes/aggmap.html).

Reads the OpenPVMapper release (parquet) and, optionally, the INSEE Filosofi 2021 1 km grid,
and writes a handful of small static files:

    static/data/agg/meta.json       grid geometry, table layout, totals, release version
    static/data/agg/outline.json    simplified outline of metropolitan France (lon/lat rings)
    static/data/agg/pv.bin.gz       installations and capacity per cell        (4 columns)
    static/data/agg/pvcls.bin.gz    same, split by unit power class             (20 columns)
    static/data/agg/insee.bin.gz    a few INSEE variables per cell              (6 columns)

Everything is aggregated on the INSPIRE 1 km grid (EPSG:3035, ids CRS3035RES1000mN{N}E{E}) and then
summed into BLOCK x BLOCK km blocks (default 2): the finest cell the viewer ever shows. All stored
columns are sums, so the viewer can add them up to any coarser resolution; ratios are always computed
in the browser as sum(numerator) / sum(denominator).

Tables are sparse (one row per non-empty cell). Each table is stored as: cell index (delta coded),
then one column after another, bytes shuffled by significance, floats truncated to a few mantissa
bits, all gzip-compressed. The page decodes them with DecompressionStream('gzip').

Usage (from the repo root):
    python3 scripts/build_agg.py --release PATH/full.parquet [--filosofi PATH/carreaux_1km_met.csv|.zip]

Needs: numpy, pandas, pyarrow, pyproj, shapely.
INSEE data: Filosofi 2021, carreaux de 1 km, Licence Ouverte (attribution: INSEE).
"""
import argparse, gzip, io, json, os, re, sys, time, zipfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--release", required=True, help="full.parquet of the OpenPVMapper release")
ap.add_argument("--filosofi", help="carreaux_1km_met.csv (or the zip that contains it); omit to skip INSEE")
ap.add_argument("--departements", default=os.path.join(ROOT, "static/data/geo/departements.geojson"),
                help="department polygons, used for the outline of the country")
ap.add_argument("--out", default=os.path.join(ROOT, "static/data/agg"))
ap.add_argument("--block", type=int, default=2, help="side (km) of the finest cell")
a = ap.parse_args()

EDGES = [3, 9, 36, 250]                # unit power classes (kWp): [0,3[ [3,9[ [9,36[ [36,250[ [250,+inf[
INSEE_VARS = ["ind", "men", "men_mais", "ind_snv", "men_pauv", "men_prop"]
KEEP_BITS = {"ind": 15, "men": 15, "men_mais": 15, "ind_snv": 11,
             **{f"k{c}_{q}": 11 for c in range(5) for q in ("all", "hc")},
             **{k: 11 for k in ("kwp_all", "kwp_hc")}}
DEFAULT_BITS = 9                       # relative error < 0.2 %: plenty for shares and means
os.makedirs(a.out, exist_ok=True)
t0 = time.time()


def log(*x): print(f"[{time.time()-t0:5.1f}s]", *x, flush=True)


# --------------------------------------------------------------------------- installations
import pyarrow.parquet as pq
from pyproj import Transformer

df = pq.read_table(a.release, columns=["centroid_lon", "centroid_lat", "kWp", "n_sources", "dpt"]).to_pandas()
df = df[df["centroid_lon"].notna() & df["centroid_lat"].notna() & ~df["dpt"].astype(str).str.startswith("97")]
log(f"{len(df):,} installations (metropolitan France)")
E, N = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform(
    df["centroid_lon"].to_numpy(), df["centroid_lat"].to_numpy())
col, row = np.floor(E / 1000).astype(np.int64), np.floor(N / 1000).astype(np.int64)
kwp = df["kWp"].fillna(0).to_numpy(np.float64)
hc = df["n_sources"].to_numpy() >= 2
cls = np.digitize(kwp, EDGES)
log(f"installations without a power estimate (counted in the smallest class): {int(df['kWp'].isna().sum()):,}")

key = row * 100000 + col
uk, inv = np.unique(key, return_inverse=True)
pv = {"rows": uk // 100000, "cols": uk % 100000}
pv["n_all"] = np.bincount(inv, minlength=len(uk))
pv["kwp_all"] = np.bincount(inv, weights=kwp, minlength=len(uk))
pv["n_hc"] = np.bincount(inv, weights=hc, minlength=len(uk))
pv["kwp_hc"] = np.bincount(inv, weights=kwp * hc, minlength=len(uk))
for c in range(5):
    for q, w in (("all", np.ones(len(df), bool)), ("hc", hc)):
        s = (cls == c) & w
        pv[f"n{c}_{q}"] = np.bincount(inv[s], minlength=len(uk))
        pv[f"k{c}_{q}"] = np.bincount(inv[s], weights=kwp[s], minlength=len(uk))
c0, c1, r0, r1 = int(pv["cols"].min()), int(pv["cols"].max()), int(pv["rows"].min()), int(pv["rows"].max())
W, H = c1 - c0 + 1, r1 - r0 + 1
log(f"{len(uk):,} non-empty 1 km cells; grid {W} x {H} km, origin ({c0}, {r0})")

# --------------------------------------------------------------------------- INSEE
insee = None
if a.filosofi:
    import pandas as pd
    fh = a.filosofi
    if fh.endswith(".zip"):
        z = zipfile.ZipFile(fh)
        fh = io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith("_met.csv"))))
    f = pd.read_csv(fh, usecols=["idcar_1km"] + INSEE_VARS)
    ids = f["idcar_1km"].str.extract(r"CRS3035RES1000mN(\d+)E(\d+)").astype(float)
    ok = ids.notna().all(axis=1)
    f, ids = f[ok], ids[ok]
    insee = {"rows": (ids[0] // 1000).astype(np.int64).to_numpy(), "cols": (ids[1] // 1000).astype(np.int64).to_numpy()}
    for v in INSEE_VARS:
        insee[v] = f[v].to_numpy(np.float64)
    log(f"INSEE: {len(f):,} cells, population {insee['ind'].sum()/1e6:.2f} M")


# --------------------------------------------------------------------------- packing
def block(z, cols):
    """Sum 1 km cells into block x block km cells, stored on the south-west 1 km cell of the block."""
    B = a.block
    bc, br = (z["cols"] - c0) // B, (z["rows"] - r0) // B
    k = br * (W // B + 2) + bc
    u, inv = np.unique(k, return_inverse=True)
    out = {"rows": r0 + B * (u // (W // B + 2)), "cols": c0 + B * (u % (W // B + 2))}
    for c in cols:
        out[c] = np.bincount(inv, weights=z[c].astype(np.float64), minlength=len(u))
    return out


def truncate(v, bits):
    u = v.astype("<f4").view("<u4").copy()
    u &= np.uint32((0xFFFFFFFF << (23 - bits)) & 0xFFFFFFFF)
    return u.view("<f4")


def shuffle(arr):
    b = np.ascontiguousarray(arr).view(np.uint8).reshape(-1, arr.itemsize)
    return b.T.copy().ravel().tobytes()


def write_table(name, z, cols, counts):
    z = block(z, cols)
    m = (z["cols"] >= c0) & (z["cols"] <= c1) & (z["rows"] >= r0) & (z["rows"] <= r1)
    idx = (z["rows"][m] - r0) * W + (z["cols"][m] - c0)
    order = np.argsort(idx)
    blob, entry = bytearray(), {"file": f"{name}.bin.gz", "n": int(m.sum()), "cols": []}

    def push(raw):
        off = len(blob)
        blob.extend(raw + b"\0" * ((-len(raw)) % 4))
        return off
    entry["idx"] = push(shuffle(np.diff(idx[order], prepend=0).astype("<u4")))
    for c in cols:
        v = z[c][m][order]
        if c in counts:
            arr, t = np.clip(np.rint(v), 0, 65535).astype("<u2"), "u16"
        else:
            arr, t = truncate(v, KEEP_BITS.get(c, DEFAULT_BITS)), "f32"
        entry["cols"].append({"id": c, "t": t, "off": push(shuffle(arr))})
    gz = gzip.compress(bytes(blob), 9)
    with open(os.path.join(a.out, entry["file"]), "wb") as fh:
        fh.write(gz)
    log(f"{entry['file']}: {entry['n']:,} cells, {len(cols)} columns, {len(gz)/1e6:.2f} MB")
    return entry


PV_TOT = ["n_all", "kwp_all", "n_hc", "kwp_hc"]
PV_CLS = [f"{p}{c}_{q}" for c in range(5) for q in ("all", "hc") for p in ("n", "k")]
is_count = lambda c: c[0] == "n" and not c.startswith("ncar")
tables = {"pv": write_table("pv", pv, PV_TOT, {c for c in PV_TOT if is_count(c)}),
          "pvcls": write_table("pvcls", pv, PV_CLS, {c for c in PV_CLS if is_count(c)})}
if insee:
    tables["insee"] = write_table("insee", insee, INSEE_VARS, set())

# --------------------------------------------------------------------------- outline
from shapely.geometry import shape
from shapely.ops import transform, unary_union
fwd = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform
back = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True).transform
polys = []
for ft in json.load(open(a.departements))["features"]:
    if str(ft["properties"].get("code", "")).startswith("97"):
        continue
    polys.append(transform(fwd, shape(ft["geometry"])).buffer(0))
land = unary_union(polys).simplify(350, preserve_topology=True)
rings = []
for p in (land.geoms if land.geom_type == "MultiPolygon" else [land]):
    if p.area >= 4e6:                      # ignore islets below 4 km2
        rings.append([[round(lo, 4), round(la, 4)] for lo, la in (back(x, y) for x, y in p.exterior.coords)])
json.dump({"land": rings}, open(os.path.join(a.out, "outline.json"), "w"), separators=(",", ":"))
log(f"outline.json: {len(rings)} rings, {os.path.getsize(os.path.join(a.out, 'outline.json'))/1024:.0f} KB")

# --------------------------------------------------------------------------- meta
try:
    ver = json.load(open(os.path.join(ROOT, "static/data/version.json")))
    ver = (ver[0] if isinstance(ver, list) else ver)
except Exception:
    ver = {}
meta = {"W": W, "H": H, "x0": c0, "y0": r0, "block": a.block, "version": ver.get("version"), "date": ver.get("date"),
        "tables": tables,
        "totals": {"n_all": int(pv["n_all"].sum()), "n_hc": int(pv["n_hc"].sum()),
                   "gwp_all": round(float(pv["kwp_all"].sum()) / 1e6, 3), "gwp_hc": round(float(pv["kwp_hc"].sum()) / 1e6, 3)}}
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), separators=(",", ":"))
log(f"meta.json written; {meta['totals']['n_all']:,} installations, {meta['totals']['gwp_all']} GWp")
assert sum(int(pv[f"n{c}_all"].sum()) for c in range(5)) == meta["totals"]["n_all"]
