# Data Access

Read-only, public API access to the DeepPVMapper detections registry — the same backend
that powers the [interactive map](https://deeppvmapper.fr/content/map.html). This page
documents the one table that is open for public read: **`detections`**. Nothing else in
the project (annotations, community verifications, usage events, moderation views) is
publicly readable — those stay insert-only or fully private, protected by
[Row Level Security](https://supabase.com/docs/guides/database/postgres/row-level-security).

If you just want the full static dataset (no API, no rate limits, no key needed), get it
from [Zenodo](https://zenodo.org/records/19188878) or
[Hugging Face](https://huggingface.co/datasets/gabrielkasmi/openpvmapper) instead — see
[Static bulk downloads](#static-bulk-downloads). Use this API when you need to query a
live, always-current subset (a bounding box, a commune, an aggregate) without downloading
the whole registry.

## Base URL & authentication

The API is a standard [Supabase](https://supabase.com)/PostgREST endpoint. Every request
needs the public **anon (publishable) key** below, sent as both the `apikey` header and a
bearer token — this is not a secret, it's the standard way PostgREST identifies the
anonymous role; the actual protection is the Row Level Security policy on the table
(public, read-only — see [Schema & access scope](#schema--access-scope)).

```
Project URL:      https://zelhliylrlktnasircwp.supabase.co
Publishable key:  sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi
```

```bash
curl "https://zelhliylrlktnasircwp.supabase.co/rest/v1/detections?select=id,surface,kwp,dpt&limit=5" \
  -H "apikey: sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi" \
  -H "Authorization: Bearer sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi"
```

No account, no sign-up, no per-user rate limiting beyond Supabase's shared project quota.
This is a small, self-hosted project (not a funded public service) — be a good citizen:
cache results client-side where you can, and prefer the [bounding-box](#get-detections_bbox-viewport-query)
or [zone](#get-detections_in_zone-exact-zone-query) RPCs over unbounded `select *` queries
on the raw table for anything wider than a handful of communes.

## Endpoints

### `GET /rest/v1/detections` — raw table access

Standard [PostgREST](https://postgrest.org/en/stable/references/api/tables_views.html)
table endpoint. Supports column selection, filtering, ordering, and pagination via
PostgREST's query syntax.

```bash
# All installations in département 33 (Gironde), capacity above 9 kWp
curl "https://zelhliylrlktnasircwp.supabase.co/rest/v1/detections?dpt=eq.33&kwp=gt.9&select=id,geom,surface,kwp,first_seen" \
  -H "apikey: sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi" \
  -H "Authorization: Bearer sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi"
```

`geom` comes back as GeoJSON-in-PostGIS's default text encoding; for a clean GeoJSON
`Feature` shape, prefer the `get_detections_*` RPCs below, which do that conversion
server-side (`ST_AsGeoJSON`). There is no row cap enforced by RLS on this endpoint — use
`limit`/`offset` (or the `Range` header) yourself for anything large, and see the note
above about preferring the RPCs for bulk/viewport-style queries.

### `POST /rest/v1/rpc/get_detections_bbox` — viewport query

Mirrors what the map itself calls when panning/zooming. Returns a GeoJSON `Feature`
array, capped at `max_count` (default 2000) — built for map viewports, not exhaustive
export.

```bash
curl "https://zelhliylrlktnasircwp.supabase.co/rest/v1/rpc/get_detections_bbox" \
  -H "apikey: sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi" \
  -H "Authorization: Bearer sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi" \
  -H "Content-Type: application/json" \
  -d '{"min_lon": -0.6, "min_lat": 44.8, "max_lon": -0.5, "max_lat": 44.9, "max_count": 2000}'
```

Each feature's `properties` carries `surface`, `kwp`, and `year` (the record's
`first_seen`) — a deliberately small subset for map rendering. Query the table endpoint
directly (above) or one of the aggregate RPCs (below) for the full attribute set.

### `POST /rest/v1/rpc/get_detections_in_zone` — exact zone query

Exact polygon containment (not bbox-approximate), uncapped by default (`max_count`
300000) — what the map's CSV/GeoJSON export button calls. Pass any GeoJSON geometry as
`zone_geometry`.

```bash
curl "https://zelhliylrlktnasircwp.supabase.co/rest/v1/rpc/get_detections_in_zone" \
  -H "apikey: sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi" \
  -H "Authorization: Bearer sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi" \
  -H "Content-Type: application/json" \
  -d '{"zone_geometry": {"type": "Polygon", "coordinates": [[[-0.6,44.8],[-0.5,44.8],[-0.5,44.9],[-0.6,44.9],[-0.6,44.8]]]}}'
```

### Aggregate RPCs — département-level stats

Pre-aggregated, so a client never has to pull individual rows just to sum/count them.
Each takes no arguments and returns one row per département (~94–96 rows).

| RPC | Returns |
|---|---|
| `get_detections_bbox` / `get_detections_in_zone` | see above |
| `dept_capacity_stats()` | `dpt`, `n_systems`, `total_kwp`, `rank_by_capacity` |
| `dept_yearly_stats()` | `dpt`, `year`, `n_systems`, `total_kwp` — evolution by `first_seen` |
| `dept_source_stats()` | `dpt`, `source_id`, `n_systems` — see [Source encoding](#source-encoding) |

```bash
curl "https://zelhliylrlktnasircwp.supabase.co/rest/v1/rpc/dept_capacity_stats" \
  -X POST -H "apikey: sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi" \
  -H "Authorization: Bearer sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi"
```

## Schema & access scope

Only `detections` is public. Everything else in the project — `annotations`,
`campaign_pool`/`verifications` (the community verification game), `events`, and their
moderation views (`annotations_pending`, `issue_reports_pending`, `verifications_summary`)
— is insert-only or fully locked behind Row Level Security; the anon key cannot read them.
If your use case needs something from one of those (e.g. aggregated, anonymized
verification stats), open an issue or get in touch — see [Contact](#contact) — rather than
assuming it's reachable the same way.

### Attribute reference

| Field | Type | Description |
|---|---|---|
| `id` | integer | Internal row id. |
| `geom` | geometry (PostGIS) | Footprint polygon/multipolygon, WGS 84 / EPSG:4326. |
| `surface` | float (m²) | Footprint area of the detected array. *Model estimate.* |
| `kwp` | float (kWp) | Estimated installed capacity (DC). *Model estimate.* |
| `tilt` | integer (°) | Estimated panel tilt from horizontal. *Model estimate.* |
| `azimuth` | integer (°) | Estimated panel orientation, clockwise from true north. *Model estimate.* |
| `first_seen` | integer (year) | Earliest imagery vintage the installation was detected in. |
| `last_seen` | integer (year) | Most recent imagery vintage it was detected in. |
| `n_vintages` | integer | Number of distinct vintages independently detecting it — a rough confidence signal. |
| `sources` | string | Comma-separated source indices (e.g. `"0,1"`) — see [Source encoding](#source-encoding). |
| `frpv_proba` | float [0–1], nullable | Match probability against the FRPV reference source, where a comparison was attempted. |
| `false_positive` | boolean, nullable | Set when flagged as not a real installation (by review or community annotation). Null ≠ confirmed true positive. |
| `false_positive_source` | string, nullable | What flagged it, when `false_positive` is set. |
| `insee` | string | INSEE commune code. |
| `dpt` | string | Two-character département code (`"2A"`/`"2B"` for Corse). |
| `rnb_id` | string, nullable | Matching building id in the French [RNB](https://rnb.beta.gouv.fr/), when confidently matched. |

**Estimated fields are model outputs, not ground truth** — treat `surface`, `kwp`, `tilt`,
`azimuth`, and `frpv_proba` as estimates with the error characteristics described in the
[Registry Audit](https://deeppvmapper.fr/content/main-results.html), not as surveyed or
self-reported values.

### Source encoding

`sources` is comma-separated because one installation can be corroborated by more than
one source.

| Index | Source | Description |
|---|---|---|
| `0` | DPVM | DeepPVMapper's own aerial-imagery detection pipeline. |
| `1` | FRPV | Matched against the FRPV reference dataset. |
| `2` | OSM | Contributed via OpenStreetMap. |
| `3` | Manual correction | Submitted through the map's annotation tools. |
| `4` | Recall sample | Recovered from a targeted recall-annotation pass. |

## Static bulk downloads

For the full registry with no API/rate limits, or if you'd rather work offline:

- **Zenodo** (permanent DOI, versioned releases): https://zenodo.org/records/19188878
- **Hugging Face**: https://huggingface.co/datasets/gabrielkasmi/openpvmapper

Both ship the same schema as above, as a single GeoJSON file.

## License & attribution

- **Data** (the `detections` table / registry, including everything returned by this
  API): [**CC-BY 4.0**](https://creativecommons.org/licenses/by/4.0/) — free to use,
  share, and adapt, including commercially, **provided you attribute the source**. No
  share-alike requirement: you are not obliged to release derivative works under the
  same license. See the [Zenodo record](https://zenodo.org/records/19188878) for the
  versioned release and the citation to use in academic work.
- **Code** (this repo, the API/RPC definitions, the map application): MIT — see
  [LICENSE](https://github.com/gabrielkasmi/deeppvmapper/blob/main/LICENSE).

**Attribution:** *DeepPVMapper — Gabriel Kasmi*, with a link back to
[deeppvmapper.fr](https://deeppvmapper.fr).

## Contact

Questions, higher-volume access needs, or something you'd like exposed that isn't listed
above — Gabriel Kasmi: [gabriel.kasmi.services@gmail.com](mailto:gabriel.kasmi.services@gmail.com).
