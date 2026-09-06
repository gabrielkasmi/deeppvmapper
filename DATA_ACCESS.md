# OpenPVMapper Public Data Contract

**Contract version:** 0.1

**Registry:** OpenPVMapper PV Detection Registry

**Access:** Public, read-only

**Format:** JSON / GeoJSON

**Coordinate reference system:** WGS 84 (EPSG:4326)

**Geographical coverage:** Metropolitan France (Mainland and Corsica)

## 1. Scope

The DeepPVMapper Public Data Contract defines the public programmatic interface to the **DeepPVMapper PV Detection Registry**.

The registry contains georeferenced detections of photovoltaic installations in France, together with estimated physical attributes, temporal information, source provenance, and selected reference-data matches.

The public interface provides:

* access to individual detection records;
* spatial queries;
* geographic-zone extraction;
* pre-computed département-level statistics;
* versioned bulk releases of the registry.

This document defines the public data contract: available resources, field definitions, data types, spatial and temporal semantics, recommended filtering, data-quality characteristics, and licensing terms.

## 2. Access

### 2.1 Base URL

```text
https://zelhliylrlktnasircwp.supabase.co
```

The API follows the [PostgREST](https://postgrest.org/en/stable/references/api/tables_views.html) query conventions.

## 2.2 Authentication

The public API is accessible using the project's **Supabase publishable API key**.

No account or user registration is required.

Public requests should include the following headers:

```http
apikey: sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi
Authorization: Bearer sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi
```

The publishable key is intended for public client-side use and does not grant access to private project resources. Access to public resources is controlled by the API's server-side access policies.

Example:

```bash
curl "https://zelhliylrlktnasircwp.supabase.co/rest/v1/detections?select=id,surface,kwp,dpt&limit=5" \
  -H "apikey: sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi" \
  -H "Authorization: Bearer sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi"
```

## 3. Resources

The API exposes four query resources.

| Resource                 | Method | Purpose                                               |
| ------------------------ | ------ | ----------------------------------------------------- |
| `detections`             | `GET`  | Query individual detection records                    |
| `get_detections_bbox`    | `POST` | Query detections within a bounding box                |
| `get_detections_in_zone` | `POST` | Query detections within an arbitrary GeoJSON geometry |
| `dept_*_stats`           | `POST` | Retrieve département-level aggregates                 |


## 4. Detection Registry

### `GET /rest/v1/detections`

Returns records from the DeepPVMapper detection registry.

The endpoint supports standard PostgREST filtering, column selection, ordering, and pagination.

#### Example

Retrieve detections in département 33 with an estimated capacity above 9 kWp:

```bash
curl "https://zelhliylrlktnasircwp.supabase.co/rest/v1/detections?dpt=eq.33&kwp=gt.9&select=id,geom,surface,kwp,first_seen" \
  -H "apikey: sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi" \
  -H "Authorization: Bearer sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi"
```

#### Query recommendations

Use explicit column selection and pagination for large queries.

For spatial extraction, prefer the spatial resources described below, which return geometries directly as GeoJSON.

## 5. Spatial Resources

### 5.1 Bounding-box query

#### `POST /rest/v1/rpc/get_detections_bbox`

Returns detections intersecting a geographic bounding box.

This resource is intended primarily for map and viewport queries.

#### Parameters

| Parameter   | Type    | Required | Description                           |
| ----------- | ------- | -------: | ------------------------------------- |
| `min_lon`   | number  |      Yes | Minimum longitude                     |
| `min_lat`   | number  |      Yes | Minimum latitude                      |
| `max_lon`   | number  |      Yes | Maximum longitude                     |
| `max_lat`   | number  |      Yes | Maximum latitude                      |
| `max_count` | integer |       No | Maximum number of returned detections |

Default `max_count`: **2,000**.

#### Example

```bash
curl "https://zelhliylrlktnasircwp.supabase.co/rest/v1/rpc/get_detections_bbox" \
  -H "apikey: sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi" \
  -H "Authorization: Bearer sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi" \
  -H "Content-Type: application/json" \
  -d '{
    "min_lon": -0.6,
    "min_lat": 44.8,
    "max_lon": -0.5,
    "max_lat": 44.9,
    "max_count": 2000
  }'
```

#### Response

Returns a GeoJSON `Feature` array.

Each feature contains:

```json
{
  "type": "Feature",
  "geometry": {},
  "properties": {
    "surface": 123.4,
    "kwp": 25.0,
    "year": 2022
  }
}
```

The `year` property corresponds to `first_seen`.

The response contains a reduced attribute set intended for spatial visualization.

---

### 5.2 Geographic-zone query

#### `POST /rest/v1/rpc/get_detections_in_zone`

Returns detections contained within a supplied GeoJSON geometry.

Unlike the bounding-box resource, the query uses the supplied geometry for exact spatial filtering.

### Parameters

| Parameter       | Type             | Required | Description                           |
| --------------- | ---------------- | -------: | ------------------------------------- |
| `zone_geometry` | GeoJSON Geometry |      Yes | Query geometry                        |
| `max_count`     | integer          |       No | Maximum number of returned detections |

Default `max_count`: **300,000**.

#### Example

```bash
curl "https://zelhliylrlktnasircwp.supabase.co/rest/v1/rpc/get_detections_in_zone" \
  -H "apikey: sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi" \
  -H "Authorization: Bearer sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi" \
  -H "Content-Type: application/json" \
  -d '{
    "zone_geometry": {
      "type": "Polygon",
      "coordinates": [[
        [-0.6,44.8],
        [-0.5,44.8],
        [-0.5,44.9],
        [-0.6,44.9],
        [-0.6,44.8]
      ]]
    }
  }'
```

The response is a GeoJSON `Feature` array.

This resource can be used with administrative boundaries or arbitrary study-area geometries.

## 6. Aggregate Resources

Pre-computed aggregate resources are provided for département-level analysis.

| Resource              | Output                                                          |
| --------------------- | --------------------------------------------------------------- |
| `dept_capacity_stats` | Capacity and system counts by département                       |
| `dept_yearly_stats`   | System counts and capacity by département and `first_seen` year |
| `dept_source_stats`   | System counts by département and source                         |

Each resource returns approximately 94–96 département-level records.

### `dept_capacity_stats`

Returns:

```text
dpt
n_systems
total_kwp
rank_by_capacity
```

Example:

```bash
curl "https://zelhliylrlktnasircwp.supabase.co/rest/v1/rpc/dept_capacity_stats" \
  -X POST \
  -H "apikey: sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi" \
  -H "Authorization: Bearer sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi"
```

## 7. Detection Schema

Each detection record follows the schema below.

| Field                   | Type     | Nullable | Unit / Format    | Definition                                                         |
| ----------------------- | -------- | -------: | ---------------- | ------------------------------------------------------------------ |
| `id`                    | integer  |       No | —                | Registry identifier                                                |
| `geom`                  | geometry |       No | EPSG:4326        | Detected PV footprint                                              |
| `surface`               | float    |       No | m²               | Estimated area of the detected PV array                            |
| `kwp`                   | float    |       No | kWp              | Estimated installed DC capacity                                    |
| `tilt`                  | integer  |       No | degrees          | Estimated panel tilt from horizontal                               |
| `azimuth`               | integer  |       No | degrees          | Estimated orientation, clockwise from true north                   |
| `first_seen`            | integer  |       No | year             | Earliest imagery vintage in which the installation was detected    |
| `last_seen`             | integer  |       No | year             | Most recent imagery vintage in which the installation was detected |
| `n_vintages`            | integer  |       No | count            | Number of distinct imagery vintages detecting the installation     |
| `sources`               | string   |       No | comma-separated  | Detection source identifiers                                       |
| `frpv_proba`            | float    |      Yes | [0,1]            | Probability associated with an FRPV reference-data match           |
| `false_positive`        | boolean  |      Yes | —                | False-positive flag                                                |
| `false_positive_source` | string   |      Yes | —                | Source of the false-positive flag                                  |
| `insee`                 | string   |       No | INSEE code       | Commune identifier                                                 |
| `dpt`                   | string   |       No | département code | French département identifier                                      |
| `rnb_id`                | string   |      Yes | RNB identifier   | Matching RNB building identifier, when available                   |

The `id` field identifies an individual registry record and should be included when referencing a specific detection in a data-quality report.


## 8. Geometry

Detection geometries are provided in **WGS 84 / EPSG:4326**.

The geometry represents the footprint of the detected PV array rather than necessarily the footprint of the underlying building.

Depending on the detection, `geom` may be a polygon or multipolygon.

## 9. Data Semantics

### 9.1 Model-derived attributes

The following attributes are model-derived estimates:

* `surface`
* `kwp`
* `tilt`
* `azimuth`
* `frpv_proba`

These values should be treated as estimates rather than surveyed measurements, declared installations, or ground-truth observations.

Methodological details and reported error characteristics are available in the [Pipeline documentation](https://deeppvmapper.fr/content/pipeline.html).

### 9.2 Temporal attributes

`first_seen` is the earliest imagery vintage in which an installation was detected.

`last_seen` is the most recent imagery vintage in which an installation was detected.

These fields describe **observed presence in the available imagery**. They do not necessarily correspond to installation, commissioning, or construction dates.

`n_vintages` records the number of distinct imagery vintages in which the installation was independently detected.

It can be used as an additional confidence or persistence signal.

### 9.3 False-positive flag

`false_positive` indicates that a detection has been flagged as a false positive.

A null value indicates that no false-positive status is recorded. It should not be interpreted as a confirmed true-positive classification.

## 10. Recommended Filtering

For applications requiring a good balance between **precision and spatial coverage**, we recommend using:

```text
frpv_proba >= 0.1
```

as a practical filtering threshold when `frpv_proba` is available.

This threshold is intended as an operating point providing a useful balance between retaining detections and reducing false positives.

`frpv_proba` is nullable. A null value indicates that an FRPV comparison was not available for the detection and should not be interpreted as a probability of zero.

The appropriate threshold may depend on the intended application. Users performing high-precision analyses may choose a more restrictive threshold, while recall-oriented applications may retain a broader set of detections.

## 11. Source Encoding

The `sources` field contains one or more comma-separated source identifiers.

| Index | Source            | Definition                                         |
| ----: | ----------------- | -------------------------------------------------- |
|   `0` | DPVM              | DeepPVMapper aerial-imagery detection pipeline     |
|   `1` | FRPV              | Match against the FRPV reference dataset           |
|   `2` | OSM               | OpenStreetMap contribution                         |
|   `3` | Manual correction | Submitted through map annotation tools             |
|   `4` | Recall sample     | Recovered during a targeted recall-annotation pass |

A detection may reference multiple sources.

For example:

```text
0
```

indicates a DeepPVMapper detection, while:

```text
0,1
```

indicates a detection associated with both DeepPVMapper and FRPV.

## 12. Data Quality and Completeness

The registry is provided **as-is** and should be considered a **PV detection dataset rather than a complete inventory of photovoltaic installations**.

Under the evaluation protocol described in the Registry Audit, the current detection pipeline has an estimated **recall of approximately 0.6**.

As a consequence:

* a substantial fraction of existing PV installations may not be represented in the registry;
* the absence of a detection must not be interpreted as evidence that no PV installation exists at a given location;
* aggregate counts should be interpreted as detected-system counts rather than exhaustive installed-system counts;
* estimates derived from the registry may be affected by incomplete detection coverage.

The reported recall depends on the evaluation dataset, imagery coverage, and evaluation protocol. Refer to the [Registry Audit](https://deeppvmapper.fr/content/main-results.html) for methodological details.

## 13. Data Quality & Disclaimer

DeepPVMapper is an **open research and collaborative data project** providing publicly accessible PV detection data and tools.

The registry is provided as an open-data and research resource. It is **not a certified, exhaustive, or authoritative inventory of photovoltaic installations**, and no guarantee is made that the registry accurately or completely represents the photovoltaic installations existing at any given location or date.

### 13.1 Data provided "as is"

The data and API are provided **"as is" and "as available"**, without warranties or representations regarding:

* accuracy;
* completeness;
* correctness;
* temporal currency;
* availability;
* continuity of service;
* fitness for a particular purpose;
* suitability for any specific technical, commercial, regulatory, financial, operational, or other use.

The information contained in the registry may include missing detections, false positives, inaccurate estimated attributes, outdated observations, classification errors, or other data-quality issues.

### 13.2 User responsibility

Users are responsible for independently assessing and validating the suitability of the data for their intended use.

In particular, users should independently verify data whenever accuracy, completeness, spatial precision, or recency is material to a decision or application.

This applies in particular to commercial, regulatory, financial, operational, planning, or other consequential uses.

### 13.3 No liability for downstream use

To the maximum extent permitted by applicable law, **DeepPVMapper, its contributors, and the project maintainers shall not be held responsible for decisions, losses, damages, costs, or other consequences arising from the use of, reliance on, or inability to use the registry or API**, including consequences resulting from incomplete, inaccurate, outdated, or missing data.

Users remain responsible for their own analyses, applications, products, services, and decisions based on the data.

### 13.4 No service-level commitment

The public API is provided on a best-effort basis.

No guarantee is made regarding uninterrupted availability, response times, retention of live data, backward compatibility of infrastructure, or continued operation of the public API.

For applications requiring stable, reproducible, or large-scale access, users should rely on the versioned bulk releases whenever possible.

### 13.5 Commercial use

Commercial use of the data is permitted under the applicable **CC BY 4.0** license.

Commercial use does not imply that DeepPVMapper provides data validation, certification, technical support, service-level commitments, warranties, or guarantees concerning the resulting application or service.

Users incorporating DeepPVMapper data into commercial products or services remain responsible for determining whether the data is sufficiently accurate and complete for their intended purpose.

## 14. Reporting Issues & Discussions

DeepPVMapper welcomes public feedback, questions, and contributions concerning the API and registry.

### GitHub Issues

Use [GitHub Issues](https://github.com/gabrielkasmi/deeppvmapper/issues) for **concrete problems or actionable reports**, including:

* API errors or broken endpoints;
* incorrect response formats;
* missing or incorrectly typed fields;
* suspected incorrect detections;
* erroneous attribute values;
* documentation errors;
* other reproducible technical problems.

When reporting an issue concerning a specific detection, include its `id` whenever possible, together with the relevant endpoint, request, error message, location, or supporting information.

### GitHub Discussions

Use the [**🔌 API & Data Discussion**](https://github.com/gabrielkasmi/deeppvmapper/discussions/19) for:

* questions about using the API;
* questions about the meaning or interpretation of the data;
* discussion of possible use cases;
* suggestions for future improvements;
* methodological or data-related discussions;
* broader feedback that does not correspond to a specific bug or error.

Public feedback may be reviewed and, where appropriate, incorporated into subsequent registry updates and contract revisions.

## 15. Live API vs. Bulk Releases

The live API provides access to the current registry.

For country-scale processing, offline analysis, or repeated access to the complete dataset, use the versioned bulk releases.

### Zenodo

[Zenodo](https://zenodo.org/records/19188878) provides the versioned research release and permanent DOI.

### Hugging Face

[Hugging Face](https://huggingface.co/datasets/gabrielkasmi/openpvmapper) provides the registry in a machine-learning-friendly format.

Bulk releases contain the core detection schema defined by this contract.

## 16. Versioning

**Current contract version:** `0.1`

The data contract version identifies the public schema and API interface.

Changes affecting:

* field names;
* field types;
* field semantics;
* endpoint parameters;
* response formats;

will result in a new contract version.

Changes to individual detection records, newly incorporated imagery, model updates, or registry corrections do not necessarily constitute a contract-version change.

Bulk dataset releases are versioned independently and should be cited using their corresponding release or DOI.

## 17. Usage

The API operates on shared project infrastructure.

For large-scale processing:

* use the bulk releases when the complete registry is required;
* use aggregate resources when individual detections are unnecessary;
* use spatial resources for geographic extraction;
* paginate large queries to the detection registry;
* cache repeated requests where appropriate.

The API is intended for programmatic access to the live registry. Bulk releases should be preferred when the complete dataset is required for offline or large-scale processing.

## 18. License

### Data

The DeepPVMapper detection registry is released under **CC BY 4.0**.

The data may be used, copied, modified, redistributed, and incorporated into commercial applications, provided that appropriate attribution is given.

There is no share-alike requirement.

The versioned [Zenodo record](https://zenodo.org/records/19188878) provides the corresponding dataset release and recommended academic citation.

#### Attribution

> DeepPVMapper — Gabriel Kasmi

with a link to:

https://deeppvmapper.fr

### Code

The DeepPVMapper codebase, including the API and map application, is released under the **MIT License**. See the repository [LICENSE](https://github.com/gabrielkasmi/deeppvmapper/blob/main/LICENSE).


## 19. Contact

For general contact:

**Gabriel Kasmi**
[gabriel.kasmi@deeppvmapper.fr](mailto:gabriel.kasmi@deeppvmapper.fr)

For API questions, data-usage questions, and broader discussion of the public data interface, use **DeepPVMapper Discussions** on GitHub.

For concrete bugs, API failures, data-quality reports, or reproducible technical issues, please raise a GitHub Issue.

For research or collaboration inquiries, please use the contact information provided in the main DeepPVMapper repository.
