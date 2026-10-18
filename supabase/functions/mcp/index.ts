// DeepPVMapper MCP server
//
// Exposes the public OpenPVMapper PV-detection registry (see DATA_ACCESS.md
// at the repo root) as MCP tools, so an LLM/agent can query real French
// rooftop-solar detection data instead of guessing.
//
// Deployed as a Supabase Edge Function; the MCP endpoint is /functions/v1/mcp/mcp.

import { McpServer, StreamableHttpTransport } from 'mcp-lite'
import { z } from 'zod'
import { Hono } from 'hono'

// --- Config -------------------------------------------------------------
//
// Defaults match the public data contract in DATA_ACCESS.md. The key is a
// Supabase "publishable" key: it is meant to be public and only grants
// access to the read-only public resources described in that contract.
// Override with `supabase secrets set` if the project or key ever changes.

// This file runs under Deno (Supabase Edge Functions) and, via stdio.ts, under Node (used by the
// Glama build). `readEnv` hides the difference.
// deno-lint-ignore no-explicit-any
const runtime = globalThis as any
function readEnv(name: string): string | undefined {
  return runtime.Deno?.env?.get(name) ?? runtime.process?.env?.[name]
}

const API_URL = readEnv('DEEPPVMAPPER_API_URL') ?? 'https://zelhliylrlktnasircwp.supabase.co'
const API_KEY =
  readEnv('DEEPPVMAPPER_API_KEY') ?? 'sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi'

const DATA_QUALITY_NOTE =
  'Data quality note: this is a detection dataset, not an exhaustive inventory ' +
  '(estimated recall ~0.6). A missing detection does not mean no PV installation ' +
  'exists there. kwp, surface, tilt and azimuth are model estimates, not surveyed values. ' +
  'In this server, quality_filter=true keeps only detections identified by 2 or more ' +
  'sources (see source_encoding in get_data_quality_reference). ' +
  'Call get_data_quality_reference for the full picture before assessing fitness for a specific use case.'

// Source catalog for the `sources` field (DATA_ACCESS.md §11).
// `sources` answers "where do we have this installation?": each listed code is a dataset/process
// that contains it. Example: "0,1" = present in both DeepPVMapper and FRPV.
// Users ask by NAME, so every source has a key, a name and aliases that resolve to its code.
const SOURCES_MEANING =
  'The `sources` field answers "where do we have this installation?": each listed code is a ' +
  'dataset or process that contains it. Example: "0,1" = present in both DeepPVMapper (DPVM) and ' +
  'FRPV; "0,2" = present in DeepPVMapper and OpenStreetMap.'

const SOURCE_CATALOG = [
  {
    code: '0',
    key: 'dpvm',
    name: 'DeepPVMapper (DPVM)',
    aliases: ['dpvm', 'deeppvmapper', 'deep pv mapper', 'kasmi', 'kasmi et al', 'kasmi et al.', 'pipeline'],
    origin:
      'Detections produced by the DeepPVMapper aerial-imagery pipeline (Kasmi et al.), run on the ' +
      'latest IGN imagery available at the time of the registry release (July 2026 for the current ' +
      'release; this will change with future releases). Source code: ' +
      'https://github.com/gabrielkasmi/deeppvmapper — PhD thesis: https://theses.fr/2024UPSLM027',
  },
  {
    code: '1',
    key: 'frpv',
    name: 'FRPV reference dataset (Thébault et al.)',
    aliases: [
      'frpv',
      'thebault',
      'thébault',
      'thebault et al',
      'thebault et al.',
      'thebaud',
      'thebaud et al',
      'applied energy',
      'comprehensive building-wise rooftop photovoltaic system detection',
    ],
    origin:
      'Building-wise rooftop PV detections from Thébault et al. (2025), "A comprehensive ' +
      'building-wise rooftop photovoltaic system detection in heterogeneous urban and rural areas: ' +
      'application to French territories", Applied Energy 388, 125630. ' +
      'https://www.sciencedirect.com/science/article/pii/S0306261925003605 (the publisher page may ' +
      'sit behind a human check).',
  },
  {
    code: '2',
    key: 'osm',
    name: 'OpenStreetMap (OSM)',
    aliases: ['osm', 'openstreetmap', 'open street map'],
    origin:
      'PV installations mapped in OpenStreetMap, taken from the latest OSM data available at the ' +
      'time of the registry release (July 2026 for the current release; this will change with ' +
      'future releases).',
  },
  {
    code: '3',
    key: 'manual',
    name: 'Manual correction',
    aliases: [
      'manual',
      'manual correction',
      'manual corrections',
      'correction',
      'manual validation',
      'precision sample',
      'precision samples',
      'annotation',
      'annotations',
      'annotator',
      'annotators',
    ],
    origin:
      'Manually validated samples (the precision samples) built when constructing the validation set ' +
      'of the DeepPVMapper evaluation paper (https://arxiv.org/abs/2609.16294). In future this ' +
      'source will also include contributions from annotators submitted through the map annotation tools.',
  },
  {
    code: '4',
    key: 'recall_sample',
    name: 'Recall sample',
    aliases: ['recall', 'recall sample', 'recall samples', 'recall annotation', 'recall_sample'],
    origin:
      'Manually validated recall samples built when constructing the validation set of the same ' +
      'paper (https://arxiv.org/abs/2609.16294): installations recovered during a targeted ' +
      'recall-annotation pass.',
  },
]

const SOURCE_LABELS: Record<string, string> = Object.fromEntries(
  SOURCE_CATALOG.map((s) => [s.code, `${s.name} [${s.key}]`]),
)

// Accepts a code (0-4, number or string) or a name/key/alias, case-insensitively.
function resolveSource(input: number | string) {
  const norm = String(input).trim().toLowerCase()
  const hit = SOURCE_CATALOG.find(
    (s) =>
      s.code === norm ||
      s.key === norm ||
      s.name.toLowerCase() === norm ||
      s.aliases.some((a) => a.toLowerCase() === norm),
  )
  if (!hit) {
    const valid = SOURCE_CATALOG.map((s) => `${s.key} (${s.code})`).join(', ')
    throw new Error(`Unknown source "${input}". Valid sources: ${valid}. Call get_source_reference for aliases.`)
  }
  return hit
}

function decodeSources(sources: unknown) {
  if (typeof sources !== 'string' || sources.length === 0) return []
  return sources.split(',').map((c) => {
    const s = SOURCE_CATALOG.find((x) => x.code === c.trim())
    return { code: c.trim(), key: s?.key ?? null, name: s?.name ?? 'unknown source code', origin: s?.origin ?? null }
  })
}

async function callRest(path: string, params: [string, string][]) {
  const url = new URL(`${API_URL}/rest/v1/${path}`)
  for (const [k, v] of params) url.searchParams.append(k, v)
  const res = await fetch(url, {
    headers: { apikey: API_KEY, Authorization: `Bearer ${API_KEY}` },
  })
  if (!res.ok) throw new Error(`DeepPVMapper API error ${res.status}: ${await res.text()}`)
  return res.json()
}

// Like callRest, but asks PostgREST for the exact total match count (via the
// `Prefer: count=exact` / `Content-Range` mechanism) while only actually
// fetching up to `rangeEnd + 1` rows. Used for aggregation: we can report an
// exact count of matches even when we only sum a bounded sample of rows.
async function callRestWithCount(path: string, params: [string, string][], rangeEnd: number) {
  const url = new URL(`${API_URL}/rest/v1/${path}`)
  for (const [k, v] of params) url.searchParams.append(k, v)
  const res = await fetch(url, {
    headers: {
      apikey: API_KEY,
      Authorization: `Bearer ${API_KEY}`,
      Prefer: 'count=exact',
      Range: `0-${rangeEnd}`,
    },
  })
  if (!res.ok) throw new Error(`DeepPVMapper API error ${res.status}: ${await res.text()}`)
  const rows = (await res.json()) as Array<Record<string, unknown>>
  const contentRange = res.headers.get('content-range') // e.g. "0-4999/12345" or "0-4999/*"
  let totalCount: number | null = null
  if (contentRange) {
    const m = contentRange.match(/\/(\d+|\*)$/)
    if (m && m[1] !== '*') totalCount = parseInt(m[1], 10)
  }
  return { rows, totalCount }
}

async function callRpc(fn: string, body: Record<string, unknown>) {
  const res = await fetch(`${API_URL}/rest/v1/rpc/${fn}`, {
    method: 'POST',
    headers: {
      apikey: API_KEY,
      Authorization: `Bearer ${API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`DeepPVMapper API error ${res.status}: ${await res.text()}`)
  return res.json()
}

function textContent(payload: unknown) {
  return { content: [{ type: 'text' as const, text: JSON.stringify(payload, null, 2) }] }
}

function errorContent(err: unknown) {
  return {
    content: [
      { type: 'text' as const, text: `Error: ${err instanceof Error ? err.message : String(err)}` },
    ],
    isError: true,
  }
}

// Computes, from a batch of already-fetched detection rows, what fraction
// show each confidence signal. This lets an agent reason about how much to
// trust a *specific query's* results, on top of the registry-wide caveats
// returned by get_data_quality_reference.
type QualityRow = { sources?: string; n_vintages?: number; frpv_proba?: number | null }

function computeQualitySummary(rows: QualityRow[]) {
  const n = rows.length
  if (n === 0) return { sample_size: 0 }
  let crossValidated = 0
  let multiVintage = 0
  let probaSum = 0
  let probaCount = 0
  for (const r of rows) {
    if (typeof r.sources === 'string' && r.sources.includes(',')) crossValidated++
    if (typeof r.n_vintages === 'number' && r.n_vintages >= 2) multiVintage++
    if (typeof r.frpv_proba === 'number') {
      probaSum += r.frpv_proba
      probaCount++
    }
  }
  return {
    sample_size: n,
    pct_cross_validated_2plus_sources: Math.round((crossValidated / n) * 1000) / 10,
    pct_detected_2plus_vintages: Math.round((multiVintage / n) * 1000) / 10,
    pct_with_frpv_reference_match: Math.round((probaCount / n) * 1000) / 10,
    avg_frpv_proba_when_available: probaCount > 0 ? Math.round((probaSum / probaCount) * 1000) / 1000 : null,
  }
}

// --- MCP server -----------------------------------------------------------

export const mcp = new McpServer({
  name: 'deeppvmapper',
  version: '0.2.0',
  schemaAdapter: (schema) => z.toJSONSchema(schema as z.ZodType),
})

mcp.tool('get_data_quality_reference', {
  description:
    'Return the DeepPVMapper/OpenPVMapper registry\'s documented data-quality characteristics: ' +
    'estimated detection recall, which fields are model-derived estimates vs. structural/observed ' +
    'fields, the source-encoding table, the recommended confidence threshold, confidence signals, ' +
    'and licensing/liability terms. Call this before advising how much to trust a result for a ' +
    'specific use case (e.g. exploratory research vs. a commercial or regulatory decision) — pair ' +
    'it with the quality_summary attached to search_detections / aggregate_detection_capacity ' +
    'results, which reflects the specific query rather than the registry as a whole.',
  inputSchema: z.object({}),
  handler: () =>
    textContent({
      estimated_recall: 0.6,
      recall_meaning:
        'Approximately 60% of existing PV installations are estimated to be represented in the ' +
        'registry under the evaluation protocol. A missing detection is not evidence of absence.',
      model_derived_fields: ['surface', 'kwp', 'tilt', 'azimuth', 'frpv_proba'],
      structural_fields: [
        'id',
        'geom',
        'first_seen',
        'last_seen',
        'n_vintages',
        'sources',
        'insee',
        'dpt',
        'rnb_id',
      ],
      recommended_quality_threshold: { field: 'frpv_proba', operator: '>=', value: 0.1 },
      quality_filter_in_this_server:
        'quality_filter=true keeps only detections identified by 2 or more sources (sources ' +
        'lists at least two codes). It is separate from the contract-recommended frpv_proba >= 0.1 ' +
        'threshold, available as min_frpv_proba.',
      confidence_signals: {
        cross_validated:
          'sources field lists 2+ distinct source codes (see source_encoding) — detection ' +
          'present in more than one source, not just the automated pipeline alone.',
        persistent:
          'n_vintages >= 2 — detected independently across multiple years of imagery, reducing ' +
          'the chance of a one-off artifact.',
      },
      sources_meaning: SOURCES_MEANING,
      source_encoding: Object.fromEntries(
        SOURCE_CATALOG.map((s) => [s.code, `${s.name} [${s.key}] — ${s.origin}`]),
      ),
      license: 'CC BY 4.0 (data), MIT (code) — commercial use permitted with attribution.',
      suitable_for: [
        'Exploratory research and academic analysis',
        'Aggregate or statistical trend estimation at département/national scale',
        'Prioritization or screening (e.g. "where is PV density highest") where false negatives are tolerable',
      ],
      use_with_caution_for: [
        'Site-specific commercial, regulatory, financial, or operational decisions — the data ' +
          'contract explicitly disclaims fitness for these without independent verification',
        'Exhaustive-inventory claims — recall ~0.6 means real coverage is understated, especially ' +
          'in areas or periods with sparser imagery',
        'Precise capacity figures for a single installation — kwp/surface/tilt/azimuth are model ' +
          'estimates, not surveyed measurements',
      ],
      source: 'https://deeppvmapper.fr (DATA_ACCESS.md, contract version 0.1)',
    }),
})

const CapacityStatsSchema = z.object({
  dpt: z
    .string()
    .optional()
    .describe('French département code, e.g. "33" for Gironde. Omit for all départements.'),
  top_n: z
    .number()
    .int()
    .positive()
    .max(96)
    .optional()
    .describe('If set and dpt is omitted, return only the top N départements by installed capacity.'),
})

mcp.tool('get_department_capacity_stats', {
  description:
    'Get installed rooftop-PV capacity (total kWp) and system counts for one or all French ' +
    'départements, from the DeepPVMapper/OpenPVMapper registry. This is a fixed, pre-computed ' +
    'department-wide aggregate with no other filters — use aggregate_detection_capacity instead ' +
    'if you need a filtered subset (e.g. only cross-validated detections). ' + DATA_QUALITY_NOTE,
  inputSchema: CapacityStatsSchema,
  handler: async (args: z.infer<typeof CapacityStatsSchema>) => {
    try {
      const rows = (await callRpc('dept_capacity_stats', {})) as Array<{
        dpt: string
        n_systems: number
        total_kwp: number
        rank_by_capacity: number
      }>
      let result = rows
      if (args.dpt) result = rows.filter((r) => r.dpt === args.dpt)
      else if (args.top_n) {
        result = [...rows].sort((a, b) => a.rank_by_capacity - b.rank_by_capacity).slice(0, args.top_n)
      }
      return textContent(result)
    } catch (err) {
      return errorContent(err)
    }
  },
})

const YearlyStatsSchema = z.object({
  dpt: z.string().optional().describe('French département code, e.g. "33". Omit for all départements.'),
})

mcp.tool('get_department_yearly_stats', {
  description:
    'Get yearly system counts and capacity by département, based on first-seen imagery year. ' +
    'Useful for tracking apparent PV deployment growth over time. ' + DATA_QUALITY_NOTE,
  inputSchema: YearlyStatsSchema,
  handler: async (args: z.infer<typeof YearlyStatsSchema>) => {
    try {
      const rows = (await callRpc('dept_yearly_stats', {})) as Array<Record<string, unknown>>
      const result = args.dpt ? rows.filter((r) => r.dpt === args.dpt) : rows
      return textContent(result)
    } catch (err) {
      return errorContent(err)
    }
  },
})

// Shared building block: both search_detections and aggregate_detection_capacity
// filter on the same dimensions (location, capacity range, cross-validation).
const DetectionFilterFields = {
  dpt: z.string().optional().describe('French département code, e.g. "33".'),
  insee: z.string().optional().describe('INSEE commune code.'),
  min_kwp: z.number().optional().describe('Minimum estimated installed capacity, in kWp.'),
  max_kwp: z.number().optional().describe('Maximum estimated installed capacity, in kWp.'),
  min_vintages: z
    .number()
    .int()
    .positive()
    .optional()
    .describe(
      'Minimum number of distinct imagery vintages (years) the installation was independently ' +
        'detected in. Use 2+ as a persistence/confidence signal, since a one-off detection in a ' +
        'single vintage is more likely to be a transient artifact.',
    ),
  cross_validated: z
    .boolean()
    .optional()
    .describe('Alias of quality_filter, kept for backward compatibility.'),
  quality_filter: z
    .boolean()
    .optional()
    .default(false)
    .describe(
      'If true, only include detections identified by 2 OR MORE sources' +
        '(the `sources` field lists at least two codes, e.g. "0,1" = DeepPVMapper pipeline + FRPV ' +
        'reference, "0,2" = pipeline + OpenStreetMap). Default false: all detections are included, ' +
        'including those found by a single source (typically the automated pipeline alone).',
    ),
  sources_include: z
    .array(z.union([z.number().int().min(0).max(4), z.string()]))
    .optional()
    .describe(
      'Only include detections present in ALL of these sources. Accepts names or codes, ' +
        'case-insensitive: dpvm/deeppvmapper/kasmi (0), frpv/thebault (1), osm/openstreetmap (2), ' +
        'manual/annotation (3), recall (4). Examples: ["osm"] = detections that are in ' +
        'OpenStreetMap; ["dpvm","frpv"] = present in both DeepPVMapper and FRPV. ' +
        'See get_source_reference for what each source is and where it comes from.',
    ),
  min_frpv_proba: z
    .number()
    .min(0)
    .max(1)
    .optional()
    .describe(
      'Minimum frpv_proba. The data contract recommends 0.1 as a precision/recall trade-off. ' +
        'Note: detections with a null frpv_proba (no FRPV comparison available) are excluded when ' +
        'this is set; null does not mean probability 0.',
    ),
  has_rnb_id: z
    .boolean()
    .optional()
    .describe(
      'true = only detections matched to an RNB building identifier (rnb_id not null); ' +
        'false = only detections without one.',
    ),
  has_frpv_proba: z
    .boolean()
    .optional()
    .describe(
      'true = only detections that have an FRPV comparison (frpv_proba not null); false = only ' +
        'those without one. A null frpv_proba means "no comparison available", not probability 0.',
    ),
  false_positive_status: z
    .enum(['flagged', 'not_flagged', 'unrecorded'])
    .optional()
    .describe(
      'flagged = false_positive is true; not_flagged = false_positive is false; unrecorded = ' +
        'false_positive is null (no status recorded — NOT a confirmed true positive).',
    ),
  min_surface: z.number().optional().describe('Minimum estimated PV array area, in m².'),
  max_surface: z.number().optional().describe('Maximum estimated PV array area, in m².'),
  min_tilt: z.number().optional().describe('Minimum estimated tilt from horizontal, in degrees.'),
  max_tilt: z.number().optional().describe('Maximum estimated tilt from horizontal, in degrees.'),
  min_azimuth: z
    .number()
    .optional()
    .describe('Minimum estimated azimuth, degrees clockwise from true north (180 = south).'),
  max_azimuth: z
    .number()
    .optional()
    .describe('Maximum estimated azimuth, degrees clockwise from true north (180 = south).'),
  min_first_seen: z
    .number()
    .int()
    .optional()
    .describe('Earliest imagery year of first detection, lower bound (e.g. 2018). Not an installation date.'),
  max_first_seen: z.number().int().optional().describe('Upper bound on first_seen imagery year.'),
  min_last_seen: z.number().int().optional().describe('Lower bound on last_seen imagery year.'),
  max_last_seen: z.number().int().optional().describe('Upper bound on last_seen imagery year.'),
  max_vintages: z
    .number()
    .int()
    .positive()
    .optional()
    .describe('Maximum number of imagery vintages (e.g. 1 = seen in a single year only).'),
}

function buildDetectionFilterParams(args: {
  dpt?: string
  insee?: string
  min_kwp?: number
  max_kwp?: number
  min_vintages?: number
  cross_validated?: boolean
  quality_filter?: boolean
  sources_include?: Array<number | string>
  min_frpv_proba?: number
  has_rnb_id?: boolean
  has_frpv_proba?: boolean
  false_positive_status?: 'flagged' | 'not_flagged' | 'unrecorded'
  min_surface?: number
  max_surface?: number
  min_tilt?: number
  max_tilt?: number
  min_azimuth?: number
  max_azimuth?: number
  min_first_seen?: number
  max_first_seen?: number
  min_last_seen?: number
  max_last_seen?: number
  max_vintages?: number
}): [string, string][] {
  const params: [string, string][] = []
  if (args.dpt) params.push(['dpt', `eq.${args.dpt}`])
  if (args.insee) params.push(['insee', `eq.${args.insee}`])
  if (args.min_kwp !== undefined) params.push(['kwp', `gte.${args.min_kwp}`])
  if (args.max_kwp !== undefined) params.push(['kwp', `lte.${args.max_kwp}`])
  if (args.min_vintages !== undefined) params.push(['n_vintages', `gte.${args.min_vintages}`])
  // sources is a comma-separated list of source codes (see DATA_ACCESS.md §11), e.g. "0,1".
  // A comma in the string means at least two distinct sources are listed.
  if (args.quality_filter || args.cross_validated) params.push(['sources', 'like.*,*'])
  // Repeated keys are ANDed by PostgREST: every listed code must appear in `sources`.
  for (const s of args.sources_include ?? []) params.push(['sources', `like.*${resolveSource(s).code}*`])
  if (args.min_frpv_proba !== undefined) params.push(['frpv_proba', `gte.${args.min_frpv_proba}`])
  if (args.has_rnb_id === true) params.push(['rnb_id', 'not.is.null'])
  if (args.has_rnb_id === false) params.push(['rnb_id', 'is.null'])
  if (args.has_frpv_proba === true) params.push(['frpv_proba', 'not.is.null'])
  if (args.has_frpv_proba === false) params.push(['frpv_proba', 'is.null'])
  if (args.false_positive_status === 'flagged') params.push(['false_positive', 'eq.true'])
  if (args.false_positive_status === 'not_flagged') params.push(['false_positive', 'eq.false'])
  if (args.false_positive_status === 'unrecorded') params.push(['false_positive', 'is.null'])
  const ranges: Array<[string, string, number | undefined]> = [
    ['surface', 'gte', args.min_surface],
    ['surface', 'lte', args.max_surface],
    ['tilt', 'gte', args.min_tilt],
    ['tilt', 'lte', args.max_tilt],
    ['azimuth', 'gte', args.min_azimuth],
    ['azimuth', 'lte', args.max_azimuth],
    ['first_seen', 'gte', args.min_first_seen],
    ['first_seen', 'lte', args.max_first_seen],
    ['last_seen', 'gte', args.min_last_seen],
    ['last_seen', 'lte', args.max_last_seen],
    ['n_vintages', 'lte', args.max_vintages],
  ]
  for (const [col, op, v] of ranges) if (v !== undefined) params.push([col, `${op}.${v}`])
  return params
}

const SearchDetectionsSchema = z.object({
  ...DetectionFilterFields,
  limit: z
    .number()
    .int()
    .positive()
    .max(200)
    .optional()
    .default(20)
    .describe('Maximum number of records to return (max 200).'),
})

mcp.tool('search_detections', {
  description:
    'Search individual rooftop-PV detections by département, commune (INSEE code), estimated ' +
    'capacity range, and/or cross-validation confidence (min_vintages, cross_validated). Returns a ' +
    'bounded list of detection records plus a quality_summary for the returned sample (use ' +
    'get_detections_in_bbox instead for map/spatial queries with geometry). ' + DATA_QUALITY_NOTE,
  inputSchema: SearchDetectionsSchema,
  handler: async (args: z.infer<typeof SearchDetectionsSchema>) => {
    try {
      const params: [string, string][] = [
        [
          'select',
          'id,surface,kwp,tilt,azimuth,first_seen,last_seen,n_vintages,sources,frpv_proba,dpt,insee,rnb_id',
        ],
        ['limit', String(args.limit)],
        ...buildDetectionFilterParams(args),
      ]
      const rows = (await callRest('detections', params)) as QualityRow[]
      return textContent({ results: rows, quality_summary: computeQualitySummary(rows) })
    } catch (err) {
      return errorContent(err)
    }
  },
})

const AggregateSchema = z.object({
  ...DetectionFilterFields,
  max_rows: z
    .number()
    .int()
    .positive()
    .max(20000)
    .optional()
    .default(5000)
    .describe(
      'Cap on the number of matching detection rows fetched to compute the capacity sum. If the ' +
        'true match count exceeds this, total_kwp is a partial lower bound and `truncated` is true ' +
        '— increase max_rows or narrow the filters (e.g. add dpt or insee) for an exact total.',
    ),
})

mcp.tool('aggregate_detection_capacity', {
  description:
    'Compute the total estimated installed capacity (kWp) and count of detections matching a set ' +
    'of filters — département, commune, capacity range, and cross-validation across sources ' +
    '(cross_validated) or imagery vintages (min_vintages) — plus a quality_summary for the summed ' +
    'sample. Unlike get_department_capacity_stats, which is a fixed pre-computed département-wide ' +
    'aggregate with no other filters, this tool sums a live filtered subset, up to max_rows ' +
    'detections. Example: "installed capacity in Gironde confirmed by at least two sources" -> ' +
    'dpt="33", cross_validated=true. ' + DATA_QUALITY_NOTE,
  inputSchema: AggregateSchema,
  handler: async (args: z.infer<typeof AggregateSchema>) => {
    try {
      const params: [string, string][] = [
        ['select', 'kwp,sources,n_vintages,frpv_proba'],
        ...buildDetectionFilterParams(args),
      ]
      const { rows, totalCount } = await callRestWithCount('detections', params, args.max_rows - 1)
      const typedRows = rows as Array<QualityRow & { kwp?: number }>
      const total_kwp = typedRows.reduce(
        (sum, r) => sum + (typeof r.kwp === 'number' ? r.kwp : 0),
        0,
      )
      const truncated = totalCount !== null && totalCount > rows.length
      return textContent({
        matched_count: totalCount ?? rows.length,
        summed_over_rows: rows.length,
        total_kwp: Math.round(total_kwp * 100) / 100,
        truncated,
        note: truncated
          ? `Only the first ${rows.length} of ${totalCount} matching detections were summed; ` +
            'total_kwp is a lower bound. Increase max_rows or narrow the filters for an exact total.'
          : undefined,
        quality_summary: computeQualitySummary(typedRows),
      })
    } catch (err) {
      return errorContent(err)
    }
  },
})

const BboxSchema = z.object({
  min_lon: z.number(),
  min_lat: z.number(),
  max_lon: z.number(),
  max_lat: z.number(),
  max_count: z
    .number()
    .int()
    .positive()
    .max(500)
    .optional()
    .default(100)
    .describe(
      'Maximum number of detections to return (max 500 here; the underlying API defaults to ' +
        '2000, capped lower to keep responses manageable for an LLM).',
    ),
})

mcp.tool('get_detections_in_bbox', {
  description:
    'Get rooftop-PV detections within a geographic bounding box (WGS84 lon/lat), including ' +
    'footprint geometry. Intended for map-style spatial queries over a small area. ' +
    DATA_QUALITY_NOTE,
  inputSchema: BboxSchema,
  handler: async (args: z.infer<typeof BboxSchema>) => {
    try {
      const rows = await callRpc('get_detections_bbox', {
        min_lon: args.min_lon,
        min_lat: args.min_lat,
        max_lon: args.max_lon,
        max_lat: args.max_lat,
        max_count: args.max_count,
      })
      return textContent(rows)
    } catch (err) {
      return errorContent(err)
    }
  },
})

const CommunityActivitySchema = z.object({
  recent_limit: z
    .number()
    .int()
    .positive()
    .max(50)
    .optional()
    .default(15)
    .describe('How many of the most recent pending contributions to list.'),
})

mcp.tool('get_community_activity', {
  description:
    'Get a snapshot of ongoing community contribution activity on the map: how many corrections ' +
    'are currently pending moderation, their breakdown by action type (add / modify / delete), and ' +
    'the most recent submissions (timestamp, action, target). Submitted free-text comments are ' +
    'intentionally excluded from this tool. This reflects unmoderated, unverified user activity, ' +
    'not the registry itself \u2014 do not present it as confirmed detection data.',
  inputSchema: CommunityActivitySchema,
  handler: async (args: z.infer<typeof CommunityActivitySchema>) => {
    try {
      const { rows: actionRows, totalCount } = await callRestWithCount(
        'annotations_pending_public',
        [['select', 'action']],
        4999,
      )
      const byAction: Record<string, number> = {}
      for (const r of actionRows as Array<{ action?: string }>) {
        const a = r.action ?? 'unknown'
        byAction[a] = (byAction[a] ?? 0) + 1
      }
      const recent = await callRest('annotations_pending_public', [
        ['select', 'created_at,action,target_id'],
        ['order', 'created_at.desc'],
        ['limit', String(args.recent_limit)],
      ])
      return textContent({
        n_pending_total: totalCount ?? actionRows.length,
        by_action: byAction,
        recent,
        note: 'Unmoderated, unverified community submissions \u2014 not yet reflected in the registry.',
      })
    } catch (err) {
      return errorContent(err)
    }
  },
})

// Approximate centroid of a GeoJSON Polygon's outer ring \u2014 fine at
// rooftop scale, where a simple vertex average and the true area centroid
// differ by a negligible amount.
function polygonCentroid(geometry: unknown): { lon: number; lat: number } | null {
  if (
    typeof geometry !== 'object' ||
    geometry === null ||
    (geometry as { type?: string }).type !== 'Polygon'
  ) {
    return null
  }
  const coords = (geometry as { coordinates?: unknown }).coordinates
  if (!Array.isArray(coords) || !Array.isArray(coords[0])) return null
  const ring = coords[0] as Array<[number, number]>
  if (ring.length === 0) return null
  let sumLon = 0
  let sumLat = 0
  for (const [lon, lat] of ring) {
    sumLon += lon
    sumLat += lat
  }
  return { lon: sumLon / ring.length, lat: sumLat / ring.length }
}

const CommunityAreaSchema = z
  .object({
    dpt: z
      .string()
      .optional()
      .describe(
        'Filter pending edits/deletions of EXISTING detections (action=modify or delete) to this ' +
          'département. Does not apply to proposed new additions (action=add), which have no ' +
          'département recorded on the pending item itself \u2014 use the bbox parameters for those.',
      ),
    min_lon: z.number().optional(),
    min_lat: z.number().optional(),
    max_lon: z.number().optional(),
    max_lat: z.number().optional(),
    scan_limit: z
      .number()
      .int()
      .positive()
      .max(2000)
      .optional()
      .default(500)
      .describe(
        'How many recent proposed additions (action=add) to scan for a bbox match. Only relevant ' +
          'when all four bbox parameters are given \u2014 there is no server-side spatial index on ' +
          'pending items, so matching is done by fetching this many of the most recent ones and ' +
          'checking their centroid against the box.',
      ),
    recent_limit: z
      .number()
      .int()
      .positive()
      .max(100)
      .optional()
      .default(30)
      .describe('Maximum number of matching items to return per category.'),
  })
  .refine(
    (a) =>
      (a.dpt !== undefined) ||
      (a.min_lon !== undefined && a.min_lat !== undefined && a.max_lon !== undefined && a.max_lat !== undefined),
    { message: 'Provide either dpt, or all four of min_lon/min_lat/max_lon/max_lat.' },
  )

mcp.tool('get_community_activity_in_area', {
  description:
    'Find pending, unmoderated community contributions relevant to a specific area, to answer ' +
    '"has anyone flagged anything here that is not in the registry yet?" Two independent filters: ' +
    'dpt finds pending edits/deletions (action=modify/delete) targeting existing detections in that ' +
    'département; a full bounding box (min_lon/min_lat/max_lon/max_lat) finds pending new additions ' +
    '(action=add) whose proposed footprint centroid falls inside it. Combine with ' +
    'get_department_capacity_stats / search_detections / get_detections_in_bbox for the confirmed ' +
    'registry picture, and present this separately and clearly as unverified, pending community ' +
    'input \u2014 not confirmed detection data. Free-text comments are never included.',
  inputSchema: CommunityAreaSchema,
  handler: async (args: z.infer<typeof CommunityAreaSchema>) => {
    try {
      const result: {
        pending_edits_or_deletions?: unknown
        pending_new_additions?: unknown
      } = {}

      if (args.dpt) {
        const rows = await callRest('annotations_pending_public', [
          ['select', 'created_at,action,target_id,target_dpt,target_insee'],
          ['action', 'in.(modify,delete)'],
          ['target_dpt', `eq.${args.dpt}`],
          ['limit', String(args.recent_limit)],
        ])
        result.pending_edits_or_deletions = rows
      }

      if (
        args.min_lon !== undefined &&
        args.min_lat !== undefined &&
        args.max_lon !== undefined &&
        args.max_lat !== undefined
      ) {
        const candidates = (await callRest('annotations_pending_public', [
          ['select', 'created_at,action,geometry'],
          ['action', 'eq.add'],
          ['limit', String(args.scan_limit)],
        ])) as Array<{ created_at: string; action: string; geometry: unknown }>

        const matches = []
        for (const c of candidates) {
          const centroid = polygonCentroid(c.geometry)
          if (
            centroid &&
            centroid.lon >= args.min_lon &&
            centroid.lon <= args.max_lon &&
            centroid.lat >= args.min_lat &&
            centroid.lat <= args.max_lat
          ) {
            matches.push({ created_at: c.created_at, action: c.action, geometry: c.geometry })
            if (matches.length >= args.recent_limit) break
          }
        }
        result.pending_new_additions = matches
      }

      return textContent({
        ...result,
        note:
          'Unmoderated, unverified community submissions \u2014 not yet reflected in the registry. ' +
          'Comments submitted with these items are intentionally excluded.',
      })
    } catch (err) {
      return errorContent(err)
    }
  },
})

// --- Counting helpers & overview tools --------------------------------------

// Exact match count via PostgREST `Prefer: count=exact`, fetching a single row only.
async function countDetections(filters: [string, string][]): Promise<number | null> {
  const { totalCount } = await callRestWithCount('detections', [['select', 'id'], ...filters], 0)
  return totalCount
}

function pct(part: number | null, whole: number | null): number | null {
  if (part === null || whole === null || whole === 0) return null
  return Math.round((part / whole) * 1000) / 10
}

type FilterArgs = Parameters<typeof buildDetectionFilterParams>[0]

const CountSchema = z.object({
  ...DetectionFilterFields,
  breakdown_rnb_id: z
    .boolean()
    .optional()
    .default(false)
    .describe(
      'If true, also return how many of the matching detections do / do not have an rnb_id ' +
        '(ignores has_rnb_id for this breakdown).',
    ),
})

mcp.tool('count_detections', {
  description:
    'Count detections matching a set of filters, exactly, without downloading rows (fast, no row ' +
    'cap). Use this for "how many ..." questions, e.g. "how many installations have an RNB id?" ' +
    '-> has_rnb_id=true; "how many are identified by 2+ sources?" -> quality_filter=true; ' +
    '"how many come from OpenStreetMap?" -> sources_include=[2]. Returns the matching count, the ' +
    'unfiltered registry total and the percentage. Counts are DETECTED systems, not an inventory. ' +
    DATA_QUALITY_NOTE,
  inputSchema: CountSchema,
  handler: async (args: z.infer<typeof CountSchema>) => {
    try {
      const filters = buildDetectionFilterParams(args as FilterArgs)
      const { has_rnb_id: _ignored, ...withoutRnb } = args as FilterArgs & { breakdown_rnb_id?: boolean }
      const baseFilters = buildDetectionFilterParams(withoutRnb as FilterArgs)
      const [matched, registryTotal, withRnb, withoutRnbCount] = await Promise.all([
        countDetections(filters),
        countDetections([]),
        args.breakdown_rnb_id ? countDetections([...baseFilters, ['rnb_id', 'not.is.null']]) : null,
        args.breakdown_rnb_id ? countDetections([...baseFilters, ['rnb_id', 'is.null']]) : null,
      ])
      return textContent({
        matched_count: matched,
        registry_total_count: registryTotal,
        pct_of_registry: pct(matched, registryTotal),
        ...(args.breakdown_rnb_id
          ? {
              rnb_id_breakdown: {
                with_rnb_id: withRnb,
                without_rnb_id: withoutRnbCount,
                pct_with_rnb_id: pct(withRnb, (withRnb ?? 0) + (withoutRnbCount ?? 0)),
              },
            }
          : {}),
        filters_applied: filters.map(([k, v]) => `${k}=${v}`),
      })
    } catch (err) {
      return errorContent(err)
    }
  },
})

const SourceBreakdownSchema = z.object({
  dpt: z.string().optional().describe('French département code, e.g. "33". Omit for all of France.'),
})

mcp.tool('get_source_breakdown', {
  description:
    'Break down detections by source code (0=DPVM pipeline, 1=FRPV reference, 2=OSM, 3=manual ' +
    'correction, 4=recall sample), for one département or all of France, with human-readable ' +
    'labels. A detection can carry several sources, so per-source counts overlap and do not add ' +
    'up to the total. Also returns how many detections are identified by 2+ sources (the ' +
    'meaning of quality_filter). ' + DATA_QUALITY_NOTE,
  inputSchema: SourceBreakdownSchema,
  handler: async (args: z.infer<typeof SourceBreakdownSchema>) => {
    try {
      const base: [string, string][] = args.dpt ? [['dpt', `eq.${args.dpt}`]] : []
      const codes = Object.keys(SOURCE_LABELS)
      const [total, multi, ...perSource] = await Promise.all([
        countDetections(base),
        countDetections([...base, ['sources', 'like.*,*']]),
        ...codes.map((c) => countDetections([...base, ['sources', `like.*${c}*`]])),
      ])
      return textContent({
        scope: args.dpt ? `département ${args.dpt}` : 'France (all départements)',
        total_detections: total,
        identified_by_2plus_sources: { count: multi, pct: pct(multi, total) },
        by_source: codes.map((c, i) => ({
          code: c,
          label: SOURCE_LABELS[c],
          count: perSource[i],
          pct_of_total: pct(perSource[i], total),
        })),
        note: 'Per-source counts overlap: a detection with sources "0,1" is counted under both 0 and 1.',
      })
    } catch (err) {
      return errorContent(err)
    }
  },
})

const DetectionDetailsSchema = z.object({
  id: z.number().int().describe('Registry identifier of the detection (the `id` field).'),
})

mcp.tool('get_detection_details', {
  description:
    'Get the full record of one detection by id: technical characteristics (surface, kwp, tilt, ' +
    'azimuth — model estimates), imagery years, sources decoded into plain labels, frpv_proba, ' +
    'false-positive flag, rnb_id and location codes, plus simple confidence signals. ' + DATA_QUALITY_NOTE,
  inputSchema: DetectionDetailsSchema,
  handler: async (args: z.infer<typeof DetectionDetailsSchema>) => {
    try {
      const rows = (await callRest('detections', [
        [
          'select',
          'id,surface,kwp,tilt,azimuth,first_seen,last_seen,n_vintages,sources,frpv_proba,' +
            'false_positive,false_positive_source,insee,dpt,rnb_id',
        ],
        ['id', `eq.${args.id}`],
        ['limit', '1'],
      ])) as Array<Record<string, unknown>>
      if (rows.length === 0) return textContent({ found: false, id: args.id })
      const r = rows[0]
      const decoded = decodeSources(r.sources)
      return textContent({
        found: true,
        ...r,
        sources_decoded: decoded,
        confidence_signals: {
          n_sources: decoded.length,
          identified_by_2plus_sources: decoded.length >= 2,
          detected_in_2plus_vintages: typeof r.n_vintages === 'number' && r.n_vintages >= 2,
          has_rnb_id: r.rnb_id !== null && r.rnb_id !== undefined,
          flagged_false_positive: r.false_positive === true,
        },
        note: 'kwp, surface, tilt, azimuth and frpv_proba are model estimates. A null false_positive is not a confirmation.',
      })
    } catch (err) {
      return errorContent(err)
    }
  },
})

const DepartmentProfileSchema = z.object({
  dpt: z.string().describe('French département code, e.g. "33".'),
})

mcp.tool('get_department_profile', {
  description:
    'One-call profile of a département: installed capacity and national rank, yearly trend ' +
    '(by first-seen imagery year), source mix, share identified by 2+ sources, and share matched ' +
    'to an RNB building id. Cross-references the pre-computed department statistics with live ' +
    'counts. ' + DATA_QUALITY_NOTE,
  inputSchema: DepartmentProfileSchema,
  handler: async (args: z.infer<typeof DepartmentProfileSchema>) => {
    try {
      const base: [string, string][] = [['dpt', `eq.${args.dpt}`]]
      const codes = Object.keys(SOURCE_LABELS)
      const [capacity, yearly, total, multi, withRnb, ...perSource] = await Promise.all([
        callRpc('dept_capacity_stats', {}) as Promise<Array<Record<string, unknown>>>,
        callRpc('dept_yearly_stats', {}) as Promise<Array<Record<string, unknown>>>,
        countDetections(base),
        countDetections([...base, ['sources', 'like.*,*']]),
        countDetections([...base, ['rnb_id', 'not.is.null']]),
        ...codes.map((c) => countDetections([...base, ['sources', `like.*${c}*`]])),
      ])
      return textContent({
        dpt: args.dpt,
        capacity_and_rank: capacity.find((r) => r.dpt === args.dpt) ?? null,
        yearly_stats: yearly.filter((r) => r.dpt === args.dpt),
        live_counts: {
          total_detections: total,
          identified_by_2plus_sources: { count: multi, pct: pct(multi, total) },
          with_rnb_id: { count: withRnb, pct: pct(withRnb, total) },
          by_source: codes.map((c, i) => ({
            code: c,
            label: SOURCE_LABELS[c],
            count: perSource[i],
            pct_of_total: pct(perSource[i], total),
          })),
        },
        note: 'Yearly figures reflect first-seen imagery year, not commissioning date. Per-source counts overlap.',
      })
    } catch (err) {
      return errorContent(err)
    }
  },
})

const FieldCoverageSchema = z.object({
  dpt: z.string().optional().describe('French département code. Omit for all of France.'),
})

mcp.tool('get_field_coverage', {
  description:
    'Completeness report: for each optional field of the registry (rnb_id, frpv_proba, ' +
    'false_positive, false_positive_source), how many detections have it filled in and what ' +
    'percentage. Answers "how many installations have an RNB id / an FRPV comparison / a ' +
    'false-positive flag?" in one call, for France or one département. For a single filtered count ' +
    'use count_detections instead. Null means "not available/recorded", never "0" or "confirmed ok". ' +
    DATA_QUALITY_NOTE,
  inputSchema: FieldCoverageSchema,
  handler: async (args: z.infer<typeof FieldCoverageSchema>) => {
    try {
      const base: [string, string][] = args.dpt ? [['dpt', `eq.${args.dpt}`]] : []
      const nn = (f: string): [string, string][] => [...base, [f, 'not.is.null']]
      const [total, rnb, frpv, fp, fpSrc, fpTrue] = await Promise.all([
        countDetections(base),
        countDetections(nn('rnb_id')),
        countDetections(nn('frpv_proba')),
        countDetections(nn('false_positive')),
        countDetections(nn('false_positive_source')),
        countDetections([...base, ['false_positive', 'eq.true']]),
      ])
      const row = (field: string, n: number | null, meaning: string) => ({
        field,
        filled: n,
        missing: n === null || total === null ? null : total - n,
        pct_filled: pct(n, total),
        meaning,
      })
      return textContent({
        scope: args.dpt ? `département ${args.dpt}` : 'France (all départements)',
        total_detections: total,
        fields: [
          row('rnb_id', rnb, 'Matched to a building in the national building repository (RNB).'),
          row('frpv_proba', frpv, 'An FRPV reference comparison was available (null = none, not probability 0).'),
          row('false_positive', fp, 'A false-positive status is recorded (null = nothing recorded, not "confirmed true").'),
          row('false_positive_source', fpSrc, 'The origin of the false-positive flag is recorded.'),
        ],
        flagged_false_positive: { count: fpTrue, pct: pct(fpTrue, total) },
      })
    } catch (err) {
      return errorContent(err)
    }
  },
})

const DEFAULT_BINS: Record<string, number[]> = {
  kwp: [0, 3, 6, 9, 36, 100, 250, 1000],
  surface: [0, 20, 40, 60, 100, 250, 1000],
  tilt: [0, 10, 20, 30, 40, 50, 90],
  azimuth: [0, 45, 90, 135, 180, 225, 270, 315, 360],
  n_vintages: [1, 2, 3, 4, 5],
  first_seen: [],
  last_seen: [],
}

const FieldDistributionSchema = z.object({
  field: z
    .enum(['kwp', 'surface', 'tilt', 'azimuth', 'n_vintages', 'first_seen', 'last_seen'])
    .describe('Numeric field to describe.'),
  dpt: z.string().optional().describe('French département code. Omit for all of France.'),
  bin_edges: z
    .array(z.number())
    .min(2)
    .max(25)
    .optional()
    .describe(
      'Custom ascending bin edges; each bin is [edge_i, edge_i+1) and the last is closed. ' +
        'Ignored for first_seen / last_seen, which are counted per year (use min_year / max_year).',
    ),
  min_year: z.number().int().optional().describe('For first_seen / last_seen: first year to count.'),
  max_year: z.number().int().optional().describe('For first_seen / last_seen: last year to count.'),
  quality_filter: z
    .boolean()
    .optional()
    .default(false)
    .describe('If true, restrict to detections identified by 2+ sources.'),
})

mcp.tool('get_field_distribution', {
  description:
    'Histogram of a numeric field (kwp, surface, tilt, azimuth, n_vintages, first_seen, ' +
    'last_seen) with exact counts per bin, for France or one département. Answers "how are ' +
    'installations distributed by size / tilt / orientation / year?". kwp, surface, tilt and ' +
    'azimuth are model estimates. first_seen/last_seen are imagery years, not installation dates. ' +
    DATA_QUALITY_NOTE,
  inputSchema: FieldDistributionSchema,
  handler: async (args: z.infer<typeof FieldDistributionSchema>) => {
    try {
      const base: [string, string][] = []
      if (args.dpt) base.push(['dpt', `eq.${args.dpt}`])
      if (args.quality_filter) base.push(['sources', 'like.*,*'])

      if (args.field === 'first_seen' || args.field === 'last_seen') {
        const from = args.min_year ?? 2010
        const to = args.max_year ?? new Date().getFullYear()
        if (to - from > 40) throw new Error('min_year/max_year span is limited to 40 years.')
        const years = Array.from({ length: to - from + 1 }, (_, i) => from + i)
        const counts = await Promise.all(years.map((y) => countDetections([...base, [args.field, `eq.${y}`]])))
        const total = await countDetections(base)
        return textContent({
          field: args.field,
          scope: args.dpt ? `département ${args.dpt}` : 'France',
          total_detections: total,
          per_year: years.map((y, i) => ({ year: y, count: counts[i], pct: pct(counts[i], total) })),
          note: 'Imagery years, not installation or commissioning dates.',
        })
      }

      const edges = args.bin_edges ?? DEFAULT_BINS[args.field]
      const bins = edges.slice(0, -1).map((lo, i) => ({ lo, hi: edges[i + 1], last: i === edges.length - 2 }))
      const [total, ...counts] = await Promise.all([
        countDetections(base),
        ...bins.map((b) =>
          countDetections([
            ...base,
            [args.field, `gte.${b.lo}`],
            [args.field, `${b.last ? 'lte' : 'lt'}.${b.hi}`],
          ]),
        ),
      ])
      const binned = counts.reduce<number>((s, c) => s + (c ?? 0), 0)
      return textContent({
        field: args.field,
        scope: args.dpt ? `département ${args.dpt}` : 'France',
        total_detections: total,
        bins: bins.map((b, i) => ({
          range: `${b.lo} to ${b.hi}${b.last ? ' (inclusive)' : ' (exclusive)'}`,
          count: counts[i],
          pct: pct(counts[i], total),
        })),
        outside_bins: total === null ? null : total - binned,
      })
    } catch (err) {
      return errorContent(err)
    }
  },
})

mcp.tool('get_field_reference', {
  description:
    'Field-by-field documentation of a detection record: type, unit, nullability, whether it is ' +
    'observed or a model estimate, how to read it, and which tool filters or reports it. Call this ' +
    'when the user asks what a field means or how reliable it is.',
  inputSchema: z.object({}),
  handler: () =>
    textContent({
      fields: {
        id: { type: 'integer', kind: 'structural', meaning: 'Registry identifier. Quote it when reporting an issue about a detection.', tools: ['get_detection_details'] },
        geom: { type: 'geometry (EPSG:4326)', kind: 'structural', meaning: 'Footprint of the detected PV array (not necessarily of the building). Polygon or multipolygon.', tools: ['get_detections_in_bbox'] },
        surface: { type: 'float, m²', kind: 'model estimate', meaning: 'Estimated area of the PV array.', filters: ['min_surface', 'max_surface'], tools: ['get_field_distribution'] },
        kwp: { type: 'float, kWp', kind: 'model estimate', meaning: 'Estimated installed DC capacity. Do not use as a precise per-site figure.', filters: ['min_kwp', 'max_kwp'], tools: ['aggregate_detection_capacity', 'get_field_distribution'] },
        tilt: { type: 'integer, degrees', kind: 'model estimate', meaning: 'Panel tilt from horizontal.', filters: ['min_tilt', 'max_tilt'], tools: ['get_field_distribution'] },
        azimuth: { type: 'integer, degrees', kind: 'model estimate', meaning: 'Orientation clockwise from true north (180 = south).', filters: ['min_azimuth', 'max_azimuth'], tools: ['get_field_distribution'] },
        first_seen: { type: 'integer, year', kind: 'observed (imagery)', meaning: 'Earliest imagery year in which it was detected. NOT an installation or commissioning date.', filters: ['min_first_seen', 'max_first_seen'], tools: ['get_department_yearly_stats', 'get_field_distribution'] },
        last_seen: { type: 'integer, year', kind: 'observed (imagery)', meaning: 'Most recent imagery year in which it was detected.', filters: ['min_last_seen', 'max_last_seen'], tools: ['get_field_distribution'] },
        n_vintages: { type: 'integer', kind: 'observed (imagery)', meaning: 'Number of distinct imagery years detecting it. 2+ = persistent, less likely a one-off artifact.', filters: ['min_vintages', 'max_vintages'], tools: ['get_field_distribution'] },
        sources: { type: 'comma-separated codes', kind: 'provenance', meaning: '0=DPVM pipeline, 1=FRPV, 2=OSM, 3=manual correction, 4=recall sample. Several codes = identified by several sources.', filters: ['quality_filter (2+ sources)', 'sources_include'], tools: ['get_source_breakdown', 'count_detections'] },
        frpv_proba: { type: 'float [0,1], nullable', kind: 'model estimate', meaning: 'Probability tied to an FRPV reference match. Null = no comparison available, not 0. Contract-recommended threshold: >= 0.1.', filters: ['min_frpv_proba', 'has_frpv_proba'], tools: ['get_field_coverage'] },
        false_positive: { type: 'boolean, nullable', kind: 'flag', meaning: 'Flagged as a false positive. Null = no status recorded, NOT a confirmed true positive.', filters: ['false_positive_status'], tools: ['get_field_coverage'] },
        false_positive_source: { type: 'string, nullable', kind: 'flag', meaning: 'Where the false-positive flag came from.', tools: ['get_field_coverage'] },
        insee: { type: 'string', kind: 'structural', meaning: 'INSEE commune code.', filters: ['insee'] },
        dpt: { type: 'string', kind: 'structural', meaning: 'Département code.', filters: ['dpt'] },
        rnb_id: { type: 'string, nullable', kind: 'structural', meaning: 'Matching building id in the national building repository (RNB), when available.', filters: ['has_rnb_id'], tools: ['count_detections', 'get_field_coverage', 'get_detection_details'] },
      },
      note: 'quality_filter (this server) = identified by 2 or more sources. It is separate from the contract-recommended frpv_proba >= 0.1 threshold, available as min_frpv_proba.',
    }),
})

const SourceReferenceSchema = z.object({
  source: z
    .string()
    .optional()
    .describe('A source name, alias or code (e.g. "frpv", "OpenStreetMap", "kasmi", "3"). Omit to list all.'),
})

mcp.tool('get_source_reference', {
  description:
    'Explain the data sources behind the registry\'s `sources` field: for each source, its code, ' +
    'short key, name, accepted aliases, and where the data comes from (with references: code, ' +
    'thesis, papers). Use it for questions like "what is FRPV?", "where does the OSM data come ' +
    'from?", "what is a manual correction?". ' + SOURCES_MEANING,
  inputSchema: SourceReferenceSchema,
  handler: (args: z.infer<typeof SourceReferenceSchema>) => {
    try {
      const list = args.source ? [resolveSource(args.source)] : SOURCE_CATALOG
      return textContent({ sources_meaning: SOURCES_MEANING, sources: list })
    } catch (err) {
      return errorContent(err)
    }
  },
})

mcp.tool('get_started', {
  description:
    'START HERE. Call this first when the user is new, asks what this server can do, or asks ' +
    'a vague question about DeepPVMapper. Returns a plain-language introduction, example ' +
    'questions, which tool answers which kind of question, a glossary and the main caveats. ' +
    'Reply to the user in their own language.',
  inputSchema: z.object({}),
  handler: () =>
    textContent({
      what_this_is:
        'DeepPVMapper / OpenPVMapper is an open registry of rooftop photovoltaic (solar panel) ' +
        'installations detected from aerial imagery across metropolitan France. Each detection ' +
        'has a location, an estimated surface and capacity (kWp), orientation, the imagery years ' +
        'it was seen in, and which sources identified it.',
      important_caveats: [
        'It is a detection dataset, not a complete inventory: estimated recall is ~0.6, so many real installations are missing. No detection does not mean no panels.',
        'surface, kwp, tilt and azimuth are model estimates, not measurements.',
        'first_seen / last_seen are imagery years, not installation or commissioning dates.',
        'Not suitable on its own for site-specific commercial, regulatory or financial decisions.',
      ],
      example_questions: [
        'What is the installed PV capacity in Gironde (33)?',
        'Which 10 départements have the most installed capacity?',
        'How has PV deployment grown in Hérault over the years?',
        'How many detections are confirmed by 2 or more sources?',
        'How many installations have an RNB building id? (and how complete is each field?)',
        'How are installations distributed by size, tilt or orientation?',
        'How many detections have been flagged as false positives?',
        'How many detections come from OpenStreetMap in the Nord?',
        'Show me the details of detection 12345.',
        'Give me a profile of département 31.',
        'What has the community flagged in this area that is not yet in the registry?',
      ],
      which_tool_for_what: {
        'Where do I start / what can you do?': 'get_started',
        'How much can I trust this data?': 'get_data_quality_reference',
        'Capacity or system count per département (pre-computed)': 'get_department_capacity_stats',
        'Growth over time per département': 'get_department_yearly_stats',
        'One-call overview of a département': 'get_department_profile',
        'How many ... (exact counts with filters)': 'count_detections',
        'Which sources identified the data': 'get_source_breakdown',
        'What is FRPV / OSM / DPVM / a manual correction, and where does it come from?': 'get_source_reference',
        'How complete is each optional field (rnb_id, frpv_proba, false_positive)': 'get_field_coverage',
        'Distribution by size / tilt / azimuth / year / vintages': 'get_field_distribution',
        'What does this field mean?': 'get_field_reference',
        'Total kWp for a filtered subset': 'aggregate_detection_capacity',
        'List individual detections': 'search_detections',
        'Everything about one detection': 'get_detection_details',
        'Detections in a map area': 'get_detections_in_bbox',
        'Pending, unverified community edits': 'get_community_activity / get_community_activity_in_area',
      },
      glossary: {
        kWp: 'Kilowatt-peak: nominal DC capacity of a PV installation.',
        'département / dpt': 'French administrative area, identified by a code such as "33" (Gironde) or "2A".',
        INSEE: 'Official code identifying a French commune.',
        'imagery vintage': 'A year of aerial imagery. n_vintages = number of distinct years the installation was detected in.',
        sources: SOURCES_MEANING + ' Names: DeepPVMapper/DPVM (0), FRPV/Thébault et al. (1), OpenStreetMap/OSM (2), manual correction (3), recall sample (4). See get_source_reference for where each comes from.',
        'quality_filter': 'In this server: keep only detections identified by 2 or more sources.',
        frpv_proba: 'Model probability tied to a match with the FRPV reference dataset; null means no comparison was available (not 0).',
        rnb_id: 'Identifier of the matching building in the French national building repository (RNB), when available.',
      },
      license_and_attribution:
        'Data: CC BY 4.0, commercial use allowed with attribution "DeepPVMapper — Gabriel Kasmi" (https://deeppvmapper.fr). ' +
        'Please do not use the data to identify or target individuals or households. ' +
        'For country-scale processing use the bulk releases (Zenodo, Hugging Face) rather than this live API.',
      links: {
        website: 'https://deeppvmapper.fr',
        zenodo: 'https://zenodo.org/records/19188878',
        huggingface: 'https://huggingface.co/datasets/gabrielkasmi/openpvmapper',
        discussions: 'https://github.com/gabrielkasmi/deeppvmapper/discussions/19',
      },
    }),
})

// --- HTTP wiring (Supabase Edge Functions pattern) -------------------------
//
// Supabase routes a request to this function at /functions/v1/<function-name>/*.
// The function is named "mcp" (see supabase/config.toml), so the final MCP
// endpoint is /functions/v1/mcp/mcp (full URL: https://<project-ref>.supabase.co/functions/v1/mcp/mcp).

const transport = new StreamableHttpTransport()
export const httpHandler = transport.bind(mcp)

const app = new Hono()
const mcpApp = new Hono()

mcpApp.get('/', (c) =>
  c.json({
    message: 'DeepPVMapper MCP server',
    endpoints: { mcp: '/mcp' },
    docs: 'https://deeppvmapper.fr',
  }),
)

mcpApp.all('/mcp', async (c) => {
  return await httpHandler(c.req.raw)
})

// Mount prefix must match the function name for Supabase's routing.
app.route('/mcp', mcpApp)

// Only start the HTTP server under Deno. When this module is imported by stdio.ts (Node), the
// server is driven over stdin/stdout instead.
if (runtime.Deno?.serve) runtime.Deno.serve(app.fetch)
