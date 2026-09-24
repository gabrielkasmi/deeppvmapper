// DeepPVMapper MCP server
//
// Exposes the public OpenPVMapper PV-detection registry (see DATA_ACCESS.md
// at the repo root) as MCP tools, so an LLM/agent can query real French
// rooftop-solar detection data instead of guessing.
//
// See DEPLOY.md in this folder for local testing and deployment steps.

import { McpServer, StreamableHttpTransport } from 'mcp-lite'
import { z } from 'zod'
import { Hono } from 'hono'

// --- Config -------------------------------------------------------------
//
// Defaults match the public data contract in DATA_ACCESS.md. The key is a
// Supabase "publishable" key: it is meant to be public and only grants
// access to the read-only public resources described in that contract.
// Override with `supabase secrets set` if the project or key ever changes.

const API_URL = Deno.env.get('DEEPPVMAPPER_API_URL') ?? 'https://zelhliylrlktnasircwp.supabase.co'
const API_KEY =
  Deno.env.get('DEEPPVMAPPER_API_KEY') ?? 'sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi'

const DATA_QUALITY_NOTE =
  'Data quality note: this is a detection dataset, not an exhaustive inventory ' +
  '(estimated recall ~0.6). A missing detection does not mean no PV installation ' +
  'exists there. kwp, surface, tilt and azimuth are model estimates, not surveyed values. ' +
  'Call get_data_quality_reference for the full picture before assessing fitness for a specific use case.'

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

const mcp = new McpServer({
  name: 'deeppvmapper',
  version: '0.1.0',
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
      confidence_signals: {
        cross_validated:
          'sources field lists 2+ distinct source codes (see source_encoding) — detection ' +
          'confirmed by independent methods, not just the automated pipeline alone.',
        persistent:
          'n_vintages >= 2 — detected independently across multiple years of imagery, reducing ' +
          'the chance of a one-off artifact.',
      },
      source_encoding: {
        '0': 'DPVM — DeepPVMapper aerial-imagery detection pipeline',
        '1': 'FRPV — match against the FRPV reference dataset',
        '2': 'OSM — OpenStreetMap contribution',
        '3': 'Manual correction — submitted through map annotation tools',
        '4': 'Recall sample — recovered during a targeted recall-annotation pass',
      },
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
    .describe(
      'If true, only include detections confirmed by at least two independent sources (e.g. the ' +
        'automated DeepPVMapper pipeline plus OpenStreetMap or the FRPV reference dataset), not just ' +
        'a single pipeline. This is a stronger confidence signal than min_vintages.',
    ),
  quality_filter: z
    .boolean()
    .optional()
    .default(true)
    .describe(
      'If true (default), only include detections with frpv_proba >= 0.1, the threshold ' +
        'recommended in the data contract for a good precision/recall trade-off.',
    ),
}

function buildDetectionFilterParams(args: {
  dpt?: string
  insee?: string
  min_kwp?: number
  max_kwp?: number
  min_vintages?: number
  cross_validated?: boolean
  quality_filter?: boolean
}): [string, string][] {
  const params: [string, string][] = []
  if (args.dpt) params.push(['dpt', `eq.${args.dpt}`])
  if (args.insee) params.push(['insee', `eq.${args.insee}`])
  if (args.min_kwp !== undefined) params.push(['kwp', `gte.${args.min_kwp}`])
  if (args.max_kwp !== undefined) params.push(['kwp', `lte.${args.max_kwp}`])
  if (args.min_vintages !== undefined) params.push(['n_vintages', `gte.${args.min_vintages}`])
  // sources is a comma-separated list of source codes (see DATA_ACCESS.md §11), e.g. "0,1".
  // A comma in the string means at least two distinct sources are listed.
  if (args.cross_validated) params.push(['sources', 'like.*,*'])
  if (args.quality_filter) params.push(['frpv_proba', 'gte.0.1'])
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
          'id,surface,kwp,tilt,azimuth,first_seen,last_seen,n_vintages,sources,frpv_proba,dpt,insee',
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

// --- HTTP wiring (Supabase Edge Functions pattern) -------------------------
//
// Supabase routes a request to this function at /functions/v1/<function-name>/*.
// The function is named "mcp" (see supabase/config.toml), so the final MCP
// endpoint is /functions/v1/mcp/mcp — see DEPLOY.md for the full URL.

const transport = new StreamableHttpTransport()
const httpHandler = transport.bind(mcp)

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

Deno.serve(app.fetch)
