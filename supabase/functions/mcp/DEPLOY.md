# Deploying the DeepPVMapper MCP server

This function is a thin wrapper: it exposes the public DeepPVMapper/OpenPVMapper
API (documented in `DATA_ACCESS.md` at the repo root) as MCP tools, using
[mcp-lite](https://www.npmjs.com/package/mcp-lite) on Supabase Edge Functions.

No global install is needed — everything below runs via `npx supabase`,
already verified to work on this machine (CLI 2.117.0).

## 0. Project ref

The current project is `zelhliylrlktnasircwp` (used throughout `DATA_ACCESS.md`).
`NOTES.md` references an older ref (`nvtjkzxoothrilrnlkym`) from before a
project change — that note is stale, ignore it.

## 1. Log in and link this repo to the project

```bash
npx supabase login
npx supabase link --project-ref zelhliylrlktnasircwp
```

`login` opens a browser to authenticate — do this from a terminal on your
own machine, not headlessly.

## 2. Test locally (optional but recommended)

```bash
npx supabase start
npx supabase functions serve --no-verify-jwt mcp
```

In another terminal:

```bash
npx @modelcontextprotocol/inspector
# then connect to http://localhost:54321/functions/v1/mcp/mcp
```

Or, if you use Claude Code locally:

```bash
claude mcp add deeppvmapper -t http http://localhost:54321/functions/v1/mcp/mcp
```

Try asking it something like "what's the installed PV capacity in Gironde
(dept 33)?" and check it calls `get_department_capacity_stats`.

## 3. Set the API credentials as secrets (optional)

The function has working defaults baked in (the same public URL and
publishable key already in `DATA_ACCESS.md`), so this step is optional.
Only do it if you want to point the function at a different project or key
without editing code:

```bash
npx supabase secrets set DEEPPVMAPPER_API_URL=https://zelhliylrlktnasircwp.supabase.co
npx supabase secrets set DEEPPVMAPPER_API_KEY=sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi
```

## 4. Deploy

```bash
npx supabase functions deploy --no-verify-jwt mcp
```

Your server is now live at:

```text
https://zelhliylrlktnasircwp.supabase.co/functions/v1/mcp/mcp
```

## 5. Custom domain (mcp.deeppvmapper.fr)

In the Supabase dashboard: **Project Settings > Custom Domains** (this
requires the Pro plan, which this project already has). Add
`mcp.deeppvmapper.fr` and follow the DNS instructions Supabase gives you —
typically a CNAME plus a TXT record for verification, added wherever
deeppvmapper.fr's DNS is managed today.

Once verified, the same endpoint is reachable at:

```text
https://mcp.deeppvmapper.fr/functions/v1/mcp/mcp
```

(the `/functions/v1/mcp/mcp` path stays the same — only the hostname changes.)

## 6. Connect a real MCP client

```bash
claude mcp add deeppvmapper -t http https://mcp.deeppvmapper.fr/functions/v1/mcp/mcp
```

Or add it as a custom remote connector in Claude Desktop / claude.ai with
that same URL.

## Troubleshooting

- If `supabase functions deploy` complains it can't find `deno.json` /
  the import map, move `supabase/functions/mcp/deno.json` to
  `supabase/functions/deno.json` (shared across functions) and retry.
- If tool calls return `DeepPVMapper API error 401`, the publishable key in
  `DATA_ACCESS.md` may have been rotated — update the secret from step 3.
- This function is read-only and stateless: it only calls the existing
  public REST/RPC endpoints, it never touches `annotations` or other
  non-public tables.
