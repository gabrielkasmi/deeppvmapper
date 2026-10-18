// stdio entry point for the DeepPVMapper MCP server (used by the Glama build).
//
// The server itself (index.ts) speaks Streamable HTTP, which is what Supabase Edge Functions
// serve. Glama and most MCP launchers expect a process that speaks JSON-RPC over stdin/stdout,
// so this file feeds each incoming JSON-RPC line to the same request handler and writes the
// responses back out. No logic lives here: tools, filters and data access are all in index.ts.
//
// Run with Node:   npx tsx supabase/functions/mcp/stdio.ts
// (needs: tsx, hono, mcp-lite, zod)

import { createInterface } from 'node:readline'
import { httpHandler } from './index.ts'

let sessionId: string | undefined
const inflight = new Set<Promise<void>>()

function emit(message: string) {
  // One JSON-RPC message per line on stdout. Re-serialise to guarantee a single line.
  process.stdout.write(JSON.stringify(JSON.parse(message)) + '\n')
}

function jsonRpcId(line: string): string | number | null {
  try {
    const parsed = JSON.parse(line)
    return parsed && (typeof parsed.id === 'string' || typeof parsed.id === 'number') ? parsed.id : null
  } catch {
    return null
  }
}

async function handle(line: string) {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: 'application/json, text/event-stream',
  }
  if (sessionId) headers['mcp-session-id'] = sessionId

  const res = await httpHandler(new Request('http://localhost/mcp', { method: 'POST', headers, body: line }))

  const sid = res.headers.get('mcp-session-id')
  if (sid) sessionId = sid

  // Notifications are acknowledged with no body.
  if (res.status === 202 || res.status === 204) return

  const body = await res.text()
  if ((res.headers.get('content-type') ?? '').includes('text/event-stream')) {
    for (const row of body.split('\n')) {
      if (row.startsWith('data:')) {
        const data = row.slice(5).trim()
        if (data) emit(data)
      }
    }
  } else if (body.trim()) {
    emit(body.trim())
  }
}

const rl = createInterface({ input: process.stdin })

rl.on('line', (raw) => {
  const line = raw.trim()
  if (!line) return
  const task: Promise<void> = handle(line)
    .catch((err) => {
      const id = jsonRpcId(line)
      if (id !== null) {
        emit(
          JSON.stringify({
            jsonrpc: '2.0',
            id,
            error: { code: -32603, message: err instanceof Error ? err.message : String(err) },
          }),
        )
      }
      console.error('deeppvmapper stdio error:', err)
    })
    .finally(() => {
      inflight.delete(task)
    })
  inflight.add(task)
})

rl.on('close', async () => {
  await Promise.allSettled([...inflight])
  process.exit(0)
})

console.error('DeepPVMapper MCP server (stdio) ready')
