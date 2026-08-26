// OpenRouter direct-from-the-browser chat client.
//
// The browser calls OpenRouter DIRECTLY (mirroring FAL) so the Livepeer topology
// orchestrator/runner never sees the user's OpenRouter key. The raw key is fetched
// from GET /api/settings/openrouter-key (owner-only) and held client-side only.
//
// Auto model resolution uses NO hardcoded model constant and no env var: provider ids
// and free tiers rotate, so the default model is derived LIVE from the current
// OpenRouter /models list — the most popular free+vision model, where "popular" is
// proxied by OpenRouter's own returned ordering (they expose no popularity field).

export const OPENROUTER_BASE = 'https://openrouter.ai/api/v1'

export class OpenRouterError extends Error {
  constructor(public status: number, message: string) {
    super(message)
  }
}

export interface OpenRouterModel {
  id: string
  name: string
  vision: boolean
  free: boolean
}

interface RawModel {
  id: string
  name?: string
  architecture?: { input_modalities?: string[]; output_modalities?: string[] }
  pricing?: { prompt?: string | number }
}

const toNum = (v: string | number | undefined): number =>
  typeof v === 'number' ? v : (typeof v === 'string' ? parseFloat(v) || 0 : 0)

/** Auth'd fetch of the OpenRouter model list, filtered to vision-capable, free-first. */
export async function fetchVisionModels(apiKey: string): Promise<OpenRouterModel[]> {
  const res = await fetch(`${OPENROUTER_BASE}/models`, {
    headers: { authorization: `Bearer ${apiKey}` },
  })
  if (!res.ok) throw new OpenRouterError(res.status, `OpenRouter models failed (${res.status})`)
  const json: { data?: RawModel[] } = await res.json()
  const out = (json.data ?? []).map((m) => ({
    id: m.id,
    name: m.name ?? m.id,
    vision: (m.architecture?.input_modalities ?? []).map((s) => s.toLowerCase()).includes('image'),
    free: toNum(m.pricing?.prompt) === 0,
  }))
  return out.filter((m) => m.vision) // PRESERVE OpenRouter's returned order = popularity proxy
}

// Freshness cache: the auto path re-confirms availability without a network call per
// request, but never trusts a stale list. Expired/absent cache -> refetch.
let modelsCache: { at: number; list: OpenRouterModel[] } | null = null
const MODELS_TTL_MS = 60_000

/** Fresh (<= TTL) vision+free list; refetches on expiry. Call at dispatch time so auto
 *  model resolution CONFIRMS availability against a current list, not the saved one. */
export async function getVisionModels(apiKey: string): Promise<OpenRouterModel[]> {
  if (modelsCache && Date.now() - modelsCache.at < MODELS_TTL_MS) return modelsCache.list
  const list = await fetchVisionModels(apiKey)
  modelsCache = { at: Date.now(), list }
  return list
}

/**
 * Resolve the model for one request. NO hardcoded model constant — provider IDs / free
 * tiers rotate, so this is derived entirely LIVE.
 *  - user has set a specific model  -> that exact model (never probed or overridden).
 *  - else (auto): the MOST POPULAR free vision model — the first free+vision entry in
 *    OpenRouter's returned order (their ordering is the popularity proxy); falls back to the
 *    first vision model if no free tier is currently listed.
 *  - no vision model at all -> throws; the caller degrades to the local-Gemma runner path.
 */
export function resolveOpenRouterModel(
  selected: string | undefined,
  models: OpenRouterModel[] | null,
): string {
  if (selected?.trim()) return selected.trim()
  const freeVision = (models ?? []).filter((m) => m.free && m.vision) // OpenRouter order kept
  if (freeVision.length) return freeVision[0].id
  const anyVision = (models ?? []).filter((m) => m.vision)
  if (anyVision.length) return anyVision[0].id
  throw new OpenRouterError(0, 'No vision model available on OpenRouter — select one in Settings')
}

export interface ChatOpts {
  maxTokens?: number
  temperature?: number
}

type ChatMessage = { role: string; content: unknown }

export async function openRouterChat(
  apiKey: string,
  model: string,
  messages: ChatMessage[],
  opts: ChatOpts = {},
): Promise<{ content: string; reasoning?: string }> {
  const body: Record<string, unknown> = {
    model,
    messages,
    max_tokens: opts.maxTokens ?? 2048,
    temperature: opts.temperature ?? 0.7,
  }
  const res = await fetch(`${OPENROUTER_BASE}/chat/completions`, {
    method: 'POST',
    headers: {
      authorization: `Bearer ${apiKey}`,
      'content-type': 'application/json',
      'x-title': 'video-creator',
    },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new OpenRouterError(res.status, `OpenRouter chat failed (${res.status})`)
  const json = (await res.json()) as {
    choices?: Array<{ message?: { content?: unknown; reasoning_content?: unknown } }>
  }
  const msg = json.choices?.[0]?.message
  return {
    content: typeof msg?.content === 'string' ? msg.content : '',
    reasoning: typeof msg?.reasoning_content === 'string' ? msg.reasoning_content : undefined,
  }
}
