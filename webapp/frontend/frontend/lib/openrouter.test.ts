import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  fetchVisionModels,
  getVisionModels,
  resolveOpenRouterModel,
  openRouterChat,
  OpenRouterError,
  OPENROUTER_BASE,
} from './openrouter'

function jsonResponse(body: unknown, ok = true, status = 200) {
  return { ok, status, json: async () => body } as unknown as Response
}

describe('fetchVisionModels', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('sends Authorization: Bearer and returns only vision models, flagging free', async () => {
    const seen: any[] = []
    vi.stubGlobal('fetch', vi.fn(async (_url: string, init?: any) => {
      seen.push(init)
      return jsonResponse({
        data: [
          { id: 'a/img-free', pricing: { prompt: '0' }, architecture: { input_modalities: ['text', 'image'] } },
          { id: 'b/img-paid', pricing: { prompt: 1.2 }, architecture: { input_modalities: ['image'] } },
          { id: 'c/txt-only', pricing: { prompt: '0' }, architecture: { input_modalities: ['text'] } },
        ],
      })
    }))
    const models = await fetchVisionModels('k')
    expect(seen[0].headers.authorization).toBe('Bearer k')
    expect(models).toHaveLength(2)
    expect(models[0]).toMatchObject({ id: 'a/img-free', free: true, vision: true })
    expect(models[1]).toMatchObject({ id: 'b/img-paid', free: false, vision: true })
  })

  it('throws OpenRouterError on non-2xx', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse({}, false, 401)))
    await expect(fetchVisionModels('k')).rejects.toThrow(OpenRouterError)
  })
})

describe('resolveOpenRouterModel (NO hardcoded constant; derives live)', () => {
  const models = [
    { id: 'paid/img', name: 'P', vision: true, free: false },
    { id: 'free/img', name: 'F', vision: true, free: true },
    { id: 'free2/img', name: 'F2', vision: true, free: true },
  ]

  it('auto picks the first free+vision in OpenRouter order (popularity proxy)', () => {
    expect(resolveOpenRouterModel(undefined, models)).toBe('free/img')
  })

  it('falls back to the first (paid) vision model when no free tier is listed', () => {
    const paidOnly = models.map((m) => ({ ...m, free: false }))
    expect(resolveOpenRouterModel(undefined, paidOnly)).toBe('paid/img')
  })

  it('throws when there is no vision model at all', () => {
    expect(() => resolveOpenRouterModel(undefined, [{ id: 'x', name: 'x', vision: false, free: true }]))
      .toThrow(OpenRouterError)
  })

  it('a user-set model wins verbatim, never probed against the list', () => {
    expect(resolveOpenRouterModel('anthropic/claude-sonnet', models)).toBe('anthropic/claude-sonnet')
  })
})

describe('getVisionModels cache', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('returns the cached list within the TTL and refetches when expired', async () => {
    const fetch = vi.fn(async () => jsonResponse({ data: [{ id: 'a/img', pricing: { prompt: '0' }, architecture: { input_modalities: ['image'] } }] }))
    vi.stubGlobal('fetch', fetch)
    await getVisionModels('k')
    await getVisionModels('k')
    expect(fetch).toHaveBeenCalledTimes(1) // second call served from cache
    await vi.advanceTimersByTimeAsync(60_001)
    await getVisionModels('k')
    expect(fetch).toHaveBeenCalledTimes(2) // expired -> refetch
    vi.setSystemTime(Date.now())
  })
})

describe('openRouterChat', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('POSTs model/messages/max_tokens/temperature and parses content + reasoning', async () => {
    let body: any
    vi.stubGlobal('fetch', vi.fn(async (_u: string, init?: any) => {
      body = JSON.parse(init.body)
      return jsonResponse({
        choices: [{ message: { content: 'enhanced!', reasoning_content: 'thinking' } }],
      })
    }))
    const res = await openRouterChat('k', 'm/img', [{ role: 'user', content: 'hi' }], { maxTokens: 99, temperature: 0.3 })
    expect(body).toMatchObject({ model: 'm/img', max_tokens: 99, temperature: 0.3 })
    expect(body.messages).toHaveLength(1)
    expect(res).toEqual({ content: 'enhanced!', reasoning: 'thinking' })
  })

  it('POSTs to the OpenRouter chat endpoint and throws on failure', async () => {
    const fetch = vi.fn(async (_url: string) => jsonResponse({}, false, 429))
    vi.stubGlobal('fetch', fetch)
    await expect(openRouterChat('k', 'm/img', [])).rejects.toThrow(OpenRouterError)
    expect(fetch.mock.calls[0][0]).toBe(`${OPENROUTER_BASE}/chat/completions`)
  })
})
