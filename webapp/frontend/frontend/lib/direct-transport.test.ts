import { afterEach, describe, it, expect, vi } from 'vitest'
import { readSSEStream } from './sse-stream'

/**
 * Fake ReadableStreamDefaultReader that mirrors the REAL browser semantics the
 * watchdog depends on: calling cancel() rejects an outstanding/pending read()
 * (and makes subsequent reads resolve {done:true}), so the read loop unwinds.
 */
function makeReader(
  onRead: (i: number) => Promise<{ done: boolean; value?: Uint8Array }> | undefined,
): ReadableStreamDefaultReader<Uint8Array> {
  let i = 0
  let pending: (() => void) | null = null
  let cancelled = false
  return {
    read: () => {
      if (cancelled) return Promise.resolve({ done: true })
      const next = onRead(i++)
      if (next) return next
      // A never-resolving source: park the read so cancel() can reject it, exactly
      // like the browser stream's pending read is errored by cancel().
      return new Promise<{ done: boolean; value?: Uint8Array }>((_resolve, reject) => {
        pending = () => reject(new Error('cancelled'))
      })
    },
    cancel: vi.fn(() => {
      cancelled = true
      pending?.()
      pending = null
      return Promise.resolve(undefined)
    }),
    releaseLock: vi.fn(),
  } as unknown as ReadableStreamDefaultReader<Uint8Array>
}

const enc = (s: string): Uint8Array => new TextEncoder().encode(s)

function collect(events: Array<[string, string]>) {
  return (event: string, data: string) => { events.push([event, data]) }
}

afterEach(() => {
  vi.useRealTimers()
})

describe('readSSEStream', () => {
  it('parses normal SSE events and resolves on EOF', async () => {
    const chunks = [
      enc('event: accepted\ndata: {"endpoint":"t2v"}\n\n'),
      enc('event: progress\ndata: {"step":1}\n\nevent: progress\ndata: {"step":2}\n\n'),
      enc('event: complete\ndata: {"ok":true}\n\n'),
    ]
    const events: Array<[string, string]> = []
    const reader = makeReader((i) =>
      i < chunks.length
        ? Promise.resolve({ done: false, value: chunks[i] })
        : Promise.resolve({ done: true }),
    )
    await readSSEStream(reader, undefined, collect(events), { idleTimeoutMs: 20000, maxDurationMs: 20000 })
    expect(events.map(([e]) => e)).toEqual(['accepted', 'progress', 'progress', 'complete'])
    expect(events.some(([e]) => e === 'error')).toBe(false)
  })

  it('fires a synthetic error when the read never resolves (silent/hung connection)', async () => {
    vi.useFakeTimers()
    const events: Array<[string, string]> = []
    const reader = makeReader((i) => {
      if (i === 0) return Promise.resolve({ done: false, value: enc('event: accepted\ndata: {}\n\n') })
      return undefined // the second read hangs forever — the true hang case
    })
    const done = readSSEStream(reader, undefined, collect(events), {
      idleTimeoutMs: 25,
      maxDurationMs: 5000,
    })
    // Let the first read resolve, then advance the fake clock to fire the idle watchdog.
    await Promise.resolve()
    await Promise.resolve()
    vi.advanceTimersByTime(30)
    await done
    expect(reader.cancel).toHaveBeenCalled()
    const err = events.find(([e]) => e === 'error')
    expect(err).toBeDefined()
    expect(String(err![1])).toMatch(/stalled/i)
  })

  it('does NOT fire a watchdog error when reads keep arriving', async () => {
    vi.useFakeTimers()
    const events: Array<[string, string]> = []
    const reader = makeReader((i) => {
      if (i >= 8) return Promise.resolve({ done: true })
      return Promise.resolve({ done: false, value: enc(': keepalive\n\n') })
    })
    const done = readSSEStream(reader, undefined, collect(events), {
      idleTimeoutMs: 1000,
      maxDurationMs: 200000,
    })
    // Drive several keeps-alive reads, advancing the clock past the idle window
    // between them — bytes keep flowing so the idle watchdog must NOT fire.
    for (let t = 0; t < 8; t++) {
      await Promise.resolve()
      vi.advanceTimersByTime(200)
    }
    await done
    expect(events.some(([e]) => e === 'error')).toBe(false)
  })

  it('fires a synthetic error when the total stream exceeds maxDurationMs', async () => {
    vi.useFakeTimers()
    const events: Array<[string, string]> = []
    // Worker heartbeats (bytes keep flowing) but never sends a terminal complete/
    // error — a hung-but-alive generation. Idle never trips; max duration must fire.
    const reader = makeReader(() =>
      Promise.resolve({ done: false, value: enc(': keepalive\n\n') }),
    )
    const done = readSSEStream(reader, undefined, collect(events), {
      idleTimeoutMs: 100000, // idle never trips
      maxDurationMs: 30,
    })
    // Advance past the max-duration cap while keepalive bytes keep arriving.
    await Promise.resolve()
    vi.advanceTimersByTime(50)
    await done
    expect(reader.cancel).toHaveBeenCalled()
    const err = events.find(([e]) => e === 'error')
    expect(err).toBeDefined()
    expect(String(err![1])).toMatch(/timed out/i)
  })
})
