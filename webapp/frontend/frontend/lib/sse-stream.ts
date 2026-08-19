/**
 * Pure SSE stream reader with a stall/max-duration watchdog. Deliberately dependency-free
 * (no electron/browser imports) so it is unit-testable in a plain Node vitest env — the same
 * pattern as media-decode.ts. direct-transport.ts builds on this for the Livepeer SSE rail.
 */

// How long a generation SSE stream may go with ZERO bytes before we call it dead.
// The runner emits a `: keepalive` comment every ~10s (and proxy_worker_sse relays the
// same for worker-streamed endpoints like extend/layer/edit), so any idle longer than
// this means the connection/pipe was dropped silently — surface an error instead of
// spinning on "generating" forever.
export const SSE_IDLE_TIMEOUT_MS = 90_000
// Total wall-clock cap for a generation SSE stream. A hung-but-alive worker (one that
// keeps heartbeating but never sends a terminal complete/error — e.g. an OOM/looped
// inference) would otherwise keep the connection "healthy" forever. This cap guarantees
// a generation ALWAYS settles; set well above the longest realistic job (cold model load
// + long denoise) so it never cuts off a genuine generation.
export const SSE_MAX_DURATION_MS = 25 * 60_000

/** Optional watchdog overrides (ms) — used by tests to shrink the windows; callers
 * leave them unset so the production constants apply. */
export interface SseWatchdogOptions {
  idleTimeoutMs?: number
  maxDurationMs?: number
}

/**
 * Drive `reader` and deliver parsed SSE events to onEvent.
 * Splits the byte stream on blank lines; a block's `event:` field names the event
 * (default "message") and its `data:` lines are joined with newlines per the SSE spec.
 *
 * WATCHDOG: guarantees the stream can't hang forever even if the peer stays connected
 * but silent. (a) If no bytes at all arrive for idleTimeoutMs the reader is cancelled
 * and a synthetic `error` event is dispatched. (b) If the whole stream runs longer than
 * maxDurationMs the same happens. This lets every consumer's error branch settle the
 * promise, so a failed/stalled generation always clears the UI's "generating" state
 * instead of wedging it.
 */
export async function readSSEStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  signal: AbortSignal | undefined,
  onEvent: (event: string, data: string) => void,
  watchdog: SseWatchdogOptions = {},
): Promise<void> {
  const decoder = new TextDecoder()
  const idleTimeoutMs = watchdog.idleTimeoutMs ?? SSE_IDLE_TIMEOUT_MS
  const maxDurationMs = watchdog.maxDurationMs ?? SSE_MAX_DURATION_MS
  let buf = ''
  let idleTimer: ReturnType<typeof setTimeout> | null = null
  let maxTimer: ReturnType<typeof setTimeout> | null = null
  let watchdogFired = false

  const failWatchdog = (message: string) => {
    if (watchdogFired) return
    watchdogFired = true
    clearTimeout(idleTimer ?? undefined)
    clearTimeout(maxTimer ?? undefined)
    onEvent('error', JSON.stringify({ error: message }))
    void reader.cancel().catch(() => {})
  }
  const armIdle = () => {
    if (idleTimer) clearTimeout(idleTimer)
    idleTimer = setTimeout(() => failWatchdog(
      `Generation stalled: no data received for ${Math.round(idleTimeoutMs / 1000)}s. ` +
      'The runner connection went silent; the generation did not complete.',
    ), idleTimeoutMs)
  }

  const onAbort = () => { void reader.cancel().catch(() => {}) }
  if (signal) {
    if (signal.aborted) onAbort()
    else signal.addEventListener('abort', onAbort, { once: true })
  }

  try {
    maxTimer = setTimeout(() => failWatchdog(
      `Generation timed out after ${Math.round(maxDurationMs / 60000)} minutes without completing.`,
    ), maxDurationMs)
    for (;;) {
      // Arm the idle timer before the read so the window also covers a read that
      // never resolves (this is the true hang case).
      armIdle()
      let chunk: { done: boolean; value?: Uint8Array }
      try {
        chunk = await reader.read()
      } catch {
        // A watchdog-driven reader.cancel() rejects the in-flight read with an
        // abort Error. The watchdog already delivered its synthetic `error` event
        // (that event is the source of truth to the consumer), so a cancel-induced
        // read rejection just means "unwind" — don't propagate it.
        break
      }
      clearTimeout(idleTimer ?? undefined)
      idleTimer = null
      if (chunk.done) break
      const { value } = chunk
      buf += decoder.decode(value, { stream: true })
      let idx: number
      while ((idx = buf.indexOf('\n\n')) !== -1) {
        const blockText = buf.slice(0, idx)
        buf = buf.slice(idx + 2)
        let event = 'message'
        const datas: string[] = []
        for (const line of blockText.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim()
          else if (line.startsWith('data:')) datas.push(line.slice(5).replace(/^\s/, ''))
        }
        if (datas.length) onEvent(event, datas.join('\n'))
      }
    }
  } finally {
    clearTimeout(idleTimer ?? undefined)
    clearTimeout(maxTimer ?? undefined)
    if (signal) signal.removeEventListener('abort', onAbort)
  }
}
