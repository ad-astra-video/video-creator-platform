/**
 * Direct transport (plans/20260811_direct_transport.md, PHASE_B_BROWSER_TICKET_FLOW.md).
 *
 * THE ONLY place the app knows how to reach a GPU runner / FAL for generation. Under the
 * locked design the D1 Worker is NOT in the media/inference path: the browser talks to the
 * runner (Livepeer) or to fal.run (FAL) DIRECTLY, and the Worker's only involvement in a
 * generation is minting the PymtHouse payment ticket (POST /sign-ticket) + auth/balance.
 *
 * Responsibilities:
 *  - resolve a capable runner from /api/providers (Worker discovery)
 *  - perform the Livepeer payment handshake: send with Livepeer-Payer-Address -> 402 ->
 *    POST /sign-ticket -> retry with Livepeer-Payment + Livepeer-Segment headers
 *  - call FAL (fal.run) directly for T2I and I2I image generation given the raw key
 *  - return the media response as a Blob for the browser-local asset store
 */

import { ApiClient } from './api-client'
import { getBlob, getBlobUrl } from './runtime/web-store'
import { logger } from './logger'

/** Fetch with an AbortSignal.timeout guard so a hung hop surfaces as an error instead of a silent await. */
function fetchWithTimeout(ms: number): RequestInit['signal'] {
  return typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function'
    ? AbortSignal.timeout(ms)
    : undefined
}

// Task -> runner endpoint (must match the live-runner /video-creator/v1 route table that the
// runner service exposes. See orchestrator.ts TASK_ENDPOINTS — we mirror the same shape here
// so the browser can POST to the runner directly).
const TASK_ENDPOINTS: Record<string, string> = {
  generate: '/video-creator/v1/t2v',
  'generate-image': '/video-creator/v1/image',
  'enhance-prompt': '/video-creator/v1/prompt-enhance',
  extend: '/video-creator/v1/extend',
  retake: '/video-creator/v1/retake',
  restyle: '/video-creator/v1/restyle',
  'restyle:extract-first-frame': '/video-creator/v1/extract-first-frame',
  'restyle:segment-subject': '/video-creator/v1/sam3',
  'restyle:style-frame': '/video-creator/v1/style',
  'ic-lora': '/video-creator/v1/ic-lora-generate',
  'ic-lora:extract-conditioning': '/video-creator/v1/extract-conditioning',
  edit: '/video-creator/v1/edit',
}
export function endpointForTask(type: string): string {
  return TASK_ENDPOINTS[type] || `/video-creator/v1/${type}`
}

export interface RunnerDto {
  runner_id: string
  url: string
  status: string
  selected: boolean
  excluded: boolean
  gpu?: { name?: string; vram_mb?: number } | null
  price_info?: { price?: number; currency?: string; unit?: string } | null
  capabilities?: Array<{ id: string; label: string }>
}

export interface SignTicketMaterial {
  payment: string
  segCreds: string
  state?: string
}


/** Clip a base64 data-URL payload down to just the base64 body (strip the data: prefix). */
function stripDataPrefix(url: string): string {
  const comma = url.indexOf(',')
  return comma >= 0 ? url.slice(comma + 1) : url
}

/**
 * Read a browser web:// asset's bytes as a base64 payload. Remote live-runner workers (extend,
 * retake, ic-lora, extract-conditioning, ...) cannot fetch a browser-local web:// key — they
 * require the actual media as base64 in the request body (e.g. `video_base64`). Returns null
 * when the path is not a readable web:// blob (caller should treat as a missing/unsupported source).
 */
export async function pathToBase64(path: string): Promise<string | null> {
  if (!path) return null
  let blob: Blob | null = null
  if (path.startsWith('web://')) {
    // Prefer the stored raw bytes; fall back to the live object URL (an asset may carry only a
    // blobUrl after restore-from-disk, or the raw bytes may have been dropped).
    blob = getBlob(path) ?? null
    if (!blob) {
      const url = getBlobUrl(path)
      if (url) {
        try { blob = await fetch(url).then((r) => r.blob()) } catch { blob = null }
      }
    }
  } else if (path.startsWith('blob:') || path.startsWith('http:') || path.startsWith('https:')) {
    try { blob = await fetch(path).then((r) => r.blob()) } catch { blob = null }
  } else if (path.startsWith('data:')) {
    return path.slice(path.indexOf(',') + 1)
  }
  if (!blob) return null
  return await new Promise<string>((resolve, reject) => {
    const fr = new FileReader()
    fr.onload = () => {
      const url = fr.result
      if (typeof url === 'string') resolve(stripDataPrefix(url))
      else reject(new Error('Could not read media blob'))
    }
    fr.onerror = () => reject(fr.error ?? new Error('Could not read media blob'))
    fr.readAsDataURL(blob as Blob)
  })
}

export function makeJobId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return crypto.randomUUID()
  return 'job-' + Math.random().toString(36).slice(2) + Date.now().toString(36)
}

/** Map a runner's price_info.unit to the Livepeer ticket `type`: hour/seconds -> live, 720p->lv2v, fixed->fixed. */
export function ticketTypeForUnit(unit?: string): string {
  switch ((unit || '').toLowerCase()) {
    case 'hour':
    case 'seconds':
    case 'live':
      return 'live'
    case '720p':
    case 'lv2v':
      return 'lv2v'
    case 'fixed':
      return 'fixed'
    default:
      return 'live'
  }
}

/**
 * Resolve a ready, capable runner. Prefers the user's selected runner when it's still ready and
 * capable; otherwise the first capable one. Returns null when none is available.
 */
// Only a runner the BROWSER can actually fetch is usable for direct (paid) generation. Demo
// placeholders (providers.ts, DEMO_RUNNERS=1) carry a fake `livepeer://runner.demo/...` url
// that fetch() rejects ("URL scheme livepeer is not supported") — such a runner must never be
// selected, or the caller gets an opaque scheme error instead of a graceful "no real runner".
function isFetchableRunnerUrl(url: string): boolean {
  try {
    const p = new URL(url)
    return p.protocol === 'http:' || p.protocol === 'https:'
  } catch {
    return false
  }
}

export async function resolveRunner(requiredCaps: string[]): Promise<RunnerDto | null> {
  const res = await ApiClient.getProviders()
  if (!res.ok) throw new Error('Failed to load providers (runner discovery)')
  const providers: RunnerDto[] = res.data.providers ?? []
  const capable = providers.filter((p) => {
    if (p.status !== 'ready') return false
    if (p.excluded) return false
    if (!isFetchableRunnerUrl(p.url)) return false
    if (requiredCaps.length === 0) return true
    const ids = new Set((p.capabilities ?? []).map((c) => c.id))
    return requiredCaps.every((c) => ids.has(c))
  })
  if (capable.length === 0) return null
  return capable.find((p) => p.selected) ?? capable[0]
}

/**
 * POST a job payload DIRECTLY to a runner, performing the Livepeer payment handshake:
 *   1) send with Livepeer-Payer-Address (no payment) -> orchestrator replies 402 + payment params
 *   2) POST /sign-ticket { paymentParams, type, manifestId, state } -> { payment, segCreds, state }
 *   3) retry the same POST with Livepeer-Payment + Livepeer-Segment + Origin
 * Returns the final Response (the orchestrator streams the generated media in the body).
 *
 * LONG-LIVED CONNECTION: the generation POST returned here is a single-shot request that STAYS
 * OPEN while the runner does inference and then streams the media back in the response body over
 * that same connection. Callers must read the media from THIS response (e.g. `const blob = await
 * res.blob()`), which blocks until the stream completes. Do NOT poll a progress endpoint or issue
 * a separate download — there is no result URL; the response body IS the result.
 */
export async function postToRunnerWithTicket(
  runner: RunnerDto,
  task: string,
  body: unknown,
  opts?: { signal?: AbortSignal; sse?: boolean },
): Promise<Response> {
  if (!isFetchableRunnerUrl(runner.url)) {
    throw new Error(
      `Runner ${runner.runner_id} has no fetchable URL (${runner.url}). This is typically a ` +
        'demo placeholder (DEMO_RUNNERS) — a real Livepeer orchestrator runner is required for paid generation.',
    )
  }
  // sse=1 rides the query string so it survives the go-livepeer reverse proxy
  // (it copies RawQuery) and tells the runner to answer as text/event-stream.
  let url = runner.url.replace(/\/+$/, '') + endpointForTask(task)
  if (opts?.sse) url += (url.includes('?') ? '&' : '?') + 'sse=1'
  // Livepeer ties the payment/auth ticket to a specific manifest. The ORCHESTRATOR
  // assigns the manifest in its 402 challenge (`manifest_id`); we must sign the ticket
  // for THAT manifest or go-livepeer rejects the retry with "mismatched manifest and
  // auth token". makeJobId() is only a fallback if the challenge omits it.
  let manifestId = makeJobId()

  // 1) Payer-address-only probe -> expect 402 + payment_params.
  const t0 = Date.now()
  const addressRes = await ApiClient.getSignerAddress()
  logger.info(`[direct] getSignerAddress ok=${addressRes.ok} (${Date.now() - t0}ms)`)
  if (!addressRes.ok || !addressRes.data.address) {
    throw new Error('No Livepeer payer address available (is the platform / platform API key configured?)')
  }
  let paymentParams: unknown = null
  let state: string | undefined
  try {
    logger.info(`[direct] probe POST ${url}`)
    const probe = await fetch(url, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'Livepeer-Payer-Address': addressRes.data.address,
      },
      body: JSON.stringify(body),
      signal: opts?.signal ?? fetchWithTimeout(20000),
    })
    logger.info(`[direct] probe status=${probe.status} (${Date.now() - t0}ms)`)
    if (probe.status === 402) {
      const probeJson = await probe.json().catch(() => null)
      paymentParams = probeJson?.payment_params ?? probeJson?.paymentParams ?? probeJson ?? {}
      // Use the orchestrator-assigned manifest id, not our own.
      if (probeJson && typeof probeJson.manifest_id === 'string' && probeJson.manifest_id) {
        manifestId = probeJson.manifest_id
      }
    } else if (probe.ok) {
      // Some orchestrators accept directly (free runner / already-paid session). Return as-is.
      return probe
    } else if (probe.status === 401 || probe.status === 403) {
      throw new Error('Runner rejected the payer address (401/403)')
    } else {
      // Not a 402 challenge and not OK — surface the body for the caller (e.g. 4xx validation).
      return probe
    }
  } catch (e) {
    if (e instanceof Error && e.name === 'AbortError') throw e
    // Network failure to the runner despite discovery — rethrow so caller can report.
    throw e
  }

  // 2) Sign the payment ticket via the Worker (the only Worker involvement in the media path).
  const type = ticketTypeForUnit(runner.price_info?.unit)
  const t1 = Date.now()
  const signed = await ApiClient.signTicket({
    paymentParams,
    type,
    manifestId,
    state,
  })
  logger.info(`[direct] signTicket ok=${signed.ok} (${Date.now() - t1}ms)`)
  if (!signed.ok) {
    throw new Error(`Could not obtain payment ticket (${signed.status}): ${JSON.stringify(signed.error)}`)
  }

  // 3) Retry with the payment material.
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'Livepeer-Payment': signed.data.payment,
    'Livepeer-Segment': signed.data.segCreds,
    Origin: window.location.origin,
  }
  logger.info(`[direct] paid retry POST ${url}`)

  return fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal: opts?.signal,
  })
}

/**
 * Generic Livepeer task over HTTP with the FULL payment handshake
 * (postToRunnerWithTicket: Livepeer-Payer-Address -> 402+params -> /sign-ticket ->
 * retry with Livepeer-Payment + Livepeer-Segment). The orchestrator signs the
 * ticket and proxies to the runner; the runner returns JSON media (video_base64 /
 * image_base64 / output_video) which we decode to a Blob.
 *
 * Mirrors the WebSocket transport's return contract { payload, mediaBlob } so
 * callers decode media identically. The WS transport is NOT usable against a
 * billing live-runner orchestrator: a browser WebSocket cannot set the
 * Livepeer-Payment/Livepeer-Segment headers, so the orchestrator rejects the job
 * with 402 "invalid live runner payment signer address". The onProgress callback
 * is accepted for source compatibility but never fires (no per-step stream over
 * HTTP).
 */
export async function postRunnerTaskWithTicket(
  runner: RunnerDto,
  task: string,
  body: unknown,
  opts?: { signal?: AbortSignal; onProgress?: (ev: RunnerProgressEvent) => void },
): Promise<{ payload: Record<string, unknown>; mediaBlob?: Blob }> {
  const res = await postToRunnerWithTicket(runner, task, body, opts)
  if (!res.ok) {
    let message = `Runner ${task} failed (${res.status})`
    try {
      const j = (await res.json()) as { error?: unknown } | null
      if (j && typeof j.error === 'string') message = j.error
    } catch { /* keep status message */ }
    throw new Error(message)
  }
  const payload = (await res.json().catch(() => null)) as Record<string, unknown> | null
  if (!payload) throw new Error('Runner returned no JSON payload')
  return { payload, mediaBlob: decodeMediaPayload(payload) }
}

// ── SSE generation transport ────────────────────────────────────────────────
// Generation tasks (image/video) stream live progress to the browser over
// Server-Sent Events instead of a WebSocket: the orchestrator reverse-proxies the
// paid POST (Livepeer-Payment headers) to the runner, and the runner answers with
// text/event-stream. The final `complete` event carries the whole result payload
// (media as base64) in ONE event. Because SSE rides a normal fetch + ReadableStream,
// it keeps the working HTTP payment rail (no browser WebSocket -> no header limits).

/**
 * Drive `reader` and deliver parsed SSE events to onEvent.
 * Splits the byte stream on blank lines; a block's `event:` field names the event
 * (default "message") and its `data:` lines are joined with newlines per the SSE spec.
 */
async function readSSEStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  signal: AbortSignal | undefined,
  onEvent: (event: string, data: string) => void,
): Promise<void> {
  const decoder = new TextDecoder()
  let buf = ''
  const onAbort = () => { void reader.cancel().catch(() => {}) }
  if (signal) {
    if (signal.aborted) onAbort()
    else signal.addEventListener('abort', onAbort, { once: true })
  }
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
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
    if (signal) signal.removeEventListener('abort', onAbort)
  }
}

/**
 * Generation task over SSE with the full Livepeer payment handshake.
 * Same rail as postRunnerTaskWithTicket but the orchestrator-proxied runner answers
 * with text/event-stream: `accepted` -> `progress`* -> single `complete` (media base64
 * in one event) or `error`. onProgress fires for each progress event (0..1 only for
 * genuine backbone steps). Falls back to plain-JSON decoding if the runner did not
 * actually stream SSE yet.
 */
export async function postRunnerTaskWithTicketSSE(
  runner: RunnerDto,
  task: string,
  body: unknown,
  opts?: { signal?: AbortSignal; onProgress?: (ev: RunnerProgressEvent) => void },
): Promise<{ payload: Record<string, unknown>; mediaBlob?: Blob }> {
  const res = await postToRunnerWithTicket(runner, task, body, { ...opts, sse: true })
  if (!res.ok) {
    let message = `Runner ${task} failed (${res.status})`
    try {
      const j = (await res.json()) as { error?: unknown } | null
      if (j && typeof j.error === 'string') message = j.error
    } catch { /* keep status message */ }
    throw new Error(message)
  }
  const ct = res.headers.get('content-type') ?? ''
  if (!ct.includes('text/event-stream') || !res.body) {
    // Runner hasn't been switched to SSE yet (or returns JSON) - behave like the old rail.
    const payload = (await res.json().catch(() => null)) as Record<string, unknown> | null
    if (!payload) throw new Error('Runner returned no JSON payload')
    return { payload, mediaBlob: decodeMediaPayload(payload) }
  }
  return new Promise((resolve, reject) => {
    let settled = false
    readSSEStream(res.body!.getReader(), opts?.signal, (event, data) => {
      if (settled) return
      if (event === 'progress') {
        let p: Record<string, unknown>
        try { p = JSON.parse(data) as Record<string, unknown> } catch { return }
        opts?.onProgress?.({
          stage: typeof p.stage === 'string' ? p.stage : null,
          message: typeof p.message === 'string' ? p.message : null,
          progress: typeof p.progress === 'number' ? p.progress : null,
        })
      } else if (event === 'complete') {
        settled = true
        let payload: Record<string, unknown>
        try { payload = JSON.parse(data) as Record<string, unknown> } catch {
          reject(new Error('Runner complete event was not valid JSON'))
          return
        }
        resolve({ payload, mediaBlob: decodeMediaPayload(payload) })
      } else if (event === 'error') {
        settled = true
        let msg = 'Runner SSE error'
        try { msg = String((JSON.parse(data) as { error?: unknown }).error ?? msg) } catch { /* keep default */ }
        reject(new Error(msg))
      }
    }).catch((e) => {
      if (!settled) { settled = true; reject(e instanceof Error ? e : new Error(String(e))) }
    })
  })
}

/** Livepeer image generation over SSE. Returns the generated image Blob. */
export async function postImageToRunnerSSE(
  runner: RunnerDto,
  body: unknown,
  opts?: { signal?: AbortSignal; onProgress?: (ev: RunnerProgressEvent) => void },
): Promise<Blob> {
  const res = await postRunnerTaskWithTicketSSE(runner, 'generate-image', body, opts)
  if (res.mediaBlob) return res.mediaBlob
  throw new Error(res.payload?.error ? String(res.payload.error) : 'Runner returned no image')
}

/**
 * Auto subject segmentation (SAM3) of the object-to-keep, via a Livepeer runner over
 * the direct transport. The browser owns the bytes of a web:// source image, so it
 * sends them as base64 in the JSON body (a remote runner cannot fetch a browser-local
 * blob — that is why the Worker rail skips this step in the web app). Performs the full
 * payment handshake and returns the subject mask. The runner returns { mask_b64, width,
 * height } — a text/mask result, NOT media, so unlike media tasks we parse JSON directly.
 */
export interface Sam3Result {
  maskB64: string
  width?: number
  height?: number
}
export async function segmentSubjectViaRunner(
  runner: RunnerDto,
  imageBase64: string,
  opts?: { mode?: 'auto' | 'text'; prompt?: string; signal?: AbortSignal },
): Promise<Sam3Result> {
  const res = await postToRunnerWithTicket(
    runner,
    'restyle:segment-subject',
    { image: imageBase64, mode: opts?.mode ?? 'auto', prompt: opts?.prompt },
    opts,
  )
  if (!res.ok) {
    let detail = `SAM3 segmentation failed (${res.status})`
    try {
      const j = (await res.json()) as { error?: unknown } | null
      if (j && typeof j.error === 'string') detail = j.error
    } catch { /* keep status message */ }
    throw new Error(detail)
  }
  const data = (await res.json().catch(() => null)) as
    | { mask_b64?: unknown; width?: unknown; height?: unknown } | null
  const maskB64 = data?.mask_b64
  if (typeof maskB64 !== 'string' || !maskB64) {
    throw new Error('SAM3 completed without a mask')
  }
  return {
    maskB64,
    width: typeof data.width === 'number' ? data.width : undefined,
    height: typeof data.height === 'number' ? data.height : undefined,
  }
}

// ── FAL (fal.run) direct path ──────────────────────────────────────────────

const FAL_BASE = 'https://fal.run'
const FAL_T2I = '/fal-ai/z-image/turbo'
const FAL_I2I = '/fal-ai/z-image/turbo/image-to-image'

async function falSubmit(path: string, falKey: string, payload: Record<string, unknown>): Promise<Blob> {
  const submit = await fetch(FAL_BASE + path, {
    method: 'POST',
    headers: { Authorization: `Key ${falKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!submit.ok) {
    const detail = (await submit.text().catch(() => '')).slice(0, 500)
    throw new Error(`FAL submit failed (${submit.status}): ${detail || 'unknown error'}`)
  }
  const json = (await submit.json().catch(() => null)) as { images?: Array<{ url?: string } | string> } | null
  const url = extractFalImageUrl(json)
  const download = await fetch(url)
  if (!download.ok) throw new Error(`FAL image download failed (${download.status})`)
  return await download.blob()
}

function extractFalImageUrl(payload: { images?: Array<{ url?: string } | string> } | null): string {
  const images = payload?.images
  if (Array.isArray(images) && images.length > 0) {
    const first = images[0]
    if (typeof first === 'string' && first) return first
    if (first && typeof first === 'object' && typeof (first as { url?: string }).url === 'string') {
      return (first as { url: string }).url
    }
  }
  throw new Error('FAL response missing image url')
}

/** FAL text-to-image. Returns the generated image Blob. */
export async function falGenerateT2I(
  falKey: string,
  p: { prompt: string; width: number; height: number; seed: number; numInferenceSteps: number },
): Promise<Blob> {
  return falSubmit(FAL_T2I, falKey, {
    prompt: p.prompt,
    image_size: { width: p.width, height: p.height },
    num_inference_steps: p.numInferenceSteps,
    seed: p.seed,
    num_images: 1,
    output_format: 'png',
    acceleration: 'regular',
    enable_safety_checker: true,
  })
}

/** FAL image-to-image (edit). imageDataUri is the source image as a data URI. Returns the edited image Blob. */
export async function falGenerateI2I(
  falKey: string,
  p: { prompt: string; imageDataUri: string; strength: number; seed: number; numInferenceSteps: number },
): Promise<Blob> {
  return falSubmit(FAL_I2I, falKey, {
    prompt: p.prompt,
    image_url: p.imageDataUri,
    strength: p.strength,
    num_inference_steps: p.numInferenceSteps,
    seed: p.seed,
    num_images: 1,
    output_format: 'png',
    acceleration: 'regular',
    enable_safety_checker: true,
  })
}

/**
 * Prompt enhancement via a Livepeer runner (direct transport). Unlike media tasks this returns
 * TEXT (the rewritten prompt), so it parses the JSON body rather than a media blob. The runner's
 * `/video-creator/v1/prompt-enhance` returns { enhancedPrompt } (or { prompt }).
 */
export async function enhancePromptViaRunner(
  runner: RunnerDto,
  body: unknown,
  opts?: { signal?: AbortSignal },
): Promise<string> {
  const res = await postToRunnerWithTicket(runner, 'enhance-prompt', body, opts)
  if (!res.ok) {
    let msg = `Enhance request failed (${res.status})`
    try {
      const errJson = (await res.json()) as { error?: unknown } | null
      if (typeof errJson?.error === 'string') msg = errJson.error
    } catch { /* keep status message */ }
    throw new Error(msg)
  }
  const data = (await res.json().catch(() => null)) as { enhancedPrompt?: unknown; prompt?: unknown } | null
  const enhanced = data?.enhancedPrompt ?? data?.prompt
  if (typeof enhanced === 'string' && enhanced.trim()) return enhanced
  throw new Error('Enhance completed without a rewritten prompt')
}

/**
 * Livepeer image generation over HTTP with the full payment handshake
 * (postToRunnerWithTicket: Livepeer-Payer-Address -> 402+params -> /sign-ticket ->
 * retry with Livepeer-Payment + Livepeer-Segment). The orchestrator signs the
 * ticket and proxies to the runner; the runner returns { image_base64 }.
 *
 * NOTE: the WebSocket transport cannot be used for a paying Livepeer orchestrator
 * because a browser WebSocket cannot set the Livepeer-Payment/Livepeer-Segment
 * headers — the orchestrator then rejects the live-runner job with
 * "invalid live runner payment signer address" (402).
 */
export async function postImageToRunner(
  runner: RunnerDto,
  body: unknown,
  opts?: { signal?: AbortSignal },
): Promise<Blob> {
  const res = await postToRunnerWithTicket(runner, 'generate-image', body, opts)
  if (!res.ok) {
    let detail = `Runner image request failed (${res.status})`
    try {
      const j = (await res.json()) as { error?: unknown } | null
      if (j && typeof j.error === 'string') detail = j.error
    } catch { /* keep status message */ }
    throw new Error(detail)
  }
  const data = (await res.json().catch(() => null)) as
    | { image_base64?: unknown; content_type?: unknown } | null
  const b64 = data?.image_base64
  if (typeof b64 !== 'string' || !b64) {
    throw new Error('Runner returned no image')
  }
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  const type = typeof data.content_type === 'string' && data.content_type ? data.content_type : 'image/png'
  return new Blob([bytes], { type })
}

// ── WebSocket generation transport ──────────────────────────────────────────

/**
 * Progress event delivered to onProgress by the WebSocket transport. progress
 * is 0..1 ONLY for genuine backbone inference steps; otherwise null (text-only).
 */
export interface RunnerProgressEvent {
  stage?: string | null
  message?: string | null
  progress?: number | null
}

/**
 * Decode the runner complete payload into a media Blob, if it carries one.
 * Video tasks return video_base64, image tasks image_base64/styled_image,
 * restyle output_video (all base64). Returns undefined for text-only payloads
 * (e.g. prompt-enhance -> enhanced_prompt).
 */
function decodeMediaPayload(payload: Record<string, unknown>): Blob | undefined {
  for (const key of ['video_base64', 'image_base64', 'output_video', 'styled_image'] as const) {
    const b64 = payload[key]
    if (typeof b64 === 'string' && b64) {
      const contentType =
        typeof payload.content_type === 'string' ? payload.content_type
        : key === 'video_base64' || key === 'output_video' ? 'video/mp4'
        : 'image/png'
      const binary = atob(b64)
      const bytes = new Uint8Array(binary.length)
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
      return new Blob([bytes], { type: contentType })
    }
  }
  return undefined
}

/**
 * Open ONE WebSocket to a runner (runner.url http->ws / https->wss +
 * /video-creator/v1/ws), send a generation task as the first message, stream
 * progress frames to onProgress, and resolve with the final complete payload
 * (media decoded to a Blob when the task produced media).
 *
 * LONG-LIVED CONNECTION: the socket stays open for the whole generation — the
 * runner streams progress and finally the result media back over the same socket
 * (no HTTP timeout, no polling, no separate download).
 */
export async function postToRunnerViaWebSocket(
  runner: RunnerDto,
  task: string,
  body: unknown,
  opts?: { signal?: AbortSignal; onProgress?: (ev: RunnerProgressEvent) => void },
): Promise<{ payload: Record<string, unknown>; mediaBlob?: Blob }> {
  const rawUrl = runner.url.replace(/\/+$/, '')
  const wss = rawUrl.replace(/^https:/, 'wss:').replace(/^http:/, 'ws:') + '/video-creator/v1/ws'
  const jobId = makeJobId()
  const requestId = makeJobId()
  // The runner routes on its endpoint name (the last segment of the TASK_ENDPOINTS path),
  // not the frontend task label (e.g. task 'generate' -> endpoint 't2v', 'generate-image'
  // -> 'image'). Resolve the endpoint name so the WS `type` matches the runner's ROUTES.
  const endpoint = endpointForTask(task).split('/').filter(Boolean).pop() || task
  return new Promise((resolve, reject) => {
    let ws: WebSocket | null = null
    try {
      ws = new WebSocket(wss)
    } catch (e) {
      reject(e instanceof Error ? e : new Error('Failed to open runner websocket'))
      return
    }
    let settled = false
    const fail = (err: Error) => {
      if (settled) return
      settled = true
      try { ws?.close() } catch { /* noop */ }
      reject(err)
    }
    ws.onopen = () => {
      ws?.send(JSON.stringify({
        type: endpoint,
        request_id: requestId,
        body: { ...(body as object), job_id: jobId },
      }))
    }
    ws.onmessage = (ev: MessageEvent) => {
      let msg: { type?: string; payload?: unknown; error?: unknown; stage?: string; message?: string; progress?: unknown }
      try { msg = JSON.parse(String(ev.data)) as typeof msg } catch { return }
      if (!msg || typeof msg !== 'object') return
      if (msg.type === 'accepted' || msg.type === 'progress') {
        opts?.onProgress?.({
          stage: typeof msg.stage === 'string' ? msg.stage : null,
          message: typeof msg.message === 'string' ? msg.message : null,
          progress: typeof msg.progress === 'number' ? msg.progress : null,
        })
      } else if (msg.type === 'complete') {
        const payload = (msg.payload ?? {}) as Record<string, unknown>
        settled = true
        try { ws?.close() } catch { /* noop */ }
        resolve({ payload, mediaBlob: decodeMediaPayload(payload) })
      } else if (msg.type === 'error') {
        fail(new Error(String(msg.error ?? 'Runner websocket error')))
      }
    }
    ws.onerror = () => fail(new Error('Runner websocket error'))
    ws.onclose = () => {
      if (!settled) fail(new Error('Runner websocket closed before completion'))
    }
    if (opts?.signal) {
      opts.signal.addEventListener('abort', () => {
        settled = true
        try { ws?.close() } catch { /* noop */ }
        reject(new DOMException('Aborted', 'AbortError'))
      }, { once: true })
    }
  })
}

/**
 * Prompt enhancement via a Livepeer runner over WebSocket. Returns the rewritten
 * prompt text (the runner complete payload carries enhanced_prompt).
 */
export async function enhancePromptViaWebSocket(
  runner: RunnerDto,
  body: unknown,
  opts?: { signal?: AbortSignal },
): Promise<string> {
  const res = await postToRunnerViaWebSocket(runner, 'prompt-enhance', body, opts)
  const enhanced = res.payload.enhanced_prompt ?? res.payload.enhancedPrompt ?? res.payload.prompt
  if (typeof enhanced === 'string' && enhanced.trim()) return enhanced
  throw new Error('Enhance completed without a rewritten prompt')
}
