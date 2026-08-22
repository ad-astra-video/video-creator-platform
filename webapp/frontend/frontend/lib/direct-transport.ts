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
import { getBillingProjectId } from './billing-context'
import { getBlob, getBlobUrl, registerBlob, setDimensions } from './runtime/web-store'
import { decodeMediaPayload } from './media-decode'
import { readSSEStream } from './sse-stream'
import { logger } from './logger'
import { isWebPlatform, discoverRunners, getRunnerDiscoveryConfig, loadExcludedRunnerUrls } from './livepeer-discovery'

/** Fetch with an AbortSignal.timeout guard so a hung hop surfaces as an error instead of a silent await. */
function fetchWithTimeout(ms: number): RequestInit['signal'] {
  return typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function'
    ? AbortSignal.timeout(ms)
    : undefined
}

// 503 (Service Unavailable) retry policy — a runner returns 503 when it has no free GPU
// (scheduler queue timeout) or its engine isn't loaded yet; both are transient and worth
// waiting out with exponential backoff before surfacing a failure. Resolve early on abort.
const MAX_503_RETRIES = 5
const MAX_503_BACKOFF_MS = 5000
function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal?.aborted) return resolve()
    const onAbort = () => { clearTimeout(timer); resolve() }
    const timer = setTimeout(() => { signal?.removeEventListener('abort', onAbort); resolve() }, ms)
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

// Task -> runner endpoint (must match the live-runner /video-creator/v1 route table that the
// runner service exposes. See orchestrator.ts TASK_ENDPOINTS — we mirror the same shape here
// so the browser can POST to the runner directly).
const TASK_ENDPOINTS: Record<string, string> = {
  generate: '/video-creator/v1/t2v',
  'generate-i2v': '/video-creator/v1/i2v',
  'generate-image': '/video-creator/v1/image',
  'enhance-prompt': '/video-creator/v1/prompt-enhance',
  extend: '/video-creator/v1/extend',
  retake: '/video-creator/v1/retake',
  restyle: '/video-creator/v1/restyle',
  'restyle:extract-first-frame': '/video-creator/v1/extract-first-frame',
  'restyle:segment-subject': '/video-creator/v1/sam3',
  'restyle:style-frame': '/video-creator/v1/style-frame',
  'ic-lora': '/video-creator/v1/ic-lora-generate',
  'ic-lora:extract-conditioning': '/video-creator/v1/extract-conditioning',
  edit: '/video-creator/v1/edit',
  layer: '/video-creator/v1/layer',
  // Bernini generation/edit rails (wan-worker, engine id `idv2v`) — mirror the
  // live-runner ROUTES table additions so the browser POSTs to the right worker.
  'bernini-t2v': '/video-creator/v1/bernini-t2v',
  'bernini-v2v': '/video-creator/v1/bernini-v2v',
  'bernini-r2v': '/video-creator/v1/bernini-r2v',
  // Post-process rails (vp-worker) — RIFE fps-boost + FlashVSR upscale + ffmpeg,
  // used for Bernini delivery targets above native (480p@16).
  'process': '/video-creator/v1/process',
  'fps-boost': '/video-creator/v1/fps-boost',
  'upscale': '/video-creator/v1/upscale',
  'ffmpeg': '/video-creator/v1/ffmpeg',
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
  /** Advertised concurrency capacity (runner GPU count) and its current usage, when the runner sends them. */
  capacity?: number
  capacityUsed?: number
  capacityAvailable?: number
  capabilities?: Array<{ id: string; label: string }>
  models?: string[]
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


/**
 * Style the restyle first frame DIRECTLY on a runner (paid Livepeer rail): hand the actual
 * image bytes (base64) to the id-v2v worker's /video-creator/v1/style-frame (FLUX.2 klein 4B)
 * instead of the Worker rail, which can't read a browser web:// asset key and simply skips
 * ("style-frame unavailable in browser without asset upload").
 */
export async function styleFrameViaRunner(
  runner: RunnerDto,
  imageBase64: string,
  prompt: string,
  opts?: { seed?: number; enhance?: boolean; quality?: 'fast' | 'balanced' | 'high'; signal?: AbortSignal },
): Promise<{ styledImageUrl: string; width?: number; height?: number; enhancedPrompt?: string }> {
  const res = await postToRunnerWithTicket(
    runner,
    'restyle:style-frame',
    {
      image: imageBase64,
      prompt,
      seed: opts?.seed,
      enhance_prompt: opts?.enhance ?? false,
      quality: opts?.quality,
    },
    opts,
  )
  if (!res.ok) {
    let detail = `First-frame style failed (${res.status})`
    try {
      const j = (await res.json()) as { error?: unknown } | null
      if (j && typeof j.error === 'string') detail = j.error
    } catch { /* keep status message */ }
    throw new Error(detail)
  }
  const data = (await res.json().catch(() => null)) as
    | { styled_image?: unknown; width?: unknown; height?: unknown; enhanced_prompt?: unknown } | null
  const styledB64 = data?.styled_image
  if (typeof styledB64 !== 'string' || !styledB64) throw new Error('First-frame style completed without an image')
  const bin = atob(styledB64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  // Register the bytes in the browser asset store so the returned value is a `web://<uuid>`
  // key that webAssetUrl() can resolve (a raw blob: URL would fall through pathToFileUrl to a
  // broken file:///blob:... link in every <img> that renders it).
  const key = registerBlob(new Blob([bytes], { type: 'image/png' }), 'styled-frame.png', 'image/png')
  if (data?.width != null && data?.height != null) {
    try { setDimensions(key, Number(data.width), Number(data.height)) } catch { /* non-fatal */ }
  }
  return {
    styledImageUrl: key,
    width: typeof data?.width === 'number' ? data.width : undefined,
    height: typeof data?.height === 'number' ? data.height : undefined,
    enhancedPrompt: typeof data?.enhanced_prompt === 'string' ? data.enhanced_prompt : undefined,
  }
}

/**
 * Run a Qwen-Image-Edit instruction edit of a base image DIRECTLY on a runner (paid
 * Livepeer rail) via /video-creator/v1/edit with the `engine` field set to `qwen-edit`
 * (the ltx-worker Qwen-Image-Edit pipeline). Used to style the restyle first frame when
 * the user picks the Qwen first-frame engine. Mirrors styleFrameViaRunner's contract:
 * the returned value is a `web://<uuid>` key (registerBlob) that webAssetUrl() can resolve.
 */
export interface EditMediaResult {
  imageUrl: string
  width?: number
  height?: number
}
function parseEditMediaResult(payload: { image?: unknown; image_b64?: unknown; edited_image?: unknown; width?: unknown; height?: unknown } | null): EditMediaResult {
  const data = (payload ?? {}) as { image?: unknown; image_b64?: unknown; edited_image?: unknown; width?: unknown; height?: unknown }
  const b64 = (typeof data.image_b64 === 'string' ? data.image_b64
    : typeof data.image === 'string' ? data.image
    : typeof data.edited_image === 'string' ? data.edited_image : null)
  if (!b64) throw new Error('Image edit completed without an image')
  const bin = atob(b64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  const key = registerBlob(new Blob([bytes], { type: 'image/png' }), 'edit.png', 'image/png')
  if (data.width != null && data.height != null) {
    try { setDimensions(key, Number(data.width), Number(data.height)) } catch { /* non-fatal */ }
  }
  return {
    imageUrl: key,
    width: typeof data.width === 'number' ? data.width : undefined,
    height: typeof data.height === 'number' ? data.height : undefined,
  }
}
export async function editImageViaRunner(
  runner: RunnerDto,
  imageBase64: string,
  prompt: string,
  opts?: { engine?: string; seed?: number; strength?: number; paddingMaskCrop?: number; enhance?: boolean; maskImage?: string; quality?: 'fast' | 'balanced' | 'high'; signal?: AbortSignal; onProgress?: (p: LayerRunProgress) => void },
): Promise<EditMediaResult> {
  const res = await postToRunnerWithTicket(
    runner,
    'edit',
    {
      image: imageBase64,
      prompt,
      engine: opts?.engine ?? 'qwen-edit',
      seed: opts?.seed,
      strength: opts?.strength,
      quality: opts?.quality,
      enhance_prompt: opts?.enhance ?? false,
      padding_mask_crop: opts?.paddingMaskCrop,
      // A base64 alpha/RGB mask limits the edit to only the masked regions
      // (Qwen-Image-Edit forwards it as mask_images; Z-Image uses inpaint).
      mask_image: opts?.maskImage,
    },
    { ...opts, sse: opts?.onProgress ? true : false },
  )
  if (!res.ok) {
    let detail = `Image edit failed (${res.status})`
    try {
      const j = (await res.json()) as { error?: unknown } | null
      if (j && typeof j.error === 'string') detail = j.error
    } catch { /* keep status message */ }
    throw new Error(detail)
  }
  const ct = res.headers.get('content-type') ?? ''
  if (opts?.onProgress && ct.includes('text/event-stream') && res.body) {
    // SSE: accepted -> progress* (per denoise step) -> single complete with the
    // edited image. Mirrors the /layer SSE consumption.
    return await new Promise<EditMediaResult>((resolve, reject) => {
      let settled = false
      readSSEStream(res.body!.getReader(), opts.signal, (event, data) => {
        if (settled) return
        if (event === 'progress') {
          let p: Record<string, unknown>
          try { p = JSON.parse(data) as Record<string, unknown> } catch { return }
          const step = typeof p.step === 'number' ? p.step : null
          const total = typeof p.total_steps === 'number' ? p.total_steps : null
          if (step != null) opts?.onProgress?.({ step, totalSteps: total != null ? total : 50 })
        } else if (event === 'complete') {
          settled = true
          try {
            const payload = JSON.parse(data) as { image?: unknown; width?: unknown; height?: unknown }
            resolve(parseEditMediaResult(payload))
          } catch (e) {
            reject(e instanceof Error ? e : new Error(String(e)))
          }
        } else if (event === 'error') {
          settled = true
          let msg = 'Image edit SSE error'
          try { msg = String((JSON.parse(data) as { error?: unknown }).error ?? msg) } catch { /* keep default */ }
          reject(new Error(msg))
        }
      }).then(() => {
        if (!settled) { settled = true; reject(new Error('Runner SSE connection closed before it completed')) }
      }).catch((e) => {
        if (!settled) { settled = true; reject(e instanceof Error ? e : new Error(String(e))) }
      })
    })
  }
  const data = (await res.json().catch(() => null)) as
    | { image?: unknown; image_b64?: unknown; edited_image?: unknown; width?: unknown; height?: unknown } | null
  return parseEditMediaResult(data)
}

export interface Sam3MaskResult {
  /** web:// key of the returned mask (white = selected item, black = rest). */
  maskKey: string
  /** raw base64 mask PNG (white-on-black, original resolution). */
  maskB64: string
  width?: number
  height?: number
  prompt: string
}

export async function sam3SelectViaRunner(
  runner: RunnerDto,
  imageBase64: string,
  prompt: string,
  opts?: { signal?: AbortSignal },
): Promise<Sam3MaskResult> {
  const res = await postToRunnerWithTicket(
    runner,
    'sam3',
    { image: imageBase64, mode: 'text', prompt },
    opts,
  )
  if (!res.ok) {
    let detail = `SAM3 selection failed (${res.status})`
    try {
      const j = (await res.json()) as { error?: unknown; detail?: unknown } | null
      if (j && typeof j.error === 'string') detail = j.error
    } catch { /* keep status */ }
    throw new Error(detail)
  }
  const data = (await res.json().catch(() => null)) as
    | { mask_b64?: unknown; width?: unknown; height?: unknown; prompt?: unknown } | null
  const b64 = typeof data?.mask_b64 === 'string' ? data.mask_b64 : null
  if (!b64) throw new Error('SAM3 selection returned no mask')
  const bin = atob(b64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  const key = registerBlob(new Blob([bytes], { type: 'image/png' }), 'sam3-mask.png', 'image/png')
  if (data?.width != null && data?.height != null) {
    try { setDimensions(key, Number(data.width), Number(data.height)) } catch { /* non-fatal */ }
  }
  return {
    maskKey: key,
    maskB64: b64,
    width: typeof data?.width === 'number' ? data.width : undefined,
    height: typeof data?.height === 'number' ? data.height : undefined,
    prompt: typeof data?.prompt === 'string' ? data.prompt : prompt,
  }
}

/**
 * Layered preprocessing: decompose an image into N RGBA layers via /video-creator/v1/layer
 * (Qwen-Image-Layered on the ltx-worker). `preview_only:true` keeps the interaction cheap —
 * the runner returns composited-on-black previews for the UI picker. Each preview is
 * re-registered as a `web://<uuid>` key so webAssetUrl() can render it. This is a
 * PREPROCESSING step: the layer output lives purely in component state and is NEVER
 * serialized into any video-request conditioning field.
 */
export interface LayerPreview {
  index: number
  /** web:// key of the composited-on-black preview thumbnail. */
  previewKey: string
  /** base64 RGBA layer image (undefined in preview-only mode). */
  rgbaB64?: string
  /** base64 alpha (mask) of the layer — undefined in preview-only mode. */
  alphaB64?: string
  /** Semantic label from the runner (e.g. foreground / midground / background). */
  label?: string
}
export interface LayerResult {
  layers: LayerPreview[]
  compositeKey?: string
  width?: number
  height?: number
}
export interface LayerRunProgress {
  step: number
  totalSteps: number
}
function parseLayerResult(payload: Record<string, unknown> | null): LayerResult {
  const data = (payload ?? {}) as {
    layers?: Array<{ index?: unknown; preview_b64?: unknown; image_b64?: unknown; rgba_b64?: unknown; alpha_b64?: unknown; label?: unknown }>
    composite?: unknown
    width?: unknown
    height?: unknown
  }
  const decodeToWebKey = (b64: string, name: string): string => {
    const bin = atob(b64)
    const bytes = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
    return registerBlob(new Blob([bytes], { type: 'image/png' }), name, 'image/png')
  }
  const rawLayers = data.layers ?? []
  const layers: LayerPreview[] = rawLayers
    .filter((l) => typeof l.preview_b64 === 'string' && (l.preview_b64 as string))
    .map((l, i) => ({
      index: typeof l.index === 'number' ? l.index : i,
      previewKey: decodeToWebKey(l.preview_b64 as string, `layer-${Date.now()}-${i}.png`),
      rgbaB64: typeof l.rgba_b64 === 'string' && (l.rgba_b64 as string) ? (l.rgba_b64 as string) : undefined,
      alphaB64: typeof l.alpha_b64 === 'string' && (l.alpha_b64 as string) ? (l.alpha_b64 as string) : undefined,
      label: typeof l.label === 'string' && (l.label as string) ? (l.label as string) : undefined,
    }))
  if (layers.length === 0) throw new Error('Layered preprocessing returned no readable layers')
  return {
    layers,
    compositeKey: typeof data.composite === 'string' ? decodeToWebKey(data.composite, 'layer-composite.png') : undefined,
    width: typeof data.width === 'number' ? data.width : undefined,
    height: typeof data.height === 'number' ? data.height : undefined,
  }
}
export async function layerImageViaRunner(
  runner: RunnerDto,
  imageBase64: string,
  layersCount: number,
  opts?: { previewOnly?: boolean; numSteps?: number; signal?: AbortSignal; onProgress?: (p: LayerRunProgress) => void },
): Promise<LayerResult> {
  const previewOnly = opts?.previewOnly ?? true
  const body = {
    image: imageBase64,
    layers: layersCount,
    preview_only: previewOnly,
    num_inference_steps: opts?.numSteps,
  }
  const res = await postToRunnerWithTicket(runner, 'layer', body, { ...opts, sse: true })
  if (!res.ok) {
    let detail = `Layered preprocessing failed (${res.status})`
    try {
      const j = (await res.json()) as { error?: unknown } | null
      if (j && typeof j.error === 'string') detail = j.error
    } catch { /* keep status message */ }
    throw new Error(detail)
  }
  const ct = res.headers.get('content-type') ?? ''
  if (!ct.includes('text/event-stream') || !res.body) {
    // Runner hasn't been switched to SSE yet — fall back to plain JSON.
    const payload = (await res.json().catch(() => null)) as Record<string, unknown> | null
    if (!payload) throw new Error('Layered preprocessing returned no readable payload')
    return parseLayerResult(payload)
  }
  // SSE: accepted -> progress* (per denoise step) -> single complete carrying the full contract.
  return new Promise((resolve, reject) => {
    let settled = false
    readSSEStream(res.body!.getReader(), opts?.signal, (event, data) => {
      if (settled) return
      if (event === 'progress') {
        let p: Record<string, unknown>
        try { p = JSON.parse(data) as Record<string, unknown> } catch { return }
        const step = typeof p.step === 'number' ? p.step : null
        const total = typeof p.total_steps === 'number' ? p.total_steps : null
        if (step != null) opts?.onProgress?.({ step, totalSteps: total != null ? total : opts?.numSteps ?? 30 })
      } else if (event === 'complete') {
        settled = true
        try {
          const payload = JSON.parse(data) as Record<string, unknown>
          resolve(parseLayerResult(payload))
        } catch (e) {
          reject(e instanceof Error ? e : new Error(String(e)))
        }
      } else if (event === 'error') {
        settled = true
        let msg = 'Layered preprocessing SSE error'
        try { msg = String((JSON.parse(data) as { error?: unknown }).error ?? msg) } catch { /* keep default */ }
        reject(new Error(msg))
      }
    }).then(() => {
      if (!settled) { settled = true; reject(new Error('Runner SSE connection closed before it completed')) }
    }).catch((e) => {
      if (!settled) { settled = true; reject(e instanceof Error ? e : new Error(String(e))) }
    })
  })
}

/**
 * Ask the multimodal Gemma agent (gemma-worker /video-creator/v1/suggest-layers)
 * how many semantic layers an image should decompose into (2-8 per the rubric).
 * Sends the image bytes + the rubric; returns the suggested count (null when the
 * agent doesn't produce a parseable number, so the caller keeps its default).
 */
export async function suggestLayersViaRunner(
  runner: RunnerDto,
  imageBase64: string,
  opts?: { signal?: AbortSignal },
): Promise<{ layers: number | null; raw: string }> {
  const res = await postToRunnerWithTicket(runner, 'suggest-layers', { image: imageBase64 }, opts)
  if (!res.ok) {
    let detail = `Layer-count suggestion failed (${res.status})`
    try {
      const j = (await res.json()) as { error?: unknown } | null
      if (j && typeof j.error === 'string') detail = j.error
    } catch { /* keep status message */ }
    throw new Error(detail)
  }
  const data = (await res.json().catch(() => null)) as { layers?: unknown; raw?: unknown } | null
  return {
    layers: typeof data?.layers === 'number' ? data.layers : null,
    raw: typeof data?.raw === 'string' ? data.raw : '',
  }
}

export async function resolveRunner(requiredCaps: string[]): Promise<RunnerDto | null> {
  // Web build: discover straight against the user's configured Discovery URL. The Worker has no
  // runner/job knowledge here (its /api/providers is control-plane discovery for the desktop
  // backend), so the browser re-hits the orchestrator directly — the same CORS-proven host as the
  // paying ticket rail. The persisted runner preference comes from settings; the browser-local
  // excluded set mirrors what RunnersSection shows.
  let providers: RunnerDto[]
  if (isWebPlatform()) {
    const cfg = getRunnerDiscoveryConfig()
    if (!cfg.discoveryUrl) return null
    const list = await discoverRunners(cfg.discoveryUrl)
    const excludedUrls = new Set(loadExcludedRunnerUrls())
    providers = list.map((p) => ({
      ...p,
      selected: p.runner_id === cfg.selectedRunnerId,
      excluded: p.excluded || excludedUrls.has(p.url),
    }))
  } else {
    const res = await ApiClient.getProviders()
    if (!res.ok) throw new Error('Failed to load providers (runner discovery)')
    providers = res.data.providers ?? []
  }
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
  // When a runner returns 503 (Service Unavailable — no free GPU / engine not loaded yet),
  // retry with exponential backoff (0.5s, 1s, 2s, 4s, then capped at 5s) before giving up.
  // Every retry re-runs the full payment handshake so each attempt gets a freshly signed
  // ticket for its own manifest. Non-503 responses are returned immediately.
  let retries = 0
  for (;;) {
    const res = await attemptInference(runner, task, body, opts)
    if (res.status === 503 && retries < MAX_503_RETRIES && !opts?.signal?.aborted) {
      retries++
      const delayMs = Math.min(MAX_503_BACKOFF_MS, 500 * 2 ** (retries - 1))
      logger.info(`[direct] ${task} 503 (retry ${retries}/${MAX_503_RETRIES}); backing off ${delayMs}ms`)
      // Free the rejected response so the connection can be reused before the next attempt.
      try { await res.body?.cancel() } catch { /* best-effort */ }
      await sleep(delayMs, opts?.signal)
      continue
    }
    return res
  }
}

/**
 * One attempt of the Livepeer payment handshake -> the final orchestrator-proxied runner
 * Response (which may be a 503 — postToRunnerWithTicket decides whether to retry).
 */
async function attemptInference(
  runner: RunnerDto,
  task: string,
  body: unknown,
  opts?: { signal?: AbortSignal; sse?: boolean },
): Promise<Response> {
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
      // Runner rejected the payer address — a signer-address payment error. Drop the
      // Worker's cached address (fire-and-forget) so the next attempt re-fetches fresh.
      void ApiClient.invalidateSignerAddress()
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
    projectId: getBillingProjectId() ?? undefined,
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

    // Guard the HEADER arrival of the paid retry: a proxy that accepts the connection
    // but never returns response headers would otherwise leave this fetch (and thus the
    // whole generation) pending forever. This bounds ONLY the header wait — for a
    // long-lived SSE body stream we must NOT abort after headers (the body is governed
    // by readSSEStream's idle/max-duration watchdog instead). So we race the HEADER
    // wait, not the body.
    const headerSignal = new AbortController()
    const payoutHeaderTimer = setTimeout(() => headerSignal.abort(), 60_000)
    const combined = combineAbortSignals(opts?.signal, headerSignal.signal)
    try {
      return await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: combined,
      })
    } catch (e) {
      if (headerSignal.signal.aborted && !(opts?.signal?.aborted)) {
        throw new Error(
          'Runner did not respond in time (60s). The orchestrator/runner connection stalled; ' +
          'no response headers were received. Please retry.',
        )
      }
      throw e
    } finally {
      clearTimeout(payoutHeaderTimer)
    }
  }

  /**
   * Return a signal that aborts when EITHER of `a` or `b` aborts. TL;DR for the paid
   * retry header guard: AbortSignal.timeout would kill the long SSE body too, but we only
   * want to bound the header wait, so we gate on a short-lived controller plus the user's
   * own abort (cancellation) signal.
   */
  function combineAbortSignals(a: AbortSignal | undefined, b: AbortSignal | undefined): AbortSignal | undefined {
    if (!a && !b) return undefined
    if (a && !b) return a
    if (b && !a) return b
    const c = new AbortController()
    const onAbort = () => c.abort()
    a!.addEventListener('abort', onAbort, { once: true })
    b!.addEventListener('abort', onAbort, { once: true })
    // Ensure it's aborted if either already was.
    if (a!.aborted || b!.aborted) c.abort()
    return c.signal
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
    }).then(() => {
      // Stream ended without a terminal complete/error event — connection dropped
      // (proxy idle timeout, runner exit) — surface a clear error instead of hanging.
      if (!settled) { settled = true; reject(new Error('Runner SSE connection closed before it completed')) }
    }).catch((e) => {
      if (!settled) { settled = true; reject(e instanceof Error ? e : new Error(String(e))) }
    })
  })
}

/** Livepeer image generation over SSE. Returns the generated image Blob + the
 * seed the runner used (when the runner echoes it back), so callers can persist
 * it as regeneration metadata. */
export interface RunnerImageResult {
  blob: Blob
  seed?: number
}
export async function postImageToRunnerSSE(
  runner: RunnerDto,
  body: unknown,
  opts?: { signal?: AbortSignal; onProgress?: (ev: RunnerProgressEvent) => void },
  /** Task/endpoint to hit. Defaults to T2I (`/image`, engines zimage|klein);
   *  pass `edit` to hit `/edit` (engines qwen-edit|zimage). */
  task: string = 'generate-image',
): Promise<RunnerImageResult> {
  const res = await postRunnerTaskWithTicketSSE(runner, task, body, opts)
  if (res.mediaBlob) {
    const seed = typeof res.payload?.seed === 'number' ? res.payload.seed
      : typeof res.payload?.seed === 'string' && res.payload.seed !== ''
        ? Number(res.payload.seed) : undefined
    return { blob: res.mediaBlob, seed }
  }
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
  const data = (await res.json().catch(() => null)) as
    | { enhanced_prompt?: unknown; enhancedPrompt?: unknown; prompt?: unknown } | null
  const enhanced = data?.enhanced_prompt ?? data?.enhancedPrompt ?? data?.prompt
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
