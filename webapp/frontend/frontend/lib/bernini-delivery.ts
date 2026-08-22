// Bernini (ByteDance) delivery-target derivation — the shared model behind the
// Generate-Videos + Edit-Video "adaptive resolution/fps" dropdowns.
//
// Bernini renders natively at 480p @ 16fps (long edge capped at --max_image_size 848).
// Every delivery target ABOVE native is reached only via the post rails (RIFE for fps,
// FlashVSR 4x for resolution, ffmpeg to downscale to a final tier). This module derives
// the reachable (resolution, fps) grid and — when a pair needs a disabled rail — hides it.
//
// Pure functions: no DOM, no backend imports, no generated OpenAPI types. Unit-tested in
// the node environment (vitest, frontend/**/*.test.ts).

export type BerniniEngine = 'bernini-1.3b' | 'bernini-14b'

/** A delivery resolution tier (+ 'raw-4x' = native FlashVSR 4x, no ffmpeg downscale). */
export type BerniniResolution = '480p' | '1080p' | '1440p' | 'raw-4x'

/** The final ffmpeg encode target for the upscale rail. 'raw' skips ffmpeg entirely. */
export type UpscaleFinal = '1080' | '1440' | 'raw'

export type FpsBoostMode = 'preserve_motion' | 'smooth'

/** Which post rail(s) a delivery target requires. */
export type PostRail = 'rife' | 'flashvsr' | 'ffmpeg'

export interface BerniniDelivery {
  resolution: BerniniResolution
  /** Delivery frame rate; > 16 requires the rife rail. */
  fps: number
  /** True iff this IS the native render (480p @ 16). */
  native: boolean
  /** Post rails required to reach this delivery from native. */
  post: PostRail[]
  /** Supported durations (seconds) at this delivery — calibrated on-box (Task 9). */
  durations: number[]
}

export interface BerniniRailsAvailable {
  rife: boolean
  flashvsr: boolean
}

export const BERNINI_NATIVE_FPS = 16
export const BERNINI_NATIVE_RESOLUTION: BerniniResolution = '480p'
/** Long edge the diffusers pipeline caps at (--max_image_size). */
export const BERNINI_MAX_IMAGE_SIZE = 848

const RES_ORDER: BerniniResolution[] = ['480p', '1080p', '1440p', 'raw-4x']
const FPS_OPTIONS: number[] = [16, 24, 30, 60]

// Native capacity by engine, in SECONDS at native. Post rails never extend duration.
// (Calibrated on-box in Task 9 — these are the reference-tuned defaults.)
const NATIVE_DURATIONS: Record<BerniniEngine, number[]> = {
  'bernini-1.3b': [2, 3, 5],
  'bernini-14b': [2, 3, 5],
}

const FINAL_BY_RESOLUTION: Partial<Record<BerniniResolution, UpscaleFinal>> = {
  '1080p': '1080',
  '1440p': '1440',
  'raw-4x': 'raw',
}

function postFor(resolution: BerniniResolution, fps: number): PostRail[] {
  const post: PostRail[] = []
  if (fps > BERNINI_NATIVE_FPS) post.push('rife')
  if (resolution === 'raw-4x') {
    post.push('flashvsr') // native 4x encode — NO ffmpeg downscale
  } else if (resolution !== BERNINI_NATIVE_RESOLUTION) {
    post.push('flashvsr', 'ffmpeg') // 4x then lanczos-downscale to the final tier
  }
  return post
}

/** All reachable deliveries for an engine, honoring which post rails are available. */
export function berniniDeliveryMatrix(
  engine: BerniniEngine,
  rails: BerniniRailsAvailable,
): BerniniDelivery[] {
  const out: BerniniDelivery[] = []
  for (const resolution of RES_ORDER) {
    for (const fps of FPS_OPTIONS) {
      const post = postFor(resolution, fps)
      if (post.includes('rife') && !rails.rife) continue
      if (post.includes('flashvsr') && !rails.flashvsr) continue
      out.push({
        resolution,
        fps,
        native: resolution === BERNINI_NATIVE_RESOLUTION && fps === BERNINI_NATIVE_FPS,
        post,
        durations: NATIVE_DURATIONS[engine],
      })
    }
  }
  return out
}

/** Reachable fps values at a given resolution (feeds the fps dropdown). */
export function berniniFpsOptions(
  resolution: BerniniResolution,
  matrix: BerniniDelivery[],
): number[] {
  return matrix
    .filter((d) => d.resolution === resolution)
    .map((d) => d.fps)
    .sort((a, b) => a - b)
}

/** Reachable resolutions at a given fps (feeds the resolution dropdown). */
export function berniniResolutionOptions(
  fps: number,
  matrix: BerniniDelivery[],
): BerniniResolution[] {
  return RES_ORDER.filter((res) => matrix.some((d) => d.fps === fps && d.resolution === res))
}

export interface BerniniPostPayload {
  // Contract matches the vp-worker /process body: fps_boost.target_fps (renamed
  // from `target` so the frontend payload is byte-identical to the worker body).
  fps_boost?: { target_fps: number; mode: FpsBoostMode }
  upscale?: { scale: 4; final: UpscaleFinal }
}

/**
 * The post rails the frontend should attach for a chosen delivery target. Native
 * (480p @ 16) returns `{}` — no post. `fps > 16` adds the RIFE fps-boost (Preserve Motion
 * by default). Any non-native resolution adds the FlashVSR upscale with the right final.
 */
export function berniniPostFor(d: { resolution: string; fps: number }): BerniniPostPayload {
  const payload: BerniniPostPayload = {}
  if (d.fps > BERNINI_NATIVE_FPS) {
    payload.fps_boost = { target_fps: d.fps, mode: 'preserve_motion' }
  }
  const final = FINAL_BY_RESOLUTION[d.resolution as BerniniResolution]
  if (final) payload.upscale = { scale: 4, final }
  return payload
}

/** The delivery target the frontend will request for a Bernini video job. */
export interface BerniniDeliveryTarget {
  engine: BerniniEngine
  resolution: BerniniResolution
  fps: number
  /** Duration in seconds at native (post rails never extend duration). */
  duration: number
}

/** Rendered-form human label for a delivery target, e.g. "1080p · 24fps". */
export function berniniDeliveryLabel(d: BerniniDeliveryTarget): string {
  return `${d.resolution} · ${d.fps}fps`
}

/**
 * Compose the EXACT body to POST to the runner's `bernini-t2v` rail from a delivery
 * target. The runner renders natively at 480p@16fps; every above-native target carries
 * `post` (the fps_boost / upscale payload, byte-identical to the vp-worker /process body)
 * so the live-runner orchestrates the post chain after the render. Native -> no `post`.
 */
export function berniniRunnerT2VBody(
  prompt: string,
  target: BerniniDeliveryTarget,
  opts?: { negativePrompt?: string; seed?: number },
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    prompt,
    // The engine id the frontend advertises; the backend (build_pipeline) dispatches
    // on model_type to the BerniniRendererPipeline (1.3B) — see runner/idv2v.
    model: target.engine,
    // Native render request; above-native comes from the post rails, not the renderer.
    resolution: BERNINI_NATIVE_RESOLUTION,
    fps: BERNINI_NATIVE_FPS,
    num_frames: target.duration * BERNINI_NATIVE_FPS,
  }
  if (opts?.negativePrompt) body.negative_prompt = opts.negativePrompt
  if (opts?.seed !== undefined) body.seed = opts.seed
  const post = berniniPostFor(target)
  if (Object.keys(post).length > 0) {
    body.post = post
  }
  return body
}
