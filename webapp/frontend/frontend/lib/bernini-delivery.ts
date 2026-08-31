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

import { SSE_MAX_DURATION_MS } from "./sse-stream";

export type BerniniEngine = "1.3b" | "14b";

/** A delivery resolution tier (+ 'raw-4x' = native FlashVSR 4x, no ffmpeg downscale). */
export type BerniniResolution = "480p" | "720p" | "1080p" | "1440p" | "raw-4x";

/** The final ffmpeg encode target for the upscale rail. 'raw' skips ffmpeg entirely. */
export type UpscaleFinal = "720" | "1080" | "1440" | "raw";

export type FpsBoostMode = "preserve_motion" | "smooth";

/** Which post rail(s) a delivery target requires. */
export type PostRail = "rife" | "flashvsr" | "ffmpeg";

export interface BerniniDelivery {
  resolution: BerniniResolution;
  /** Delivery frame rate; > 16 requires the rife rail. */
  fps: number;
  /** True iff this IS the native render (480p @ 16). */
  native: boolean;
  /** Post rails required to reach this delivery from native. */
  post: PostRail[];
  /** Supported durations (seconds) at this delivery — calibrated on-box (Task 9). */
  durations: number[];
}

export interface BerniniRailsAvailable {
  rife: boolean;
  flashvsr: boolean;
}

export const BERNINI_NATIVE_FPS = 16;
/** Bernini's native frame count per clip (480p @ 16fps = ~5.06s). */
export const BERNINI_NATIVE_FRAMES = 81;

/** Long v2v window the BACKEND renders per chunk (frames). MUST stay aligned
 *  with runner/idv2v/bernini.py NATIVE_FRAMES (33): the manager splits source
 *  videos longer than this into consecutive non-overlapping windows, each its
 *  own native pass, then concatenates. The frontend uses this to size the SSE
 *  max-duration watchdog so a multi-chunk job isn't killed at the default
 *  25-min cap. */
export const BERNINI_V2V_CHUNK_FRAMES = 33;

/** How many backend v2v chunks a source of `totalFrames` frames splits into
 *  (used to scale the generation watchdog ceiling). */
export function berniniV2VChunkCount(totalFrames: number): number {
  if (!Number.isFinite(totalFrames) || totalFrames <= 0) return 1;
  return Math.max(1, Math.ceil(totalFrames / BERNINI_V2V_CHUNK_FRAMES));
}

/** SSE max-duration ceiling for a v2v generation of `totalFrames` frames. Each
 *  chunk is a near-native render (~2s of video, tens of seconds to minutes of
 *  wall time), so a length-N chunked job runs ~N times a single render. Scale
 *  the default 25-min cap by the chunk count so the frontend watchdog never
 *  cuts off a legitimately long job; the 90-s idle timeout still catches real
 *  stalls (silent-but-alive worker). */
export function berniniV2VMaxDurationMs(totalFrames: number): number {
  return SSE_MAX_DURATION_MS * berniniV2VChunkCount(totalFrames);
}
export const BERNINI_NATIVE_RESOLUTION: BerniniResolution = "480p";
/** Long edge the diffusers pipeline caps at (--max_image_size). */
export const BERNINI_MAX_IMAGE_SIZE = 848;

const RES_ORDER: BerniniResolution[] = [
  "480p",
  "720p",
  "1080p",
  "1440p",
  "raw-4x",
];
const FPS_OPTIONS: number[] = [16, 24, 30, 60];

// Native capacity by engine, in SECONDS at native. Post rails never extend duration.
// (Calibrated on-box in Task 9 — these are the reference-tuned defaults.)
const NATIVE_DURATIONS: Record<BerniniEngine, number[]> = {
  "1.3b": [2, 3, 5],
  "14b": [2, 3, 5],
};

const FINAL_BY_RESOLUTION: Partial<Record<BerniniResolution, UpscaleFinal>> = {
  "720p": "720",
  "1080p": "1080",
  "1440p": "1440",
  "raw-4x": "raw",
};

function postFor(resolution: BerniniResolution, fps: number): PostRail[] {
  const post: PostRail[] = [];
  if (fps > BERNINI_NATIVE_FPS) post.push("rife");
  if (resolution === "raw-4x") {
    post.push("flashvsr"); // native 4x encode — NO ffmpeg downscale
  } else if (resolution !== BERNINI_NATIVE_RESOLUTION) {
    post.push("flashvsr", "ffmpeg"); // 4x then lanczos-downscale to the final tier
  }
  return post;
}

/** All reachable deliveries for an engine, honoring which post rails are available. */
export function berniniDeliveryMatrix(
  engine: BerniniEngine,
  rails: BerniniRailsAvailable,
): BerniniDelivery[] {
  const out: BerniniDelivery[] = [];
  for (const resolution of RES_ORDER) {
    for (const fps of FPS_OPTIONS) {
      const post = postFor(resolution, fps);
      if (post.includes("rife") && !rails.rife) continue;
      if (post.includes("flashvsr") && !rails.flashvsr) continue;
      out.push({
        resolution,
        fps,
        native:
          resolution === BERNINI_NATIVE_RESOLUTION &&
          fps === BERNINI_NATIVE_FPS,
        post,
        durations: NATIVE_DURATIONS[engine],
      });
    }
  }
  return out;
}

/** Reachable fps values at a given resolution (feeds the fps dropdown). */
export function berniniFpsOptions(
  resolution: BerniniResolution,
  matrix: BerniniDelivery[],
): number[] {
  return matrix
    .filter((d) => d.resolution === resolution)
    .map((d) => d.fps)
    .sort((a, b) => a - b);
}

/** Reachable resolutions at a given fps (feeds the resolution dropdown). */
export function berniniResolutionOptions(
  fps: number,
  matrix: BerniniDelivery[],
): BerniniResolution[] {
  return RES_ORDER.filter((res) =>
    matrix.some((d) => d.fps === fps && d.resolution === res),
  );
}

export interface BerniniPostPayload {
  // Contract matches the vp-worker /process body: fps_boost.target_fps (renamed
  // from `target` so the frontend payload is byte-identical to the worker body).
  fps_boost?: { target_fps: number; mode: FpsBoostMode };
  upscale?: { scale: 4; final: UpscaleFinal };
}

/**
 * The post rails the frontend should attach for a chosen delivery target. Native
 * (480p @ 16) returns `{}` — no post. `fps > 16` adds the RIFE fps-boost (Preserve Motion
 * by default). Any non-native resolution adds the FlashVSR upscale with the right final.
 */
export function berniniPostFor(d: {
  resolution: string;
  fps: number;
}): BerniniPostPayload {
  const payload: BerniniPostPayload = {};
  if (d.fps > BERNINI_NATIVE_FPS) {
    payload.fps_boost = { target_fps: d.fps, mode: "preserve_motion" };
  }
  const final = FINAL_BY_RESOLUTION[d.resolution as BerniniResolution];
  if (final) payload.upscale = { scale: 4, final };
  return payload;
}

/** The delivery target the frontend will request for a Bernini video job. */
export interface BerniniDeliveryTarget {
  engine: BerniniEngine;
  resolution: BerniniResolution;
  fps: number;
  /** Duration in seconds at native (post rails never extend duration). */
  duration: number;
  /** Optional explicit native frame count override. When unset the render is the
   *  spec'd BERNINI_NATIVE_FRAMES (81) clip. v2v sets it to the SOURCE video's full
   *  frame count so the edit covers the entire input instead of only ~5s. */
  numFrames?: number;
}

/** Rendered-form human label for a delivery target, e.g. "1080p · 24fps". */
export function berniniDeliveryLabel(d: BerniniDeliveryTarget): string {
  return `${d.resolution} · ${d.fps}fps`;
}

/** A Bernini video operation (what the user asked for). */
export type BerniniOperation = "t2v" | "v2v" | "r2v";

export interface BerniniTaskSpec {
  /** Runner task string the browser POSTs to (endpointForTask -> /video-creator/v1/...). */
  task: string;
  /** The runner `capabilities[].id` a capable runner must advertise (resolveRunner). */
  capability: string;
}

/**
 * Map a Bernini operation to the runner task + capability the browser must target.
 * Mirrors the live-runner ROUTES / capability advertisement (wan-worker). This is the
 * whole "engine decides -> backend route-based" resolution the plan requires.
 */
export function berniniTaskFor(op: BerniniOperation): BerniniTaskSpec {
  const key = `bernini-${op}`;
  return { task: key, capability: key };
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
  opts?: {
    negativePrompt?: string;
    seed?: number;
    turbo?: boolean;
    /** Explicit native denoise step count (40 = full quality on the non-turbo
     *  path). When unset the renderer falls back to its own default. */
    numInferenceSteps?: number;
  },
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    prompt,
    // The engine id the frontend advertises; the backend (build_pipeline) dispatches
    // on model_type to the BerniniRendererPipeline (1.3B) — see runner/idv2v.
    model: target.engine,
    // Native render request; above-native comes from the post rails, not the renderer.
    resolution: BERNINI_NATIVE_RESOLUTION,
    fps: BERNINI_NATIVE_FPS,
    // Bernini's native output is the fixed 81-frame clip at 16fps (~5s); a caller
    // may override with target.numFrames (v2v renders the SOURCE video's full frame
    // count natively so the whole input is covered). Post rails never extend duration.
    num_frames: target.numFrames ?? BERNINI_NATIVE_FRAMES,
  };
  if (opts?.negativePrompt) body.negative_prompt = opts.negativePrompt;
  if (opts?.seed !== undefined) body.seed = opts.seed;
  // Explicit denoise step count (40 = full quality non-turbo). server.py forwards
  // num_inference_steps straight to the bernini_cli job, which honors it.
  if (opts?.numInferenceSteps !== undefined) {
    body.num_inference_steps = opts.numInferenceSteps;
  }
  // Turbo (14B only): ask the renderer for the 4-step LightX2V distill via the
  // rzgar LoRA instead of the default 40-step render. Unsure that the backend is
  // fp8 (14B); setting it on the 1.3B engine is a no-op there.
  if (opts?.turbo) body.turbo = true;
  const post = berniniPostFor(target);
  if (Object.keys(post).length > 0) {
    body.post = post;
  }
  return body;
}

/**
 * Motion-preserving Bernini video edit (v2v). Mirrors t2v body construction but adds
 * the source clip the worker edits in place (skip the FLUX.2 first-frame accept).
 */
export function berniniRunnerV2VBody(
  prompt: string,
  sourceVideo: string,
  target: BerniniDeliveryTarget,
  opts?: { negativePrompt?: string; seed?: number; turbo?: boolean; numInferenceSteps?: number },
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    ...berniniRunnerT2VBody(prompt, target, opts),
    // The Bernini worker's decode_source_media reads the source clip under `video` —
    // NOT `source_video` (which the older restyle/idv2v rail uses). v2v edits in place.
    video: sourceVideo,
  };
  return body;
}

/**
 * Reference-image → video (r2v, 1.3B). `references` are base64-encoded reference
 * images; the worker conditions the edit on them (multi-reference native to 1.3B).
 */
export function berniniRunnerR2VBody(
  prompt: string,
  references: string[],
  target: BerniniDeliveryTarget,
  opts?: { negativePrompt?: string; seed?: number; turbo?: boolean; numInferenceSteps?: number },
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    ...berniniRunnerT2VBody(prompt, target, opts),
    references,
  };
  return body;
}

/**
 * The vp-worker `/process` combined post body derived from a delivery target's rails.
 * Produced independently of the render payload so a caller can post-process an
 * EXISTING clip (Edit Video / standalone Process) without re-rendering.
 */
export interface BerniniProcessPayload {
  fps_boost?: { target_fps: number; mode: FpsBoostMode };
  upscale?: { scale: 4; final: UpscaleFinal };
}

export function berniniProcessFor(target: {
  resolution: BerniniResolution;
  fps: number;
}): BerniniProcessPayload {
  return berniniPostFor(target) as BerniniProcessPayload;
}
