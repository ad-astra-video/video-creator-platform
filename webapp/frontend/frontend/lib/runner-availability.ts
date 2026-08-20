/**
 * Pure, testable runner-availability detection for the webapp.
 *
 * The webapp's Generate / Regenerate buttons should be disabled ONLY when there is no
 * capable runner available (and Livepeer is enabled). This module centralises that check so
 * the button-disable rule is one source of truth: given the set of discovered runners and the
 * required capabilities, which are actually usable (ready, not excluded, fetchable)?
 *
 * It is intentionally framework-free so it can be unit-tested with vitest (the frontend has a
 * vitest setup for pure dep-free lib modules, see the video-creator-platform-dev skill).
 */

export interface RunnerAvailabilityRunner {
  url: string
  runner_id: string
  status: string
  excluded: boolean
  capabilities?: Array<{ id: string }>
}

/**
 * True when the given capabilities are all served by `runner` AND the runner is actually usable:
 * ready, not excluded by the user, and its URL is fetchable by the browser (so a job could
 * actually be dispatched to it). `requiredCaps` empty means "any capable runner is fine".
 */
export function isRunnerCapable(
  runner: RunnerAvailabilityRunner,
  requiredCaps: string[],
  isFetchable: (url: string) => boolean = (url) => url.length > 0,
): boolean {
  if (runner.status !== 'ready') return false
  if (runner.excluded) return false
  if (!isFetchable(runner.url)) return false
  if (requiredCaps.length === 0) return true
  const ids = new Set((runner.capabilities ?? []).map((c) => c.id))
  return requiredCaps.every((c) => ids.has(c))
}

/** How many of `runners` can serve every capability in `requiredCaps`. 0 = no runner available. */
export function countCapableRunners(
  runners: RunnerAvailabilityRunner[],
  requiredCaps: string[],
  isFetchable?: (url: string) => boolean,
): number {
  return runners.reduce((count, runner) => (
    isRunnerCapable(runner, requiredCaps, isFetchable) ? count + 1 : count
  ), 0)
}

// ---------------------------------------------------------------------------
// Possible-capacity estimation (pure, best-effort).
//
// A runner advertises its GPU VRAM (`gpu.vram_mb`) and the model(s) it can run. From those we
// can estimate how many concurrent generations it could host in VRAM. This is an ESTIMATE only:
// it assumes a single device budget and that the whole model must be resident at once, both of
// which the real runner may not match (offloading, multiple GPUs, per-request streaming). It is
// intentionally conservative and always labeled "(est.)" in the UI — never presented as fact.
// ---------------------------------------------------------------------------

/**
 * Approximate resident-VRAM footprint (in MiB) for the remote generated-video/image models the
 * app can send to a runner. Kept here (not inferred) because the runner only advertises model id
 * strings, not their memory cost. Values are rough 2026 figures from this stack's own GPU
 * measurements and should be tuned/expanded as reality changes. Unknown ids return null so the
 * caller can fall back to showing raw VRAM instead of a fabricated number.
 */
export const RUNNER_MODEL_FOOTPRINT_MB: Record<string, number> = {
  ltx: 20480, // LTX-2 video, ~bf16/fp8 resident
  'ltx-2b': 20480,
  'ltx-2b-video': 20480,
  hidream: 18432, // hidream 8B image
  'qwen-edit': 30720, // Qwen-Image-Edit fp8
  'qwen-image-edit': 30720,
  'z-image': 13312, // Z-Image turbo
  'zimage': 13312,
  klein: 13312, // FLUX.2 klein 4B
  'flux-klein': 13312,
  'flux.2-klein': 13312,
}

const VRAM_OVERHEAD_RATIO = 0.9 // leave headroom for activations / KV / transient frames

/** Largest known footprint among `modelIds`, in MiB. null when none are known. */
export function largestModelFootprintMb(modelIds: string[]): number | null {
  let max: number | null = null
  for (const m of modelIds) {
    const f = RUNNER_MODEL_FOOTPRINT_MB[m.trim().toLowerCase()]
    if (f != null && (max == null || f > max)) max = f
  }
  return max
}

/**
 * Estimate how many concurrent generations `vramMb` could host for the given `modelIds`, using
 * the LARGEST advertised model as the worst case. Returns null when VRAM is unknown or no known
 * model footprint matched (caller should then show raw VRAM only, not a guess).
 */
export function estimateRunnerCapacity(vramMb: number | undefined, modelIds: string[]): number | null {
  if (vramMb == null || vramMb <= 0) return null
  const largest = largestModelFootprintMb(modelIds)
  if (largest == null || largest <= 0) return null
  const usable = vramMb * VRAM_OVERHEAD_RATIO
  const n = Math.floor(usable / largest)
  return n >= 1 ? n : null
}
