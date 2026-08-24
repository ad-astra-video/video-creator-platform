// Best-effort browser probe for a video asset's frame rate, used by the Edit-Video
// (Bernini v2v) rail so an edited clip can default to matching its source's fps.
//
// There is no `video.fps` property in the DOM, so we estimate it by rendering the
// source into an offscreen (muted) <video> and measuring `mediaTime` advances across
// consecutive requestVideoFrameCallback presentations. For CFR sources this is the
// true frame cadence; for VFR it yields the average. DOM-only (never imported by the
// node-environment vitest suite). Pure best-effort: every failure path returns null,
// never throws, and the caller falls back to the native render rate.

type RvfcCallback = (now: number, meta: { mediaTime: number }) => void

interface RvfcVideo {
  requestVideoFrameCallback: (cb: RvfcCallback) => void
}

/** Common target frame rates to snap to so 29.97 -> 30, 23.976 -> 24, etc. */
const COMMON_FPS = [24, 25, 30, 48, 50, 60]

function snapFps(raw: number): number {
  for (const nice of COMMON_FPS) {
    if (Math.abs(raw - nice) <= 2) return nice
  }
  return Math.max(1, Math.min(120, Math.round(raw)))
}

/**
 * Estimate the frame rate of a playable video URL (blob:, http(s)://, or data:).
 * Resolves to the average fps when it can be measured, or null when the source isn't
 * a decodable video, requestVideoFrameCallback is unavailable, or sampling fails.
 */
export async function measureVideoFps(url: string, samples = 14): Promise<number | null> {
  if (typeof document === 'undefined' || typeof HTMLVideoElement === 'undefined') return null
  const proto = HTMLVideoElement.prototype as unknown as Partial<RvfcVideo>
  if (typeof proto.requestVideoFrameCallback !== 'function') return null

  const v = document.createElement('video')
  v.muted = true
  v.playsInline = true
  v.preload = 'auto'
  v.src = url

  try {
    await new Promise<void>((resolve, reject) => {
      v.onloadeddata = () => resolve()
      v.onerror = () => reject(new Error('video load failed'))
    })
    await v.play()
  } catch {
    return null
  }

  try {
    const times = (v as unknown as RvfcVideo)
    const mediaTimes: number[] = await new Promise<number[]>((resolve) => {
      const collected: number[] = []
      const cb: RvfcCallback = (_now, meta) => {
        collected.push(meta.mediaTime)
        if (collected.length >= samples) {
          resolve(collected)
          return
        }
        times.requestVideoFrameCallback(cb)
      }
      times.requestVideoFrameCallback(cb)
    })

    if (mediaTimes.length < 3) return null
    const deltas = mediaTimes
      .slice(1)
      .map((t, i) => t - mediaTimes[i])
      .filter((d) => d > 0 && d < 1.5) // drop seeks / stalls
    if (deltas.length < 2) return null
    const avg = deltas.reduce((sum, d) => sum + d, 0) / deltas.length
    if (!(avg > 0)) return null
    return snapFps(1 / avg)
  } finally {
    v.pause()
    v.removeAttribute('src')
    v.load()
  }
}
