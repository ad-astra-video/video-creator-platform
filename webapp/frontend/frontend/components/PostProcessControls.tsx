import type { BerniniProcessPayload, UpscaleFinal, FpsBoostMode } from '../lib/bernini-delivery'

// Shared post-process rail controls (RIFE fps-boost + FlashVSR upscale), reused by
// Generate Videos, Edit Video, and standalone "Process clip" → vp-worker /process.
// Deliberately CONTROLLED (no internal state): the parent owns the payload and each
// toggle maps 1:1 to the vp-worker /process body fields (fps_boost.target_fps/mode,
// upscale.scale/final), honoring the user's "toggle switches, no un-X" preference.

const FPS_OPTIONS = [16, 24, 30, 60]

interface PostProcessControlsProps {
  value: BerniniProcessPayload
  onChange: (next: BerniniProcessPayload) => void
  disabled?: boolean
}

export function PostProcessControls({ value, onChange, disabled = false }: PostProcessControlsProps) {
  const fps = value.fps_boost?.target_fps
  const upscale = value.upscale
  const mode: FpsBoostMode = value.fps_boost?.mode ?? 'preserve_motion'

  const setFpsBoost = (on: boolean, target = 24) => {
    if (!on) {
      const { fps_boost: _drop, ...rest } = value
      onChange(rest)
      return
    }
    onChange({ ...value, fps_boost: { target_fps: target === fps ? target : fps ?? target, mode } })
  }

  const setUpscale = (on: boolean, final: UpscaleFinal = '1080') => {
    if (!on) {
      const { upscale: _drop, ...rest } = value
      onChange(rest)
      return
    }
    onChange({ ...value, upscale: { scale: 4, final } })
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        {/* FPS Boost (RIFE) */}
        <label className="flex items-center gap-1.5 cursor-pointer select-none text-[11px] text-zinc-300">
          <input
            type="checkbox"
            checked={fps !== undefined}
            disabled={disabled}
            onChange={(e) => setFpsBoost(e.target.checked)}
            className="accent-emerald-500"
          />
          FPS Boost (RIFE)
        </label>
        {fps !== undefined && (
          <select
            value={fps}
            disabled={disabled}
            onChange={(e) => onChange({ ...value, fps_boost: { target_fps: Number(e.target.value), mode } })}
            className="bg-zinc-800 border border-zinc-700 rounded-md px-2 py-1 text-[11px] text-white"
          >
            {FPS_OPTIONS.map((f) => (
              <option key={f} value={f}>{f}fps</option>
            ))}
          </select>
        )}
        {fps !== undefined && (
          <label className="flex items-center gap-1.5 cursor-pointer select-none text-[11px] text-zinc-400">
            <input
              type="checkbox"
              checked={mode === 'smooth'}
              disabled={disabled}
              onChange={(e) => onChange({
                ...value,
                fps_boost: { target_fps: fps, mode: e.target.checked ? 'smooth' : 'preserve_motion' },
              })}
              className="accent-emerald-500"
            />
            Smooth (preserve-motion off)
          </label>
        )}

        {/* Upscale (FlashVSR) */}
        <label className="flex items-center gap-1.5 cursor-pointer select-none text-[11px] text-zinc-300">
          <input
            type="checkbox"
            checked={upscale !== undefined}
            disabled={disabled}
            onChange={(e) => setUpscale(e.target.checked)}
            className="accent-emerald-500"
          />
          Upscale (FlashVSR)
        </label>
        {upscale !== undefined && (
          <select
            value={upscale.final}
            disabled={disabled}
            onChange={(e) => onChange({ ...value, upscale: { scale: 4, final: e.target.value as UpscaleFinal } })}
            className="bg-zinc-800 border border-zinc-700 rounded-md px-2 py-1 text-[11px] text-white"
          >
            <option value="1080">1080p</option>
            <option value="1440">1440p</option>
            <option value="raw">4x native</option>
          </select>
        )}
      </div>
    </div>
  )
}
