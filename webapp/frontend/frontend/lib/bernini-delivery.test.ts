// Unit tests for the pure Bernini delivery-matrix derivation (node vitest).
import { describe, it, expect } from 'vitest'
import {
  berniniDeliveryMatrix,
  berniniFpsOptions,
  berniniResolutionOptions,
  berniniPostFor,
  BERNINI_NATIVE_FPS,
  BERNINI_NATIVE_RESOLUTION,
} from './bernini-delivery'

describe('berniniDeliveryMatrix', () => {
  it('has exactly one native (480p @ 16) delivery requiring no post', () => {
    const m = berniniDeliveryMatrix('bernini-1.3b', { rife: true, flashvsr: true })
    const native = m.filter((d) => d.native)
    expect(native).toHaveLength(1)
    expect(native[0]).toMatchObject({
      resolution: BERNINI_NATIVE_RESOLUTION,
      fps: BERNINI_NATIVE_FPS,
      post: [],
    })
  })

  it('hides rife-dependent rows when the rife rail is disabled', () => {
    const withRife = berniniDeliveryMatrix('bernini-1.3b', { rife: true, flashvsr: true })
    const noRife = berniniDeliveryMatrix('bernini-1.3b', { rife: false, flashvsr: true })
    expect(noRife.some((d) => d.fps > BERNINI_NATIVE_FPS)).toBe(false)
    expect(withRife.some((d) => d.fps > BERNINI_NATIVE_FPS)).toBe(true)
  })

  it('hides flashvsr-dependent rows when the vsr rail is disabled', () => {
    const noVsr = berniniDeliveryMatrix('bernini-1.3b', { rife: true, flashvsr: false })
    expect(noVsr.every((d) => d.resolution === BERNINI_NATIVE_RESOLUTION)).toBe(true)
  })

  it('raw-4x requires flashvsr but no ffmpeg', () => {
    const m = berniniDeliveryMatrix('bernini-1.3b', { rife: true, flashvsr: true })
    const raw = m.find((d) => d.resolution === 'raw-4x' && d.fps === 16)
    expect(raw?.post).toEqual(['flashvsr'])
  })

  it('1080p at high fps chains rife + flashvsr + ffmpeg', () => {
    const m = berniniDeliveryMatrix('bernini-14b', { rife: true, flashvsr: true })
    const row = m.find((d) => d.resolution === '1080p' && d.fps === 60)
    expect(row?.post).toEqual(['rife', 'flashvsr', 'ffmpeg'])
  })
})

describe('dropdown derivation', () => {
  const matrix = berniniDeliveryMatrix('bernini-14b', { rife: true, flashvsr: true })
  it('fps options at 480p include native 16 plus rife-derived', () => {
    expect(berniniFpsOptions('480p', matrix)).toEqual([16, 24, 30, 60])
  })
  it('resolution options at 16fps include all tiers', () => {
    expect(berniniResolutionOptions(16, matrix)).toEqual(['480p', '1080p', '1440p', 'raw-4x'])
  })
})

describe('berniniPostFor', () => {
  it('native delivery attaches no post', () => {
    expect(berniniPostFor({ resolution: '480p', fps: 16 })).toEqual({})
  })
  it('fps-only boost sets preserve_motion target', () => {
    expect(berniniPostFor({ resolution: '480p', fps: 30 })).toEqual({
      fps_boost: { target: 30, mode: 'preserve_motion' },
    })
  })
  it('raw-4x attaches upscale with final raw and no ffmpeg requirement', () => {
    expect(berniniPostFor({ resolution: 'raw-4x', fps: 16 })).toEqual({
      upscale: { scale: 4, final: 'raw' },
    })
  })
  it('1080p @ 30 combines fps_boost + upscale final 1080', () => {
    expect(berniniPostFor({ resolution: '1080p', fps: 30 })).toEqual({
      fps_boost: { target: 30, mode: 'preserve_motion' },
      upscale: { scale: 4, final: '1080' },
    })
  })
})
