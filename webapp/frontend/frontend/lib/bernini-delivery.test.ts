// Unit tests for the pure Bernini delivery-matrix derivation (node vitest).
import { describe, it, expect } from 'vitest'
import {
  berniniDeliveryMatrix,
  berniniFpsOptions,
  berniniResolutionOptions,
  berniniPostFor,
  berniniRunnerT2VBody,
  berniniRunnerV2VBody,
  berniniRunnerR2VBody,
  berniniProcessFor,
  berniniDeliveryLabel,
  berniniTaskFor,
  BERNINI_NATIVE_FPS,
  BERNINI_NATIVE_RESOLUTION,
  type BerniniDeliveryTarget,
} from './bernini-delivery'

describe('berniniDeliveryMatrix', () => {
  it('has exactly one native (480p @ 16) delivery requiring no post', () => {
    const m = berniniDeliveryMatrix('1.3b', { rife: true, flashvsr: true })
    const native = m.filter((d) => d.native)
    expect(native).toHaveLength(1)
    expect(native[0]).toMatchObject({
      resolution: BERNINI_NATIVE_RESOLUTION,
      fps: BERNINI_NATIVE_FPS,
      post: [],
    })
  })

  it('hides rife-dependent rows when the rife rail is disabled', () => {
    const withRife = berniniDeliveryMatrix('1.3b', { rife: true, flashvsr: true })
    const noRife = berniniDeliveryMatrix('1.3b', { rife: false, flashvsr: true })
    expect(noRife.some((d) => d.fps > BERNINI_NATIVE_FPS)).toBe(false)
    expect(withRife.some((d) => d.fps > BERNINI_NATIVE_FPS)).toBe(true)
  })

  it('hides flashvsr-dependent rows when the vsr rail is disabled', () => {
    const noVsr = berniniDeliveryMatrix('1.3b', { rife: true, flashvsr: false })
    expect(noVsr.every((d) => d.resolution === BERNINI_NATIVE_RESOLUTION)).toBe(true)
  })

  it('raw-4x requires flashvsr but no ffmpeg', () => {
    const m = berniniDeliveryMatrix('1.3b', { rife: true, flashvsr: true })
    const raw = m.find((d) => d.resolution === 'raw-4x' && d.fps === 16)
    expect(raw?.post).toEqual(['flashvsr'])
  })

  it('1080p at high fps chains rife + flashvsr + ffmpeg', () => {
    const m = berniniDeliveryMatrix('14b', { rife: true, flashvsr: true })
    const row = m.find((d) => d.resolution === '1080p' && d.fps === 60)
    expect(row?.post).toEqual(['rife', 'flashvsr', 'ffmpeg'])
  })
})

describe('dropdown derivation', () => {
  const matrix = berniniDeliveryMatrix('14b', { rife: true, flashvsr: true })
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
      fps_boost: { target_fps: 30, mode: 'preserve_motion' },
    })
  })
  it('raw-4x attaches upscale with final raw and no ffmpeg requirement', () => {
    expect(berniniPostFor({ resolution: 'raw-4x', fps: 16 })).toEqual({
      upscale: { scale: 4, final: 'raw' },
    })
  })
  it('1080p @ 30 combines fps_boost + upscale final 1080', () => {
    expect(berniniPostFor({ resolution: '1080p', fps: 30 })).toEqual({
      fps_boost: { target_fps: 30, mode: 'preserve_motion' },
      upscale: { scale: 4, final: '1080' },
    })
  })
})

describe('berniniDeliveryLabel', () => {
  it('renders a compact resolution-fps label', () => {
    expect(berniniDeliveryLabel({ engine: '1.3b', resolution: '1080p', fps: 24, duration: 3 }))
      .toBe('1080p · 24fps')
  })
})

describe('berniniRunnerT2VBody', () => {
  const base: BerniniDeliveryTarget = {
    engine: '1.3b',
    resolution: '480p',
    fps: 16,
    duration: 3,
  }

  it('native target: renders at native 480p@16 with no post', () => {
    const body = berniniRunnerT2VBody('a red fox', base)
    expect(body).toMatchObject({
      prompt: 'a red fox',
      model: '1.3b',
      resolution: '480p',
      fps: 16,
      num_frames: 48,
    })
    expect(body.post).toBeUndefined()
  })

  it('above-native target: carries the delivery post payload', () => {
    const body = berniniRunnerT2VBody('a red fox', {
      ...base,
      resolution: '1080p',
      fps: 30,
    })
    expect(body.post).toEqual({
      fps_boost: { target_fps: 30, mode: 'preserve_motion' },
      upscale: { scale: 4, final: '1080' },
    })
  })

  it('raw-4x upscales 4x with final raw and no ffmpeg', () => {
    const body = berniniRunnerT2VBody('a red fox', { ...base, resolution: 'raw-4x' })
    expect(body.post).toEqual({ upscale: { scale: 4, final: 'raw' } })
  })

  it('honors negative prompt and seed when provided', () => {
    const body = berniniRunnerT2VBody('a red fox', base, { negativePrompt: 'blurry', seed: 42 })
    expect(body.negative_prompt).toBe('blurry')
    expect(body.seed).toBe(42)
  })

  it('computes num_frames from duration at native fps', () => {
    expect(berniniRunnerT2VBody('x', { ...base, duration: 5 }).num_frames).toBe(80)
  })
})

describe('berniniTaskFor', () => {
  it('maps each operation to its bernini-* runner task and capability', () => {
    expect(berniniTaskFor('t2v')).toEqual({ task: 'bernini-t2v', capability: 'bernini-t2v' })
    expect(berniniTaskFor('v2v')).toEqual({ task: 'bernini-v2v', capability: 'bernini-v2v' })
    expect(berniniTaskFor('r2v')).toEqual({ task: 'bernini-r2v', capability: 'bernini-r2v' })
  })
})

describe('berniniRunnerV2VBody', () => {
  const base: BerniniDeliveryTarget = { engine: '1.3b', resolution: '480p', fps: 16, duration: 3 }

  it('sends the source clip under `video` (Bernini worker contract)', () => {
    const body = berniniRunnerV2VBody('make it sunset', 'AAABase64VideoAA', base)
    expect(body.video).toBe('AAABase64VideoAA')
    expect(body.resolution).toBe('480p')
    expect(body.fps).toBe(16)
    expect(body.post).toBeUndefined()
  })

  it('carries the post payload for above-native delivery', () => {
    const body = berniniRunnerV2VBody('make it sunset', 'AAABase64VideoAA', { ...base, resolution: '1080p', fps: 30 })
    expect(body.post).toEqual({
      fps_boost: { target_fps: 30, mode: 'preserve_motion' },
      upscale: { scale: 4, final: '1080' },
    })
  })
})

describe('berniniRunnerR2VBody', () => {
  const base: BerniniDeliveryTarget = { engine: '1.3b', resolution: '480p', fps: 16, duration: 3 }

  it('attaches reference images under references[]', () => {
    const refs = ['ref1base64', 'ref2base64']
    const body = berniniRunnerR2VBody('compose around these', refs, base)
    expect(body.references).toEqual(refs)
  })

  it('still renders natively and honors seed', () => {
    const body = berniniRunnerR2VBody('compose', ['ref'], { ...base, duration: 5 }, { seed: 7 })
    expect(body.references).toEqual(['ref'])
    expect(body.num_frames).toBe(80)
    expect(body.seed).toBe(7)
  })
})

describe('berniniProcessFor', () => {
  it('native delivery => empty process payload', () => {
    expect(berniniProcessFor({ resolution: '480p', fps: 16 })).toEqual({})
  })
  it('derives the vp-worker /process post body from a delivery target', () => {
    expect(berniniProcessFor({ resolution: '1440p', fps: 60 })).toEqual({
      fps_boost: { target_fps: 60, mode: 'preserve_motion' },
      upscale: { scale: 4, final: '1440' },
    })
  })
  it('raw-4x => upscale with final raw only', () => {
    expect(berniniProcessFor({ resolution: 'raw-4x', fps: 16 })).toEqual({
      upscale: { scale: 4, final: 'raw' },
    })
  })
})
