import { describe, expect, it } from 'vitest'
import {
  RUNNER_MODEL_FOOTPRINT_MB,
  largestModelFootprintMb,
  estimateRunnerCapacity,
  runnerCapacity,
} from './runner-availability'

describe('largestModelFootprintMb', () => {
  it('returns the largest known footprint for recognized model ids', () => {
    expect(largestModelFootprintMb(['ltx', 'z-image'])).toBe(RUNNER_MODEL_FOOTPRINT_MB.ltx)
    expect(largestModelFootprintMb(['z-image', 'ltx'])).toBe(RUNNER_MODEL_FOOTPRINT_MB.ltx)
    expect(largestModelFootprintMb(['qwen-edit', 'klein'])).toBe(RUNNER_MODEL_FOOTPRINT_MB['qwen-edit'])
  })

  it('is case/whitespace tolerant', () => {
    expect(largestModelFootprintMb(['  LTX '])).toBe(RUNNER_MODEL_FOOTPRINT_MB.ltx)
  })

  it('returns null when no model is recognized', () => {
    expect(largestModelFootprintMb(['unknown-model', 'foo'])).toBeNull()
    expect(largestModelFootprintMb([])).toBeNull()
  })
})

describe('runnerCapacity', () => {
  it('prefers the advertised capacity_available over any estimate', () => {
    expect(runnerCapacity({ capacity: 3, capacityAvailable: 3, vramMb: 24576, modelIds: ['z-image'] })).toBe(3)
  })

  it('reflects current usage via capacity_available', () => {
    expect(runnerCapacity({ capacity: 3, capacityAvailable: 1, vramMb: 24576, modelIds: ['z-image'] })).toBe(1)
  })

  it('falls back to raw capacity when available is absent', () => {
    expect(runnerCapacity({ capacity: 3, vramMb: 24576, modelIds: ['z-image'] })).toBe(3)
  })

  it('falls back to the VRAM estimate only when no capacity is advertised', () => {
    expect(runnerCapacity({ vramMb: 24576, modelIds: ['z-image'] })).toBe(1)
    expect(runnerCapacity({ vramMb: 24576, modelIds: ['unknown'] })).toBeNull()
    expect(runnerCapacity({})).toBeNull()
  })
})

describe('estimateRunnerCapacity', () => {
  it('computes concurrent generations from VRAM and the largest model footprint', () => {
    // 24 GB (24576 MiB) card, smallest-known model (13312 MiB z-image): usable = 0.9*24576 = 22118,
    // floor(22118 / 13312) = 1.
    expect(estimateRunnerCapacity(24576, ['z-image'])).toBe(1)
    // 2x that VRAM -> 3 concurrent (44237 usable / 13312 = 3.3, floor 3).
    expect(estimateRunnerCapacity(49152, ['z-image'])).toBe(3)
  })

  it('uses the LARGEST advertised model as the worst case', () => {
    // Mixing a cheap and an expensive model: capacity is driven by the expensive one.
    expect(estimateRunnerCapacity(24576, ['z-image'])).toBe(1)
    expect(estimateRunnerCapacity(24576, ['z-image', 'qwen-edit'])).toBeNull() // qwen (30GB) doesn't fit 24GB
    expect(estimateRunnerCapacity(36000, ['z-image', 'qwen-edit'])).toBe(1)
  })

  it('returns null when VRAM is unavailable or no footprint matches (no fabrication)', () => {
    expect(estimateRunnerCapacity(undefined, ['ltx'])).toBeNull()
    expect(estimateRunnerCapacity(0, ['ltx'])).toBeNull()
    expect(estimateRunnerCapacity(24576, ['unknown'])).toBeNull()
  })
})
