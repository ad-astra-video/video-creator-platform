import { describe, it, expect } from 'vitest'
import { getVideoGenerationModelSpecs } from './video-generation-model-specs'

// Behavior contract: the runner now advertises ONE "bernini" pipeline whose
// spec.options lists the engines it serves (fast "1.3b" | detailed "14b").
// getVideoGenerationModelSpecs must re-materialize those options as distinct
// selectable pipelines (each pipeline id IS the engine, so settings.model is
// "1.3b"/"14b"), WITHOUT aliasing Bernini's limits onto the LTX "fast" matrix.
describe('Bernini pipeline -> options expansion', () => {
  const berniniCollapsed = {
    pipeline: 'bernini',
    spec: { display_name: 'Bernini', options: ['1.3b', '14b'] },
  }
  const fast = {
    pipeline: 'fast',
    spec: {
      display_name: 'LTX-2.3',
      supported_resolutions_durations: { '720p': { fps_to_durations: { '24': [4, 6, 8] } } },
    },
  }

  it('expands a bernini pipeline into one option per engine', () => {
    const out = getVideoGenerationModelSpecs(
      { api_models: [], local_models: [berniniCollapsed, fast] as never[] },
      { useApiSpecs: false },
    )
    const bernini = out.filter((m) => (m.pipeline as string) === '1.3b' || (m.pipeline as string) === '14b')
    expect(bernini.map((m) => m.pipeline)).toEqual(['1.3b', '14b'])
    expect(bernini.map((m) => (m.spec as { display_name: string }).display_name)).toEqual([
      'Bernini 1.3B',
      'Bernini 14B',
    ])
  })

  it('does not alias Bernini onto the LTX fast matrix', () => {
    const out = getVideoGenerationModelSpecs(
      { api_models: [], local_models: [berniniCollapsed, fast] as never[] },
      { useApiSpecs: false },
    )
    const bern = out.find((m) => (m.pipeline as string) === '1.3b')
    // Bernini stays marker-only (no LTX resolution/fps matrix inherited).
    expect((bern?.spec as { supported_resolutions_durations?: unknown }).supported_resolutions_durations).toBeUndefined()
  })

  it('keeps a bare bernini pipeline as a marker when it advertises no options', () => {
    const out = getVideoGenerationModelSpecs(
      { api_models: [], local_models: [{ pipeline: 'bernini', spec: { display_name: 'Bernini' } }, fast] as never[] },
      { useApiSpecs: false },
    )
    const bern = out.filter((m) => (m.pipeline as string) === 'bernini')
    expect(bern).toHaveLength(1)
    expect(bern[0].pipeline).toBe('bernini')
  })
})
