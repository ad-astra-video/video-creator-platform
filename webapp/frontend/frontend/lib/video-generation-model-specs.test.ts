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

  it('gives each engine its own 16fps Bernini matrix (not the LTX alias), filtered per engine', () => {
    const out = getVideoGenerationModelSpecs(
      { api_models: [], local_models: [berniniCollapsed, fast] as never[] },
      { useApiSpecs: false },
    )
    const one = out.find((m) => (m.pipeline as string) === '1.3b')
    const fourteen = out.find((m) => (m.pipeline as string) === '14b')
    const s1 = (one?.spec as { supported_resolutions_durations?: Record<string, { fps_to_durations?: Record<string, number[]> }> }).supported_resolutions_durations ?? {}
    const s14 = (fourteen?.spec as { supported_resolutions_durations?: Record<string, { fps_to_durations?: Record<string, number[]> }> }).supported_resolutions_durations ?? {}
    // Bernini's own limits on 16fps (its native render fps), NOT LTX's 24fps band.
    expect(Object.keys(s1)).toEqual(['480p'])
    expect(s1['480p']?.fps_to_durations?.['16']).toEqual([5])
    // 14b -> 480p + 720p.
    expect(Object.keys(s14)).toEqual(['480p', '720p'])
    // It never inherits the LTX fast matrix (''720p' @ 24fps).
    expect(s14['720p']?.fps_to_durations?.['24']).toBeUndefined()
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
