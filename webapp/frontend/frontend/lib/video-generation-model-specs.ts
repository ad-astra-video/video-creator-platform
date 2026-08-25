import type { components } from '../generated/backend-openapi'

export type VideoGenerationModelSpecsResponse = components['schemas']['GenerateVideoModelsSpecsResponse']
export type VideoGenerationModelSpecItem = components['schemas']['LTXVideoGenerationModelSpecItem']
export type VideoGenerationResolutionSpec = components['schemas']['LTXVideoGenerationResolutionSpec']
export type VideoGenerationPipeline = components['schemas']['GenerateVideoRequest']['model'] | 'ltx-2.5'
export type VideoGenerationResolution = components['schemas']['GenerateVideoRequest']['resolution']
export type VideoGenerationDuration = components['schemas']['GenerateVideoRequest']['duration']
export type VideoGenerationFps = components['schemas']['GenerateVideoRequest']['fps']
export type VideoGenerationAspectRatio = components['schemas']['GenerateVideoRequest']['aspectRatio']

// Resolution-aware extend ceiling advertised by the runner's model-spec metadata (see
// runner/live_runner/specs.py -- build_extend_capability). The runner is the authority on how
// many "seconds to add" it can actually run at each output resolution on its own GPU.
export interface ExtendCapability {
  context_window_seconds?: number
  min_duration_seconds?: number
  max_duration_seconds?: Record<string, number> // resolution key (e.g. "540p") -> max seconds
}

export function getExtendCapability(item: VideoGenerationModelSpecItem): ExtendCapability | null {
  const ext = (item.spec as { extend?: ExtendCapability }).extend
  return ext && ext.max_duration_seconds ? ext : null
}

export interface VideoGenerationSettingsShape {
  model: string
  duration: number
  videoResolution: string
  fps: number
  aspectRatio?: string
  audio?: boolean
}

export interface ResolvedVideoGenerationOptions {
  modelOptions: VideoGenerationModelSpecItem[]
  resolutionOptions: VideoGenerationResolution[]
  fpsOptions: VideoGenerationFps[]
  durationOptions: VideoGenerationDuration[]
  selectedModel: VideoGenerationPipeline | null
  selectedResolution: VideoGenerationResolution | null
  selectedFps: VideoGenerationFps | null
  selectedDuration: VideoGenerationDuration | null
  hasCompatibleOptions: boolean
}

type DurationSelectionMode = 'preserve' | 'smallest_valid'

interface ResolveVideoGenerationOptionsParams<T extends VideoGenerationSettingsShape> {
  settings: T
  modelSpecs: VideoGenerationModelSpecItem[]
  hasAudio?: boolean
  minimumDuration?: number
  durationSelection?: DurationSelectionMode
}

function getResolutionMap(
  item: VideoGenerationModelSpecItem,
  options: { hasAudio: boolean },
): Record<string, VideoGenerationResolutionSpec> {
  const { hasAudio } = options
  if (hasAudio && item.spec.a2v_supported_resolutions_durations) {
    return item.spec.a2v_supported_resolutions_durations
  }
  return item.spec.supported_resolutions_durations
}

function getResolutionEntries(
  item: VideoGenerationModelSpecItem,
  options: { hasAudio: boolean },
): Array<[VideoGenerationResolution, VideoGenerationResolutionSpec]> {
  return Object.entries(getResolutionMap(item, options)).map(([resolution, spec]) => [
    resolution as VideoGenerationResolution,
    spec,
  ])
}

function getDurationsForFps(
  resolutionSpec: VideoGenerationResolutionSpec,
  fps: VideoGenerationFps,
): VideoGenerationDuration[] {
  return (resolutionSpec.fps_to_durations[String(fps)] ?? []) as VideoGenerationDuration[]
}

function filterDurationsByMinimum(
  durations: VideoGenerationDuration[],
  minimumDuration: number | undefined,
): VideoGenerationDuration[] {
  if (minimumDuration === undefined) return durations
  return durations.filter((duration) => duration >= minimumDuration)
}

function getCompatibleFps(
  resolutionSpec: VideoGenerationResolutionSpec,
  options: { minimumDuration: number | undefined },
): VideoGenerationFps[] {
  const { minimumDuration } = options
  return Object.keys(resolutionSpec.fps_to_durations).map((fps) => Number(fps) as VideoGenerationFps).filter((fps) => (
    filterDurationsByMinimum(getDurationsForFps(resolutionSpec, fps), minimumDuration).length > 0
  ))
}

function getCompatibleResolutionEntries(
  item: VideoGenerationModelSpecItem,
  options: { hasAudio: boolean; minimumDuration: number | undefined },
): Array<[VideoGenerationResolution, VideoGenerationResolutionSpec]> {
  return getResolutionEntries(item, { hasAudio: options.hasAudio }).filter(([, resolutionSpec]) => (
    getCompatibleFps(resolutionSpec, { minimumDuration: options.minimumDuration }).length > 0
  ))
}

function getCompatibleModelOptions(
  modelSpecs: VideoGenerationModelSpecItem[],
  options: { hasAudio: boolean; minimumDuration: number | undefined },
): VideoGenerationModelSpecItem[] {
  const { hasAudio, minimumDuration } = options
  if (minimumDuration === undefined) return modelSpecs
  return modelSpecs.filter((item) => (
    getCompatibleResolutionEntries(item, { hasAudio, minimumDuration }).length > 0
  ))
}

function chooseOption<T>(current: string | number, options: T[]): T | null {
  return options.find((option) => option === current) ?? options[0] ?? null
}

export function getVideoGenerationModelSpecs(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  options: { useApiSpecs: boolean },
): VideoGenerationModelSpecItem[] {
  const { useApiSpecs } = options
  if (!specs) return []
  const list = useApiSpecs ? specs.api_models : specs.local_models
  return expandBerniniOptions(inheritLimitsFromFast(list))
}

/** The picker label for a Bernini engine option (fast 1.3B | detailed 14B). */
const BERNINI_OPTION_LABELS: Record<string, string> = {
  '1.3b': 'Bernini 1.3B',
  '14b': 'Bernini 14B',
}

/**
 * Expand the runner's single "bernini" pipeline (spec.options = engine ids) into
 * one selectable model per engine option. The runner now advertises ONE "bernini"
 * entry whose options list serves fast (1.3B) and detailed (14B); here we
 * re-materialize them as distinct pipelines — each pipeline id IS the engine
 * (settings.model = "1.3b"/"14b"), so the model picker offers both and the
 * generate/edit paths can filter runners by the chosen option. A "bernini" entry
 * with no options is kept as a bare marker fallback.
 */
function expandBerniniOptions(modelSpecs: VideoGenerationModelSpecItem[]): VideoGenerationModelSpecItem[] {
  const out: VideoGenerationModelSpecItem[] = []
  for (const item of modelSpecs) {
    // The OpenAPI pipeline union is "fast" | "pro", but the runner also
    // advertises a "bernini" pipeline at runtime — compare as string.
    if ((item.pipeline as string) !== 'bernini') {
      out.push(item)
      continue
    }
    const spec = item.spec as unknown as { options?: string[] }
    const opts = Array.isArray(spec.options) ? spec.options : []
    if (opts.length === 0) {
      out.push(item)
      continue
    }
    for (const opt of opts) {
      out.push({
        ...item,
        pipeline: opt as VideoGenerationModelSpecItem['pipeline'],
        spec: {
          ...(item.spec as object),
          display_name: BERNINI_OPTION_LABELS[opt] ?? opt,
        } as VideoGenerationModelSpecItem['spec'],
      })
    }
  }
  return out
}

/**
 * Materialize model specs that omit their own resolution/fps/duration matrix by
 * ALIASING the "fast" (LTX-2.3) limits -- same 22B fp8-cast footprint, same VRAM
 * budget. The runner advertises LTX-2.5 as a minimal marker (display_name only, no
 * matrix) to keep the orchestrator heartbeat metadata under its 1024-byte cap; here
 * we give it the full 2.3 matrix client-side so the options picker works identically.
 * Models that do carry their own matrix pass through untouched.
 */
function inheritLimitsFromFast(
  modelSpecs: VideoGenerationModelSpecItem[],
): VideoGenerationModelSpecItem[] {
  const fast = modelSpecs.find((item) => item.pipeline === 'fast')
  if (!fast) return modelSpecs
  const fastLimits = fast.spec.supported_resolutions_durations
  return modelSpecs.map((item) => {
    if (item.pipeline === 'fast') return item
    // Bernini is a distinct engine (native 480p @ 16fps) delivered through the
    // RIFE / FlashVSR post rails — its options come from lib/bernini-delivery.ts,
    // NEVER from the LTX-2.3 limits alias. Leave it marker-only (no matrix) so
    // the runner heartbeat metadata stays lean and the picker treats it as its
    // own selectable T2V model.
    if (item.pipeline?.startsWith('bernini')) return item
    if (item.spec.supported_resolutions_durations) return item
    return { ...item, spec: { ...item.spec, supported_resolutions_durations: fastLimits } }
  })
}

export function resolveVideoGenerationOptions<T extends VideoGenerationSettingsShape>({
  settings,
  modelSpecs,
  hasAudio = false,
  minimumDuration,
  durationSelection = 'preserve',
}: ResolveVideoGenerationOptionsParams<T>): ResolvedVideoGenerationOptions {
  const modelOptions = getCompatibleModelOptions(modelSpecs, { hasAudio, minimumDuration })
  const selectedModelItem = modelOptions.find((item) => item.pipeline === settings.model) ?? modelOptions[0] ?? null
  if (!selectedModelItem) {
    return {
      modelOptions,
      resolutionOptions: [],
      fpsOptions: [],
      durationOptions: [],
      selectedModel: null,
      selectedResolution: null,
      selectedFps: null,
      selectedDuration: null,
      hasCompatibleOptions: false,
    }
  }

  const resolutionEntries = getCompatibleResolutionEntries(selectedModelItem, { hasAudio, minimumDuration })
  const resolutionOptions = resolutionEntries.map(([resolution]) => resolution)
  const selectedResolution = chooseOption(settings.videoResolution, resolutionOptions)
  if (!selectedResolution) {
    return {
      modelOptions,
      resolutionOptions,
      fpsOptions: [],
      durationOptions: [],
      selectedModel: selectedModelItem.pipeline,
      selectedResolution: null,
      selectedFps: null,
      selectedDuration: null,
      hasCompatibleOptions: false,
    }
  }

  const selectedResolutionSpec = resolutionEntries.find(([resolution]) => resolution === selectedResolution)?.[1] ?? null
  if (!selectedResolutionSpec) {
    return {
      modelOptions,
      resolutionOptions,
      fpsOptions: [],
      durationOptions: [],
      selectedModel: selectedModelItem.pipeline,
      selectedResolution,
      selectedFps: null,
      selectedDuration: null,
      hasCompatibleOptions: false,
    }
  }

  const fpsOptions = getCompatibleFps(selectedResolutionSpec, { minimumDuration })
  const selectedFps = chooseOption(settings.fps, fpsOptions)
  if (!selectedFps) {
    return {
      modelOptions,
      resolutionOptions,
      fpsOptions,
      durationOptions: [],
      selectedModel: selectedModelItem.pipeline,
      selectedResolution,
      selectedFps: null,
      selectedDuration: null,
      hasCompatibleOptions: false,
    }
  }

  const durationOptions = filterDurationsByMinimum(
    getDurationsForFps(selectedResolutionSpec, selectedFps),
    minimumDuration,
  )
  const selectedDuration = durationSelection === 'smallest_valid'
    ? durationOptions[0] ?? null
    : chooseOption(settings.duration, durationOptions)

  return {
    modelOptions,
    resolutionOptions,
    fpsOptions,
    durationOptions,
    selectedModel: selectedModelItem.pipeline,
    selectedResolution,
    selectedFps,
    selectedDuration,
    hasCompatibleOptions: selectedDuration !== null,
  }
}

export function sanitizeVideoGenerationSettings<T extends VideoGenerationSettingsShape>(
  settings: T,
  modelSpecs: VideoGenerationModelSpecItem[],
  options: {
    hasAudio?: boolean
    minimumDuration?: number
    durationSelection?: DurationSelectionMode
  } = {},
): T | null {
  const resolved = resolveVideoGenerationOptions({
    settings,
    modelSpecs,
    hasAudio: options.hasAudio,
    minimumDuration: options.minimumDuration,
    durationSelection: options.durationSelection,
  })
  if (
    !resolved.hasCompatibleOptions
    || !resolved.selectedModel
    || !resolved.selectedResolution
    || !resolved.selectedFps
    || !resolved.selectedDuration
  ) {
    return null
  }

  return {
    ...settings,
    model: resolved.selectedModel,
    videoResolution: resolved.selectedResolution,
    fps: resolved.selectedFps,
    duration: resolved.selectedDuration,
    aspectRatio: (settings.aspectRatio === '9:16' ? '9:16' : '16:9') as VideoGenerationAspectRatio,
  }
}

export function areVideoGenerationSettingsEquivalent<T extends VideoGenerationSettingsShape>(
  left: T,
  right: T,
): boolean {
  return (
    left.model === right.model
    && left.duration === right.duration
    && left.videoResolution === right.videoResolution
    && left.fps === right.fps
    && (left.aspectRatio ?? '16:9') === (right.aspectRatio ?? '16:9')
    && (left.audio ?? false) === (right.audio ?? false)
  )
}
