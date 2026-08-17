import { useEffect, useState } from 'react'
import { ApiClient } from '../lib/api-client'
import { isWebPlatform, discoverRunners, getRunnerDiscoveryConfig } from '../lib/livepeer-discovery'
import type { VideoGenerationModelSpecItem, VideoGenerationModelSpecsResponse } from '../lib/video-generation-model-specs'

interface VideoGenerationModelSpecsState {
  modelSpecs: VideoGenerationModelSpecsResponse | null
  isLoading: boolean
  errorMessage: string | null
}

/**
 * Dedupe runner-advertised model specs by pipeline::display_name — mirrors the Worker's
 * mergeLocalModels (src/routes/catalog.ts) so the browser builds the same list the Worker
 * used to build by hitting /api/generate/models-specs.
 */
function mergeLocalModels(specs: unknown[]): VideoGenerationModelSpecItem[] {
  const seen = new Set<string>()
  const out: VideoGenerationModelSpecItem[] = []
  for (const item of specs) {
    if (!item || typeof item !== 'object') continue
    const rec = item as Record<string, unknown>
    const spec = rec.spec && typeof rec.spec === 'object' ? rec.spec : null
    if (!spec) continue
    const pipeline = typeof rec.pipeline === 'string' ? rec.pipeline : ''
    const specRec = spec as Record<string, unknown>
    const displayName = typeof specRec.display_name === 'string' ? specRec.display_name : ''
    const key = `${pipeline}::${displayName}`
    if (seen.has(key)) continue
    seen.add(key)
    out.push(rec as unknown as VideoGenerationModelSpecItem)
  }
  return out
}

/**
 * Web build: video model specs come straight from the runner's discovery metadata — the runner
 * advertises its real resolution/fps/duration capabilities (metadata.model_specs), which is the
 * authoritative source now that discovery is client-side and the Worker no longer serves
 * /api/generate/models-specs. The LTX-API slot (api_models) is deliberately empty: the static web
 * app's video path is runner-driven.
 */
async function fetchWebModelSpecs(discoveryUrl?: string): Promise<VideoGenerationModelSpecsResponse> {
  const url = discoveryUrl ?? getRunnerDiscoveryConfig().discoveryUrl
  if (!url) return { api_models: [], local_models: [] }
  const runners = await discoverRunners(url)
  const advertised = runners
    .filter((r) => r.status === 'ready' && Array.isArray(r.modelSpecs))
    .map((r) => r.modelSpecs as unknown[])
  return { api_models: [], local_models: mergeLocalModels(advertised) }
}

export function useVideoGenerationModelSpecs(discoveryUrl?: string): VideoGenerationModelSpecsState {
  const [state, setState] = useState<VideoGenerationModelSpecsState>({
    modelSpecs: null,
    isLoading: true,
    errorMessage: null,
  })

  useEffect(() => {
    const abortController = new AbortController()
    let isActive = true

    void (async () => {
      try {
        if (isWebPlatform()) {
          const data = await fetchWebModelSpecs(discoveryUrl)
          if (!isActive) return
          setState({ modelSpecs: data, isLoading: false, errorMessage: null })
          return
        }
        const result = await ApiClient.getGenerateVideoModelSpecs(undefined, {
          signal: abortController.signal,
        })
        if (!isActive) return
        if (result.ok) {
          setState({ modelSpecs: result.data, isLoading: false, errorMessage: null })
        } else {
          setState({ modelSpecs: null, isLoading: false, errorMessage: result.error.message })
        }
      } catch (e) {
        if (!isActive) return
        setState({
          modelSpecs: null,
          isLoading: false,
          errorMessage: e instanceof Error ? e.message : 'Failed to load model specs.',
        })
      }
    })()

    return () => {
      isActive = false
      abortController.abort()
    }
  }, [discoveryUrl])

  return state
}
