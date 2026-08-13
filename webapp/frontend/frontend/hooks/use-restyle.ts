import { useCallback, useState } from 'react'
import { withGenerationActive } from '../lib/generation-active'
import { logger } from '../lib/logger'
import { resolveRunner, postRunnerTaskWithTicket } from '../lib/direct-transport'
import { useAppSettings } from '../contexts/AppSettingsContext'

export type RestyleModel = 'fast' | 'regular'

export interface RestyleSubmitParams {
  videoPath: string
  stylizedImagePath: string
  prompt: string
  // id-v2v model variant: "fast" = FusionX (~8 steps), "regular" = original (30).
  model?: RestyleModel
  maxFrames?: number
  inferenceSteps?: number
  cfgScale?: number
  // Rotate per Redo so re-running the same restyle yields a distinct take.
  seed?: number
  // Ask the id-v2v worker's Gemma LLM to rewrite/expand the prompt before the
  // restyle. (A blank/auto prompt is always auto-captioned from the source video.)
  enhancePrompt?: boolean
}

export interface RestyleResult {
  videoPath: string
  // Gemma LLM artifacts returned from the id-v2v worker, saved for the UI to
  // show for reference. null when Gemma didn't run.
  videoCaption?: string | null
  enhancedPrompt?: string | null
}

interface UseRestyleState {
  isRestyling: boolean
  restyleStatus: string
  restyleError: string | null
  result: RestyleResult | null
}


// Map backend phase to a user-friendly message.
function getPhaseMessage(phase: string): string {
  switch (phase) {
    case 'connecting_remote':
    case 'sending_to_remote':
      return 'Connecting to remote runner...'
    case 'connecting':
      return 'Opening websocket...'
    case 'preprocessing':
      return 'Preparing frames...'
    case 'generating':
      return 'Generating (restyle)...'
    case 'decoding':
      return 'Decoding video...'
    case 'finalizing':
      return 'Finalizing output...'
    case 'upscaling':
      return 'Upscaling to full resolution...'
    case 'downloading_result':
    case 'downloading_output':
      return 'Downloading output...'
    case 'complete':
      return 'Complete!'
    default:
      return 'Restyling...'
  }
}


export function useRestyle() {
  const { settings } = useAppSettings()
  const [state, setState] = useState<UseRestyleState>({
    isRestyling: false,
    restyleStatus: '',
    restyleError: null,
    result: null,
  })


  const submitRestyle = useCallback(async (params: RestyleSubmitParams) => {
    if (!params.videoPath || !params.stylizedImagePath) return

    setState({
      isRestyling: true,
      restyleStatus: 'Connecting to remote runner...',
      restyleError: null,
      result: null,
    })

    await withGenerationActive(async () => {
      // Direct transport requires a configured Livepeer runner; without it there is no
      // remote backend to dispatch the restyle to.
      if (!(settings.hasLivepeerDiscoveryUrl && settings.livepeerDiscoveryUrl.trim())) {
        const msg = 'Restyle requires Livepeer runners. Configure a Livepeer discovery URL in Settings.'
        logger.error(`Restyle error: ${msg}`)
        setState({ isRestyling: false, restyleStatus: '', restyleError: msg, result: null })
        return
      }

      const runner = await resolveRunner(['restyle'])
      if (!runner) {
        const msg = 'No capable Livepeer runner is currently available for restyling.'
        logger.error(`Restyle error: ${msg}`)
        setState({ isRestyling: false, restyleStatus: '', restyleError: msg, result: null })
        return
      }

      const res = await postRunnerTaskWithTicket(runner, 'restyle', {
        video_path: params.videoPath,
        stylized_image_path: params.stylizedImagePath,
        prompt: params.prompt,
        model: params.model ?? 'fast',
        max_frames: params.maxFrames ?? 81,
        // 0 = let the backend resolve steps from the model (fast ~8, regular 30).
        inference_steps: params.inferenceSteps ?? 0,
        cfg_scale: params.cfgScale ?? 5.0,
        seed: params.seed,
        enhance_prompt: params.enhancePrompt ?? false,
      }, {
        onProgress: (ev) => {
          const msg = getPhaseMessage(ev.stage || '')
          const isBackbone = ev.stage === 'generating'
          const pct = typeof ev.progress === 'number' ? ev.progress : null
          const showPct = isBackbone && pct !== null
          setState(prev => ({
            ...prev,
            restyleStatus: showPct ? `${msg} ${Math.min(Math.round(pct * 100), 99)}%` : msg,
          }))
        },
      })

      if (!res.mediaBlob) {
        const err = res.payload?.error ? String(res.payload.error) : 'Runner returned no media'
        logger.error(`Restyle error: ${err}`)
        setState({ isRestyling: false, restyleStatus: '', restyleError: err, result: null })
        return
      }

      // The runner streams the generated media back over the WebSocket — store it as a local
      // object URL for the project asset store.
      const videoPath = URL.createObjectURL(res.mediaBlob)
      const videoCaption = res.payload?.video_caption ? String(res.payload.video_caption) : null
      const enhancedPrompt = res.payload?.enhanced_prompt ? String(res.payload.enhanced_prompt) : null
      setState({
        isRestyling: false,
        restyleStatus: 'Restyle complete!',
        restyleError: null,
        result: { videoPath, videoCaption, enhancedPrompt },
      })
    })
  }, [settings])

  const resetRestyle = useCallback(() => {
    setState({
      isRestyling: false,
      restyleStatus: '',
      restyleError: null,
      result: null,
    })
  }, [])

  return {
    submitRestyle,
    resetRestyle,
    isRestyling: state.isRestyling,
    restyleStatus: state.restyleStatus,
    restyleError: state.restyleError,
    restyleResult: state.result,
  }
}
