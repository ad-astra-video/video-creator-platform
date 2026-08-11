import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiClient } from '../lib/api-client'
import { withGenerationActive } from '../lib/generation-active'
import { logger } from '../lib/logger'

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

const POLLING_INTERVAL_MS = 500

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
  const [state, setState] = useState<UseRestyleState>({
    isRestyling: false,
    restyleStatus: '',
    restyleError: null,
    result: null,
  })

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  // Stop polling if the hook unmounts mid-generation.
  useEffect(() => stopPolling, [stopPolling])

  const submitRestyle = useCallback(async (params: RestyleSubmitParams) => {
    if (!params.videoPath || !params.stylizedImagePath) return

    setState({
      isRestyling: true,
      restyleStatus: 'Connecting to remote runner...',
      restyleError: null,
      result: null,
    })

    await withGenerationActive(async () => {
      // Poll the backend's generation progress while the restyle request is
      // in flight (the backend streams real remote progress over a websocket
      // to the live-runner, so this shows live stages/percent rather than a
      // static "Restyling" that hides a minuter-scale generation).
      let stopped = false
      pollRef.current = setInterval(async () => {
        if (stopped) return
        const r = await ApiClient.getGenerationProgress()
        if (!r.ok || stopped) return
        const data = r.data
        if (data.status === 'complete') {
          setState(prev => ({ ...prev, restyleStatus: 'Finalizing...' }))
          return
        }
        const msg = getPhaseMessage(data.phase)
        // The % is only meaningful for backbone inference iterations; every
        // other step (connecting/preprocessing/decoding/upscaling/finalizing)
        // updates the text only. So render the % purely when phase is
        // 'generating' and a numeric progress is present.
        const isBackbone = data.phase === 'generating'
        const pct = typeof data.progress === 'number' ? data.progress : null
        const showPct = isBackbone && pct !== null
        setState(prev => ({
          ...prev,
          restyleStatus: showPct ? `${msg} ${Math.min(pct, 99)}%` : msg,
        }))
      }, POLLING_INTERVAL_MS)

      const result = await ApiClient.restyle({
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
      })
      stopped = true
      stopPolling()

      if (!result.ok) {
        logger.error(`Restyle error: ${result.error.message}`)
        setState({
          isRestyling: false,
          restyleStatus: '',
          restyleError: result.error.message,
          result: null,
        })
        return
      }

      const payload = result.data

      if (payload.status === 'cancelled') {
        setState({
          isRestyling: false,
          restyleStatus: 'Cancelled',
          restyleError: null,
          result: null,
        })
        return
      }

      setState({
        isRestyling: false,
        restyleStatus: 'Restyle complete!',
        restyleError: null,
        result: {
          videoPath: payload.video_path,
          videoCaption: payload.video_caption ?? null,
          enhancedPrompt: payload.enhanced_prompt ?? null,
        },
      })
    })
  }, [stopPolling])

  const resetRestyle = useCallback(() => {
    stopPolling()
    setState({
      isRestyling: false,
      restyleStatus: '',
      restyleError: null,
      result: null,
    })
  }, [stopPolling])

  return {
    submitRestyle,
    resetRestyle,
    isRestyling: state.isRestyling,
    restyleStatus: state.restyleStatus,
    restyleError: state.restyleError,
    restyleResult: state.result,
  }
}
