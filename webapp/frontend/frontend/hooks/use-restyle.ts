import { useCallback, useState } from 'react'
import { withGenerationActive } from '../lib/generation-active'
import { logger } from '../lib/logger'
import { resolveRunner, postRunnerTaskWithTicket } from '../lib/direct-transport'
import { getBlob, isWebPath } from '../lib/runtime/web-store'
import { GENERATION_RECOVERY_KEY, GENERATION_RECOVERY_TS_KEY } from './use-generation'

/** Read a browser Blob (e.g. a web:// image/video) as a base64 payload for a runner. */
function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const fr = new FileReader()
    fr.onload = () => {
      const url = fr.result
      if (typeof url === 'string') {
        const comma = url.indexOf(',')
        resolve(comma >= 0 ? url.slice(comma + 1) : url)
      } else {
        reject(new Error('Could not read media blob'))
      }
    }
    fr.onerror = () => reject(fr.error ?? new Error('Could not read media blob'))
    fr.readAsDataURL(blob)
  })
}

/** Resolve a browser asset path (web://key → Blob) to base64, or null if not resolvable. */
async function assetToBase64(path: string | null | undefined): Promise<string | null> {
  if (!path) return null
  if (isWebPath(path)) {
    const blob = getBlob(path)
    return blob ? blobToBase64(blob) : null
  }
  // Only web:// blobs are resolvable in-browser; anything else (electron disk path,
  // backend path) must be handled by the backend Worker rail instead.
  return null
}

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
      try {
      // Direct transport requires a configured Livepeer runner; without it there is no
      // remote backend to dispatch the restyle to.
      const runner = await resolveRunner(['restyle'])
      if (!runner) {
        const msg = 'No capable Livepeer runner is currently available for restyling.'
        logger.error(`Restyle error: ${msg}`)
        setState({ isRestyling: false, restyleStatus: '', restyleError: msg, result: null })
        return
      }

      // The id-v2v worker cannot read browser asset paths (web:// blobs) — it needs the
      // actual media bytes. Convert the source video + accepted styled first frame to
      // base64 and send them under the worker's expected field names (source_video /
      // stylized_first_frame). Only resolvable web:// assets can be base64'd in-browser;
      // if a media path isn't resolvable here (e.g. a backend/electron path) we keep the
      // path form for the backend Worker rail to resolve server-side.
      const sourceB64 = await assetToBase64(params.videoPath)
      const styledB64 = await assetToBase64(params.stylizedImagePath)
      const usingB64 = !!(sourceB64 && styledB64)
      const body: Record<string, unknown> = {
        prompt: params.prompt,
        model: params.model ?? 'fast',
        max_frames: params.maxFrames ?? 81,
        // 0 = let the backend resolve steps from the model (fast ~8, regular 30).
        inference_steps: params.inferenceSteps ?? 0,
        cfg_scale: params.cfgScale ?? 5.0,
        seed: params.seed,
        enhance_prompt: params.enhancePrompt ?? false,
      }
      if (usingB64) {
        body.source_video = sourceB64
        body.stylized_first_frame = styledB64
      } else {
        body.video_path = params.videoPath
        body.stylized_image_path = params.stylizedImagePath
      }
      logger.info(`[restyle] direct rail media=${usingB64 ? 'base64' : 'path'}`)
      const res = await postRunnerTaskWithTicket(runner, 'restyle', body, {
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
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Restyle failed'
        logger.error(`Restyle error: ${msg}`)
        setState({ isRestyling: false, restyleStatus: '', restyleError: msg, result: null })
      } finally {
        // Always release the global generation lock on settle. A thrown fetch/SSE/worker
        // error must not leak GENERATION_RECOVERY_KEY — otherwise every Generate button
        // greys out until localStorage is manually cleared (mirrors use-extend).
        window.localStorage.removeItem(GENERATION_RECOVERY_KEY)
        window.localStorage.removeItem(GENERATION_RECOVERY_TS_KEY)
      }
    })
  }, [])

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
