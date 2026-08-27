import { useCallback, useState } from 'react'
import { withGenerationActive } from '../lib/generation-active'
import { logger } from '../lib/logger'
import { resolveRunner, postRunnerTaskWithTicketSSE } from '../lib/direct-transport'
import { getBlob, isWebPath, registerBlob } from '../lib/runtime/web-store'
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

/**
 * The id-v2v worker reports its real denoise step as a message string, both in the
 * single-clip core ("step K/M") and the multi-clip path ("clip N/total step K/M").
 * Extract K/M so the chat-history card can render a live "step K/M" alongside the
 * overall progress fraction.
 */
function parseStep(message?: string): { step?: number; totalSteps?: number } {
  const m = /step\s+(\d+)\s*\/\s*(\d+)/i.exec(message || '')
  if (m) return { step: parseInt(m[1], 10), totalSteps: parseInt(m[2], 10) }
  return {}
}

export type RestyleModel = 'fast' | 'regular'

/** Live restyle progress mirrored to a chat-history generation card. */
export interface RestyleProgressEvent {
  /** Overall backbone fraction 0..1 (only while generating; null otherwise). */
  progress?: number | null
  /** Current denoise step within the active clip, when the worker reports it. */
  step?: number
  /** Denoise steps in the active clip, when the worker reports it. */
  totalSteps?: number
  /** Raw worker status text (e.g. "clip 1/2 step 12/30" or "Preparing frames..."). */
  statusMessage?: string
  /** Backend phase (preprocessing | generating | decoding | ...). */
  stage?: string
}

export interface RestyleSubmitParams {
  videoPath: string
  stylizedImagePath: string
  prompt: string
  // id-v2v model variant: "fast" = FusionX (~8 steps), "regular" = original (30).
  model?: RestyleModel
  // Output fps: 24 | 25 | 30 | 'auto'. 'auto' (default) = encode the returned
  // restyle at the source video's own fps so it plays at the same rate/duration
  // as the input. An explicit 24/25/30 overrides that.
  fps?: 24 | 25 | 30 | 'auto'
  maxFrames?: number
  inferenceSteps?: number
  cfgScale?: number
  // Rotate per Redo so re-running the same restyle yields a distinct take.
  seed?: number
  // Ask the id-v2v worker's Gemma LLM to rewrite/expand the prompt before the
  // restyle. (A blank/auto prompt is always auto-captioned from the source video.)
  enhancePrompt?: boolean
  // Live progress callback (the chat-history card consumes this to mirror the
  // runner's real per-step progress).
  onProgress?: (ev: RestyleProgressEvent) => void
}

export interface RestyleResult {
  videoPath: string
  // Gemma LLM artifacts returned from the id-v2v worker, saved for the UI to
  // show for reference. null when Gemma didn't run.
  videoCaption?: string | null
  enhancedPrompt?: string | null
}

export interface RestyleOutcome {
  error: string | null
  result: RestyleResult | null
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

  const submitRestyle = useCallback(async (params: RestyleSubmitParams): Promise<RestyleOutcome> => {
    if (!params.videoPath || !params.stylizedImagePath) {
      return { error: 'Missing source video or styled frame.', result: null }
    }
    let outcome: RestyleOutcome = { error: 'Restyle did not run', result: null }

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
          outcome = { error: msg, result: null }
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
          // Only send fps when the user picked an explicit rate; 'auto' is the
          // default and lets the worker match the source video's fps.
          ...(params.fps && params.fps !== 'auto' ? { fps: params.fps } : {}),
          ...(params.maxFrames != null ? { max_frames: params.maxFrames } : {}),
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
        const res = await postRunnerTaskWithTicketSSE(runner, 'restyle', body, {
          onProgress: (ev) => {
            const msg = getPhaseMessage(ev.stage || '')
            const isBackbone = ev.stage === 'generating'
            const pct = typeof ev.progress === 'number' ? ev.progress : null
            const showPct = isBackbone && pct !== null
            setState(prev => ({
              ...prev,
              restyleStatus: showPct ? `${msg} ${Math.min(Math.round(pct * 100), 99)}%` : msg,
            }))
            // Mirror the REAL worker progress (fraction + per-step "clip x step k/m"
            // text) to the chat-history card. The worker's message is authoritative —
            // only the generating stage carries a numeric fraction (no fabricated %).
            params.onProgress?.({
              progress: pct,
              statusMessage:
                typeof ev.message === 'string' && ev.message.trim()
                  ? ev.message
                  : msg,
              stage: ev.stage ?? undefined,
              ...parseStep(ev.message ?? undefined),
            })
          },
        })

        if (!res.mediaBlob) {
          const err = res.payload?.error ? String(res.payload.error) : 'Runner returned no media'
          logger.error(`Restyle error: ${err}`)
          setState({ isRestyling: false, restyleStatus: '', restyleError: err, result: null })
          outcome = { error: err, result: null }
          return
        }

        // Store the restyle video in the browser asset store as a web:// key that
        // webAssetUrl() can resolve (a raw blob: URL would fall through to a broken
        // file:///blob%3A... link and the preview shows "Not allowed to load local
        // resource"). Mirrors styleFrameViaRunner's registerBlob pattern.
        const videoPath = registerBlob(
          res.mediaBlob,
          'restyled.mp4',
          (res.mediaBlob as Blob).type || 'video/mp4',
        )

        const videoCaption = res.payload?.video_caption ? String(res.payload.video_caption) : null
        const enhancedPrompt = res.payload?.enhanced_prompt ? String(res.payload.enhanced_prompt) : null
        const result: RestyleResult = { videoPath, videoCaption, enhancedPrompt }
        setState({
          isRestyling: false,
          restyleStatus: 'Restyle complete!',
          restyleError: null,
          result,
        })
        outcome = { error: null, result }
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Restyle failed'
        logger.error(`Restyle error: ${msg}`)
        setState({ isRestyling: false, restyleStatus: '', restyleError: msg, result: null })
        outcome = { error: msg, result: null }
      } finally {
        // Always release the global generation lock on settle. A thrown fetch/SSE/worker
        // error must not leak GENERATION_RECOVERY_KEY — otherwise every Generate button
        // greys out until localStorage is manually cleared (mirrors use-extend).
        window.localStorage.removeItem(GENERATION_RECOVERY_KEY)
        window.localStorage.removeItem(GENERATION_RECOVERY_TS_KEY)
      }
    })

    return outcome
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
