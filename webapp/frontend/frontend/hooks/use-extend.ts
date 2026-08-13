import { useCallback, useState } from 'react'
import { withGenerationActive } from '../lib/generation-active'
import { logger } from '../lib/logger'
import { resolveRunner, postRunnerTaskWithTicket, pathToBase64 } from '../lib/direct-transport'
import { GENERATION_RECOVERY_KEY } from './use-generation'

export type ExtendDirection = 'start' | 'end'

// Seconds-to-add presets (matches LTX Studio). API allows 2–20s.
export const EXTEND_SECONDS = [4, 6, 8, 10, 12] as const
export const DEFAULT_EXTEND_SECONDS = 4

export interface ExtendSubmitParams {
  videoPath: string
  duration: number
  prompt: string
  mode: ExtendDirection
  resolution?: { width: number; height: number }
}

export interface ExtendResult {
  videoPath: string
}

interface UseExtendState {
  isExtending: boolean
  extendStatus: string
  extendError: string | null
  result: ExtendResult | null
}


export function useExtend() {
  const [state, setState] = useState<UseExtendState>({
    isExtending: false,
    extendStatus: '',
    extendError: null,
    result: null,
  })

  const submitExtend = useCallback(async (params: ExtendSubmitParams) => {
    if (!params.videoPath) return

    logger.info(`Extend: submit started videoPath=${params.videoPath} duration=${params.duration} mode=${params.mode}`)
    setState({ isExtending: true, extendStatus: 'Generating', extendError: null, result: null })

    await withGenerationActive(async () => {
      try {
        const runner = await resolveRunner(['extend'])
        if (runner) logger.info(`Extend: runner resolved ${runner.runner_id}`)
        else logger.info('Extend: NO capable extend runner resolved')

        if (!runner) {
          const msg = 'No capable Livepeer runner is currently available for extending.'
          logger.error(`Extend error: ${msg}`)
          setState({ isExtending: false, extendStatus: '', extendError: msg, result: null })
          return
        }

        // The remote worker cannot fetch a browser-local web:// source video — it requires the
        // actual bytes as video_base64 in the body (otherwise worker 500s with KeyError).
        const videoBase64 = await pathToBase64(params.videoPath)
        if (videoBase64) logger.info(`Extend: source bytes base64 ${(videoBase64.length / 1024 / 1024).toFixed(2)} MB`)
        else logger.info(`Extend: pathToBase64 returned null (path=${params.videoPath})`)
        if (!videoBase64) {
          const msg = `The source video cannot be read as bytes (path=${params.videoPath}). It must be a browser asset before it can be extended.`
          logger.error(`Extend error: ${msg}`)
          setState({ isExtending: false, extendStatus: '', extendError: msg, result: null })
          return
        }

        const res = await postRunnerTaskWithTicket(runner, 'extend', {
          video_base64: videoBase64,
          prompt: params.prompt,
          // Worker extends by FRAMES, not seconds. 24fps (LTX band) preserves the chosen duration.
          extendFrames: Math.round(params.duration * 24),
          mode: params.mode,
          seed: 42,
          fps: 24,
          resolution: params.resolution,
        }, {
          onProgress: (ev) => {
            if (ev.stage === 'generating') {
              setState(prev => ({ ...prev, extendStatus: ev.message || 'Extending...' }))
            }
          },
        })
        if (!res.mediaBlob) {
          const err = res.payload?.error ? String(res.payload.error) : 'Runner returned no media'
          logger.error(`Extend error: ${err}`)
          setState({ isExtending: false, extendStatus: '', extendError: err, result: null })
          return
        }

        // The runner streams the generated media back over the WebSocket — store it as a local
        // object URL for the project asset store.
        const videoPath = URL.createObjectURL(res.mediaBlob)
        setState({
          isExtending: false,
          extendStatus: 'Extend complete!',
          extendError: null,
          result: { videoPath },
        })
      } catch (err) {
        // Any throw (network fetch to runner, blob read, AbortError) surfaces as a visible
        // error instead of leaving isExtending stuck true with no request sent.
        const msg = err instanceof Error ? err.message : 'Extend failed'
        logger.error(`Extend error: ${msg}`)
        setState({ isExtending: false, extendStatus: '', extendError: msg, result: null })
      } finally {
        // Always clear the in-flight marker on settle so a failed/oversized extend can't
        // leak GENERATION_RECOVERY_KEY and wedge the global generation lock (every Generate
        // button greys out until localStorage is manually cleared). The GenSpace completion
        // effect clears it on success too; this guarantees the error path is covered.
        window.localStorage.removeItem(GENERATION_RECOVERY_KEY)
      }
    })
  }, [])

  const resetExtend = useCallback(() => {
    setState({ isExtending: false, extendStatus: '', extendError: null, result: null })
  }, [])

  return {
    submitExtend,
    resetExtend,
    isExtending: state.isExtending,
    extendStatus: state.extendStatus,
    extendError: state.extendError,
    extendResult: state.result,
  }
}
