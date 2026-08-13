import { useCallback, useState } from 'react'
import { withGenerationActive } from '../lib/generation-active'
import { logger } from '../lib/logger'
import { resolveRunner, postRunnerTaskWithTicket } from '../lib/direct-transport'

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

    setState({ isExtending: true, extendStatus: 'Generating', extendError: null, result: null })

    await withGenerationActive(async () => {
      const runner = await resolveRunner(['extend'])
      if (!runner) {
        const msg = 'No capable Livepeer runner is currently available for extending.'
        logger.error(`Extend error: ${msg}`)
        setState({ isExtending: false, extendStatus: '', extendError: msg, result: null })
        return
      }

      const res = await postRunnerTaskWithTicket(runner, 'extend', {
        video_path: params.videoPath,
        duration: params.duration,
        prompt: params.prompt,
        mode: params.mode,
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
