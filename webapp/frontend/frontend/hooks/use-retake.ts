import { useCallback, useState } from 'react'
import { withGenerationActive } from '../lib/generation-active'
import { logger } from '../lib/logger'
import { resolveRunner, postRunnerTaskWithTicketSSE, pathToBase64 } from '../lib/direct-transport'

export type RetakeMode = 'replace_audio_and_video' | 'replace_video' | 'replace_audio'

export interface RetakeSubmitParams {
  videoPath: string
  startTime: number
  duration: number
  prompt: string
  mode: RetakeMode
  resolution?: { width: number; height: number }
}

export interface RetakeResult {
  videoPath: string
}


interface UseRetakeState {
  isRetaking: boolean
  retakeStatus: string
  retakeError: string | null
  result: RetakeResult | null
}

export function useRetake() {
  const [state, setState] = useState<UseRetakeState>({
    isRetaking: false,
    retakeStatus: '',
    retakeError: null,
    result: null,
  })

  const submitRetake = useCallback(async (params: RetakeSubmitParams) => {
    if (!params.videoPath) return

    setState({
      isRetaking: true,
      retakeStatus: 'Generating',
      retakeError: null,
      result: null,
    })

    await withGenerationActive(async () => {
      const runner = await resolveRunner(['t2v'])
      if (!runner) {
        const msg = 'No capable Livepeer runner is currently available for retake.'
        logger.error(`Retake error: ${msg}`)
        setState({ isRetaking: false, retakeStatus: '', retakeError: msg, result: null })
        return
      }

      // The remote worker needs the source clip's actual bytes (video_base64); it cannot read
      // a browser-local web:// key. Without this the worker 500s with KeyError('video_base64').
      const videoBase64 = await pathToBase64(params.videoPath)
      if (!videoBase64) {
        const msg = 'The source video must be a browser asset before it can be retaken.'
        logger.error(`Retake error: ${msg}`)
        setState({ isRetaking: false, retakeStatus: '', retakeError: msg, result: null })
        return
      }

      let res
      try {
        res = await postRunnerTaskWithTicketSSE(runner, 'retake', {
          video_base64: videoBase64,
          startTime: params.startTime,
          duration: params.duration,
          prompt: params.prompt,
          mode: params.mode,
          seed: 42,
          fps: 24,
          resolution: params.resolution,
        }, {
          onProgress: (ev) => {
            if (ev.stage === 'generating') {
              setState(prev => ({ ...prev, retakeStatus: ev.message || 'Retaking...' }))
            }
          },
        })
      } catch (e) {
        const err = e instanceof Error ? e.message : String(e)
        logger.error(`Retake error: ${err}`)
        setState({ isRetaking: false, retakeStatus: '', retakeError: err, result: null })
        return
      }

      if (!res.mediaBlob) {
        const err = res.payload?.error ? String(res.payload.error) : 'Runner returned no media'
        logger.error(`Retake error: ${err}`)
        setState({ isRetaking: false, retakeStatus: '', retakeError: err, result: null })
        return
      }

      setState({
        isRetaking: false,
        retakeStatus: 'Retake complete!',
        retakeError: null,
        result: { videoPath: URL.createObjectURL(res.mediaBlob) },
      })
    })
  }, [])

  const resetRetake = useCallback(() => {
    setState({
      isRetaking: false,
      retakeStatus: '',
      retakeError: null,
      result: null,
    })
  }, [])

  return {
    submitRetake,
    resetRetake,
    isRetaking: state.isRetaking,
    retakeStatus: state.retakeStatus,
    retakeError: state.retakeError,
    retakeResult: state.result,
  }
}
