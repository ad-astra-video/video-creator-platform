import { useCallback, useState } from 'react'
import { withGenerationActive } from '../lib/generation-active'
import { logger } from '../lib/logger'
import { resolveRunner, postRunnerTaskWithTicket } from '../lib/direct-transport'
import { useAppSettings } from '../contexts/AppSettingsContext'

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
  const { settings } = useAppSettings()
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
      if (!(settings.hasLivepeerDiscoveryUrl && settings.livepeerDiscoveryUrl.trim())) {
        const msg = 'Remote retake requires Livepeer runners. Configure a Livepeer discovery URL in Settings.'
        logger.error(`Retake error: ${msg}`)
        setState({ isRetaking: false, retakeStatus: '', retakeError: msg, result: null })
        return
      }

      const runner = await resolveRunner(['t2v'])
      if (!runner) {
        const msg = 'No capable Livepeer runner is currently available for retake.'
        logger.error(`Retake error: ${msg}`)
        setState({ isRetaking: false, retakeStatus: '', retakeError: msg, result: null })
        return
      }

      const res = await postRunnerTaskWithTicket(runner, 'retake', {
        video_path: params.videoPath,
        start_time: params.startTime,
        duration: params.duration,
        prompt: params.prompt,
        mode: params.mode,
        resolution: params.resolution,
      }, {
        onProgress: (ev) => {
          if (ev.stage === 'generating') {
            setState(prev => ({ ...prev, retakeStatus: ev.message || 'Retaking...' }))
          }
        },
      })

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
  }, [settings])

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
