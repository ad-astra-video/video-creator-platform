import { useState, useCallback, useRef, useEffect } from 'react'
import type { GenerationSettings } from '../components/SettingsPanel'
import { ApiClient, type ApiSuccessOf } from '../lib/api-client'
import {
  falGenerateI2I,
  falGenerateT2I,
  makeJobId,
  postImageToRunner,
  postRunnerTaskWithTicket,
  resolveRunner,
  type RunnerDto,
} from '../lib/direct-transport'
import { createLocalGenerationError, type GenerationError } from '../lib/generation-errors'
import { withGenerationActive } from '../lib/generation-active'
import { useAppSettings } from '../contexts/AppSettingsContext'

const POLLING_INTERVAL_MS = 2000

export const GENERATION_RECOVERY_KEY = 'ltx-generation-recovery'

export interface GenerationRecoveryContext {
  projectId: string
  prompt: string
  // Absent for ic-lora/retake: those recover as standalone video assets (Phase 1),
  // so there are no video/image settings to restore.
  settings?: GenerationSettings
  inputImageUrl?: string
  inputAudioUrl?: string
  genType?: 'image' | 'enhance'
  // Whatever generation id the backend reported at the moment this marker was written — i.e.
  // immediately BEFORE this generation started. The handler that starts a generation loads its
  // pipeline (can take many seconds — worse for image models loading checkpoint shards) before
  // it ever reports a new id, so a poll can otherwise be looking at a stale, unrelated id/result
  // that predates this marker entirely. Once a later poll observes a DIFFERENT id, that's proof
  // (single global generation slot) that this marker's own generation has started — see
  // checkAndConsumeRecovery in lib/generation-recovery.ts.
  baselineId: string | null
  // Set once a poll observes an id different from baselineId — i.e. once this marker's own
  // generation is confirmed to exist. Distinct from baselineId: a LATER id change past this point
  // means a DIFFERENT generation superseded ours (not that ours just started), which must NOT be
  // imported under this marker.
  generationId?: string
}

interface GenerationState {
  isGenerating: boolean
  progress: number
  statusMessage: string
  videoPath: string | null
  imagePath: string | null
  imagePaths: string[]
  error: GenerationError | null
}


interface UseGenerationReturn extends GenerationState {
  generate: (prompt: string, imagePath: string | null, settings: GenerationSettings, audioPath?: string | null) => Promise<void>
  generateImage: (prompt: string, settings: GenerationSettings, editSource?: string | null) => Promise<void>
  cancel: () => void
  reset: () => void
  resumeIfRunning: () => Promise<'running' | 'complete' | 'none'>
}

const IMAGE_SHORT_SIDE_BY_RESOLUTION: Record<string, number> = {
  '1080p': 1080,
  '1440p': 1440,
  '2048p': 2048,
}

const IMAGE_ASPECT_RATIO_VALUE: Record<string, number> = {
  '1:1': 1,
  '16:9': 16 / 9,
  '9:16': 9 / 16,
  '4:3': 4 / 3,
  '3:4': 3 / 4,
  '21:9': 21 / 9,
}

function getImageDimensions(settings: GenerationSettings): { width: number; height: number } {
  const shortSide = IMAGE_SHORT_SIDE_BY_RESOLUTION[settings.imageResolution]
  if (!shortSide) {
    throw new Error(`Unsupported image resolution mapping: ${settings.imageResolution}`)
  }

  const ratio = IMAGE_ASPECT_RATIO_VALUE[settings.imageAspectRatio]
  if (!ratio) {
    throw new Error(`Unsupported image aspect ratio mapping: ${settings.imageAspectRatio}`)
  }

  if (ratio >= 1) {
    return { width: Math.round(shortSide * ratio), height: shortSide }
  }
  return { width: shortSide, height: Math.round(shortSide / ratio) }
}

/** Convert an ArrayBuffer to a base64 data URI (used for FAL I2I edit uploads). */
function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  let binary = ''
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i])
  }
  return btoa(binary)
}

// Map phase to user-friendly message
function getPhaseMessage(phase: string): string {
  switch (phase) {
    case 'validating_request':
      return 'Validating request...'
    case 'uploading_image':
      return 'Uploading image...'
    case 'uploading_audio':
      return 'Uploading audio...'
    case 'loading_model':
      return 'Loading model...'
    case 'encoding_text':
      return 'Encoding prompt...'
    case 'inference':
      return 'Generating...'
    case 'downloading_output':
      return 'Downloading output...'
    case 'decoding':
      return 'Decoding video...'
    case 'complete':
      return 'Complete!'
    default:
      return 'Generating...'
  }
}

export function useGeneration(): UseGenerationReturn {
  const { settings: appSettings, shouldImageGenerateWithFalApi, refreshSettings } = useAppSettings()
  const [state, setState] = useState<GenerationState>({
    isGenerating: false,
    progress: 0,
    statusMessage: '',
    videoPath: null,
    imagePath: null,
    imagePaths: [],
    error: null,
  })

  const abortControllerRef = useRef<AbortController | null>(null)
  const recoveryIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const clearRecoveryPolling = () => {
    if (recoveryIntervalRef.current) {
      clearInterval(recoveryIntervalRef.current)
      recoveryIntervalRef.current = null
    }
  }

  useEffect(() => clearRecoveryPolling, [])

  // Re-attach to a generation that was running OR finished while the frontend was
  // unmounted. Polls the backend progress endpoint; localStorage recovery context
  // (inputs, settings incl. loras) is owned by the caller (GenSpace). Returns the
  // recovered status so the caller can restore context for 'running' AND 'complete'
  // (a generation that finished during the unmount window still needs its metadata).
  const resumeIfRunning = useCallback(async (): Promise<'running' | 'complete' | 'none'> => {
    const apply = (data: ApiSuccessOf<'getGenerationProgress'>): 'running' | 'complete' | 'other' => {
      if (data.status === 'complete' && data.result != null) {
        const vp = typeof data.result === 'string' ? data.result : null
        const ips = Array.isArray(data.result) ? data.result : []
        setState({
          isGenerating: false, progress: 100, statusMessage: 'Complete!',
          videoPath: vp, imagePath: ips[0] ?? null, imagePaths: ips, error: null,
        })
        return 'complete'
      }
      if (data.status === 'running') {
        setState(prev => ({
          ...prev, isGenerating: true, progress: data.progress,
          statusMessage: getPhaseMessage(data.phase),
        }))
        return 'running'
      }
      setState(prev => ({ ...prev, isGenerating: false, statusMessage: '' }))
      return 'other'
    }

    const initial = await ApiClient.getGenerationProgress()
    if (!initial.ok) return 'none'
    const status = apply(initial.data)
    if (status === 'complete') return 'complete'
    if (status !== 'running') return 'none'

    clearRecoveryPolling()
    recoveryIntervalRef.current = setInterval(async () => {
      const r = await ApiClient.getGenerationProgress()
      if (!r.ok) return
      if (apply(r.data) !== 'running') clearRecoveryPolling()
    }, POLLING_INTERVAL_MS)
    return 'running'
  }, [])

  const generate = useCallback(async (
    prompt: string,
    imagePath: string | null,
    settings: GenerationSettings,
    audioPath?: string | null,
  ) => {
    const statusMsg = settings.model === 'pro'
      ? 'Loading Pro model & generating...'
      : 'Generating video...'

    setState({
      isGenerating: true,
      progress: 0,
      statusMessage: statusMsg,
      videoPath: null,
      imagePath: null,
      imagePaths: [],
      error: null,
    })

    const abortController = new AbortController()
    abortControllerRef.current = abortController
    let progressInterval: ReturnType<typeof setInterval> | null = null
    let shouldApplyPollingUpdates = true

    await withGenerationActive(async () => {
      try {
        // Prepare JSON body
        const body: Record<string, unknown> = {
          prompt,
          model: settings.model,
          duration: settings.duration,
          resolution: settings.videoResolution,
          fps: settings.fps,
          audio: settings.audio,
          cameraMotion: settings.cameraMotion,
          negativePrompt: (settings as { negativePrompt?: string }).negativePrompt ?? '',
          aspectRatio: settings.aspectRatio || '16:9',
        }
        if (imagePath) {
          body.imagePath = imagePath
        }
        if (audioPath) {
          body.audioPath = audioPath
        }
        if (settings.loras?.length) {
          body.loras = settings.loras.map(l => ({ ref: l.ref, scale: l.scale }))
        }

        // Direct transport is the ONLY video backend in the webapp (/api/generate is 410 —
        // the Worker no longer carries the media path). Gate on the Livepeer video toggle, not
        // on the browser's discovery-URL string: runner resolution happens on the Worker via
        // /api/providers (using its own D1/env discovery config), so a configured/toggle-on
        // runner must be attempted even when appSettings.livepeerDiscoveryUrl is empty here.
        const useDirectTransport = appSettings.livepeerVideoEnabled !== false

        // DIRECT transport: resolve a capable runner (Worker discovery) and do the Livepeer
        // payment handshake + read the resulting media from the runner response.
        if (useDirectTransport) {
          const runner = await resolveRunner(['t2v'])
          if (!runner) {
            throw new Error('No available Livepeer runner for video generation')
          }
          const res = await postRunnerTaskWithTicket(
            runner,
            'generate',
            { ...body, jobId: makeJobId() },
            {
              signal: abortController.signal,
              onProgress: (ev) => {
                if (!shouldApplyPollingUpdates) return
                let displayProgress = 0
                if (ev.stage === 'generating' && typeof ev.progress === 'number') {
                  displayProgress = Math.min(Math.round(ev.progress * 95), 95)
                }
                setState(prev => ({
                  ...prev,
                  progress: displayProgress,
                  statusMessage: ev.message || 'Generating...',
                }))
              },
            },
          )
          shouldApplyPollingUpdates = false
          if (!res.mediaBlob) {
            throw new Error(res.payload?.error ? String(res.payload.error) : 'Runner returned no media')
          }
          const objectUrl = URL.createObjectURL(res.mediaBlob)
          setState({
            isGenerating: false,
            progress: 100,
            statusMessage: 'Complete!',
            videoPath: objectUrl,
            imagePath: null,
            imagePaths: [],
            error: null,
          })
          return
        }

        // The /api/generate fallback is gone in the direct-transport design (Worker returns
        // 410). If we reach here, no direct runner was available — fail with a clear message
        // instead of a confusing 410 from the Worker.
        setState(prev => ({
          ...prev,
          isGenerating: false,
          error: createLocalGenerationError(
            'Video generation requires a Livepeer runner (direct transport). Configure a Livepeer discovery URL in Settings and ensure a t2v-capable runner is ready.',
          ),
        }))
        return

      } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') {
          setState(prev => ({
            ...prev,
            isGenerating: false,
            statusMessage: 'Cancelled',
          }))
        } else {
          setState(prev => ({
            ...prev,
            isGenerating: false,
            error: createLocalGenerationError(error instanceof Error ? error.message : 'Unknown error'),
          }))
        }
      } finally {
        shouldApplyPollingUpdates = false
        if (progressInterval) {
          clearInterval(progressInterval)
        }
      }
    })
  }, [appSettings])

  const cancel = useCallback(async () => {
    // Abort the fetch request
    abortControllerRef.current?.abort()
    
    // Also tell the backend to cancel
    void ApiClient.cancelGeneration()
    
    setState(prev => ({
      ...prev,
      isGenerating: false,
      statusMessage: 'Cancelled',
    }))
  }, [])

  const generateImage = useCallback(async (
    prompt: string,
    settings: GenerationSettings,
    editSource?: string | null,
  ) => {
    const isEditing = !!editSource

    const openFalConnectDialog = () => {
      window.dispatchEvent(new CustomEvent('open-api-gateway', {
        detail: {
          requiredKeys: ['fal'],
          title: 'Connect FAL AI',
          description: `FAL AI is required for ${isEditing ? 'editing' : 'generating'} images with Z Image Turbo when API generations are enabled.`,
          blocking: false,
        },
      }))
    }

    const numImages = settings.variations || 1

    setState({
      isGenerating: true,
      progress: 0,
      statusMessage: isEditing
        ? 'Editing image...'
        : numImages > 1 ? `Generating ${numImages} images...` : 'Generating image...',
      videoPath: null,
      imagePath: null,
      imagePaths: [],
      error: null,
    })

    const abortController = new AbortController()
    abortControllerRef.current = abortController

    await withGenerationActive(async () => {
      let progressInterval: ReturnType<typeof setInterval> | null = null
      try {
        // Skip prompt enhancement for T2I - use original prompt directly
        const finalPrompt = prompt

        // Edit runs at the source image's resolution; width/height are ignored server-side.
        const dims = isEditing ? { width: 1024, height: 1024 } : getImageDimensions(settings)
        const numSteps = settings.imageSteps || (isEditing ? 8 : 4)

        // 1) Livepeer runner first (preferred). Non-fatal: fall through to other
        // providers when no capable livepeer runner is resolvable.
        //
        // Pays through the HTTP ticket rail (Livepeer-Payer-Address -> 402+payment
        // params -> POST /sign-ticket -> retry with Livepeer-Payment+Livepeer-Segment)
        // because a browser WebSocket cannot set those payment headers — against a
        // billing orchestrator the payment-less WS is rejected with 402
        // "invalid live runner payment signer address" (the orchestrator signs and
        // proxies; it needs a valid signed payment on the job).
        if (appSettings.livepeerImageEnabled) {
          let livepeerRunner: RunnerDto | null = null
          try {
            livepeerRunner = await resolveRunner(['image'])
          } catch {
            livepeerRunner = null
          }
          if (livepeerRunner) {
            const imageBody = {
              prompt: finalPrompt,
              width: dims.width,
              height: dims.height,
              numSteps,
              numImages,
              seed: 42,
              guidanceScale: 0.0,
              strength: isEditing ? (settings.imageEditStrength ?? 0.6) : 0.6,
              keepSubject: false,
              ...(isEditing ? { imagePath: editSource } : {}),
            }
            const mediaBlob = await postImageToRunner(livepeerRunner, imageBody, {
              signal: abortController.signal,
            })
            const objectUrl = URL.createObjectURL(mediaBlob)
            setState({
              isGenerating: false,
              progress: 100,
              statusMessage: 'Complete!',
              videoPath: null,
              imagePath: objectUrl,
              imagePaths: [objectUrl],
              error: null,
            })
            return
          }
        }

        // 2) Fall back to other API providers — FAL (Z-Image Turbo).
        if (shouldImageGenerateWithFalApi) {
          const settingsResult = await ApiClient.getSettings()
          const hasFalApiKey = settingsResult.ok ? settingsResult.data.hasFalApiKey : appSettings.hasFalApiKey
          if (!hasFalApiKey) {
            if (settingsResult.ok) void refreshSettings()
            openFalConnectDialog()
            return
          }
          const keyRes = await ApiClient.getFalApiKey()
          if (!keyRes.ok || !keyRes.data.falApiKey) {
            throw new Error('FAL API key required')
          }
          const falKey = keyRes.data.falApiKey
          const seed = 42
          let blob: Blob
          if (isEditing) {
            if (!editSource) throw new Error('Edit source image required')
            const srcRes = await fetch(editSource, { signal: abortController.signal })
            if (!srcRes.ok) throw new Error('Failed to load edit source image')
            const srcBuf = await srcRes.arrayBuffer()
            blob = await falGenerateI2I(falKey, {
              prompt: finalPrompt,
              imageDataUri: `data:image/png;base64,${arrayBufferToBase64(srcBuf)}`,
              strength: settings.imageEditStrength ?? 0.6,
              seed,
              numInferenceSteps: numSteps,
            })
          } else {
            blob = await falGenerateT2I(falKey, {
              prompt: finalPrompt,
              width: dims.width,
              height: dims.height,
              seed,
              numInferenceSteps: numSteps,
            })
          }
          const objectUrl = URL.createObjectURL(blob)
          setState({
            isGenerating: false,
            progress: 100,
            statusMessage: 'Complete!',
            videoPath: null,
            imagePath: objectUrl,
            imagePaths: [objectUrl],
            error: null,
          })
          return
        }

        // 3) No backend available.
        throw new Error('No image backend available. Enable a Livepeer image runner or connect a FAL AI key.')
      } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') {
          setState(prev => ({
            ...prev,
            isGenerating: false,
            statusMessage: 'Cancelled',
          }))
        } else {
          setState(prev => ({
            ...prev,
            isGenerating: false,
            error: createLocalGenerationError(error instanceof Error ? error.message : 'Unknown error'),
          }))
        }
      } finally {
        if (progressInterval) {
          clearInterval(progressInterval)
        }
      }
    })
  }, [appSettings, refreshSettings, shouldImageGenerateWithFalApi])

  const reset = useCallback(() => {
    clearRecoveryPolling()
    localStorage.removeItem(GENERATION_RECOVERY_KEY)
    setState({
      isGenerating: false,
      progress: 0,
      statusMessage: '',
      videoPath: null,
      imagePath: null,
      imagePaths: [],
      error: null,
    })
  }, [])

  return {
    ...state,
    generate,
    generateImage,
    cancel,
    reset,
    resumeIfRunning,
  }
}
