import { useState, useCallback, useRef, useEffect, type RefObject } from 'react'
import type { GenerationSettings } from '../components/SettingsPanel'
import { ApiClient } from '../lib/api-client'
import { activeRecoveryMarkerExists } from '../lib/generation-progress-poll'
import {
  falGenerateI2I,
  falGenerateT2I,
  makeJobId,
  pathToBase64,
  postImageToRunnerSSE,
  postRunnerTaskWithTicketSSE,
  resolveRunner,
  type RunnerDto,
  type RunnerProgressEvent,
} from '../lib/direct-transport'
import { createLocalGenerationError, type GenerationError } from '../lib/generation-errors'
import { withGenerationActive } from '../lib/generation-active'
import { useAppSettings } from '../contexts/AppSettingsContext'
import {
  berniniRunnerT2VBody,
  berniniRunnerR2VBody,
  berniniTaskFor,
  BERNINI_NATIVE_FPS,
  BERNINI_NATIVE_FRAMES,
  type BerniniEngine,
  type BerniniResolution,
} from '../lib/bernini-delivery'

/** True when a GenerationSettings.model is a Bernini engine id. */
function isBerniniModel(model: string): model is BerniniEngine {
  return model === '1.3b' || model === '14b'
}


export const GENERATION_RECOVERY_KEY = 'ltx-generation-recovery'
// Parallel timestamp for the same marker: a lease so a crash-leaked marker can't permanently
// wedge the global generation lock. Written by writeRecoveryContext when the marker goes live;
// refreshed while a generation is genuinely in flight (see generation-progress-poll); any marker
// older than the lease (or lacking a timestamp, i.e. from a pre-lease build) is stale and purged.
export const GENERATION_RECOVERY_TS_KEY = 'ltx-generation-recovery-ts'
export const GENERATION_RECOVERY_LEASE_MS = 10 * 60 * 1000

// The web build injects a stub electronAPI (main.tsx -> createWebElectronAPI) whose
// platform is 'web'; the desktop has no such element. Same detection the global
// generation-lock and livepeer-discovery use. On web there is no local backend slot and
// no direct rail after a reload, so a leftover recovery marker can never correspond to a
// genuinely-running job — it must never be treated as one (see resumeIfRunning).
function isWebPlatform(): boolean {
  try {
    return (window as unknown as { electronAPI?: { platform?: string } }).electronAPI?.platform === 'web'
  } catch {
    return false
  }
}

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
  generate: (prompt: string, imagePath: string | null, settings: GenerationSettings,
              audioPath?: string | null, onProgress?: (ev: RunnerProgressEvent) => void) => Promise<void>
  generateImage: (prompt: string, settings: GenerationSettings, editSource?: string | null) => Promise<void>
  cancel: () => void
  reset: () => void
  resumeIfRunning: () => Promise<'running' | 'complete' | 'none'>
  /** Seed used by the most recent image generation (text-to-image or edit), so
   *  callers can persist it as regeneration metadata. Null before any image gen. */
  latestImageSeedRef: RefObject<number | null>
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
  const latestImageSeedRef = useRef<number | null>(null)

  const clearRecoveryPolling = () => {
    if (recoveryIntervalRef.current) {
      clearInterval(recoveryIntervalRef.current)
      recoveryIntervalRef.current = null
    }
  }

  useEffect(() => clearRecoveryPolling, [])

  // HARD watchdog: while isGenerating is true, force it back to false after a max
  // wall-clock cap no matter what the transport does. The SSE reader has its own
  // idle (90s) + max-duration (25min) watchdog that makes a generation promise
  // ALWAYS settle, but this is the final guarantee at the UI-state layer: even if
  // some future path forgets to end its await, the webapp can never be wedged on
  // "generating". On fire it aborts the in-flight request and surfaces an error.
  const GENERATION_MAX_MS = 27 * 60 * 1000

  useEffect(() => {
    if (!state.isGenerating) return
    const timer = setTimeout(() => {
      abortControllerRef.current?.abort()
      setState(prev => {
        if (!prev.isGenerating) return prev
        return {
          ...prev,
          isGenerating: false,
          statusMessage: 'Generation timed out',
          error: createLocalGenerationError(
            `Generation did not complete within ${Math.round(GENERATION_MAX_MS / 60000)} minutes. ` +
            'It was stopped to avoid the app hanging. Please retry.',
          ),
        }
      })
    }, GENERATION_MAX_MS)
    return () => clearTimeout(timer)
  }, [state.isGenerating])

  // Re-attach to a generation that was running OR finished while the frontend was
  // unmounted. Polls the backend progress endpoint; localStorage recovery context
  // (inputs, settings incl. loras) is owned by the caller (GenSpace). Returns the
  // recovered status so the caller can restore context for 'running' AND 'complete'
  // (a generation that finished during the unmount window still needs its metadata).
  const resumeIfRunning = useCallback(async (): Promise<'running' | 'complete' | 'none'> => {
    // The webapp has no backend generation-job row — the /api/generation/progress endpoint was
    // removed, so there is no server progress/resume or server 'complete' result to restore
    // (a reloaded tab's direct-rail result is gone). The only in-flight signal is the local
    // recovery marker: if its lease is live, treat the generation as still running and restore
    // a generic generating state; the owning flow clears the marker on completion.
    if (isWebPlatform()) {
      // WEB: no local backend slot AND the direct-rail SSE is lost on reload — nothing can be
      // resumed. A leftover recovery marker (a generation whose process died before its finally,
      // or that finished while this tab was closed) is therefore NOT proof a job is still
      // running. Treating it as such would wedge isGenerating/'Generating...' and disable the
      // Generate button forever even though a runner is available — exactly the generation lock
      // the webapp must not have. Clear the marker and report none so a new generation starts
      // immediately; a stale marker self-heals on the next mount.
      clearRecoveryPolling()
      localStorage.removeItem(GENERATION_RECOVERY_KEY)
      localStorage.removeItem(GENERATION_RECOVERY_TS_KEY)
      setState(prev => ({ ...prev, isGenerating: false, statusMessage: '' }))
      return 'none'
    }
    if (activeRecoveryMarkerExists()) {
      setState(prev => ({ ...prev, isGenerating: true, progress: 0, statusMessage: 'Generating...' }))
      return 'running'
    }
    clearRecoveryPolling()
    setState(prev => ({ ...prev, isGenerating: false, statusMessage: '' }))
    return 'none'
  }, [])

  const generate = useCallback(async (
    prompt: string,
    imagePath: string | null,
    settings: GenerationSettings,
    audioPath?: string | null,
    onProgress?: (ev: RunnerProgressEvent) => void,
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
        const lorasArr: Array<Record<string, unknown>> = settings.loras?.length
          ? settings.loras.map(l => ({ ref: l.ref, scale: l.scale }))
          : []
        // Option-A custom LoRA (user-supplied HF URL). Added to the runner loras array as
        // {custom_url, scale?, hf_token?} — the runner's _resolve_loras downloads it from an
        // allowlisted https host with the optional token.
        const customLora = (settings as { customLora?: { url?: string; token?: string; scale?: number } }).customLora
        if (customLora?.url) {
          lorasArr.push({ custom_url: customLora.url, scale: customLora.scale ?? 1.0, hf_token: customLora.token || undefined })
        }
        if (lorasArr.length) {
          body.loras = lorasArr
        }

        // Direct transport is the ONLY video backend in the webapp (/api/generate is 410 — the
        // Worker no longer carries the media path). Gate on the Livepeer video toggle. On the
        // web build resolveRunner() discovers straight against the configured Discovery URL
        // (client-side, from appSettings.livepeerDiscoveryUrl) — no Worker /api/providers call.
        const useDirectTransport = appSettings.livepeerVideoEnabled !== false

        // DIRECT transport: resolve a capable runner (Worker discovery) and do the Livepeer
        // payment handshake + read the resulting media from the runner response.
        if (useDirectTransport) {
          // ── Bernini rail: when the user picks a Bernini engine, route to the
          //    wan-worker /video-creator/v1/bernini-{t2v,r2v} rail. Native 480p@16
          //    render; above-native delivery is requested via the `post` payload the
          //    live-runner orchestrates on the vp-worker (/process). The runner (not
          //    the frontend) decides the upstream engine from `model`.
          if (isBerniniModel(settings.model)) {
            const engine: BerniniEngine = settings.model
            // A start image attached => reference-image-to-video (r2v, multi-ref
            // native to 1.3B) so the generation is actually conditioned on the image;
            // otherwise plain text-to-video.
            const isImageToVideo = !!imagePath
            const spec = berniniTaskFor(isImageToVideo ? 'r2v' : 't2v')
            const runner = await resolveRunner([spec.capability], { model: engine })
            if (!runner) {
              throw new Error('No available Livepeer runner for Bernini video generation')
            }
            const target = {
              engine,
              resolution: (body.resolution ?? '480p') as BerniniResolution,
              fps: typeof body.fps === 'number' ? (body.fps as number) : BERNINI_NATIVE_FPS,
              duration: Math.min(Math.max(typeof body.duration === 'number' ? (body.duration as number) : 3, 1), 5),
            }
            const negativePrompt = (settings as { negativePrompt?: string }).negativePrompt
            // The start image MUST be sent as base64 bytes — a remote worker can't
            // resolve a browser web:// or blob: path.
            let berniniPayload: Record<string, unknown>
            if (isImageToVideo) {
              const image_b64 = await pathToBase64(imagePath)
              if (!image_b64) {
                throw new Error('Could not read the start image for Bernini image-to-video generation')
              }
              berniniPayload = berniniRunnerR2VBody(prompt, [image_b64], target, { negativePrompt })
            } else {
              berniniPayload = berniniRunnerT2VBody(prompt, target, { negativePrompt })
            }
            const res = await postRunnerTaskWithTicketSSE(
              runner,
              spec.task,
              {
                ...berniniPayload,
                jobId: makeJobId(),
                // Bernini's native output IS the 81-frame clip at 16fps (~5s). The model
                // spec advertises fps=16 + a duration that represents those 81 frames;
                // send the native frame count exactly so the delivered clip is the full
                // render (duration*fps at 5s would give 80, not the intended 81).
                num_frames: BERNINI_NATIVE_FRAMES,
              },
              {
                signal: abortController.signal,
                onProgress: (ev) => {
                  if (!shouldApplyPollingUpdates) return
                  let displayProgress = 0
                  if (ev.stage === 'generating' && typeof ev.progress === 'number') {
                    displayProgress = Math.min(Math.round(ev.progress * 95), 95)
                  }
                  setState(prev => ({ ...prev, progress: displayProgress, statusMessage: ev.message || 'Generating...' }))
                  onProgress?.(ev)
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

          // ── Generic (LTX) video rail: route by conditioning — a start image attached
          //    => image-to-video (runner /i2v with the image bytes), no image => t2v.
          //    The image MUST be sent as base64 bytes — a remote worker can't resolve a
          //    browser web:// or blob: path, so never send the path for i2v.
          const isI2V = !!imagePath
          const runner = await resolveRunner(isI2V ? ['i2v'] : ['t2v'])
          if (!runner) {
            throw new Error(
              isI2V
                ? 'No available Livepeer runner for image-to-video generation'
                : 'No available Livepeer runner for video generation',
            )
          }
          if (isI2V) {
            const image_b64 = await pathToBase64(imagePath)
            if (!image_b64) {
              throw new Error('Could not read the start image for image-to-video generation')
            }
            body.image_base64 = image_b64
            delete body.imagePath
          }
          const res = await postRunnerTaskWithTicketSSE(
            runner,
            isI2V ? 'generate-i2v' : 'generate',
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
                onProgress?.(ev)
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
            const imageSeed = Math.floor(Math.random() * 0x7fffffff)
            // Engine selection: image EDIT -> Qwen-Image-Edit (default) vs Z-Image
            // keep-subject; text-to-image -> Z-Image Turbo (default) vs FLUX.2 Klein 4B.
            // Forwarded to the runner's /video-creator/v1/edit|image so the engine is
            // selected server-side.
            const engine = isEditing
              ? (settings.imageEditEngine ?? 'qwen-edit')
              : (settings.imageModel ?? 'zimage')
            // Edits go to the /edit endpoint (accepts qwen-edit|zimage) and MUST carry
            // the source image as base64 bytes — a remote runner can't fetch a browser
            // web:// or blob: path. T2I goes to /image (accepts zimage|klein) with a prompt.
            const srcEditTask = isEditing ? 'edit' : 'generate-image'
            let imageBody: Record<string, unknown>
            if (isEditing) {
              if (!editSource) throw new Error('Edit source image required')
              const image = await pathToBase64(editSource)
              if (!image) throw new Error('Could not read the edit source image')
              imageBody = {
                prompt: finalPrompt,
                image,
                engine,
                seed: imageSeed,
                quality: settings.imageEditQuality ?? 'balanced',
                strength: settings.imageEditStrength ?? 0.6,
                keepSubject: false,
              }
            } else {
              // Worker owns the per-model Fast/Balanced/High -> step map; send the
              // quality NAME so Z-Image/Klein/HiDream each get their own step count.
              imageBody = {
                prompt: finalPrompt,
                width: dims.width,
                height: dims.height,
                quality: settings.imageQuality ?? 'balanced',
                numImages,
                seed: imageSeed,
                engine,
              }
            }
            const imgRes = await postImageToRunnerSSE(livepeerRunner, imageBody, {
              signal: abortController.signal,
              onProgress: (ev) => {
                setState(prev => ({
                  ...prev,
                  statusMessage: ev.message || prev.statusMessage,
                }))
              },
            }, srcEditTask)
            // Persist the ACTUAL seed the runner used (it echoes it back), falling
            // back to the one we sent — so metadata always shows a real seed.
            latestImageSeedRef.current = typeof imgRes.seed === 'number' ? imgRes.seed : imageSeed
            const objectUrl = URL.createObjectURL(imgRes.blob)
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
          const seed = Math.floor(Math.random() * 0x7fffffff)
          latestImageSeedRef.current = seed
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
    latestImageSeedRef,
  }
}
