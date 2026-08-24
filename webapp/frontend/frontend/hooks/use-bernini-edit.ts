import { useCallback, useState } from 'react'
import { withGenerationActive } from '../lib/generation-active'
import { logger } from '../lib/logger'
import { resolveRunner, postRunnerTaskWithTicketSSE } from '../lib/direct-transport'
import { getBlob, isWebPath, registerBlob } from '../lib/runtime/web-store'
import { GENERATION_RECOVERY_KEY, GENERATION_RECOVERY_TS_KEY } from './use-generation'
import {
  berniniRunnerV2VBody,
  berniniRunnerR2VBody,
  berniniTaskFor,
  BERNINI_NATIVE_FPS,
  BERNINI_NATIVE_RESOLUTION,
  type BerniniEngine,
  type BerniniResolution,
  type BerniniOperation,
} from '../lib/bernini-delivery'

// Edit-Video Bernini rail (use-restyle analog). The frontend decides the goal ->
// endpoint (v2v motion-preserving edit | r2v reference-image -> video) and the
// backend stays strictly route-based. Render happens natively at 480p@16; any
// above-native delivery is requested via the `post` payload the live-runner
// orchestrates on the vp-worker (/process).

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
  return null
}

export interface BerniniEditSubmitParams {
  // The operation/goal determines the endpoint (v2v | r2v).
  operation: Exclude<BerniniOperation, 't2v'>
  // Source video (v2v edits it in place).
  videoPath: string
  // Reference images (r2v, 1.3B multi-reference). web:// asset paths.
  referencePaths?: string[]
  prompt: string
  engine: BerniniEngine
  // Native render fps; >16 asks the runner for a RIFE fps-boost post rail.
  fps?: number
  // Delivery resolution; above native asks for the FlashVSR post rail.
  resolution?: BerniniResolution
  duration?: number
  seed?: number
  negativePrompt?: string
}

export interface BerniniEditResult {
  videoPath: string
}

interface UseBerniniEditState {
  isEditing: boolean
  editStatus: string
  editError: string | null
  result: BerniniEditResult | null
}

function getPhaseMessage(phase: string): string {
  switch (phase) {
    case 'connecting_remote':
    case 'sending_to_remote':
      return 'Connecting to remote runner...'
    case 'preprocessing':
      return 'Preparing frames...'
    case 'generating':
      return 'Generating (Bernini)...'
    case 'decoding':
      return 'Decoding video...'
    case 'finalizing':
      return 'Finalizing output...'
    case 'complete':
      return 'Complete!'
    default:
      return 'Editing...'
  }
}

export function useBerniniEdit() {
  const [state, setState] = useState<UseBerniniEditState>({
    isEditing: false,
    editStatus: '',
    editError: null,
    result: null,
  })

  const submitBerniniEdit = useCallback(async (params: BerniniEditSubmitParams) => {
    if (!params.videoPath || !params.prompt) return

    setState({
      isEditing: true,
      editStatus: 'Connecting to remote runner...',
      editError: null,
      result: null,
    })

    await withGenerationActive(async () => {
      try {
        const spec = berniniTaskFor(params.operation)
        const runner = await resolveRunner([spec.capability])
        if (!runner) {
          const msg = 'No capable Livepeer runner is currently available for Bernini editing.'
          logger.error(`Bernini edit error: ${msg}`)
          setState({ isEditing: false, editStatus: '', editError: msg, result: null })
          return
        }

        const videoB64 = await assetToBase64(params.videoPath)
        if (!videoB64) {
          const msg = 'Could not read the source video for this edit.'
          logger.error(`Bernini edit error: ${msg}`)
          setState({ isEditing: false, editStatus: '', editError: msg, result: null })
          return
        }

        const target = {
          engine: params.engine,
          resolution: params.resolution ?? BERNINI_NATIVE_RESOLUTION,
          fps: params.fps ?? BERNINI_NATIVE_FPS,
          duration: Math.min(Math.max(params.duration ?? 3, 1), 5),
        }

        const opts = {
          negativePrompt: params.negativePrompt,
          seed: params.seed,
        }

        let body: Record<string, unknown>
        if (params.operation === 'r2v') {
          const refB64 = (params.referencePaths ?? []).filter(Boolean)
          const refs: string[] = []
          for (const p of refB64) {
            const b = await assetToBase64(p)
            if (b) refs.push(b)
          }
          if (refs.length === 0) {
            const msg = 'Reference-image editing (r2v) requires at least one reference image.'
            logger.error(`Bernini edit error: ${msg}`)
            setState({ isEditing: false, editStatus: '', editError: msg, result: null })
            return
          }
          body = berniniRunnerR2VBody(params.prompt, refs, target, opts)
        } else {
          body = berniniRunnerV2VBody(params.prompt, videoB64, target, opts)
        }

        logger.info(`[bernini-edit] rail=${spec.task} media=base64`)
        const res = await postRunnerTaskWithTicketSSE(runner, spec.task, body, {
          onProgress: (ev) => {
            const msg = getPhaseMessage(ev.stage || '')
            const isBackbone = ev.stage === 'generating'
            const pct = typeof ev.progress === 'number' ? ev.progress : null
            const showPct = isBackbone && pct !== null
            setState(prev => ({
              ...prev,
              editStatus: showPct ? `${msg} ${Math.min(Math.round(pct * 100), 99)}%` : msg,
            }))
          },
        })

        if (!res.mediaBlob) {
          const err = res.payload?.error ? String(res.payload.error) : 'Runner returned no media'
          logger.error(`Bernini edit error: ${err}`)
          setState({ isEditing: false, editStatus: '', editError: err, result: null })
          return
        }

        const videoPath = registerBlob(
          res.mediaBlob,
          'bernini-edit.mp4',
          (res.mediaBlob as Blob).type || 'video/mp4',
        )

        setState({
          isEditing: false,
          editStatus: 'Edit complete!',
          editError: null,
          result: { videoPath },
        })
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Bernini edit failed'
        logger.error(`Bernini edit error: ${msg}`)
        setState({ isEditing: false, editStatus: '', editError: msg, result: null })
      } finally {
        window.localStorage.removeItem(GENERATION_RECOVERY_KEY)
        window.localStorage.removeItem(GENERATION_RECOVERY_TS_KEY)
      }
    })
  }, [])

  const resetBerniniEdit = useCallback(() => {
    setState({
      isEditing: false,
      editStatus: '',
      editError: null,
      result: null,
    })
  }, [])

  return {
    submitBerniniEdit,
    resetBerniniEdit,
    isEditing: state.isEditing,
    editStatus: state.editStatus,
    editError: state.editError,
    berniniEditResult: state.result,
  }
}
