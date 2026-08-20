// Global background-generation task manager.
//
// The generation is dispatched HERE (not inside a view component that may unmount when the
// user navigates), so an in-flight call keeps running after the user leaves the project /
// switches views, and its finished output is persisted into the right project folder on disk.
//
// There is deliberately NO single-flight lock: every submit starts its own independent runner
// call, so the user can fire many generations while runners are available. The runner's own
// 503 retry/backoff absorbs "no free GPU right now" capacity — the browser just needs to start
// the call.
//
// On success the manager is the SOLE place that persists the output (addVisualAssetToProject →
// disk copy + thumbnails via the web electronAPI shim, then addAsset into the project model), so
// navigating away can never drop the save, and there is exactly one asset per completed task.
//
// Completion is surfaced only through the persistent task list (no toasts): a task is marked
// completed/error and the user clears rows from the bottom-right panel as they please.

import React, { createContext, useCallback, useContext, useMemo, useReducer, useRef } from 'react'
import { useProjects } from './ProjectContext'
import { addVisualAssetToProject } from '../lib/asset-copy'
import {
  type GenerationKind,
  type GenerationTask,
  generationTaskReducer,
  makeId,
} from '../lib/generation-tasks'
import { isWebPath } from '../lib/runtime/web-store'
import { logger } from '../lib/logger'

export interface GenerationRunContext {
  /** Optional coarse progress 0..1 used to render a progress bar in the task list. */
  onProgress?: (p: number) => void
}

export interface SubmitGenerationInput {
  label: string
  kind: GenerationKind
  /** Project that receives the finished asset. Pass null to skip project persistence. */
  projectId?: string | null
  /**
   * Performs the actual generation and MUST resolve to a `web://<uuid>` asset key when done.
   * It may be an SSE/WS/HTTP runner call of any duration; the provider keeps the promise alive
   * independent of any component that called submitGenerationTask.
   */
  run: (ctx: GenerationRunContext) => Promise<string>
}

interface GenerationTaskContextType {
  tasks: GenerationTask[]
  runningCount: number
  submitGenerationTask: (input: SubmitGenerationInput) => string
  // result-tracking API for callers that keep driving their own transport (e.g. GenSpace's
  // existing use-generation) but want the output in the task list.
  trackGeneration: (input: { label: string; kind: GenerationKind; projectId?: string | null }) => string
  generationCompleted: (id: string, label: string, assetKey: string) => void
  generationFailed: (id: string, label: string, error: string) => void
  clearTask: (id: string) => void
  clearCompleted: () => void
}

const GenerationTaskContext = createContext<GenerationTaskContextType | null>(null)

export function GenerationTaskProvider({ children }: { children: React.ReactNode }) {
  const { addAsset } = useProjects()
  const [tasks, dispatch] = useReducer(generationTaskReducer, [])
  const nextTasksRef = useRef<Map<string, SubmitGenerationInput['run']>>(new Map())

  const persistCompleted = useCallback(
    async (assetKey: string, projectId: string | null, kind: GenerationKind, prompt: string | undefined) => {
      if (!projectId || !isWebPath(assetKey)) return
      try {
        const copied = await addVisualAssetToProject(assetKey, projectId, kind)
        if (!copied) throw new Error('disk copy failed (no assets folder selected?)')
        addAsset(projectId, {
          type: kind,
          path: copied.path,
          bigThumbnailPath: copied.bigThumbnailPath,
          smallThumbnailPath: copied.smallThumbnailPath,
          width: copied.width,
          height: copied.height,
          prompt: prompt || '',
          resolution: '',
        })
      } catch (e) {
        // Persisting to the project folder is best-effort: the raw web:// asset is still in the
        // browser store (IndexedDB-mirrored), so the user can save/download it manually.
        logger.warn(`Failed to persist completed task to project folder: ${e instanceof Error ? e.message : String(e)}`)
      }
    },
    [addAsset],
  )

  const submitGenerationTask = useCallback(
    (input: SubmitGenerationInput): string => {
      const id = makeId('gen')
      const now = Date.now()
      const projectId = input.projectId ?? null
      const kind = input.kind
      dispatch({
        type: 'ADD',
        task: { id, label: input.label, projectId, kind, status: 'queued', createdAt: now },
      })
      nextTasksRef.current.set(id, input.run)

      // Start on the next microtask so the task row is visible ("queued") before it flips to running.
      void (async () => {
        const run = nextTasksRef.current.get(id)
        if (!run) return
        nextTasksRef.current.delete(id)
        dispatch({ type: 'START', id, at: Date.now() })
        try {
          const assetKey = await run({ onProgress: () => { /* progress bar reserved for a later slice */ } })
          if (!isWebPath(assetKey)) throw new Error('generation returned an invalid asset reference')
          await persistCompleted(assetKey, projectId, kind, undefined)
          dispatch({ type: 'COMPLETE', id, at: Date.now(), assetKey })
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e)
          dispatch({ type: 'FAIL', id, at: Date.now(), error: msg })
        }
      })()

      return id
    },
    [persistCompleted],
  )

  const trackGeneration = useCallback(
    (input: { label: string; kind: GenerationKind; projectId?: string | null }): string => {
      const id = makeId('gen')
      const now = Date.now()
      dispatch({
        type: 'ADD',
        task: { id, label: input.label, projectId: input.projectId ?? null, kind: input.kind, status: 'running', startedAt: now, createdAt: now },
      })
      return id
    },
    [],
  )

  const generationCompleted = useCallback((id: string, _label: string, assetKey: string) => {
    dispatch({ type: 'COMPLETE', id, at: Date.now(), assetKey })
  }, [])

  const generationFailed = useCallback((id: string, _label: string, error: string) => {
    dispatch({ type: 'FAIL', id, at: Date.now(), error })
  }, [])

  const clearTask = useCallback((id: string) => dispatch({ type: 'REMOVE', id }), [])
  const clearCompleted = useCallback(() => dispatch({ type: 'CLEAR_COMPLETED' }), [])

  const running = useMemo(() => tasks.filter(t => t.status === 'running' || t.status === 'queued').length, [tasks])

  const value = useMemo<GenerationTaskContextType>(
    () => ({ tasks, runningCount: running, submitGenerationTask, trackGeneration, generationCompleted, generationFailed, clearTask, clearCompleted }),
    [tasks, running, submitGenerationTask, trackGeneration, generationCompleted, generationFailed, clearTask, clearCompleted],
  )

  return <GenerationTaskContext.Provider value={value}>{children}</GenerationTaskContext.Provider>
}

export function useGenerationTasks(): GenerationTaskContextType {
  const ctx = useContext(GenerationTaskContext)
  if (!ctx) throw new Error('useGenerationTasks must be used within a GenerationTaskProvider')
  return ctx
}
