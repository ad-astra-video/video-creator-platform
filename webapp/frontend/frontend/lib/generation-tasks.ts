// Pure task-queue state for the background generation manager.
// Kept dependency-free (no React, no electronAPI) so it is trivially unit-testable.

export type GenerationKind = 'video' | 'image'
export type GenerationTaskStatus = 'queued' | 'running' | 'completed' | 'error'

export interface GenerationTask {
  id: string
  label: string
  /** Which project the finished output should be saved into. */
  projectId: string | null
  kind: GenerationKind
  status: GenerationTaskStatus
  error?: string
  /** web:// key of the produced asset (set on completion). */
  assetKey?: string
  createdAt: number
  startedAt?: number
  completedAt?: number
}

export interface TaskToast {
  id: string
  message: string
  tone: 'success' | 'error' | 'info'
  createdAt: number
}

/** Cap the on-screen task list so a long burst can't pin unbounded memory. */
export const MAX_TASKS = 100
/** Auto-dismiss a toast after this many ms. */
export const TOAST_TTL_MS = 6500

export function makeId(prefix: string): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return `${prefix}-${crypto.randomUUID()}`
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

export type TasksAction =
  | { type: 'ADD'; task: GenerationTask }
  | { type: 'START'; id: string; at: number }
  | { type: 'COMPLETE'; id: string; at: number; assetKey: string }
  | { type: 'FAIL'; id: string; at: number; error: string }
  | { type: 'REMOVE'; id: string }
  | { type: 'CLEAR_COMPLETED' }

export function generationTaskReducer(tasks: GenerationTask[], action: TasksAction): GenerationTask[] {
  switch (action.type) {
    case 'ADD':
      return [action.task, ...tasks].slice(0, MAX_TASKS)
    case 'START':
      return tasks.map(t => (t.id === action.id && t.status === 'queued' ? { ...t, status: 'running', startedAt: action.at } : t))
    case 'COMPLETE':
      return tasks.map(t => (t.id === action.id && (t.status === 'running' || t.status === 'queued')
        ? { ...t, status: 'completed', assetKey: action.assetKey, completedAt: action.at }
        : t))
    case 'FAIL':
      return tasks.map(t => (t.id === action.id && (t.status === 'running' || t.status === 'queued')
        ? { ...t, status: 'error', error: action.error, completedAt: action.at }
        : t))
    case 'REMOVE':
      return tasks.filter(t => t.id !== action.id)
    case 'CLEAR_COMPLETED':
      return tasks.filter(t => t.status !== 'completed')
    default:
      return tasks
  }
}

export function runningCount(tasks: GenerationTask[]): number {
  return tasks.filter(t => t.status === 'running' || t.status === 'queued').length
}

export function elidedLabel(label: string, max = 48): string {
  return label.length > max ? `${label.slice(0, max - 1)}…` : label
}
