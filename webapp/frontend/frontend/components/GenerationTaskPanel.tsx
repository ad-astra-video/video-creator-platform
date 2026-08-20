// Fixed bottom-right generation task list.
// Reads the global GenerationTaskProvider (mounted above routing), so the list survives
// navigation and shows every background generation the user has fired. No toasts: tasks are
// marked completed/error in the list and the user clears rows as needed.

import { useEffect, useState } from 'react'
import { CheckCircle2, ChevronDown, ListTodo, Loader2, X, XCircle } from 'lucide-react'
import { useGenerationTasks } from '../contexts/GenerationTaskContext'
import { elidedLabel } from '../lib/generation-tasks'

function relativeTime(ms: number): string {
  const s = Math.max(0, Math.round((Date.now() - ms) / 1000))
  if (s < 5) return 'just now'
  if (s < 60) return `${s}s ago`
  const m = Math.round(s / 60)
  if (m < 60) return `${m}m ago`
  return `${Math.round(m / 60)}h ago`
}

function StatusIcon({ status }: { status: string }) {
  if (status === 'running') return <Loader2 className="h-4 w-4 animate-spin text-zinc-300" />
  if (status === 'completed') return <CheckCircle2 className="h-4 w-4 text-emerald-400" />
  if (status === 'error') return <XCircle className="h-4 w-4 text-red-400" />
  return <Loader2 className="h-4 w-4 text-zinc-500" /> // queued
}

/** Live-updating ("5s ago") labels. */
function useNow(interval = 15000): number {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), interval)
    return () => clearInterval(t)
  }, [interval])
  return now
}

export function GenerationTaskPanel() {
  const { tasks, runningCount: running, clearCompleted, clearTask } = useGenerationTasks()
  useNow()

  if (tasks.length === 0) return null

  return (
    <div className="fixed bottom-4 right-4 z-[80] w-96 max-w-[calc(100vw-2rem)]">
      <div className="overflow-hidden rounded-xl border border-zinc-700/80 bg-zinc-900/95 shadow-2xl backdrop-blur">
        <div className="flex w-full items-center justify-between gap-2 px-3 py-2 text-sm font-medium text-zinc-200">
          <span className="flex items-center gap-2">
            <ListTodo className="h-4 w-4 text-zinc-400" />
            Tasks
            {running > 0 ? (
              <span className="flex items-center gap-1 rounded-full bg-zinc-800 px-2 py-0.5 text-xs text-zinc-300">
                <Loader2 className="h-3 w-3 animate-spin" />
                {running} running
              </span>
            ) : (
              <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-xs text-zinc-400">{tasks.length} total</span>
            )}
          </span>
          <ChevronDown className="h-4 w-4 text-zinc-500" />
        </div>

        <div className="border-t border-zinc-800">
          <div className="max-h-72 overflow-y-auto">
            {tasks.map(t => (
              <div key={t.id} className="flex items-start gap-2 border-b border-zinc-800/60 px-3 py-2 last:border-0">
                <div className="mt-0.5 shrink-0"><StatusIcon status={t.status} /></div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-sm text-zinc-200" title={t.label}>{elidedLabel(t.label)}</span>
                    <span className="shrink-0 text-[10px] uppercase tracking-wide text-zinc-500">{t.kind}</span>
                  </div>
                  <div className="mt-0.5 flex items-center gap-2 text-[11px] text-zinc-500">
                    <span>{relativeTime(t.completedAt ?? t.startedAt ?? t.createdAt)}</span>
                    {t.status === 'completed' && <span className="text-emerald-500">saved to project</span>}
                    {t.status === 'queued' && <span>queued</span>}
                  </div>
                  {t.status === 'error' && (
                    <div className="mt-1 line-clamp-2 text-xs text-red-300" title={t.error}>
                      {elidedLabel(t.error ?? 'Generation failed', 90)}
                    </div>
                  )}
                </div>
                {(t.status === 'completed' || t.status === 'error') && (
                  <button
                    onClick={() => clearTask(t.id)}
                    className="shrink-0 rounded p-0.5 text-zinc-500 hover:bg-white/10 hover:text-zinc-200"
                    aria-label="Dismiss task"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            ))}
          </div>
          {tasks.some(t => t.status === 'completed') && (
            <div className="border-t border-zinc-800 px-3 py-1.5 text-right">
              <button onClick={clearCompleted} className="text-xs text-zinc-500 hover:text-zinc-300">Clear completed</button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
