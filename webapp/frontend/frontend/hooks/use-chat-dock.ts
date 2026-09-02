import { useCallback, useEffect, useReducer, useRef } from 'react'
import {
  chatDockReducer,
  initialChatDockState,
  nextDockId,
  type ChatDockMessage,
  type DockPlacement,
  type DockView,
  type GenerationMessage,
  type GenerationStatus,
  type LLMTraceMessage,
} from '../lib/chat-dock'

/**
 * Owns the right-side chat-dock state: the message history (generations + chats)
 * and the dock's layout (collapsed/expanded, docked/floating). A thin
 * useReducer wrapper over the pure `chatDockReducer` in lib/chat-dock.
 */
/** A generation card left in 'running' past this cap is auto-marked 'error', so a task that
 *  never finishes (or whose completion handler was lost on a transport without an SSE
 *  watchdog) can't wedge the history on 'running' forever. Set slightly above the longest
 *  legitimate generation (SSE 25min max + 27min UI cap) so a real run isn't cut short.
 *  Client-side tracking only — it flips the card, the underlying runner abort is the
 *  generation rail's own job. */
const GENERATION_CARD_TIMEOUT_MS = 30 * 60 * 1000
const GENERATION_TIMEOUT_MESSAGE = "This task didn't finish, so tracking was stopped."

export function useChatDock() {
  const [state, dispatch] = useReducer(chatDockReducer, initialChatDockState)
  const timeoutRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())

  const addGeneration = useCallback((prompt: string, mode: string) => {
    const id = nextDockId('gen')
    dispatch({ type: 'add_generation', prompt, mode, id })
    return id
  }, [])

  const updateGeneration = useCallback(
    (id: string, patch: Partial<Pick<GenerationMessage, 'status' | 'resultPath' | 'stillPath' | 'error' | 'progress' | 'step' | 'totalSteps' | 'statusMessage' | 'preview'>>) => {
      dispatch({ type: 'update_generation', id, patch })
    },
    [],
  )

  const markGenerationDone = useCallback(
    (id: string, resultPath: string, stillPath?: string) =>
      updateGeneration(id, { status: 'done', resultPath, stillPath: stillPath ?? resultPath }),
    [updateGeneration],
  )

  const markGenerationError = useCallback(
    (id: string, error: string) => updateGeneration(id, { status: 'error', error }),
    [updateGeneration],
  )

  const addChat = useCallback((role: 'user' | 'assistant', text: string) => {
    dispatch({ type: 'add_chat', role, text })
  }, [])

  const addLlmTrace = useCallback(
    (payload: Omit<LLMTraceMessage, 'kind' | 'id' | 'createdAt'>) => {
      dispatch({
        type: 'add_llm_trace',
        label: payload.label,
        sent: payload.sent,
        response: payload.response,
        reasoning: payload.reasoning,
        appliedTo: payload.appliedTo,
      })
    },
    [],
  )

  const setView = useCallback((view: DockView) => dispatch({ type: 'set_view', view }), [])
  const setPlacement = useCallback(
    (placement: DockPlacement) => dispatch({ type: 'set_placement', placement }),
    [],
  )
  const toggleCollapse = useCallback(
    () => dispatch({ type: 'set_view', view: state.view === 'expanded' ? 'collapsed' : 'expanded' }),
    [state.view],
  )
  const togglePlacement = useCallback(
    () =>
      dispatch({
        type: 'set_placement',
        placement: state.placement === 'docked' ? 'floating' : 'docked',
      }),
    [state.placement],
  )
  // Mark generation cards whose produced asset was deleted from the project, so
  // the live timeline shows a "deleted" placeholder instead of a broken preview
  // (and they are dropped entirely when the project is reopened).
  const markGenerationDeleted = useCallback((paths: string[]) => {
    if (paths.length === 0) return
    dispatch({ type: 'mark_generation_deleted', paths })
  }, [])

  const clear = useCallback(() => dispatch({ type: 'clear' }), [])

  // Client-side timeout backstop: schedule a timer per running generation card; if it's
  // STILL running when the cap elapses, mark it 'error' so it stops showing as running.
  useEffect(() => {
    const running = state.messages.filter(
      (m): m is GenerationMessage => m.kind === 'generation' && m.status === 'running',
    )
    const timers = timeoutRef.current
    // Drop timers for cards that are no longer running / were removed.
    for (const id of [...timers.keys()]) {
      if (!running.some(m => m.id === id)) {
        clearTimeout(timers.get(id)!)
        timers.delete(id)
      }
    }
    for (const m of running) {
      if (timers.has(m.id)) continue
      timers.set(
        m.id,
        setTimeout(() => {
          timers.delete(m.id)
          dispatch({
            type: 'update_generation',
            id: m.id,
            patch: { status: 'error', error: GENERATION_TIMEOUT_MESSAGE },
          })
        }, GENERATION_CARD_TIMEOUT_MS),
      )
    }
  }, [state.messages])

  // Clear all card timers on unmount.
  useEffect(() => {
    const timers = timeoutRef.current
    return () => {
      for (const t of timers.values()) clearTimeout(t)
      timers.clear()
    }
  }, [])

  // User asks to stop tracking a stuck task client-side: mark its card 'error' immediately.
  // Display-only — the underlying generation rail is cancelled separately via its own cancel.
  const stopTracking = useCallback((id: string) => {
    dispatch({
      type: 'update_generation',
      id,
      patch: { status: 'error', error: 'Stopped tracking (you cancelled it).' },
    })
  }, [])

  // Bulk-restore persisted history (loaded from the project's timeline JSON).
  const replaceMessages = useCallback((messages: ChatDockMessage[]) => {
    dispatch({ type: 'replace_messages', messages })
  }, [])

  return {
    messages: state.messages as ChatDockMessage[],
    view: state.view,
    placement: state.placement,
    addGeneration,
    updateGeneration,
    markGenerationDone,
    markGenerationError,
    markGenerationDeleted,
    addChat,
    addLlmTrace,
    stopTracking,
    setView,
    setPlacement,
    toggleCollapse,
    togglePlacement,
    clear,
    replaceMessages,
  }
}

export type { GenerationStatus }
