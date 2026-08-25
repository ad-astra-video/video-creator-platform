import { useEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { ChevronLeft, ChevronRight, Maximize2, Minimize2, MessageSquare, Trash2, GripHorizontal } from 'lucide-react'
import type { ChatDockMessage, DockPlacement, DockView } from '../lib/chat-dock'
import { DockMessageList } from './chat-dock-messages'

/**
 * Right-side chat dock for the GenSpace shell — the prompt panel that hosts the
 * message history (generations vs chats).
 *
 * Three layout states, driven by `view` + `placement`:
 *   - collapsed rail      (view='collapsed')       — a slim icon rail pinned to
 *     the canvas's right edge; keeps the canvas maximized.
 *   - expanded, docked    (view='expanded', placement='docked') — a fixed-width
 *     column on the right holding the history + the prompt content.
 *   - floating modal      (placement='floating')   — the same content detached
 *     into a draggable floating panel over the canvas (the "undock" state).
 *
 * ``children`` is the prompt/settings bar (the existing PromptBar) pinned to the
 * panel's bottom; the message history sits above it and is independently
 * scrollable.
 */
export interface ChatDockProps {
  messages: ChatDockMessage[]
  view: DockView
  placement: DockPlacement
  modeLabel?: string
  onToggleCollapse: () => void
  onTogglePlacement: () => void
  onClear: () => void
  /** Optional: mark a running generation card as stopped (client-side tracking stop). */
  onStopTrack?: (id: string) => void
  /** The prompt / settings content pinned at the bottom of the dock. */
  children: ReactNode
  /** Column width when expanded/docked or floating (px). */
  width?: number
  /** Called with the new width while the user drags the dock's resize handle. */
  onWidthChange?: (width: number) => void
}

export const DOCK_MIN_WIDTH = 360
export const DOCK_MAX_WIDTH = 720
export const DOCK_DEFAULT_WIDTH = 480

export function ChatDock({
  messages,
  view,
  placement,
  modeLabel,
  onToggleCollapse,
  onTogglePlacement,
  onClear,
  onStopTrack,
  children,
  width = DOCK_DEFAULT_WIDTH,
  onWidthChange,
}: ChatDockProps) {
  // ── Floating drag state ────────────────────────────────────────────────────
  const [pos, setPos] = useState(() => ({
    x: typeof window !== 'undefined' ? Math.max(16, window.innerWidth - width - 24) : 0,
    y: 72,
  }))
  const dragRef = useRef<{
    active: boolean
    startX: number
    startY: number
    origX: number
    origY: number
  }>({ active: false, startX: 0, startY: 0, origX: 0, origY: 0 })

  useEffect(() => {
    if (placement !== 'floating') return
    const move = (e: PointerEvent) => {
      const d = dragRef.current
      if (!d.active) return
      setPos({ x: d.origX + (e.clientX - d.startX), y: d.origY + (e.clientY - d.startY) })
    }
    const up = () => { dragRef.current.active = false }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
  }, [placement])

  const onDragStart = (e: React.PointerEvent) => {
    if (placement !== 'floating') return
    dragRef.current = { active: true, startX: e.clientX, startY: e.clientY, origX: pos.x, origY: pos.y }
  }

  // ── Resize (drag a left/right edge to change width) ────────────────────────
  // deltaFn(dx) -> newWidth: docked LEFT edge shrinks when dragged right
  // (startW + (startX - clientX)); floating RIGHT edge grows (startW + (clientX - startX)).
  const onResizeStart = (e: React.PointerEvent, deltaFn: (dx: number) => number) => {
    if (!onWidthChange) return
    if (e.button !== 0) return
    e.preventDefault()
    e.stopPropagation()
    const startX = e.clientX
    const move = (ev: PointerEvent) => {
      const next = Math.min(DOCK_MAX_WIDTH, Math.max(DOCK_MIN_WIDTH, deltaFn(ev.clientX - startX)))
      onWidthChange(Math.round(next))
    }
    const up = () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
      window.removeEventListener('pointercancel', up)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    window.addEventListener('pointercancel', up)
  }

  const dockResizeHandle = (deltaFn: (dx: number) => number) => (
    <div
      onPointerDown={(e) => onResizeStart(e, deltaFn)}
      className="absolute top-0 bottom-0 w-1.5 cursor-col-resize hover:bg-emerald-500/50 active:bg-emerald-500/70 transition-colors select-none"
      title="Drag to resize"
      aria-hidden="true"
    />
  )

  // Auto-scroll history to the newest message.
  const listRef = useRef<HTMLDivElement>(null)
  const lastLen = useRef(messages.length)
  useEffect(() => {
    if (messages.length === lastLen.current) return
    lastLen.current = messages.length
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight
  }, [messages.length])

  const header = (dragHandle: boolean) => (
    <div
      className={`flex items-center gap-1 px-2.5 py-2 border-b border-zinc-800 flex-shrink-0 select-none ${
        dragHandle ? 'cursor-grab active:cursor-grabbing' : ''
      }`}
      onPointerDown={dragHandle ? onDragStart : undefined}
    >
      {dragHandle && <GripHorizontal className="h-3.5 w-3.5 text-zinc-600" />}
      <MessageSquare className="h-3.5 w-3.5 text-emerald-400" />
      <span className="text-[11px] font-semibold text-white flex-1 truncate">
        Project Chat
        {modeLabel && (
          <span className="ml-2 text-[9px] font-normal uppercase tracking-wide text-zinc-500">{modeLabel}</span>
        )}
      </span>
      <button
        type="button"
        onClick={onClear}
        title="Clear history"
        className="p-1 rounded-md text-zinc-500 hover:text-white hover:bg-zinc-800 transition-colors"
      >
        <Trash2 className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        onClick={onTogglePlacement}
        title={placement === 'floating' ? 'Dock to canvas edge' : 'Undock as floating panel'}
        className="p-1 rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
      >
        {placement === 'floating' ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
      </button>
      <button
        type="button"
        onClick={onToggleCollapse}
        title="Collapse to rail"
        className="p-1 rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
      >
        <ChevronRight className="h-3.5 w-3.5" />
      </button>
    </div>
  )

  const body = () => (
    <div className="flex flex-col min-h-0 flex-1">
      <div ref={listRef} className="flex-1 min-h-0 overflow-y-auto p-3">
        <DockMessageList messages={messages} onStopTrack={onStopTrack} />
      </div>
      {/* Pinned prompt / settings content */}
      <div className="flex-shrink-0 border-t border-zinc-800 p-2">{children}</div>
    </div>
  )

  // ── Collapsed rail — canvas maximized, dock tucked to the right edge ────────
  if (view === 'collapsed' && placement !== 'floating') {
    return (
      <div className="h-full w-9 flex-shrink-0 border-l border-zinc-800 bg-zinc-900/80 flex flex-col items-center py-2 gap-1.5">
        <button
          type="button"
          onClick={onToggleCollapse}
          title="Expand chat dock"
          className="p-1.5 rounded-md text-emerald-400 hover:bg-zinc-800 transition-colors"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
        <button
          type="button"
          onClick={() => { if (view === 'collapsed') onToggleCollapse(); onTogglePlacement() }}
          title="Undock as floating panel"
          className="p-1.5 rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
        >
          <Maximize2 className="h-4 w-4" />
        </button>
        {messages.length > 0 && (
          <span className="mt-0.5 text-[9px] text-zinc-500 tabular-nums">{messages.length}</span>
        )}
      </div>
    )
  }

  // ── Floating modal — detached + draggable over the canvas ──────────────────
  // Rendered via a PORTAL to document.body: the dock is mounted inside GenSpace's
  // docked-layout wrapper, which collapses to width 0 when placement = floating
  // (dockReserved = 0). A `position: fixed` child trapped in a transformed / width-0
  // ancestor box gets clipped and disappears; portaling it to body keeps it
  // positioned against the viewport (same pattern as the fixed EditOptions popover).
  if (placement === 'floating') {
    return createPortal(
      <div
        className="fixed z-[90] flex flex-col rounded-2xl border border-zinc-700 bg-zinc-900 shadow-2xl overflow-hidden relative"
        style={{ left: pos.x, top: pos.y, width }}
        data-dock="floating"
      >
        {header(true)}
        {body()}
        {dockResizeHandle((dx) => width + dx)}
      </div>,
      document.body,
    )
  }

  // ── Expanded docked panel ───────────────────────────────────────────────────
  return (
    <div
      className="h-full flex-shrink-0 border-l border-zinc-800 bg-zinc-900/60 flex flex-col overflow-hidden relative"
      style={{ width }}
      data-dock="docked"
    >
      {dockResizeHandle((dx) => width - dx)}
      {header(false)}
      {body()}
    </div>
  )
}
