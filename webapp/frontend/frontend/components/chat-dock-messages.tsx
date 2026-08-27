import { Image as ImageIcon, Video, Loader2, AlertTriangle, Trash2 } from 'lucide-react'
import { webAssetUrl } from '../lib/file-url'
import { useState } from 'react'
import type { ChatDockMessage, GenerationMessage, ChatMessage, LLMTraceMessage } from '../lib/chat-dock'

/**
 * Renders the chat-dock message history with a HARD visual distinction between
 * the two message kinds:
 *
 *   - A **generation** is a media card — a bordered tile with a result
 *     thumbnail (video/image), the prompt, a mode chip and a status pill. It
 *     reads as "this prompt produced media".
 *   - A **chat** is a plain speech bubble along one side, with no media frame.
 *     It reads as "a conversation".
 *
 * Kept as its own module (not inlined in ChatDock) because the entry renderers
 * are a clean boundary — the GenSpace shell and ChatDock both only need to feed
 * messages in and get differentiated UI back.
 */

const IMAGE_EXT = /\.(png|jpe?g|webp|gif|bmp|avif)$/i

function resultIsImage(path: string): boolean {
  return IMAGE_EXT.test(path)
}

function GenerationCard({ msg, onStopTrack }: { msg: GenerationMessage; onStopTrack?: (id: string) => void }) {
  const hasResult = !!msg.resultPath
  const isImage = msg.resultPath ? resultIsImage(msg.resultPath) : false
  return (
    <div
      className={`rounded-xl border overflow-hidden flex flex-col ${
        msg.status === 'error'
          ? 'border-red-500/40 bg-red-500/5'
          : 'border-zinc-800 bg-zinc-900/60'
      }`}
      data-kind="generation"
    >
      {/* Media / result strip */}
      <div className="relative aspect-video bg-zinc-950 flex items-center justify-center overflow-hidden">
        {msg.deleted && (
          <div className="flex flex-col items-center gap-1.5 p-3 text-center">
            <Trash2 className="h-5 w-5 text-zinc-500" />
            <span className="text-[10px] uppercase tracking-wide text-zinc-400">Deleted</span>
          </div>
        )}
        {msg.status === 'running' && (
          <div className="flex flex-col items-center gap-2 text-zinc-500">
            <Loader2 className="h-6 w-6 animate-spin text-emerald-400" />
            <span className="text-[10px] uppercase tracking-wide">Generating…</span>
          </div>
        )}
        {msg.status === 'error' && (
          <div className="flex flex-col items-center gap-1 p-3 text-center">
            <AlertTriangle className="h-5 w-5 text-red-400" />
            <span className="text-[10px] text-red-300 px-2">{msg.error || 'Generation failed'}</span>
          </div>
        )}
        {hasResult && !msg.deleted && (
          <div className="w-full h-full">
            {isImage ? (
              <img src={webAssetUrl(msg.resultPath!)} alt={msg.prompt} className="w-full h-full object-contain" />
            ) : msg.stillPath ? (
              <img src={webAssetUrl(msg.stillPath)} alt={msg.prompt} className="w-full h-full object-contain" />
            ) : (
              <video
                src={webAssetUrl(msg.resultPath!)}
                muted
                playsInline
                preload="metadata"
                disablePictureInPicture
                className="w-full h-full object-contain"
              />
            )}
          </div>
        )}
        {/* Mode chip */}
        <div className="absolute top-1.5 left-1.5 flex items-center gap-1 rounded-md bg-black/60 backdrop-blur px-1.5 py-0.5 text-[9px] uppercase tracking-wide text-zinc-300">
          {msg.mode === 'video' ? <Video className="h-3 w-3" /> : <ImageIcon className="h-3 w-3" />}
          {msg.mode}
        </div>
        {/* Status pill */}
        <div
          className={`absolute top-1.5 right-1.5 rounded-full px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wide ${
            msg.deleted
              ? 'bg-zinc-600/80 text-zinc-100'
              : msg.status === 'done'
                ? 'bg-emerald-500/80 text-black'
                : msg.status === 'error'
                  ? 'bg-red-500/80 text-white'
                  : 'bg-zinc-700/80 text-zinc-200'
          }`}
        >
          {msg.deleted ? 'deleted' : msg.status}
        </div>
      </div>
      {/* Prompt */}
      <div className="px-2.5 py-2">
        <p className="text-[11px] text-zinc-200 leading-snug line-clamp-2">{msg.prompt}</p>
        {msg.status === 'running' && (msg.progress != null || msg.statusMessage) && (
          <div className="mt-1.5">
            {msg.statusMessage && (
              <p className="text-[10px] text-zinc-400 leading-snug">{msg.statusMessage}</p>
            )}
            {msg.progress != null && (
              <div className="mt-1">
                <div className="h-1 rounded-full bg-zinc-800 overflow-hidden">
                  <div
                    className="h-full bg-emerald-400 transition-all duration-300 ease-out"
                    style={{ width: `${Math.round(Math.min(Math.max(msg.progress, 0), 1) * 100)}%` }}
                  />
                </div>
                {msg.step != null && msg.totalSteps != null ? (
                  <p className="text-[9px] uppercase tracking-wide text-zinc-500 mt-1">
                    step {msg.step}/{msg.totalSteps}
                  </p>
                ) : (
                  <p className="text-[9px] uppercase tracking-wide text-zinc-500 mt-1">
                    {Math.round(Math.min(Math.max(msg.progress, 0), 1) * 100)}%
                  </p>
                )}
              </div>
            )}
          </div>
        )}
        {msg.status === 'running' && onStopTrack && (
          <button
            type="button"
            onClick={() => onStopTrack(msg.id)}
            title="Stop tracking this task client-side"
            className="mt-1.5 text-[10px] uppercase tracking-wide text-zinc-500 hover:text-red-300 transition-colors"
          >
            Stop tracking
          </button>
        )}
      </div>
    </div>
  )
}

function ChatBubble({ msg }: { msg: ChatMessage }) {
  const mine = msg.role === 'user'
  return (
    <div className={`flex ${mine ? 'justify-end' : 'justify-start'}`} data-kind="chat">
      <div
        className={`max-w-[85%] rounded-2xl px-3 py-2 text-[12px] leading-snug whitespace-pre-wrap break-words ${
          mine
            ? 'bg-emerald-700/70 text-white rounded-br-sm'
            : 'bg-zinc-800 text-zinc-200 rounded-bl-sm'
        }`}
      >
        {msg.text}
      </div>
    </div>
  )
}

/** Destination chip label for an `appliedTo` value. */
const APPLIED_TO_LABEL: Record<NonNullable<LLMTraceMessage['appliedTo']>, string> = {
  prompt: '→ prompt bar',
  layer: '→ layer count',
  gap: '→ gap prompt',
  chat: '→ assistant chat',
  none: '',
}

/**
 * Extract a readable "message sent" string from the raw `sent` payload. The
 * payload can be either a list of OpenAI-style messages (most callers) or a
 * request-inputs object whose primary field is the prompt (the Enhance rail).
 * Falls back to a JSON string so nothing is ever lost from the visible record.
 */
function extractSentText(sent: unknown): string {
  if (Array.isArray(sent)) {
    const parts: string[] = []
    for (const m of sent) {
      if (!m || typeof m !== 'object') continue
      const content = (m as { content?: unknown }).content
      if (typeof content === 'string') {
        if (content.trim()) parts.push(content)
      } else if (Array.isArray(content)) {
        // OpenAI content blocks: { type: 'text', text: ... }
        const text = content
          .map((b) => (b && typeof b === 'object' && typeof (b as { text?: unknown }).text === 'string' ? (b as { text: string }).text : ''))
          .join('\n')
        if (text.trim()) parts.push(text)
      }
    }
    if (parts.length) return parts.join('\n\n')
    return JSON.stringify(sent)
  }
  if (sent && typeof sent === 'object') {
    const prompt = (sent as { prompt?: unknown }).prompt
    if (typeof prompt === 'string' && prompt.trim()) return prompt
    return JSON.stringify(sent)
  }
  return sent === null || sent === undefined ? '' : String(sent)
}

/**
 * A read-only record of one LLM round-trip — what was sent, the response, and a
 * collapsed-by-default reasoning section when the model returned one. Rendered
 * to read like a chat window: the message that went in (a "sent" bubble), the
 * model's reasoning on its own collapsible toggle, and the response (a reply
 * bubble). This is an informational card — the response itself is routed to its
 * real destination (prompt bar, layer UI, gap prompt, assistant chat) by the
 * caller, unchanged; this card just mirrors the round-trip.
 */
function LLMTraceCard({ msg }: { msg: LLMTraceMessage }) {
  const [showReasoning, setShowReasoning] = useState(false)
  const dest = msg.appliedTo ? APPLIED_TO_LABEL[msg.appliedTo] : ''
  const sentText = extractSentText(msg.sent)
  return (
    <div
      className="rounded-xl border border-zinc-800 bg-zinc-900/40 overflow-hidden flex flex-col"
      data-kind="llm_trace"
    >
      <div className="flex items-center gap-2 px-2.5 py-1.5 border-b border-zinc-800">
        <span className="rounded bg-fuchsia-500/20 text-fuchsia-300 px-1.5 py-0.5 text-[9px] uppercase tracking-wide font-medium">
          LLM
        </span>
        <span className="text-[11px] text-zinc-300 font-medium">{msg.label}</span>
        {dest && <span className="ml-auto text-[10px] text-zinc-500">{' '}{dest}</span>}
      </div>
      <div className="px-2.5 py-2 space-y-2">
        {/* Message sent — the user-facing input to the model, as a bubble. */}
        <div className="flex justify-end">
          <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-emerald-700/70 text-white px-3 py-2 text-[11px] leading-snug whitespace-pre-wrap break-words">
            {sentText || <span className="text-white/60">(empty request)</span>}
          </div>
        </div>
        {/* Collapsible reasoning (start collapsed). */}
        {msg.reasoning ? (
          <div className="flex justify-start">
            <div className="max-w-[85%]">
              <button
                type="button"
                onClick={() => setShowReasoning((v) => !v)}
                className="text-[10px] uppercase tracking-wide text-zinc-500 hover:text-zinc-300 flex items-center gap-1"
              >
                <span
                  className={`inline-block transition-transform ${showReasoning ? 'rotate-90' : ''}`}
                >
                  ▸
                </span>
                {showReasoning ? 'Hide reasoning' : 'Show reasoning'}
              </button>
              {showReasoning && (
                <div className="mt-1 rounded-md border border-zinc-800 bg-zinc-950/40 p-2">
                  <pre className="text-[10px] leading-snug text-zinc-500 whitespace-pre-wrap break-words max-h-40 overflow-y-auto">
                    {msg.reasoning}
                  </pre>
                </div>
              )}
            </div>
          </div>
        ) : null}
        {/* Response — the model's reply, as a bubble. */}
        <div className="flex justify-start">
          <div className="max-w-[85%] rounded-2xl rounded-bl-sm bg-zinc-800 text-zinc-200 px-3 py-2 text-[11px] leading-snug whitespace-pre-wrap break-words">
            {msg.response}
          </div>
        </div>
      </div>
    </div>
  )
}

export function DockMessageList({
  messages,
  emptyTitle = 'No activity yet',
  onStopTrack,
}: {
  messages: ChatDockMessage[]
  emptyTitle?: string
  onStopTrack?: (id: string) => void
}) {
  if (messages.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center text-zinc-600 px-6">
        <p className="text-[12px]">{emptyTitle}</p>
        <p className="text-[10px] mt-1">
          Generations (media results) and chats appear here as distinct cards.
        </p>
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-2">
      {messages.map((m) =>
        m.kind === 'generation' ? (
          <GenerationCard key={m.id} msg={m} onStopTrack={onStopTrack} />
        ) : m.kind === 'llm_trace' ? (
          <LLMTraceCard key={m.id} msg={m} />
        ) : (
          <ChatBubble key={m.id} msg={m} />
        ),
      )}
    </div>
  )
}
