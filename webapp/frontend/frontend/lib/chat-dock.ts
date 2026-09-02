/**
 * Chat-dock message history — the right-side panel that replaces the bottom
 * prompt bar in the GenSpace shell.
 *
 * The dock models KINDS of messages that MUST render distinctly so a user can
 * tell a *generation* (a prompt that produced media), a *chat* (a conversational
 * message) and an *LLM trace* (a record of an LLM round-trip) apart at a glance:
 *
 *   - `generation`: a prompt submitted to a generation rail (image / video /
 *     edit-subtask). Carries the prompt, the mode rail, a live status, and the
 *     path of the produced result once complete.
 *   - `chat`: a conversational message (user or assistant) with no media.
 *   - `llm_trace`: an informational record of an LLM round-trip — what was sent
 *     to the model, the response, and an optional reasoning section. Additive:
 *     the caller still routes the response to its real destination (prompt bar,
 *     layer UI, gap prompt, or assistant chat); the trace card is just the
 *     visible request/response record.
 *
 * The reducer is pure (no React / DOM deps) so it is unit-testable in the
 * frontend's vitest suite (files matching the ``.test.ts`` convention).
 */

export type GenerationStatus = 'running' | 'done' | 'error'

export interface GenerationMessage {
  kind: 'generation'
  id: string
  prompt: string
  /** The mode rail the generation ran on (video / image / edit / restyle / ...). */
  mode: string
  status: GenerationStatus
  /** Media asset path produced by the generation, once available. */
  resultPath: string | null
  /** A still image (frame) for the result, used as the thumbnail in history.
   *  For image results this is the image itself; for video results it's the
   *  precomputed small/big thumbnail so the card shows a real frame instead of
   *  a black <video> element until interaction. Falls back to resultPath. */
  stillPath: string | null
  error?: string
  /** True once the produced result asset was deleted from the project. */
  deleted?: boolean
  /** 0..1 overall progress while running (best-effort; only set when the rail
   *  reports it — e.g. Bernini per-step denoise progress over SSE). */
  progress?: number
  /** Current denoising step / total, when the rail reports per-step progress. */
  step?: number
  totalSteps?: number
  /** Human-readable phase status while running (e.g. 'Generating (Bernini)...'). */
  statusMessage?: string
  /** Ephemeral per-chunk preview (base64 JPEG) shown in the card while running;
   *  removed once the generation completes. */
  preview?: string
  createdAt: number
}

export interface ChatMessage {
  kind: 'chat'
  id: string
  role: 'user' | 'assistant'
  text: string
  createdAt: number
}

export interface LLMTraceMessage {
  kind: 'llm_trace'
  id: string
  /** Readable card label, e.g. 'Enhance', 'Layer suggestion', 'Gap fill', 'Agent chat'. */
  label: string
  /** What was sent to the LLM (the OpenAI-format messages). */
  sent: unknown
  /** The LLM's content response. */
  response: string
  /** Reasoning content, when present (rendered collapsed). */
  reasoning?: string
  /** Where the response is also applied, for the chip. Behavior is unchanged. */
  appliedTo?: 'prompt' | 'layer' | 'gap' | 'chat' | 'none'
  createdAt: number
}

export type ChatDockMessage = GenerationMessage | ChatMessage | LLMTraceMessage

/** Dock visual state. */
export type DockView = 'expanded' | 'collapsed'
/** Docked to the canvas edge vs. detached into a floating modal. */
export type DockPlacement = 'docked' | 'floating'

export interface ChatDockState {
  messages: ChatDockMessage[]
  view: DockView
  placement: DockPlacement
}

export type ChatDockAction =
  | { type: 'add_generation'; prompt: string; mode: string; id?: string }
  | { type: 'update_generation'; id: string; patch: Partial<Pick<GenerationMessage, 'status' | 'resultPath' | 'stillPath' | 'error' | 'deleted' | 'progress' | 'step' | 'totalSteps' | 'statusMessage' | 'preview'>> }
  | { type: 'add_chat'; role: ChatMessage['role']; text: string }
  | {
      type: 'add_llm_trace'
      label: string
      sent: unknown
      response: string
      reasoning?: string
      appliedTo?: LLMTraceMessage['appliedTo']
    }
  | { type: 'mark_generation_deleted'; paths: string[] }
  | { type: 'replace_messages'; messages: ChatDockMessage[] }
  | { type: 'set_view'; view: DockView }
  | { type: 'set_placement'; placement: DockPlacement }
  | { type: 'clear' }

export const initialChatDockState: ChatDockState = {
  messages: [],
  view: 'expanded',
  placement: 'docked',
}

let seq = 0
/** Monotonic id for a new message. `seq` guards against same-ms collisions. */
export function nextDockId(prefix: string): string {
  seq += 1
  return `${prefix}-${Date.now().toString(36)}-${seq}`
}

export function chatDockReducer(
  state: ChatDockState,
  action: ChatDockAction,
): ChatDockState {
  switch (action.type) {
    case 'add_generation':
      return {
        ...state,
        messages: [
          ...state.messages,
          {
            kind: 'generation',
            id: action.id ?? nextDockId('gen'),
            prompt: action.prompt,
            mode: action.mode,
            status: 'running',
            resultPath: null,
            stillPath: null,
            createdAt: Date.now(),
          },
        ],
      }
    case 'update_generation':
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.kind === 'generation' && m.id === action.id
            ? { ...m, ...action.patch }
            : m,
        ),
      }
    case 'add_chat':
      return {
        ...state,
        messages: [
          ...state.messages,
          {
            kind: 'chat',
            id: nextDockId('chat'),
            role: action.role,
            text: action.text,
            createdAt: Date.now(),
          },
        ],
      }
    case 'add_llm_trace':
      return {
        ...state,
        messages: [
          ...state.messages,
          {
            kind: 'llm_trace',
            id: nextDockId('llm'),
            label: action.label,
            sent: action.sent,
            response: action.response,
            ...(action.reasoning !== undefined ? { reasoning: action.reasoning } : {}),
            ...(action.appliedTo !== undefined ? { appliedTo: action.appliedTo } : {}),
            createdAt: Date.now(),
          },
        ],
      }
    case 'mark_generation_deleted': {
      const paths = action.paths
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.kind === 'generation' &&
          ((m.resultPath && paths.includes(m.resultPath)) ||
            (m.stillPath && paths.includes(m.stillPath)))
            ? { ...m, deleted: true }
            : m,
        ),
      }
    }
    case 'replace_messages':
      return { ...state, messages: action.messages }
    case 'set_view':
      return { ...state, view: action.view }
    case 'set_placement':
      return { ...state, placement: action.placement }
    case 'clear':
      return { ...state, messages: [] }
    default:
      return state
  }
}

/**
 * Should a restored generation card be hidden from a project timeline?
 * True when the result was explicitly marked deleted, or (when the caller
 * has the project's current asset paths) its result is no longer a project
 * asset — i.e. it was deleted from ANY surface (GenSpace grid, Video Editor,
 * etc.). When `assetPaths` is null the caller can't verify, so only the
 * explicit flag drops it (avoids false positives while assets are unknown).
 */
export function shouldDropRestoredGeneration(
  m: GenerationMessage,
  assetPaths: ReadonlySet<string> | null,
): boolean {
  if (m.deleted) return true
  if (assetPaths && m.resultPath && !assetPaths.has(m.resultPath)) return true
  return false
}
