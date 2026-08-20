import { GENERATION_RECOVERY_KEY, GENERATION_RECOVERY_TS_KEY, GENERATION_RECOVERY_LEASE_MS, type GenerationRecoveryContext } from '../hooks/use-generation'

// The webapp has NO backend generation-job rows (browser WS/HTTP generations never write one),
// so the GET /api/generation/progress endpoint has been REMOVED entirely from the webapp — there
// is no server progress to poll. Generation progress is resolved purely from browser-local
// state: the GENERATION_RECOVERY_KEY marker in localStorage IS the "generation in flight" signal.
export interface GenerationProgress {
  ok: true
  data: { id: null; status: 'running' | 'idle'; progress: number; phase: string; result: undefined }
}
type Listener = (result: GenerationProgress) => void

// Returns the in-flight generation marker ONLY if its lease is still live. A marker written by
// a pre-lease build has no timestamp (or one that's expired because the owning tab died without
// settling) — both are stale by definition and are purged here, so a leaked marker can never
// permanently wedge the global generation lock.
function activeRecoveryMarker(): GenerationRecoveryContext | null {
  const raw = localStorage.getItem(GENERATION_RECOVERY_KEY)
  if (!raw) return null
  const ts = Number(localStorage.getItem(GENERATION_RECOVERY_TS_KEY) ?? NaN)
  if (!Number.isFinite(ts) || Date.now() - ts > GENERATION_RECOVERY_LEASE_MS) {
    localStorage.removeItem(GENERATION_RECOVERY_KEY)
    localStorage.removeItem(GENERATION_RECOVERY_TS_KEY)
    return null
  }
  try {
    return JSON.parse(raw) as GenerationRecoveryContext
  } catch {
    localStorage.removeItem(GENERATION_RECOVERY_KEY)
    localStorage.removeItem(GENERATION_RECOVERY_TS_KEY)
    return null
  }
}

// Refresh the lease while a generation is genuinely in flight, so a legitimately long job isn't
// released early; on a crash the poll stops and the lease naturally expires.
function touchLease(): void {
  if (!localStorage.getItem(GENERATION_RECOVERY_KEY)) return
  localStorage.setItem(GENERATION_RECOVERY_TS_KEY, String(Date.now()))
}

export function activeRecoveryMarkerExists(): boolean {
  return activeRecoveryMarker() != null
}

function browserLocalProgress(): GenerationProgress {
  return {
    ok: true,
    data: { id: null, status: activeRecoveryMarkerExists() ? 'running' : 'idle', progress: 0, phase: '', result: undefined },
  }
}

const POLL_INTERVAL_MS = 3000
const MARKER_CHECK_INTERVAL_MS = 3000

const listeners = new Set<Listener>()
let interval: ReturnType<typeof setInterval> | null = null
let lastResult: GenerationProgress | null = null

// Purely local poll: no async I/O and no network — just re-reads the recovery-marker lease and
// notifies listeners with the running/idle snapshot.
function poll(): void {
  touchLease()
  const result = browserLocalProgress()
  lastResult = result
  listeners.forEach(listener => listener(result))
}

// The global generation lock and the background recovery watcher both need the same
// generation-progress poll while mounted together (any project open); sharing one interval
// instead of one per subscriber keeps this cheap.
export function subscribeToGenerationProgress(listener: Listener): () => void {
  listeners.add(listener)
  if (lastResult) listener(lastResult)
  if (!interval) {
    poll()
    interval = setInterval(poll, POLL_INTERVAL_MS)
  }
  return () => {
    listeners.delete(listener)
    if (listeners.size === 0 && interval) {
      clearInterval(interval)
      interval = null
      // Otherwise a later generation's fresh subscribe (see subscribeWhileGenerationMayBeActive)
      // immediately replays this now-unrelated snapshot before its own first poll ever completes
      // - e.g. reporting "running" from a previous, unrelated session for a moment.
      lastResult = null
    }
  }
}

// Every generate-starting call site (see GenSpace.tsx's writeRecoveryContext calls) writes a
// recovery marker into localStorage BEFORE it starts, in this same renderer process — so no
// marker anywhere is proof nothing here could be running, checkable with a local read instead of
// a network call. Gates the shared poll on that: idle app (no generation ever started, or one
// that already finished and was consumed) costs nothing for the whole session, since this
// re-checks on the same cadence as the poll it gates.
export function subscribeWhileGenerationMayBeActive(listener: Listener): () => void {
  let unsubscribe: (() => void) | null = null
  let latest: GenerationProgress | null = null
  const trackingListener: Listener = result => {
    latest = result
    listener(result)
  }
  const sync = () => {
    const hasMarker = activeRecoveryMarkerExists()
    if (hasMarker) {
      if (!unsubscribe) unsubscribe = subscribeToGenerationProgress(trackingListener)
      return
    }
    // No marker doesn't mean safe to stop yet: a marker's writer can clear it eagerly, straight
    // off its own HTTP response (Enhance does — see GenSpace.tsx's runEnhance), without ever
    // going through a poll cycle first. If we unsubscribed immediately here, a shared poll tick
    // still mid-flight (or one that simply hasn't fired again yet) never gets to redeliver the
    // real terminal status, and the listener (e.g. useGlobalGenerationLock's isRunning) is stuck
    // on whatever "running" snapshot it last saw — forever, since nothing calls it again. Only
    // stop once a poll has actually caught up and confirmed we're not running.
    if (unsubscribe && latest && (!latest.ok || latest.data.status !== 'running')) {
      unsubscribe()
      unsubscribe = null
    }
  }
  sync()
  const interval = setInterval(sync, MARKER_CHECK_INTERVAL_MS)
  return () => {
    clearInterval(interval)
    unsubscribe?.()
  }
}
