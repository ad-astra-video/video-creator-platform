import { useEffect, useState } from 'react'
import { subscribeWhileGenerationMayBeActive, activeRecoveryMarkerExists } from '../lib/generation-progress-poll'

// The webapp (electronAPI.platform === 'web') has NO local backend generation slot: generation
// runs on a remote Livepeer runner (dispatched via the direct/SSE transport). There is no shared
// single-slot state to serialize, and the recovery marker / lease machinery exists only to protect
// the desktop build's single local-GPU slot. Applying the lock in the webapp is not just pointless
// — it's actively harmful: a stale ltx-generation-recovery marker (e.g. left by a crash or a
// request that died before its finally) makes this hook initialize `true` and stay `true` forever,
// permanently greying out every Generate/Enhance button even though a runner is available and
// could accept the job. Skip the lock entirely on the web platform.
function isWebPlatform(): boolean {
  try {
    return (window as unknown as { electronAPI?: { platform?: string } }).electronAPI?.platform === 'web'
  } catch {
    return false
  }
}

// Only one generation can run at a time across the whole app (single global backend slot), but
// each project's GenSpace only tracks its OWN local isGenerating-style state — it has no idea a
// different project (or the same one, reconnected via a stale click) is already occupying that
// slot. Without this, Generate stays clickable in project B while project A is mid-generation;
// the request 409s, but only after writeRecoveryContext already overwrote A's in-flight recovery
// marker with B's (now-failed) one. Polling here lets Generate disable proactively instead.
// No marker anywhere means nothing CAN be running (see subscribeWhileGenerationMayBeActive), so
// idle starts unlocked and costs no network call; once a marker exists and we're actually
// polling, an unconfirmed/failed poll is treated as locked rather than silently trusting "not
// running" — that unconfirmed-failure gap is exactly what previously let Generate stay clickable
// during another project's generation. The initial state has to check the marker too, not just
// hardcode false: a page refresh resets this hook's React state from scratch while another
// project's marker (and its still-running backend generation) survives in localStorage, and the
// first poll takes a network round trip to resolve — that gap is otherwise the same unconfirmed
// window all over again, just re-opened on every reload instead of only at first app launch.
export function useGlobalGenerationLock(): boolean {
  // Desktop-only: the webapp has no local backend slot, so no global lock (see above).
  const [isWeb] = useState(isWebPlatform)
  const [isRunning, setIsRunning] = useState(() => activeRecoveryMarkerExists())

  useEffect(() => {
    if (isWeb) return
    return subscribeWhileGenerationMayBeActive(result => {
      setIsRunning(result.ok ? result.data.status === 'running' : true)
    })
  }, [isWeb])

  return isWeb ? false : isRunning
}
