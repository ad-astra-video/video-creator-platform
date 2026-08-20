import { useCallback, useEffect, useRef, useState } from 'react'
import { useAppSettings } from '../contexts/AppSettingsContext'
import {
  discoverRunners,
  isWebPlatform,
  loadExcludedRunnerUrls,
  type DiscoveredRunner,
} from '../lib/livepeer-discovery'
import { countCapableRunners } from '../lib/runner-availability'

// Which capability ids a given rail requires in order to run. Kept in one place so the
// button-disable rule and the generation dispatch agree on what "has a runner" means.
export const VIDEO_RUNNER_CAPS = ['t2v']
export const IMAGE_RUNNER_CAPS = ['image']

export interface RunnerAvailability {
  /** True while the first discovery for a configured URL is still resolving. */
  loading: boolean
  /** True when at least one ready, non-excluded t2v/i2v-capable runner is available. */
  videoAvailable: boolean
  /** True when at least one ready, non-excluded image-capable runner is available. */
  imageAvailable: boolean
  /** Count of ready, non-excluded capable runners per rail (0 = none). */
  videoRunnerCount: number
  imageRunnerCount: number
  /** Re-run discovery now (call after a generation settles to unlock relaunching immediately). */
  refresh: () => void
}

/**
 * Tracks Livepeer runner availability for the generate/regenerate rails.
 *
 * The webapp's Generate button must be disabled ONLY when Livepeer is enabled and there is no
 * capable runner available. This hook provides that signal. It is a no-op (reports available)
 * when Livepeer is not enabled (no Discovery URL configured) — the disable rule only applies
 * to the Livepeer rail, so a user without Livepeer never gets gated here.
 */
export function useRunnerAvailability(
  caps: string[] = VIDEO_RUNNER_CAPS,
  refreshMs = 20_000,
): RunnerAvailability {
  const { settings } = useAppSettings()
  const livepeerEnabled = settings.hasLivepeerDiscoveryUrl === true
  const discoveryUrl = settings.livepeerDiscoveryUrl

  const [runners, setRunners] = useState<DiscoveredRunner[]>([])
  const [loading, setLoading] = useState<boolean>(livepeerEnabled)

  const capsRef = useRef(caps)
  capsRef.current = caps
  const enabledRef = useRef(livepeerEnabled)
  enabledRef.current = livepeerEnabled
  const urlRef = useRef(discoveryUrl)
  urlRef.current = discoveryUrl

  const refresh = useCallback(async () => {
    // Not Livepeer-enabled (no Discovery URL): nothing to gate on -> report available.
    if (!enabledRef.current || !urlRef.current) {
      setRunners([])
      setLoading(false)
      return
    }
    setLoading(true)
    try {
      const list = await discoverRunners(urlRef.current)
      const excludedUrls = new Set(
        isWebPlatform() ? loadExcludedRunnerUrls() : [],
      )
      setRunners(list.map((r) => ({
        ...r,
        excluded: r.excluded || excludedUrls.has(r.url),
      })))
    } catch {
      // Unreachable Discovery URL -> treat as no runners (button disabled when livepeer on).
      setRunners([])
    } finally {
      setLoading(false)
    }
  }, [])

  // Initial + whenever the Discovery URL / persisted runner preference changes.
  useEffect(() => {
    void refresh()
  }, [refresh, discoveryUrl, settings.livepeerSelectedRunnerId])

  // Periodic refresh so runner availability stays current (admission changes as GPUs load up).
  useEffect(() => {
    if (!livepeerEnabled) return
    const id = setInterval(() => { void refresh() }, refreshMs)
    return () => clearInterval(id)
  }, [livepeerEnabled, refresh, refreshMs])

  const videoRunnerCount = countCapableRunners(runners, capsRef.current)
  const imageRunnerCount = countCapableRunners(runners, IMAGE_RUNNER_CAPS)

  return {
    loading,
    videoAvailable: !livepeerEnabled || videoRunnerCount > 0,
    imageAvailable: !livepeerEnabled || imageRunnerCount > 0,
    videoRunnerCount,
    imageRunnerCount,
    refresh,
  }
}
