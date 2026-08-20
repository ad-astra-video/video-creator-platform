/**
 * Pure, testable runner-availability detection for the webapp.
 *
 * The webapp's Generate / Regenerate buttons should be disabled ONLY when there is no
 * capable runner available (and Livepeer is enabled). This module centralises that check so
 * the button-disable rule is one source of truth: given the set of discovered runners and the
 * required capabilities, which are actually usable (ready, not excluded, fetchable)?
 *
 * It is intentionally framework-free so it can be unit-tested with vitest (the frontend has a
 * vitest setup for pure dep-free lib modules, see the video-creator-platform-dev skill).
 */

export interface RunnerAvailabilityRunner {
  url: string
  runner_id: string
  status: string
  excluded: boolean
  capabilities?: Array<{ id: string }>
}

/**
 * True when the given capabilities are all served by `runner` AND the runner is actually usable:
 * ready, not excluded by the user, and its URL is fetchable by the browser (so a job could
 * actually be dispatched to it). `requiredCaps` empty means "any capable runner is fine".
 */
export function isRunnerCapable(
  runner: RunnerAvailabilityRunner,
  requiredCaps: string[],
  isFetchable: (url: string) => boolean = (url) => url.length > 0,
): boolean {
  if (runner.status !== 'ready') return false
  if (runner.excluded) return false
  if (!isFetchable(runner.url)) return false
  if (requiredCaps.length === 0) return true
  const ids = new Set((runner.capabilities ?? []).map((c) => c.id))
  return requiredCaps.every((c) => ids.has(c))
}

/** How many of `runners` can serve every capability in `requiredCaps`. 0 = no runner available. */
export function countCapableRunners(
  runners: RunnerAvailabilityRunner[],
  requiredCaps: string[],
  isFetchable?: (url: string) => boolean,
): number {
  return runners.reduce((count, runner) => (
    isRunnerCapable(runner, requiredCaps, isFetchable) ? count + 1 : count
  ), 0)
}
