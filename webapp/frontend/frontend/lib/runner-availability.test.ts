import { describe, it, expect } from 'vitest'
import { countCapableRunners, isRunnerCapable, type RunnerAvailabilityRunner } from './runner-availability'

const readyT2V = (over: Partial<RunnerAvailabilityRunner> = {}): RunnerAvailabilityRunner => ({
  url: 'https://orchestrator.example/discovery',
  runner_id: 'runner-a',
  status: 'ready',
  excluded: false,
  capabilities: [{ id: 't2v' }, { id: 'image' }],
  ...over,
})

describe('isRunnerCapable', () => {
  it('accepts a ready, non-excluded runner that serves every required cap', () => {
    expect(isRunnerCapable(readyT2V(), ['t2v'])).toBe(true)
  })

  it('treats ANY capable runner as fine when requiredCaps is empty', () => {
    expect(isRunnerCapable(readyT2V(), [])).toBe(true)
  })

  it('rejects a busy runner even when it has the caps', () => {
    expect(isRunnerCapable(readyT2V({ status: 'busy' }), ['t2v'])).toBe(false)
  })

  it('rejects an offline runner', () => {
    expect(isRunnerCapable(readyT2V({ status: 'offline' }), ['t2v'])).toBe(false)
  })

  it('rejects a user-excluded runner', () => {
    expect(isRunnerCapable(readyT2V({ excluded: true }), ['t2v'])).toBe(false)
  })

  it('rejects a runner missing a required capability', () => {
    const runner = readyT2V({ capabilities: [{ id: 't2v' }] })
    expect(isRunnerCapable(runner, ['image'])).toBe(false)
    expect(isRunnerCapable(runner, ['t2v'])).toBe(true)
  })

  it('rejects a runner with an unfetchable (empty) URL', () => {
    expect(isRunnerCapable(readyT2V({ url: '' }), ['t2v'])).toBe(false)
  })
})

describe('countCapableRunners', () => {
  it('counts only ready, non-excluded, capable runners', () => {
    const runners = [
      readyT2V(),                                  // cap t2v ✔
      readyT2V({ runner_id: 'b', status: 'busy' }), // busy ✘
      readyT2V({ runner_id: 'c', excluded: true }), // excluded ✘
      readyT2V({ runner_id: 'd', capabilities: [{ id: 'image' }] }), // no t2v ✘
    ]
    expect(countCapableRunners(runners, ['t2v'])).toBe(1)
  })

  it('returns 0 when no runner serves the required caps', () => {
    const runners = [readyT2V({ capabilities: [{ id: 'image' }] })]
    expect(countCapableRunners(runners, ['t2v'])).toBe(0)
    expect(countCapableRunners(runners, ['image'])).toBe(1)
  })

  it('returns 0 for an empty list (the "no runners available" state)', () => {
    expect(countCapableRunners([], ['t2v'])).toBe(0)
  })
})
