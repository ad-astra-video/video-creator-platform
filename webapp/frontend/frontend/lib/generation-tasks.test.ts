import { describe, expect, it } from 'vitest'
import {
  type GenerationTask,
  generationTaskReducer,
  runningCount,
  elidedLabel,
  MAX_TASKS,
} from './generation-tasks'

function seed(overrides: Partial<GenerationTask> = {}): GenerationTask {
  return {
    id: 'gen-1',
    label: 'forest night',
    projectId: 'project-a',
    kind: 'video',
    status: 'queued',
    createdAt: 1000,
    ...overrides,
  }
}

describe('generationTaskReducer', () => {
  it('ADD prepends the task', () => {
    const next = generationTaskReducer([], { type: 'ADD', task: seed() })
    expect(next).toHaveLength(1)
    expect(next[0].id).toBe('gen-1')
    expect(next[0].status).toBe('queued')
  })

  it('ADD caps the list at MAX_TASKS', () => {
    let tasks: GenerationTask[] = []
    for (let i = 0; i < MAX_TASKS + 5; i++) {
      tasks = generationTaskReducer(tasks, { type: 'ADD', task: seed({ id: `t${i}` }) })
    }
    expect(tasks).toHaveLength(MAX_TASKS)
  })

  it('START marks a queued task running with a timestamp', () => {
    const t = generationTaskReducer([seed()], { type: 'START', id: 'gen-1', at: 2000 })[0]
    expect(t.status).toBe('running')
    expect(t.startedAt).toBe(2000)
  })

  it('COMPLETE sets completed, asset key and completion time; run stays counted as done', () => {
    const [t] = generationTaskReducer([seed({ status: 'running' })], {
      type: 'COMPLETE', id: 'gen-1', at: 3000, assetKey: 'web://abc',
    })
    expect(t.status).toBe('completed')
    expect(t.assetKey).toBe('web://abc')
    expect(t.completedAt).toBe(3000)
  })

  it('FAIL records the error and time', () => {
    const [t] = generationTaskReducer([seed({ status: 'running' })], {
      type: 'FAIL', id: 'gen-1', at: 3000, error: 'runner busy',
    })
    expect(t.status).toBe('error')
    expect(t.error).toBe('runner busy')
  })

  it('does not complete a task already terminal', () => {
    const [t] = generationTaskReducer([seed({ status: 'completed' })], {
      type: 'FAIL', id: 'gen-1', at: 3000, error: 'x',
    })
    expect(t.status).toBe('completed')
  })

  it('REMOVE deletes by id', () => {
    expect(generationTaskReducer([seed()], { type: 'REMOVE', id: 'gen-1' })).toHaveLength(0)
  })

  it('CLEAR_COMPLETED keeps only non-completed tasks', () => {
    const tasks = [
      seed({ id: 'a', status: 'completed' }),
      seed({ id: 'b', status: 'running' }),
      seed({ id: 'c', status: 'error' }),
      seed({ id: 'd', status: 'queued' }),
    ]
    const next = generationTaskReducer(tasks, { type: 'CLEAR_COMPLETED' })
    expect(next.map(t => t.id)).toEqual(['b', 'c', 'd'])
  })
})

describe('runningCount', () => {
  it('counts queued + running only', () => {
    const tasks = [
      seed({ id: 'a', status: 'running' }),
      seed({ id: 'b', status: 'queued' }),
      seed({ id: 'c', status: 'completed' }),
    ]
    expect(runningCount(tasks)).toBe(2)
  })
})

describe('elidedLabel', () => {
  it('truncates long labels preserving a trailing ellipsis', () => {
    expect(elidedLabel('short')).toBe('short')
    expect(elidedLabel('x'.repeat(60), 10)).toBe('xxxxxxxxx…')
  })
})
