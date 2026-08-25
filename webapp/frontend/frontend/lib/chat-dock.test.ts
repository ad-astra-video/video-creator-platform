import { describe, it, expect } from 'vitest'
import {
  chatDockReducer,
  initialChatDockState,
  shouldDropRestoredGeneration,
  type ChatDockState,
  type GenerationMessage,
} from './chat-dock'

describe('chatDockReducer', () => {
  it('starts empty and expanded-docked', () => {
    expect(initialChatDockState.messages).toEqual([])
    expect(initialChatDockState.view).toBe('expanded')
    expect(initialChatDockState.placement).toBe('docked')
  })

  it('adds a running generation entry', () => {
    const s = chatDockReducer(initialChatDockState, {
      type: 'add_generation',
      prompt: 'a puppy',
      mode: 'image',
    })
    expect(s.messages).toHaveLength(1)
    expect(s.messages[0]).toMatchObject({
      kind: 'generation',
      prompt: 'a puppy',
      mode: 'image',
      status: 'running',
      resultPath: null,
    })
  })

  it('updates a generation in place by id (status + result)', () => {
    let s = chatDockReducer(initialChatDockState, {
      type: 'add_generation',
      prompt: 'a puppy',
      mode: 'video',
    })
    const id = s.messages[0].id
    s = chatDockReducer(s, {
      type: 'update_generation',
      id,
      patch: { status: 'done', resultPath: 'video://out.mp4' },
    })
    expect(s.messages[0]).toMatchObject({
      kind: 'generation',
      status: 'done',
      resultPath: 'video://out.mp4',
    })
    // No-op update of an unknown id leaves the list content untouched.
    const before = s
    const after = chatDockReducer(s, {
      type: 'update_generation',
      id: 'gen-nope',
      patch: { status: 'error' },
    })
    expect(after.messages).toEqual(before.messages)
    expect(after.messages[0]).toMatchObject({ status: 'done', resultPath: 'video://out.mp4' })
  })

  it('drops restored generations marked deleted or whose result is gone from the project', () => {
    const mk = (id: string, resultPath: string | null, deleted?: boolean): GenerationMessage => ({
      kind: 'generation',
      id,
      prompt: 'p',
      mode: 'video',
      status: 'done',
      resultPath,
      stillPath: null,
      deleted,
      createdAt: 0,
    })
    const present = new Set(['a.mp4'])
    expect(shouldDropRestoredGeneration(mk('1', 'a.mp4'), present)).toBe(false) // still exists
    expect(shouldDropRestoredGeneration(mk('2', 'gone.mp4'), present)).toBe(true) // deleted from project
    expect(shouldDropRestoredGeneration(mk('3', 'a.mp4', true), present)).toBe(true) // explicitly deleted
    expect(shouldDropRestoredGeneration(mk('4', 'gone.mp4'), null)).toBe(false) // unknown asset set -> keep
    expect(shouldDropRestoredGeneration(mk('5', null), present)).toBe(false) // no result -> keep
  })

  it('marks matching generation entries deleted by result or still path', () => {
    // A done generation with a media result (path + thumbnail still).
    let s = chatDockReducer(initialChatDockState, { type: 'add_generation', prompt: 'p', mode: 'video' })
    const id = s.messages[0].id
    s = chatDockReducer(s, {
      type: 'update_generation',
      id,
      patch: { status: 'done', resultPath: 'proj/a.mp4', stillPath: 'proj/thumbs/a.jpg' },
    })
    // Deleting by the thumbnail/still path marks the card deleted.
    s = chatDockReducer(s, { type: 'mark_generation_deleted', paths: ['proj/thumbs/a.jpg'] })
    expect(s.messages[0]).toMatchObject({ status: 'done', resultPath: 'proj/a.mp4', deleted: true })
    // Non-matching paths leave entries untouched.
    s = chatDockReducer(initialChatDockState, { type: 'add_generation', prompt: 'q', mode: 'image' })
    s = chatDockReducer(s, { type: 'mark_generation_deleted', paths: ['other.png'] })
    expect(s.messages[0]).not.toHaveProperty('deleted')
  })

  it('adds chat entries and keeps kind distinct from generation', () => {
    let s = chatDockReducer(initialChatDockState, {
      type: 'add_chat',
      role: 'user',
      text: 'hello',
    })
    s = chatDockReducer(s, { type: 'add_generation', prompt: 'x', mode: 'edit' })
    expect(s.messages[0].kind).toBe('chat')
    expect(s.messages[1].kind).toBe('generation')
    expect(s.messages[0]).toMatchObject({ kind: 'chat', role: 'user', text: 'hello' })
  })

  it('adds an llm_trace entry with sent/response/reasoning', () => {
    let s = chatDockReducer(initialChatDockState, {
      type: 'add_llm_trace',
      label: 'Enhance',
      sent: [{ role: 'user', content: 'make it cinematic' }],
      response: 'A cinematic scene.',
      reasoning: 'Because it is dark.',
      appliedTo: 'prompt',
    })
    expect(s.messages).toHaveLength(1)
    expect(s.messages[0]).toMatchObject({
      kind: 'llm_trace',
      label: 'Enhance',
      response: 'A cinematic scene.',
      reasoning: 'Because it is dark.',
      appliedTo: 'prompt',
    })
    // trace without reasoning stores no reasoning field
    s = chatDockReducer(s, {
      type: 'add_llm_trace',
      label: 'Gap fill',
      sent: [],
      response: 'a bridge',
      appliedTo: 'gap',
    })
    expect(s.messages).toHaveLength(2)
    expect(s.messages[1]).toMatchObject({
      kind: 'llm_trace',
      label: 'Gap fill',
      appliedTo: 'gap',
    })
    expect(s.messages[1]).not.toHaveProperty('reasoning')
  })

  it('keeps llm_trace distinct from generation and chat', () => {
    let s = chatDockReducer(initialChatDockState, { type: 'add_chat', role: 'user', text: 'hi' })
    s = chatDockReducer(s, { type: 'add_generation', prompt: 'x', mode: 'video' })
    s = chatDockReducer(s, {
      type: 'add_llm_trace',
      label: 'Layer suggestion',
      sent: [],
      response: '3',
      appliedTo: 'layer',
    })
    expect(s.messages.map((m) => m.kind)).toEqual(['chat', 'generation', 'llm_trace'])
  })

  it('marks a running generation as error (stop-tracking / timeout path)', () => {
    let s = chatDockReducer(initialChatDockState, { type: 'add_generation', prompt: 'p', mode: 'video' })
    const id = s.messages[0].id
    expect(s.messages[0]).toMatchObject({ status: 'running' })
    s = chatDockReducer(s, {
      type: 'update_generation',
      id,
      patch: { status: 'error', error: 'Stopped tracking.' },
    })
    expect(s.messages[0]).toMatchObject({ kind: 'generation', status: 'error', error: 'Stopped tracking.' })
    expect(s.messages).toHaveLength(1)
  })

  it('toggles view and placement independently', () => {
    let s: ChatDockState = chatDockReducer(initialChatDockState, { type: 'set_view', view: 'collapsed' })
    expect(s.view).toBe('collapsed')
    s = chatDockReducer(s, { type: 'set_placement', placement: 'floating' })
    expect(s.placement).toBe('floating')
    expect(s.view).toBe('collapsed')
    s = chatDockReducer(s, { type: 'set_placement', placement: 'docked' })
    expect(s.placement).toBe('docked')
  })

  it('clears all messages but keeps layout state', () => {
    let s = chatDockReducer(initialChatDockState, { type: 'add_chat', role: 'user', text: 'a' })
    s = chatDockReducer(s, { type: 'add_generation', prompt: 'p', mode: 'image' })
    s = chatDockReducer(s, { type: 'set_view', view: 'collapsed' })
    s = chatDockReducer(s, { type: 'clear' })
    expect(s.messages).toEqual([])
    expect(s.view).toBe('collapsed')
  })
})
