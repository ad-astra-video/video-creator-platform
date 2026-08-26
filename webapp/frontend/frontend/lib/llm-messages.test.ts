import { describe, expect, it } from 'vitest'
import {
  buildEnhanceMessages,
  buildLayerSuggestMessages,
  buildGapSuggestMessages,
  DEFAULT_T2V_SYSTEM_PROMPT,
  DEFAULT_I2V_SYSTEM_PROMPT,
  DEFAULT_EXTEND_SYSTEM_PROMPT,
  DEFAULT_RETAKE_SYSTEM_PROMPT,
  LAYER_SUGGEST_RUBRIC,
} from './llm-messages'

const B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='

describe('buildEnhanceMessages', () => {
  it('plain text -> T2V system prompt with a plain-text user message', () => {
    const msgs = buildEnhanceMessages({ prompt: 'a cat' })
    expect(msgs[0].content).toBe(DEFAULT_T2V_SYSTEM_PROMPT)
    expect(msgs[1]).toEqual({ role: 'user', content: 'user prompt: a cat' })
  })

  it('single image -> I2V system prompt + image_url part + text part', () => {
    const msgs = buildEnhanceMessages({ prompt: 'a cat', imageBase64: B64 })
    expect(msgs[0].content).toBe(DEFAULT_I2V_SYSTEM_PROMPT)
    const content = msgs[1].content as unknown[]
    expect(content[0]).toMatchObject({ type: 'image_url' })
    expect((content[0] as any).image_url.url).toContain('data:image/png;base64,')
    expect((content[1] as any).text).toBe('User Raw Input Prompt: a cat.')
  })

  it('context_frames -> EXTEND system prompt with frames; direction note appended', () => {
    const msgs = buildEnhanceMessages({ prompt: 'keep going', contextFrames: [B64], task: 'extend', direction: 'pan left' })
    expect(msgs[0].content).toBe(DEFAULT_EXTEND_SYSTEM_PROMPT)
    const content = msgs[1].content as unknown[]
    expect(content.slice(0, 1)[0]).toMatchObject({ type: 'image_url' })
    expect((content[1] as any).text).toContain('Extend direction: pan left.')
  })

  it('retake chooses the retake system prompt', () => {
    const msgs = buildEnhanceMessages({ prompt: 'fix the jet', contextFrames: [B64], task: 'retake' })
    expect(msgs[0].content).toBe(DEFAULT_RETAKE_SYSTEM_PROMPT)
  })

  it('custom overrides replace the default system prompt', () => {
    const msgs = buildEnhanceMessages({ prompt: 'x' }, { enhancerT2V: 'custom t2v' })
    expect(msgs[0].content).toBe('custom t2v')
  })
})

describe('buildLayerSuggestMessages', () => {
  it('uses the rubric + image + how-many question shape', () => {
    const msgs = buildLayerSuggestMessages(B64)
    expect(msgs[0].content).toBe(LAYER_SUGGEST_RUBRIC)
    const content = msgs[1].content as unknown[]
    expect(content[0]).toMatchObject({ type: 'image_url' })
    expect((content[1] as any).text).toBe('How many layers should this image be decomposed into?')
  })
})

describe('buildGapSuggestMessages', () => {
  it('includes frames, duration and mode text', () => {
    const msgs = buildGapSuggestMessages({
      gapDuration: 2,
      mode: 'before',
      beforeFrame: B64,
      afterFrame: B64,
      beforePrompt: 'a beach',
      afterPrompt: 'a storm',
    })
    const content = msgs[1].content as unknown[]
    // 2 frames + 1 text
    const imgs = content.filter((c: any) => c.type === 'image_url')
    expect(imgs).toHaveLength(2)
    const text = content[content.length - 1] as any
    expect(text.text).toContain('2s timeline gap')
    expect(text.text).toContain('mode: before')
    expect(text.text).toContain('Before summary: a beach')
  })

  it('substitutes the duration into the rubric', () => {
    const msgs = buildGapSuggestMessages({ gapDuration: 5, mode: 'after' })
    // GAP_FILL_RUBRIC holds a {duration} placeholder; the builder replaces it — the
    // sent system prompt must not contain an unresolved placeholder.
    expect(String(msgs[0].content)).not.toContain('{duration}')
    expect(String(msgs[0].content)).toContain('~5s')
  })
})
