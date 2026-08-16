import { describe, it, expect } from 'vitest'
import { decodeMediaPayload, MEDIA_PAYLOAD_KEYS } from './media-decode'

// 1x1 red PNG, base64
const PNG_1x1 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=='

describe('decodeMediaPayload', () => {
  it('decodes the /image T2I key ("image") into an image/png Blob', async () => {
    const blob = decodeMediaPayload({ image: PNG_1x1, content_type: 'image/png', engine: 'zimage' })
    expect(blob).toBeInstanceOf(Blob)
    expect(blob!.type).toBe('image/png')
    expect(blob!.size).toBeGreaterThan(0)
    const buf = new Uint8Array(await blob!.arrayBuffer())
    // PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
    const head = Array.from(buf.slice(0, 8))
    expect(head).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
  })

  it('defaults to image/png when content_type is absent', async () => {
    const blob = decodeMediaPayload({ image: PNG_1x1 })
    expect(blob!.type).toBe('image/png')
  })

  it('decodes image_base64 (legacy/edit rail)', () => {
    const blob = decodeMediaPayload({ image_base64: PNG_1x1 })
    expect(blob!.type).toBe('image/png')
  })

  it('decodes styled_image (style-frame rail)', () => {
    const blob = decodeMediaPayload({ styled_image: PNG_1x1 })
    expect(blob!.type).toBe('image/png')
  })

  it('does NOT treat text-only payloads as media', () => {
    expect(decodeMediaPayload({ enhanced_prompt: 'x' })).toBeUndefined()
    expect(decodeMediaPayload({})).toBeUndefined()
  })

  it('treats empty/non-string media fields as absent', () => {
    expect(decodeMediaPayload({ image: '' })).toBeUndefined()
    expect(decodeMediaPayload({ image_base64: 123 } as Record<string, unknown>)).toBeUndefined()
  })

  it('media key set includes /image T2I "image" (regression guard)', () => {
    // This is the regression test for "no image returned": the image-worker's
    // /video-creator/v1/image responds with {"image": <b64>}, and if that key
    // ever falls out of the accepted set, T2I silently drops the result.
    expect(MEDIA_PAYLOAD_KEYS).toContain('image')
  })
})
