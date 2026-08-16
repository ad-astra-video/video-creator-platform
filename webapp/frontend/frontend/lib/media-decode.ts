/**
 * Decode a runner "complete" payload into a media Blob, if it carries one.
 *
 * This is the ONLY place the webapp turns a runner result's base64 bytes into a
 * displayable Blob. Keep it free of browser-only imports (no ApiClient, no
 * http/fetch, no DOM) so it can be unit-tested under vitest in a plain Node env.
 *
 * Media tasks return one of these base64 keys (all __base64-encoded bytes):
 *   - video tasks  -> `video_base64`
 *   - /image T2I   -> `image`            (image-worker /video-creator/v1/image)
 *   - /edit        -> `image_base64` / `image` / `edited_image`
 *   - /style-frame -> `styled_image`
 *   - restyle      -> `output_video`
 *
 * Text-only payloads (e.g. prompt-enhance -> `enhanced_prompt`) carry none of
 * these and return undefined.
 *
 * NOTE: `image_base64` vs `image` both appear in real payloads (edit vs image
 * endpoints), so the key set is deliberately a superset. If a future endpoint
 * changes its response key, update this list — a mismatch here silently drops
 * the result ("no image returned") while the runner itself encoded it fine.
 */
export function decodeMediaPayload(payload: Record<string, unknown>): Blob | undefined {
  for (const key of ['video_base64', 'image_base64', 'image', 'output_video', 'styled_image'] as const) {
    const b64 = payload[key]
    if (typeof b64 === 'string' && b64) {
      const contentType =
        typeof payload.content_type === 'string' ? payload.content_type
        : key === 'video_base64' || key === 'output_video' ? 'video/mp4'
        : 'image/png'
      const binary = atob(b64)
      const bytes = new Uint8Array(binary.length)
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
      return new Blob([bytes], { type: contentType })
    }
  }
  return undefined
}

/**
 * The ordered set of media keys decodeMediaPayload accepts, exported so tests
 * (and future audit) can assert the exact list is correct/comprehensive.
 */
export const MEDIA_PAYLOAD_KEYS = [
  'video_base64',
  'image_base64',
  'image',
  'output_video',
  'styled_image',
] as const
