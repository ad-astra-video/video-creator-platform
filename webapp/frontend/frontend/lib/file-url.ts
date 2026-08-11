/**
 * Converts a filesystem path to a properly encoded file:// URL.
 *
 *   /Users/me/my file.mp4   → file:///Users/me/my%20file.mp4
 *   C:\Users\me\video#1.mp4 → file:///C:/Users/me/video%231.mp4
 */
export function pathToFileUrl(filePath: string): string {
  // Normalize Windows separators
  let normalized = filePath.replace(/\\/g, '/')

  // Ensure leading slash (Windows drive letters like C:/ need one prepended)
  if (!normalized.startsWith('/')) {
    normalized = '/' + normalized
  }

  // Encode each path segment individually so we don't encode the slashes
  const encoded = normalized
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/')

  return 'file://' + encoded
}

/**
 * Web-aware asset URL resolver.
 *
 * In the browser (serverless web app) there is no real filesystem — media is stored in
 * the in-memory web asset store under `web://<uuid>` keys mapped to Blob/object URLs.
 * `pathToFileUrl` would turn those into unplayable `file://` URLs, so anything rendered
 * (`<video src>`, `<img src>`) must resolve web keys to their blob URL first. Plain disk
 * paths (the Electron/desktop case) fall back to `pathToFileUrl`.
 *
 * Returns '' for an unknown/blank web key (caller should treat as no source).
 */
import { isWebPath, getBlobUrl } from './runtime/web-store'

export function webAssetUrl(p: string | null | undefined): string {
  if (!p) return ''
  if (isWebPath(p)) {
    return getBlobUrl(p) ?? ''
  }
  return pathToFileUrl(p)
}
