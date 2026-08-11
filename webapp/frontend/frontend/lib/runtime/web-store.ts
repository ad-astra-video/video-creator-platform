// Web asset store: a session-persistent, in-browser stand-in for the OS filesystem that
// `window.electronAPI` referenced. Every "file path" the UI deals with is a synthetic
// `web://<uuid>` key mapped to a real File/Blob + object URL + optional metadata.
//
// This is the foundation `fs-access.ts` (Phase 3.1) upgrades to a user-selected real folder
// via the File System Access API. Today it already gives the app genuinely working file
// semantics in the browser (open, save, download) with no backend.

export interface StoredAsset {
  key: string
  name: string
  mimeType: string
  size: number
  kind: 'file' | 'blob'
  blobUrl: string
  width?: number
  height?: number
  duration?: number
}

const assets = new Map<string, StoredAsset>()

const PREFIX = 'web://'

export function isWebPath(p: string | null | undefined): p is string {
  return typeof p === 'string' && p.startsWith(PREFIX)
}

export function registerBlob(data: Blob | ArrayBuffer, name = 'file', mimeType?: string): string {
  const blob: Blob =
    data instanceof ArrayBuffer ? new Blob([data], { type: mimeType ?? 'application/octet-stream' }) : data
  const key = `${PREFIX}${crypto.randomUUID()}`
  const rec: StoredAsset = {
    key,
    name,
    mimeType: blob.type || mimeType || 'application/octet-stream',
    size: blob.size,
    kind: 'blob',
    blobUrl: URL.createObjectURL(blob),
  }
  assets.set(key, rec)
  return key
}

export function registerFile(file: File): string {
  const key = `${PREFIX}${crypto.randomUUID()}`
  const rec: StoredAsset = {
    key,
    name: file.name,
    mimeType: file.type,
    size: file.size,
    kind: 'file',
    blobUrl: URL.createObjectURL(file),
  }
  assets.set(key, rec)
  return key
}

export function getAsset(key: string): StoredAsset | undefined {
  return assets.get(key)
}

export function getBlobUrl(key: string): string | undefined {
  return assets.get(key)?.blobUrl
}

export function listAssetKeys(): string[] {
  return [...assets.keys()]
}

export function removeAsset(key: string): void {
  const a = assets.get(key)
  if (a) URL.revokeObjectURL(a.blobUrl)
  assets.delete(key)
}

/**
 * Read a stored asset back as a data URL + mime, mirroring the old readLocalFile IPC.
 * Falls back to an empty result if the key is unknown (never throws).
 */
export function readDataUrl(key: string): { data: string; mimeType: string } {
  const a = assets.get(key)
  if (!a) return { data: '', mimeType: 'text/plain' }
  return { data: a.blobUrl, mimeType: a.mimeType }
}

/** Apply measured dimensions/duration for an image or video asset (thumbs + previews). */
export function setDimensions(key: string, width: number, height: number): void {
  const a = assets.get(key)
  if (a) {
    a.width = width
    a.height = height
  }
}

export function setDuration(key: string, duration: number): void {
  const a = assets.get(key)
  if (a) a.duration = duration
}

/** Probe an image or video object URL and store its intrinsic size. Real media metadata. */
export async function measureMedia(key: string, kind: 'video' | 'image'): Promise<{ width: number; height: number }> {
  const a = assets.get(key)
  if (!a) throw new Error('Unknown asset')
  const url = a.blobUrl
  if (kind === 'image') {
    const img = new Image()
    img.src = url
    await img.decode()
    setDimensions(key, img.naturalWidth, img.naturalHeight)
    return { width: img.naturalWidth, height: img.naturalHeight }
  }
  return await new Promise((resolve, reject) => {
    const v = document.createElement('video')
    v.preload = 'metadata'
    v.src = url
    v.onloadedmetadata = () => {
      setDimensions(key, v.videoWidth, v.videoHeight)
      if (Number.isFinite(v.duration)) setDuration(key, v.duration)
      resolve({ width: v.videoWidth, height: v.videoHeight })
    }
    v.onerror = () => reject(new Error('Could not read video metadata'))
  })
}

/** Extract a frame from a video at seekTime using an offscreen canvas (real, no ffmpeg). */
export async function extractFrame(
  key: string,
  seekTime: number,
  width?: number,
  quality = 0.92,
): Promise<string> {
  const a = assets.get(key)
  if (!a) throw new Error('Unknown asset')
  const v = document.createElement('video')
  v.muted = true
  v.playsInline = true
  v.preload = 'auto'
  v.src = a.blobUrl
  await new Promise<void>((resolve, reject) => {
    v.onloadeddata = () => resolve()
    v.onerror = () => reject(new Error('Could not load video for frame extraction'))
  })
  if (Number.isFinite(v.duration) && seekTime < v.duration) v.currentTime = Math.max(0, seekTime)
  await new Promise<void>((resolve) => {
    v.onseeked = () => resolve()
    // If the seek is a no-op (element already at that time) the event may not fire.
    setTimeout(resolve, 400)
  })
  const w = width ? Math.round(width) : v.videoWidth
  const scale = w / v.videoWidth
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = Math.round(v.videoHeight * scale)
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Canvas unavailable')
  ctx.drawImage(v, 0, 0, canvas.width, canvas.height)
  const blob: Blob | null = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', quality))
  if (!blob) throw new Error('Frame encoding failed')
  return registerBlob(blob, `${a.name.split('.').slice(0, -1).join('.') || 'frame'}-${Math.round(seekTime * 1000)}.jpg`, 'image/jpeg')
}
