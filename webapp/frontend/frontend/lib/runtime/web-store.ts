// Web asset store: a session-persistent, in-browser stand-in for the OS filesystem that
// `window.electronAPI` referenced. Every "file path" the UI deals with is a synthetic
// `web://<uuid>` key mapped to a real File/Blob + object URL + optional metadata.
//
// This is the foundation `fs-access.ts` (Phase 3.1) upgrades to a user-selected real folder
// via the File System Access API. Today it already gives the app genuinely working file
// semantics in the browser (open, save, download) with no backend.
//
// Persistence: the in-memory `assets` Map is session-only. Projects in localStorage reference
// `web://<uuid>` keys, so to make previously-saved images/videos survive a browser reload we
// mirror every stored asset's bytes into IndexedDB (keyed by the same web:// key) and rebuild
// the object URLs on startup via `restoreAssets()`.

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

/** Private field carrying the raw bytes so they can be mirrored to IndexedDB. */
interface StoredAssetWithData {
  data?: Blob
}

const assets = new Map<string, StoredAsset>()

const PREFIX = 'web://'

// ---- IndexedDB persistence -------------------------------------------------

const DB_NAME = 'vcp-asset-store'
const DB_STORE = 'assets'

interface PersistedAsset {
  key: string
  name: string
  mimeType: string
  kind: 'file' | 'blob'
  size: number
  data: Blob
  width?: number
  height?: number
  duration?: number
}

function idbAvailable(): boolean {
  return typeof window !== 'undefined' && 'indexedDB' in window
}

function openAssetDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(DB_STORE)) db.createObjectStore(DB_STORE, { keyPath: 'key' })
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

function toPersisted(rec: StoredAsset): PersistedAsset | null {
  const data = (rec as StoredAssetWithData).data
  if (!data) return null
  return {
    key: rec.key,
    name: rec.name,
    mimeType: rec.mimeType,
    kind: rec.kind,
    size: rec.size || data.size,
    data,
    width: rec.width,
    height: rec.height,
    duration: rec.duration,
  }
}

/** Mirror an asset's bytes into IndexedDB. Best-effort (fire-and-forget). */
function persistRecord(rec: StoredAsset): void {
  if (!idbAvailable()) return
  const p = toPersisted(rec)
  if (!p) return
  void (async () => {
    try {
      const db = await openAssetDB()
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(DB_STORE, 'readwrite')
        tx.objectStore(DB_STORE).put(p)
        tx.oncomplete = () => resolve()
        tx.onerror = () => reject(tx.error)
      })
    } catch {
      /* quota exceeded / private mode — persistence is best-effort */
    }
  })()
}

async function deletePersisted(key: string): Promise<void> {
  if (!idbAvailable()) return
  try {
    const db = await openAssetDB()
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(DB_STORE, 'readwrite')
      tx.objectStore(DB_STORE).delete(key)
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
  } catch {
    /* ignore */
  }
}

/**
 * Rebuild the in-memory store from IndexedDB. Call once at app startup (before render) so
 * saved images/videos referenced by localStorage projects load again after a browser reload.
 */
export async function restoreAssets(): Promise<void> {
  if (!idbAvailable()) return
  try {
    const db = await openAssetDB()
    const entries = await new Promise<PersistedAsset[]>((resolve, reject) => {
      const tx = db.transaction(DB_STORE, 'readonly')
      const req = tx.objectStore(DB_STORE).getAll()
      req.onsuccess = () => resolve((req.result as PersistedAsset[]) ?? [])
      req.onerror = () => reject(req.error)
    })
    let restored = 0
    for (const p of entries) {
      if (!p || !p.key || !p.data || assets.has(p.key)) continue
      const rec: StoredAsset = {
        key: p.key,
        name: p.name || 'file',
        mimeType: p.mimeType || 'application/octet-stream',
        size: p.size || p.data.size,
        kind: p.kind === 'file' ? 'file' : 'blob',
        blobUrl: URL.createObjectURL(p.data),
      }
      ;(rec as StoredAssetWithData).data = p.data
      if (p.width != null) rec.width = p.width
      if (p.height != null) rec.height = p.height
      if (p.duration != null) rec.duration = p.duration
      assets.set(p.key, rec)
      restored++
    }
    if (restored > 0) console.log(`[web-store] restored ${restored} saved asset(s)`)
  } catch (e) {
    console.error('[web-store] failed to restore saved assets:', e)
  }
}

// ---- store API --------------------------------------------------------------

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
  ;(rec as StoredAssetWithData).data = blob
  assets.set(key, rec)
  persistRecord(rec)
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
  ;(rec as StoredAssetWithData).data = file
  assets.set(key, rec)
  persistRecord(rec)
  return key
}

export function getAsset(key: string): StoredAsset | undefined {
  return assets.get(key)
}

export function getBlobUrl(key: string): string | undefined {
  return assets.get(key)?.blobUrl
}

/** The raw bytes behind a key (for writing to disk), or undefined when unknown. */
export function getBlob(key: string): Blob | undefined {
  return (assets.get(key) as StoredAssetWithData | undefined)?.data
}

/**
 * Register bytes that came from a durable source (our own IndexedDB mirror or the user's
 * project-assets folder) under an already-known `web://<uuid>` key. Used on startup to rebuild
 * the store so saved images render again after a reload. Returns false if the key is already
 * live (the fresher mapping wins). Unlike registerBlob/registerFile this does NOT re-persist,
 * to avoid a write-back loop.
 */
export function registerPersistedAsset(key: string, blob: Blob, name: string, mimeType: string): boolean {
  if (key.startsWith(PREFIX) && assets.has(key)) return false
  const rec: StoredAsset = {
    key,
    name: name || 'file',
    mimeType: mimeType || blob.type || 'application/octet-stream',
    size: blob.size,
    kind: 'blob',
    blobUrl: URL.createObjectURL(blob),
  }
  ;(rec as StoredAssetWithData).data = blob
  assets.set(key, rec)
  return true
}

export function listAssetKeys(): string[] {
  return [...assets.keys()]
}

export function removeAsset(key: string): void {
  const a = assets.get(key)
  if (a) URL.revokeObjectURL(a.blobUrl)
  assets.delete(key)
  void deletePersisted(key)
}

/**
 * Discard ONLY the IndexedDB-persisted mirror of an asset while keeping it live in memory
 * (object URL + bytes) for the current session. Inverse of persistRecord — used once the
 * asset's bytes are durably written to the user's project-assets folder on disk, so disk
 * items don't also occupy IndexedDB. Reload re-derives the key from the folder rescan
 * (registerPersistedAsset, which never re-mirrors to IndexedDB).
 */
export function unpersistAsset(key: string): void {
  void deletePersisted(key)
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
    persistRecord(a)
  }
}

export function setDuration(key: string, duration: number): void {
  const a = assets.get(key)
  if (a) {
    a.duration = duration
    persistRecord(a)
  }
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
export const store = {
  isWebPath,
  registerBlob,
  registerFile,
  getAsset,
  getBlobUrl,
  getBlob,
  registerPersistedAsset,
  listAssetKeys,
  removeAsset,
  unpersistAsset,
  readDataUrl,
  setDimensions,
  setDuration,
  measureMedia,
  extractFrame,
  restoreAssets,
}

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
