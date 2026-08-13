// File System Access API wrapper (Chromium-only; Firefox/Safari fall back to OPFS + folder
// export, surfaced via supportsFileSystemAccess()). Persists the user-chosen project-assets
// directory handle in IndexedDB so its permission can be re-requested on later visits instead
// of re-picking the folder every time.
//
// This is also where saved project assets become REAL files on disk. Each project gets its own
// subfolder under the chosen folder (`<folder>/<projectId>/<uuid>.<ext>`), so projects stay
// isolated on disk and projects load their assets from disk on reload:
//   - `saveAssetToProjectFolder` writes a saved image/video into that project's subfolder.
//   - `deleteAssetFromProjectFolder` removes an asset's file when the item is deleted.
//   - `listProjectFolderAssets` rescans every project subfolder and returns the files re-keyed
//     to `web://<uuid>` so the caller can re-register them into the web store.

const DB_NAME = 'vcp-fs'
const STORE = 'dirs'
const KEY = 'project-assets'

// Chromium-only File System Access API surface not present in this repo's lib.dom typings
// (same reason `requestPermission`, `createWritable`, `values`, `getFileHandle` aren't typed).
interface FsWritableLike {
  write(data: Blob): Promise<void>
  close(): Promise<void>
}
interface FsFileHandleLike {
  createWritable(): Promise<FsWritableLike>
}
interface FsDirEntryLike {
  kind: string
  name: string
  getFile(): Promise<File>
}
interface FsDirLike {
  getFileHandle(name: string, opts?: { create?: boolean }): Promise<FsFileHandleLike>
  getDirectoryHandle(name: string, opts?: { create?: boolean }): Promise<FsDirLike>
  values(): AsyncIterableIterator<FsDirEntryLike>
  removeEntry(name: string, opts?: { recursive?: boolean }): Promise<void>
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE)
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

export function supportsFileSystemAccess(): boolean {
  return typeof window !== 'undefined' && 'showDirectoryPicker' in window
}

export async function getProjectAssetsHandle(): Promise<FileSystemDirectoryHandle | null> {
  if (!('indexedDB' in window)) return null
  try {
    const db = await openDB()
    return await new Promise<FileSystemDirectoryHandle | null>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readonly')
      const req = tx.objectStore(STORE).get(KEY)
      req.onsuccess = () => resolve((req.result as FileSystemDirectoryHandle) ?? null)
      req.onerror = () => reject(req.error)
    })
  } catch {
    return null
  }
}

export async function saveProjectAssetsHandle(handle: FileSystemDirectoryHandle): Promise<void> {
  if (!('indexedDB' in window)) return
  const db = await openDB()
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite')
    tx.objectStore(STORE).put(handle, KEY)
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

/** The folder label (or '') for the current project-assets location. */
export async function getProjectAssetsName(): Promise<string> {
  const h = await getProjectAssetsHandle()
  return h ? h.name : ''
}

/**
 * Prompt the user to choose the folder where project assets are saved. Persists the handle
 * so future visits only need a permission re-grant, not a fresh pick.
 * Throws if the browser lacks the File System Access API or the user cancels.
 */
export async function pickProjectAssetsFolder(): Promise<FileSystemDirectoryHandle> {
  const picker = (window as unknown as { showDirectoryPicker?: (opts?: unknown) => Promise<FileSystemDirectoryHandle> }).showDirectoryPicker
  if (!picker) throw new Error('This browser does not support choosing a folder. Please use Chrome, Edge or Opera.')
  const handle = await picker({ id: 'project-assets', mode: 'readwrite', startIn: 'documents' })
  await saveProjectAssetsHandle(handle)
  return handle
}

/** Re-request read/write permission on a previously-saved handle (returns true if granted). */
export async function requestProjectAssetsPermission(): Promise<boolean> {
  const h = await getProjectAssetsHandle()
  if (!h) return false
  try {
    const opts = { mode: 'readwrite' } as const
    // @ts-expect-error - FileSystemHandle.requestPermission is Chromium-only and not in all lib dom types.
    const state = await h.requestPermission(opts)
    return state === 'granted'
  } catch {
    return true // permission already held (persisted handle) — treat as usable
  }
}

/** The saved-assets folder wrapped in a read/write-permissioned handle, or null. */
async function requireWriteableAssetsHandle(): Promise<FileSystemDirectoryHandle | null> {
  const h = await getProjectAssetsHandle()
  if (!h) return null
  const ok = await requestProjectAssetsPermission()
  return ok ? h : null
}

/** Map a MIME type to a short filename extension (falls back to the saved name's ext). */
function extForMime(mimeType: string, fallbackName: string): string {
  const byMime: Array<[string, string]> = [
    ['image/png', 'png'],
    ['image/jpeg', 'jpg'],
    ['image/webp', 'webp'],
    ['image/gif', 'gif'],
    ['video/mp4', 'mp4'],
    ['video/webm', 'webm'],
    ['video/quicktime', 'mov'],
    ['audio/mpeg', 'mp3'],
    ['application/json', 'json'],
  ]
  for (const [m, e] of byMime) if (mimeType.includes(m)) return e
  if (fallbackName && fallbackName.includes('.')) {
    const last = fallbackName.split('.').pop()!
    if (/^[a-z0-9]{1,5}$/i.test(last)) return last
  }
  return 'bin'
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

function uuidFromAssetsKey(key: string): string | null {
  const prefix = 'web://'
  if (!key.startsWith(prefix)) return null
  const uuid = key.slice(prefix.length)
  return UUID_RE.test(uuid) ? uuid : null
}

export interface ProjectFolderAsset {
  key: string
  data: Blob
  name: string
  mimeType: string
}

/**
 * Write a saved project asset into `<folder>/<projectId>/<uuid>.<ext>`. The filename preserves
 * the `web://<uuid>` key so a later rescan can rebuild the same key. Best-effort: returns false
 * (no throw) when no folder is selected, the project id is missing, or the write fails.
 */
export async function saveAssetToProjectFolder(
  projectId: string,
  key: string,
  data: Blob,
  name: string,
  mimeType: string,
): Promise<boolean> {
  const uuid = uuidFromAssetsKey(key)
  if (!uuid || !projectId) return false
  const root = await requireWriteableAssetsHandle()
  if (!root) return false
  try {
    const r = root as unknown as FsDirLike
    const projDir = await r.getDirectoryHandle(projectId, { create: true })
    const filename = `${uuid}.${extForMime(mimeType || data.type, name)}`
    let fh: FsFileHandleLike
    try {
      fh = await projDir.getFileHandle(filename)
    } catch {
      fh = await projDir.getFileHandle(filename, { create: true })
    }
    const writable = await fh.createWritable()
    await writable.write(data)
    await writable.close()
    return true
  } catch (e) {
    console.warn('[fs-access] failed to write project asset to folder:', e)
    return false
  }
}

/**
 * Delete an asset's file (`<uuid>.<ext>`) from `<folder>/<projectId>/`. Best-effort; returns
 * true only if a matching file was actually removed. Never throws.
 */
export async function deleteAssetFromProjectFolder(projectId: string, key: string): Promise<boolean> {
  const uuid = uuidFromAssetsKey(key)
  if (!uuid || !projectId) return false
  const root = await requireWriteableAssetsHandle()
  if (!root) return false
  try {
    const r = root as unknown as FsDirLike
    let projDir: FsDirLike
    try {
      projDir = await r.getDirectoryHandle(projectId)
    } catch {
      return false // no subfolder for this project — nothing to delete
    }
    for await (const entry of projDir.values()) {
      if (entry.kind !== 'file') continue
      const dot = entry.name.lastIndexOf('.')
      const stem = dot > 0 ? entry.name.slice(0, dot) : entry.name
      if (stem === uuid) {
        await projDir.removeEntry(entry.name)
        return true
      }
    }
  } catch (e) {
    console.warn('[fs-access] failed to delete project asset from folder:', e)
  }
  return false
}

/**
 * Scan every project subfolder under the chosen folder for saved assets (files named
 * `<uuid>.<ext>`) and return them re-keyed to `web://<uuid>` so the caller can re-register
 * them into the store. Returns an empty list when there's no folder or permission isn't
 * granted. Never throws.
 */
export async function listProjectFolderAssets(): Promise<ProjectFolderAsset[]> {
  const root = await requireWriteableAssetsHandle()
  if (!root) return []
  const out: ProjectFolderAsset[] = []
  try {
    const r = root as unknown as FsDirLike
    for await (const projEntry of r.values()) {
      if (projEntry.kind !== 'directory') continue
      let projDir: FsDirLike
      try {
        projDir = await r.getDirectoryHandle(projEntry.name)
      } catch {
        continue
      }
      try {
        for await (const entry of projDir.values()) {
          if (entry.kind !== 'file') continue
          const dot = entry.name.lastIndexOf('.')
          const stem = dot > 0 ? entry.name.slice(0, dot) : entry.name
          if (!UUID_RE.test(stem)) continue // only files we wrote (uuid-named)
          const file = await entry.getFile()
          out.push({ key: `web://${stem}`, data: file as Blob, name: entry.name, mimeType: file.type })
        }
      } catch {
        /* skip a project dir we can't read */
      }
    }
  } catch (e) {
    console.warn('[fs-access] failed to scan project-assets folder:', e)
  }
  return out
}
