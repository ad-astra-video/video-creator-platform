// File System Access API wrapper (Chromium-only; Firefox/Safari fall back to OPFS + folder
// export, surfaced via supportsFileSystemAccess()). Persists the user-chosen project-assets
// directory handle in IndexedDB so its permission can be re-requested on later visits instead
// of re-picking the folder every time.

const DB_NAME = 'vcp-fs'
const STORE = 'dirs'
const KEY = 'project-assets'

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
