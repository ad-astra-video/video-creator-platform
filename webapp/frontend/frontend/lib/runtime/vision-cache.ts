/**
 * Disk-backed result cache for analysis-style runner calls keyed by the input
 * image's content hash (see direct-transport's suggest-layers / SAM3 caching).
 *
 * IndexedDB is the browser's persistent, disk-backed store — the same layer the
 * fs-access module uses for its project-assets directory handle — so the cache
 * survives a full page reload and app restart, and is rehydrated from disk on
 * load (ensureLoaded). Values are JSON-serializable; masks / layer payloads are
 * stored as base64 text, so the store can hold them comfortably.
 *
 * The whole store is one object `{ <cacheKey>: value }` under a single row, so
 * a failure is contains a single write. Never throws to the caller.
 */

const DB_NAME = 'vcp-vision-cache'
const STORE = 'cache'
const ROW = 'entries'

const cache = new Map<string, unknown>()
let ready: Promise<void> | null = null

function hasIdb(): boolean {
  return typeof indexedDB !== 'undefined'
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

/** Load persisted entries into the in-memory map once; safe to call repeatedly. */
function ensureLoaded(): Promise<void> {
  if (ready) return ready
  ready = (async () => {
    if (!hasIdb()) return
    try {
      const db = await openDB()
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE, 'readonly')
        const req = tx.objectStore(STORE).get(ROW)
        req.onsuccess = () => {
          const raw = req.result as Record<string, unknown> | undefined
          if (raw) for (const [k, v] of Object.entries(raw)) cache.set(k, v)
          resolve()
        }
        req.onerror = () => reject(req.error)
      })
      db.close()
    } catch (e) {
      console.warn('[vision-cache] failed to hydrate from disk:', e)
    }
  })()
  return ready
}

function persist(): Promise<void> {
  return (async () => {
    if (!hasIdb()) return
    try {
      const db = await openDB()
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE, 'readwrite')
        tx.objectStore(STORE).put(Object.fromEntries(cache), ROW)
        tx.oncomplete = () => resolve()
        tx.onerror = () => reject(tx.error)
      })
      db.close()
    } catch (e) {
      console.warn('[vision-cache] failed to persist to disk:', e)
    }
  })()
}

/** Read a cached value (undefined when absent). Hydrates the store first. */
export async function getCachedValue<T>(key: string): Promise<T | undefined> {
  await ensureLoaded()
  return cache.get(key) as T | undefined
}

/** Store a value and write the store to disk. */
export async function setCachedValue(key: string, value: unknown): Promise<void> {
  await ensureLoaded()
  cache.set(key, value)
  await persist()
}
