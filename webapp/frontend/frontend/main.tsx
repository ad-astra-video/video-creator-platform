import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { installProjectStorageDevtools } from './lib/project-storage-devtools'
import { createWebElectronAPI } from './lib/runtime/web-electron-api'
import { listProjectFolderAssets } from './lib/runtime/fs-access'
import { registerPersistedAsset, restoreAssets } from './lib/runtime/web-store'
import './index.css'

// Web runtime: install a real browser implementation of the Electron bridge. In Electron this
// is provided by the preload script; in the static web app we provide it here so the (unchanged)
// view/component code can call window.electronAPI and get genuine browser behavior.
if (typeof window !== 'undefined' && !window.electronAPI) {
  window.electronAPI = createWebElectronAPI()
}

installProjectStorageDevtools()

// Rehydrate the in-memory web asset store BEFORE the first render so that previously-saved
// images/videos (referenced by localStorage projects as `web://<uuid>` keys) render again after
// a browser reload. First restore from the IndexedDB mirror, then re-scan the user's chosen
// project-assets folder and register any real files found on disk. Both are best-effort and
// never throw.
void restoreAssets().then(async () => {
  try {
    const folderAssets = await listProjectFolderAssets()
    let fromFolder = 0
    for (const a of folderAssets) if (registerPersistedAsset(a.key, a.data, a.name, a.mimeType)) fromFolder++
    if (fromFolder > 0) console.log(`[web-store] restored ${fromFolder} asset(s) from project folder`)
  } catch {
    /* best-effort */
  }
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  )
})
