import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { installProjectStorageDevtools } from './lib/project-storage-devtools'
import { createWebElectronAPI } from './lib/runtime/web-electron-api'
import './index.css'

// Web runtime: install a real browser implementation of the Electron bridge. In Electron this
// is provided by the preload script; in the static web app we provide it here so the (unchanged)
// view/component code can call window.electronAPI and get genuine browser behavior.
if (typeof window !== 'undefined' && !window.electronAPI) {
  window.electronAPI = createWebElectronAPI()
}

installProjectStorageDevtools()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
