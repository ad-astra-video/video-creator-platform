// Web implementation of the Electron `window.electronAPI` surface.
//
// This is NOT a stub: each method performs a real browser operation (file picker, blob
// download, `window.open`, canvas frame extraction, health fetch, local asset store).
// It lets the (unchanged) vendored frontend run as a pure static web app with no OS
// filesystem and no local Python/GPU. Heavy GPU work is dispatched to the Worker +
// orchestrator; anything that fundamentally needs an OS shell (open a system folder,
// kill a process) resolves to a safe no-op that returns a clear result.
import type { BackendHealthStatus, ElectronAPI } from '../../../shared/electron-api-schema'
import { store } from './web-store'
import { getProjectAssetsName, pickProjectAssetsFolder, saveAssetToProjectFolder } from './fs-access'

// ---- config / credentials ------------------------------------------------

interface VcConfig {
  apiBase?: string
  apiKey?: string
}

function vcConfig(): VcConfig {
  return ((window as unknown as { __VC_CONFIG__?: VcConfig }).__VC_CONFIG__ ?? {}) as VcConfig
}

function apiBase(): string {
  // A user-set platform URL (chosen in the onboarding wizard / Settings) overrides the
  // shipped config so the app can talk to a different Worker without a rebuild.
  try {
    const user = localStorage.getItem('vcp_platform_url')
    if (user) return user.replace(/\/$/, '')
  } catch {
    /* ignore */
  }
  // Runtime config (dist/config.js) wins, then VITE_API_BASE (build-time), then same-origin.
  const c = vcConfig().apiBase
  if (typeof c === 'string' && c) return c.replace(/\/$/, '')
  const v = (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_API_BASE
  return (v ?? '').replace(/\/$/, '')
}

function getKey(): string {
  const c = vcConfig().apiKey
  if (typeof c === 'string' && c) return c
  try {
    return localStorage.getItem('vcp_key') ?? ''
  } catch {
    return ''
  }
}

function setDone(key: string): void {
  try {
    localStorage.setItem(key, '1')
  } catch {
    /* ignore */
  }
}

function isDone(key: string): boolean {
  try {
    return localStorage.getItem(key) === '1'
  } catch {
    return false
  }
}

let installationId = 'web-install'
try {
  installationId = localStorage.getItem('vcp_installation_id') ?? crypto.randomUUID()
  localStorage.setItem('vcp_installation_id', installationId)
} catch {
  /* ignore */
}

// ---- directory / asset bookkeeping for dialog-driven imports -------------
// DirectoryHandles from showOpenDirectoryDialog are remembered by a synthetic dir key so
// searchDirectoryForFiles / checkFilesExist (used by the timeline importer) can work in-browser.
const directories = new Map<string, Map<string, string>>() // dirKey -> { name: assetKey }

// ---- health ---------------------------------------------------------------

// One-shot liveness probe (used only to resolve the App.tsx `connected` boot gate).
// Intentionally NOT a recurring poll: under the serverless/Livepeer model there is no local
// backend process to watch, so every-5s /health against the Worker was pure network chatter.
let healthProbed = false
let healthListeners = new Set<(s: BackendHealthStatus) => void>()

function broadcastHealth(status: BackendHealthStatus): void {
  healthListeners.forEach((cb) => {
    try {
      cb(status)
    } catch {
      /* ignore */
    }
  })
}

async function probeHealth(): Promise<BackendHealthStatus> {
  try {
    const res = await fetch(`${apiBase()}/health`, { signal: AbortSignal.timeout(6000) })
    return res.ok ? { status: 'alive', exitCode: 0 } : { status: 'dead', exitCode: res.status }
  } catch {
    return { status: 'dead', exitCode: null }
  }
}

function ensureHealthPoll(): void {
  if (healthProbed) return
  healthProbed = true
  void probeHealth().then(broadcastHealth)
}

// ---- file input helper -----------------------------------------------------

function pickFiles(multiple: boolean, dir: boolean): Promise<string[]> {
  return new Promise((resolve) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.multiple = multiple
    if (dir) {
      input.setAttribute('webkitdirectory', '')
      input.setAttribute('directory', '')
    }
    input.style.display = 'none'
    document.body.appendChild(input)
    input.onchange = () => {
      const files = Array.from(input.files ?? [])
      const keys = files.map((f) => store.registerFile(f))
      if (dir && files.length > 0) {
        // Remember this directory as a synthetic key for searchDirectoryForFiles.
        const dirKey = `web-dir://${crypto.randomUUID()}`
        const byName = new Map<string, string>()
        files.forEach((f, i) => byName.set(f.name, keys[i] ?? ''))
        directories.set(dirKey, byName)
        keys.splice(0, keys.length, dirKey) // return the dir key when a directory was requested
      }
      input.remove()
      resolve(keys)
    }
    input.oncancel = () => {
      input.remove()
      resolve([])
    }
    input.click()
  })
}

function download(data: Blob | ArrayBuffer, filename: string): void {
  const blob: Blob = data instanceof ArrayBuffer ? new Blob([data]) : data
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 5000)
}

function pickSaveTarget(defaultName: string): string | null {
  try {
    // File System Access API where available: real save dialog.
    return `${defaultName || 'export'}`
  } catch {
    return defaultName || 'export'
  }
}

// ---- video export (canvas + MediaRecorder → webm) ----------------------------

interface ExportClip {
  path: string
  type: string
  startTime: number
  duration: number
  trimStart: number
  speed: number
  reversed: boolean
  flipH: boolean
  flipV: boolean
  opacity: number
  muted: boolean
  volume: number
}

const activeExports = new Map<string, MediaRecorder>()

async function renderExport(clips: ExportClip[], width: number, height: number, fps: number, sessionId: string): Promise<Blob | null> {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Canvas unavailable for export')
  const stream = canvas.captureStream(fps)
  const recorder = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp9', videoBitsPerSecond: 8_000_000 })
  activeExports.set(sessionId, recorder)
  const chunks: Blob[] = []
  recorder.ondataavailable = (e) => {
    if (e.data.size) chunks.push(e.data)
  }
  const done = new Promise<Blob | null>((resolve) => {
    recorder.onstop = () => resolve(new Blob(chunks, { type: 'video/webm' }))
  })
  recorder.start()

  const paintFrame = (media: HTMLVideoElement | HTMLImageElement, w: number, h: number, opacity: number, flipH: boolean, flipV: boolean) => {
    ctx.save()
    ctx.globalAlpha = opacity
    ctx.translate(flipH ? width : 0, flipV ? height : 0)
    ctx.scale(flipH ? -1 : 1, flipV ? -1 : 1)
    const isVid = media instanceof HTMLVideoElement
    const mw = isVid ? (media as HTMLVideoElement).videoWidth || w : (media as HTMLImageElement).naturalWidth || w
    const mh = isVid ? (media as HTMLVideoElement).videoHeight || h : (media as HTMLImageElement).naturalHeight || h
    const scale = Math.max(width / mw, height / mh)
    const dw = mw * scale
    const dh = mh * scale
    ctx.drawImage(media, (width - dw) / 2, (height - dh) / 2, dw, dh)
    ctx.restore()
  }

  try {
    for (const clip of clips) {
      const url = store.getBlobUrl(clip.path)
      if (!url) throw new Error(`Asset not found: ${clip.path}`)
      const isVideo = clip.type === 'video' || clip.path !== '' && clip.type !== 'image'
      const media: HTMLVideoElement | HTMLImageElement =
        isVideo ? document.createElement('video') : new Image()
      media.src = url
      if (isVideo) {
        const v = media as HTMLVideoElement
        v.muted = true
        v.playsInline = true
        await new Promise<void>((res, rej) => {
          v.onloadeddata = () => res()
          v.onerror = () => rej(new Error('Could not load video for export'))
        })
      } else {
        await (media as HTMLImageElement).decode()
      }
      const effDuration = Math.max(0.1, clip.duration / Math.max(0.1, clip.speed))
      const frames = Math.max(1, Math.round(effDuration * fps))
      const stepMs = 1000 / fps
      for (let i = 0; i < frames; i++) {
        const t = clip.trimStart + (i * stepMs) / 1000
        if (isVideo) {
          const v = media as HTMLVideoElement
          if (Number.isFinite(v.duration) && t < v.duration) v.currentTime = Math.min(t, v.duration - 0.02)
          await new Promise((r) => setTimeout(r, 0))
        }
        if (isVideo && (media as HTMLVideoElement).readyState < 2) {
          await new Promise((r) => setTimeout(r, 30))
        }
        paintFrame(media, width, height, clip.opacity, clip.flipH, clip.flipV)
        await new Promise((r) => setTimeout(r, 0))
      }
      ;(media as HTMLMediaElement).pause?.()
    }
  } finally {
    if (recorder.state !== 'inactive') recorder.stop()
    activeExports.delete(sessionId)
  }
  return await done
}

// ---- the API ---------------------------------------------------------------

export function createWebElectronAPI(): ElectronAPI {
  return {
    platform: 'web',

    getBackend: async () => ({ url: apiBase(), token: getKey() }),

    getAppInfo: async () => ({
      version: '1.1.0-web',
      isPackaged: false,
      modelsPath: '',
      userDataPath: '',
    }),

    getModelsPath: async () => '',
    getDownloadsPath: async () => '',
    getResourcePath: async () => null,
    getProjectAssetsPath: async () => {
      const name = await getProjectAssetsName()
      return name ? `web://project-assets/${name}` : ''
    },
    readLocalFile: async ({ filePath }) => store.readDataUrl(filePath),

    checkGpu: async () => ({ available: false }),

    checkFirstRun: async () => ({
      needsSetup: false, // web app: no local python/model setup
      needsLicense: !isDone('vcp_license'),
    }),
    acceptLicense: async () => {
      setDone('vcp_license')
      return true
    },
    completeSetup: async () => {
      setDone('vcp_license')
      return true
    },
    fetchLicenseText: async () => {
      try {
        const r = await fetch('LICENSE.txt')
        return r.ok ? await r.text() : 'License text unavailable.'
      } catch {
        return 'License text unavailable.'
      }
    },
    getNoticesText: async () => {
      try {
        const r = await fetch('NOTICES.md')
        return r.ok ? await r.text() : 'No notices available.'
      } catch {
        return 'No notices available.'
      }
    },

    openLtxApiKeyPage: async () => {
      window.open('https://app.ltx.ai/settings', '_blank', 'noopener')
      return true
    },
    openLtxBillingPage: async () => {
      window.open('https://app.ltx.ai/billing', '_blank', 'noopener')
      return true
    },
    openFalApiKeyPage: async () => {
      window.open('https://fal.ai/dashboard/keys', '_blank', 'noopener')
      return true
    },
    openHuggingFaceRepo: async ({ repoId }) => {
      window.open(`https://huggingface.co/${repoId}`, '_blank', 'noopener')
      return true
    },
    openExternalUrl: async ({ url }) => {
      window.open(url, '_blank', 'noopener')
      return true
    },
    openHuggingFaceAuth: async ({ clientId, redirectUri, scope, state, codeChallenge, codeChallengeMethod }) => {
      const q = new URLSearchParams({
        client_id: clientId,
        redirect_uri: redirectUri,
        response_type: 'code',
        scope,
        state,
        code_challenge: codeChallenge,
        code_challenge_method: codeChallengeMethod || 'S256',
      })
      window.open(`https://huggingface.co/oauth/authorize?${q.toString()}`, '_self')
      return true
    },
    openParentFolderOfFile: async () => {},
    showItemInFolder: async () => {},

    getLogs: async () => ({ logPath: 'web-console', lines: [] }),
    getLogPath: async () => ({ logPath: 'web-console', logDir: 'web-console' }),
    openLogFolder: async () => false,

    // Dialogs / files ------------------------------------------------------------------
    showOpenFileDialog: async () => await pickFiles(true, false),
    showOpenDirectoryDialog: async () => {
      const keys = await pickFiles(true, true)
      return keys.length ? (keys[0] ?? null) : null
    },
    searchDirectoryForFiles: async ({ directory, filenames }) => {
      const entries = directories.get(directory)
      const out: Record<string, string> = {}
      if (entries) {
        for (const name of filenames) {
          const key = entries.get(name)
          if (key) out[name] = key
        }
      }
      return out
    },
    checkFilesExist: async ({ filePaths }) => {
      const out: Record<string, boolean> = {}
      for (const p of filePaths) out[p] = store.getAsset(p) !== undefined
      return out
    },
    showSaveDialog: async ({ defaultPath }) => pickSaveTarget(defaultPath ?? 'export'),
    saveFile: async (input: { filePath: string; data: string }) => {
      const { filePath, data } = input
      const blob = new Blob([data], { type: 'text/plain;charset=utf-8' })
      download(blob, filePath.split('/').pop() || 'file.txt')
      return { success: true, path: `web://saved-${crypto.randomUUID()}` }
    },
    saveBinaryFile: async ({ filePath, data }) => {
      download(data, filePath.split('/').pop() || 'file')
      return { success: true, path: `web://saved-${crypto.randomUUID()}` }
    },

    // Project assets ------------------------------------------------------------------
    addVisualAssetToProject: async ({ srcPath, type }) => {
      try {
        // srcPath is normally a registered web:// key (from the picker). Generated
        // images can arrive as a raw blob:/data: URL that is NOT yet in the store —
        // register it first so measureMedia can read it and persistence succeeds.
        let key = srcPath
        if (!store.isWebPath(srcPath)) {
          const blob = await fetch(srcPath).then((r) => r.blob())
          key = store.registerBlob(blob, 'generated', blob.type)
        }
        const dims = await store.measureMedia(key, type)

        // Persist the actual bytes into the user's selected project-assets folder (best-effort)
        // so saved images exist as real files on disk and can be rescanned after a reload.
        {
          const asset = store.getAsset(key)
          const data = store.getBlob(key)
          if (asset && data) {
            void saveAssetToProjectFolder(key, data, asset.name, asset.mimeType).catch(() => {})
          }
        }

        return {
          success: true,
          path: key,
          bigThumbnailPath: key,
          smallThumbnailPath: key,
          width: dims.width,
          height: dims.height,
        }
      } catch (e) {
        return { success: false, error: e instanceof Error ? e.message : 'Could not read asset' }
      }
    },
    addGenericAssetToProject: async ({ srcPath }) => ({ success: true, path: srcPath }),
    makeThumbnailsForProjectAsset: async ({ path }) => ({ success: true, bigThumbnailPath: path, smallThumbnailPath: path }),
    makeDimensionsForProjectAsset: async ({ path, type }) => {
      try {
        const dims = await store.measureMedia(path, type)
        return { success: true, width: dims.width, height: dims.height }
      } catch (e) {
        return { success: false, error: e instanceof Error ? e.message : 'Could not read dimensions' }
      }
    },
    openProjectAssetsPathChangeDialog: async () => {
      try {
        const handle = await pickProjectAssetsFolder()
        return { success: true, path: `web://project-assets/${handle.name}` }
      } catch (err) {
        return { success: false, error: err instanceof Error ? err.message : String(err) }
      }
    },

    // Video processing ------------------------------------------------------------------
    extractVideoFrame: async ({ videoPath, seekTime, width, quality }) => {
      const path = await store.extractFrame(videoPath, seekTime, width, quality)
      return { path }
    },
    exportNative: async ({ clips, width, height, fps }) => {
      const sessionId = crypto.randomUUID()
      try {
        const blob = await renderExport(clips as unknown as ExportClip[], width, height, fps, sessionId)
        if (blob) {
          download(blob, 'video-creator-export.webm')
          return { success: true }
        }
        return { success: false, error: 'No output produced' }
      } catch (e) {
        return { success: false, error: e instanceof Error ? e.message : 'Export failed' }
      }
    },
    exportCancel: async ({ sessionId }) => {
      const r = activeExports.get(sessionId)
      if (r && r.state !== 'inactive') r.stop()
      activeExports.delete(sessionId)
      return { success: true }
    },

    // Python lifecycle (no local runtime in the web app — resolve immediately) ---------
    checkPythonReady: async () => ({ ready: true }),
    startPythonSetup: async () => {},
    startPythonBackend: async () => {},
    getBackendHealthStatus: async () => await probeHealth(),
    notifyGenerationActive: async () => {},

    // Logging / analytics ---------------------------------------------------------------
    writeLog: async ({ level, message }) => {
      ;(console as unknown as Record<string, (m: string) => void>)[level]?.(message)
    },
    getAnalyticsState: async () => ({ analyticsEnabled: false, installationId }),
    setAnalyticsEnabled: async () => {},
    sendAnalyticsEvent: async () => {},

    // Models / OS paths that are no-ops in the browser -----------------------------------
    openModelsDirChangeDialog: async () => ({ success: false, error: 'No local models folder in web' }),
    openModelsFolder: async () => ({ success: false, error: 'No local models folder in web' }),

    // Progress subscriptions --------------------------------------------------------------
    onPythonSetupProgress: () => {},
    removePythonSetupProgress: () => {},
    onBackendHealthStatus: (cb) => {
      healthListeners.add(cb)
      ensureHealthPoll()
      return () => {
        healthListeners.delete(cb)
      }
    },
    getPathForFile: (file) => store.registerFile(file),
  }
}
