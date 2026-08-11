# Electron bridge → web replacement map

Source: `video-creator/frontend` (read-only reference). Every `window.electronAPI.*` call site the
frontend makes, grouped by capability, with its web replacement and disposition. This is the input
for the `RuntimeBridge` interface (webapp `frontend/lib/runtime/bridge.ts`, Phase 1) and for what
Phase 5 deletes.

Count: **52 invocations across 48 files**: lines (files) — counted from a full grep of
`window\.electronAPI\.` in `frontend/**/*.{ts,tsx}`. Dispositions: `keep-web` (reimplement with a
browser API), `keep-remote` (a Worker/runner endpoint), `drop` (Electron-only local-runtime, no web
equivalent).

---

## A. File dialogs & OS filesystem → browser file APIs

| Method | Call sites | Web replacement | Disp |
|---|---|---|---|
| `showOpenFileDialog` | `VideoPreviewPanel`, `ImportTimelineModal`, `ICLoraPanel` (×2) | `<input type="file" multiple>` / drag-drop / `showOpenFilePicker` | keep-web |
| `showOpenDirectoryDialog` | `ImportTimelineModal` | `<input webkitdirectory>` / `showDirectoryPicker` | keep-web |
| `showSaveDialog` | `useTimelineXmlExport`, `useSubtitleImportExport` | `showSaveFilePicker` / blob download | keep-web |
| `saveFile` | `VideoEditorTimelineEditingPanel`, `useTimelineXmlExport` | blob download (`URL.createObjectURL`) / `showSaveFilePicker` | keep-web |
| `searchDirectoryForFiles` | `ImportTimelineModal` (×2) | enumerate chosen `DirectoryHandle` (fs-access.ts) | keep-web(rework) |
| `checkFilesExist` | `ImportTimelineModal` (×2) | `handle.getFileHandle(name)` check | keep-web(rework) |
| `addVisualAssetToProject` | `lib/asset-copy.ts` | write into user-selected folder via fs-access.ts | keep-web(rework) |
| `addGenericAssetToProject` | `lib/asset-copy.ts` | same | keep-web(rework) |
| `extractVideoFrame` | `VideoEditor`, `VideoEditorTimelineEditingPanel` (×3), `useRegeneration` | Worker/runner endpoint (asset → frame) — no local ffmpeg | keep-remote |

## B. Python/runtime lifecycle → DROP (no local runtime)

| Method | Call sites | Disposition |
|---|---|---|
| `startPythonSetup` | `PythonSetup.tsx` | drop |
| `removePythonSetupProgress` | `PythonSetup.tsx` | drop |
| `onPythonSetupProgress` | `PythonSetup.tsx` | drop |
| `getResourcePath` | `PythonSetup.tsx`, `FirstRunSetup.tsx` | drop (or static asset URL) |
| `startPythonBackend` | `App.tsx` | drop |
| `checkPythonReady` | `App.tsx` | drop |
| `checkFirstRun` | `App.tsx` | replace with web "connect account" flow |
| `completeSetup` | `App.tsx` | replace with web provision/accept (keep-web) |
| `acceptLicense` | `App.tsx` | web EULA modal (keep-web) |
| `getBackend` | `lib/backend.ts` | web-bridge returns Worker base URL (keep-web) |
| `getBackendHealthStatus` | `use-backend.ts`, `AppSettingsContext.tsx` | poll `GET {worker}/health` (keep-remote) |
| `onBackendHealthStatus` | `use-backend.ts`, `AppSettingsContext.tsx` | subscribe to health poll (keep-remote) |
| `notifyGenerationActive` | `lib/generation-active.ts` (×2) | in-app toast (keep-web) |

## C. Paths / app info / logs / licenses → serve from origin or Worker

| Method | Call sites | Web replacement | Disp |
|---|---|---|---|
| `getProjectAssetsPath` / `openProjectAssetsPathChangeDialog` | `SettingsModal` | chosen-folder handle (fs-access.ts) | keep-web(rework) |
| `getModelsPath` / `openModelsDirChangeDialog` / `openModelsFolder` | `VideoEditorTimelineEditingPanel`, `settings/BaseModelSection` | remote model catalog via Worker (no local models dir) | drop/rework |
| `getLogs` / `openLogFolder` | `LogViewer` | Worker logs endpoint / in-app log view | keep-remote(rework) |
| `getAppInfo` | `SettingsModal` | static app info from origin | keep-web |
| `getNoticesText` | `SettingsModal` | served from origin (Web App Manifest/NOTICES) | keep-web |
| `fetchLicenseText` | `SettingsModal`, `FirstRunSetup` | served from origin (LICENSE) | keep-web |

## D. External links & HF auth → browser

| Method | Call sites | Web replacement | Disp |
|---|---|---|---|
| `openHuggingFaceRepo` | `SettingsModal`, `LoraLibraryModal`, `LoraInfoPopover` | `window.open(url,'_blank','noopener')` | keep-web |
| `openHuggingFaceAuth` | `hooks/use-hf-auth.ts` | redirect/popup to Worker `/api/auth/huggingface/*` OAuth callback | keep-remote |
| `openLtxApiKeyPage` | `App.tsx`, `LtxApiKeyInput` | `window.open(...)` | keep-web |
| `openFalApiKeyPage` | `App.tsx`, `SettingsModal` | `window.open(...)` | keep-web |
| `openExternalUrl` | `settings/CreditsPanel`, `library/LibraryItemCard` | `window.open(...)` | keep-web |

---

## Summary of what the bridge needs to expose in the browser

Only three capabilities actually need V8/browser-native pieces:

1. **Files**: open/save via FS Access API + `<input>` (drop the API gateway for these).
2. **Frame extraction**: `extractVideoFrame` moves to a Worker/runner endpoint.
3. **Health/provision**: `getBackend`, health status, provision → Worker REST. Everything else in A–D
   is either dropped (local runtime) or trivially a `window.open` / static file.

Python/runtime methods (B) have **no** web equivalent and are deleted at Phase 5. The `RuntimeBridge`
interface (Phase 1) therefore only needs: `getBackend`, `onHealth`/`getHealth`, `notifyGenerationActive`,
plus the file-open/save/import surface — everything else either never reaches the bridge or is dropped.
