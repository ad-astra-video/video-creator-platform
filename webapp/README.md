# Video Creator — Web App (static)

The full Video Creator UI (all task panels + the timeline editor) as a **static web app** with no
Electron, no local Python, and no local GPU. The heavy work runs on the Cloudflare Worker + GPU
runners behind a Livepeer orchestrator (see `../plans/20260810_183000-webapp-transition.md`).

```
webapp/
  PLAN.md        <- moved to ../plans/ (gitignored planning notes)
  frontend/      <- the Vite static app (this is what you build and deploy)
  docs/          <- Phase 0 artifacts: electronAPI bridge map, API contract, architecture
```

## Layout

- `frontend/` is the vendored frontend (copied from the read-only `video-creator` desktop repo)
  plus a **web-only** configuration. The desktop repo is never modified; re-sync by copying
  `frontend/frontend/*` and `shared/*` over the vendored copy.
- `frontend/frontend/lib/runtime/web-electron-api.ts` + `web-store.ts` provide a real browser
  implementation of the Electron bridge (file pickers, blob downloads, `window.open`, canvas
  frame extraction, media-probe dimensions, MediaRecorder webm export, health fetch) so the app
  runs in a browser with zero Electron/Python.
- `frontend/frontend/lib/runtime/fs-access.ts` (Phase 3.1) upgrades the asset store to a
  user-selected folder via the File System Access API when it lands.

## Build

```
cd frontend
pnpm install
pnpm typecheck
pnpm build          # -> dist/  (pure HTML/CSS/JS + assets)
```

## Run locally

```
pnpm preview        # serves dist/ at http://localhost:4173
```

or serve `frontend/dist/` with any static server (python -m http.server, lite-server, nginx…).

## Point at your Worker (no rebuild)

Edit `dist/config.js` **after** the build (it ships in every deploy root):

```js
window.__VC_CONFIG__ = {
  apiBase: '',                          // '' = same origin; or https://your-worker.workers.dev
  apiKey: '',                           // optional per-user key (else supplied in-app -> localStorage `vcp_key`)
}
```

`apiBase` is read before the build-time `VITE_API_BASE`. This lets a local static deployment
talk to a remote Worker without touching the bundle.

## Deploy

Upload `frontend/dist/` to Cloudflare Pages (or any static host). If Pages and the Worker are on
the same origin, set `apiBase: ''`. Otherwise set `apiBase` to the Worker URL in `config.js`.

## What works today vs. needs the backend

**Works (static, no backend):** app boot, first-run license flow, Home/Projects navigation,
timeline editor + all task panels render, local file open/save/download, media probing/frame
extraction, webm export.

**Needs the Worker + orchestrator/runners (Phase 2/3):** live generation, restyle/extend/retake/
LoRA over the network, live progress (WebSocket), credits/checkout. These dispatch real jobs from
the browser once the Worker is deployed and an orchestrator is stood up (see plan Decision 5).
