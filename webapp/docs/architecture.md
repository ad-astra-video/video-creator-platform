# Architecture lock — fully serverless static web app

Decision record for the Electron → static web app transition. Locked by the user 2026-08-10; matches
the Decisions table in `plans/20260810_183000-webapp-transition.md`.

## Topology

```
Browser (static frontend — Cloudflare Pages)
   │  HTTPS + per-user API key (REST/JSON)  +  WebSocket for live job progress
   ▼
video-creator-platform (Cloudflare Worker — the only edge; no Python server between)
   ├─ reimplemented API + dispatch layer   (TypeScript; mirrors webapp/docs/api-contract.md)
   ├─ auth: per-user API key (/provision, /recover)
   ├─ credits & checkout (Stripe → PymtHouse → D1 ledger), decrement before dispatch
   ├─ orchestrator client: discover runners + submit jobs
   └─ WS relay: browser <-> Worker <-> orchestrator
        │
        ▼
Livepeer orchestrator (stood up AFTER implementation, for e2e)
        └─► GPU runners (idv2v, ltx-worker, …) — model weights + execution stay here

Project/media assets: LOCAL, user-selected folder via File System Access API (no upload).
```

## Locked decisions

1. **No Python anywhere (Decision 1 → b).** The existing `video-creator/backend` is read-only
   reference only — never deployed. The Cloudflare Worker reimplements the API + dispatch layer in
   TypeScript. **There is no server between the browser and the orchestrator except Cloudflare edge.**
2. **Static hosting → Cloudflare Pages.** Frontend build (`vite build`) deploys to Pages; the Worker
   serves the API + WebSocket on the same origin.
3. **Assets → local.** User media lives in a user-selected folder (File System Access API);
   `webapp/frontend/lib/runtime/fs-access.ts` wraps it. D1 stores only project metadata/paths.
   OPFS + folder-export fallback on Firefox/Safari (Chromium-only caveat).
4. **Progress → WebSocket** between the browser and the orchestrator (Worker-relayed). The current
   REST-poll progress endpoints (`/api/generation/progress`, `/download/progress`) become fallback.
5. **Orchestrator stood up after implementation** for the end-to-end test (Phase 5.4). Until then,
   dispatch + WS are built behind testable seams and validated against mocks / a scripted WS sequence.
6. **Scope cut:** park any task that can't reach a remote runner cleanly; don't silently drop it.
7. **`video-creator` is READ-ONLY.** Never modify it. Frontend is vendored into `video-creator-platform/webapp/`.

## Component responsibilities

- **Browser (webapp/):** all UI, the timeline editor, local-folder asset store, WS client, auth key
  storage (localStorage/IndexedDB), per-user REST calls.
- **Cloudflare Pages:** serves the static frontend build + SPA fallback.
- **Cloudflare Worker (video-creator-platform/src):** reimplemented API routes (dispatch + catalog +
  settings + HF OAuth + platform), per-user auth, credit ledger, D1 job/project-metadata records,
  orchestrator client (runner discovery + job submission), WS relay.
- **Livepeer orchestrator:** runner discovery, job routing, payment/voucher sign-off; stood up post-impl.
- **GPU runners (idv2v, ltx-worker, …):** model weights + actual inference/task execution.

## Non-goals / out of scope

- No GPU execution on the edge, no model weights in the Worker/Pages.
- No local media upload to the platform (assets stay on the user's machine).
- No desktop/installer, no local python, no local CUDA.

## Data model (D1) — additions

- `jobs`: id, user_id, type (t2v/restyle/image/extend/...), status, runner, timestamps, WS channel.
- `projects`: id, user_id, name, asset path manifest (metadata only — no blobs).
- (existing `accounts`, `api_keys`, `codes`, `idempotency`, `payments` unchanged.)

## Verification gates

- Phase 1.7: frontend runs as static files in a browser, zero `window.electronAPI` at runtime.
- Phase 2.x: every route in `api-contract.md` implemented + `curl`-checked; `wrangler dev` WS accepts a
  key and relays a scripted progress sequence; `pnpm smoke` covers dispatch → job row → credit decrement.
- Phase 5.4: clean-browser e2e with the orchestrator stood up — all task types + timeline editor.
