// Runtime configuration for the static Video Creator web app.
//
// This file is copied verbatim into the deploy root (dist/config.js) so you can point the
// frontend at your Worker WITHOUT rebuilding. Edit it in your deployed copy and reload.
//
//   apiBase  - the Worker / API origin. '' = same origin (Cloudflare Pages + Worker on one
//              domain). For a local static server talking to a remote Worker, set the full
//              URL, e.g.  https://video-creator-api.your-worker.workers.dev
//   apiKey   - OPTIONAL per-user key. Leave '' to let the user supply one in-app (stored in
//              localStorage as `vcp_key`).
window.__VC_CONFIG__ = {
  apiBase: '',
  apiKey: '',
}
