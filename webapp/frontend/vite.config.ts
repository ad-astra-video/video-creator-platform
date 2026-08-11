import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// Web-only build for the static Video Creator web app (served by Cloudflare Pages /
// any static host). No Electron plugins — this must produce pure html/css/js in ./dist.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './frontend'),
    },
  },
  base: './', // relative asset paths so dist/ deploys under any path
  build: { outDir: 'dist' },
  server: {
    // SPA fallback so deep links (/project/...) return index.html in dev.
    // API + WebSocket traffic is handled by the Worker; point VITE_API_BASE at it (Phase 2).
    fs: { allow: ['.'] },
  },
})
