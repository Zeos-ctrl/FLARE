import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server proxies API calls to the FastAPI backend (default: flare-api's
// own default port, 8420 — deliberately not 8000/3000/5000, which are common
// defaults for other local dev tools). Override with FLARE_API if you run
// the backend elsewhere.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.FLARE_API || 'http://localhost:8420',
        changeOrigin: true,
      },
    },
  },
})
