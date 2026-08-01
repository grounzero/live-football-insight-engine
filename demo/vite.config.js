import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The demo is served by FastAPI in production (`make demo`), so the build is
// emitted into the package's static directory. In development `npm run dev`
// proxies API calls to the running service instead.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../src/football_insights/serving/static',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/ready': 'http://127.0.0.1:8000',
      '/model': 'http://127.0.0.1:8000',
      '/metrics': 'http://127.0.0.1:8000',
      '/insights': 'http://127.0.0.1:8000',
      '/replay': 'http://127.0.0.1:8000',
    },
  },
})
