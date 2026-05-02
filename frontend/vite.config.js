import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
    headers: {
      // unsafe-none lets Firebase Auth + Google's gapi script perform
      // window.close / window.closed checks against the cross-origin sign-in
      // popup without Chrome printing COOP warnings.
      'Cross-Origin-Opener-Policy': 'unsafe-none',
      'Cross-Origin-Embedder-Policy': 'unsafe-none',
    },
  },
})
