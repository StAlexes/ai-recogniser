import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const CORE_URL = 'http://127.0.0.1:8765';

const proxy = {
  '/api': { target: CORE_URL, changeOrigin: false },
  '/healthz': { target: CORE_URL, changeOrigin: false },
} as const;

export default defineConfig({
  plugins: [react()],
  // Relative asset paths work both from the core's HTTP server and from file://.
  base: './',
  server: {
    port: 5173,
    strictPort: true,
    proxy,
  },
  preview: {
    port: 4173,
    strictPort: true,
    proxy,
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2022',
    sourcemap: false,
    chunkSizeWarningLimit: 1024,
  },
});
