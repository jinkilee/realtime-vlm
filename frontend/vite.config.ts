import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server (frontend container): serves the UI with HMR and proxies the backend container.
const backend = process.env.BACKEND_URL ?? "http://backend:8000";

export default defineConfig({
  plugins: [react()],
  // node_modules lives in the image (read-only for the host UID) -> keep Vite's cache elsewhere
  cacheDir: "/tmp/.vite",
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": backend,
      "/event-files": backend,
      "/ws": { target: backend, ws: true },
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
