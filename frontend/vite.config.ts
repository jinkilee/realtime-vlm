import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the backend (app on :8000) so the UI can be developed with hot reload.
const backend = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": backend,
      "/event-files": backend,
      "/ws": { target: backend, ws: true },
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
