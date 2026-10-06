import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev: `npm run dev` proxies /api -> FastAPI on :8000 (override with API_PORT). Production: nginx does the same proxying.
const apiPort = process.env.API_PORT || "8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: { "/api": { target: `http://localhost:${apiPort}`, changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, "") } },
  },
});
