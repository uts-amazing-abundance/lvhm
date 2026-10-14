import { createReadStream, cpSync, existsSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig, type Plugin } from 'vite';

/** The SO-101 model, meshes and calibration live in ../assets/so101; expose them under /so101/. */
function so101Assets(): Plugin {
  const src = path.resolve(__dirname, '../assets/so101');
  const types: Record<string, string> = { '.xml': 'application/xml', '.json': 'application/json', '.stl': 'application/octet-stream' };
  let outDir = 'dist';
  let isBuild = false;
  return {
    name: 'so101-assets',
    configResolved(cfg) { outDir = cfg.build.outDir; isBuild = cfg.command === 'build' && cfg.mode !== 'test'; },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? '').split('?')[0];
        if (!url.startsWith('/so101/')) return next();
        const rel = path.normalize(decodeURIComponent(url.slice('/so101/'.length)));
        const file = path.join(src, rel);
        if (!file.startsWith(src) || !existsSync(file) || !statSync(file).isFile()) return next();
        res.setHeader('Content-Type', types[path.extname(file)] ?? 'application/octet-stream');
        res.setHeader('Cache-Control', 'max-age=3600');
        createReadStream(file).pipe(res);
      });
    },
    closeBundle() {
      if (!isBuild) return; // vitest also drives the Vite pipeline
      cpSync(src, path.resolve(__dirname, outDir, 'so101'), { recursive: true });
    },
  };
}

// The app is a static page. The trace JSON lives in public/data and is fetched at run time,
// so a rebuilt export (python gui/build_dashboard.py's input) only needs copying there.
export default defineConfig({
  base: './',
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  plugins: [react(), tailwindcss(), so101Assets()],
  server: { port: 5173, open: false },
  build: { outDir: 'dist', sourcemap: false },
  // The MuJoCo module resolves its .wasm next to itself; pre-bundling would break that.
  optimizeDeps: { exclude: ['@mujoco/mujoco'] },
});
