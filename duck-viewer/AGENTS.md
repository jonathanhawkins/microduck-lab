# duck-viewer — agent guide

Next.js + react-three-fiber viewer for Microduck policies. `README.md` is the
authoritative doc — architecture diagram, controls, panel behavior, and the
"Notes for future work" section listing the GPU pitfalls already hit. Read it
before changing rendering or protocol code.

Rules of the road:

- The lab (`microduck_local`'s `duck-lab`) owns all simulation truth; this
  app only renders frames and sends `{"cmd"}/{"reset"}/teach` messages. Don't
  put physics or reward logic here.
- Rendering is deliberately light (merged geoms per body, no shadow maps, DOM
  labels). The first version lost the WebGL context in embedded browsers from
  560 shadow-casting meshes and GPU glyph atlases — watch for
  `THREE.WebGLRenderer: Context Lost` before adding GPU-heavy effects.
- Duck colors come from MJCF material rgba streamed per geom; a few materials
  the CAD export got wrong are overridden by name in `Duck.tsx`
  (`MATERIAL_FIX`).
- Panel states, chat history, and the camera persist in `localStorage`; the
  duck roster persists server-side. Keep both working when you touch state.
- `?lab=host:port` overrides the default lab address (`127.0.0.1:8788`) —
  useful for pointing a dev viewer at a scratch lab.
- Verify with `npm run build` (type checks) and by driving a real lab; the
  HUD's live numbers make protocol regressions obvious.

The block below is maintained automatically by `next dev` — leave it in place.

<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->
