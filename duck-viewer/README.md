# duck-viewer

Next.js + Three.js (react-three-fiber) web viewer for Microduck policies — watch
many training runs walk side by side in the browser instead of the native MuJoCo
viewer. The pattern is lifted from jenga-stacker's web viewer (mesh geometry
extracted straight from the compiled MuJoCo model, no asset pipeline), upgraded
from recorded replays to live WebSocket streaming.

```
┌──────────────────────────┐  GET /scene (meshes + colors, once)  ┌────────────┐
│ duck-farm (Python)       │ ───────────────────────────────────▶ │ Next.js    │
│ microduck_local          │   WS /ws ~25 Hz body poses + stats   │ duck-viewer│
│ one CPU-MuJoCo env per   │ ◀─────────────────────────────────── │ r3f canvas │
│ policy, real-time 50 Hz  │   {"cmd": [vx,vy,wz]} / {"reset"}    │ + HUD      │
└──────────────────────────┘                                      └────────────┘
```

## Run it

```bash
# 1. the farm (from microduck_local/) — each arg is one duck
uv run duck-farm --checkpoints runs/first-gait ../microduck/policies/alpha_walking.onnx

# 2. the viewer
cd duck-viewer && npm run dev     # then open the printed localhost URL
```

`?farm=host:port` on the page URL points it at a different farm (a scratch
server on another port, a farm on another machine); default `127.0.0.1:8788`.

Duck sources: a run dir (`runs/my-run`, uses/exports `policy.onnx`), any
`.onnx` file (shipped alphas work), or `--checkpoints <run>` to line up one
duck per training checkpoint — watching a policy learn across 500k-step
snapshots is the point of this thing.

**Ducks are driven only by their RL policies** — no teleop. Walking policies
follow the server's auto demo script (their velocity-command input, the same
interface the real robot's gamepad uses); trick policies (`teach-*`, 🎓, 🤝)
get zero commands and just do their trick. The **keyboard flies the camera**,
Maya/Blender-style: drag to orbit, scroll/two-finger-vertical to zoom,
**two-finger horizontal swipe** to slide laterally (natural-scrolling
direction; browser back-swipe is suppressed over the scene, and panels keep
native scrolling), **A/D** slide, **W/S·↑↓** dolly, **←/→** orbit, **Q/E**
rise/fall, **Shift+R** reset view — all held keys move smoothly (velocity × dt).
The one non-camera key is **R**, which **restarts the sim** (`{"reset": true}`
to the farm): every duck's episode drops back to step zero at the same moment,
which is what makes a side-by-side comparison legible. HUD shows per-duck
episode time, fall count, reward-rate EMA (dimmed for trick ducks — it scores
the walking recipe), a system-stats strip (cpu/mem/training steps-per-second),
and collapses to a pill via its — button. (Manual drive commands still exist
at the protocol level — `FarmClient.sendCmd` — for a future gamepad page;
the UI deliberately doesn't expose them.)

## The panels

- **🧠 policies** (top-right): every assignable brain — shipped Pollen
  policies, local runs, checkpoints. Drag a chip onto a duck (or click to arm,
  then click a duck) to hot-swap its brain mid-stride. Auto-refreshes when a
  training run finishes.
- **🎓 teach the duck** (bottom-right): chat a trick ("stand on one leg"), see
  the reward recipe in plain English, watch the live score curve + per-term
  bars while the 🎓 trainee duck improves snapshot by snapshot. When a run
  ends, the recipe's **weight sliders unlock**: drag them and either
  "↻ retrain" (fresh) or "✨ fine-tune" (keep what it learned, adjust) —
  that's the reward-shaping loop with no Python involved.
- **helpers**: ＋ on the training row spawns a 🤝 helper duck — another
  viewer of the same live policy. Helpers do **not** add trainer workers
  (that *lowered* steps/s while the farm was open). ✕ removes it.

Panel states, chat history, and the camera persist in localStorage; the duck
roster itself persists server-side (`microduck_local/farm-state.json`) across
farm restarts.

## Notes for future work

- The scene payload is ~20 MB raw (gzipped over the wire, one-time). If it ever
  matters: quantize verts or move to binary/Draco.
- Poses stream as JSON at 25 Hz (~16 bodies × 7 floats per duck) — binary
  framing is the next lever, far from needed at this scale.
- Rendering stays deliberately light: geoms merged per body (~16 draw calls per
  duck), no shadow maps, DOM labels. The first version (560 shadow-casting
  meshes + drei `Text` GPU glyph atlases) lost the WebGL context in the
  embedded browser — keep an eye on `THREE.WebGLRenderer: Context Lost` if you
  add GPU-heavy effects back.
- Duck colors are the MJCF material rgba streamed per geom, carried through the
  per-body merge as a vertex-color channel (so per-part color costs zero extra
  draw calls). A few materials the OnShape export got wrong vs the printed
  robot (eye ring, soft mouth, shoes) are overridden by name in `Duck.tsx`
  (`MATERIAL_FIX`); against a farm too old to stream colors the viewer falls
  back to one guessed color per body.
- The farm loop is single-threaded Python: 8 ducks × 50 Hz ≈ 5% of one core.
  Dozens of ducks are fine; hundreds would want the envs in a worker pool.
