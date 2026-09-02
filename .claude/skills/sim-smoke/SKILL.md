---
name: sim-smoke
description: Look at the /sim world page the way a user would — bring up the lab in world mode and the viewer, open the page in headless Chromium, press keys, screenshot it, and READ the screenshot and console. Use after touching world_server / world / sensors / brain or the viewer's sim files, whenever "does the page actually render/stream" is the question, and to capture a picture of a scenario for a PR. Trigger on: "smoke test /sim", "screenshot the sim page", "is the world page working", "show me the room".
---

Two scripts, both from the repo root:

1. **Bring the stack up** (idempotent; `--restart` relaunches the lab, which
   you need after editing any lab Python — the running process holds stale
   code):

   ```
   bash .claude/skills/sim-smoke/bringup.sh living-room --restart
   ```

   Built-in scenarios: `empty-floor`, `wall-test`, `living-room`
   (`GET /scenarios` lists user ones too). Logs: `$TMPDIR/sim-smoke/`.

2. **Screenshot the page** (needs Playwright; on the web runner it is in the
   global node modules and Chromium is at `/opt/pw-browsers/chromium`; on a
   Mac `npm i -g playwright && npx playwright install chromium` and unset
   `CHROMIUM_PATH`):

   ```
   node .claude/skills/sim-smoke/shot.mjs --out /tmp/sim.png --keys "Escape"
   node .claude/skills/sim-smoke/shot.mjs --keys "1" --out /tmp/sim-d0.png   # select duck 1
   ```

   Then **Read the PNG** and check: the `● live` badge (frames flowing),
   walls/boxes/ducks present, ToF dots on the surfaces the ducks face, the
   inspector heatmap painted, RTF ≈ 1.00 in the top bar. The script prints
   the browser console minus dev-server noise — a `pageerror` line is a bug.

Numbers to trust over the picture: `curl -s :8788/world` for per-duck falls
and presets, and `curl -s :8788/replay/ring | tail -c 600` for the last
recorded frames (once record/replay exists).

Gotchas: never `pkill -f duck-lab` from an agent shell (the pattern matches
your own command line and kills your shell; bringup.sh uses `duck-la[b]`);
the container has no EGL, so do not set `MUJOCO_GL=egl` here; headless
software GL is slower than a Mac GPU, so give the page a few seconds.
