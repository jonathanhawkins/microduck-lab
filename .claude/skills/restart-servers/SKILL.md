---
name: restart-servers
description: Restart the microduck dev stack — the duck-lab backend (farm, :8788) and the duck-viewer dev server (:63317). Use when the viewer shows "offline", after editing viz_server/behaviors (the farm holds stale code), or whenever the user asks to restart the dev/backend servers.
---

Restart both microduck servers, in this order:

1. **WARNING — training dies with the farm.** A farm restart kills any live
   teach job. Its checkpoint survives; resume after with
   `POST /teach {"text": ..., "initFrom": "<run>"}`. The script refuses on
   its own when a trainer is up — a `train_behavior` (duck) or a
   `microduck_local.train` (G1 task) python process, or the lab's own
   `GET /teach/status` saying `"running": true` — but check first anyway,
   **as a separate command**, never combined with the restart:

   ```
   curl -s -m 2 http://127.0.0.1:8788/teach/status
   ```

   Only `MICRODUCK_RESTART_FORCE=1` overrides the refusal.

2. Run the script (stops the lab, restarts it cwd-proof and
   entry-point-rename-proof, verifies :8788 health, then bounces the viewer):

   ```
   bash .claude/skills/restart-servers/restart.sh
   ```

   After a lab-only code change (viz_server, behaviors, world, brain…) leave
   the viewer dev server alone — it hot-reloads on its own:

   ```
   bash .claude/skills/restart-servers/restart.sh --backend-only
   ```

   Pass `--fresh` plus `.onnx` paths to reseed the duck roster; with no args
   the saved roster (lab-state.json) is kept.

   **Only the process LISTENING on :8788 is stopped** (`lsof … -sTCP:LISTEN`,
   TERM, then KILL by pid after 10 s). It used to `pkill -f duck-lab`, which
   matched every lab on the machine: on 2026-09-17 another session had a
   scratch lab on :8799 (`--world pitch-3v3`) that the old script would have
   killed along with the real one. A scratch lab on another port — with its
   own `LAB_STATE_PATH=…` so it never rewrites the real `lab-state.json` —
   survives a restart of :8788 now.

3. The script now starts the viewer too, detached, by calling `viewer.sh`.
   Nothing more to do. To (re)start only the viewer:

   ```
   bash .claude/skills/restart-servers/viewer.sh          # no-op if already up
   bash .claude/skills/restart-servers/viewer.sh --force  # restart regardless
   ```

   **Do NOT use `preview_start` for the viewer.** The managed preview makes the
   dev server a child of the session's tool runtime, so it is reaped on a
   session or model switch — that killed it on 2026-09-02 and looked like a
   crash. `viewer.sh` reparents it to launchd (PPID 1), like the backend, so it
   survives. Open the viewer with `navigate` to a normal browser tab instead.

   **Use `localhost`, not `127.0.0.1`, for the viewer.** Next 16 serves the
   HTML to either host but 403s its own `/_next` dev resources when the host is
   not in `allowedDevOrigins`, so `127.0.0.1:63317` renders a blank page with
   an empty `<body>` and console 403s — it looks like the app is broken when
   only the origin is wrong. `http://localhost:63317/sim` renders fine. (The
   backend is the opposite: curl it at 127.0.0.1:8788, and note `/` is a 404
   there because it has no root route — use `/joints` for a health check.)

4. Confirm frames flow: one WS read from ws://127.0.0.1:8788/ws should return
   a frame with a non-empty `ducks` list. Tell the user to refresh the page.

Logs: backend `microduck_local/lab-server.log`, viewer
`duck-viewer/viewer-server.log` (both gitignored).

Known gotchas these scripts already handle: the shell cwd resets to the parent
repo (breaks bare `uv run`); the backend entry point was renamed
(duck-farm → duck-lab) and may churn again; a stale Next lock file can block
viewer spawns (`viewer.sh` kills the port holder with `lsof` first); a managed
preview would not survive a session restart (hence the detached launch). And — since 2026-09-10 — **each
server starts in its own session** (`detach` in both scripts: Python's
`start_new_session`, because macOS has no `setsid`). `nohup … &` alone left
the server in the tool call's process group, and when the Claude harness
reaped a tool call it had backgrounded, the lab went down with it hours after
"backend UP". The `sim-smoke` bring-up delegates every launch here for that
reason; from an agent shell, never start these servers any other way.
