---
name: restart-servers
description: Restart the microduck dev stack — the duck-lab backend (farm, :8788) and the duck-viewer dev server (:63317). Use when the viewer shows "offline", after editing viz_server/behaviors (the farm holds stale code), or whenever the user asks to restart the dev/backend servers.
---

Restart both microduck servers, in this order:

1. **WARNING — training dies with the farm.** If a teach job is running
   (check `curl -s http://127.0.0.1:8788/teach/stop` is NOT needed — just ask
   or look at the viewer), a farm restart kills it. Its checkpoint survives;
   resume after with `POST /teach {"text": ..., "initFrom": "<run>"}`.

2. Run the script (kills both servers, restarts the backend cwd-proof and
   entry-point-rename-proof, verifies :8788 health):

   ```
   bash .claude/skills/restart-servers/restart.sh
   ```

   Pass `--fresh` plus `.onnx` paths to reseed the duck roster; with no args
   the saved roster (lab-state.json) is kept.

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
preview would not survive a session restart (hence the detached launch).
