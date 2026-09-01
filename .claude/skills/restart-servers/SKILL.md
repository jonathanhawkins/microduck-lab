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

3. Start the viewer with the managed preview: `preview_start` with name
   `duck-viewer` (launch.json pins port 63317). Verify with a curl to
   http://127.0.0.1:63317/ (expect 200).

4. Confirm frames flow: one WS read from ws://127.0.0.1:8788/ws should return
   a frame with a non-empty `ducks` list. Tell the user to refresh the page.

Known gotchas this script already handles: the shell cwd resets to the parent
repo (breaks bare `uv run`); the backend entry point was renamed
(duck-farm → duck-lab) and may churn again; a stale Next lock file can block
viewer spawns (the kill + managed preview_start path avoids it).
