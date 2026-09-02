#!/usr/bin/env bash
# Bring up the lab in WORLD mode and the viewer dev server for a smoke test.
# Usage: bash .claude/skills/sim-smoke/bringup.sh [scenario] [--restart]
# Idempotent: leaves a healthy pair alone; --restart kills and relaunches the
# lab (needed after editing viz_server/world_server/brain — the process holds
# stale code). Logs land in $TMPDIR/sim-smoke/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCENARIO="${1:-living-room}"
LOGS="${TMPDIR:-/tmp}/sim-smoke"; mkdir -p "$LOGS"
if [[ "${2:-}" == "--restart" || "${1:-}" == "--restart" ]]; then
  # A bracketed regex so this script's own command line never matches.
  for p in $(pgrep -f 'bin/duck-la[b]' || true); do kill "$p" || true; done
  sleep 1
fi
if ! curl -sf -o /dev/null http://127.0.0.1:8788/world; then
  ( cd "$ROOT/microduck_local" && LAB_STATE_PATH="$LOGS/lab-state.json" \
      nohup uv run duck-lab --fresh --world "$SCENARIO" --port 8788 > "$LOGS/lab.log" 2>&1 & )
  for _ in $(seq 1 40); do curl -sf -o /dev/null http://127.0.0.1:8788/world && break; sleep 0.5; done
fi
curl -sf -o /dev/null http://127.0.0.1:8788/world && echo "lab: up ($(curl -s http://127.0.0.1:8788/world | head -c 60)…)" || { echo "lab failed:"; tail -20 "$LOGS/lab.log"; exit 1; }
if ! curl -sf -o /dev/null http://127.0.0.1:63317/sim; then
  ( cd "$ROOT/duck-viewer" && PORT=63317 nohup npm run dev > "$LOGS/viewer.log" 2>&1 & )
  for _ in $(seq 1 60); do curl -sf -o /dev/null http://127.0.0.1:63317/sim && break; sleep 1; done
fi
curl -sf -o /dev/null http://127.0.0.1:63317/sim && echo "viewer: up" || { echo "viewer failed:"; tail -20 "$LOGS/viewer.log"; exit 1; }
