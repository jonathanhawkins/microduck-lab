#!/bin/bash
# Restart the microduck backend (duck-lab farm on :8788). Cwd-proof, rename-proof.
# Usage: restart.sh [--fresh] [policy.onnx ...]   (--fresh drops the saved roster)
set -u
# Repo-relative: works from any clone location.
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ML=$ROOT/microduck_local
LOG=$ML/lab-server.log

# GUARD: a farm restart kills any live teach job with it. This has now
# happened twice by accident (2026-08-31, both times a check combined into
# the same command as the restart). The script itself refuses now:
# (match python trainer processes only — monitor/grep shells that merely
# mention the name in their command text must not trip the guard)
if pgrep -fl 'train_behavio[r]' 2>/dev/null | grep -q python && [ "${MICRODUCK_RESTART_FORCE:-0}" != "1" ]; then
  echo "REFUSING to restart: a teach job is training (train_behavior running)."
  echo "Stop it first:  curl -s -X POST http://127.0.0.1:8788/teach/stop"
  echo "or, to kill it deliberately:  MICRODUCK_RESTART_FORCE=1 restart.sh ..."
  exit 1
fi

echo "[1/3] stopping old servers..."
pkill -f "duck-lab" 2>/dev/null; pkill -f "duck-farm" 2>/dev/null
pkill -f "viz_server import main" 2>/dev/null
pkill -f "duck-viewer/node_modules/.bin/next" 2>/dev/null
pkill -f "next dev -p 63317" 2>/dev/null
sleep 2

echo "[2/3] starting backend (duck-lab :8788)..."
# Entry-point name has churned (duck-farm -> duck-lab); resolve from pyproject.
ENTRY=$(grep -oE '^(duck-[a-z]+) *= *"microduck_local.viz_server:main"' "$ML/pyproject.toml" | cut -d' ' -f1)
ENTRY=${ENTRY:-duck-lab}
MICRODUCK_ACTUATOR=bam nohup uv run --directory "$ML" "$ENTRY" --port 8788 "$@" > "$LOG" 2>&1 &

for i in $(seq 1 40); do
  sleep 2
  curl -s -m 2 http://127.0.0.1:8788/joints >/dev/null 2>&1 && { echo "backend UP on :8788"; ok=1; break; }
done
if [ "${ok:-0}" != 1 ]; then
  if grep -q "no ducks" "$LOG" 2>/dev/null; then
    echo "empty roster — retrying with the default walker..."
    MICRODUCK_ACTUATOR=bam nohup uv run --directory "$ML" "$ENTRY" --port 8788       ../microduck/policies/alpha_walking.onnx > "$LOG" 2>&1 &
    for i in $(seq 1 40); do
      sleep 2
      curl -s -m 2 http://127.0.0.1:8788/joints >/dev/null 2>&1 && { echo "backend UP on :8788"; ok=1; break; }
    done
  fi
fi
if [ "${ok:-0}" != 1 ]; then
  echo "backend FAILED — last log lines:"; tail -8 "$LOG"; exit 1
fi
echo "[3/3] viewer port 63317 freed — now start it with preview_start (name: duck-viewer)."
