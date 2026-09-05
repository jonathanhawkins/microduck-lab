#!/bin/bash
# Start the duck-viewer dev server (:63317) DETACHED, so it outlives the
# Claude Code session that started it.
#
# Why this exists: the managed preview (preview_start) makes the dev server a
# child of the session's tool runtime, so it is reaped on a session or model
# switch — that is what killed it on 2026-09-02. nohup + a background job here
# reparents it away from the session, the same trick restart.sh uses for the
# backend, and it then survives.
#
# Usage: viewer.sh [--force]   (--force restarts even if one is already up)
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DV=$ROOT/duck-viewer
LOG=$DV/viewer-server.log
PORT=63317

up() { curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" 2>/dev/null; }

if [ "${1:-}" != "--force" ] && [ "$(up)" = "200" ]; then
  echo "viewer already UP on :$PORT (use --force to restart)"; exit 0
fi

echo "[1/2] stopping any old viewer on :$PORT..."
pkill -f "duck-viewer/node_modules/.bin/next" 2>/dev/null
pkill -f "next dev -p $PORT" 2>/dev/null
# Anything still holding the port (a stale Next lock leaves one behind).
lsof -ti :$PORT 2>/dev/null | xargs -r kill 2>/dev/null
sleep 2

echo "[2/2] starting viewer detached (:$PORT)..."
cd "$DV" || { echo "no duck-viewer at $DV"; exit 1; }
PORT=$PORT nohup npm run dev > "$LOG" 2>&1 &
disown 2>/dev/null

for i in $(seq 1 40); do
  sleep 2
  if [ "$(up)" = "200" ]; then
    echo "viewer UP on :$PORT (pid $(lsof -ti :$PORT 2>/dev/null | head -1), log: $LOG)"
    exit 0
  fi
done

echo "viewer FAILED — last log lines:"; tail -15 "$LOG"; exit 1
