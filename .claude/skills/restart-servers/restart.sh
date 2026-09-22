#!/bin/bash
# Restart the microduck backend (duck-lab farm on :8788). Cwd-proof, rename-proof.
# Usage: restart.sh [--backend-only] [--fresh] [policy.onnx ...]
#   --backend-only  leave the viewer dev server (:63317) running — a lab-only
#                   code change has no reason to bounce it
#   --fresh         drop the saved roster (passed through to duck-lab)
#
# Only the process LISTENING on :8788 is stopped. This used to `pkill -f
# duck-lab`, which matches every lab on the machine — on 2026-09-17 another
# session had a scratch lab up on :8799 (`--world pitch-3v3 --fresh`) while
# the :8788 lab needed a restart, and the old script would have taken it
# down too. A port names exactly one lab; a process name does not.
set -u
# Repo-relative: works from any clone location.
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ML=$ROOT/microduck_local
LOG=$ML/lab-server.log
PORT=8788

BACKEND_ONLY=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --backend-only) BACKEND_ONLY=1 ;;
    *) ARGS+=("$a") ;;
  esac
done

# GUARD: a farm restart kills any live teach job with it. This has now
# happened twice by accident (2026-08-31, both times a check combined into
# the same command as the restart). The script itself refuses now, on two
# independent readings:
# (1) trainer processes. BOTH trainers: the duck's `train_behavior` and a
#     task run (`python -m microduck_local.train --robot g1 ...`), which the
#     old pattern missed. The bracket keeps the pattern from matching this
#     pgrep's own command line; `grep -i python` keeps monitor/grep shells
#     that merely mention the name in their command text from tripping it.
# (2) the lab's own answer, when it is up: GET /teach/status `running` is
#     what the lab reads off its trainer subprocess (the authoritative one —
#     see the endpoint's docstring on why the process table lies).
if [ "${MICRODUCK_RESTART_FORCE:-0}" != "1" ]; then
  if pgrep -fl 'microduck_local\.train_behavio[r]|microduck_local\.trai[n] ' 2>/dev/null | grep -qi python; then
    echo "REFUSING to restart: a trainer process is running:"
    # pid + argv from `-m` on: the interpreter path alone is 100+ characters.
    pgrep -fl 'microduck_local\.train_behavio[r]|microduck_local\.trai[n] ' 2>/dev/null | grep -i python \
      | awk '{ out = $1 " python"; for (i = 3; i <= NF && i <= 9; i++) out = out " " $i; print out }' | sort -u
    echo "Stop it first:  curl -s -X POST http://127.0.0.1:$PORT/teach/stop"
    echo "or, to kill it deliberately:  MICRODUCK_RESTART_FORCE=1 restart.sh ..."
    exit 1
  fi
  case "$(curl -s -m 2 "http://127.0.0.1:$PORT/teach/status" 2>/dev/null)" in
    *'"running":true'*|*'"running": true'*)
      echo "REFUSING to restart: the lab on :$PORT says a teach job is training."
      echo "Stop it first:  curl -s -X POST http://127.0.0.1:$PORT/teach/stop"
      echo "or, to kill it deliberately:  MICRODUCK_RESTART_FORCE=1 restart.sh ..."
      exit 1 ;;
  esac
fi

# Stop whatever LISTENS on a port — and nothing else. TERM first; a listener
# still holding the port after 10 s gets KILL, by pid, never by name.
stop_port() {
  local port=$1 pids
  pids=$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null)
  if [ -z "$pids" ]; then
    echo "nothing listening on :$port"
    return 0
  fi
  for pid in $pids; do
    # argv without the interpreter path, which is 100+ characters on its own
    echo "stopping pid $pid: $(ps -o command= -p "$pid" 2>/dev/null | awk '{ for (i = 2; i <= NF; i++) printf "%s ", $i }' | cut -c1-90)"
    kill "$pid" 2>/dev/null
  done
  for _ in $(seq 1 20); do
    sleep 0.5
    lsof -nP -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 || return 0
  done
  for pid in $pids; do
    echo "pid $pid still holds :$port after 10 s — kill -9"
    kill -9 "$pid" 2>/dev/null
  done
  sleep 1
}

# Launch a server DETACHED IN ITS OWN SESSION. `nohup … &` is not enough: the
# child stays in this shell's process group, and when the Claude harness
# reaps a tool call it has backgrounded it kills that whole group — on
# 2026-09-10 that took the lab down hours after a bring-up had printed
# "lab: up". macOS has no `setsid`, so the new session comes from Python
# (start_new_session). Usage: detach LOGFILE cmd args…  (env prefixes pass
# through; the log is truncated first, as the old `> LOG` did).
detach() {
  local log=$1; shift
  : > "$log"
  python3 - "$log" "$@" <<'PY'
import subprocess, sys
log, cmd = sys.argv[1], sys.argv[2:]
with open(log, "ab") as fh:
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
print(f"detached pid {p.pid} (its own session)")
PY
}

echo "[1/3] stopping the lab on :$PORT..."
stop_port "$PORT"
if [ "$BACKEND_ONLY" != 1 ]; then
  pkill -f "duck-viewer/node_modules/.bin/next" 2>/dev/null
  pkill -f "next dev -p 63317" 2>/dev/null
  sleep 2
fi

echo "[2/3] starting backend (duck-lab :$PORT)..."
# Entry-point name has churned (duck-farm -> duck-lab); resolve from pyproject.
ENTRY=$(grep -oE '^(duck-[a-z]+) *= *"microduck_local.viz_server:main"' "$ML/pyproject.toml" | cut -d' ' -f1)
ENTRY=${ENTRY:-duck-lab}
# ${ARGS[@]+...}: bash 3.2 (macOS) trips `set -u` on an empty array otherwise.
MICRODUCK_ACTUATOR=bam detach "$LOG" uv run --directory "$ML" "$ENTRY" --port "$PORT" ${ARGS[@]+"${ARGS[@]}"}

for i in $(seq 1 40); do
  sleep 2
  curl -s -m 2 "http://127.0.0.1:$PORT/joints" >/dev/null 2>&1 && { echo "backend UP on :$PORT"; ok=1; break; }
done
if [ "${ok:-0}" != 1 ]; then
  if grep -q "no ducks" "$LOG" 2>/dev/null; then
    echo "empty roster — retrying with the default walker..."
    MICRODUCK_ACTUATOR=bam detach "$LOG" uv run --directory "$ML" "$ENTRY" --port "$PORT" ../microduck/policies/alpha_walking.onnx
    for i in $(seq 1 40); do
      sleep 2
      curl -s -m 2 "http://127.0.0.1:$PORT/joints" >/dev/null 2>&1 && { echo "backend UP on :$PORT"; ok=1; break; }
    done
  fi
fi
if [ "${ok:-0}" != 1 ]; then
  echo "backend FAILED — last log lines:"; tail -8 "$LOG"; exit 1
fi
if [ "$BACKEND_ONLY" = 1 ]; then
  echo "[3/3] --backend-only: viewer (:63317) left as it was"
  exit 0
fi
echo "[3/3] starting viewer (:63317) detached..."
# Detached on purpose: the managed preview (preview_start) ties the dev server
# to the session's tool runtime and it is reaped on a session/model switch.
bash "$(dirname "$0")/viewer.sh" --force
