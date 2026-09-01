#!/bin/bash
# Render the ACTIVE teach run's live.onnx under the trainer's OWN env knobs
# (actuator, spawn mix, gates — read from the live trainer process), so what
# you look at is what training is actually practicing right now.
# Usage: watch.sh [--seconds N] [--episodes N] [extra render-rollout args...]
set -u
# pipefail: the render is piped through `tail`, whose exit status would
# otherwise mask a crashed render-rollout and still print the SHEETS line.
set -o pipefail
# Repo-relative: works from any clone location.
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ML=$ROOT/microduck_local
P=$(pgrep -fl 'train_behavio[r]' | grep python | head -1 | awk '{print $1}')
if [ -z "${P:-}" ]; then echo "no trainer running"; exit 1; fi
# Newest run dir that has a live.onnx = the active run.
RUN=""
for d in $(ls -t "$ML/runs"); do
  [ -f "$ML/runs/$d/live.onnx" ] && RUN=$d && break
done
[ -z "$RUN" ] && { echo "no run with live.onnx found"; exit 1; }
# Harvest MICRODUCK_* from the trainer's environment. Values can contain
# SPACES (MICRODUCK_CLIP carries a clip name like "my walk"), so split on the
# next KEY= boundary instead of on whitespace — `tr ' ' '\n'` silently
# truncated such a value and rendered against the wrong clip — and carry the
# results in an array so the quoting survives into render-rollout.
# Seeded with a no-op so the expansions below are never EMPTY: macOS ships
# bash 3.2, where `set -u` + "${EMPTY[@]}" aborts with "unbound variable"
# (a CLI-launched trainer exports no MICRODUCK_* at all, so this is the
# ordinary case, not an edge one).
ENV_ARGS=(--seconds 8)
while IFS= read -r -d '' kv; do
  ENV_ARGS+=(--env "$kv")
done < <(ps eww "$P" | python3 -c '
import re, sys
for m in re.finditer(r"(?:^| )(MICRODUCK_[A-Z0-9_]+)=(.*?)(?= [A-Za-z_][A-Za-z0-9_]*=|$)",
                     sys.stdin.read()):
    sys.stdout.write(f"{m.group(1)}={m.group(2)}\0")
')
OUT=${WATCH_OUT:-/tmp/watch-$RUN}
echo "watching $RUN under: ${ENV_ARGS[*]}"
cd "$ML" && nice -n 10 uv run render-rollout --policy "runs/$RUN/live.onnx" \
  "${ENV_ARGS[@]}" --episodes 2 --out "$OUT" "$@" 2>&1 | tail -3
rc=$?
[ $rc -ne 0 ] && { echo "render-rollout failed (exit $rc) — no sheets written"; exit $rc; }
echo "SHEETS: $OUT/ep0_sheet.png $OUT/ep1_sheet.png"
