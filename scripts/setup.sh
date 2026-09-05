#!/usr/bin/env bash
# One-command workspace setup — a Mac (Apple Silicon or Intel) or Linux.
#
#   git clone <this repo> microduck-workspace && cd microduck-workspace && ./scripts/setup.sh
#
# Clones the two upstream Pollen repos NEXT TO this checkout at the shas CI
# pins (the contract, golden-bit and symmetry tests are measured against
# those exact models and policies), syncs the Python env with uv, installs
# the viewer's npm packages, and runs the quick contract tests. Re-running it
# is safe: it only moves the upstream checkouts to the pinned shas.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
ws="$(dirname "$here")"
RL_SHA="$(sed -n 's/.*git -C microduck_rl checkout \([0-9a-f]\{40\}\).*/\1/p' "$here/AGENTS.md" | head -1)"
MD_SHA="$(sed -n 's/.*git -C microduck checkout \([0-9a-f]\{40\}\).*/\1/p' "$here/AGENTS.md" | head -1)"
[ -n "$RL_SHA" ] && [ -n "$MD_SHA" ] || { echo "could not read the pinned upstream shas from AGENTS.md"; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1 — $2"; exit 1; }; }
need git "https://git-scm.com"
need uv "https://docs.astral.sh/uv/  (brew install uv)"
need npm "https://nodejs.org  (brew install node)"

clone_at() {  # repo dir sha
  if [ ! -d "$ws/$2/.git" ]; then
    echo "→ cloning pollen-robotics/$1 into $ws/$2"
    git clone -q "https://github.com/pollen-robotics/$1" "$ws/$2"
  fi
  if [ "$(git -C "$ws/$2" rev-parse HEAD)" != "$3" ]; then
    git -C "$ws/$2" fetch -q origin
    git -C "$ws/$2" checkout -q "$3"
  fi
  echo "✓ $2 at $(git -C "$ws/$2" rev-parse --short HEAD) (pinned)"
}
clone_at microduck_rl microduck_rl "$RL_SHA"
clone_at microduck microduck "$MD_SHA"

echo "→ uv sync (microduck_local)"
(cd "$here/microduck_local" && uv sync -q)
echo "→ npm install (duck-viewer)"
(cd "$here/duck-viewer" && npm install --silent --no-audit --no-fund)

for b in follow-v1 follow-v2; do
  [ -f "$here/microduck_local/brains/$b/brain.onnx" ] && echo "✓ shipped brain $b" || echo "! brains/$b/brain.onnx missing (git lfs? a partial clone?)"
done

echo "→ contract smoke tests"
(cd "$here/microduck_local" && uv run --with pytest pytest tests/test_env_contract.py tests/test_world.py tests/test_brain.py -q -p no:cacheprovider)

cat <<MSG

Ready. From microduck_local/:
  uv run --with pytest pytest tests/          # the whole suite (~4 min); the golden-bit
                                             # files SKIP until you record this Mac's:
  MICRODUCK_RECORD_GOLDENS=1 uv run --with pytest pytest tests/test_step_perf_parity.py tests/test_bam_perf_parity.py
  uv run duck-lab --world pitch               # then, from duck-viewer/: npm run dev → /sim
  uv run eval-pitch --seeds 4 --seconds 300 --jobs 4
  uv run eval-tidy --seeds 16 --seconds 300 --jobs 4
MSG
