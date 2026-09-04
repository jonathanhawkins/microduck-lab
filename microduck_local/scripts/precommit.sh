#!/usr/bin/env bash
# The check to run BEFORE every commit. Under a second, and it catches the
# two ways this repo has actually been broken:
#
#   1. A syntax error committed without running anything. Twice in one
#      session: an unterminated docstring in `brain/tracker.py` broke every
#      import and killed three running batteries, and nobody noticed until a
#      battery's log showed the traceback an hour later.
#   2. An import orphaned by an edit — `math` and `contract` left behind in
#      `eval_pitch.py` after the code using them moved to `world/metrics.py`.
#
# The full suite is ~12 minutes, which is why it gets skipped and why these
# got through. This is not a substitute for it: run `pytest tests/` before
# pushing anything that changes behaviour. This is the gate that makes
# "I'll just commit this doc tweak" safe.
#
#   ./scripts/precommit.sh
set -euo pipefail
cd "$(dirname "$0")/.."

uv run ruff check src/ tests/

# ruff parses; it does not EXECUTE. A module that parses can still fail at
# import (a bad relative import, a missing name in an `from x import y`), so
# import the entry points that every battery goes through.
uv run python -c "
import microduck_local.eval_pitch, microduck_local.eval_tidy
import microduck_local.eval_striker, microduck_local.walker_facts
import microduck_local.trace_tidy, microduck_local.train_brain
import microduck_local.world_server, microduck_local.viz_server
print('imports ok')"

echo "precommit ok"
