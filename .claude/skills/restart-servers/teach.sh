#!/bin/bash
# Launch training IN THE VIEWER (farm /teach) — never CLI-headless.
# Usage: teach.sh "spin in place" [--from RUN] [--steps N] [--stage K]
set -u
TEXT=${1:?'usage: teach.sh "<behavior text>" [--from run] [--steps N] [--stage K]'}; shift
FROM=""; STEPS=""; STAGE=""
while [ $# -gt 0 ]; do case $1 in
  --from) FROM=${2:?--from needs a run name}; shift 2;;
  --steps) STEPS=${2:?--steps needs a number}; shift 2;;
  --stage) STAGE=${2:?--stage needs a number}; shift 2;;
  *) echo "unknown arg $1"; exit 1;; esac; done
# Body built by json.dumps, not shell interpolation: a quote or backslash in
# the trick text (teach.sh 'hold a "one leg" stand') produced invalid JSON,
# and the 422 came back as the useless "REFUSED: None".
BODY=$(TEXT="$TEXT" FROM="$FROM" STEPS="$STEPS" STAGE="$STAGE" python3 -c '
import json, os, sys
body = {"text": os.environ["TEXT"]}
if os.environ["FROM"]:
    body["initFrom"] = os.environ["FROM"]
for key, env in (("steps", "STEPS"), ("startStage", "STAGE")):
    raw = os.environ[env]
    if raw:
        try:
            body[key] = int(raw)
        except ValueError:
            sys.exit(f"{env.lower()} must be a whole number, got {raw!r}")
print(json.dumps(body))') || exit 1
OUT=$(curl -s -m 30 -X POST http://127.0.0.1:8788/teach -H 'Content-Type: application/json' -d "$BODY")
if [ -z "$OUT" ]; then
  echo "farm not reachable on :8788 — run restart-servers first"; exit 1
fi
echo "$OUT" | python3 -c "
import json,sys
d=json.load(sys.stdin)
if not d.get('matched'):
    print('REFUSED:', d.get('message')); sys.exit(1)
j=d.get('job',{}); sp=j.get('stage') or {}
print('training in the viewer:', j.get('runName'),
      ('| stage %s/%s %s'%(sp.get('idx'),sp.get('count'),sp.get('label',''))) if sp else '')"
