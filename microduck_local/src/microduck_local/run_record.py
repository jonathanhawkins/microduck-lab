"""What a training run WAS, in the run's own directory: `record.json`.

    uv run describe-run teach-g1_imitate-g1_front_kick-25ac56-s2 \\
        --title "Front kick (G1)" \\
        --note "8/8 seeds hold 20 s, apex 0.62 m (102% of the clip)" --pick

A run's NAME is an identifier — `--init-from` addresses it, the directory is
called it, the docs quote it — so it never changes. What a PERSON reads is
this record: the lab's policy palette shows `title` on the chip and `note` in
its tooltip, and `pick` marks the one stage of a curriculum chain that
actually measured best.

This exists because of a specific failure. Three G1 imitation chains were
trained through the lab, and the panel showed five rows of
`teach-g1_imitate-g1_front_kick-<hash>-sN` with nothing to tell them apart —
while the chain's own "whole trick" button pointed at the FINAL stage, which
in that chain was the worst one (3/8 seeds surviving, against 8/8 two stages
earlier). The measurements existed; they were in a chat log and a roadmap
file, which is exactly where the person looking at the palette is not. A run
that cannot say what it is gets deleted or, worse, shipped.

`brain.json` is the same idea for the navigation brains (`describe-brain`),
but it carries their observation contract and lives under `brains/` — a walk
or task policy is not one, so this is a separate, smaller file rather than a
pretense of being one.

WRITING IT IS NOT OPTIONAL FOR A TRAINER: `train.py` writes a default record
at the end of every run (what task, which clip, which curriculum rung, what
it warm-started from) so the palette is never blank again. The parts a
machine cannot know — whether the run was any good — are left empty for a
person or an eval to fill in with `describe-run` or `write_measurement`.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

RECORD_NAME = "record.json"

# Everything `describe-run` may set by hand. `measured` is written by an eval
# (write_measurement) rather than typed, so it is not in this list.
FIELDS = ("title", "description", "note", "group")


def record_path(run_dir: Path) -> Path:
    return Path(run_dir) / RECORD_NAME


def read_record(run_dir: Path) -> dict[str, Any]:
    """The run's record, or {} when it has none or the file is unreadable.

    Unreadable is not fatal on purpose: this is a LABEL. A half-written or
    hand-edited record must never stop the palette from listing the policy
    it belongs to (the same reasoning as `load_clips` skipping a bad clip).
    """
    try:
        data = json.loads(record_path(run_dir).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_record(run_dir: Path, **fields: Any) -> dict[str, Any]:
    """Merge `fields` into the run's record and write it atomically.

    A None value leaves that field alone; `write_record(d, note=None)` is a
    no-op, and `write_record(d, note="")` clears it. Atomic because the lab
    polls `discover_policies` on a timer and a half-written file read at the
    wrong instant would blank a row (see the atomic-write rule in AGENTS.md).
    """
    run_dir = Path(run_dir)
    meta = read_record(run_dir)
    for key, value in fields.items():
        if value is not None:
            meta[key] = value
    path = record_path(run_dir)
    fd, tmp = tempfile.mkstemp(dir=str(run_dir), prefix=RECORD_NAME + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(meta, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return meta


def write_measurement(run_dir: Path, tool: str, numbers: dict[str, Any],
                      note: str | None = None, pick: bool | None = None) -> dict[str, Any]:
    """Record what an evaluation measured, and optionally make this the pick.

    `tool` names the script so the number can be reproduced ("eval_imitate.py
    --seeds 8"); a later run of the same tool REPLACES its entry rather than
    appending, because the honest current reading is the last one taken —
    a growing list of stale numbers is how a run ends up quoted at its best
    ever rather than its present.
    """
    meta = read_record(run_dir)
    measured = dict(meta.get("measured") or {})
    measured[tool] = numbers
    return write_record(run_dir, measured=measured, note=note, pick=pick)


def read_label(run_dir: Path) -> dict[str, Any]:
    """What the POLICY PALETTE should show for this run: title, note, pick.

    Two sources, because two trainers write two files. `record.json` is this
    module's and wins. A duck trick run has no run.json at all — it carries
    `behavior.json`, which has had `title`/`description` fields since the
    teach panel existed but was only ever read by the /train page, so 647
    trick runs sat in the palette as bare directory names while their titles
    were on disk the whole time. Reading both here is what makes the palette
    show everything that has ever been named.
    """
    rec = read_record(run_dir)
    out = {k: rec[k] for k in ("title", "note", "pick") if rec.get(k)}
    if out.get("title"):
        return out
    try:
        beh = json.loads((Path(run_dir) / "behavior.json").read_text())
    except (OSError, ValueError):
        return out
    if not isinstance(beh, dict):
        return out
    title = beh.get("title")
    if not title and beh.get("behavior"):
        clip = beh.get("clip")
        title = f"{beh['behavior']}" + (f" · “{clip}”" if clip else "")
    if title:
        out["title"] = str(title)
    return out


def _contract_id(run_json: dict[str, Any], robot: str) -> str | None:
    """The policy contract this run speaks, by id — or None.

    `run.json`'s own `"contract"` first, because the trainer wrote it from
    the body a moment earlier and reading it costs nothing. The registry is
    the fallback, for a record written from an OLD run.json
    (`describe-run --backfill`) that records only a robot name.

    Never raises, and never guesses. A body whose assets are absent cannot
    be loaded (`registry.get` raises with the fetch command) — and a LABEL
    must not be able to take down the run it labels, which is this module's
    rule from its first line (`read_record` swallowing a corrupt file for
    the same reason). A record with no `contract_id` reads as "not
    recorded", which is exactly what every run before this key says too.
    """
    declared = run_json.get("contract")
    if isinstance(declared, dict) and declared.get("id"):
        return str(declared["id"])
    try:
        from .robots import registry
        return str(registry.get(robot).contract().id)
    except Exception:
        return None


def default_record(run_name: str, run_json: dict[str, Any],
                   behavior_title: str | None = None,
                   stage_env: dict[str, str] | None = None) -> dict[str, Any]:
    """The record a TRAINER writes with no human involved.

    Everything here is a fact the trainer already has — never a judgement
    about how the run went. The title is what the lab called the job, which
    is what the person watching it saw; the description says what was
    trained, on which body, against which clip, from which parent, and at
    which curriculum rung, because those are the four things that make one
    hash in a list of hashes different from the next one.
    """
    robot = str(run_json.get("robot") or "microduck")
    task = str(run_json.get("task") or "walk")
    contract_id = _contract_id(run_json, robot)
    kw = run_json.get("env_kwargs") or {}
    clip = kw.get("clip_name") or run_json.get("clip")
    title = behavior_title or (f"{task} ({robot})" if robot != "microduck" else task)
    # The lab's own job title already names the clip ("Perform “x”"), so
    # appending it there produced `Perform “x” · “x”`.
    if clip and str(clip) not in title:
        title = f"{title} · “{clip}”"
    bits = [f"{task} on the {robot}"]
    if clip:
        bits.append(f"tracking the clip “{clip}”")
    init = run_json.get("init_from")
    if init:
        bits.append(f"warm-started from {Path(str(init)).name}")
    # The clip is reported on its own line above; it is not a curriculum rung.
    rungs = {k: v for k, v in (stage_env or {}).items()
             if k.startswith("MICRODUCK_") and k != "MICRODUCK_CLIP"}
    if rungs:
        bits.append("curriculum rung " + ", ".join(f"{k}={v}" for k, v in sorted(rungs.items())))
    bits.append(f"{int(run_json.get('steps') or 0):,} steps")
    return {
        "title": title,
        "description": "; ".join(bits) + ".",
        "run_name": run_name,
        "robot": robot,
        # The contract ID ONLY, where `run.json` carries the whole record
        # (`robots/policy_contract.py`). Two files, two audiences: `run.json`
        # is what the exporter and the lab read, and this is what a PERSON
        # reads — the id is the part that is legible in a palette tooltip,
        # and a slot table there would be noise. Absent (None) rather than
        # guessed when the body cannot be loaded — see `_contract_id`.
        "contract_id": contract_id,
        "task": task,
        "clip": clip,
        "init_from": Path(str(init)).name if init else None,
        "stage_env": rungs or None,
        # Deliberately absent until something measures it: `note`, `pick`,
        # `measured`. A trainer does not get to say its own run was good.
    }
