"""Give a training run a human name, a measured note, and the ★ pick.

    uv run describe-run teach-g1_imitate-g1_front_kick-25ac56-s2 \\
        --title "Front kick (G1)" \\
        --note "8/8 seeds hold 20 s, apex 0.62 m (102% of the clip)" --pick
    uv run describe-run teach-g1_imitate-g1_front_kick-25ac56-s3 \\
        --note "over-shoots to 0.92 m and loses the recovery: 3/8" --no-pick
    uv run describe-run teach-g1_imitate-g1_front_kick-25ac56-s2 --show

The run's NAME never changes — `--init-from` and the docs address it. This
writes `runs/<name>/record.json`, which is what the lab's policy palette
actually shows: `title` on the chip, `note` in its tooltip, and `--pick` on
the ONE stage of a curriculum chain worth using, so the panel's chain button
stops pointing at whichever stage happened to be last.

Write the FINDING into `--note`, in numbers, once an eval has produced one
("8/8 seeds hold 20 s, apex 0.62 m"). The note is what the next person reads
instead of re-running the experiment. `--pick` is a claim that something
measured this stage against its siblings; never set it on a hunch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .run_record import FIELDS, default_record, read_label, read_record, write_record
from .train import RUNS_DIR


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", help="run directory name under runs/")
    ap.add_argument("--title", default=None, help="short human name for the chip")
    ap.add_argument("--description", default=None,
                    help="what it was for, and what it found once it resolves")
    ap.add_argument("--note", default=None,
                    help="one MEASURED sentence, shown in the chip's tooltip")
    ap.add_argument("--group", default=None, help="free-form grouping label")
    pick = ap.add_mutually_exclusive_group()
    pick.add_argument("--pick", dest="pick", action="store_true", default=None,
                      help="this is the stage of its chain worth using — the "
                           "palette's chain button points here")
    pick.add_argument("--no-pick", dest="pick", action="store_false",
                      help="clear the pick flag")
    ap.add_argument("--show", action="store_true", help="print the record and exit")
    ap.add_argument("--backfill", action="store_true",
                    help="derive title and description from the run's own "
                         "run.json (for runs trained before records existed); "
                         "never overwrites a title or description already set")
    args = ap.parse_args(argv)

    run_dir = RUNS_DIR / args.run
    if not run_dir.is_dir():
        print(f"no run directory {run_dir}", file=sys.stderr)
        return 2
    if args.show:
        print(json.dumps(read_record(run_dir), indent=2))
        return 0
    given = {f: getattr(args, f) for f in FIELDS}
    if args.backfill:
        # Only the facts the trainer already wrote, and only where the record
        # is silent: a hand-written title is worth more than a derived one,
        # and this is run over hundreds of old directories at once.
        derived = _derived_record(run_dir)
        if derived is None:
            print(f"{run_dir.name}: nothing to derive a title from "
                  "(no readable run.json or behavior.json)", file=sys.stderr)
            return 2
        have = read_record(run_dir)
        for key in ("title", "description"):
            if not have.get(key) and given.get(key) is None:
                given[key] = derived[key]
    if all(v is None for v in given.values()) and args.pick is None:
        print("nothing to write — pass --title/--description/--note/--group/"
              "--pick, or --show", file=sys.stderr)
        return 2
    # Only ONE stage of a chain may be the pick: setting it here clears it on
    # the siblings, because two picks is the same "which one do I use?"
    # question this file exists to answer.
    if args.pick:
        for sibling in _chain_siblings(run_dir):
            if read_record(sibling).get("pick"):
                write_record(sibling, pick=False)
    meta = write_record(run_dir, pick=args.pick, **given)
    print(json.dumps(meta, indent=2))
    return 0


def _derived_record(run_dir: Path) -> dict[str, str] | None:
    """Title and description from whichever file the trainer wrote —
    `run.json` (train.py) or `behavior.json` (train_behavior) — or None when
    neither says anything. One function for both so `--backfill` cannot take
    a shortcut past the rest of main()'s work, which is how its first draft
    skipped the pick-uniqueness step for trick runs."""
    try:
        run_json = json.loads((run_dir / "run.json").read_text())
    except (OSError, ValueError):
        title = read_label(run_dir).get("title")
        if not title:
            return None
        return {"title": title,
                "description": f"{run_dir.name} — named from its behavior.json."}
    rec = default_record(run_dir.name, run_json)
    return {"title": rec["title"], "description": rec["description"]}


def _chain_siblings(run_dir: Path) -> list[Path]:
    """Every OTHER stage directory of `run_dir`'s curriculum chain."""
    name = run_dir.name
    if "-s" not in name:
        return []
    prefix = name.rsplit("-s", 1)[0]
    if not prefix or not name.rsplit("-s", 1)[1].isdigit():
        return []
    return [d for d in run_dir.parent.iterdir()
            if d.is_dir() and d != run_dir and d.name.startswith(prefix + "-s")
            and d.name[len(prefix) + 2:].isdigit()]


if __name__ == "__main__":
    sys.exit(main())
