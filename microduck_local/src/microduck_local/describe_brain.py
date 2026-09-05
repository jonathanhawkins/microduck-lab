"""Give a trained brain a human name and a short description.

    uv run describe-brain p-n256-s31 --title "Capacity sweep: 256-256" \\
        --description "256-256 MLPs against p-n128 on the same six seeds: +0.006, unresolved."
    uv run describe-brain z1-s81 --show

A run's NAME is an identifier — it is what --init-from, learned:<name> and
select-brain address, and what the directory is called — so it never
changes. What people READ is the title and description this writes into
brains/<name>/brain.json; the /train page shows them in place of the name.
A board of runs called p-batch-s14 and z1 was unreadable a day after it
was made, which is the reason this exists.

Write the FINDING into the description once the experiment resolves ("+0.000
paired, ±0.012 — neutral, shipped on"): the description is where the next
person learns what a run was for and whether it was worth running.

Only `title`, `description` and `group` are touched; the recipe and
`selected` are preserved as they are.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .brain.learned import brains_dir

FIELDS = ("title", "description", "group")

# Use-case groups, in the order the /train page and the /sim brain menu list
# them. A group is the QUESTION a set of runs was made to answer, not the
# knob they turned — lineage (init_from) is recorded too rarely to group by.
GROUPS = {
    "shipped-followers": "Followers (shipped)",
    "trainer-ab": "Trainer defect A/B (seed 7)",
    "early-stop": "Early stop",
    "paired-sweeps": "Paired-seed sweeps (seeds 11-14)",
    "capacity": "Network capacity (seeds 31-36)",
    "null-pair": "Null pair (seeds 81-84)",
    "other": "Other",
}


def describe(run_dir: Path, title: str | None, description: str | None, group: str | None = None) -> dict:
    """Write the given fields (None = leave alone) into run_dir/brain.json and
    return the resulting metadata. Missing brain.json is an error: a run
    with no contract is not a brain, and this must not create one."""
    path = run_dir / "brain.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} — not a trained brain (no brain.json)")
    meta = json.loads(path.read_text())
    if group is not None and group not in GROUPS:
        raise ValueError(f"group must be one of {', '.join(GROUPS)} (got {group!r})")
    for key, value in (("title", title), ("description", description), ("group", group)):
        if value is not None:
            meta[key] = value.strip()
    path.write_text(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="brains/<run> to describe")
    ap.add_argument("--title", default=None, metavar="TEXT", help="the human name shown in place of the run name")
    ap.add_argument("--description", default=None, metavar="TEXT",
                    help="one or two sentences: what it tests, against what, and what it found")
    ap.add_argument("--group", default=None, choices=sorted(GROUPS),
                    help="the use case this run belongs to — how the /train page and the /sim brain menu file it")
    ap.add_argument("--show", action="store_true", help="print the current title and description and exit")
    args = ap.parse_args(argv)

    run_dir = brains_dir() / args.run
    if args.show or (args.title is None and args.description is None and args.group is None):
        meta = json.loads((run_dir / "brain.json").read_text())
        for key in FIELDS:
            print(f"{key}: {meta.get(key) or '—'}")
        if not args.show:
            print("(pass --title and/or --description to set them)", file=sys.stderr)
        return
    meta = describe(run_dir, args.title, args.description, args.group)
    print(f"{args.run}: {meta.get('title') or '—'}")
