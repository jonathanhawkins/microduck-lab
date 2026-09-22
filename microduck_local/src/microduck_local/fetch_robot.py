"""Download one body's assets.

    uv run fetch-robot g1                        # the Unitree G1's MJCF, meshes and walker
    uv run fetch-robot menagerie:unitree_go2     # any of MuJoCo Menagerie's 71 models
    uv run fetch-robot                           # list what this install knows

One command per body was one script per body — `fetch-g1` (still an alias, the
README and `scripts/setup.sh` name it). The download lives on the body now
(`Body.fetch`), so the lab's ⤓ button, this CLI and `setup.sh` all reach the
same code, and a third robot adds a registry entry rather than a console
script.

A NAMESPACED id is the one case that cannot work that way, and it is the whole
point of `docs/mars-roadmap.md` §7: `menagerie:unitree_go2` has no body to ask
for its own download, because the body is READ from the files the download
brings. So an id whose prefix is a discovery namespace
(`robots/registry._NAMESPACES`) routes to that namespace's `fetch(name)`
instead, and the body exists on the far side of it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from .robots import registry


def _fetch_namespaced(body_id: str, prefix: str) -> int:
    """`fetch-robot menagerie:<name>` — download, then describe what arrived.

    Returns a process exit code. The LICENCE is printed by the namespace
    itself (it differs per model and the file travels with the download), and
    what this adds is the body the files turned into: its joints, its measured
    stage pitch and its contract id, which is the proof that nothing per-robot
    was needed to read them.
    """
    name = body_id.split(":", 1)[1]
    if not name:
        print(f"{body_id!r} names no model — try "
              f"`uv run fetch-robot {prefix}:unitree_go2`", file=sys.stderr)
        return 1
    try:
        dest = registry.namespace_module(prefix).fetch(name)
    except (RuntimeError, OSError, ValueError) as e:
        print(f"{body_id}: {e}", file=sys.stderr)
        return 1
    print(f"{body_id} assets at {dest}")
    try:
        body = registry.get(body_id)
    except Exception as e:                    # downloaded but unreadable
        print(f"{body_id} downloaded but does not build: {e}", file=sys.stderr)
        return 1
    print(f"  {body.num_joints} joints, spawn keyframe "
          f"{body.stand_keyframe!r}, stage pitch "
          f"{body.lab_spacing_m:.3f} m ({body.extra.get('measured_width_m')} m "
          f"across), scene {body.extra.get('scene_source')}")
    print(f"  {body.contract().describe()}")
    return 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Not `choices=`: a body this install does not know should be answered by
    # the registry's message (which lists the ids AND the setup command),
    # not by argparse's "invalid choice".
    ap.add_argument("body", nargs="?", default=None,
                    help=f"which body to fetch ({', '.join(registry.ids())})")
    args = ap.parse_args(argv)

    if not args.body:
        print("bodies this install knows:")
        for body_id in registry.ids():
            hint = registry.setup_hint(body_id)
            # The CONTRACT ID beside the state, because "which robot do I
            # have" and "what do its policies speak" are the same question
            # asked twice — and this is the one listing a person runs before
            # handing a .onnx to someone (robots/policy_contract.py).
            contract_id = ""
            try:
                body = registry.get(body_id)
                state = "ready" if body.ready() else "not fetched"
                contract_id = body.contract().id
            except KeyError:
                state = "not fetched"
            except NotImplementedError:
                contract_id = "no contract declared"
            # A trailing space, not only a width: a namespaced contract id is
            # 34 characters (`mjcf-menagerie:trs_so_arm100-18-v0`) and ran
            # straight into the command with none. Budgeting characters is
            # what this repo has already been caught doing — pad, and let a
            # long id push its own row wider.
            print(f"  {body_id:<24} {state:<12} {contract_id:<26} "
                  + (f"({hint})" if hint else "(ships with the checkout)"))
        # A catalogue cannot be listed: 71 models, and naming them here would
        # be the per-robot table this whole seam exists to delete. So the
        # listing shows what is FETCHED and says how to add one.
        for prefix in registry._NAMESPACES:
            print(f"\nand any model from the {prefix} catalogue, read straight "
                  f"from its MJCF:\n  uv run fetch-robot {prefix}:<model> "
                  "(e.g. menagerie:unitree_go2 — the directory name in "
                  "github.com/google-deepmind/mujoco_menagerie)")
        return

    # A namespaced id that is NOT already fetched has no body to ask, so it
    # goes to the catalogue. One that IS fetched goes the normal way, and
    # `Body.fetch()` on it is the no-op that returns its directory.
    prefix = registry.namespace_of(args.body)
    if prefix and args.body not in registry.registry():
        # `sys.exit(0)` would still raise `SystemExit`, and every other
        # success path in this file returns — a caller that imports `main`
        # (the tests, and `scripts/setup.sh`'s python) should not have to
        # catch an exception to learn that a download worked.
        code = _fetch_namespaced(args.body, prefix)
        if code:
            sys.exit(code)
        return

    try:
        body = registry.get(args.body)
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        sys.exit(1)
    try:
        dest = body.fetch()
    except NotImplementedError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except (subprocess.CalledProcessError, RuntimeError, OSError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    print(f"{body.id} assets at {dest}")


if __name__ == "__main__":
    main()
