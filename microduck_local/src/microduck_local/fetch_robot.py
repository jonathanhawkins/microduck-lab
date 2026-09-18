"""Download one body's assets.

    uv run fetch-robot g1        # the Unitree G1's MJCF, meshes and walker
    uv run fetch-robot           # list what this install knows

One command per body was one script per body — `fetch-g1` (still an alias, the
README and `scripts/setup.sh` name it). The download lives on the body now
(`Body.fetch`), so the lab's ⤓ button, this CLI and `setup.sh` all reach the
same code, and a third robot adds a registry entry rather than a console
script.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from .robots import registry


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
            print(f"  {body_id:<12} {state:<12} {contract_id:<18}"
                  + (f"({hint})" if hint else "(ships with the checkout)"))
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
