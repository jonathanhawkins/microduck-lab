"""Where the bit-exact rollout goldens live, and for which machine.

The parity tests (test_step_perf_parity.py, test_bam_perf_parity.py) pin
rollouts to the last bit of float64. That is a property of ONE machine's
MuJoCo build, libm and numba code paths, so goldens are stored per
platform in tests/goldens/<name>-<system>-<machine>.json and a platform
without a recording SKIPS (with the command to make one) rather than
failing on numbers it never produced. Record on the platform in question:

    MICRODUCK_RECORD_GOLDENS=1 uv run --with pytest pytest tests/test_step_perf_parity.py tests/test_bam_perf_parity.py

The file records the upstream model sha and the library versions it was
made against; a mismatch on either is reported first, because a golden
that moved with a model re-export (2026-09: the CAD re-export moved every
per-term sum in the 5th digit) is a recapture, not a regression.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from microduck_local import contract as C

GOLDENS = Path(__file__).parent / "goldens"
RECORD = os.environ.get("MICRODUCK_RECORD_GOLDENS", "") not in ("", "0")


def platform_key() -> str:
    return f"{platform.system()}-{platform.machine()}".lower()


def provenance() -> dict:
    sha = "unknown"
    try:
        sha = subprocess.run(["git", "-C", str(C.MICRODUCK_RL_DIR), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        pass
    return {"microduck_rl": sha, "mujoco": mujoco.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "platform": platform.platform()}


def path(name: str) -> Path:
    return GOLDENS / f"{name}-{platform_key()}.json"


def load(name: str) -> dict | None:
    p = path(name)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save(name: str, data: dict) -> Path:
    p = path(name)
    GOLDENS.mkdir(exist_ok=True)
    p.write_text(json.dumps({"provenance": provenance(), "data": data}, indent=1, sort_keys=True) + "\n")
    return p


def skip_reason(name: str) -> str:
    return (f"no golden for this platform ({path(name).name}); record one here with "
            f"MICRODUCK_RECORD_GOLDENS=1 pytest tests/test_{name}.py")


def check_provenance(golden: dict) -> str:
    """'' if the golden was made against what is installed now, else what differs."""
    want, have = golden.get("provenance", {}), provenance()
    diff = [f"{k}: golden {want.get(k)} / now {have[k]}" for k in ("microduck_rl", "mujoco", "numpy")
            if want.get(k) != have[k]]
    return "; ".join(diff)
