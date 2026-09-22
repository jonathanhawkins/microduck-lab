"""Download the Unitree G1 MJCF, meshes and walker ONNX into `.cache/unitree_g1`.

    uv run fetch-g1
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .robots.g1 import CACHE_DIR, G1_REPO, g1_ready


def fetch(dest: Path | None = None) -> Path:
    dest = dest or CACHE_DIR
    if g1_ready(dest):
        print(f"G1 already at {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / (dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    print(f"cloning {G1_REPO} → {dest} (~140 MB meshes)…")
    subprocess.run(
        ["git", "clone", "--depth", "1", G1_REPO, str(tmp)],
        check=True)
    git = tmp / ".git"
    if git.exists():
        shutil.rmtree(git)
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    if not g1_ready(dest):
        raise RuntimeError(f"clone finished but {dest} is missing g1.xml / walker.onnx / assets")
    print(f"G1 ready at {dest}")
    return dest


def main() -> None:
    try:
        fetch()
    except (subprocess.CalledProcessError, RuntimeError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
