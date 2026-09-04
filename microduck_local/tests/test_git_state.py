"""brain.json records the commit a brain was trained at.

The /train page diffs runs by their recipe; without this, two runs trained
from different code with the same flags were indistinguishable after the
fact — and every one of the 49 runs on disk before it landed still is.
"""

import subprocess
from pathlib import Path

from microduck_local.train_brain import git_state

WORKSPACE = Path(__file__).resolve().parents[2]


def test_git_state_matches_the_workspace_checkout():
    want = subprocess.run(["git", "-C", str(WORKSPACE), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True)
    if want.returncode != 0:
        import pytest
        pytest.skip("workspace is not a git checkout")
    got = git_state()
    assert got["git_sha"] == want.stdout.strip()
    assert isinstance(got["git_dirty"], bool)


def test_git_state_is_empty_outside_a_repo(tmp_path):
    """Provenance must never fail a training run: no repo, no keys."""
    assert git_state(tmp_path) == {}
