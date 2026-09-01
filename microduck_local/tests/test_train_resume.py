"""train-behavior warm restarts: the SB3 counter semantic the CLI compensates
for, and a real two-stage subprocess run proving --init-from continues the same
run with --steps as an ABSOLUTE target. The subprocess stage is the slowest
test in the suite (~1-2 min of tiny PPO at --envs 2)."""

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def test_sb3_continue_is_additive_not_absolute():
    """Pin the fact train_behavior compensates for: learn(total_timesteps=T,
    reset_num_timesteps=False) runs T MORE steps (_setup_learn adds the current
    counter back in). If an SB3 upgrade changes this, the --init-from
    remaining-budget math must change with it."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from microduck_local.walk_env import MicroduckWalkEnv

    venv = DummyVecEnv([lambda: MicroduckWalkEnv(seed=0)])
    model = PPO("MlpPolicy", venv, n_steps=32, batch_size=32, n_epochs=1,
                device="cpu")
    model.learn(total_timesteps=64)
    assert model.num_timesteps == 64
    model.learn(total_timesteps=64, reset_num_timesteps=False)
    assert model.num_timesteps == 128  # additive — NOT an absolute target


def _run(args: list[str], runs_dir: Path) -> None:
    env = {**os.environ, "MICRODUCK_RUNS_DIR": str(runs_dir)}
    res = subprocess.run(
        [sys.executable, "-m", "microduck_local.train_behavior", *args],
        env=env, cwd=str(PROJECT), timeout=240,
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"train-behavior failed:\n{res.stdout}\n{res.stderr}"


def _rollout_steps(run_dir: Path) -> list[int]:
    lines = [json.loads(ln)
             for ln in (run_dir / "progress.jsonl").read_text().splitlines()]
    return [ln["steps"] for ln in lines if "elapsed_s" in ln]


def test_init_from_continues_step_count(tmp_path):
    weights = '{"crouch_height": 2.0}'
    base = ["crouch", "--envs", "2", "--run-name", "resume-test",
            "--snap-steps", "6000", "--weights-json", weights]

    _run([*base, "--steps", "20000"], tmp_path)
    run_dir = tmp_path / "resume-test"
    # A snapshot is restart material: all three files, none half-written.
    for f in ("live.onnx", "model.zip", "vecnormalize.pkl", "policy.onnx"):
        assert (run_dir / f).exists(), f
    first_stage = _rollout_steps(run_dir)
    assert 20000 <= first_stage[-1] < 22000
    assert json.loads((run_dir / "behavior.json").read_text())["weights"] == {
        "crouch_height": 2.0}

    # Continue the SAME run with more envs — exactly what TrainingJob.scale()
    # launches. 40k is absolute: were SB3's additive semantic left
    # uncompensated, this stage would run to ~60k.
    _run([*base, "--steps", "40000", "--init-from", str(run_dir)], tmp_path)
    rollouts = _rollout_steps(run_dir)
    assert 40000 <= rollouts[-1] < 45000
    # One file, one monotone counter across both processes.
    assert rollouts == sorted(rollouts)
    assert len(rollouts) > len(first_stage)
    # The recorded scorecard survived the restart.
    assert json.loads((run_dir / "behavior.json").read_text())["weights"] == {
        "crouch_height": 2.0}
    last = json.loads((run_dir / "progress.jsonl").read_text().splitlines()[-1])
    assert last.get("done") is True and last["total"] == 40000


def test_export_ships_the_final_policy(tmp_path):
    """A run deploys its FINAL policy.

    Best-checkpoint selection was tried and removed: scoring on
    `keep_pace * ep_len` is ~90% ep_len, so it collapsed into
    "longest-surviving", and `_run_speed` pays 0.29 at zero velocity — a
    motionless duck scored well. On all three `run` jobs that shipped under it
    the FINAL policy was faster than the selected one (0.359 vs 0.236 m/s,
    0.112 vs 0.046, 0.343 vs 0.320). Any future criterion must score ACHIEVED
    GROUND SPEED, not a reward term.
    """
    from stable_baselines3 import PPO

    runs = tmp_path / "runs"
    _run(["stand", "--envs", "2", "--steps", "12000", "--run-name", "finalchk",
          "--snap-steps", "4000"], runs)
    run_dir = runs / "finalchk"
    assert (run_dir / "policy.onnx").exists()
    assert not (run_dir / "best_model.zip").exists(), "best-selection is gone"

    done = [json.loads(ln) for ln in (run_dir / "progress.jsonl").read_text().splitlines()
            if ln.strip()][-1]
    assert done.get("done") is True
    final = PPO.load(str(run_dir / "model"), device="cpu")
    assert final.num_timesteps >= 12000
