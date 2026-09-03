"""Plateau detection on a training reward curve — stop a run that stopped learning.

Measured motivation: `brains/follow-v4` was trained for 2M decisions and its
`ep_rew` curve was flat from ~900k — roughly half the wall clock bought
nothing. Trick runs show the same shape (the headstand chain's stages all
flatten well before their budget). A run that has stopped improving is not
just wasted time: on this harness the machine is the bottleneck, so a
half-hour of flat curve is a half-hour another recipe did not get.

Torch-free on purpose, same rule as `ppo_hparams.py`: `train_behavior.py`
forks its SubprocVecEnv workers BEFORE importing torch (a torch-initialized
parent has OpenMP/Accelerate thread pools that deadlock on a macOS fork), and
this module is imported at the top of that file. Stdlib + numpy only —
never import torch, stable_baselines3, or anything that pulls them in.

The detector is deliberately dumb and pure so it can be tested on synthetic
series without a trainer (`tests/test_plateau.py`): feed it one
`(steps, ep_rew)` pair per PPO rollout — exactly the series
`ProgressCallback` already writes to progress.jsonl — and it answers one
question, "has the smoothed best stopped moving?".

    det = PlateauDetector(patience=8, min_steps=400_000, rel=0.01, window=10)
    for steps, ep_rew in rollouts:
        if det.update(steps, ep_rew):
            break   # plateaued

OFF BY DEFAULT (`patience=0`). This repo's rule is that no change to how a
run spends its budget becomes a default without a seed-matched A/B at matched
step counts (AGENTS.md, "Performance work" and verification discipline #4) —
and early stopping is exactly that kind of change, since a curve that looks
flat can still be consolidating. Opt in per run with `--plateau-patience`, or
with $MICRODUCK_PLATEAU_PATIENCE for runs launched by the lab's /teach
endpoint, which passes no flags (same escape hatch as MICRODUCK_LR_START).
"""

from __future__ import annotations

import os
from collections import deque

import numpy as np

# patience 0 = disabled. The other three are only consulted once someone has
# turned it on, so they are sized for a real trick run: ~10 rollouts of
# smoothing (at 32 envs a rollout is 8192 steps, so the window is ~80k
# steps), no firing before 400k steps because early training legitimately
# plateaus while the value function catches up, and 1% relative improvement
# as the bar for "still learning".
DEFAULT_PATIENCE = 0
DEFAULT_MIN_STEPS = 400_000
DEFAULT_REL = 0.01
DEFAULT_WINDOW = 10

ENV_PATIENCE = "MICRODUCK_PLATEAU_PATIENCE"
ENV_MIN_STEPS = "MICRODUCK_PLATEAU_MIN_STEPS"
ENV_REL = "MICRODUCK_PLATEAU_REL"
ENV_WINDOW = "MICRODUCK_PLATEAU_WINDOW"


def _env(name: str, default, cast):
    """$NAME parsed as `cast`, falling back to `default` on junk.

    Same shape as train_behavior's `_env_lr`: a malformed env var must not
    take a training run down, it just means "not set".
    """
    try:
        return cast(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_defaults() -> dict[str, float]:
    """CLI defaults, env-overridable. Read at parser-build time, not import
    time, so a test (or a lab process that sets the var before launching a
    trainer) sees its own environment."""
    return {
        "patience": _env(ENV_PATIENCE, DEFAULT_PATIENCE, int),
        "min_steps": _env(ENV_MIN_STEPS, DEFAULT_MIN_STEPS, int),
        "rel": _env(ENV_REL, DEFAULT_REL, float),
        "window": _env(ENV_WINDOW, DEFAULT_WINDOW, int),
    }


class PlateauDetector:
    """Rolling-window plateau detector over one scalar series.

    One `update(steps, value)` per evaluation (one PPO rollout). The value is
    smoothed by a `window`-long rolling mean — a single rollout's `ep_rew` is
    far too noisy to judge, since it is a mean over whatever episodes happened
    to finish. A plateau is declared when that smoothed value has failed to
    beat the best smoothed value ever seen by more than `rel` (relative) for
    `patience` consecutive evaluations.

    Two guards against stopping a run that is still learning:

    * `best` only moves on a *significant* improvement, so a slow-but-steady
      climb of half a percent per rollout accumulates and eventually resets
      the counter instead of being ratcheted away (Keras' `min_delta`
      semantic). A noisy rising curve therefore survives.
    * `min_steps` is a warmup: evaluations before it can raise `best` but
      never age the counter, because early training legitimately looks flat
      while the critic catches up. With `patience` evaluations all required
      after the warmup, the detector cannot fire before `min_steps`.

    `update` is sticky: once it returns True it keeps returning True, so a
    caller that polls it after the fact reads a decision, not a fresh one.
    """

    def __init__(self, patience: int = DEFAULT_PATIENCE,
                 min_steps: int = DEFAULT_MIN_STEPS,
                 rel: float = DEFAULT_REL,
                 window: int = DEFAULT_WINDOW,
                 label: str = "ep_rew"):
        # `label` names the series being watched, for the stop message only.
        # It is not always ep_rew: `train-brain --probe-every` feeds this the
        # DETERMINISTIC benchmark score, because on a measured 2M run in-band
        # was flat from 250k while ep_rew climbed the whole way — a stop
        # watching reward would never have fired.
        self.label = str(label)
        self.patience = int(patience)
        self.min_steps = int(min_steps)
        self.rel = float(rel)
        self.window = max(1, int(window))
        self._vals: deque[float] = deque(maxlen=self.window)
        self.best: float | None = None
        self.best_steps = 0
        self.smoothed: float | None = None
        self.stale = 0
        self.evals = 0
        self.fired = False
        self.fired_steps = 0

    @property
    def enabled(self) -> bool:
        """patience <= 0 is the off switch — `update` is then a no-op."""
        return self.patience > 0

    def update(self, steps: int, value: float) -> bool:
        """Feed one evaluation. Returns True once (and forever after) plateaued."""
        if not self.enabled or self.fired:
            return self.fired
        self._vals.append(float(value))
        if len(self._vals) < self.window:
            return False  # not enough history to smooth: not an evaluation yet
        self.evals += 1
        self.smoothed = smoothed = float(np.mean(self._vals))
        # Relative to |best| so the bar has the right size whatever the
        # reward scale, and stays a real bar for a negative best.
        if self.best is None or smoothed > self.best + self.rel * abs(self.best):
            self.best, self.best_steps, self.stale = smoothed, int(steps), 0
            return False
        if int(steps) < self.min_steps:
            return False  # warmup: flat early training is not a plateau
        self.stale += 1
        if self.stale >= self.patience:
            self.fired, self.fired_steps = True, int(steps)
        return self.fired

    def summary(self) -> str:
        """One line for the trainer's stdout / train.log."""
        best = "n/a" if self.best is None else f"{self.best:.3f}"
        return (f"smoothed {self.label} ({self.window}-rollout mean) has not improved "
                f"by >{self.rel:.1%} for {self.stale} rollouts "
                f"(best {best} at {self.best_steps} steps); "
                f"stopping at {self.fired_steps} steps")

    def record(self, trained_steps: int) -> dict:
        """Fields to merge into progress.jsonl's terminating line.

        The run still reports `done` with its full `steps`/`total` — it
        completed, it just completed early — so `stopped` and `trained_steps`
        are what tell a later reader why the curve is short. Without them a
        plateau-stopped run is indistinguishable from a crashed one.
        """
        return {
            "stopped": "plateau",
            "trained_steps": int(trained_steps),
            "plateau": {
                "signal": self.label,
                "at_steps": self.fired_steps,
                "best": None if self.best is None else round(self.best, 4),
                "best_steps": self.best_steps,
                "stale": self.stale,
                "patience": self.patience,
                "min_steps": self.min_steps,
                "rel": self.rel,
                "window": self.window,
            },
        }
