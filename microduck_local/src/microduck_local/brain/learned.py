"""A trained brain running in the world: `learned:<name>` in the registry.

`train-brain` writes brains/<name>/brain.onnx (obs normalizer baked in,
output already clamped to the intent bounds) and brain.json. This loads it
and turns `Senses` into the same 80-float observation the env trained on
(`brain_env.senses_to_obs`), at the env's decision rate, holding the last
intent between decisions — exactly the cadence a 10 Hz brain would run at
on the robot.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .brain_env import ACT_HIGH, ACT_LOW, BRAIN_OBS_DIM, ObsBuilder, onnx_infer
from .runtime import Intent, Senses, age_inputs


def brains_dir() -> Path:
    return Path(os.environ.get("MICRODUCK_BRAINS_DIR",
                               Path(__file__).resolve().parents[3] / "brains"))


class LearnedBrain:
    kind = "learned"

    def __init__(self, name: str, decide_every: int | None = None):
        d = brains_dir() / name
        onnx = d / "brain.onnx"
        if not onnx.exists():
            raise ValueError(f"no exported brain at {onnx}")
        meta = json.loads((d / "brain.json").read_text()) if (d / "brain.json").exists() else {}
        self.name = name
        self.kind = f"learned:{name}"
        self.target_cls = meta.get("target_cls", "person")
        self.decide_every = int(decide_every or meta.get("decide_every", 5))
        self.obs_version = int(meta.get("obs_version", 1))      # brains before the version key are version 1
        self.builder = ObsBuilder(self.target_cls, self.obs_version)
        self.infer = onnx_infer(onnx)
        self.state = "learned"
        self.last_action = np.zeros(3, np.float32)
        self.last_seen_t: float | None = None
        self._tick = 0
        self._senses: Senses | None = None

    def reset(self) -> None:
        self.last_action[:] = 0.0
        self.last_seen_t = None
        self.builder.reset()
        self._tick = 0
        self._senses = None

    def inputs(self) -> dict:
        if self._senses is None:
            return {}
        out = age_inputs(self._senses, 0.25, 0.4)
        out["target"] = None if self.last_seen_t is None else {
            "bearing": None, "range": None, "since": round(self._senses.t - self.last_seen_t, 2)}
        return out

    def step(self, senses: Senses) -> Intent:
        self._senses = senses
        if self._tick % self.decide_every == 0:
            obs = self.builder(senses, self.last_action)
            self.last_seen_t = self.builder.last_seen_t
            a = np.clip(self.infer(obs.astype(np.float32)), ACT_LOW, ACT_HIGH)
            self.last_action = a.astype(np.float32)
            seen = obs[65] > 0.5
            self.state = "tracking" if seen else "lost"
        self._tick += 1
        return Intent(twist=tuple(float(v) for v in self.last_action), note=self.state)


__all__ = ["BRAIN_OBS_DIM", "LearnedBrain", "brains_dir"]
