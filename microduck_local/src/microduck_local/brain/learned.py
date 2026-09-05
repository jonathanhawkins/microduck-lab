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
from .runtime import Intent, Senses, age_inputs, brain_view


def learned_index() -> list[dict]:
    """One entry per exported learned brain, with what people READ about it:
    {name, title, group, description}. The /sim brain menu files the entries
    under their group and shows the title; the name stays the value it
    sends back, because learned:<name> is the identifier."""
    out = []
    d = brains_dir()
    if not d.exists():
        return out
    for onnx in sorted(d.glob("*/brain.onnx")):
        meta = {}
        try:
            meta = json.loads((onnx.parent / "brain.json").read_text())
        except (OSError, ValueError):
            pass
        out.append({"name": onnx.parent.name, "title": meta.get("title"),
                    "group": meta.get("group"), "description": meta.get("description")})
    return out


def brains_dir() -> Path:
    return Path(os.environ.get("MICRODUCK_BRAINS_DIR",
                               Path(__file__).resolve().parents[3] / "brains"))


class LearnedBrain:
    kind = "learned"

    def __init__(self, name: str, decide_every: int | None = None):
        # A bare name is a shipped brain under `brains/`; anything that looks
        # like a path is loaded from there instead. That is what lets
        # `select-brain` score a mid-run CHECKPOINT on the real benchmark
        # without first shipping it as brain.onnx — the deterministic probe
        # the reward curve cannot stand in for.
        d = Path(name) if ("/" in name or "\\" in name) else brains_dir() / name
        onnx = d / "brain.onnx"
        if not onnx.exists():
            raise ValueError(f"no exported brain at {onnx}")
        meta = json.loads((d / "brain.json").read_text()) if (d / "brain.json").exists() else {}
        self.name = d.name
        self.kind = f"learned:{name}"
        self.target_cls = meta.get("target_cls", "person")
        self.decide_every = int(decide_every or meta.get("decide_every", 5))
        self.obs_version = int(meta.get("obs_version", 1))      # brains before the version key are version 1
        # The intent bounds are part of THIS brain's contract, so they come
        # from its own record — not from the module constants, which a later
        # run may have widened. Clipping a brain trained at wz +-2.0 back to
        # the current default would silently halve its turn rate at inference
        # and make any A/B over the bound measure nothing. Brains written
        # before the keys existed fall back to the constants they used.
        self.act_low = np.asarray(meta.get("act_low", ACT_LOW), np.float32)
        self.act_high = np.asarray(meta.get("act_high", ACT_HIGH), np.float32)
        self.builder = ObsBuilder(self.target_cls, self.obs_version)
        self.infer = onnx_infer(onnx)
        self.state = "learned"
        self.last_action = np.zeros(3, np.float32)
        self.last_seen_t: float | None = None
        # What the network saw and said at the last decision (view()).
        self.last_obs: np.ndarray | None = None
        self.last_raw: np.ndarray | None = None
        self._tick = 0
        self._senses: Senses | None = None

    def reset(self) -> None:
        self.last_action[:] = 0.0
        self.last_seen_t = None
        self.last_obs = None
        self.last_raw = None
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
            raw = np.asarray(self.infer(obs.astype(np.float32)), np.float32)
            a = np.clip(raw, self.act_low, self.act_high)
            self.last_action = a.astype(np.float32)
            self.last_obs = obs.astype(np.float32)
            self.last_raw = raw
            seen = obs[65] > 0.5
            self.state = "tracking" if seen else "lost"
        self._tick += 1
        return Intent(twist=tuple(float(v) for v in self.last_action), note=self.state)

    def view(self) -> dict | None:
        """The last decision, for the frame's `brain.view` (runtime.brain_view)."""
        return brain_view(self.last_obs, self.last_raw, self.last_action, self.act_low, self.act_high,
                          self.obs_version, self.decide_every)


__all__ = ["BRAIN_OBS_DIM", "LearnedBrain", "brains_dir"]
