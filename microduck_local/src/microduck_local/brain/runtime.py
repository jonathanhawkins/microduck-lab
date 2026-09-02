"""The brain runtime (roadmap 2.1): senses in, intents out, at its own rate.

A `Brain` never touches physics. Each control tick the world hands it a
`Senses` snapshot — the newest ToF frame and detection frame with their
AGES — and it returns an `Intent`: a twist (the robot's `robot.move`) and a
head pose (`robot.head`). Freshness gating is the brain's job, not the
sensor's: a controller that keeps steering on a 2-second-old detection is
the classic failure the /sim inspector exists to show, so `Senses` carries
ages and `Brain.stale` says what "too old" means for this brain.

Kinds registered here are what the page's brain picker offers:
  wander  — cruise on the ToF, turn toward the open side (controllers.Wander)
  follow  — keep a person (or a duck) ahead at a set distance (controllers.Follow)
  script  — no brain: the drive script / manual command steers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..sensors.detector import DetectionFrame
from ..sensors.tof import TofFrame


@dataclass
class Senses:
    t: float
    tof: TofFrame | None = None
    tof_age: float | None = None
    det: DetectionFrame | None = None
    det_age: float | None = None
    speed: float | None = None           # heading-frame forward speed, m/s

    def fresh_tof(self, max_age: float) -> TofFrame | None:
        return self.tof if (self.tof is not None and self.tof_age is not None
                            and self.tof_age <= max_age) else None

    def fresh_det(self, max_age: float) -> DetectionFrame | None:
        return self.det if (self.det is not None and self.det_age is not None
                            and self.det_age <= max_age) else None


@dataclass
class Intent:
    twist: tuple[float, float, float] = (0.0, 0.0, 0.0)
    head: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)   # neck_pitch, head_pitch, head_yaw, head_roll
    note: str = ""                       # one-line "why" for the inspector


class Brain(Protocol):
    kind: str
    state: str

    def step(self, senses: Senses) -> Intent: ...
    def reset(self) -> None: ...
    def inputs(self) -> dict: ...        # what it looked at, with freshness, for the inspector


@dataclass
class BrainRegistry:
    kinds: dict[str, type] = field(default_factory=dict)

    def register(self, kind: str, cls: type) -> None:
        self.kinds[kind] = cls

    def make(self, kind: str, **kw) -> Brain:
        try:
            return self.kinds[kind](**kw)
        except KeyError:
            raise ValueError(f"unknown brain kind {kind!r}; have {sorted(self.kinds)}") from None


REGISTRY = BrainRegistry()


def payload(brain: Brain | None, intent: Intent | None, mode: str) -> dict:
    """The frame's per-duck brain block."""
    if mode == "manual" or brain is None:
        return {"kind": "manual" if mode == "manual" else "script", "state": mode,
                "cmd": [round(float(v), 3) for v in (intent.twist if intent else (0, 0, 0))],
                "inputs": {}}
    return {"kind": brain.kind, "state": brain.state,
            "cmd": [round(float(v), 3) for v in (intent.twist if intent else (0, 0, 0))],
            "head": [round(float(v), 3) for v in (intent.head if intent else (0, 0, 0, 0))],
            "note": intent.note if intent else "",
            "inputs": brain.inputs()}


def _round(v) -> float | None:
    return None if v is None else round(float(v), 3)


def age_inputs(senses: Senses, tof_max: float, det_max: float) -> dict:
    return {
        "tof": {"age": _round(senses.tof_age), "stale": senses.fresh_tof(tof_max) is None, "max": tof_max},
        "det": {"age": _round(senses.det_age), "stale": senses.fresh_det(det_max) is None, "max": det_max,
                "n": 0 if senses.det is None else len(senses.det.detections)},
    }


__all__ = ["Brain", "BrainRegistry", "Intent", "REGISTRY", "Senses", "age_inputs", "payload", "np"]
