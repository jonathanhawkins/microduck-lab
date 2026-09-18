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

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..sensors.detector import DetectionFrame
from ..sensors.lidar import LidarFrame
from ..sensors.tof import TofFrame
from .graph import graph_key


@dataclass
class Senses:
    t: float
    tof: TofFrame | None = None
    tof_age: float | None = None
    det: DetectionFrame | None = None
    det_age: float | None = None
    speed: float | None = None           # heading-frame forward speed, m/s
    # What the robot's own telemetry adds: odometry (x, y, yaw — true here
    # for now; roadmap 1.7 makes it drift), whether the beak holds something
    # (servo load on the robot), and whether a skill cycle is running.
    odom: tuple[float, float, float] | None = None
    holding: bool = False
    skill: str | None = None
    bumped: bool = False                 # the body is touching another body (contacts here; IMU / servo loads on the robot)
    # A planar 360-degree scan, for a body that carries one (MARS's chassis
    # lid: `sensors/lidar.py`). A brain written for the duck never sees this
    # and does not have to: a wheeled body's `tof` above is filled by
    # `sensors.lidar.tof_from_lidar`, so `wander` and `follow` run on it
    # unchanged. This channel is here for a brain that wants the whole turn —
    # what is BEHIND the robot, which no ToF can answer.
    lidar: LidarFrame | None = None
    lidar_age: float | None = None
    # The ACHIEVED joint positions of a body with an arm, by joint name — what
    # Innate's `/mars/arm/state` publishes, and the counterpart of
    # `Intent.arm`'s commanded targets. A body with no arm leaves it None.
    #
    # It exists because the two are NOT the same, by design: `mars.arm_servo`
    # carries Innate's own structural compliance and backlash
    # (`STRUCT_STIFFNESS` 25 N*m/rad, `ARM_BACKLASH_RAD` 0.055), so a
    # commanded pose is reached with a few hundredths of a radian of sag.
    # MEASURED in a room at the pick pose: joint2 -0.067 rad and joint3
    # -0.048 rad off the command, which is **28.5 mm of claw height and
    # 11.5 mm of reach** — three times the margin a 58.8 mm jaw has on a
    # 40 mm block, and it is why `brain/tidy_arm.py` measures the sag at a
    # hover pose and pre-compensates the descent rather than trusting its own
    # forward kinematics. A learned policy reads the same numbers in
    # `robots/mars.OBS_ARM_QPOS` and closes the same loop by training.
    arm: Mapping[str, float] | None = None

    def fresh_lidar(self, max_age: float) -> LidarFrame | None:
        return self.lidar if (self.lidar is not None and self.lidar_age is not None
                              and self.lidar_age <= max_age) else None

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
    beak: str | None = None              # "open" | "close" — the mouth servo, outside the policy
    skill: str | None = None             # ask the reflex tier for a skill cycle ("ground_pick")
    # Joint position targets for a body with an ARM, by joint name (MARS:
    # `robots/mars.DRIVEN_JOINTS`). None leaves the arm where it is; a body
    # with no arm ignores it, the way a body with no beak ignores `beak`.
    # A NAMED mapping and not a vector, because an arm's joints are not a
    # policy's action block: a brain that only wants the gripper to close
    # says so in one key (`robots/mars_drive.MarsDriver.set_arm`), and a
    # vector would have made every such brain carry all seven numbers and
    # the order they go in.
    arm: Mapping[str, float] | None = None


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
        if kind.startswith("learned:"):
            from .learned import LearnedBrain  # lazy: onnxruntime
            return LearnedBrain(kind.split(":", 1)[1], **kw)
        try:
            return self.kinds[kind](**kw)
        except KeyError:
            raise ValueError(f"unknown brain kind {kind!r}; have {sorted(self.available())}") from None

    def available(self) -> list[str]:
        """Registered kinds plus every exported learned brain on disk."""
        from .learned import brains_dir
        out = sorted(self.kinds)
        d = brains_dir()
        if d.exists():
            out += [f"learned:{p.parent.name}" for p in sorted(d.glob("*/brain.onnx"))]
        return out


REGISTRY = BrainRegistry()


def attach_world(brain, world, robot_id: str):
    """Hand a brain the world it lives in and its own id, if it asked for them.

    A `Brain` is stepped on `Senses` alone and that is the contract — nothing
    here is a back door to physics. What a brain may legitimately need is the
    ROOM's own geometry, which is not a sense and is not a constant either:
    `brain/tidy_arm.TidyArm` reads `Basket.rim` to decide how high to hold the
    toy before it opens the jaw, and a tray of a different depth is a
    different number. A brain constructed with `truth=True` — a CONTROL ARM,
    never a benchmark — also reads the toy's true pose through this.

    Duck-typed (`hasattr`), so no brain has to grow slots it does not use and
    this is one call at every construction site instead of a branch on kind.
    """
    if hasattr(brain, "world"):
        brain.world = world
    if hasattr(brain, "robot_id"):
        brain.robot_id = robot_id
    return brain


def payload(brain: Brain | None, intent: Intent | None, mode: str) -> dict:
    """The frame's per-duck brain block."""
    if mode == "manual" or brain is None:
        # The graph key rides this branch too. Taking the wheel does not
        # change which graph the duck's brain is drawn on — and dropping the
        # key here blanked the inspector's state panel for as long as drive
        # mode was on, which is exactly when someone is steering into the
        # states they want to watch.
        # `state` must name a node of the `graph` beside it, or the panel
        # lights nothing and counts an off-chart visit: "manual" and "auto"
        # are declared in no graph. With a brain, report the state it is
        # actually parked in (it is not stepped while you drive, so this is
        # the last real one); only a duck with no brain at all falls back to
        # the mode, and it carries no graph either.
        return {"kind": "manual" if mode == "manual" else "script",
                "state": mode if brain is None else brain.state,
                "graph": None if brain is None else graph_key(brain),
                "cmd": [round(float(v), 3) for v in (intent.twist if intent else (0, 0, 0))],
                "inputs": {}}
    out = {"kind": brain.kind, "state": brain.state,
            # Which state graph the inspector should draw this duck on
            # (brain/graph.py). A key, not the table: the tables go once in
            # the world-info message, this rides every frame.
            "graph": graph_key(brain),
            "cmd": [round(float(v), 3) for v in (intent.twist if intent else (0, 0, 0))],
            "head": [round(float(v), 3) for v in (intent.head if intent else (0, 0, 0, 0))],
            "note": intent.note if intent else "",
            "beak": intent.beak if intent else None,
            "skill": intent.skill if intent else None,
            "inputs": brain.inputs()}
    # A learned brain can also say what its network SAW and SAID at the last
    # decision — the /sim inspector draws it. Rule brains have no such thing.
    view = brain.view() if hasattr(brain, "view") else None
    if view is not None:
        out["view"] = view
    return out


def brain_view(obs, raw, clipped, low, high, obs_version: int, decide_every: int) -> dict | None:
    """The wire shape of one decision: the observation vector the network
    read, the action as the network returned it, the action after the
    intent clip, and the bounds. For a brain exported by train_brain the
    two actions are equal — the graph clamps to its own bounds — so the
    saturated-mean tell (the log_std trap) is an action PINNED on a bound,
    which the viewer marks; `raw` is kept for a brain whose graph does not
    clamp. None before the first decision — there is nothing to show yet."""
    if obs is None or raw is None:
        return None

    def r(xs) -> list[float]:
        return [round(float(v), 3) for v in np.asarray(xs, np.float32).reshape(-1)]

    return {"obs": r(obs), "obs_version": int(obs_version), "decide_every": int(decide_every),
            "act": {"raw": r(raw), "clipped": r(clipped), "low": r(low), "high": r(high)}}


def _round(v) -> float | None:
    return None if v is None else round(float(v), 3)


def age_inputs(senses: Senses, tof_max: float, det_max: float) -> dict:
    """The freshness row per input channel, for the `/sim` inspector.

    **The range channel is named after the DEVICE the body has.** A duck's
    range sense is its 8x8 ToF and reports as `tof`; a body carrying a planar
    scanner reports as `lidar` and ships NO `tof` row, because on that body
    there is no such sensor — the 8x8 the brain gates on was adapted from the
    scan (`world/arena.World.senses_tof`) and a row labelled `tof` on a robot
    with no ToF is a claim about the hardware, not a note about the stream.
    The two are the same QUANTITY by construction: `senses_tof` keeps the
    SCAN's timestamp through the adaptation, so `tof_age` and `lidar_age` are
    one number and the gate `max` is this brain's own range-freshness bound
    either way.

    The scan decides it (`senses.lidar`), not a robot id, so no brain and
    nothing here has to know which bodies carry one. A wheeled body whose
    first scan has not landed yet has neither channel to report and falls
    back to the `tof` row — which is honest ("range sense: never"), and at
    6 Hz with `LidarSensor._next_t` starting at 0.0 it lasts until the first
    poll of the first step.
    """
    if senses.lidar is not None:
        rng = {"lidar": {"age": _round(senses.lidar_age),
                         "stale": senses.fresh_lidar(tof_max) is None, "max": tof_max}}
    else:
        rng = {"tof": {"age": _round(senses.tof_age),
                       "stale": senses.fresh_tof(tof_max) is None, "max": tof_max}}
    return {
        **rng,
        "det": {"age": _round(senses.det_age), "stale": senses.fresh_det(det_max) is None, "max": det_max,
                "n": 0 if senses.det is None else len(senses.det.detections)},
    }


__all__ = ["Brain", "BrainRegistry", "Intent", "REGISTRY", "Senses", "age_inputs",
           "attach_world", "brain_view", "graph_key", "payload", "np"]
