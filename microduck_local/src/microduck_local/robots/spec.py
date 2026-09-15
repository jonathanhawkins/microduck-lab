"""What the walking env needs to know about a robot, as data.

`walk_env.py` grew around one body and resolved it by hard-coded name:
``trunk_base``, ``floor``, ``left_foot_collision``, the 14 names in
``contract.JOINT_NAMES``. Everything else in the file is already generic —
the rewards, the domain randomization, the shared-model plumbing, the
step cache. A `RobotSpec` is that handful of names lifted out, so a second
body (the Unitree G1) can reuse the env instead of forking it.

The duck's spec lives in `contract.MICRODUCK` and is bit-for-bit what the
env used to hard-code: the golden-bit tests are the proof (tests/goldens/).
Nothing here changes the 61-obs / 14-action deployment contract — a spec
DESCRIBES a robot, it does not redefine the duck's.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Effector:
    """A point on the body the 🎬 editor can DRAG: an inverse-kinematics
    target (pose.PoseScratch.solve_ik). `body` is a MuJoCo body name; the
    joints the solver may move are found by walking that body's parents up
    to the base, so a renamed link fails at construction like every other
    name in a spec. `point` is where on the body the target sits, in the
    body frame — the literal string "sole" asks for the centre of the foot's
    footprint at floor level (measured from the collision geoms at STAND),
    which is the point an animator means when they drag a foot."""

    id: str                                   # "left_foot" — the wire name
    label: str                                # "left foot" — the panel name
    body: str
    kind: str = "foot"                        # foot | hand | head
    point: str | tuple[float, float, float] | None = None


@dataclass(frozen=True, eq=False)     # eq=False: ndarray fields have no bool eq
class RobotSpec:
    """One body the env can walk.

    Every field is either a NAME the compiled model is asked for (so a
    model revision that renames a link fails loudly at construction rather
    than silently walking on the wrong geometry) or a dimension.
    """

    id: str                                   # "microduck" | "g1"
    joint_names: tuple[str, ...]              # policy/action order
    default_pose: np.ndarray                  # rad, len == num_joints
    obs_dim: int
    base_body: str                            # the IMU body: trunk / pelvis
    gyro_sensor: str                          # mjOBJ_SENSOR name, 3 floats
    foot_geoms: Mapping[str, tuple[str, ...]] # {"left": (...), "right": (...)}
    scene_fn: Callable[[], Path]              # default training scene
    # target = default_pose + action * action_scale. The duck's contract is
    # scale 1.0 (contract.py); the G1's walker.onnx carries a per-joint scale.
    action_scale: np.ndarray | None = None
    action_clip: float = 4.0
    floor_geom: str = "floor"
    stand_keyframe: str = "STAND"
    # Joints the pose reward holds at the default (the duck pays on legs
    # only — its head has its own command-tracking term).
    pose_joint_ids: np.ndarray | None = None
    # Termination, in the base's own body frame / world z.
    fall_gravity_z: float = -0.342            # tilted > 70 deg
    fall_height: float = 0.07                 # m, base body world z
    # Domain randomization: the base body plus whatever else upstream offsets.
    com_bodies: tuple[str, ...] = ()
    # Actor observation noise, in obs order (uniform half-widths).
    noise_gyro: float = 0.03
    noise_gravity: float = 0.01
    noise_joint_pos: float = 0.001
    noise_joint_vel: float = 0.25
    # Command ranges the env samples (m/s, m/s, rad/s).
    lin_vel_x_range: tuple[float, float] = (-0.4, 0.4)
    lin_vel_y_range: tuple[float, float] = (-0.3, 0.3)
    ang_vel_z_range: tuple[float, float] = (-1.0, 1.0)
    # The "forward" command branch clamps |vx| to at least this (mjlab
    # rel_forward_envs). It is a floor on what counts as a walking order, and
    # a robot whose gait does not start below some speed needs its own.
    min_forward_cmd: float = 0.3
    # Where the twist command sits in the observation, as (start, stop). The
    # obs normalizer needs it: a policy cloned under a PINNED command has
    # ~zero variance in these slots, so the first real command it is shown
    # normalizes to thousands and clips — see train._seed_command_stats.
    twist_obs_slice: tuple[int, int] = (48, 51)
    # Velocity pushes, m/s on the base's world xy (upstream's DR).
    push_vel_range: tuple[float, float] = (-0.3, 0.3)
    # How much floor ONE SLOT on the lab stage gets, centre to centre (m).
    # The lab lays its roster out in a square grid and each slot is pitched by
    # its own robot (viz_server.lab_slot_offsets), because one duck-sized
    # constant put six 1.3 m G1 helpers inside each other.
    # Measured at each robot's own STAND keyframe — the widest HORIZONTAL
    # extent of the whole body's geom AABBs, which is what must not overlap:
    #     microduck  0.1845 m wide  ->  0.65 m   (3.52x, the pitch the viewer
    #                                             has always drawn: this default)
    #     g1         0.5338 m wide  ->  1.88 m   (the same 3.52x; robots/g1.py)
    # tests/test_lab_robots.py re-measures both widths from the models, so a
    # model revision that changes a body's size fails here rather than quietly
    # crowding the stage. (Scaling on standing height instead — 0.277 m vs
    # 1.307 m — would give the G1 3.07 m: also defensible, simply further apart
    # than the stage camera can frame.)
    lab_spacing_m: float = 0.65
    # --- the 🎬 animation editor (pose.py serves these to the viewer) ------
    # Panel section label per joint, in joint_names order (None: one group).
    joint_groups: tuple[str, ...] | None = None
    # The draggable IK targets — feet, hands — see Effector.
    effectors: tuple[Effector, ...] = ()
    # 🎮 rig controls: coupled-joint macro sliders, served verbatim to the
    # viewer (duck-viewer/lib/rig.ts documents the shape; contract.py carries
    # the duck's set, robots/g1.py the G1's). Each is a direction in
    # (joints + rootPitch) space keyed by joint NAME, so a control naming a
    # joint this body lacks is dropped by the viewer, never mis-wired.
    rig_controls: tuple[Mapping[str, object], ...] = ()
    # Sole vertices within this of the sole's lowest point, standing, are
    # its flat (the duck's sole has a 5 mm fillet; 1 mm picks the flat).
    sole_tol: float = 0.001
    # A foot whose lowest point is this far above the other foot's is in
    # the air. Millimetres on a 25 cm duck; a 1.3 m humanoid needs more.
    ground_tol: float = 0.005
    # Free-form, for tools that want to say something about the robot.
    title: str = ""
    extra: Mapping[str, object] = field(default_factory=dict)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def num_actions(self) -> int:
        return len(self.joint_names)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """Policy output -> position target (radians)."""
        if self.action_scale is None:
            return self.default_pose + action
        return self.default_pose + action * self.action_scale

    def mirror_joint_perm(self) -> np.ndarray:
        """left_X <-> right_X by name; everything else maps to itself."""
        names = list(self.joint_names)
        out = []
        for n in names:
            if n.startswith("left_"):
                partner = "right_" + n[len("left_"):]
            elif n.startswith("right_"):
                partner = "left_" + n[len("right_"):]
            else:
                partner = n
            out.append(names.index(partner) if partner in names else names.index(n))
        return np.array(out, dtype=np.int64)


def registry() -> dict[str, RobotSpec]:
    """Every robot the training stack can build, by id.

    Imported lazily: the G1 spec touches `.cache/unitree_g1`, which most
    machines have never fetched, and asking for the duck must not care.
    """
    from .. import contract as C

    out = {C.MICRODUCK.id: C.MICRODUCK}
    try:
        from .g1 import G1_SPEC
    except Exception:                          # assets missing / import error
        return out
    out[G1_SPEC.id] = G1_SPEC
    return out


def get(robot_id: str) -> RobotSpec:
    reg = registry()
    if robot_id not in reg:
        raise KeyError(
            f"unknown robot {robot_id!r} — have {sorted(reg)}"
            + ("; `uv run fetch-g1` adds the G1" if robot_id == "g1" else ""))
    return reg[robot_id]
