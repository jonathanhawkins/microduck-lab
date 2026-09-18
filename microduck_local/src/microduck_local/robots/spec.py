"""What the walking env needs to know about a robot, as data.

`walk_env.py` grew around one body and resolved it by hard-coded name:
``trunk_base``, ``floor``, ``left_foot_collision``, the 14 names in
``contract.JOINT_NAMES``. Everything else in the file is already generic —
the rewards, the domain randomization, the shared-model plumbing, the
step cache. A `RobotSpec` is that handful of names lifted out, so a second
body (the Unitree G1) can reuse the env instead of forking it.

A `RobotSpec` is a `Body` (`robots/body.py` — what the LAB needs) PLUS the
walking env's names and windows. The split exists because the next body
planned does not walk: `docs/mars-roadmap.md` §1. Nothing moved out of the
walker's half, so the duck and the G1 keep every field they had.

The duck's spec lives in `contract.MICRODUCK` and is bit-for-bit what the
env used to hard-code: the golden-bit tests are the proof (tests/goldens/).
Nothing here changes the 61-obs / 14-action deployment contract — a spec
DESCRIBES a robot, it does not redefine the duck's.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .body import Body, BodyBase, _no_scene_fn


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


@dataclass(frozen=True, eq=False, kw_only=True)   # eq=False: ndarray fields
class RobotSpec(BodyBase):
    """One body the env can walk.

    Every field is either a NAME the compiled model is asked for (so a
    model revision that renames a link fails loudly at construction rather
    than silently walking on the wrong geometry) or a dimension.

    `id`, `title`, `noun`, `kind`, `joint_names`, `joint_groups`,
    `default_pose`, `obs_dim` and `lab_spacing_m` come from `BodyBase`: they
    are what the lab, the viewer and `/sim` ask any body, walker or not.
    """

    base_body: str                            # the IMU body: trunk / pelvis
    gyro_sensor: str                          # mjOBJ_SENSOR name, 3 floats
    foot_geoms: Mapping[str, tuple[str, ...]] # {"left": (...), "right": (...)}
    # `scene_fn` (the default training scene) and `stand_keyframe` are
    # `BodyBase` fields now — every body has a model and a spawn pose, walker
    # or not. Declaring them here was what made a non-walker impossible.
    # `__post_init__` keeps a walker's own loudness: they were REQUIRED
    # keywords on this class, and a RobotSpec without a scene still fails at
    # construction rather than when something tries to compile it.
    # target = default_pose + action * action_scale. The duck's contract is
    # scale 1.0 (contract.py); the G1's walker.onnx carries a per-joint scale.
    action_scale: np.ndarray | None = None
    action_clip: float = 4.0
    floor_geom: str = "floor"
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
    # --- the 🎬 animation editor (pose.py serves these to the viewer) ------
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
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """A walker with no scene is a construction error, as it always was.

        `scene_fn` was a required keyword on this class before it moved up to
        `BodyBase` (where it needs a default, because a body may be declared
        before its model exists). Inheriting a default would have turned
        "you forgot the scene" from a TypeError here into a
        NotImplementedError somewhere in the trainer, so the loudness is
        re-stated rather than lost.
        """
        if self.scene_fn is _no_scene_fn:
            raise TypeError(
                f"RobotSpec {self.id!r} has no scene_fn — the walking env "
                "spawns from that scene's keyframe, so it is required")

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

    # ------------------------------------------------------ the Body contract
    #
    # ONLY the answers that hold for ANY walker, computed from the spec
    # itself. Everything that names a robot's assets — its meshes, its
    # shipped policies, its env, how it attaches — is on the body:
    # `robots/microduck.MicroduckBody` and `robots/g1.G1Body`. A generic
    # default that happened to be one robot's would be inherited in silence,
    # and arrive as a wrong picture and a wrong policy instead of an error.

    def ready(self) -> bool:
        """The training scene is on disk.

        Generic: a walker is set up when the scene it trains in exists.
        Cheap for a body whose `scene_fn` is a path, and False rather than an
        exception for one whose `scene_fn` GENERATES a scene from assets that
        are missing — `tests/test_body_conformance.py` uses the same rule as
        its fallback, and `G1Body` narrows it to the meshes and the ONNX too.
        """
        try:
            return Path(self.scene_fn()).is_file()
        except Exception:
            return False


def registry() -> dict[str, Body]:
    """Every robot the training stack can build, by id.

    A thin alias for `robots/registry.registry()`, which is where bodies now
    come from (including ones a pip-installed plugin adds). Kept because
    `viz_server`, `export_onnx`, `motion`, `pose` and two test modules import
    it from here. The values are `Body`, not `RobotSpec`: a non-walker is a
    legal registry entry, and a caller that needs `foot_geoms` should ask the
    body for the walker's half rather than assume every entry has one.
    """
    from . import registry as _registry
    return _registry.registry()


def get(robot_id: str) -> Body:
    """`robots/registry.get()`, under the name the tree already imports."""
    from . import registry as _registry
    return _registry.get(robot_id)


__all__ = ["Body", "BodyBase", "Effector", "RobotSpec", "get", "registry"]
