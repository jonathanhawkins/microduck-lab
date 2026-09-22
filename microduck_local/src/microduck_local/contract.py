"""The deployment contract, mirrored from microduck_rl.

Single source of truth for everything that must match the runtime and the
official training stack. Values are copied from:
  - microduck_rl/scripts/infer_policy.py  (DEFAULT_POSE, obs order, 50 Hz timing)
  - microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py
    (action scale 1.0, command ranges, obs noise magnitudes)

Obs layout (61D, order is the hot-swap contract — never reorder):
  [ base_ang_vel(3), projected_gravity(3), joint_pos_rel(14), joint_vel(14),
    last_action(14), twist_cmd(3), head_pose_cmd(4), body_pose_cmd(6) ]
Action (14D): target = DEFAULT_POSE + action  (scale 1.0, radians).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .robots.microduck import MicroduckBody
from .robots.spec import Effector

# microduck_rl checkout providing the MJCF models. Sibling of this project by
# default; override with MICRODUCK_RL_DIR for a non-standard layout.
MICRODUCK_RL_DIR = Path(
    os.environ.get(
        "MICRODUCK_RL_DIR",
        Path(__file__).resolve().parents[3] / "microduck_rl",
    )
)
SCENE_WALK_XML = MICRODUCK_RL_DIR / "src/mjlab_microduck/robot/microduck/scene_walk.xml"
# Full-collision scene (head/trunk/hips can rest on the floor) — required by
# inverted/ground tricks (headstand); the walk scene strips those contacts.
SCENE_ALL_XML = MICRODUCK_RL_DIR / "src/mjlab_microduck/robot/microduck/scene.xml"

# Joint order (14 servos) — identical to model order in robot_walk.xml and to
# infer_policy.py's DEFAULT_POSE: 0-4 left leg, 5-8 neck/head, 9-13 right leg.
JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
NUM_JOINTS = 14
LEG_JOINT_IDS = np.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13])
HEAD_JOINT_IDS = np.array([5, 6, 7, 8])

# STAND2 pose (matches HOME_FRAME in microduck_constants.py and the STAND keyframe).
DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,        # left leg
    0.3491, 0.3491, 0.0, 0.0,                      # neck/head
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,          # right leg
], dtype=np.float32)

# Timing — infer_policy.py sets model.opt.timestep = 0.005, decimation 4 → 50 Hz.
PHYSICS_DT = 0.005
DECIMATION = 4
CTRL_DT = PHYSICS_DT * DECIMATION  # 0.02 s

# Obs block sizes, in contract order.
OBS_DIM = 61
CMD_DIM = 13  # twist(3) + head_pose(4) + body_pose(6)

# Command ranges (velocity env cfg — fixed, no widening curriculum).
LIN_VEL_X_RANGE = (-0.4, 0.4)
LIN_VEL_Y_RANGE = (-0.3, 0.3)
ANG_VEL_Z_RANGE = (-1.0, 1.0)
# Keep-alive ranges for the unused command slots (velocity env initial ranges):
# neurons for these inputs must stay alive for later curricula / other tasks.
HEAD_CMD_RANGES = ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))
BODY_CMD_RANGES = tuple(((-0.005, 0.005),) * 3 + ((-0.05, 0.05),) * 3)

# Actor observation noise (uniform, from the velocity env cfg).
NOISE_GYRO = 0.03
NOISE_GRAVITY = 0.01
NOISE_JOINT_POS = 0.001
NOISE_JOINT_VEL = 0.25


# ---------------------------------------------------------------- robot spec
#
# The same constants above, as the data `walk_env` resolves a body with (see
# robots/spec.py). This is a DESCRIPTION of the duck, not a second source of
# truth: every field is read from the constants above, so the deployment
# contract still lives in exactly one place.
# 🎬 editor data for the duck, served by pose.PoseScratch.meta(). The rig
# controls are the set the viewer shipped in duck-viewer/lib/rig.ts, moved
# here so a second body can carry its own; the coefficient signs come from
# the WORLD hinge axes at STAND (x forward, y left, z up):
#
#   left  hip_pitch +y   knee −y   ankle +y      hip_roll +x   hip_yaw −z
#   right hip_pitch −y   knee +y   ankle −y      hip_roll +x   hip_yaw −z
#   neck_pitch −y   head_pitch +y
#
# A flat foot needs the leg's world-pitch sum (root + hip + knee + ankle) to
# stay constant; every control below preserves it, and every pair of controls
# is orthogonal in joint space (rig sliders read 0 until used).
JOINT_GROUPS = ("left leg",) * 5 + ("head + neck",) * 4 + ("right leg",) * 5

EFFECTORS = (
    Effector("left_foot", "left foot", "ankle_left", "foot", "sole"),
    Effector("right_foot", "right foot", "ankle_right", "foot", "sole"),
    Effector("head", "head", "jaw_soft", "head", (0.0, 0.0, 0.0)),
)

RIG_CONTROLS = (
    {"id": "squat", "label": "squat", "hint": "+ crouch",
     "title": "fold both legs symmetrically, feet flat, trunk upright — the ⇕ "
              "handle drags this when no other control is selected",
     "parts": {"left_hip_pitch": -1, "left_knee": -2, "left_ankle": -1,
               "right_hip_pitch": 1, "right_knee": 2, "right_ankle": 1},
     "pick": ["left_knee", "right_knee"],
     "handle": {"joint": "root", "offset": [-0.105, 0, 0.03]}},
    {"id": "lean", "label": "lean", "hint": "+ fwd",
     "title": "the trunk pitches while the legs counterbalance, feet flat",
     "parts": {"root": 1,
               "left_hip_pitch": -1 / 3, "left_knee": 1 / 3, "left_ankle": -1 / 3,
               "right_hip_pitch": 1 / 3, "right_knee": -1 / 3, "right_ankle": 1 / 3},
     "pick": ["root"],
     "handle": {"joint": "root", "offset": [-0.105, 0, 0.115]}},
    {"id": "swingL", "label": "L swing", "hint": "+ fwd",
     "title": "swing the whole left leg forward/back about the hip, foot kept "
              "level — pair with R swing for a stride",
     "parts": {"left_hip_pitch": -1, "left_ankle": 1},
     "pick": ["left_hip_pitch"],
     "handle": {"joint": "left_hip_pitch", "offset": [0, 0.07, 0]}},
    {"id": "swingR", "label": "R swing", "hint": "+ fwd",
     "title": "swing the whole right leg forward/back about the hip, foot kept "
              "level — pair with L swing for a stride",
     "parts": {"right_hip_pitch": 1, "right_ankle": -1},
     "pick": ["right_hip_pitch"],
     "handle": {"joint": "right_hip_pitch", "offset": [0, -0.07, 0]}},
    {"id": "sway", "label": "sway", "hint": "hips ±",
     "title": "both hip rolls together — swing the legs sideways under the trunk",
     "parts": {"left_hip_roll": 1, "right_hip_roll": 1},
     "pick": ["left_hip_roll", "right_hip_roll"],
     "handle": {"joint": "root", "offset": [0, 0.115, 0.01]}},
    {"id": "stance", "label": "stance", "hint": "+ wide",
     "title": "hip rolls apart — widen or narrow the stance",
     "parts": {"left_hip_roll": -1, "right_hip_roll": 1},
     "pick": [],
     "handle": {"joint": "root", "offset": [0, -0.115, 0.01]}},
    {"id": "twist", "label": "twist", "hint": "hips ±",
     "title": "both hip yaws together — pivot the hips against the feet",
     "parts": {"left_hip_yaw": 1, "right_hip_yaw": 1},
     "pick": ["left_hip_yaw", "right_hip_yaw"],
     "handle": {"joint": "root", "offset": [-0.145, 0, -0.025]}},
    {"id": "toes", "label": "toes", "hint": "+ out",
     "title": "hip yaws apart — duck-foot or pigeon-toe the stance",
     "parts": {"left_hip_yaw": -1, "right_hip_yaw": 1},
     "pick": ["left_ankle", "right_ankle"],
     "handle": {"joint": "left_ankle", "offset": [0.075, 0, 0.015]}},
    {"id": "look", "label": "look", "hint": "+ down",
     "title": "neck and head pitch share the motion — one radian of control "
              "is one radian of gaze",
     "parts": {"neck_pitch": -0.5, "head_pitch": 0.5},
     "pick": ["neck_pitch", "head_pitch", "head_yaw", "head_roll"],
     "handle": {"joint": "head_pitch", "offset": [-0.02, 0, 0.115]}},
)

MICRODUCK = MicroduckBody(
    id="microduck",
    title="Microduck",
    noun="duck",
    joint_groups=JOINT_GROUPS,
    effectors=EFFECTORS,
    rig_controls=RIG_CONTROLS,
    joint_names=JOINT_NAMES,
    default_pose=DEFAULT_POSE,
    obs_dim=OBS_DIM,
    base_body="trunk_base",
    gyro_sensor="imu_ang_vel",
    foot_geoms={"left": ("left_foot_collision",),
                "right": ("right_foot_collision",)},
    scene_fn=lambda: SCENE_WALK_XML,
    action_scale=None,               # contract: target = DEFAULT_POSE + action
    pose_joint_ids=LEG_JOINT_IDS,
    # walk_env's own thresholds, verbatim (FALL_GRAVITY_Z / FALL_HEIGHT).
    fall_gravity_z=-0.342,
    fall_height=0.07,
    # Upstream HEAD_BODY_NAMES; walk_env.HEAD_COM_BODIES is the same tuple.
    com_bodies=("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft",
                "bearing_roll"),
    noise_gyro=NOISE_GYRO,
    noise_gravity=NOISE_GRAVITY,
    noise_joint_pos=NOISE_JOINT_POS,
    noise_joint_vel=NOISE_JOINT_VEL,
    lin_vel_x_range=LIN_VEL_X_RANGE,
    lin_vel_y_range=LIN_VEL_Y_RANGE,
    ang_vel_z_range=ANG_VEL_Z_RANGE,
)


def quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate `vec` by the inverse of quaternion [w, x, y, z] — verbatim from infer_policy.py.

    The crosses are unrolled by hand: `np.cross` on single 3-vectors spends
    ~15 us in moveaxis/broadcast plumbing and this runs 4x per control step
    (obs + rewards). Each component below is the exact multiply-subtract
    np.cross performs, so the result is bit-identical (held to the bit by
    test_bam_perf_parity.py).
    """
    w = quat_wxyz[0]
    x, y, z = quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    v0, v1, v2 = vec[0], vec[1], vec[2]
    t0 = (y * v2 - z * v1) * 2   # t = cross(xyz, vec) * 2
    t1 = (z * v0 - x * v2) * 2
    t2 = (x * v1 - y * v0) * 2
    return np.array((v0 - w * t0 + (y * t2 - z * t1),
                     v1 - w * t1 + (z * t0 - x * t2),
                     v2 - w * t2 + (x * t1 - y * t0)), dtype=np.float32)


# The play world's rolling resistance (`world/scenario.py` `Ball.rolling`): a
# short carpet, the floor a home robot lives on. Mirrored here so the training
# and bench scene is the same ball as the match.
BALL_ROLLING = 0.002


def scene_walk_ball_xml() -> Path:
    """The walk scene with upstream's 70 mm / 15 g kick ball (roadmap item 7,
    4c revisit): `scene_walk.xml` rewritten beside SYMLINKS to every file of
    the robot directory - MuJoCo resolves meshes relative to the main file -
    with `ball.xml` included last and every keyframe padded by the ball's
    seven qpos, since a free joint changes nq and upstream's own
    scene_ball.xml drops its keyframes for exactly that reason. Generated on
    demand under microduck_local/.cache; the pinned upstream checkout is
    never written to. The walk env indexes joints by address, so nothing
    else in it changes: measured, the shipped walker walks this scene
    without a fall."""
    import re
    src, d = SCENE_WALK_XML, SCENE_WALK_XML.parent
    out = Path(__file__).resolve().parents[2] / ".cache" / "scene"
    out.mkdir(parents=True, exist_ok=True)
    for entry in d.iterdir():
        link = out / entry.name
        target = entry.resolve()
        if entry.name == "ball.xml":
            # THE BALL THE ROBOT ACTUALLY PLAYS WITH. Upstream's ball.xml
            # carries `friction="0.5 0.005 0.0001"` on a geom with no
            # `condim`, and MuJoCo's default of 3 applies the SLIDING
            # coefficient only - so its rolling value is silently ignored and
            # the ball rolls until a wall stops it. `world/compose.py` fixed
            # exactly this for the play world on 2026-09-06 (condim 6,
            # `Ball.rolling`); the training and bench scene kept the frictionless
            # one, so the kick was trained and benched on a ball that behaves
            # nothing like the one it meets in a match. Found by the
            # 2026-09-08 review, which is also why the bench's 0% whiff and
            # its 1.0-1.3 m travel numbers were never comparable with play.
            # Patched here rather than in the pinned checkout, which is never
            # written to.
            txt = target.read_text()
            if 'name="ball_geom"' in txt and "condim" not in txt:
                txt = txt.replace('name="ball_geom"', 'name="ball_geom" condim="6"')
                txt = txt.replace('friction="0.5 0.005 0.0001"', f'friction="0.5 0.005 {BALL_ROLLING}"')
            if link.is_symlink():
                link.unlink()
            if not link.exists() or link.read_text() != txt:
                tmp = link.with_name(f".{link.name}.{os.getpid()}.tmp")
                tmp.write_text(txt)
                os.replace(tmp, link)
            continue
        if link.is_symlink():
            if os.readlink(link) == str(target):
                continue
            link.unlink()
        elif link.exists():
            link.unlink()
        try:
            link.symlink_to(target)
        except FileExistsError:                       # a sibling vec-env worker got there first
            pass
    xml = src.read_text().replace("</mujoco>", '    <include file="ball.xml"/>\n</mujoco>')
    xml = re.sub(r'qpos="([^"]+)"', lambda m: f'qpos="{" ".join(m.group(1).split())} 0.3 0 0.035 1 0 0 0"', xml)
    p = out / "scene_walk_ball.xml"
    if not p.exists() or p.read_text() != xml:
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")        # atomic: a worker never reads a half-written scene
        tmp.write_text(xml)
        os.replace(tmp, p)
    return p
