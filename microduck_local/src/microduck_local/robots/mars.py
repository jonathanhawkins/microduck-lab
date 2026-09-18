"""Innate's MARS: the lab's third body, and the first that does not walk.

A differential-drive base carrying a 5-joint arm with a parallel-jaw gripper
and a pitching head — 40 cm reach, 250 g payload, 1.37 kg of printed parts in
the description. `docs/mars-roadmap.md` §0 has the robot, the servos and the
measured Phase 0 numbers; §1 says why it is a `Body` and not a `RobotSpec`:

    it has no feet, it cannot fall, and the thing this harness calls "the
    policy" — a velocity-command walker — does not exist for it.

Threading it through `RobotSpec` would be the architecture-scale version of
the mistake `AGENTS.md` warns about most, so `MarsBody` is a `BodyBase`: the
lab's contract, none of the walker's. What it has instead of a gait is an
ARM, and that is Phase 4.

**The assets are Innate's, and so is the recipe.** `mars.urdf`, `arm.srdf`
and nine STL meshes come from
[innate-inc/innate-os](https://github.com/innate-inc/innate-os) (Apache-2.0)
at the pinned sha below — 7.2 MB, downloaded file by file with a sha256 each
rather than cloned (the repo is 185 MB). Everything in `robot_spec()` and
`arm_servo()` is a port of their own MuJoCo driver
(`ros2_ws/src/mars_bot/mars_sim_driver/{world,core}.py`: `load_robot_spec`,
`add_planar_base`, `tune_contacts`, `style_robot_geoms`, `_apply_control`,
`joint2_min_target`), and every constant taken from it names the function it
came from, so a drive or a grasp that is wrong here is wrong there too.

**What is NOT here.** The base drive — their velocity PD through
`xfrc_applied`, station keeping and the `cmd_vel` watchdog — lives in
`robots/mars_drive.py`, which `driver()` hands out (Phase 3a); its 360-degree
lidar is `sensors/lidar.py`. Read that module before changing `ARM_HOME` or
the planar base: one constant of Innate's could NOT be ported verbatim (their
`KP_YAW` is unstable at this repo's 5 ms step) and the reason is measured
there. **The arm env is `robots/mars_env.py`** (Phase 4a) — `MarsArmEnv`,
which `env_class()` hands out, with its `reach` and `pick` tasks as
`Behavior`s in `behaviors/mars_tasks.py`; it is the thing that FILLS the
observation layout declared below, and `scripts/probe_mars_reach.py` /
`scripts/probe_mars_pick.py` are the eyes for it. `place` is still Phase 4c.
This module is the download, the body, the model, the arm/head servo, the
senses' mounts and the viewer's mesh dump.

MEASURED on this Mac, `scene_xml()` compiled (2026-09-17):

    nq=11 nv=11 nu=0 nbody=18 ngeom=59 nmesh=9, 1.365 kg
    2 s at ARM_HOME on `arm_servo`: max |q - home| 0.00344 rad,
        base z drift 0.0 mm, planar drift 1e-5 m / 1.4e-4 rad, ncon <= 6
    134,700 physics steps/s bare, 80,000 with the servo in python
        (the duck's own scene: 70,400 — MARS is 0.52x a duck per step)

`nu == 0` is not a mistake: mars.urdf has no `<actuator>` block because
Innate's driver commands positions through `qfrc_applied`, and `arm_servo`
is that. Nothing here writes `data.ctrl`.
"""

from __future__ import annotations

import hashlib
import math
import os
import urllib.request
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from .. import contract as C
from .body import BodyBase, RobotFrames
from .policy_contract import PolicyContract, Slot, declare

_ROOT = Path(__file__).resolve().parents[3]  # microduck_local/

# --------------------------------------------------------------- the assets
#
# innate-os at the sha docs/mars-roadmap.md §0 pins (2026-09-17). A revision
# is a different robot: the collision boxes, the finger blades and ARM_HOME
# are all hand-measured in that tree, so the sha travels with the manifest.
INNATE_OS_SHA = "0ca73670ae20940e381de8b4356ac9855f66e25a"
INNATE_OS_LICENCE = "Apache-2.0 (innate-inc/innate-os)"
RAW_BASE = ("https://raw.githubusercontent.com/innate-inc/innate-os/"
            f"{INNATE_OS_SHA}/ros2_ws/src/mars_bot/mars_description")

#: Where `fetch()` puts them. `MICRODUCK_MARS_DIR` moves the cache, the way
#: `MICRODUCK_G1_DIR` does, so a worktree can verify without touching the
#: main checkout's download.
CACHE_DIR = Path(os.environ.get("MICRODUCK_MARS_DIR")
                 or _ROOT / ".cache" / "innate_mars")
#: The package directory the URDF's `package://mars_description/` resolves to.
#: Kept as a real directory name because `load_robot_spec` rewrites the URI to
#: a path and MuJoCo then reads `meshes/` relative to it.
ASSET_SUBDIR = "mars_description"

#: (path under `mars_description/`, sha256). Every file `fetch()` downloads
#: and `ready()` looks for — measured from the pinned revision, so a moved
#: tag, a truncated download or a proxy's error page is refused instead of
#: compiled. 11 files, 7.2 MB (base.STL is 5.4 MB of it).
ASSETS: tuple[tuple[str, str], ...] = (
    ("urdf/mars.urdf",
     "d5690f75f3d9eb21d1c6dc4683126b1fe854355d202f4f5a4bae71700b0ded14"),
    ("urdf/arm.srdf",
     "1e042c33ea7220e2a91017605493e1c2c27a510a4bb12e45b71a27c7f1e60145"),
    ("meshes/base.STL",
     "e725a419bbe85a9346a0187aff6b9c1fd3bcf9cf6cc1f1c4164777134b31f5ca"),
    ("meshes/head.STL",
     "3852748e253644f9e9d710e92d11a15fee06cc9eeb4ed0d595eeedfebc03bd55"),
    ("meshes/link1.STL",
     "9eb892e67efe447e802009cbbeaf46c30c86d46c08563858c9d26f1cee286587"),
    ("meshes/link2.STL",
     "d8fe401af1ee98323ff128eeaab64945533ccf6e5a836ae1252fb571b6d70182"),
    ("meshes/link3.STL",
     "71956e0cd0b868c46200af5e1b0a5191207fc056e0878fbba9dffbdd023add40"),
    ("meshes/link4.STL",
     "2f1c87f188ddd71479f30d2e20c488b3d7eb30b4d19d4aec572e70d4c0529fae"),
    ("meshes/link5.STL",
     "e668681a8f62060382c73f11630a1a752b11332012791886b1e21bfceeff54d0"),
    ("meshes/link61.STL",
     "b57b1b055e3d0bb8eda43ae23604219a1155c01f4fcecfea6d3560494280e40e"),
    ("meshes/link62.STL",
     "44df6c24e9af8cc1bc5a0feb179d23c7039c590e45dff7e5a388363f79b5f0ba"),
)

SCENE_NAME = "scene_mars.xml"
DOWNLOAD_TIMEOUT_S = 60.0

# ----------------------------------------------------------- what MARS is
#
# The six joints `/mars/arm/state` reports, in that order. `joint6M` is the
# mirrored finger (a `<mimic>` in the URDF, driven at -1 by the servo below)
# and is NOT a policy joint; `joint_head` is a separate command slot, like
# the duck's neck, so it is not in `joint_names` either.
ARM_JOINTS: tuple[str, ...] = ("joint1", "joint2", "joint3", "joint4",
                               "joint5", "joint6")
HEAD_JOINT = "joint_head"
#: (mimic, source, multiplier) — innate world.MIMIC_JOINT.
MIMIC_JOINT: tuple[str, str, float] = ("joint6M", "joint6", -1.0)
#: Every joint the servo drives, in innate's own order (world.DRIVEN_JOINTS).
DRIVEN_JOINTS: tuple[str, ...] = ARM_JOINTS + (HEAD_JOINT,)
#: The planar base's DoFs (see `add_planar_base`).
BASE_JOINTS: tuple[str, ...] = ("base_x", "base_y", "base_yaw")
BASE_BODY = "base_link"
#: Frames the URDF carries that Phase 3's senses will mount on.
LIDAR_SITE = "base_laser"          # 360 deg 2-D lidar on the chassis lid
CAMERA_BODY = "head_camera_left"   # the stereo head camera's left eye
EFFECTOR_BODY = "ee_link"          # the gripper's tool point
FINGER_LINKS: tuple[str, str] = ("link61", "link62")

#: innate world.ARM_HOME, to the digit — the webapp's ARM_HOME_POSITIONS.
ARM_HOME: dict[str, float] = {
    "joint1": 1.445009902188274,
    "joint2": -1.3882526130365052,
    "joint3": 1.517106999218899,
    "joint4": 0.44638840927472156,
    "joint5": -0.08897088569736719,
    "joint6": 0.0015339807878856412,
    "joint_head": 0.0,
}
#: `default_pose`: ARM_HOME for the six policy joints. The head's home is 0
#: and rides in its own command slot.
DEFAULT_POSE = np.array([ARM_HOME[j] for j in ARM_JOINTS], np.float32)
#: Panel sections for the 🎬 editor, in `joint_names` order.
JOINT_GROUPS: tuple[str, ...] = ("arm",) * 5 + ("gripper",)
HOME_KEY = "HOME"
FLOOR_GEOM = "floor"

# Widest HORIZONTAL extent of the whole body at the HOME keyframe (m), and
# the lab-stage pitch derived from it, the way robots/g1.BODY_WIDTH_M was.
# MEASURED off the compiled scene (geom AABBs in world axes, arm FOLDED at
# ARM_HOME — `tests/test_body_conformance._widest_horizontal_extent_m`
# re-measures it, so a URDF revision that changes the robot's size fails
# there rather than quietly crowding the stage):
#
#     HOME   extent (x, y, z) = 0.4135 x 0.3665 x 0.4483 m
#     qpos0  extent (x, y, z) = 0.6555 x 0.3628 x 0.4483 m
#
# So the ROADMAP'S 1.09 m estimate was low, twice over: it came from the
# probe's y extent (0.311 m, rbound-based, collision geoms only) at qpos0,
# where the arm points straight out. What actually neighbours a slot is the
# body's LENGTH — 0.4135 m at HOME, the rear tray overhanging the chassis by
# 76 mm plus the folded arm — which is why `_widest_horizontal_extent_m`
# takes the larger of x and y. 3.52x that (the duck's ratio) is 1.4555.
BODY_WIDTH_M = 0.4135
LAB_SPACING_M = 1.46

# ------------------------------------------------------- the head camera's lens
#
# Innate's CALIBRATED intrinsics, from `docs/mars-roadmap.md` §0: the stereo
# head camera is 640x480 on the wire with fx 200.3 / fy 267.3 / cx 319.1 /
# cy 248.7. A pinhole lens's full field of view is 2*atan(half_px / f), so
#
#     hfov = 2*atan(320 / 200.3) = 2.0237 rad = 115.95 deg
#     vfov = 2*atan(240 / 267.3) = 1.4633 rad =  83.84 deg
#
# and they are DERIVED here rather than typed, because this lab has already
# paid for a hard-coded field of view twice: the duck's default was the stock
# lens's full-array figure for a camera the robot does not run that way
# (`sensors/detector.py`'s block), and an uncalibrated equidistant default
# inflated every bearing by 1.58x near the axis (AGENTS.md, "Uncalibrated lens
# is a bearing GAIN"). A calibration is two numbers and an arctangent; a
# degree figure is a number somebody once wrote down.
#
# Note how WIDE this is: 116 degrees horizontally is the same class of lens
# the duck turned out to have, and 84 vertically is wider still. The
# projection stays `pinhole` (the detector's default) because these ARE
# pinhole intrinsics — Innate publishes fx/fy/cx/cy for a pinhole model — and
# no distortion coefficients come with them, so the frame edges are as wrong
# here as they are for the duck, and for the same reason.
CAMERA_PX = (640, 480)
CAMERA_FX, CAMERA_FY = 200.3, 267.3
CAMERA_HFOV_DEG = 2.0 * math.degrees(math.atan(CAMERA_PX[0] / 2 / CAMERA_FX))
CAMERA_VFOV_DEG = 2.0 * math.degrees(math.atan(CAMERA_PX[1] / 2 / CAMERA_FY))
#: The head camera's depth range (Innate's published spec, §0): 0.4-6 m. The
#: detector here is geometric and has one range knob, so the FAR end is what
#: it takes; nothing models the 0.4 m near cutoff.
CAMERA_MAX_RANGE_M = 6.0
#: Innate's own detection cadence is not published, so the lab's default rate
#: stands (`sensors/detector.DetectorSpec.rate_hz`) and this constant exists
#: to say that it is a LAB number and not the robot's — the same honesty the
#: duck's 10 Hz carries against its robot's measured 2 Hz.
CAMERA_RATE_HZ = 10.0

# ------------------------------------------------------- the scanner's own body
#
# A range return closer than this to the BASE ORIGIN is MARS looking at
# itself. MEASURED (`sensors/lidar.py`'s module docstring): at `ARM_HOME` the
# folded arm's `link5` blocks 4 of the 360 rays, at 7-10 degrees, from
# 0.1562 m off the laser — and the laser sits 76.4 mm behind the base origin,
# so in the base frame those returns are ~0.08 m out at ~15 degrees, well
# inside the chassis. A consumer that reads them as obstacles steers away
# from its own elbow for a whole run.
#
# 0.12 m and not the half-width (0.207 m): the cut must be big enough to
# cover the elbow and small enough that it cannot hide anything a brain needs.
# It cannot hide an obstacle that matters, because the chassis's own front
# face is ~0.17 m ahead of the origin and its wheels are wider than 0.12 m —
# something 12 cm from the base origin is already inside the robot. The
# scanner's own 0.15 m minimum range is a further floor underneath this.
FOOTPRINT_M = 0.12

# ------------------------------------------------- the v1 observation layout
#
# `robots/mars_env.MarsArmEnv` fills this, and its `_SLOTS` is asserted
# against `MarsBody.contract()` at construction — name by name and width by
# width — so the env may not have a layout of its own.
#
# It was declared HERE and BEFORE that env existed, which is the part worth
# keeping: `obs_dim` is what the lab refuses a wrong policy by, what the
# exporter shapes the graph from, and what the conformance suite checks, so
# the body had to speak one width from the day it was listed rather than one
# that appeared when the first env was written. It cost Phase 4a nothing to
# fill a table it did not get to choose, which is the evidence that declaring
# first was the right way round.
#
# One fixed layout per body, zero-padded, exactly as the duck's 61 floats are
# (`contract.py`): a task that does not use a slot sends zeros rather than
# re-packing, because a policy is hot-swapped behind a width and two layouts
# at one width cross silently (§6.3's contract ids are the eventual fix).
OBS_ARM_QPOS = slice(0, 6)        # rad, the six joints, absolute (not rel)
OBS_ARM_QVEL = slice(6, 12)       # rad/s
OBS_HEAD_PITCH = slice(12, 13)    # rad, joint_head
OBS_GRIPPER_LOAD = slice(13, 14)  # N*m at joint6 — what "holding" reads
#
# WHAT THAT N*m IS, and it changed once. Phase 4a filled this slot with the
# torque `arm_servo` WROTE (`qfrc_applied`), and Phase 4b measured that this
# cannot be the holding signal: `set_arm` clamps the close target to the hard
# stop, so shutting on air ends with zero position error and zero torque — and
# while the blades are TRAVELLING through air the same torque is saturated at
# -2 N*m, so one sample cannot tell "closing" from "holding". MEASURED
# (`scripts/probe_mars_pick.py --scripted`, 2 ms, the playroom block):
#
#                                   qfrc_constraint   qfrc_applied  contacts
#     open, empty                        0.0000         -0.0000       0 /  0
#     shut on AIR, at the stop           0.0000         -0.0000       0 /  0
#     shut claw driven into the FLOOR    0.0000         -0.0000       0 /  0
#     jaws resting open ON the block     0.0035         -0.0063       1 /  1
#     HOLDING the block                  1.9703         -2.0000       4 / 12
#
# The three EMPTY rows read 0.0000 in every pose measured, which is the split
# that matters. The "resting" row is pose-dependent (0.0035 where the jaws
# merely touch, 1.82 where the approach has pressed them onto the block) and
# is the right answer either way: a blade carrying an object's reaction IS
# loaded by it. `mars_env.HOLD_LOAD_NM` documents what that buys and what it
# does not.
#
# ...and through a whole close on AIR, `qfrc_applied` is saturated at -2 N*m
# for 8 of the 40 control steps (the travel) before it falls to 0 at the stop,
# while `qfrc_constraint` is 0.0000 for all 40. One sample of the servo torque
# cannot tell "closing" from "holding"; one sample of the constraint can.
#
# So the slot carries `qfrc_constraint` at joint6 — the torque the OBJECT
# feeds back through the blade — which is 0.0000 in every empty case and
# ~2 N*m with something in the claw. On hardware it is not a register: an
# Innate code skill reconstructs it from `present_load` AND the encoder (the
# real servo stalls at its current limit on air too, so load alone says
# nothing — `arm_qpos[5]` short of `GRIPPER_CLOSED_ON_AIR_RAD` is the other
# half). `reach` is unaffected in practice: under its trained policy the old
# slot was non-zero on 7.6% of steps and the new one is 0.0000 throughout.
OBS_LAST_ACTION = slice(14, 22)   # the previous action, all 8 slots
OBS_TARGET_BASE = slice(22, 25)   # xyz of the task target in the BASE frame
OBS_TARGET_SEEN = slice(25, 26)   # 1.0 when the detector has it this step
OBS_BASE_TWIST = slice(26, 28)    # measured (vx, wz) of the planar base
OBS_RESERVED = slice(28, 32)      # zeros: room for a task's own slots
#: The MARS policy contract, v1. 32 = 28 used + 4 reserved.
OBS_DIM = 32

# Actions: six joint targets plus the base twist the wheels get. 8, not 6 —
# `MarsBody.num_actions` overrides `BodyBase`'s one-action-per-joint because
# a wheeled body's action vector is not its joint vector. An arm-only task
# sends zeros in the base pair.
ACT_ARM = slice(0, 6)
ACT_BASE_TWIST = slice(6, 8)      # (vx m/s, wz rad/s)
NUM_ACTIONS = 8
#: Innate's policy-defined skills tick at 25 Hz, so a lab policy and a real
#: code skill run the same clock (docs/mars-roadmap.md Phase 4).
CONTROL_HZ = 25.0
#: MARS's policy contract id (`robots/policy_contract.py`). "arm" and not
#: "walk" because that is the whole design decision of §1: the thing this
#: harness calls a policy does not exist for a wheeled body, and what it has
#: instead is an arm. `v1` is the layout declared above, unfilled — the id
#: exists from the day the body is listed so that the FIRST exported MARS
#: policy is self-describing, rather than a 32-wide file that has to be
#: recognised by its width.
CONTRACT_ID = "mars-arm-32-v1"


def _obs_slots() -> tuple[Slot, ...]:
    """The v1 layout, straight off the `OBS_*` slices above.

    Built from the slices rather than re-typed, so the table and the
    constants Phase 4's `MarsArmEnv` will index with cannot disagree — the
    duck's and the G1's tables are derived from their joint counts for the
    same reason.

    `reserved` is a NAMED slot, not a gap: `PolicyContract.tiles()` refuses
    an undescribed float, and "four zeros a task may claim" is information
    that a hole in the table would throw away.
    """
    return (
        Slot("arm_qpos", OBS_ARM_QPOS.start, OBS_ARM_QPOS.stop),
        Slot("arm_qvel", OBS_ARM_QVEL.start, OBS_ARM_QVEL.stop),
        Slot("head_pitch", OBS_HEAD_PITCH.start, OBS_HEAD_PITCH.stop),
        Slot("gripper_load", OBS_GRIPPER_LOAD.start, OBS_GRIPPER_LOAD.stop),
        Slot("last_action", OBS_LAST_ACTION.start, OBS_LAST_ACTION.stop),
        Slot("target_base", OBS_TARGET_BASE.start, OBS_TARGET_BASE.stop),
        Slot("target_seen", OBS_TARGET_SEEN.start, OBS_TARGET_SEEN.stop),
        Slot("base_twist", OBS_BASE_TWIST.start, OBS_BASE_TWIST.stop),
        Slot("reserved", OBS_RESERVED.start, OBS_RESERVED.stop),
    )

# --------------------------------------------- innate's contact tuning
#
# All from world.py's `tune_contacts` and its constant block; their comments
# are summarised, not replaced — read theirs for the measurements.
#: Frictionless drive wheels: the planar base pins z, so a tangent wheel
#: answers every step with ~50 N of spurious normal force whose friction cone
#: glues the base. The chassis box, 7 mm narrower, does the gripping.
WHEEL_GEOMS: tuple[str, str] = ("base_wheel_left", "base_wheel_right")
FINGER_CONDIM = 6                      # boxes need torsion, spheres rolling
FINGER_FRICTION = (2.0, 0.05, 0.02)    # (slide, torsion, roll)
FINGER_SOLREF = (0.005, 1.0)
FINGER_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2)
FINGER_DAMPING = 1.0                   # sets the ~0.45 s full close
FINGER_ARMATURE = 1e-4                 # else 12 g blades sink into the grasp
#: The real claw's hard stop is past nominal zero: closed on air the encoder
#: reads this. Unclamped, a -0.6 close target scissors the blades through
#: each other (world.GRIPPER_CLOSED_ON_AIR_RAD).
GRIPPER_CLOSED_ON_AIR_RAD = -0.085

#: |`qfrc_constraint` at joint6| above which the claw is HOLDING something.
#:
#: It lives HERE, with the table above that decided it, because it is a fact
#: about the ROBOT and two consumers need it: `robots/mars_env.HOLD_LOAD_NM`
#: (which re-exports this name, so every 4b number and every planted break in
#: `tests/test_mars_pick.py` still reads the same float) and
#: `robots/mars_drive.MarsDriver.held_body`, which is how a MARS in a `/sim`
#: room fills `Senses.holding`. `mars_drive` cannot import `mars_env` — that
#: module pulls gymnasium in — so a constant shared by the env and the driver
#: has exactly one honest home.
#:
#: **What it separates is EMPTY from LOADED**, and the margin is two orders:
#: every empty pose measured reads 0.0000 (open, shut on air at the stop, the
#: shut claw driven 0.25 rad into the floor) and the block in the claw reads
#: 1.89-2.03. It does NOT separate "gripped" from "resting against" — that
#: reading is pose-dependent and is the right answer either way, because a
#: blade carrying 1.8 N*m of an object's reaction IS loaded by it.
HOLD_LOAD_NM = 1.0

#: The world timestep MARS's CLAW needs — Innate's own (`Body.physics_dt`).
#:
#: 2 ms, MEASURED, and it is the one lever: at this repo's 5 ms a grasp does
#: not slip, it EJECTS (the block leaves at 0.08-6.6 m of travel in the two
#: seconds after a lift, a contact impulse the coarse step cannot integrate),
#: and the scripted pick holds 4 of 16 spots against 14 of 16 at 2 ms. The
#: elliptic cone and `impratio 10` are a NULL at either step. Read by
#: `world/scenario.robot_physics_dt`, so a room built for a MARS gets the
#: clock its claw needs without any builder naming this number.
GRASP_PHYSICS_DT = 0.002

#: How fast a COMMANDED joint target may travel, rad/s.
#:
#: `[datasheet — the XL430-W250-T shoulder pair's 61 rpm no-load at 12 V; not
#: measured here]`. It is the servo's own speed, which is why it lives with
#: the robot: `MarsDriver.set_arm` takes an ABSOLUTE target and applies no
#: limit at all, so every writer has to carry this, and 4a's transferable bug
#: was exactly a limit that existed on one side and not the other.
#:
#: **What it costs to forget.** MEASURED twice. In the env (4a): with the
#: target applied instantly, 60 of 60 random episodes ended in a
#: self-collision inside 1-5 control steps, because a 3 rad step every 40 ms
#: drives the arm straight through the chassis. In a ROOM (Phase 5, the first
#: `tidy_arm` run): `brain/tidy_arm.py` wrote its IK solution as one
#: `Intent.arm` and the servo slammed it — the 20 g block was flung out of the
#: room (found at (5.2, 7.2) m in a 3.5 x 3.0 m room) on 6 of 6 picks, which
#: reads exactly like the 5 ms EJECTION failure and is not it. Both consumers
#: now slew: `mars_env.MAX_TARGET_RATE_RAD_S` is this name, and
#: `TidyArmParams` slews its own commanded target at it.
MAX_TARGET_RATE_RAD_S = 6.0

# ------------------------------------------------------- innate's arm servo
#
# core.py: the arm and head run position PD through `qfrc_applied`, because
# the URDF has no `<actuator>` block and their real node commands positions.
# That is why a compiled MARS has `nu == 0` and `arm_servo` writes forces.
KP_JOINT = 50.0
KD_JOINT = 1.0
EFFORT_LIMIT = 50.0            # N*m
#: The gripper runs on mars.urdf's real joint6 rating: 50 N*m on a 45 mm
#: finger is 1.1 kN of pinch, which ejects what it grabs. 2 N*m is the ~44 N
#: the actual servo delivers, and its velocity term is zero because the
#: finger's 2e-5 inertia makes an explicit -kd*qvel unstable at this
#: timestep — FINGER_DAMPING damps it instead (core.KD_GRIPPER).
GRIPPER_EFFORT_LIMIT = 2.0     # N*m
KD_GRIPPER = 0.0
#: arm_control.cpp's "intelligent joint limits": when joint1 swings the arm
#: across the robot's front arc, joint2 may not stay folded up — the arm must
#: duck UNDER the head rather than sweep through it. The real floor is -0.5,
#: but the simplified collision boxes still overlap ~9 mm there, so the sim
#: ducks to -0.25 (core.JOINT2_GUARD_MIN).
JOINT2_GUARD_MIN = -0.25
#: Structural sag past the encoders (gear play, link flex): the LINK settles
#: below the servo angle under gravity load. Innate's estimates (~19 mm at
#: the pick pose) until measured on a real arm. OFF by default here — see
#: `arm_servo`'s `sag` argument for why.
STRUCT_STIFFNESS = 25.0        # N*m/rad, per arm joint
ARM_BACKLASH_RAD = 0.055
BACKLASH_TANH_NM = 0.05

# ------------------------------------------------------------ innate's look
#
# world.style_robot_geoms. Applied to the viewer's DUMP rather than to a
# render: the lab ships colours per geom and the browser paints them.
ORANGE_LINKS = frozenset({"link1", "link3", "link5"})
BRIGHT_ORANGE = (1.0, 0.5, 0.0, 1.0)
#: matt_black (0.05) lifted to charcoal, so the chassis reads as a shape
#: rather than a silhouette.
CHARCOAL = 0.16
DARK_RGB_MEAN = 0.4            # below this mean, a geom is "matt black"
#: Frame markers the URDF draws as 5 mm spheres. Hidden, as in their viewer.
HIDDEN_MARKER_LINKS = frozenset({"ee_link", "head_camera_left",
                                 "head_camera_right"})
#: MuJoCo's URDF importer puts `<visual>` geoms in group 1 and `<collision>`
#: geoms in group 0 (MEASURED on this URDF, 12 visual / 46 collision). The
#: dump asks for the visual group and for meshes only, which is also what
#: hides the collision boxes and the marker spheres.
VISUAL_GROUP = 1
COLLISION_GROUP = 3            # where style sweeps collision geoms (hidden)


# --------------------------------------------------------------- the assets

def cache_dir() -> Path:
    return CACHE_DIR


def asset_dir(dest: Path | None = None) -> Path:
    """The `mars_description` directory inside the cache."""
    return (dest or CACHE_DIR) / ASSET_SUBDIR


def urdf_path(dest: Path | None = None) -> Path:
    return asset_dir(dest) / "urdf/mars.urdf"


def mars_ready(dest: Path | None = None) -> bool:
    """Are all 11 files present?

    Presence, not sha256: `ready()` is asked per roster change and per policy
    load in the lab, and re-hashing 7.2 MB on each of those is a cost nobody
    asked for. The hashes are what `fetch()` verifies before a file is put in
    place, so a file that is HERE was verified when it arrived — and
    `tests/test_mars.py` pins that a corrupted one is refused on the next
    fetch rather than silently kept.
    """
    d = asset_dir(dest)
    return all((d / rel).is_file() for rel, _sha in ASSETS)


def require_mars(dest: Path | None = None) -> Path:
    if not mars_ready(dest):
        raise FileNotFoundError(
            f"Innate MARS assets not in {asset_dir(dest)} — run "
            "`uv run fetch-robot mars` (mars.urdf + arm.srdf + 9 STLs, "
            f"7.2 MB, {INNATE_OS_LICENCE} at {INNATE_OS_SHA[:12]})")
    return dest or CACHE_DIR


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Fetch one raw file to `dest`. The seam `tests/test_mars.py` replaces.

    Separated from `fetch()` on purpose: every other property of the download
    — idempotence, the sha256 refusal, the atomic replace — is then testable
    with no network at all.
    """
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as r:
        dest.write_bytes(r.read())


def fetch(dest: Path | None = None) -> Path:
    """Download the description into the cache, verifying every sha256.

    Idempotent: a file that is present and hashes correctly is skipped, so a
    second run costs 11 hashes and no bytes. Each download lands on a temp
    name in the target directory and is `os.replace`d into place only after
    its hash matches — the repo's atomic-write rule, and here it also means a
    half-written STL can never be imported by a parallel worker (AGENTS.md,
    "Atomic writes and live imports").

    A hash that does not match is a RuntimeError naming both hashes. It is
    the one check that can tell a moved revision, a truncated transfer and a
    captive-portal HTML page apart from the robot.
    """
    root = dest or CACHE_DIR
    d = asset_dir(root)
    fresh = 0
    for rel, want in ASSETS:
        out = d / rel
        if out.is_file() and _sha256(out) == want:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.name}.{os.getpid()}.part")
        url = f"{RAW_BASE}/{rel}"
        print(f"[mars] {rel} <- {url}")
        try:
            _download(url, tmp)
            got = _sha256(tmp)
            if got != want:
                raise RuntimeError(
                    f"{rel} from innate-os {INNATE_OS_SHA[:12]} hashes "
                    f"{got[:12]}, the manifest says {want[:12]} — refusing "
                    "it (robots/mars.ASSETS pins the revision)")
            os.replace(tmp, out)
        finally:
            tmp.unlink(missing_ok=True)
        fresh += 1
    if not mars_ready(root):
        missing = [rel for rel, _ in ASSETS if not (d / rel).is_file()]
        raise RuntimeError(f"MARS fetch finished but {missing} are missing")
    print(f"[mars] {fresh} file(s) downloaded, {len(ASSETS) - fresh} already "
          f"verified — {d} ({INNATE_OS_LICENCE})")
    return root


# ---------------------------------------------------------------- the model

def load_robot_spec(urdf: Path | None = None) -> mujoco.MjSpec:
    """mars.urdf with its `<visual>` STL meshes and its FRAMES intact.

    Innate's own two rewrites (world.load_robot_spec):

    * MuJoCo's URDF importer DISCARDS visuals unless the embedded
      `<mujoco><compiler discardvisual="false"/></mujoco>` override is
      present, and
    * it cannot resolve `package://` URIs.

    And a third this lab needs, `fusestatic="false"`. The URDF importer
    defaults it ON, which merges every jointless frame into its parent —
    `base_laser`, `ee_link`, `head_camera_left/right`, `base_footprint` and
    the two optical frames, 8 of the 18 bodies. Three reasons not to, in
    order of how much they cost:

    1. **The body list would depend on WHEN the spec was compiled.** MEASURED
       on this URDF: `robot_spec().compile()` alone gives 10 bodies and
       1.3650 kg, and attaching that ALREADY-COMPILED spec into a world gives
       10 bodies and **1.3340 kg** — the 31 g of marker links is silently
       dropped the second time. The lab streams body poses as a positional
       list built from one model and drawn by another (`visual_scene()` vs
       the composed world), so two compiles that disagree on the body list is
       a robot drawn with its parts on the wrong joints.
    2. Phase 3 mounts the senses on `base_laser` and `head_camera_left`, and
       Phase 4's reward reads `ee_link`. A fused frame has no name to
       resolve, and `mj_name2id` answers -1 — the silent failure this repo's
       spec fields exist to prevent.
    3. With it off, the numbers match `docs/mars-roadmap.md` §0's measured
       Phase 0 row (nbody 18, 1.365 kg), which was taken through an attach.

    All three are text edits on the way in, so the file on disk stays
    byte-identical to the manifest's sha256.
    """
    path = urdf or urdf_path()
    if not path.is_file():
        require_mars()                        # raises with the fetch command
    pkg = path.parent if (path.parent / "meshes").is_dir() else path.parent.parent
    text = path.read_text().replace("package://mars_description/",
                                    str(pkg.resolve()) + "/")
    text = text.replace(
        '<robot name="mars_bot">',
        '<robot name="mars_bot"><mujoco>'
        '<compiler discardvisual="false" fusestatic="false"/></mujoco>')
    return mujoco.MjSpec.from_string(text)


def add_planar_base(spec: mujoco.MjSpec) -> None:
    """(x, y, yaw) on `base_link` — innate's world.add_planar_base.

    Their reason, kept verbatim because it is the whole design: "a wheeled
    chassis can't pitch, and a free joint lets the arm's reaction torque tip
    the 0.89 kg base over". It is also what makes MARS a body that cannot
    fall, which is why the conformance suite's topple case is legged-only.
    """
    base = spec.body(BASE_BODY)
    for name, jtype, axis in (
        (BASE_JOINTS[0], mujoco.mjtJoint.mjJNT_SLIDE, (1, 0, 0)),
        (BASE_JOINTS[1], mujoco.mjtJoint.mjJNT_SLIDE, (0, 1, 0)),
        (BASE_JOINTS[2], mujoco.mjtJoint.mjJNT_HINGE, (0, 0, 1)),
    ):
        joint = base.add_joint()
        joint.name = name
        joint.type = jtype
        joint.axis = axis


def tune_contacts(spec: mujoco.MjSpec) -> None:
    """innate's world.tune_contacts: everything the URDF cannot say.

    The collision SHAPES are all in mars.urdf (one description their browser
    viewer draws from too). What is set here is MuJoCo's contact model and the
    finger servo: frictionless wheels, a grasp contact model on the blades,
    the finger pair excluded (their hub pins overlap ~1 mm at joint6 = 0, and
    `arm.srdf` disables the same pair for MoveIt), and the gripper's range
    clamped to the real hard stop.
    """
    for name in WHEEL_GEOMS:
        wheel = spec.geom(name)
        wheel.condim = 1
        # Else the floor's condim 3 wins the pair and the friction is back.
        wheel.priority = 1

    for link in FINGER_LINKS:
        blades = [g for g in spec.body(link).geoms if g.contype]
        if not blades:
            raise RuntimeError(
                f"{link}: no collision geometry in mars.urdf — the finger "
                "blades are what a grasp is made of; check ASSETS' revision")
        for geom in blades:
            geom.priority = 2        # the finger's params govern every pair
            geom.condim = FINGER_CONDIM
            geom.friction = FINGER_FRICTION
            geom.solref = FINGER_SOLREF
            geom.solimp = FINGER_SOLIMP

    spec.add_exclude(bodyname1=FINGER_LINKS[0], bodyname2=FINGER_LINKS[1])

    mimic_name, source_name, _mult = MIMIC_JOINT
    j6 = spec.joint(source_name)
    j6m = spec.joint(mimic_name)
    j6.range = [GRIPPER_CLOSED_ON_AIR_RAD, j6.range[1]]
    j6m.range = [j6m.range[0], -GRIPPER_CLOSED_ON_AIR_RAD]
    for name in (source_name, mimic_name):
        joint = spec.joint(name)
        # A hinge reads only the first entry; the URDF's arm-sized damping=5
        # would otherwise cap the close at EFFORT/5 rad/s.
        joint.damping = [FINGER_DAMPING, 0.0, 0.0]
        joint.armature = FINGER_ARMATURE


def robot_spec() -> mujoco.MjSpec:
    """One MARS as an `MjSpec`, ready to `attach` — innate's recipe, applied.

    The single definition of "a MARS" for the lab slot, the `/sim` world and
    (Phase 4) the training env, the way `g1.g1_spec()` is for the G1.
    """
    spec = load_robot_spec()
    spec.option.timestep = C.PHYSICS_DT
    add_planar_base(spec)
    tune_contacts(spec)
    return spec


def _home_qpos(model: mujoco.MjModel, prefix: str = "") -> np.ndarray:
    """The HOME pose as a full qpos vector: base at the origin, arm folded.

    Addressed by joint NAME, never by index. The planar base's three DoFs
    come first in qpos today, and writing `qpos[3:]` would be an assumption
    that a URDF revision could break in silence.
    """
    qpos = np.zeros(model.nq)
    for name, target in ARM_HOME.items():
        qpos[model.joint(prefix + name).qposadr[0]] = target
    mimic_name, source_name, mult = MIMIC_JOINT
    qpos[model.joint(prefix + mimic_name).qposadr[0]] = mult * ARM_HOME[source_name]
    return qpos


def _scene_spec(timestep: float | None = None) -> mujoco.MjSpec:
    """The standalone scene: one MARS, a floor, a light, a HOME keyframe.

    On the lab's `C.PHYSICS_DT` (5 ms) by default, not Innate's 2 ms: a
    `/sim` world is ONE model with one timestep, and this is it. MEASURED at
    both, the 2 s hold is identical to five decimal places (0.00344 rad), so
    the finger tuning survives the coarser step for a HOLD — which was the
    thing worth checking, since their contact model was tuned at 2 ms on a
    2e-5 inertia blade.

    **A GRASP is the case where it does not survive, and `timestep` is here
    because of it.** Phase 4b put the playroom's 4 cm / 20 g block between the
    blades, closed, lifted 10 cm and held 2 s at 16 spots in the shell
    (`scripts/probe_mars_pick.py --scripted`): **4/16 at 5 ms, 14/16 at
    2 ms**. What fails at 5 ms is not slip but EJECTION — the block leaves at
    0.08-6.6 m of travel in the two seconds after the lift — which is the
    coarse step's contact impulse, not the friction cone (the world's own
    pyramidal cone scores the same 14/16 at 2 ms). So `robots/mars_env.py`'s
    `pick` compiles this scene at 2 ms with decimation 20, and `reach`, which
    never closes on anything, keeps the world's 5 ms.
    """
    spec = robot_spec()
    # Innate's own option block (world.build_world_xml): implicitfast damps
    # the single-step impulse spikes contacts can produce, and elliptic cone
    # with impratio 10 is what stops a grasped object creeping out of a
    # closed claw (MuJoCo docs, "Preventing slip"). PHASE 3: a `/sim` world
    # composes every body into ONE model and `attach` keeps the PARENT's
    # options, so a MARS in a room runs on the world's block, not this one —
    # `world/compose.py` already special-cases the G1's the same way.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    if timestep is not None:
        spec.option.timestep = float(timestep)
    w = spec.worldbody
    light = w.add_light()
    light.pos = [0.0, 0.0, 3.0]
    light.dir = [0.0, 0.0, -1.0]
    floor = w.add_geom()
    floor.name = FLOOR_GEOM
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [10.0, 10.0, 0.1]
    # The wheels rest exactly on z = 0 (their centres are one radius up) and
    # the planar base pins z there, so there is no stand height to measure —
    # the one thing about a wheeled body that is simpler than a walker.
    #
    # The keyframe is sized from a SEPARATE compile of the robot, not from
    # this spec: `MjSpec.compile()` normalises the spec it is called on, and
    # the spec returned here is the one `to_xml()` and `attach()` consume. A
    # floor plane and a light add no DoFs, so the robot's own nq, nv and
    # joint addresses are the scene's.
    probe = robot_spec().compile()
    key = spec.add_key()
    key.name = HOME_KEY
    key.qpos = _home_qpos(probe)
    key.qvel = np.zeros(probe.nv)
    # No `key.ctrl`: nu == 0. Innate's arm is driven through `qfrc_applied`
    # (there is no `<actuator>` block in mars.urdf), and `arm_servo` is how.
    return spec


def scene_spec(timestep: float | None = None) -> mujoco.MjSpec:
    """The standalone scene as an editable `MjSpec`, for a task that adds to it.

    `robots/mars_env.py`'s `pick` needs a block in the scene and a 2 ms step
    (see `_scene_spec`), and a task's furniture does not belong in this
    module: what belongs here is the ROBOT and the one scene every body in
    the lab is entitled to. A caller that adds a free-jointed body must also
    widen the HOME keyframe's `qpos`, which is `MjSpec.key(HOME_KEY)`.
    """
    return _scene_spec(timestep)


def write_scene_xml(spec: mujoco.MjSpec, path: Path) -> Path:
    """Write `spec` to `path` atomically, and only when the content differs.

    `scene_xml`'s body, shared so a task scene gets the same two properties
    for free: the file lands beside the assets (so `meshdir` still resolves)
    and a vec-env worker can never import a half-written one (AGENTS.md,
    "Atomic writes and live imports").
    """
    xml = spec.to_xml()
    if not path.exists() or path.read_text() != xml:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(xml)
        os.replace(tmp, path)
    return path


def scene_xml() -> Path:
    """Path to the generated scene, written under the cache.

    Beside the assets so `meshdir` still resolves, and never into the
    downloaded files. Atomic (temp + `os.replace`) and rewritten only when
    the content differs — `g1.g1_scene_xml`'s pattern, for its reason: a
    vec-env worker must never import a half-written scene.
    """
    require_mars()
    return write_scene_xml(_scene_spec(), CACHE_DIR / SCENE_NAME)


@lru_cache(maxsize=1)
def model() -> mujoco.MjModel:
    """The compiled standalone scene. Cached: the meshes cost ~0.2 s."""
    return mujoco.MjModel.from_xml_path(str(scene_xml()))


# ---------------------------------------------------------------- the servo

def joint2_min_target(joint1_target: float, full_min: float) -> float:
    """joint2's target floor for a given joint1 target.

    innate core.joint2_min_target — the same piecewise ramp as
    `arm_control.cpp`'s `applyLimitsAndConvertToEncoder`, with the sim joint
    range's lower bound as the full limit. Ported because the guard is a real
    constraint on where a policy may put the arm: at joint1 inside the front
    arc, a folded joint2 sweeps the arm THROUGH the head.

    At ARM_HOME it is a no-op (joint1 = 1.445 is past the arc's edge, so the
    full range applies), which is why the hold test cannot be passing because
    of it.
    """
    if joint1_target < -1.35 or joint1_target >= 1.25:
        return full_min
    if joint1_target < -1.0:
        t = -(joint1_target + 1.0) / 0.35
    elif joint1_target < 1.0:
        t = 0.0
    else:
        t = (joint1_target - 1.0) / 0.25
    return JOINT2_GUARD_MIN + t * (full_min - JOINT2_GUARD_MIN)


def servo_addresses(model: mujoco.MjModel, prefix: str = "") -> dict[str, tuple[int, int]]:
    """{joint name: (qpos address, dof address)} for every driven joint.

    Raises `KeyError` on a name the model does not have, which is the point:
    a renamed link in a URDF revision fails here instead of servoing a joint
    that does not exist. Pass the result back into `arm_servo` to keep it out
    of a 50 Hz loop (Phase 3's driver will, the way `G1Walker` caches its
    addresses in `__init__`).
    """
    out: dict[str, tuple[int, int]] = {}
    for name in DRIVEN_JOINTS + (MIMIC_JOINT[0],):
        joint = model.joint(prefix + name)
        out[name] = (int(joint.qposadr[0]), int(joint.dofadr[0]))
    return out


def arm_servo(model: mujoco.MjModel, data: mujoco.MjData,
              targets: Mapping[str, float], prefix: str = "",
              *, sag: bool = False,
              adr: Mapping[str, tuple[int, int]] | None = None) -> None:
    """Hold the arm and head at `targets` — innate core._apply_control's
    position half, through `qfrc_applied`.

    KP 50 / KD 1 clamped at 50 N*m, the two gripper fingers on the real
    servo's 2 N*m with no velocity term, the mimic finger driven to
    -joint6's TARGET (not its measurement — their own choice: the geartrain
    is what mirrors, so a blade dragged off by a contact is not fought by the
    other blade), and joint2's floor re-clamped from joint1 every step, like
    the real node re-clamps every control cycle.

    `targets` names joints; anything it leaves out holds its ARM_HOME value,
    so a caller that only wants the gripper to close says so in one key.

    `sag=True` adds their structural-sag model (the LINK settles
    `gravity/STRUCT_STIFFNESS + backlash` below the servo angle). OFF by
    default because on the real robot the sag lives PAST the encoders and
    `/joint_states` reports the encoder side — until this harness models that
    reporting layer (Phase 4's obs), switching it on would make `qpos` — and
    therefore the observation — disagree with the arm by up to 0.055 rad
    while claiming to be more honest.

    The base drive is NOT here: Phase 3, `robots/mars_drive.py`.
    """
    adr = adr if adr is not None else servo_addresses(model, prefix)
    full = {**ARM_HOME, **targets}
    j2_min = joint2_min_target(full["joint1"],
                               float(model.joint(prefix + "joint2").range[0]))
    mimic_name, source_name, mult = MIMIC_JOINT
    for name in DRIVEN_JOINTS:
        qadr, dadr = adr[name]
        target = full[name]
        if name == "joint2":
            target = max(target, j2_min)
        if sag:
            bias = float(data.qfrc_bias[dadr])
            target -= (bias / STRUCT_STIFFNESS
                       + ARM_BACKLASH_RAD * math.tanh(bias / BACKLASH_TANH_NM))
        if name == source_name:
            kd, limit = KD_GRIPPER, GRIPPER_EFFORT_LIMIT
        else:
            kd, limit = KD_JOINT, EFFORT_LIMIT
        torque = KP_JOINT * (target - data.qpos[qadr]) - kd * data.qvel[dadr]
        data.qfrc_applied[dadr] = min(max(torque, -limit), limit)
    mq, md = adr[mimic_name]
    torque = (KP_JOINT * (mult * full[source_name] - data.qpos[mq])
              - KD_GRIPPER * data.qvel[md])
    data.qfrc_applied[md] = min(max(torque, -GRIPPER_EFFORT_LIMIT),
                                GRIPPER_EFFORT_LIMIT)


# --------------------------------------------------------------- the viewer

def style_visual_geoms(model: mujoco.MjModel, prefix: str = "") -> None:
    """innate's world.style_robot_geoms, applied to a model about to be
    DUMPED rather than rendered: orange arm links, charcoal instead of matt
    black, hidden frame markers, collision boxes swept into a hidden group.

    The lab ships a colour per geom and the browser paints it, so this is how
    a MARS arrives orange in the viewer without the viewer knowing anything
    about MARS (`docs/mars-roadmap.md` §6.4: a material `kind` per geom, not
    a component per robot).

    Robot geoms only, by `prefix`: everything else in a composed world owns
    the group it was built with, and sweeping those into the hidden group
    would erase them from every render.
    """
    for i in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 model.geom_bodyid[i]) or ""
        if prefix and not body.startswith(prefix):
            continue
        link = body[len(prefix):] if prefix else body
        if model.geom_contype[i] == 1:                 # a <collision> geom
            model.geom_group[i] = COLLISION_GROUP
        elif link in ORANGE_LINKS:
            model.geom_rgba[i] = BRIGHT_ORANGE
        elif link in HIDDEN_MARKER_LINKS:
            model.geom_rgba[i, 3] = 0.0
        elif float(model.geom_rgba[i, :3].mean()) < DARK_RGB_MEAN:
            model.geom_rgba[i, :3] = CHARCOAL


@lru_cache(maxsize=1)
def visual_scene() -> dict:
    """The viewer's mesh dump for one MARS. Cached: 9 meshes, ~7 MB of STL.

    Built from the robot alone (no floor, no light) so the body list is
    exactly what an `attach` into a room produces, and styled first so the
    colours travel with the geometry.

    The dump itself is `g1.extract_visual_scene` — generic apart from which
    geom group it reads, which is now an argument. It keeps the collision
    boxes and the three marker spheres out for free: it takes MESH geoms
    only, and every one of those in this URDF is a `<visual>`.
    """
    from .g1 import extract_visual_scene
    m = robot_spec().compile()
    style_visual_geoms(m)
    return extract_visual_scene(m, group=VISUAL_GROUP)


# ------------------------------------------------------------------ the body

class MarsBody(BodyBase):
    """MARS's half of the lab's `Body` contract — a wheeled body, not a spec.

    `BodyBase`, deliberately: no `foot_geoms`, no `fall_height`, no gyro, no
    twist ranges. Everything a walker would demand of it is either absent
    from the robot or means something else, and inheriting those fields is
    exactly how a new body ends up sagging into another robot's proxy.

    Not a `@dataclass` of its own — like `MicroduckBody` and `G1Body` it adds
    no fields, only answers.
    """

    #: How far from the base origin a range return is MARS looking at itself
    #: (`FOOTPRINT_M`, measured there). Not a `Body` field: only a body with a
    #: planar scanner has anything to say about it, and the one consumer
    #: (`world/arena.WorldRobot`) reads it with a `getattr` default of 0.
    footprint_m = FOOTPRINT_M
    #: What a gripper READING means on this robot: the load past which the
    #: claw counts as holding (`HOLD_LOAD_NM`) and the servo's own torque
    #: clamp (`GRIPPER_EFFORT_LIMIT`), which is the full scale of any bar
    #: drawn from it.
    #:
    #: Declared here for `footprint_m`'s reason, and it is the whole point of
    #: the pair being attributes rather than an import: the `/sim` inspector's
    #: load bar needs both numbers, and reading them out of `robots/mars.py`
    #: inside `world_server.tof_payload` would put a robot's name back in the
    #: frame builder — the one hack `docs/mars-roadmap.md` §6.5 exists to
    #: delete. A body with no claw declares neither and `WorldRobot` answers
    #: None, so the frame simply carries no `gripper` block.
    hold_load_nm = HOLD_LOAD_NM
    gripper_limit_nm = GRIPPER_EFFORT_LIMIT

    # 6 joint targets + (vx, wz). `BodyBase`'s default is one action per
    # joint, which is right for a body whose actions ARE its joints.
    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    def contract(self) -> PolicyContract:
        """A CODE SKILL contract, and untested on hardware — it says both.

        The vocabulary matters here (`docs/mars-roadmap.md` §1): in Innate's
        stack a *learned skill* is an ACT checkpoint trained from teleop
        demonstrations with both cameras in, at 25 Hz. A policy trained in
        this lab is NOT one of those, and calling it one would promise a
        provenance it does not have. What it is is a *code skill* — their own
        Python plugin, running onnxruntime and streaming
        `/mars/arm/commands` + `/cmd_vel` — so that is what `deploy` says,
        with "untested on hardware" in the same sentence because nothing in
        this repo has ever driven a MARS.

        Declared before `MarsArmEnv` existed, for the reason the layout above
        is: a body has to speak one contract from the day it is listed. Phase
        4a's exports carry exactly this record, stamped into the ONNX's own
        `metadata_props`, and `tests/test_mars_env.py` reads it back off a
        trained file.

        **`deploy` names the ACTION MAP, and it has to.** The eight floats a
        MARS policy emits are +-1 under every map this lab trains — what
        changes is whether the six arm floats are a joint POSITION about
        ARM_HOME or an INCREMENT on a commanded target (`mars_env`'s
        `ACTION_MODES`; Phase 4a-2 measured the second one into the default).
        No part of an ONNX graph says which, the numbers look identical
        either way, and a consumer that holds a rate or integrates a position
        drives a different controller. The sentence is built from
        `mars_env`'s own constants rather than typed, so the file and the
        code cannot drift; a run that overrode the default records it in
        `run.json`'s `env_kwargs`, which is the one place to check.
        """
        from .mars_env import DEFAULT_ACTION_MODE, action_map_sentence
        return declare(
            self,
            id=CONTRACT_ID,
            rate_hz=CONTROL_HZ,
            slots=_obs_slots(),
            deploy="code skill: an Innate Python skill running the ONNX and "
                   "streaming /mars/arm/commands + /cmd_vel at 25 Hz — "
                   "untested on hardware. Clip the output to +-1 first "
                   "(nothing clips at inference and the policy never saw an "
                   "unclipped value); then actions[6:8] are (vx m/s, "
                   "wz rad/s), zero for arm-only tasks, and actions[0:6] are "
                   f"— under this lab's DEFAULT map, {DEFAULT_ACTION_MODE} — "
                   f"{action_map_sentence(DEFAULT_ACTION_MODE)}. A run that "
                   "trained a different map says so in its run.json "
                   "env_kwargs.action_mode; check there before deploying one")

    # ------------------------------------------------------------- assets

    def ready(self) -> bool:
        """All 11 downloaded files present.

        Not the base class's "is the training scene on disk": `scene_fn`
        GENERATES that scene, so the generic answer would be False until
        something asked for it and then True forever.
        """
        return mars_ready()

    def fetch(self) -> Path:
        return fetch()

    # ------------------------------------------------------------- viewer

    def visual_scene(self) -> dict:
        return visual_scene()

    def look(self) -> str:
        """The viewer's material set. The colours are already in the dump
        (`style_visual_geoms`); this names the table the browser adds gloss,
        metalness and the wheels' rubber with — Phase 1b/viewer work."""
        return "mars"

    # ----------------------------------------------------------- training

    def env_class(self, task: str = "walk") -> type:
        """The env that trains `task` on MARS. Phase 4a: `reach`; 4b: `pick`.

        `MarsArmEnv` is gymnasium and its OWN base class, not
        `MicroduckWalkEnv` — see `robots/mars_env.py`. A body that quietly
        returned the duck's env would train an arm against foot-contact
        rewards, which is why the unknown-task branch names what there is
        rather than falling back to anything.

        Imported lazily, like the G1's: a machine with no MARS assets must
        still be able to train the duck.

        **One class PER TASK, not one class with a task argument.**
        `train.make_env` uses `--task` to choose the class and then constructs
        it with the shared kwargs, so a body that answers the same class for
        two tasks trains whichever one the constructor defaults to — which is
        what a `--task pick` run did before `MarsPickEnv` existed.
        """
        from .mars_env import TASKS, MarsArmEnv, MarsPickEnv
        if task == "pick":
            return MarsPickEnv
        if task == "reach":
            return MarsArmEnv
        raise SystemExit(
            f"unknown --task {task!r} for {self.id} (have: "
            f"{', '.join(TASKS)}) — MARS does not walk, and `place` is a "
            "later rung of docs/mars-roadmap.md Phase 4")

    def shipped_policies(self) -> tuple[dict, ...]:
        """MARS ships no policy this lab can run.

        Innate's learned skills are ACT checkpoints trained from teleop
        demonstrations on their cloud — not ONNX, not a 32-float
        observation — and `innate-os` vendors none of them. So the palette
        shows no MARS group, and the conformance suite's shipped-idle hold
        skips this body and holds it with `arm_servo` instead.
        """
        return ()

    def train_env_kwargs(self, args) -> dict:
        """MARS's per-body knobs: the arm action map, and `pick`'s rung.

        `--actuator bam` is the flag that must not be allowed through, and it
        is REFUSED rather than dropped: joints 4-6 and the head are XL330s,
        which is the servo BAM was identified on, while joints 1-3 are
        XL430/XC430 with no fit here, so v1 drives every joint with Innate's
        own PD (`docs/mars-roadmap.md` §6.6 — per-joint servo models are the
        fix). Pretending one switch covers the arm is the thing to avoid; so
        is accepting a flag and silently discarding it, which is AGENTS.md's
        rule 0 ("a knob that changes nothing is broken, not null"). The G1
        refuses the same flag in the same place.

        `--action-mode` is here rather than in a new environment variable
        because it belongs in the run's `env_kwargs` (§6.8: new per-body knobs
        go in the recipe's `env` dict and the run's record, both of which
        already exist and are recorded). What an action MEANS is not
        recoverable from a policy file, so it has to be on the run.

        Left out entirely when the flag is, so the env's own
        `DEFAULT_ACTION_MODE` stays the single definition of the default and
        a run that did not ask is not recorded as having asked.

        **`MICRODUCK_MARS_PICK_RUNG` is an ENVIRONMENT variable and not a
        flag, and that is the one difference from `--action-mode` above.** It
        is a CURRICULUM knob: the lab's teach job runs a `Behavior`'s stages
        by exporting each `CurriculumStage.env` dict into the trainer
        subprocess, and it does not rewrite argv — which is exactly how the
        G1's `MICRODUCK_G1_COMMAND_MIX` reaches `command_mix`. It still lands
        in `run.json`'s `env_kwargs` from here, so the rung a run trained on
        is recorded whether the lab or a shell set it.
        """
        if getattr(args, "actuator", None) == "bam":
            raise SystemExit(
                "--actuator bam is this repo's XL330 identification — it "
                "describes MARS's joints 4-6 and head but NOT its XL430/XC430 "
                "shoulder (joints 1-3), so v1 trains every joint on Innate's "
                "own position PD (docs/mars-roadmap.md §6.6)")
        kw: dict = {}
        mode = getattr(args, "action_mode", None)
        if mode:
            from .mars_env import RUNG_SCALE_RAD
            if mode == "rung":
                # The CLI's fourth name is not a fourth MAP: it is `absolute`
                # in a narrower box, which is the one combination worth naming
                # because Phase 4a's plan named it. `RUNG_SCALE_RAD` carries
                # the measurement that refuted it.
                kw["action_mode"] = "absolute"
                kw["action_scale_rad"] = RUNG_SCALE_RAD
            else:
                kw["action_mode"] = mode
        if getattr(args, "task", None) == "pick":
            from .mars_env import PICK_RUNGS
            raw = os.environ.get("MICRODUCK_MARS_PICK_RUNG")
            if raw:
                try:
                    rung = int(raw)
                except ValueError as e:
                    raise SystemExit(
                        f"MICRODUCK_MARS_PICK_RUNG={raw!r} is not an integer; "
                        f"the rungs are {', '.join(str(r) for r in PICK_RUNGS)}"
                    ) from e
                if rung not in PICK_RUNGS:
                    raise SystemExit(
                        f"MICRODUCK_MARS_PICK_RUNG={rung} is not a rung of "
                        f"the pick ladder ({', '.join(str(r) for r in PICK_RUNGS)}"
                        ") — a spawn box nobody measured is not a curriculum")
                kw["pick_rung"] = rung
        return kw

    # ---------------------------------------------------------- /sim world

    def attach(self, spec, prefix: str, frame) -> None:
        """Put one MARS in a world model under `prefix`.

        The child's `<option>` is written to the PARENT's timestep first.
        `MjSpec.attach` keeps the parent's option block either way — that is
        documented in `world/compose.compose` and measured — but it WARNS when
        the two disagree, and once rooms carry their own clock
        (`Scenario.physics_dt`: a MARS room is 2 ms) every compose of one
        printed a `UserWarning` about a value that was never in doubt. Saying
        it explicitly is also the more honest spelling: the room's clock is
        the robot's clock.
        """
        robot = robot_spec()
        robot.option.timestep = spec.option.timestep
        spec.attach(robot, prefix=prefix, frame=frame)

    def driver(self, model, prefix: str):
        """One `MarsDriver` on this model — `robots/mars_drive.py`.

        Innate's base velocity PD (KP forward 200 / lateral 40 / yaw 3),
        station keeping after 0.4 s of quiet, the 0.5 s `cmd_vel` watchdog
        and `arm_servo`, with their constants named. `G1Walker`'s slot in the
        contract, filled by a controller rather than an ONNX session, which
        is the whole reason a wheeled body moves in a room before anything is
        trained.

        Imported here rather than at module level: `mars_drive` imports this
        module for the joint names and the servo, and a top-level import
        would be a cycle.
        """
        from .mars_drive import MarsDriver
        return MarsDriver(model, prefix)

    def frames(self) -> RobotFrames:
        """MARS's root link and its one pitching joint, for `/sim`.

        `base_link` is the URDF's root and carries the planar base's three
        DoFs (`add_planar_base`), so it is the body every other body of a MARS
        hangs off and the subtree a composed world slices out.

        **The sign is -1 and it is read off the two models, not chosen.** The
        duck's `head_pitch` turns about +y (`contract.py`'s axis table), so a
        POSITIVE command looks DOWN — which is what every brain's `head_down`
        constant means. `joint_head` in mars.urdf is `axis="0 -1 0"`, so a
        positive angle here looks UP. Hence `joint_head = -head_pitch`, and a
        brain that asks to look down looks down. Getting it backwards is a
        silent failure: the arm still moves, the servo still tracks, and the
        camera looks at the ceiling.

        The duck's other three gaze slots have no joint on this robot and are
        dropped by `world/arena.WorldRobot.set_cmd`, which documents what that
        costs: `neck_pitch` is the first joint of a two-joint gaze chain whose
        combined depression is measured on the duck, and MARS has one pitch
        DoF; `head_yaw` and `head_roll` do not exist here at all, because MARS
        turns its whole base to look sideways.
        """
        return RobotFrames(base=BASE_BODY, head_pitch_joint=HEAD_JOINT,
                           head_pitch_sign=-1.0)

    def make_sensors(self, model, prefix: str, *, presets, targets, seed) -> dict:
        """MARS's two sense channels: the 360-degree lidar and the head camera.

        `robots/body.py`'s hook, for the reason it exists: the arena used to
        mount sensors by the DUCK's site names, and MARS has neither a `tof`
        site nor a `head_camera` site. What it has is a `base_laser` frame on
        the chassis lid and a `head_camera_left` link on the pitching head.

        * **lidar** — `sensors/lidar.LidarSensor` at the device's own 6 Hz,
          0.15-6 m, 360 rays. Two arguments are not defaults and both are
          measured (that module's docstring): `base_body` so every frame
          carries the 76 mm the laser sits behind the base origin, and the
          exclusion that the sensor resolves to the mount's PARENT — without
          it all 360 rays return 0.04-0.10 m off the turret box that models
          the scanner's own housing.
        * **detector** — `sensors/detector.Detector` on the camera BODY (the
          URDF frame is x-forward, so it is the detector's own convention),
          with Innate's calibrated lens rather than `DetectorSpec.from_env()`:
          `MICRODUCK_CAMERA` is a knob for the DUCK's sensor variants and a
          battery that sets it is asking a question about the duck's camera,
          not about this one.

        The scenario's `tof` field is the entry's RANGE-SENSOR preset
        (`world/scenario.Duck`'s docstring), so on MARS it names the LIDAR's
        noise — which is why a MARS written with `"tof": null` has no range
        sensor and the lab falls its brain back to `script`, exactly as a
        blind duck does. `LidarNoise.datasheet` is a PLACEHOLDER shape and
        says so in its own docstring; nothing is tuned against it.
        """
        from ..sensors import Detector, DetectorNoise, DetectorSpec, LidarNoise, LidarSensor
        out: dict = {}
        lidar_preset = presets.get("tof")
        if lidar_preset is not None:
            out["lidar"] = LidarSensor(
                model, prefix + LIDAR_SITE, noise=LidarNoise.preset(lidar_preset),
                seed=seed(), base_body=prefix + BASE_BODY)
        det_preset = presets.get("detector")
        if det_preset is not None:
            out["detector"] = Detector(
                model, body=prefix + CAMERA_BODY,
                spec=DetectorSpec(fov_h_deg=CAMERA_HFOV_DEG,
                                  fov_v_deg=CAMERA_VFOV_DEG,
                                  max_range_m=CAMERA_MAX_RANGE_M,
                                  rate_hz=CAMERA_RATE_HZ),
                noise=DetectorNoise.preset(det_preset), targets=targets,
                seed=seed())
        return out


MARS = MarsBody(
    id="mars",
    title="Innate MARS",
    noun="MARS",
    kind="wheeled",
    joint_names=ARM_JOINTS,
    joint_groups=JOINT_GROUPS,
    default_pose=DEFAULT_POSE,
    obs_dim=OBS_DIM,
    lab_spacing_m=LAB_SPACING_M,
    scene_fn=scene_xml,
    stand_keyframe=HOME_KEY,
    # A lab slot with no task named runs `reach`: MARS does not walk, so
    # `BodyBase`'s "walk" default would reach `env_class`'s raise for every
    # roster slot rather than for a typo. `reach` and not `pick` because it
    # is the cheaper scene (5 ms, no free body) and the honest starting rung
    # — the ⭐ in the panel goes to whichever a measurement earns
    # (docs/mars-roadmap.md §5, open decision 2).
    default_task="reach",
    # A room with a MARS in it that must GRASP runs at 2 ms, not the world's
    # 5 ms — the measurement is on `GRASP_PHYSICS_DT`, and declaring it here
    # is what lets `world/scenario.robot_physics_dt` build such a room without
    # naming this robot.
    physics_dt=GRASP_PHYSICS_DT,
)


__all__ = ["ARM_HOME", "ARM_JOINTS", "ASSETS", "BASE_BODY", "CACHE_DIR",
           "CAMERA_BODY", "CAMERA_HFOV_DEG", "CAMERA_VFOV_DEG", "CONTRACT_ID",
           "FOOTPRINT_M", "HEAD_JOINT", "INNATE_OS_SHA", "LIDAR_SITE", "MARS",
           "OBS_DIM", "MarsBody", "arm_servo",
           "fetch", "mars_ready", "model", "robot_spec", "scene_spec",
           "scene_xml", "servo_addresses", "visual_scene", "write_scene_xml"]
