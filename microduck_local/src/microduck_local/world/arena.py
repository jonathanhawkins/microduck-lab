"""`World`: N robots, their reflex policies and their sensors in one mjData.

The lab used to give every duck a private env (one mjData each, stepped in
turn). A world composes them into one model so a room costs one `mj_step`
per substep, ducks can collide with each other and with objects, and a
sensor on one duck can see another. Per duck it reproduces exactly what the
walk env does at inference time — the 61-obs build (joint_vel lagged one
step, no noise), the `DEFAULT_POSE + clip(action)` actuator write, the fall
test — which `tests/test_arena.py` locks step-for-step against
`MicroduckWalkEnv`. What it deliberately does NOT do is rewards or domain
randomization: a world is for watching, driving and sensing, and for the
brain layer on top; reflex training keeps its own env.

Command semantics match the lab's `Duck.set_cmd`: the policy is
compass-blind, so a straight-ahead command closes a heading-hold loop on the
duck's measured yaw, the way the robot runtime would.

**Two kinds of body live in `World.ducks`**, and they are separate classes on
purpose (`docs/mars-roadmap.md` §6.5):

    WorldDuck   a POLICY-stepped body: 61 observations in, 14 actions out,
                once per 50 Hz control tick, plus the beak and the skill
                cycles. Sized by `contract.NUM_JOINTS` throughout.
    WorldRobot  a DRIVER-stepped body: `Body.driver()` is a controller, not a
                network, and it runs EVERY physics step. MARS's wheeled base
                is the first (`robots/mars_drive.py`).

Threading `if robot != "microduck"` through `WorldDuck` was the other option
and it is the shape of mistake this repo has paid for most (`AGENTS.md`,
"Retarget a term, don't delete it"): the duck's fields are a walker's —
`prev_joint_vel` for a one-step obs lag, `kp_base` for a kick window's
standing gain, `down_until` for a get-up — and a wheeled base answers none of
them. What the two share is the plumbing around the body and not the body:
odometry (`World._odom_step` reads `trunk_pos` and `yaw` off either), the bump
sense, the sensor poll, the brain hand-off. Those are functions of the World,
so they are shared by being written once there.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from .. import contract as C
from ..sensors import Detector, Target, TofSensor
from ..sensors.lidar import tof_from_lidar
from ..walk_env import MicroduckWalkEnv
from .compose import (
    DUCK_ROBOT,
    DuckAddress,
    compose,
    duck_prefix,
    mouth_frac_for_gape,
    mouth_target,
    spawn_duck,
)
from .scenario import PICKABLE_KINDS, Person, Scenario


def mars_joint_names(body) -> tuple[str, ...]:
    """The joint names a body's TELEMETRY reports, in its own order.

    `Body.joint_names` plus its gaze joint: for MARS that is the six arm
    joints and `joint_head`, which is exactly `mars.DRIVEN_JOINTS` and exactly
    what `MarsDriver.set_arm` accepts. Asked of the BODY rather than imported
    from one robot's module, so `WorldRobot.arm_qpos` names no robot.
    """
    head = getattr(body.frames(), "head_pitch_joint", None)
    names = tuple(body.joint_names)
    return names + ((head,) if head and head not in names else ())


def _body_of(robot_id: str):
    """The `Body` a scenario entry names (`robots/registry.get`).

    Imported inside the function because `robots/microduck.py` reaches back
    into `world/compose.py` for the attach sequence, and `robots/mars.py`'s
    driver is what steps a MARS here — a module-level import either way is a
    cycle. The registry is rebuilt per call and nothing in it is expensive
    (`robots/registry.registry`'s docstring), so this is a dict lookup.
    """
    from ..robots.registry import get
    return get(robot_id)

# The shipped ground-pick cycle (upstream microduck_ground_pick_env_cfg.py):
# a 4 s period encoded as [cos 2πφ, sin 2πφ, 0] in the twist slots; the beak
# tip bottoms out ~2 cm above the floor, ~8 cm ahead of the trunk (12.7 cm
# from a sagged, unpowered stand), for φ ∈ [0.2, 0.42] (measured in this
# world), and the runtime hands back to
# the walker at φ = 0.7. The mouth servo is scripted, outside the policy:
# here it closes at φ = 0.38, inside that window.
GROUND_PICK_PERIOD_S = 4.0
GROUND_PICK_CLOSE_PHI = 0.38
GROUND_PICK_END_PHI = 0.7
# How long the beak stays open after a drop (World._mouth). Long enough to read
# at 50 Hz in the viewer; the release itself is instant - it deactivates the
# weld, and a massless bill moving afterwards cannot push the toy.
MOUTH_DROP_S = 0.6
# The shipped kicks (ball_kick_left / ball_kick_right) run as a WINDOW, not a
# phase: the robot hands the reflex tier to the kick network for
# `kick_duration` (0.5 s in robotd's control.rs) with an all-zero command,
# then back to the walker. Same protocol here.
KICK_S = 0.5
# A kick ONNX whose sidecar says `"sensed": true` was trained with the BALL in
# its four head command slots (roadmap 12as, `behaviors/lastmetre.py`) instead
# of the all-zero block the vendored pair was trained on. This is the class the
# tracker must hold for it to have anything to say.
SENSED_BALL_CLS = "ball"
# …and the SIM-ONLY ABLATION that prices what the track costs it (roadmap
# 12as follow-up H). `MICRODUCK_SENSED_TRUTH=1` makes the four slots carry the
# recipe's own projection of the TRUE ball instead of the tracker's estimate of
# it, which is the upper bound a perfect tracker could ever hand this kick.
#
# THIS CAN NEVER SHIP AND IS NOT A KNOB TO LEAVE ON. The robot has no truth:
# there is no `qpos` for the ball on hardware, so an arm measured under this
# flag describes a world that does not exist. It is here to answer ONE
# question — how much of a sensed foot's in-play gap to its bench is the
# tracker's ~5.5 cm placement error (`brain/tracker.py::_place`) — and
# `World.sensed_truth` is printed at construction so a battery cannot run it
# by accident and quote the number as play.
SENSED_TRUTH_ENV = "MICRODUCK_SENSED_TRUTH"
# `=1` (or true/yes) swaps ALL FOUR slots — and therefore THREE things at once:
# where the ball is (placement), whether this tick saw it (`seen`), and how old
# the held estimate is (confidence). Follow-up H measured all three moving
# together and could not say which one the kick pays for, so the knob also
# takes the two SPLITS that separate them (roadmap 12as follow-up I):
#
#   xy     the truth path's PLACEMENT (slots 51/52, bearing and range off the
#          recipe's projection) with the TRACK's own freshness (53/54)
#   fresh  the TRACK's placement (51/52, off `Track.bearing_from`/`range_from`
#          exactly as the played path reads it) with the truth path's `seen`
#          and confidence (53/54)
#
# Both are composed from the same two producers `=1` and the played path
# already use — nothing is re-derived — so `xy` + `fresh` between them cover
# exactly the four slots `=1` swaps. Every mode is the same SIM-ONLY ablation
# and none of them can ship.
SENSED_TRUTH_MODES = ("all", "xy", "fresh")
SENSED_TRUTH_OFF = ("", "0", "false", "False")
SENSED_TRUTH_ALL = ("1", "true", "True", "yes")
# A goal this soon after a kick is the kick's; the rest were walked into.
# (Until 2026-09-06 the ball had no rolling resistance, so a chase at
# 0.45 m/s sent a bumped ball as far as a kick did - to the boards; now a
# kick at 1.4 m/s reaches a goal 1.5 m off in ~1.3 s and stops after
# 3.5 m, a walked-into ball after 0.7 m - see `Ball.rolling`.)
# `soccer_score` reports both counts; eval-pitch prints them.
KICK_GOAL_S = 4.0
# …and at the STANDING tuning: robotd's standing transition fires on that
# all-zero command, so the window runs at `standing_action_scale` (1.0 —
# the same whole action this world always applies) and the softened
# standing gain, `standing_gain_ratio` × the walking Kp (control.rs).
STANDING_GAIN_RATIO = 0.8
SKILLS = {"ground_pick": "alpha_ground_pick.onnx", "kick_left": "ball_kick_left.onnx",
          "kick_right": "ball_kick_right.onnx"}
PICK_REACH_AHEAD = 0.078     # where the tip lands, ahead of the trunk origin (m), standing on the walker
PICK_REACH_LEFT = 0.014      # …and a touch to the left (the beak is not on the centreline)
GRASP_TOL_XY = 0.04          # a toy centre within this of the tip can be grasped (the beak is ~2 cm wide)
GRASP_TOL_Z = 0.045

Infer = Callable[[np.ndarray], np.ndarray]


@dataclass
class OdomNoise:
    """How the (x, y, yaw) a brain gets drifts from the truth (roadmap 1.7).
    The robot's odometry is dead reckoning from leg kinematics + the IMU's
    yaw: distance is over/under-counted by a per-run scale (foot slip),
    yaw integrates a gyro bias, and both get per-step noise. The presets
    are ASSUMPTIONS in the absence of a measurement on the robot — the
    point is that a brain must survive them, not their exact size."""
    scale_sigma: float = 0.0        # per-run distance scale error, 1σ (fraction)
    yaw_bias_sigma: float = 0.0     # per-run gyro bias, 1σ (rad/s)
    step_sigma: float = 0.0         # per-step position noise, 1σ (m per m walked)
    yaw_step_sigma: float = 0.0     # per-step yaw noise, 1σ (rad per rad turned)

    @staticmethod
    def preset(name: str) -> "OdomNoise":
        if name == "ideal":
            return OdomNoise()
        if name == "datasheet":
            return OdomNoise(scale_sigma=0.03, yaw_bias_sigma=np.deg2rad(0.3), step_sigma=0.02, yaw_step_sigma=0.02)
        if name == "hostile":
            return OdomNoise(scale_sigma=0.08, yaw_bias_sigma=np.deg2rad(1.0), step_sigma=0.05, yaw_step_sigma=0.05)
        raise ValueError(f"unknown odom preset {name!r}")
_NEG_Z = np.array([0.0, 0.0, -1.0], np.float32)
_E_FWD = np.array([1.0, 0.0, 0.0])


def zero_infer(obs: np.ndarray) -> np.ndarray:
    return np.zeros(C.NUM_JOINTS, np.float32)


@dataclass
class WorldDuck:
    id: str
    adr: DuckAddress
    spawn: tuple[float, float, float]
    infer: Infer = zero_infer
    policy_id: str | None = None
    tof: TofSensor | None = None
    detector: Detector | None = None
    max_steps: int = int(round(30.0 / C.CTRL_DT))
    # Manipulation state (roadmap 12.2/12.3): what the beak holds, and the
    # skill cycle the reflex tier is running instead of the walker.
    holding: str | None = None
    beak_closed: bool = False
    # The 15th servo, as an opening fraction: 0 shut, 1 wide (World._mouth).
    # It is NOT in the obs contract and no policy writes it - on the robot the
    # mouth is the app's, driven by `RobotMouth`, and here it is the world's.
    mouth: float = 0.0
    mouth_open_until: float = 0.0
    # Dead-reckoned pose the brain gets (World.odom): the truth plus OdomNoise.
    odom_preset: str = "ideal"
    odom_noise: OdomNoise = field(default_factory=OdomNoise)
    odom_est: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _odom_true_prev: np.ndarray | None = None
    _odom_scale: float = 1.0
    _odom_yaw_bias: float = 0.0
    skill: str | None = None
    skill_t0: float = 0.0
    skill_infer: Infer | None = None
    skill_sensed: bool = False         # …and was it trained to SEE the ball (roadmap 12as)?
    kp_base: np.ndarray | None = None      # the model's actuator Kp for this duck, restored after a kick
    gain_ratio: float = 1.0
    grasp_attempts: int = 0
    grasp_successes: int = 0
    last_grasp_err: float | None = None   # xy distance tip→nearest toy at the last close (m)
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(C.NUM_JOINTS, np.float32))
    prev_joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(C.NUM_JOINTS, np.float32))
    twist_cmd: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    head_cmd: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))
    body_cmd: np.ndarray = field(default_factory=lambda: np.zeros(6, np.float32))
    falls: int = 0
    down_until: float = -1.0           # lying where it fell until then (World.getup_s); -1: up
    up_since: float = -1.0             # …when it first read upright again during a get-up; -1: not yet
    down_since: float = -1.0           # …and when this spell on the floor began, to price it; -1: up
    step_count: int = 0
    bumped_t: float = -1e9             # when a body of this duck last touched another duck or a person
    episodes: int = 0
    _hold_yaw: float | None = None

    @property
    def root_body(self) -> int:
        """The body every other body of this robot hangs off — the duck's
        `trunk_base`. The name a `World` loop uses when it must not care which
        KIND of body it holds (`duck_bodies`, `_geom_owner`); `WorldRobot` has
        the same property over its own root."""
        return self.adr.trunk_body

    # -- state readers (all straight off mjData, no caching) -------------------
    def trunk_quat(self, data: mujoco.MjData) -> np.ndarray:
        return data.xquat[self.adr.trunk_body]

    def trunk_pos(self, data: mujoco.MjData) -> np.ndarray:
        return data.xpos[self.adr.trunk_body]

    def projected_gravity(self, data: mujoco.MjData) -> np.ndarray:
        return C.quat_rotate_inverse(self.trunk_quat(data), _NEG_Z)

    def joint_vel(self, data: mujoco.MjData) -> np.ndarray:
        return data.qvel[self.adr.joint_qvel].astype(np.float32)

    def yaw(self, data: mujoco.MjData) -> float:
        q = data.qpos[self.adr.root_qpos + 3:self.adr.root_qpos + 7]
        return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                1 - 2 * (q[2] ** 2 + q[3] ** 2)))

    def heading_speed(self, data: mujoco.MjData) -> float:
        """Forward speed in the yaw-heading frame (the walk env's
        `heading_lin_vel`[0]): pays for covering ground, not for diving."""
        v = data.qvel[self.adr.root_qvel:self.adr.root_qvel + 3]
        R = data.xmat[self.adr.trunk_body].reshape(3, 3)
        fwd = R @ _E_FWD
        fwd[2] = 0.0
        n = float(np.linalg.norm(fwd))
        fwd = fwd / n if n > 1e-9 else _E_FWD
        return float(v @ fwd)

    def fallen(self, data: mujoco.MjData) -> bool:
        g = self.projected_gravity(data)
        return bool(g[2] > MicroduckWalkEnv.FALL_GRAVITY_Z
                    or self.trunk_pos(data)[2] < MicroduckWalkEnv.FALL_HEIGHT)

    # -- the contract ----------------------------------------------------------
    def obs(self, data: mujoco.MjData) -> np.ndarray:
        """The 61-obs vector, in contract order, noise-free. Advances the
        one-step joint_vel lag, so call it exactly once per control step."""
        a = self.adr
        obs = np.empty(C.OBS_DIM, np.float32)
        obs[0:3] = data.sensordata[a.gyro_adr:a.gyro_adr + 3]
        obs[3:6] = self.projected_gravity(data)
        obs[6:20] = (data.qpos[a.joint_qpos] - C.DEFAULT_POSE).astype(np.float32)
        obs[20:34] = self.prev_joint_vel
        self.prev_joint_vel = self.joint_vel(data)
        obs[34:48] = self.last_action
        obs[48:51] = self.twist_cmd
        obs[51:55] = self.head_cmd
        obs[55:61] = self.body_cmd
        return obs

    def set_cmd(self, data: mujoco.MjData, twist, head=None) -> None:
        tw = np.asarray(twist, np.float32).copy()
        if tw[0] > 0.05 and abs(float(tw[2])) < 1e-6:
            yaw = self.yaw(data)
            if self._hold_yaw is None:
                self._hold_yaw = yaw
            err = yaw - self._hold_yaw
            err = float(np.arctan2(np.sin(err), np.cos(err)))
            tw[2] = float(np.clip(-4.0 * err, -1.0, 1.0))
        else:
            self._hold_yaw = None
        self.twist_cmd[:] = tw
        self.head_cmd[:] = 0.0 if head is None else np.asarray(head, np.float32)


class WorldRobot:
    """One driver-stepped body in a room — a MARS today (`robots/mars.py`).

    The counterpart of `WorldDuck` for a body whose reflex tier is a
    CONTROLLER rather than a policy. `docs/mars-roadmap.md` Phase 3 is why
    that is the cheap half of putting a wheeled base in a room: "for a
    differential-drive base the reflex tier IS the base controller, so a MARS
    drives on the day it is attached" — there is no gait to train first.

    **The cadences are two and they are not the same.** The COMMAND is 50 Hz,
    the duck's control tick, because that is when a brain decides and when the
    lab streams. The DRIVER runs every 5 ms physics step, because it must:
    MEASURED (`robots/mars_drive.py`'s docstring) at a 20 ms tick the forward
    velocity loop's gain is `KP_FORWARD * dt / m` = 3.04, past the explicit
    loop's bound of 2, and a 3 s run at 0.3 m/s ends going BACKWARDS at
    2.6 m/s having covered nothing. `World.step` therefore calls
    `driver.step(data)` inside its substep loop.

    **A command is a lease.** `MarsDriver` carries Innate's 0.5 s `cmd_vel`
    watchdog, so unlike a duck's `twist_cmd` — which persists until something
    overwrites it — a MARS that stops being commanded STOPS. That is the
    robot's own behaviour and it is kept: a brain that dies, or a socket that
    drops, leaves the base still rather than driving at the last thing it was
    told. `set_cmd` stamps `data.time`, so the world's clock is the
    watchdog's and a slow host makes a robot late, never runaway.

    **No heading hold.** `WorldDuck.set_cmd` closes a yaw loop on the duck's
    measured yaw because the shipped walker is compass-blind and drifts.
    MARS's drive is a body-frame velocity PD on a planar base reading its own
    yaw rate, and MEASURED it does not need one: 0.3 m/s for 5 s ends 6.7 mm
    off the line at -0.0045 rad. Adding one would be a second loop around a
    loop that already tracks.

    **`fallen()` is always False.** The base is three planar DoFs by
    construction (`mars.add_planar_base`: "a wheeled chassis can't pitch, and
    a free joint lets the arm's reaction torque tip the 0.89 kg base over"),
    so there is no attitude to lose and the World's fall / get-up / respawn
    machinery never fires on one. `falls` stays 0 for the life of the run,
    which is what the frame reports.
    """

    # The duck's gaze intent is gated: the shipped walker's observation
    # carries a head command block and it never trained with one, so a brain's
    # `Intent.head` only reaches a duck that opted in (`world_server`'s
    # `head_cmds`). A wheeled body has no such observation — its head is a
    # position servo outside every policy — so there is nothing to disturb and
    # the gaze is always applied.
    head_always = True

    def __init__(self, robot_id: str, spec, body, model: mujoco.MjModel,
                 sensors: dict | None = None, max_steps: int = 2 ** 62):
        self.id = robot_id
        self.robot = spec.robot
        self.body = body
        self.spawn = spec.spawn
        self.prefix = f"{robot_id}/"
        self.model = model
        self.driver = body.driver(model, self.prefix)
        # The body DECLARES its root link and its gaze joint
        # (`robots/body.RobotFrames`); the world asks. It is not a table here,
        # which is the pattern `docs/mars-roadmap.md` §6.5 exists to delete —
        # and it means a body from a pip-installed plugin needs no edit to
        # this file to stand in a room.
        self.frames = body.frames()
        self.root_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.prefix + self.frames.base)
        if self.root_body < 0:
            raise KeyError(
                f"{robot_id}: no body {self.prefix + self.frames.base!r} in the model — "
                f"{body.id!r}'s frames() names a root link this model does not have")
        #: Returns closer than this to the base origin are the robot looking
        #: at itself — `robots/mars.FOOTPRINT_M`, measured there. A body that
        #: has not declared one keeps every return.
        self.footprint_m = float(getattr(body, "footprint_m", 0.0))
        self.sensors: dict = dict(sensors or {})
        # The three channel names `Senses` has a field for. `tof` is None on
        # this body and stays None: a planar scan reaches the duck's ToF
        # brains through `sensors.lidar.tof_from_lidar`, which is the World's
        # job at the moment it builds the senses, not a second sensor here.
        self.tof = self.sensors.get("tof")
        self.detector = self.sensors.get("detector")
        self.lidar = self.sensors.get("lidar")
        # The duck-shaped surface the lab's frame builder, the inspector and
        # `record-world` read off every body in `World.ducks`. Each one is
        # either real for a wheeled body or a documented constant.
        self.policy_id: str | None = None      # no walker: `Body.driver` is the reflex tier
        self.falls = 0                         # a planar base cannot topple
        self.step_count = 0
        self.episodes = 0
        self.max_steps = int(max_steps)
        self.holding: str | None = None        # Phase 4: the gripper's own grasp
        self.beak_closed = False
        self.mouth = 0.0
        self.skill: str | None = None
        self.twist_cmd = np.zeros(3, np.float32)
        self.head_cmd = np.zeros(4, np.float32)
        self.bumped_t = -1e9
        self.down_until = -1.0
        self.up_since = -1.0
        self.down_since = -1.0
        self.last_action = np.zeros(body.num_actions, np.float32)
        self.prev_joint_vel = np.zeros(body.num_joints, np.float32)
        self._arm_cmd: dict[str, float] = {}   # what the last `Intent.arm` asked for
        # Odometry: `World._odom_step` and `_odom_reset` are body-agnostic —
        # they read `trunk_pos` and `yaw` — so a MARS gets the same dead
        # reckoning and the same drift presets a duck does, for free.
        self.odom_preset = spec.odom
        self.odom_noise = OdomNoise.preset(spec.odom)
        self.odom_est = np.zeros(3)
        self._odom_true_prev: np.ndarray | None = None
        self._odom_scale = 1.0
        self._odom_yaw_bias = 0.0
        self._joint_qvel = np.array(
            [int(model.joint(self.prefix + n).dofadr[0]) for n in body.joint_names])
        self._scene_bodies: list[str] | None = None

    # -- state readers ---------------------------------------------------------
    def trunk_pos(self, data: mujoco.MjData) -> np.ndarray:
        """Where the robot is. The base body's origin, which for a planar base
        is on the floor plane at the chassis's own reference point."""
        return data.xpos[self.root_body]

    def yaw(self, data: mujoco.MjData) -> float:
        """Heading, WRAPPED to +-pi.

        `MarsDriver.pose` returns the planar hinge's qpos, which accumulates —
        a robot that has turned twice reads past 2*pi. Wrapped here so that
        this is the same quantity `WorldDuck.yaw` is (a quaternion's heading),
        which is what `World._odom_step` and every brain reading
        `senses.odom[2]` expect."""
        yaw = self.driver.pose(data)[2]
        return float(np.arctan2(np.sin(yaw), np.cos(yaw)))

    def heading_speed(self, data: mujoco.MjData) -> float:
        """Forward speed in the base's own frame — `MarsDriver.velocity`'s
        first component, which is the quantity the drive PD regulates, so
        what the lab shows and what the controller sees cannot disagree."""
        return float(self.driver.velocity(data)[0])

    def joint_vel(self, data: mujoco.MjData) -> np.ndarray:
        """The arm's joint velocities. Only the World's respawn path reads
        this (to seed `prev_joint_vel`); a wheeled body has no obs lag."""
        return data.qvel[self._joint_qvel].astype(np.float32)

    def fallen(self, data: mujoco.MjData) -> bool:
        return False

    # -- commands --------------------------------------------------------------
    def set_cmd(self, data: mujoco.MjData, twist, head=None) -> None:
        """One 50 Hz command: a body-frame twist, and optionally a gaze.

        `WorldDuck.set_cmd`'s signature, so `world_server.drive` and every
        test drive a MARS with the same call. Only `vx` and `wz` reach the
        base — a differential-drive chassis cannot strafe, so a brain's `vy`
        is dropped rather than quietly turned into something.

        The gaze maps ONE component: the duck's `head_pitch` (intent slot 1)
        onto MARS's `joint_head`, with a SIGN FLIP that is read off the two
        models and not guessed. The duck's `head_pitch` turns about +y
        (`contract.py`'s axis table) so positive is looking DOWN, which is
        what every brain's `head_down` constant means. MARS's `joint_head` has
        `axis="0 -1 0"` in mars.urdf, so positive is looking UP. Hence
        `joint_head = -head_pitch`, and `MarsDriver.set_arm` clamps it to the
        joint's own MJCF range (the URDF's +-0.3491 rad, +-20 degrees) — so a
        brain asking for the duck's 0.6 rad look-down gets all 0.349 rad MARS
        has and nothing breaks.

        The duck's OTHER three slots are dropped, and that is a real gap
        rather than a rounding: `neck_pitch` is the first joint of a two-joint
        gaze chain whose combined depression is measured on the duck
        (`brain/controllers.py`: 0.75 + 0.43 per unit), and MARS has one pitch
        DoF. Summing them would need a gain nobody has measured on this robot.
        `head_yaw` and `head_roll` have no joint at all here — MARS turns its
        whole base to look sideways.
        """
        tw = np.asarray(twist, np.float32)
        self.twist_cmd[:] = tw[:3]
        self.driver.set_cmd(float(tw[0]), float(tw[2]), float(data.time))
        self.head_cmd[:] = 0.0 if head is None else np.asarray(head, np.float32)
        self._push_arm()

    def set_arm(self, targets) -> None:
        """A brain's `Intent.arm`: joint position targets by name.

        ABSOLUTE, not a nudge, because `MarsDriver.set_arm` is — a joint the
        mapping leaves out holds `ARM_HOME`. So this REPLACES what the last
        `Intent.arm` asked for rather than merging with it, and a brain that
        wants two joints held says both every tick."""
        self._arm_cmd = {str(k): float(v) for k, v in dict(targets).items()}
        self._push_arm()

    def _push_arm(self) -> None:
        """Write the arm AND the gaze in one call — the reason this exists.

        `MarsDriver.set_arm` is absolute over the whole driven set, so two
        independent writers (a brain's `Intent.arm` and the head component of
        its `Intent.head`) would each silently return the other's joints to
        `ARM_HOME` — a gaze intent every tick would have pinned the arm home
        for as long as a brain looked anywhere. One writer, one union, and the
        explicit arm mapping WINS when it names the head joint itself: a brain
        that commands `joint_head` directly means it, and the gaze is the
        fallback for one that only says "look down".
        """
        targets = dict(self._arm_cmd)
        joint = self.frames.head_pitch_joint
        if joint is not None and joint not in targets:
            targets[joint] = self.frames.head_pitch_sign * float(self.head_cmd[1])
        self.driver.set_arm(targets)

    def arm_targets(self) -> dict[str, float]:
        """The targets as the driver clamped and stored them."""
        return self.driver.arm_targets()

    # -- lifecycle -------------------------------------------------------------
    def spawn_at(self, data: mujoco.MjData, x: float, y: float, yaw: float) -> None:
        """`spawn_duck`'s job for this body: HOME pose, at rest, applied
        forces cleared. Delegated to the driver, which owns both force
        channels and must not leave the last step's drive push behind."""
        self.driver.spawn(data, x, y, yaw)
        self.step_count = 0
        self.last_action[:] = 0.0
        self.head_cmd[:] = 0.0
        self.twist_cmd[:] = 0.0
        self._arm_cmd = {}
        self.holding = None

    def arm_qpos(self, data: mujoco.MjData) -> dict[str, float] | None:
        """The ACHIEVED joint positions, by name — `Senses.arm`.

        What Innate's `/mars/arm/state` publishes, and NOT what `arm_targets`
        returns: the servo carries their structural compliance and backlash,
        so a commanded pose arrives with a few hundredths of a radian of sag
        (MEASURED at the pick pose: 28.5 mm of claw height). A brain that
        wants the claw somewhere has to read this.

        None for a body whose driver has no joint table, the way `held_body`
        answers -1 for one with no claw.
        """
        adr = getattr(self.driver, "adr", None)
        if not adr:
            return None
        # The DRIVEN set only. `servo_addresses` also carries the mirrored
        # finger (`mars.MIMIC_JOINT`), which is a constraint and not a
        # commandable joint — Innate's own `/mars/arm/state` does not report
        # it and `Intent.arm` refuses it, so neither does this.
        return {name: float(data.qpos[adr[name][0]]) for name in mars_joint_names(self.body)
                if name in adr}

    def held_body(self, data: mujoco.MjData) -> int:
        """Which body this robot's gripper has hold of, or -1.

        Delegated to the driver, which owns the predicate
        (`MarsDriver.held_body`: |constraint torque at joint6| past
        `mars.HOLD_LOAD_NM` AND a blade contact with a body that is not part
        of the robot). A driver with no claw answers -1 the way a body with no
        beak ignores `Intent.beak` — the body vocabulary is per channel, and
        `World.sense_grip` turns this into the pickable id that
        `Senses.holding`, the events log and the tidy overlay all read.
        """
        fn = getattr(self.driver, "held_body", None)
        return -1 if fn is None else int(fn(data))

    def scene_bodies(self) -> list[str]:
        """The body names the viewer's scene lists, in ITS order, stripped of
        the prefix and with the world first.

        The stream maps a robot's pose list onto `GET /scene?robot=<id>`'s
        body list POSITIONALLY (`viz_server._robot_scene_to_env` resolves the
        same list by name), so the two have to agree body for body or the
        robot is drawn with its parts on the wrong joints — the exact failure
        `robots/mars.load_robot_spec` turned `fusestatic` off to prevent.

        Derived from THIS model's attached subtree rather than by calling
        `Body.visual_scene()`, which would parse 7 MB of STL to read a list of
        names. That the two agree is a claim, so it is a TEST and not a
        comment: `tests/test_world_robot.py` compares this list against
        `mars.visual_scene()["bodies"]` name for name, the way
        `tests/test_arena.py` compares the composed duck against
        `scene_model()`.
        """
        if self._scene_bodies is None:
            sub = [b for b in range(self.model.nbody)
                   if self.model.body_rootid[b] == self.root_body]
            self._scene_bodies = ["world"] + [
                self.model.body(b).name[len(self.prefix):] for b in sub]
        return self._scene_bodies

    def bodies_payload(self, data: mujoco.MjData) -> list[list[float]]:
        """Poses for `scene_bodies()`, world first as an identity pose — the
        shape `world_server`'s frame puts under a duck's `bodies`."""
        out = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
        for name in self.scene_bodies()[1:]:
            b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.prefix + name)
            p, q = data.xpos[b], data.xquat[b]
            out.append([round(float(v), 4) for v in (*p, *q)])
        return out


class WorldPerson:
    """A person the ducks can follow: a mocap capsule, or a Unitree G1.

    Capsule: infinite-mass mocap, slid along waypoints. G1: the Lucky Robots
    MJCF attached under this id's prefix, driven by walker.onnx with the same
    waypoint twist as a command. Detector class stays "person" either way.
    """

    def __init__(self, model: mujoco.MjModel, spec: Person):
        self.spec = spec
        self.id = spec.id
        self.robot = None
        if spec.kind == "g1":
            from ..robots.g1 import G1Walker
            self.robot = G1Walker(model, spec.id)
            self.body = self.robot.pelvis
            self.mocap = -1
        else:
            self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.id)
            self.mocap = int(model.body_mocapid[self.body])
        self.x, self.y, self.yaw = spec.pos[0], spec.pos[1], spec.yaw
        self.wp = 0
        # A private copy: G1 yield used to `pop` spec.path until one point
        # remained, then the robot stood still for the rest of the session.
        self.route = list(spec.path)
        self.cmd: np.ndarray | None = None      # possessed: heading-frame twist
        self.possessed = False
        self.waiting = 0.0                       # s stood behind a duck in the way (polite walkers)
        self.yields = 0                          # waypoints given up on
    WAIT_S = 2.5                                 # ...before stepping on to the next waypoint

    def reset(self, data: mujoco.MjData) -> None:
        self.x, self.y, self.yaw = self.spec.pos[0], self.spec.pos[1], self.spec.yaw
        self.wp = 0
        self.route = list(self.spec.path)
        self.cmd = None
        self.waiting = 0.0
        self.yields = 0
        if self.robot is not None:
            self.robot.spawn(data, self.x, self.y, self.yaw)
        else:
            self.write(data)

    def write(self, data: mujoco.MjData) -> None:
        if self.mocap < 0:
            return
        data.mocap_pos[self.mocap] = [self.x, self.y, self.spec.height / 2]
        data.mocap_quat[self.mocap] = [np.cos(self.yaw / 2), 0.0, 0.0, np.sin(self.yaw / 2)]

    def sync(self, data: mujoco.MjData) -> None:
        """Read the G1 pelvis out of physics after mj_step. Capsules no-op."""
        if self.robot is None:
            return
        self.x, self.y, self.yaw = self.robot.pose(data)

    def fallen(self, data: mujoco.MjData) -> bool:
        return bool(self.robot is not None and self.robot.fallen(data))

    def blocked_by(self, blockers, target_yaw: float) -> bool:
        """A polite walker's rule: something (a duck's trunk) inside
        `yield_m` and within 70 degrees of the way it is about to walk."""
        r = self.spec.yield_m
        if r <= 0:
            return False
        for bx, by in blockers:
            dx, dy = bx - self.x, by - self.y
            if np.hypot(dx, dy) < r and abs(np.arctan2(np.sin(np.arctan2(dy, dx) - target_yaw),
                                                          np.cos(np.arctan2(dy, dx) - target_yaw))) < 1.22:
                return True
        return False

    def step(self, data: mujoco.MjData, dt: float, blockers=()) -> None:
        if self.robot is not None:
            self._step_g1(data, dt, blockers)
            return
        if self.possessed and self.cmd is not None:
            vx, vy, wz = (float(v) for v in self.cmd)
            self.yaw += wz * dt
            c, s_ = np.cos(self.yaw), np.sin(self.yaw)
            self.x += (vx * c - vy * s_) * dt
            self.y += (vx * s_ + vy * c) * dt
        elif self.spec.path and self.spec.speed > 0:
            tx, ty = self.spec.path[self.wp]
            dx, dy = tx - self.x, ty - self.y
            dist = float(np.hypot(dx, dy))
            if dist < 0.05:
                self.wp = (self.wp + 1) % len(self.spec.path)
            else:
                target_yaw = float(np.arctan2(dy, dx))
                err = float(np.arctan2(np.sin(target_yaw - self.yaw), np.cos(target_yaw - self.yaw)))
                if self.spec.yield_m > 0 and abs(err) > 1.0:
                    # A polite walker turns in place toward a new waypoint
                    # before stepping (a mocap walker arcs, through whatever
                    # is beside it).
                    self.yaw += float(np.clip(err, -1.5 * dt, 1.5 * dt))
                    self.write(data)
                    return
                if self.blocked_by(blockers, target_yaw) or self.blocked_by(blockers, self.yaw):
                    # Stand (facing the way it wants to go); give the
                    # waypoint up after a while - a person steps around.
                    self.yaw += float(np.clip(err, -1.5 * dt, 1.5 * dt))
                    self.waiting += dt
                    if self.waiting >= self.WAIT_S:
                        self.waiting = 0.0
                        self.yields += 1
                        if len(self.spec.path) > 1:
                            self.spec.path.pop(self.wp)
                            self.wp %= len(self.spec.path)
                        else:
                            self.wp = (self.wp + 1) % len(self.spec.path)
                    self.write(data)
                    return
                self.waiting = 0.0
                self.yaw += float(np.clip(err, -1.5 * dt, 1.5 * dt))
                step = min(self.spec.speed * dt, dist)
                self.x += step * np.cos(self.yaw)
                self.y += step * np.sin(self.yaw)
        self.write(data)

    def _step_g1(self, data: mujoco.MjData, dt: float, blockers) -> None:
        """Waypoint / possess twist → G1 walk policy. Pose comes from physics."""
        vx = vy = wz = 0.0
        if self.possessed and self.cmd is not None:
            vx, vy, wz = (float(v) for v in self.cmd)
        elif self.route and self.spec.speed > 0:
            tx, ty = self.route[self.wp]
            dx, dy = tx - self.x, ty - self.y
            dist = float(np.hypot(dx, dy))
            if dist < 0.25:
                self.wp = (self.wp + 1) % len(self.route)
            else:
                target_yaw = float(np.arctan2(dy, dx))
                err = float(np.arctan2(np.sin(target_yaw - self.yaw),
                                       np.cos(target_yaw - self.yaw)))
                wz = float(np.clip(err / max(dt, 1e-3), -1.0, 1.0))
                turning = abs(err) > 0.8
                blocked = self.spec.yield_m > 0 and (
                    self.blocked_by(blockers, target_yaw) or self.blocked_by(blockers, self.yaw))
                if blocked:
                    self.waiting += dt
                    self.yields += 1 if self.waiting == dt else 0
                    # Do NOT pop waypoints. A follower that stands at the G1's
                    # feet used to erase the whole tour, then the G1 idled.
                else:
                    self.waiting = 0.0
                if not turning and not blocked:
                    vx = float(self.spec.speed)
                elif blocked and self.waiting > self.WAIT_S:
                    # Duck has been in the way long enough — walk anyway at
                    # half speed rather than freeze the scene.
                    vx = float(self.spec.speed) * 0.5
        assert self.robot is not None
        self.robot.cmd[:] = (vx, vy, wz)
        self.robot.control(data)

    def payload(self, data: mujoco.MjData | None = None) -> dict:
        # Pose: capsule centre, or G1 pelvis. Viewer uses this for the
        # capsule fallback and the contact blob.
        if self.robot is None or data is None:
            pose = [round(self.x, 4), round(self.y, 4), round(self.spec.height / 2, 4),
                    round(float(np.cos(self.yaw / 2)), 4), 0.0, 0.0,
                    round(float(np.sin(self.yaw / 2)), 4)]
            bodies = None
        else:
            bodies = self.robot.bodies_payload(data, self.robot.scene_bodies)
            pose = bodies[1] if len(bodies) > 1 else [
                round(self.x, 4), round(self.y, 4), 0.79,
                round(float(np.cos(self.yaw / 2)), 4), 0.0, 0.0,
                round(float(np.sin(self.yaw / 2)), 4)]
        out = {"id": self.id, "kind": "person", "waiting": self.waiting > 0,
               "pose": pose, "possessed": self.possessed}
        if self.spec.kind != "capsule":
            out["robot"] = self.spec.kind
        if bodies is not None:
            out["bodies"] = bodies
        return out


class World:
    def __init__(self, scenario: Scenario, infer_for: dict[str, Infer] | None = None,
                 max_episode_s: float | None = None, seed: int | None = None, getup_s: float = 0.0,
                 ball_out_s: float = 0.0, getup_infer: Infer | None = None):
        # No episode timeout by default: a world is a place, not an episode
        # (a 30 s default once respawned a duck mid-delivery, toy and all).
        # Training envs pass their own horizon.
        max_episode_s = float("inf") if max_episode_s is None else max_episode_s
        self.scenario = scenario
        # A MICRODUCK_SKILL_<NAME> override naming a missing file is a typo
        # that used to disable the kick silently (start_skill refuses it, no
        # event): a battery then measured "the brain never kicks". Said here,
        # at build, before anything is measured (code review, 2026-09-08).
        for name in SKILLS:
            ov = os.environ.get(f"MICRODUCK_SKILL_{name.upper()}")
            if ov and not Path(ov).exists():
                raise FileNotFoundError(f"MICRODUCK_SKILL_{name.upper()}={ov!r}: no such file")
        # FALLS COST TIME (roadmap Track 4 s6 B.1, the second half): with
        # `getup_s` > 0 a fallen duck lies where it fell, on a zero command,
        # for that long before it is respawned - the stand-in for a get-up
        # policy (every RoboCup humanoid must recover unaided; ours has no
        # floor-to-stand yet), which on a robot costs 10-20 s. 0: the
        # respawn as it always was, and every number measured before this.
        self.getup_s = float(getup_s)
        # THE GET-UP, for real (roadmap B.1 / bead mdl-0ad). `getup_s` alone is
        # a stand-in: the fallen duck lies on a zero command for that long and
        # is then TELEPORTED upright. Give this a policy and the duck gets up
        # instead - it is driven by `getup_infer` while it is down, and the
        # moment it is upright again the walker has it back, with no teleport.
        # `getup_s` becomes the TIMEOUT: a duck that cannot make it up in that
        # long is respawned as before, so a stuck duck cannot stall a battery.
        # MEASURED (12 seeds a pose, honest BAM, obs noise + domain
        # randomisation + action delay, deterministic ONNX): the shipped
        # `alpha_stand.onnx` gets up from lying on its back, front and side
        # 100% of the time in 0.2-1.3 s, while the walker manages 0 of 24 -
        # which is why a fallen duck stays down today. So this is a CONTROLLER
        # SWITCH and not a policy that had to be trained.
        # None = the teleport stand-in, bit for bit every number measured
        # before this.
        self.getup_infer = getup_infer
        # …and it has to STAY up for this long before the walker gets it back.
        # Measured the hard way: with no dwell at all, one real fall in seed 0
        # of a 3v3 battery became TWENTY-FIVE counted falls, 0.1-0.3 s apart -
        # `fallen()` is a threshold on projected gravity and trunk height, and
        # a duck handed back to the walker the tick it first crosses that line
        # is still on its way up, so it drops straight back over it. The fall
        # count is what exposed it (3 falls with the teleport, 30 with the
        # get-up on the same 24 seeds); the diagnosis was reading the fall
        # TIMES, which were tenths of a second apart and so cannot be separate
        # topples. This is the get-up's own settle, not a metric patch.
        self.getup_hold_s = 0.3
        self.getups = 0                    # falls the duck got up from by itself
        self.getup_timeouts = 0            # …and falls that ran `getup_s` out and were respawned
        # …and how long each of those spells on the floor actually lasted, so a
        # battery can quote the PRICE of a fall rather than assume it (the bench
        # says 0.2-1.2 s to stand; `getup_s` is only the ceiling). One entry a
        # spell, get-ups and timeouts alike, in the order they finished.
        self.getup_down_s: list[float] = []
        # BALL OUT (roadmap Track 4 item 11b): what a referee does on a
        # walled table. A ball at rest within `ball_out_m` of the boards for
        # `ball_out_s` seconds is placed `ball_out_in` in from that wall
        # (velocity zeroed, `ball_outs` counted, `soccer_score["ballOuts"]`).
        # Measured, 12 seeds x 300 s of 3v3: the ball is at the boards 72%
        # of a run and no kick is ever taken there; with this, kicks 2.9 ->
        # 7.7 a run, dead-ball time -72 s, possession +8 s/min, progress
        # +0.19 (all p < 0.005), falls flat. 0 = off, bit for bit every
        # number measured before it; eval-pitch takes --ball-out-s and the
        # lab's pitch builtins turn it on (world_server).
        self.ball_out_s = float(ball_out_s)
        self.ball_out_m, self.ball_out_in = 0.20, 0.45
        # …and never placed ON a duck. The duck that was lining up on the ball
        # is standing about `kick_ahead` + `kick_side` (~0.12 m) from it when
        # the rule fires, so the spot the ball is moved to is exactly where a
        # body may be. A ball dropped inside one interpenetrates and the
        # solver flings both apart — the same failure `_clear_of_persons`
        # exists for on the respawn path (the physics audit's "fling").
        self.ball_out_clear = 0.25
        self._ball_rest_t0: float | None = None
        self.ball_outs = 0
        # …and a SEQUENCE beside it, so a harness can notice a throw-in the way
        # it notices a goal. `ball_outs` is a count nobody watched: the World
        # teleported the ball and told no brain (roadmap 12s).
        self.ball_out_seq = 0
        # THE SENSED KICK (roadmap 12as). Read once, at build, off the same
        # sidecars `kick_exits` aims with: a pair without the field is the
        # vendored blind pair and every number measured before this.
        self._skill_sensed = {n: World.skill_sensed(n) for n in SKILLS}
        self._sensed_kick = any(self._skill_sensed[n] for n in SKILLS if n.startswith("kick"))
        self._ball_trackers: dict = {}
        # The truth ablation, read ONCE here and never again (playbook rule 0:
        # the flag a battery thinks it is measuring has to be readable off the
        # World it built). Off is the only shipping value and costs one
        # attribute: no state is kept and no random number is drawn.
        self.sensed_truth_mode = World.sensed_truth_env_mode()
        self.sensed_truth = self.sensed_truth_mode != ""
        self._truth_sense: dict = {}
        self._truth_cam: dict = {}
        self._truth_seed = scenario.seed if seed is None else seed
        self._truth_rng = None
        if self.sensed_truth:
            what = {"all": "all four slots", "xy": "the PLACEMENT slots only (51/52); 53/54 stay the track's",
                    "fresh": "the FRESHNESS slots only (53/54); 51/52 stay the track's",
                    }[self.sensed_truth_mode]
            print(f"[world] {SENSED_TRUTH_ENV}={self.sensed_truth_mode} — the sensed kick reads the TRUE "
                  f"ball through the recipe's projection in {what}. SIM-ONLY ABLATION: the robot has no "
                  "truth, this can never ship.", flush=True)
        self.model = compose(scenario)
        # PHYSICS STEPS PER 50 Hz TICK, read off the compiled model rather
        # than from `C.DECIMATION`. The control tick is fixed — it is when a
        # brain decides, when the lab streams and what every sensor rate is
        # scheduled against — and the timestep underneath it is the room's
        # (`Scenario.physics_dt`: a MARS that must grasp needs 2 ms, and the
        # grasp table is in that field's block). At the default 5 ms this is
        # exactly `C.DECIMATION` = 4, which is what keeps `tests/test_arena.py`'s
        # step-for-step lock against the walk env exact.
        self.substeps = int(round(C.CTRL_DT / float(self.model.opt.timestep)))
        self.data = mujoco.MjData(self.model)
        self.t = 0.0
        self.tick = 0
        self.rng = np.random.default_rng(scenario.seed if seed is None else seed)
        infer_for = infer_for or {}
        self.ducks: dict[str, WorldDuck] = {}
        self.persons: dict[str, WorldPerson] = {
            p.id: WorldPerson(self.model, p) for p in scenario.persons}
        # What a detector can find: every duck's trunk, every ball, every person.
        # A duck's colour is its team's colorway — the one thing about another
        # duck a camera could really read (world/compose.paint_team paints it).
        # A non-duck body is NOT a detector target, and that is a stated gap
        # rather than an oversight: `DETECT_CLASSES` is the robot's own YOLO
        # class plus the sim-only ones this repo writes brains for, and
        # "a wheeled base with an arm" is not one of them. A duck's camera
        # therefore does not see a MARS; a MARS's LIDAR does see the duck,
        # geometrically, because a ray does not need a class. Giving it a
        # borrowed class would be worse than the gap — a `Target` on a body
        # this world has no id for reads `xpos[-1]`, the LAST body in the
        # model, and reports a detection of something somewhere else.
        targets = [Target(d.id, "duck", mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                                          f"{d.id}/trunk_base"), 0.10, color=d.team)
                   for d in scenario.ducks if d.robot == DUCK_ROBOT]
        targets += [Target(f"ball{i}", "ball", mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                                                f"ball{i}"), b.radius)
                    for i, b in enumerate(scenario.balls)]
        targets += [Target(p.id, "person", self.persons[p.id].body, p.radius, height=p.height) for p in scenario.persons]
        # A pitch's four goal POSTS, as landmarks a camera can classify
        # (roadmap Track 4 s6 C.2): the goal is a scored line with no
        # geometry, so they are fixed-position targets on the mouth line,
        # post-high, 5 cm round. What a self-localiser has to steer by.
        if scenario.goal_width > 0:
            gx, gw = scenario.floor[0] / 2 - 0.25, scenario.goal_width / 2   # the mouth line (goal_for) and half-width
            targets += [Target(f"post_{side}_{lr}", "post", -1, 0.05, pos=(sx * gx, sy * gw, 0.15))
                        for side, sx in (("right", 1.0), ("left", -1.0)) for lr, sy in (("l", 1.0), ("r", -1.0))]
        self.pickables: dict[str, int] = {
            t.id: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, t.id) for t in scenario.pickables}
        self.pickable_kind = {t.id: t.kind for t in scenario.pickables}
        # ...and the reverse, for `sense_grip`: a gripper reports the BODY it
        # has hold of and the rest of the world speaks toy ids.
        self._pickable_of_body: dict[int, str] = {b: t for t, b in self.pickables.items()}
        targets += [Target(t.id, "toy", self.pickables[t.id],
                           max(PICKABLE_KINDS[t.kind]["size"]) / 2) for t in scenario.pickables]
        self.basket = scenario.basket
        self.team_of = {d.id: d.team for d in scenario.ducks}
        # Soccer (first form): a pitch counts goals on both short walls.
        self.goal_width = float(scenario.goal_width)
        self.goals = {"left": 0, "right": 0}
        # A goal restarts play from a kickoff (below): this counter says one
        # happened, for the brains that must forget their plan; the hold
        # keeps every walker on a zero command until play restarts.
        self.goal_seq = 0
        self.last_goal: str | None = None
        self.kickoff_hold_s = 1.0
        self.kickoff_until = -1.0
        # THE GAME STATE (roadmap Track 4 s6 B.3). Every league runs a
        # GameController: after a goal the team that CONCEDED kicks off and
        # the other side stands off until the ball is in play. The World is
        # that controller here: `kickoff_team` (None for the first kickoff,
        # which is contested, as it always was), the ball's spot, and a
        # window; `game_state` is "set" during the hold, "kickoff" until the
        # ball has left the spot by `kickoff_moved_m` or `kickoff_free_s`
        # have passed, then "playing". Nothing here penalises a duck that
        # crosses early - the brains obey it (brain/team.py `Team.waits`,
        # `ChaseParams.kickoff_wait`).
        self.kickoff_team: str | None = None
        self.kickoff_free_s = 10.0
        self.kickoff_moved_m = 0.1
        self.kickoff_ball: tuple[float, float] | None = None
        self.last_kick_t = -1e9            # when a kick skill last started (attribution, KICK_GOAL_S)
        self.last_kick_duck: str | None = None    # …and which duck took it
        # Who the World says put the last goal in: the duck whose kick was
        # inside KICK_GOAL_S at the moment the ball crossed, else None (it
        # was walked in). Exactly the test `goals_kicked` / `goals_bumped`
        # splits on, recorded per goal so a caller that knows the ROSTER —
        # `PitchMetrics`, which owns the duck→team map — can turn it into
        # "this team scored" or "this team scored on itself" without
        # re-deriving the window and disagreeing with the split above.
        self.goal_credit_duck: str | None = None
        self.goals_kicked = 0
        self.goals_bumped = 0
        self._ball_joint: int | None = None
        if self.goal_width > 0 and scenario.balls:
            bb = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball0")
            for j in range(self.model.njnt):
                if int(self.model.jnt_bodyid[j]) == bb and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    self._ball_joint = j
        if scenario.basket is not None:
            targets.append(Target("basket", "basket",
                                  mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "basket_marker"), 0.12))
        self.skills: dict[str, Infer] = {}
        max_steps = (2**62 if not np.isfinite(max_episode_s)
                     else int(round(max_episode_s / C.CTRL_DT)))
        # A body mounts its OWN sensors (`robots/body.Body.make_sensors`): the
        # duck's `tof` and `head_camera` site names used to be spelled out
        # here, which is one of the two hacks `docs/mars-roadmap.md` §6.5
        # counts against world mode. The seed callable is what keeps the move
        # honest — the duck still draws its ToF seed before its detector's,
        # off this world's RNG, in this order, so every golden bit and every
        # seeded soccer number is the number it was.
        def next_seed() -> int:
            return int(self.rng.integers(0, 2**31 - 1))

        for d in scenario.ducks:
            body = _body_of(d.robot)
            sensors = body.make_sensors(
                self.model, duck_prefix(d.id),
                presets={"tof": d.tof, "detector": d.detector},
                targets=targets, seed=next_seed)
            if d.robot != DUCK_ROBOT:
                self.ducks[d.id] = WorldRobot(d.id, d, body, self.model,
                                              sensors=sensors, max_steps=max_steps)
                continue
            self.ducks[d.id] = WorldDuck(
                id=d.id, adr=DuckAddress.resolve(self.model, d.id), spawn=d.spawn,
                infer=infer_for.get(d.id, zero_infer),
                policy_id=d.policy, tof=sensors.get("tof"), detector=sensors.get("detector"),
                odom_preset=d.odom, odom_noise=OdomNoise.preset(d.odom),
                max_steps=max_steps)
        # Every body that a CONTROLLER steps rather than a policy, in roster
        # order. `step()` iterates this inside its substep loop, so on a
        # duck-only world it is empty and the loop body is unchanged — which
        # is what keeps `tests/test_arena.py`'s step-for-step lock exact.
        self._robots: list[WorldRobot] = [d for d in self.ducks.values()
                                          if isinstance(d, WorldRobot)]
        # Body ranges per robot: an attached subtree is contiguous after its
        # root, so the viewer's per-robot body list is one slice.
        self.duck_bodies: dict[str, slice] = {}
        for d in self.ducks.values():
            sub = [b for b in range(self.model.nbody)
                   if self.model.body_rootid[b] == d.root_body]
            assert sub == list(range(sub[0], sub[-1] + 1)), "robot subtree not contiguous"
            self.duck_bodies[d.id] = slice(sub[0], sub[-1] + 1)
        # Who owns each geom, for the bump sense: ducks 0..n-1, persons n.., -1 the rest (floor, walls, ball, toys).
        self._geom_owner = np.full(self.model.ngeom, -1, dtype=np.int64)
        self._owner_duck: list[WorldDuck] = list(self.ducks.values())
        for k, d in enumerate(self._owner_duck):
            s = self.duck_bodies[d.id]
            self._geom_owner[(self.model.geom_bodyid >= s.start) & (self.model.geom_bodyid < s.stop)] = k
        for k, p in enumerate(self.persons.values()):
            owner = len(self._owner_duck) + k
            if p.robot is not None:
                for b in p.robot.body_ids:
                    self._geom_owner[self.model.geom_bodyid == b] = owner
            else:
                self._geom_owner[self.model.geom_bodyid == p.body] = owner
        # THE BOARDS, as a thing a robot can hit (`wall_bumps`). Static
        # scenery is every geom of the worldbody — walls, the cove's facets, a
        # massless box, the basket's plates — and the FLOOR is the one to
        # leave out, because a walker's soles are on it every step. That
        # leaves exactly "a robot touched something built into the room",
        # which is the number Phase 3's bar is stated in and which nothing
        # else here measures: `_stamp_bumps` counts body-on-BODY contacts (the
        # `Senses.bumped` channel), and a wall is not a body owner.
        self._wall_geom = ((self.model.geom_bodyid == 0)
                           & (self.model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE))
        #: Separate EPISODES of board contact per robot id — a run of ticks in
        #: contact counts once, which is what a person means by "it bumped the
        #: wall". `wall_ticks` is the duration beside it, in control ticks.
        self.wall_bumps: dict[str, int] = {d: 0 for d in self.ducks}
        self.wall_ticks: dict[str, int] = {d: 0 for d in self.ducks}
        self._on_wall: set[str] = set()
        # Dynamic objects (free bodies that are not ducks): streamed each frame.
        self.objects: list[tuple[str, str, int]] = []
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            b = int(self.model.jnt_bodyid[j])
            name = self.model.body(b).name
            if "/" in name or name in self.persons:
                continue
            kind = ("ball" if name.startswith("ball") else
                    "toy" if name in self.pickables else "box")
            self.objects.append((name, kind, b))
        # Rolling cost of one control step, split physics / sensors (ms),
        # EMA over ~1 s of ticks — the /sim perf HUD reads it.
        self.perf = {"stepMs": 0.0, "sensorMs": 0.0, "encodeMs": 0.0}
        self.reset()

    # -- lifecycle ------------------------------------------------------------
    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.t = 0.0
        self.tick = 0
        self.goals = {"left": 0, "right": 0}
        self.last_goal = None
        self.kickoff_until = -1.0
        self.kickoff_team = None
        self.kickoff_ball = None
        self.last_kick_t = -1e9
        self.last_kick_duck = None
        self.goal_credit_duck = None
        self.goals_kicked = self.goals_bumped = 0
        self.ball_outs, self.ball_out_seq, self._ball_rest_t0 = 0, 0, None
        self.getups = self.getup_timeouts = 0
        self.getup_down_s = []
        # The board counters are keyed to the clock that just went back to
        # zero (`WorldState.restart`'s list of everything that outlived a
        # reset is what this is guarding against).
        for rid in self.wall_bumps:
            self.wall_bumps[rid] = self.wall_ticks[rid] = 0
        self._on_wall = set()
        for p in self.persons.values():
            p.reset(self.data)
        for d in self.ducks.values():
            self._respawn(d)
        mujoco.mj_forward(self.model, self.data)
        for d in self.ducks.values():
            d.prev_joint_vel = d.joint_vel(self.data)

    # A mocap capsule has infinite mass: a duck respawned INSIDE one is driven
    # out at 4-6 m/s (the physics audit's "fling" was this loop - a person
    # walking over a fallen duck's spawn point, every tick a new fall). The
    # spawn steps aside, perpendicular to the person's heading, to this
    # clearance from the capsule's surface.
    RESPAWN_CLEAR_M = 0.15

    def _clear_of_persons(self, x: float, y: float) -> tuple[float, float]:
        for p in self.persons.values():
            need = p.spec.radius + self.RESPAWN_CLEAR_M
            dx, dy = x - p.x, y - p.y
            if math.hypot(dx, dy) >= need:
                continue
            sx, sy = -math.sin(p.yaw), math.cos(p.yaw)          # the person's left
            if dx * sx + dy * sy < 0:
                sx, sy = -sx, -sy                                # ...or right, whichever side the spawn is on
            x, y = p.x + sx * need, p.y + sy * need
        return x, y

    def _price_spell(self, d: WorldDuck) -> None:
        """Record how long this spell on the floor lasted, as it ends. A fall's
        price is the thing a get-up changes, and it is not `getup_s`: that is
        only the ceiling a stuck duck hits."""
        if d.down_since >= 0.0:
            self.getup_down_s.append(round(self.t - d.down_since, 3))
            d.down_since = -1.0

    def _respawn(self, d) -> None:
        if isinstance(d, WorldRobot):
            x, y, yaw = d.spawn
            x, y = self._clear_of_persons(x, y)
            d.spawn_at(self.data, x, y, yaw)
            self._odom_reset(d, x, y, yaw)
            d.episodes += 1
            d.bumped_t = -1e9
            # Every channel the body declared, whatever they are — the reason
            # `Body.make_sensors` returns a dict and the arena does not name
            # the sensors a robot has.
            for s in d.sensors.values():
                s.reset()
            return
        # The DUCK's path, from here down, byte for byte what it was: the
        # spawn, the odometry, the beak, the skill, the gain, its two
        # sensors and the tracker, in the order they were.
        x, y, yaw = d.spawn
        x, y = self._clear_of_persons(x, y)
        spawn_duck(self.model, self.data, d.adr, x, y, yaw)
        self._odom_reset(d, x, y, yaw)
        d.last_action[:] = 0.0
        d.step_count = 0
        d.down_until = -1.0
        d.up_since = -1.0
        d.down_since = -1.0
        d._hold_yaw = None
        d.episodes += 1
        self.release(d)
        # …and the beak deadline goes with it. `release()` only re-arms this
        # when something was actually held, so a duck that dropped a toy at
        # t=182 kept `mouth_open_until = 182.9` through `World.reset()` — the
        # clock then restarts at 0 and `_mouth` holds the bill wide open for
        # the next 182 seconds of the new run.
        d.mouth_open_until = 0.0          # the field's own "shut" value, not a second sentinel
        d.skill = None
        d.skill_infer = None
        self._set_gain_ratio(d, 1.0)
        if d.tof is not None:
            d.tof.reset()
        if d.detector is not None:
            d.detector.reset()
        d.skill_sensed = False
        if d.id in self._ball_trackers:
            self._ball_trackers[d.id].reset()
        self._truth_sense.pop(d.id, None)      # the ablation's memory goes with the track's

    def reset_duck(self, duck_id: str) -> None:
        d = self.ducks[duck_id]
        self._respawn(d)
        mujoco.mj_forward(self.model, self.data)
        d.prev_joint_vel = d.joint_vel(self.data)

    def set_policy(self, duck_id: str, infer: Infer | None, policy_id: str | None = None) -> None:
        d = self.ducks[duck_id]
        d.infer = infer or zero_infer
        d.policy_id = policy_id

    # -- one 50 Hz control step -----------------------------------------------
    # -- manipulation (roadmap 12.2 / 12.3) ------------------------------------
    def mouth_tip(self, d: WorldDuck) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, d.adr.prefix + "mouth_tip")
        return self.data.site_xpos[sid]

    def _eq_id(self, d: WorldDuck, toy: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{d.id}/hold/{toy}")

    # -- soccer ---------------------------------------------------------------
    def _check_goal(self) -> None:
        j = self._ball_joint
        q = int(self.model.jnt_qposadr[j])
        x, y = float(self.data.qpos[q]), float(self.data.qpos[q + 1])
        hx = self.scenario.floor[0] / 2 - 0.25             # the walls sit 0.25 m inside the floor's edge
        if abs(y) < self.goal_width / 2 and abs(x) > hx - 0.08:
            side = "right" if x > 0 else "left"
            self.goals[side] += 1
            self.last_goal = side
            self.kickoff_team = self.team_defending(side)     # the side that conceded restarts
            self.goal_seq += 1
            if self.t - self.last_kick_t <= KICK_GOAL_S:
                self.goals_kicked += 1
                self.goal_credit_duck = self.last_kick_duck
            else:
                self.goals_bumped += 1
                self.goal_credit_duck = None
            self.kickoff()

    def _check_ball_out(self) -> None:
        """The ball-out rule (see `ball_out_s` in `__init__`): a ball at rest
        against the boards for long enough is placed back in play."""
        j = self._ball_joint
        q, v = int(self.model.jnt_qposadr[j]), int(self.model.jnt_dofadr[j])
        x, y = float(self.data.qpos[q]), float(self.data.qpos[q + 1])
        speed = float(np.hypot(self.data.qvel[v], self.data.qvel[v + 1]))
        hx, hy = self.scenario.floor[0] / 2 - 0.25, self.scenario.floor[1] / 2 - 0.25
        if not (min(hx - abs(x), hy - abs(y)) < self.ball_out_m and speed < 0.05):
            self._ball_rest_t0 = None
            return
        if self._ball_rest_t0 is None:
            self._ball_rest_t0 = self.t
            return
        if self.t - self._ball_rest_t0 < self.ball_out_s:
            return
        m = self.ball_out_in
        nx, ny = float(np.clip(x, -hx + m, hx - m)), float(np.clip(y, -hy + m, hy - m))
        half = self.goal_width / 2
        if nx != x and half > 0 and abs(ny) < half:
            # It came off an END board (the x-clip bound) from inside the
            # goal's own y-band: a ball that has stopped in the mouth without
            # crossing (`_check_goal` needs |x| > hx - 0.08). Clipping x alone
            # would place it on a PENALTY SPOT, squarely in front of the goal
            # it was about to go into - taking a tap-in away from one side and
            # handing the other a centred close-range chance, which is not
            # what a throw-in does. Put it beside the mouth instead.
            ny = float(np.clip(math.copysign(half + self.ball_out_m, ny or 1.0), -hy + m, hy - m))
        nx, ny = self._clear_of_ducks(nx, ny, hx, hy)
        self.data.qpos[q:q + 3] = [nx, ny, self.scenario.balls[0].radius + 0.005]
        self.data.qvel[v:v + 6] = 0.0
        self.ball_outs += 1
        self.ball_out_seq += 1        # harnesses watch this like `goal_seq` (brain/team.py throw_in_brains)
        for trk in self._ball_trackers.values():
            trk.reset()               # …and the ball a sensed kick remembers goes with it, exactly as the
                                      # brains' does (world_server.after_step): the referee moved it.
        self._truth_sense.clear()
        self._ball_rest_t0 = None

    def _clear_of_ducks(self, x: float, y: float, hx: float, hy: float) -> tuple[float, float]:
        """Step a ball-out placement off any duck standing on it (see
        `ball_out_clear`), pushed straight out from that duck and kept inside
        the boards. Two passes: moving clear of one body can walk into
        another, and on a crowded pitch the second pass settles it."""
        m = self.ball_out_in
        for _ in range(2):
            moved = False
            for d in self.ducks.values():
                p = d.trunk_pos(self.data)
                dx, dy = x - float(p[0]), y - float(p[1])
                r = math.hypot(dx, dy)
                if r >= self.ball_out_clear:
                    continue
                if r < 1e-6:
                    dx, dy, r = 1.0, 0.0, 1.0          # dead centre: any direction will do
                x = float(np.clip(float(p[0]) + dx / r * self.ball_out_clear, -hx + m, hx - m))
                y = float(np.clip(float(p[1]) + dy / r * self.ball_out_clear, -hy + m, hy - m))
                moved = True
            if not moved:
                break
        return x, y

    def kickoff(self) -> None:
        """Restart play: the ball on the centre spot (a few centimetres of
        random nudge, so two mirror-image ducks do not meet nose to nose),
        every duck back on its spawn, and `kickoff_hold_s` of zero command
        so play resumes from standing ducks and not from the heap at the
        goal mouth. Brains are the caller's: `goal_seq` says a goal
        happened (brain/team.py `kickoff_brains` resets what they should
        forget and keeps what they should not — the kicks they took)."""
        j = self._ball_joint
        if j is None:
            return
        q, v = int(self.model.jnt_qposadr[j]), int(self.model.jnt_dofadr[j])
        nx, ny = self.rng.uniform(-0.05, 0.05, 2)
        self.data.qpos[q:q + 7] = [nx, ny, self.scenario.balls[0].radius + 0.005, 1.0, 0.0, 0.0, 0.0]
        self.data.qvel[v:v + 6] = 0.0
        self.kickoff_ball = (float(nx), float(ny))
        for d in self.ducks.values():
            self._respawn(d)
            d.set_cmd(self.data, (0.0, 0.0, 0.0))
        mujoco.mj_forward(self.model, self.data)
        for d in self.ducks.values():
            d.prev_joint_vel = d.joint_vel(self.data)
        self.kickoff_until = self.t + self.kickoff_hold_s

    @property
    def in_kickoff(self) -> bool:
        return self.t < self.kickoff_until

    def team_defending(self, side: str) -> str | None:
        """The team whose own mouth is `side` ("left" / "right"): the one a
        ball crossing it scores AGAINST, which kicks off after. None when no
        duck on the pitch belongs to a team."""
        for d in self.ducks.values():
            g = self.goal_for(d)
            tm = self.team_of.get(d.id)
            if g is None or tm is None:
                continue
            if ("right" if g[0] > 0 else "left") != side:
                return tm
        return None

    @property
    def game_state(self) -> str:
        """"set" (the kickoff hold), "kickoff" (the conceding side's ball,
        still on the spot, inside the window) or "playing"."""
        if self.in_kickoff:
            return "set"
        if self.kickoff_team is not None and self.kickoff_ball is not None \
                and self.t < self.kickoff_until + self.kickoff_free_s:
            b = self.ball_xy()
            if b is not None and math.dist(b, self.kickoff_ball) < self.kickoff_moved_m:
                return "kickoff"
        return "playing"

    def goal_for(self, d: WorldDuck) -> tuple[float, float] | None:
        """The goal this duck attacks (world = odometry-at-spawn frame): the
        mouth its team is declared to attack (`Scenario.attacks`), else the one
        its spawn heading faces. None off a pitch.

        The declaration exists because the heading rule cannot answer for a
        roster placed by hand — a defender is placed facing its OWN goal — and
        an undeclared team whose ducks disagree is refused by
        `validate_scenario` rather than resolved here."""
        if self.goal_width <= 0:
            return None
        hx = self.scenario.floor[0] / 2 - 0.25
        mouth = self.scenario.attacks.get(self.team_of.get(d.id) or "")
        if mouth is not None:
            return (hx if mouth == "right" else -hx), 0.0
        return (hx if math.cos(d.spawn[2]) >= 0 else -hx), 0.0

    def ball_xy(self) -> tuple[float, float] | None:
        """The ball's planar position, unrounded. `soccer_score` rounds to the
        millimetre for the stream, which is right for a payload and wrong for
        an accumulator: eval-pitch sums per-step displacements over 15 000
        control ticks, and mm-quantised differences random-walk into the
        answer. None off a pitch (no ball / no goals)."""
        if self._ball_joint is None:
            return None
        q = int(self.model.jnt_qposadr[self._ball_joint])
        return float(self.data.qpos[q]), float(self.data.qpos[q + 1])

    def soccer_score(self) -> dict | None:
        if self._ball_joint is None:
            return None
        q = int(self.model.jnt_qposadr[self._ball_joint])
        return {"left": self.goals["left"], "right": self.goals["right"],
                "ball": [round(float(self.data.qpos[q]), 3), round(float(self.data.qpos[q + 1]), 3)],
                "lastGoal": self.last_goal, "kickoff": round(max(0.0, self.kickoff_until - self.t), 2),
                "kicked": self.goals_kicked, "bumped": self.goals_bumped,
                "state": self.game_state, "kickoffTeam": self.kickoff_team,
                "ballOuts": self.ball_outs}

    # -- odometry (roadmap 1.7) ---------------------------------------------
    def _odom_reset(self, d: WorldDuck, x: float, y: float, yaw: float) -> None:
        n = d.odom_noise
        d.odom_est[:] = (x, y, yaw)
        d._odom_true_prev = np.array([x, y, yaw])
        d._odom_scale = 1.0 + float(self.rng.normal(0.0, n.scale_sigma)) if n.scale_sigma else 1.0
        d._odom_yaw_bias = float(self.rng.normal(0.0, n.yaw_bias_sigma)) if n.yaw_bias_sigma else 0.0

    def _odom_step(self, d: WorldDuck) -> None:
        """Dead-reckon one control step: the TRUE motion in the body frame,
        scaled, biased and noised per OdomNoise, integrated in the estimate's
        own frame — so a yaw error bends the whole path after it."""
        pos = d.trunk_pos(self.data)
        yaw = d.yaw(self.data)
        if d._odom_true_prev is None:
            self._odom_reset(d, float(pos[0]), float(pos[1]), yaw)
            return
        px, py, pyaw = d._odom_true_prev
        dx, dy = float(pos[0]) - px, float(pos[1]) - py
        c, s_ = np.cos(pyaw), np.sin(pyaw)
        fwd, left = c * dx + s_ * dy, -s_ * dx + c * dy          # body-frame step
        dyaw = float(np.arctan2(np.sin(yaw - pyaw), np.cos(yaw - pyaw)))
        d._odom_true_prev = np.array([pos[0], pos[1], yaw])
        n = d.odom_noise
        if n.scale_sigma or n.yaw_bias_sigma or n.step_sigma or n.yaw_step_sigma:
            ds = float(np.hypot(fwd, left))
            fwd *= d._odom_scale
            left *= d._odom_scale
            if n.step_sigma and ds > 0:
                fwd += float(self.rng.normal(0.0, n.step_sigma * ds))
                left += float(self.rng.normal(0.0, n.step_sigma * ds))
            dyaw += d._odom_yaw_bias * C.CTRL_DT
            if n.yaw_step_sigma and dyaw:
                dyaw += float(self.rng.normal(0.0, n.yaw_step_sigma * abs(dyaw)))
        eyaw = d.odom_est[2]
        d.odom_est[0] += np.cos(eyaw) * fwd - np.sin(eyaw) * left
        d.odom_est[1] += np.sin(eyaw) * fwd + np.cos(eyaw) * left
        d.odom_est[2] = float(np.arctan2(np.sin(eyaw + dyaw), np.cos(eyaw + dyaw)))

    def odom(self, d: WorldDuck) -> tuple[float, float, float]:
        """The (x, y, yaw) a brain gets: the truth under the `ideal` preset."""
        return float(d.odom_est[0]), float(d.odom_est[1]), float(d.odom_est[2])

    def set_odom_preset(self, d: WorldDuck, name: str) -> None:
        d.odom_preset, d.odom_noise = name, OdomNoise.preset(name)
        pos = d.trunk_pos(self.data)
        self._odom_reset(d, float(pos[0]), float(pos[1]), d.yaw(self.data))

    def grasp(self, d: WorldDuck, tol_xy: float = GRASP_TOL_XY, tol_z: float = GRASP_TOL_Z) -> str | None:
        """Close the beak: the nearest toy within tolerance of the mouth tip
        gets WELDED to the jaw (grasp as an attachment event, roadmap 12.2).
        Success is a curve in alignment error, not physics: p = 1 at zero
        error, 0 at the tolerance. Returns the toy id or None."""
        d.beak_closed = True
        if d.holding is not None or not self.pickables:
            return d.holding
        tip = self.mouth_tip(d)
        best, best_err, nearest = None, 9.0, 9.0
        for toy, b in self.pickables.items():
            if any(o.holding == toy for o in self.ducks.values()):
                continue
            p = self.data.xpos[b]
            exy = float(np.hypot(p[0] - tip[0], p[1] - tip[1]))
            ez = abs(float(p[2] - tip[2]))
            nearest = min(nearest, exy)
            if exy < tol_xy and ez < tol_z and exy < best_err:
                best, best_err = toy, exy
        d.grasp_attempts += 1
        d.last_grasp_err = None if nearest >= 9.0 else nearest
        if best is None:
            return None
        # Gentle in the middle, zero at the edge: a 2 cm miss on a 3 cm
        # window still grasps 3 times in 4. A model, not physics (12.2).
        p_ok = 1.0 - (best_err / tol_xy) ** 2
        if self.rng.random() > p_ok:
            return None
        eq = self._eq_id(d, best)
        jaw = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, d.adr.prefix + "jaw_soft")
        b = self.pickables[best]
        # Weld data: anchor (in body1) + relative pose of body2 in body1, so
        # the toy keeps exactly the pose it was caught in.
        Rj = self.data.xmat[jaw].reshape(3, 3)
        rel_p = Rj.T @ (self.data.xpos[b] - self.data.xpos[jaw])
        qj, qb = self.data.xquat[jaw], self.data.xquat[b]
        qj_inv = np.array([qj[0], -qj[1], -qj[2], -qj[3]])
        rel_q = np.zeros(4)
        mujoco.mju_mulQuat(rel_q, qj_inv, qb)
        self.model.eq_data[eq, 0:3] = 0.0
        self.model.eq_data[eq, 3:6] = rel_p
        self.model.eq_data[eq, 6:10] = rel_q
        self.model.eq_data[eq, 10] = 1.0
        self.data.eq_active[eq] = 1
        d.holding = best
        d.grasp_successes += 1
        return best

    def release(self, d: WorldDuck) -> str | None:
        d.beak_closed = False
        toy = d.holding
        if toy is None:
            return None
        # Arm the drop-open window only for a release that released SOMETHING.
        # Spawning calls this on every duck to clear its hands, and arming it
        # up here left every duck in a fresh world gaping for MOUTH_DROP_S.
        d.mouth_open_until = self.t + MOUTH_DROP_S
        self.data.eq_active[self._eq_id(d, toy)] = 0
        d.holding = None
        return toy

    def _mouth(self, d: WorldDuck) -> None:
        """Drive the 15th servo. The beak reaches open, snaps shut on the
        grab - as far as the toy in it allows - and opens again to drop.

        This is the whole of mouth control, as it is on the robot: the 14
        actions a policy returns are scattered around this joint, never onto
        it (`duck-control`'s MOUTH_INDEX), so nothing here can disturb a
        gait."""
        if d.holding is not None:
            # A 40 mm block is wider than the 36 mm gape, so the bill goes
            # AROUND it rather than closing through it; a 10 mm brick nearly
            # shuts. `grasp` is an attachment either way (roadmap 12.2).
            kind = self.pickable_kind.get(d.holding)
            want = 1.0 if kind is None else mouth_frac_for_gape(min(PICKABLE_KINDS[kind]["size"]))
        elif d.skill == "ground_pick" and not d.beak_closed:
            want = 1.0                       # reaching: open on the way down
        elif self.t < d.mouth_open_until:
            want = 1.0                       # just dropped one
        else:
            want = 0.0                       # at rest a duck's beak is shut
        d.mouth = want
        if d.adr.mouth_act >= 0:
            self.data.ctrl[d.adr.mouth_act] = mouth_target(want)

    def _set_gain_ratio(self, d: WorldDuck, ratio: float) -> None:
        """Scale this duck's position-actuator Kp (gain and the matching
        bias term) — the standing gain a kick window runs at on the robot."""
        if d.kp_base is None:
            d.kp_base = self.model.actuator_gainprm[d.adr.actuators, 0].copy()
        if ratio == d.gain_ratio:
            return
        kp = d.kp_base * ratio
        self.model.actuator_gainprm[d.adr.actuators, 0] = kp
        self.model.actuator_biasprm[d.adr.actuators, 1] = -kp
        d.gain_ratio = ratio

    # Skills trained HERE, vendored under microduck_local/policies/<skill>/,
    # preferred over the shipped Hub file when present (the kicks: roadmap
    # item 7, behaviors/kick.py - 0% whiff from every gaze pose on the bench,
    # whiff 61 -> 40% in play). A sidecar .json beside the ONNX carries what
    # the brain must know about it (`exit_rad`: the kick's exit angle off the
    # body). MICRODUCK_SKILL_<NAME>=path still wins over both.
    LOCAL_SKILLS = {"kick_left": "kick/kick_left.onnx", "kick_right": "kick/kick_right.onnx"}

    @staticmethod
    def skill_path(name: str) -> Path | None:
        """The policy file a skill runs from: the env override, else a local
        export under policies/, else the shipped file. None for no skill."""
        if name not in SKILLS:
            return None
        from ..brain.brain_env import POLICIES_DIR  # noqa: PLC0415
        override = os.environ.get(f"MICRODUCK_SKILL_{name.upper()}")
        if override:
            return Path(override)                                    # checked to exist when the World is built
        local = World.LOCAL_SKILLS.get(name)
        if local:
            p = Path(__file__).resolve().parents[3] / "policies" / local
            if p.exists():
                return p
        return POLICIES_DIR / SKILLS[name]

    @staticmethod
    def kick_exits() -> tuple[float, float] | None:
        """(left, right) exit angles (rad, off the body) of the kicks this
        world will run, from the sidecars beside local exports; None when
        either kick is the shipped one (the brain keeps its measured
        defaults)."""
        out = []
        for name in ("kick_left", "kick_right"):
            p = World.skill_path(name)
            side = p.with_suffix(".json") if p is not None else None
            if side is None or not side.exists():
                return None
            try:
                out.append(float(json.loads(side.read_text())["exit_rad"]))
            except (ValueError, KeyError, OSError, TypeError):      # a sidecar written before the exit was measured (null)
                return None
        return out[0], out[1]

    @staticmethod
    def skill_sidecar(name: str) -> dict:
        """Everything the sidecar .json beside a skill's ONNX says — the same
        file `kick_exits` reads `exit_rad` out of. `{}` when there is no
        sidecar, or it is not readable, or it is not an object."""
        p = World.skill_path(name)
        side = p.with_suffix(".json") if p is not None else None
        if side is None or not side.exists():
            return {}
        try:
            out = json.loads(side.read_text())
        except (ValueError, OSError):
            return {}
        return out if isinstance(out, dict) else {}

    @staticmethod
    def skill_sensed(name: str) -> bool:
        """Was this skill's ONNX trained with the ball in its four HEAD
        command slots (roadmap 12as)? The sidecar says so with
        `"sensed": true`. WITHOUT the field — every sidecar written before
        2026-09-10, the vendored kicks included — this is False and the skill
        gets the all-zero command block it was trained on, to the bit."""
        return bool(World.skill_sidecar(name).get("sensed", False))

    @staticmethod
    def sensed_truth_env_mode() -> str:
        """Which of the SIM-ONLY truth ablations the environment asks for:
        "" (off, the only shipping value), "all", "xy" or "fresh" — see
        SENSED_TRUTH_ENV. A value that is none of these RAISES rather than
        falling back to a mode, because the failure it prevents is a battery
        running an arm it did not mean to and quoting the number."""
        raw = os.environ.get(SENSED_TRUTH_ENV, "").strip()
        if raw in SENSED_TRUTH_OFF:
            return ""
        if raw in SENSED_TRUTH_ALL:
            return "all"
        if raw in SENSED_TRUTH_MODES:
            return raw
        raise ValueError(f"{SENSED_TRUTH_ENV}={raw!r} is not a mode: "
                         f"one of {SENSED_TRUTH_OFF + SENSED_TRUTH_ALL + SENSED_TRUTH_MODES}")

    def _ball_tracker(self, d: WorldDuck):
        """This duck's ball track, for a SENSED kick to read. The brain's own
        `Tracker` (brain/tracker.py) over this duck's own detector frames and
        its own odometry — the same class, the same uncertainty model off the
        same detector preset, so what the kick reads is what a `Chase` on this
        duck reads. Built on first use and kept WARM: a tracker started at the
        swing has no memory to coast, and the sighting arrives on only 58% of
        swings (12ak) — the other half is exactly what the held estimate is
        for."""
        trk = self._ball_trackers.get(d.id)
        if trk is None:
            from ..brain.tracker import Tracker, TrackerParams  # noqa: PLC0415
            preset = next((s.detector for s in self.scenario.ducks if s.id == d.id), None)
            trk = self._ball_trackers[d.id] = Tracker(TrackerParams.for_detector(preset))
        return trk

    def _sensed_head(self, d: WorldDuck) -> None:
        """Write the ball into the four head slots in the recipe's units —
        `behaviors/lastmetre.py`'s docstring, off the tracker instead of off
        the truth:

          [51] bearing in the DUCK's own yaw frame, psi / (pi/2), + = LEFT
          [52] ground range / LM_RANGE_SCALE, clipped to 0..1
          [53] 1.0 while the detector's last frame had the ball in it
          [54] freshness: 1.0 on that frame, fading exp(-age / LM_MEM_TAU)

        and all four zero while nothing is known, which is what the recipe
        spawns with. The scale and the decay are IMPORTED from the recipe, so
        the trained units and the played units cannot drift apart.

        Under the sim-only ablation (SENSED_TRUTH_ENV) some or all of the four
        come from `truth_slots` instead. The mix is COMPOSED here and nowhere
        else: each half is whichever producer's own output, unmodified, so
        "xy" and "fresh" between them are exactly "all" and neither can drift
        from the played path they are being compared against."""
        hc = d.head_cmd
        hc[:] = 0.0
        mode = self.sensed_truth_mode
        if mode:
            if mode == "all":
                hc[:] = self.truth_slots(d)   # the sim-only ablation; see SENSED_TRUTH_ENV
                return
            truth = self.truth_slots(d)
            self._track_head(d, hc)           # the played path, into the same buffer…
            half = slice(0, 2) if mode == "xy" else slice(2, 4)
            hc[half] = truth[half]            # …then ONE half of it replaced by the truth's
            return
        self._track_head(d, hc)

    def _track_head(self, d: WorldDuck, hc: np.ndarray) -> None:
        """The four slots off the TRACK — the played path, and the arithmetic
        `_sensed_head` had inline before the splits existed. Writes into `hc`,
        which the caller has already zeroed, and leaves it all-zero when the
        duck knows nothing about a ball."""
        from ..behaviors.lastmetre import LM_MEM_TAU, LM_RANGE_SCALE  # noqa: PLC0415
        trk = self._ball_trackers.get(d.id)
        tr = None if trk is None else trk.best(SENSED_BALL_CLS, self.t, min_hits=1)
        if tr is None:
            return
        x, y, yaw = self.odom(d)
        # Off `xy` whenever the track has a position (roadmap 12ar): `bearing`
        # and `range` are the pose at the last HIT turned by yaw alone, so they
        # stop meaning bearing the moment the duck WALKS — and a kick window is
        # entered walking.
        bearing = tr.bearing if tr.xy is None else tr.bearing_from((x, y), yaw)
        rng = tr.range if tr.xy is None else tr.range_from((x, y))
        hc[0] = float(np.clip(bearing / (math.pi / 2), -1.0, 1.0))
        hc[1] = float(np.clip(rng / LM_RANGE_SCALE, 0.0, 1.0))
        last = d.detector.last if d.detector is not None else None
        seen = last is not None and tr.last_t == last.t     # the newest frame is the one that hit
        hc[2] = 1.0 if seen else 0.0
        hc[3] = 1.0 if seen else float(math.exp(-max(0.0, tr.age(self.t)) / LM_MEM_TAU))

    # -- the truth ablation (SIM-ONLY, see SENSED_TRUTH_ENV) --------------------
    def _head_camera(self, d: WorldDuck):
        """(position, forward, image-right, image-up) of THIS duck's head
        camera, world frame — `behaviors/ball.py::_ball_camera`'s convention on
        the composed model's per-duck `<prefix>head_camera` element, so the
        truth arm projects through the same optics the recipe trained on."""
        cid = self._truth_cam.get(d.id)
        if cid is None:
            cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, d.adr.prefix + "head_camera")
            if cid < 0:
                raise RuntimeError(f"duck {d.id} has no head_camera element")
            self._truth_cam[d.id] = cid
        R = self.data.cam_xmat[cid]
        return self.data.cam_xpos[cid], R[2::3], -R[1::3], -R[0::3]

    def truth_update(self, d: WorldDuck) -> None:
        """SIM-ONLY. One control step of the RECIPE's own sensing
        (`behaviors/lastmetre.py::_lm_sense`) run against the TRUE ball: the
        same FOV, the same `LM_DETECT_EVERY` cadence, the same `LM_JITTER` on
        the reported angles, the same `_lm_ground_point` placement off the head
        encoders, the same `LM_MEM_TAU` fade. Everything the recipe's detector
        does is kept; the ONE thing removed is the tracker — association,
        smoothing and odometry coasting — and with it its placement error.

        Held per duck and stepped EVERY tick (not only inside a kick window),
        because that is what training did: a memory that only runs during the
        0.5 s swing would start every window blind and measure that instead.

        The jitter comes from a stream of its own (`_truth_rng`, seeded off the
        world's seed), so an arm running under this flag cannot move the
        world's RNG and nothing outside the ablation changes.
        """
        from ..behaviors.ball import _BALL_KNOBS  # noqa: PLC0415
        from ..behaviors.lastmetre import (  # noqa: PLC0415
            LM_DETECT_EVERY,
            LM_JITTER,
            LM_MAX_RANGE,
            LM_MEM_TAU,
            _lm_ground_point,
        )
        st = self._truth_sense.get(d.id)
        if st is None:
            st = self._truth_sense[d.id] = {"world": None, "conf": 0.0, "det_tick": -(10 ** 9),
                                            "seen": False}
        if self._truth_rng is None:
            self._truth_rng = np.random.default_rng((int(self._truth_seed or 0) * 2 + 90_210) % (2 ** 32))
        j = self._ball_joint
        if j is None:
            return
        q = int(self.model.jnt_qposadr[j])
        bx, by, bz = (float(self.data.qpos[q]), float(self.data.qpos[q + 1]), float(self.data.qpos[q + 2]))
        cam, fwd, right, up = self._head_camera(d)
        half_h = math.radians(_BALL_KNOBS["MICRODUCK_BALL_HFOV_DEG"]) / 2
        half_v = math.radians(_BALL_KNOBS["MICRODUCK_BALL_VFOV_DEG"]) / 2
        vx, vy, vz = bx - float(cam[0]), by - float(cam[1]), bz - float(cam[2])
        dist = math.sqrt(vx * vx + vy * vy + vz * vz)
        f = vx * fwd[0] + vy * fwd[1] + vz * fwd[2]
        if f > 1e-6:
            ax = math.atan2(vx * right[0] + vy * right[1] + vz * right[2], f) / half_h
            ay = math.atan2(vx * up[0] + vy * up[1] + vz * up[2], f) / half_v
            seen = -1.0 < ax < 1.0 and -1.0 < ay < 1.0 and dist < LM_MAX_RANGE
        else:
            ax = ay = 0.0
            seen = False
        if self.tick - st["det_tick"] >= LM_DETECT_EVERY:
            st["det_tick"] = self.tick
            st["seen"] = seen
            if seen:
                r = self._truth_rng
                a_h = (ax + float(r.uniform(-LM_JITTER, LM_JITTER))) * half_h
                a_v = (ay + float(r.uniform(-LM_JITTER, LM_JITTER))) * half_v
                p = _lm_ground_point(cam, fwd, right, up, a_h, a_v)
                if p is not None:
                    st["world"], st["conf"] = p, 1.0
        if not st["seen"]:
            st["conf"] *= math.exp(-C.CTRL_DT / LM_MEM_TAU)

    def truth_slots(self, d: WorldDuck) -> np.ndarray:
        """SIM-ONLY. The four head slots `truth_update`'s held estimate makes,
        in the recipe's units — the tail of `_lm_sense`, off the same odometry
        pose `_sensed_head` reads the track from, so the two arms differ in the
        ESTIMATE and in nothing else."""
        from ..behaviors.lastmetre import LM_RANGE_SCALE  # noqa: PLC0415
        out = np.zeros(4, np.float32)
        st = self._truth_sense.get(d.id)
        if st is None:
            return out
        if st["world"] is not None:
            x, y, yaw = self.odom(d)
            dx, dy = st["world"][0] - x, st["world"][1] - y
            c, s = math.cos(yaw), math.sin(yaw)
            ahead, beside = c * dx + s * dy, -s * dx + c * dy      # + beside = to the LEFT
            out[0] = float(np.clip(math.atan2(beside, ahead) / (math.pi / 2), -1.0, 1.0))
            out[1] = float(np.clip(math.hypot(ahead, beside) / LM_RANGE_SCALE, 0.0, 1.0))
        out[2] = 1.0 if st["seen"] else 0.0
        out[3] = float(np.clip(st["conf"], 0.0, 1.0))
        return out

    def start_skill(self, d: WorldDuck, name: str) -> bool:
        """Hand the reflex tier to a skill policy for one cycle (the robot's
        own pattern: hard swap in, auto swap back): ground_pick (a phase
        cycle) or kick_left / kick_right (a 0.5 s window)."""
        if name not in SKILLS or d.skill is not None:
            return False
        if name not in self.skills:
            from ..brain.brain_env import onnx_infer
            path = self.skill_path(name)                 # env override, local export, or the shipped file
            if not path.exists():
                return False
            self.skills[name] = onnx_infer(path)
        d.skill, d.skill_t0, d.skill_infer = name, self.t, self.skills[name]
        d.skill_sensed = self._skill_sensed.get(name, False)
        d._hold_yaw = None
        if name.startswith("kick"):
            self._set_gain_ratio(d, STANDING_GAIN_RATIO)
            self.last_kick_t, self.last_kick_duck = self.t, d.id
        if d.holding is None:
            d.beak_closed = False          # a cycle starts with an open, empty beak
        return True

    def sense_grip(self, d) -> str | None:
        """Refresh a driver-stepped body's `holding` from its gripper, and
        return the toy id it has hold of (or None).

        **A MARS's grasp is PHYSICS, not an attachment.** A duck's is a weld
        the World switches on (`grasp`, roadmap 12.2), because a soft bill
        closing around a toy is not something a rigid convex hull does. MARS's
        two blades are real geoms with Innate's own contact model on them
        (`mars.FINGER_*`), and 4b measured that at 2 ms they hold the 4 cm
        block 14 spots of 16 — so there is nothing to weld and nothing to
        model: the claw either has it or it does not, and this reads which.
        That is also why `pick`/`release` are not events a MARS emits. The
        transition of THIS value is the event, which is what `record-world`'s
        log and the `/sim` overlay already watch (`d.holding` changing).

        Called once a tick, after the substep loop, beside the sensor polls —
        `Senses.holding` is a sense.
        """
        bid = d.held_body(self.data)
        d.holding = self._pickable_of_body.get(bid) if bid >= 0 else None
        return d.holding

    def in_basket(self, toy: str) -> bool:
        """Is this toy in the basket? GEOMETRY, and body-agnostic on purpose.

        Nothing here asks who put it there or how: the footprint of the tray
        and a height under the rim plus 5 cm. So the tidy score, `eval-tidy`'s
        count and `record-world`'s overlay measure a MARS's arm and a duck's
        beak with the same instrument, and "toys in the basket" is one number
        across bodies rather than two definitions that could drift.
        """
        if self.basket is None:
            return False
        p = self.data.xpos[self.pickables[toy]]
        bx, by = self.basket.pos
        sx, sy = self.basket.size[0] / 2, self.basket.size[1] / 2
        return bool(abs(p[0] - bx) < sx and abs(p[1] - by) < sy and p[2] < self.basket.rim + 0.05)

    def tidy_score(self) -> dict:
        n = len(self.pickables)
        done = sum(self.in_basket(t) for t in self.pickables)
        return {"total": n, "inBasket": done, "held": [d.holding for d in self.ducks.values() if d.holding]}

    def apply_intent(self, d, intent) -> None:
        """Route a brain's non-twist intents to the reflex tier.

        A driver-stepped body takes `arm` and ignores `skill` and `beak`: it
        has no skill-policy tier (its reflex tier IS the driver) and no beak.
        Ignoring rather than raising, for the same reason a duck ignores
        `arm`: a brain is written against the intent vocabulary and not
        against one robot, and `wander` setting no arm and `tidy` asking for
        a beak must both be legal on both bodies."""
        if isinstance(d, WorldRobot):
            if intent.arm:
                d.set_arm(intent.arm)
            return
        if intent.skill:
            self.start_skill(d, intent.skill)
        if intent.beak == "close" and not d.beak_closed:
            self.grasp(d)
        elif intent.beak == "open" and d.beak_closed:
            self.release(d)

    def _sense_bumps(self, pairs: list[np.ndarray]) -> None:
        """Snapshot the current contact list's geom pairs into `pairs`. On the
        robot a bump is the IMU and the servo loads; here it is the contact
        list, read after EVERY physics substep of a tick (`step`) so a touch
        that lasts one substep still counts - the last substep alone missed
        three in four. The owner lookup happens once a tick (`_stamp_bumps`);
        measured against the last-substep-only scan, the four snapshots cost
        3-8% of a world step (10-20 us of 0.2-0.35 ms, pitch and playroom).
        Under collision "all" (the default) every body of a duck is in this:
        trunk, head, legs, feet; under "walk" the soles and the
        self-collision slivers. The floor, the boards, the ball and the toys
        are not owners."""
        n = self.data.ncon
        if n:
            pairs.append(self.data.contact.geom1[:n].copy())
            pairs.append(self.data.contact.geom2[:n].copy())

    def _stamp_bumps(self, pairs: list[np.ndarray]) -> None:
        """Every duck whose body touched ANOTHER owner's (a duck's or a
        person's) in any substep of the tick: its `bumped_t` is now. And,
        separately, every robot that touched the BOARDS (`wall_bumps`).

        The two are different questions and are counted apart on purpose.
        `bumped_t` is the `Senses.bumped` channel a brain reads — "something
        that can move is against me" — and its owner table has no entry for
        scenery, so a duck walking into a wall has never been `bumped` and
        must not start being (every soccer number is measured on that
        meaning). The board count is an INSTRUMENT: Phase 3's bar for a
        wheeled body is stated in wall contacts, and no brain reads it.
        """
        if not pairs or len(self._owner_duck) == 0:
            return
        own, wall = self._geom_owner, self._wall_geom
        g1, g2 = np.concatenate(pairs[0::2]), np.concatenate(pairs[1::2])
        o1, o2 = own[g1], own[g2]
        nd = len(self._owner_duck)
        # The boards, first, because it is unconditional: a tick with no
        # body-on-body contact can still be a tick against a wall.
        on_wall: set[str] = set()
        touch = np.flatnonzero((wall[g2] & (o1 >= 0)) | (wall[g1] & (o2 >= 0)))
        if touch.size:
            for k in set(o1[touch].tolist()) | set(o2[touch].tolist()):
                if 0 <= k < nd:
                    on_wall.add(self._owner_duck[k].id)
        for rid in on_wall:
            self.wall_ticks[rid] += 1
            if rid not in self._on_wall:
                self.wall_bumps[rid] += 1        # a new episode, not another tick of one
        self._on_wall = on_wall
        hit = (o1 >= 0) & (o2 >= 0) & (o1 != o2)
        if not hit.any():
            return
        for k in set(o1[hit].tolist()) | set(o2[hit].tolist()):
            if k < nd:
                self._owner_duck[k].bumped_t = self.t

    def bumped(self, d: WorldDuck, within: float = 0.1) -> bool:
        return self.t - d.bumped_t <= within

    def _skill_cmd(self, d: WorldDuck) -> Infer | None:
        """While a skill runs, it owns the command block; returns the infer to
        use, ending the cycle at its exit phase."""
        if d.skill is None:
            return None
        if d.skill.startswith("kick"):
            if self.t - d.skill_t0 >= KICK_S:
                d.skill, d.skill_infer, d.skill_sensed = None, None, False
                d.twist_cmd[:] = 0.0
                self._set_gain_ratio(d, 1.0)
                return None
            d.twist_cmd[:] = 0.0                  # the kick's observation carries an all-zero command
            if d.skill_sensed:
                self._sensed_head(d)              # …except the four the SENSED pair was trained to read
            else:
                d.head_cmd[:] = 0.0
            d.body_cmd[:] = 0.0
            return d.skill_infer
        phi = (self.t - d.skill_t0) / GROUND_PICK_PERIOD_S
        if phi >= GROUND_PICK_END_PHI:
            d.skill, d.skill_infer = None, None
            d.twist_cmd[:] = 0.0
            return None
        d.twist_cmd[:] = (np.cos(2 * np.pi * phi), np.sin(2 * np.pi * phi), 0.0)
        d.head_cmd[:] = 0.0
        d.body_cmd[:] = 0.0
        if phi >= GROUND_PICK_CLOSE_PHI and not d.beak_closed:
            self.grasp(d)
        return d.skill_infer

    def step(self) -> None:
        t0 = time.perf_counter()
        m, data = self.model, self.data
        hold = self.in_kickoff
        for d in self.ducks.values():
            if hold or d.down_until > self.t:  # kickoff, or lying where it fell: stand, whatever the brain asked
                d.set_cmd(data, (0.0, 0.0, 0.0))
            if isinstance(d, WorldRobot):
                # A driver-stepped body's 50 Hz work is the command, which
                # `set_cmd` already did (here for a hold, or in the caller's
                # `drive()` for a brain). Everything else it needs happens in
                # the substep loop below, where its controller must run.
                continue
            if self._sensed_kick and d.detector is not None:
                ox, oy, oyaw = self.odom(d)       # the ball track a sensed kick reads, kept warm
                self._ball_tracker(d).update(d.detector.last, self.t, oyaw, (ox, oy))
                if self.sensed_truth:
                    self.truth_update(d)      # …and, under the ablation, the truth projection beside it
            skill = self._skill_cmd(d)
            obs = d.obs(data)
            # A duck that is DOWN is driven by the get-up policy when there is
            # one. The zero command above is exactly what a standing behaviour
            # wants, so the observation it sees is the one it was trained on.
            down = self.getup_infer is not None and d.down_until > self.t
            raw = np.asarray((skill or (self.getup_infer if down else d.infer))(obs), np.float32)
            d.last_action = raw.copy()
            data.ctrl[d.adr.actuators] = C.DEFAULT_POSE + raw.clip(-4.0, 4.0)
            self._mouth(d)
        blockers = [tuple(d.trunk_pos(data)[:2]) for d in self.ducks.values()] if any(
            p.spec.yield_m > 0 for p in self.persons.values()) else ()
        for p in self.persons.values():
            p.step(data, C.CTRL_DT, blockers)
        pairs: list[np.ndarray] = []
        for _ in range(self.substeps):
            # A controller-stepped body runs EVERY physics step, not once a
            # control tick. MEASURED (`robots/mars_drive.py`): decimated to
            # 50 Hz the base's velocity loop has gain 3.04 — past the explicit
            # loop's bound of 2 — and a 0.3 m/s run ends going backwards at
            # 2.6 m/s. `self._robots` is EMPTY on a duck-only world, so this
            # loop body is what it always was and `tests/test_arena.py`'s
            # step-for-step lock against the walk env is exact.
            for r in self._robots:
                r.driver.step(data)
            mujoco.mj_step(m, data)
            self._sense_bumps(pairs)
        # What the walk env does after its substep loop (`_refresh_derived`,
        # physics audit item 6): `mj_step` integrates AFTER it computed
        # kinematics and sensors, so without this the IMU blocks of the obs,
        # the trunk height and every site a sensor rays from describe the
        # state one substep (5 ms) before `qpos`. The same four calls, so
        # tests/test_arena.py's step-for-step lock holds to the bit.
        mujoco.mj_kinematics(m, data)
        mujoco.mj_comPos(m, data)
        mujoco.mj_comVel(m, data)
        mujoco.mj_sensorVel(m, data)
        for p in self.persons.values():
            p.sync(data)
            if p.fallen(data):
                p.reset(data)
                mujoco.mj_forward(m, data)
                p.sync(data)
        self.t += C.CTRL_DT
        self.tick += 1
        self._stamp_bumps(pairs)
        for d in self.ducks.values():
            d.step_count += 1
            if d.down_until >= 0.0:
                if self.getup_infer is not None and not d.fallen(data):
                    if d.up_since < 0.0:
                        d.up_since = self.t   # first tick upright: start the dwell
                    if self.t - d.up_since >= self.getup_hold_s:
                        self._price_spell(d)
                        d.down_until = d.up_since = -1.0   # up and STAYING up: the walker has it back
                        self.getups += 1
                elif self.getup_infer is not None and d.fallen(data):
                    d.up_since = -1.0         # back over the line: the dwell starts again
                if d.down_until >= 0.0 and self.t >= d.down_until:  # the clock, or the get-up's timeout
                    self._price_spell(d)
                    if self.getup_infer is not None:
                        self.getup_timeouts += 1
                    self._respawn(d)
                    mujoco.mj_forward(m, data)
                    d.prev_joint_vel = d.joint_vel(data)
                continue                      # still down: counted at the fall, nothing more to do
            if d.fallen(data):
                d.falls += 1
                if self.getup_s > 0.0:
                    d.down_until, d.down_since = self.t + self.getup_s, self.t
                    continue
                self._respawn(d)
                mujoco.mj_forward(m, data)
                d.prev_joint_vel = d.joint_vel(data)
            elif d.step_count >= d.max_steps:
                self._respawn(d)
                mujoco.mj_forward(m, data)
                d.prev_joint_vel = d.joint_vel(data)
        for d in self.ducks.values():
            self._odom_step(d)
        if self._ball_joint is not None:
            self._check_goal()
            if self.ball_out_s > 0 and self._ball_joint is not None:
                self._check_ball_out()
        t1 = time.perf_counter()
        for d in self.ducks.values():
            if isinstance(d, WorldRobot):
                # Every channel this body declared, each at its own device
                # rate: the lidar's `maybe_scan` beside the ToF's `sample`
                # (`LidarSensor.maybe_scan` schedules off the grid for the
                # same reason `TofSensor.sample` does — a 6 Hz sensor polled
                # every step must not drift into a 5.9 Hz one).
                if d.lidar is not None:
                    d.lidar.maybe_scan(data, self.t)
                if d.detector is not None:
                    d.detector.sample(data, self.t)
                # The claw is a sense too (`sense_grip`): a driver-stepped
                # body's grasp is physics, so "am I holding something" is
                # READ each tick rather than remembered from an event.
                self.sense_grip(d)
                continue
            if d.tof is not None:
                d.tof.sample(data, self.t)
            if d.detector is not None:
                d.detector.sample(data, self.t)
        t2 = time.perf_counter()
        a = 0.02
        self.perf["stepMs"] += a * ((t1 - t0) * 1e3 - self.perf["stepMs"])
        self.perf["sensorMs"] += a * ((t2 - t1) * 1e3 - self.perf["sensorMs"])

    # -- payloads for the lab stream ------------------------------------------
    def duck_pose(self, duck_id: str) -> list[list[float]]:
        s = self.duck_bodies[duck_id]
        out = []
        for b in range(s.start, s.stop):
            p, q = self.data.xpos[b], self.data.xquat[b]
            out.append([round(float(v), 4) for v in (*p, *q)])
        return out

    def objects_payload(self) -> list[dict]:
        out = []
        held = {d.holding: d.id for d in self.ducks.values() if d.holding}
        for name, kind, b in self.objects:
            p, q = self.data.xpos[b], self.data.xquat[b]
            item = {"id": name, "kind": kind, "pose": [round(float(v), 4) for v in (*p, *q)]}
            if kind == "toy":
                item["toy"] = self.pickable_kind[name]
                item["held"] = held.get(name)
                item["inBasket"] = self.in_basket(name)
            out.append(item)
        return out

    def senses_tof(self, d) -> tuple:
        """(frame, age) for the 8x8 ToF a brain reads off this body.

        A duck's is its own sensor. A body with a planar scanner and no ToF
        gets one ADAPTED from the newest scan (`sensors.lidar.tof_from_lidar`,
        whose docstring has the approximation and what it costs), which is how
        `wander` and `follow` drive a MARS with nothing in
        `brain/controllers.py` edited. The age is the SCAN's age, unchanged by
        the adaptation: a 6 Hz scanner's frame is up to 167 ms old and a brain
        gating on freshness must see that, not a zero stamped by the
        conversion.

        One place, so the lab's stream, `record-world` and a test cannot
        disagree about what a wheeled body's brain was handed.
        """
        if d.tof is not None:
            f = d.tof.last
            return (f, None if f is None else self.t - f.t)
        lidar = getattr(d, "lidar", None)
        f = None if lidar is None else lidar.last
        if f is None:
            return (None, None)
        return (tof_from_lidar(f, footprint_m=getattr(d, "footprint_m", 0.0)),
                self.t - f.t)

    def sensors_payload(self, duck_id: str) -> dict | None:
        d = self.ducks[duck_id]
        out: dict = {}
        lidar = getattr(d, "lidar", None)
        if lidar is not None and lidar.last is not None:
            out["lidar"] = {**lidar.last.as_payload(),
                            "age": round(self.t - lidar.last.t, 4)}
        if d.tof is not None and d.tof.last is not None:
            out["tof"] = {**d.tof.last.as_payload(), "age": round(self.t - d.tof.last.t, 4)}
        if d.detector is not None and d.detector.last is not None:
            f = d.detector.last
            out["det"] = {"t": round(f.t, 4), "age": round(self.t - f.t, 4),
                          "fov": [d.detector.spec.fov_h_deg, d.detector.spec.fov_v_deg],
                          "items": [x.as_payload() for x in f.detections]}
        return out or None

    def persons_payload(self) -> list[dict]:
        return [p.payload(self.data) for p in self.persons.values()]

    def possess(self, person_id: str | None) -> None:
        """Hand one person to a human (None releases all). A possessed person
        follows `cmd` instead of its path."""
        for p in self.persons.values():
            p.possessed = p.id == person_id
            if not p.possessed:
                p.cmd = None
