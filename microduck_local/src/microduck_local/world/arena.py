"""`World`: N ducks, their reflex policies and their sensors in one mjData.

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
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np

from .. import contract as C
from ..sensors import Detector, DetectorNoise, Target, TofNoise, TofSensor
from ..walk_env import MicroduckWalkEnv
from .compose import DuckAddress, compose, spawn_duck
from .scenario import PICKABLE_KINDS, Person, Scenario

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
# The shipped kicks (ball_kick_left / ball_kick_right) run as a WINDOW, not a
# phase: the robot hands the reflex tier to the kick network for
# `kick_duration` (0.5 s in robotd's control.rs) with an all-zero command,
# then back to the walker. Same protocol here.
KICK_S = 0.5
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
    step_count: int = 0
    episodes: int = 0
    _hold_yaw: float | None = None

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


class WorldPerson:
    """A mocap capsule walking its waypoints at a set speed, or driven by a
    possessing human (`cmd` = [vx, vy, wz] in its own heading frame)."""

    def __init__(self, model: mujoco.MjModel, spec: Person):
        self.spec = spec
        self.id = spec.id
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.id)
        self.mocap = int(model.body_mocapid[self.body])
        self.x, self.y, self.yaw = spec.pos[0], spec.pos[1], spec.yaw
        self.wp = 0
        self.cmd: np.ndarray | None = None      # possessed: heading-frame twist
        self.possessed = False

    def reset(self, data: mujoco.MjData) -> None:
        self.x, self.y, self.yaw = self.spec.pos[0], self.spec.pos[1], self.spec.yaw
        self.wp = 0
        self.cmd = None
        self.write(data)

    def write(self, data: mujoco.MjData) -> None:
        data.mocap_pos[self.mocap] = [self.x, self.y, self.spec.height / 2]
        data.mocap_quat[self.mocap] = [np.cos(self.yaw / 2), 0.0, 0.0, np.sin(self.yaw / 2)]

    def step(self, data: mujoco.MjData, dt: float) -> None:
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
                self.yaw += float(np.clip(err, -1.5 * dt, 1.5 * dt))
                step = min(self.spec.speed * dt, dist)
                self.x += step * np.cos(self.yaw)
                self.y += step * np.sin(self.yaw)
        self.write(data)

    def payload(self) -> dict:
        return {"id": self.id, "kind": "person",
                "pose": [round(self.x, 4), round(self.y, 4), round(self.spec.height / 2, 4),
                         round(float(np.cos(self.yaw / 2)), 4), 0.0, 0.0, round(float(np.sin(self.yaw / 2)), 4)],
                "possessed": self.possessed}


class World:
    def __init__(self, scenario: Scenario, infer_for: dict[str, Infer] | None = None,
                 max_episode_s: float | None = None, seed: int | None = None):
        # No episode timeout by default: a world is a place, not an episode
        # (a 30 s default once respawned a duck mid-delivery, toy and all).
        # Training envs pass their own horizon.
        max_episode_s = float("inf") if max_episode_s is None else max_episode_s
        self.scenario = scenario
        self.model = compose(scenario)
        self.data = mujoco.MjData(self.model)
        self.t = 0.0
        self.tick = 0
        self.rng = np.random.default_rng(scenario.seed if seed is None else seed)
        infer_for = infer_for or {}
        self.ducks: dict[str, WorldDuck] = {}
        self.persons: dict[str, WorldPerson] = {
            p.id: WorldPerson(self.model, p) for p in scenario.persons}
        # What a detector can find: every duck's trunk, every ball, every person.
        targets = [Target(d.id, "duck", mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                                          f"{d.id}/trunk_base"), 0.10)
                   for d in scenario.ducks]
        targets += [Target(f"ball{i}", "ball", mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                                                f"ball{i}"), b.radius)
                    for i, b in enumerate(scenario.balls)]
        targets += [Target(p.id, "person", self.persons[p.id].body, p.radius, height=p.height) for p in scenario.persons]
        self.pickables: dict[str, int] = {
            t.id: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, t.id) for t in scenario.pickables}
        self.pickable_kind = {t.id: t.kind for t in scenario.pickables}
        targets += [Target(t.id, "toy", self.pickables[t.id],
                           max(PICKABLE_KINDS[t.kind]["size"]) / 2) for t in scenario.pickables]
        self.basket = scenario.basket
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
        for d in scenario.ducks:
            adr = DuckAddress.resolve(self.model, d.id)
            tof = None
            if d.tof is not None and adr.tof_site >= 0:
                tof = TofSensor(self.model, site=adr.prefix + "tof",
                                noise=TofNoise.preset(d.tof),
                                seed=int(self.rng.integers(0, 2**31 - 1)),
                                base_body=adr.trunk_body)
            det = None
            if d.detector is not None:
                det = Detector(self.model, site=adr.prefix + "head_camera",
                               noise=DetectorNoise.preset(d.detector), targets=targets,
                               seed=int(self.rng.integers(0, 2**31 - 1)))
            self.ducks[d.id] = WorldDuck(
                id=d.id, adr=adr, spawn=d.spawn, infer=infer_for.get(d.id, zero_infer),
                policy_id=d.policy, tof=tof, detector=det,
                odom_preset=d.odom, odom_noise=OdomNoise.preset(d.odom),
                max_steps=(2**62 if not np.isfinite(max_episode_s)
                           else int(round(max_episode_s / C.CTRL_DT))))
        # Body ranges per duck: the attached subtree is contiguous after the
        # trunk, so the viewer's per-duck body list is one slice.
        self.duck_bodies: dict[str, slice] = {}
        for d in self.ducks.values():
            sub = [b for b in range(self.model.nbody)
                   if self.model.body_rootid[b] == d.adr.trunk_body]
            assert sub == list(range(sub[0], sub[-1] + 1)), "duck subtree not contiguous"
            self.duck_bodies[d.id] = slice(sub[0], sub[-1] + 1)
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
        for p in self.persons.values():
            p.reset(self.data)
        for d in self.ducks.values():
            self._respawn(d)
        mujoco.mj_forward(self.model, self.data)
        for d in self.ducks.values():
            d.prev_joint_vel = d.joint_vel(self.data)

    def _respawn(self, d: WorldDuck) -> None:
        x, y, yaw = d.spawn
        spawn_duck(self.model, self.data, d.adr, x, y, yaw)
        self._odom_reset(d, x, y, yaw)
        d.last_action[:] = 0.0
        d.step_count = 0
        d._hold_yaw = None
        d.episodes += 1
        self.release(d)
        d.skill = None
        d.skill_infer = None
        self._set_gain_ratio(d, 1.0)
        if d.tof is not None:
            d.tof.reset()
        if d.detector is not None:
            d.detector.reset()

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
            self.goal_seq += 1
            self.kickoff()

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

    def goal_for(self, d: WorldDuck) -> tuple[float, float] | None:
        """The goal this duck attacks (world = odometry-at-spawn frame): the
        one its team is placed to face, by its spawn heading. None off a pitch."""
        if self.goal_width <= 0:
            return None
        hx = self.scenario.floor[0] / 2 - 0.25
        return (hx if math.cos(d.spawn[2]) >= 0 else -hx), 0.0

    def soccer_score(self) -> dict | None:
        if self._ball_joint is None:
            return None
        q = int(self.model.jnt_qposadr[self._ball_joint])
        return {"left": self.goals["left"], "right": self.goals["right"],
                "ball": [round(float(self.data.qpos[q]), 3), round(float(self.data.qpos[q + 1]), 3)],
                "lastGoal": self.last_goal, "kickoff": round(max(0.0, self.kickoff_until - self.t), 2)}

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
        self.data.eq_active[self._eq_id(d, toy)] = 0
        d.holding = None
        return toy

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

    def start_skill(self, d: WorldDuck, name: str) -> bool:
        """Hand the reflex tier to a skill policy for one cycle (the robot's
        own pattern: hard swap in, auto swap back): ground_pick (a phase
        cycle) or kick_left / kick_right (a 0.5 s window)."""
        if name not in SKILLS or d.skill is not None:
            return False
        if name not in self.skills:
            from ..brain.brain_env import POLICIES_DIR, onnx_infer
            path = POLICIES_DIR / SKILLS[name]
            if not path.exists():
                return False
            self.skills[name] = onnx_infer(path)
        d.skill, d.skill_t0, d.skill_infer = name, self.t, self.skills[name]
        d._hold_yaw = None
        if name.startswith("kick"):
            self._set_gain_ratio(d, STANDING_GAIN_RATIO)
        if d.holding is None:
            d.beak_closed = False          # a cycle starts with an open, empty beak
        return True

    def in_basket(self, toy: str) -> bool:
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

    def apply_intent(self, d: WorldDuck, intent) -> None:
        """Route a brain's non-twist intents to the reflex tier."""
        if intent.skill:
            self.start_skill(d, intent.skill)
        if intent.beak == "close" and not d.beak_closed:
            self.grasp(d)
        elif intent.beak == "open" and d.beak_closed:
            self.release(d)

    def _skill_cmd(self, d: WorldDuck) -> Infer | None:
        """While a skill runs, it owns the command block; returns the infer to
        use, ending the cycle at its exit phase."""
        if d.skill is None:
            return None
        if d.skill.startswith("kick"):
            if self.t - d.skill_t0 >= KICK_S:
                d.skill, d.skill_infer = None, None
                d.twist_cmd[:] = 0.0
                self._set_gain_ratio(d, 1.0)
                return None
            d.twist_cmd[:] = 0.0                  # the kick's observation carries an all-zero command
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
            if hold:                          # kickoff: stand, whatever the brain asked
                d.set_cmd(data, (0.0, 0.0, 0.0))
            skill = self._skill_cmd(d)
            obs = d.obs(data)
            raw = np.asarray((skill or d.infer)(obs), np.float32)
            d.last_action = raw.copy()
            data.ctrl[d.adr.actuators] = C.DEFAULT_POSE + raw.clip(-4.0, 4.0)
        for p in self.persons.values():
            p.step(data, C.CTRL_DT)
        for _ in range(C.DECIMATION):
            mujoco.mj_step(m, data)
        self.t += C.CTRL_DT
        self.tick += 1
        for d in self.ducks.values():
            d.step_count += 1
            if d.fallen(data):
                d.falls += 1
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
        t1 = time.perf_counter()
        for d in self.ducks.values():
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

    def sensors_payload(self, duck_id: str) -> dict | None:
        d = self.ducks[duck_id]
        out: dict = {}
        if d.tof is not None and d.tof.last is not None:
            out["tof"] = {**d.tof.last.as_payload(), "age": round(self.t - d.tof.last.t, 4)}
        if d.detector is not None and d.detector.last is not None:
            f = d.detector.last
            out["det"] = {"t": round(f.t, 4), "age": round(self.t - f.t, 4),
                          "fov": [d.detector.spec.fov_h_deg, d.detector.spec.fov_v_deg],
                          "items": [x.as_payload() for x in f.detections]}
        return out or None

    def persons_payload(self) -> list[dict]:
        return [p.payload() for p in self.persons.values()]

    def possess(self, person_id: str | None) -> None:
        """Hand one person to a human (None releases all). A possessed person
        follows `cmd` instead of its path."""
        for p in self.persons.values():
            p.possessed = p.id == person_id
            if not p.possessed:
                p.cmd = None
