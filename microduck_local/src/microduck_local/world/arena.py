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

import time
from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np

from .. import contract as C
from ..sensors import Detector, DetectorNoise, Target, TofNoise, TofSensor
from ..walk_env import MicroduckWalkEnv
from .compose import DuckAddress, compose, spawn_duck
from .scenario import Person, Scenario

Infer = Callable[[np.ndarray], np.ndarray]
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
                 max_episode_s: float = 30.0, seed: int | None = None):
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
        targets += [Target(p.id, "person", self.persons[p.id].body, p.radius) for p in scenario.persons]
        for d in scenario.ducks:
            adr = DuckAddress.resolve(self.model, d.id)
            tof = None
            if d.tof is not None and adr.tof_site >= 0:
                tof = TofSensor(self.model, site=adr.prefix + "tof",
                                noise=TofNoise.preset(d.tof),
                                seed=int(self.rng.integers(0, 2**31 - 1)))
            det = None
            if d.detector is not None:
                det = Detector(self.model, site=adr.prefix + "head_camera",
                               noise=DetectorNoise.preset(d.detector), targets=targets,
                               seed=int(self.rng.integers(0, 2**31 - 1)))
            self.ducks[d.id] = WorldDuck(
                id=d.id, adr=adr, spawn=d.spawn, infer=infer_for.get(d.id, zero_infer),
                policy_id=d.policy, tof=tof, detector=det,
                max_steps=int(round(max_episode_s / C.CTRL_DT)))
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
            kind = "ball" if name.startswith("ball") else "box"
            self.objects.append((name, kind, b))
        # Rolling cost of one control step, split physics / sensors (ms),
        # EMA over ~1 s of ticks — the /sim perf HUD reads it.
        self.perf = {"stepMs": 0.0, "sensorMs": 0.0}
        self.reset()

    # -- lifecycle ------------------------------------------------------------
    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.t = 0.0
        self.tick = 0
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
        d.last_action[:] = 0.0
        d.step_count = 0
        d._hold_yaw = None
        d.episodes += 1
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
    def step(self) -> None:
        t0 = time.perf_counter()
        m, data = self.model, self.data
        for d in self.ducks.values():
            obs = d.obs(data)
            raw = np.asarray(d.infer(obs), np.float32)
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
        for name, kind, b in self.objects:
            p, q = self.data.xpos[b], self.data.xquat[b]
            out.append({"id": name, "kind": kind,
                        "pose": [round(float(v), 4) for v in (*p, *q)]})
        return out

    def sensors_payload(self, duck_id: str) -> dict | None:
        d = self.ducks[duck_id]
        out: dict = {}
        if d.tof is not None and d.tof.last is not None:
            out["tof"] = {**d.tof.last.as_payload(), "age": round(self.t - d.tof.last.t, 4)}
        if d.detector is not None and d.detector.last is not None:
            f = d.detector.last
            out["det"] = {"t": round(f.t, 4), "age": round(self.t - f.t, 4),
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
