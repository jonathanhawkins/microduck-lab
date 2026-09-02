"""`BrainEnv`: reinforcement learning for the brain layer (roadmap 3.1).

A gymnasium env whose ACTION is an intent — a twist for the unchanged
reflex walker underneath — decided every `decide_every` control steps
(10 Hz by default), and whose OBSERVATION is what a brain can actually get
on the robot: the 64 ToF zones, the nearest detection of the target class
(bearing, elevation, width, range, confidence, age), the last intent, and
the duck's own heading speed. The walker is a frozen ONNX (the shipped
`alpha_walking` unless told otherwise), so the policy being trained is a
small MLP over ~80 features and physics costs what it costs today.

Domain randomization is on the SENSES (noise preset drawn per episode,
extra dropouts, latency), never on physics: the roadmap's claim is that
the sim2real gap of a brain is perception, and this is where that claim
gets tested.

The first task is follow-me (3.2): a person walks a random path; reward
for holding a distance band and a small bearing, penalties for losing
sight, for walking into things (ToF bumper), and for jerky intents. The
scripted `Follow` controller runs the same scenario as the baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np

from .. import contract as C
from ..sensors import DetectorNoise, TofNoise
from ..world import Duck, Person, Scenario, World
from ..world.scenario import TOF_PRESETS
from .runtime import Senses
from .tracker import Tracker

POLICIES_DIR = C.MICRODUCK_RL_DIR.parent / "microduck" / "policies"

# Observation layout (contract for exported brains — keep in this order):
#   [0:64]   ToF zones, metres (0 = no target), clipped to 4 m
#   [64]     ToF age, s (clipped 1)
#   [65:71]  target: seen(0/1), bearing, elevation, width, range (clipped 4), conf
#   [71]     detection age, s (clipped 1)
#   [72]     time since the target was last seen, s (clipped 5)
#   [73:76]  last intent twist
#   [76]     heading speed, m/s
#   [77:80]  reserved (zero) in version 1; version 2 (below) fills them
#
# Version 2 (the tracker + odometry features, brain.json "obs_version": 2):
#   [65:71]  the TRACK of the target (brain/tracker.py: smoothed, coasting
#            with the body through misses): hit-this-frame(0/1), bearing,
#            elevation, width, range (clipped 4), conf
#   [72]     time since the track was last HIT, s (clipped 5)
#   [77]     track coasting (0/1: a track exists but was not hit this frame)
#   [78]     own yaw rate from odometry, rad/s (clipped ±3)
#   [79]     track confirmed (0/1: two hits or more)
# Same 80 floats, same walker underneath: a version-1 brain still runs on a
# version-1 observation (LearnedBrain reads the version from brain.json).
BRAIN_OBS_DIM = 80
BRAIN_OBS_VERSION = 2
BRAIN_ACT_DIM = 3          # vx, vy, wz (vy is emitted but kept small by the walker)
ACT_LOW = np.array([-0.2, -0.3, -1.0], np.float32)
ACT_HIGH = np.array([0.6, 0.3, 1.0], np.float32)


def senses_to_obs(s: Senses, target_cls: str, last_action: np.ndarray,
                  last_seen_t: float | None, tracker: Tracker | None = None,
                  yaw_rate: float = 0.0, det_max_age: float = 0.4) -> tuple[np.ndarray, float | None]:
    """The brain observation contract (layout above), from what a brain gets.
    Returns (obs, updated last_seen_t). Shared by BrainEnv (training) and
    LearnedBrain (inference in the world), so the two can never drift.
    With a `tracker` (version 2) the target slots come from its track of
    the target class and the reserved slots carry the coasting flag, the
    odometry yaw rate and the confirmation; without one, version 1."""
    o = np.zeros(BRAIN_OBS_DIM, np.float32)
    if s.tof is not None:
        o[0:64] = np.clip(s.tof.depth_mm.reshape(-1) / 1000.0, 0.0, 4.0)
        o[64] = min(s.tof_age or 0.0, 1.0)
    else:
        o[64] = 1.0
    if tracker is not None:
        det = s.fresh_det(det_max_age)
        new_frame = det is not None and det.t != tracker._last_frame_t
        tracker.update(det, s.t, None if s.odom is None else s.odom[2])
        tr = tracker.best(target_cls, s.t, min_hits=1)
        if tr is not None:
            hit = new_frame and tr.last_t == det.t                # a NEW frame's detection updated the track
            o[65:71] = [1.0 if hit else 0.0, tr.bearing, tr.elevation, tr.width, min(tr.range, 4.0), tr.conf]
            o[77] = 0.0 if hit else 1.0
            o[79] = 1.0 if tr.hits >= tracker.p.confirm_hits else 0.0
            if hit:
                last_seen_t = tr.last_t
        o[78] = float(np.clip(yaw_rate, -3.0, 3.0))
    else:
        tgt = None
        if s.det is not None:
            cands = [x for x in s.det.detections if x.cls == target_cls]
            if cands:
                tgt = min(cands, key=lambda x: x.range_est)
        if tgt is not None:
            o[65:71] = [1.0, tgt.bearing, tgt.elevation, tgt.width, min(tgt.range_est, 4.0), tgt.conf]
            last_seen_t = s.t
    o[71] = min(s.det_age if s.det_age is not None else 1.0, 1.0)
    o[72] = 5.0 if last_seen_t is None else min(s.t - last_seen_t, 5.0)
    o[73:76] = last_action
    o[76] = s.speed or 0.0
    return o, last_seen_t


class ObsBuilder:
    """The per-brain state `senses_to_obs` needs across calls: the tracker
    (version 2), the last-seen clock and the previous heading for the yaw
    rate. One in the env, one in the LearnedBrain — same code path."""

    def __init__(self, target_cls: str, version: int = BRAIN_OBS_VERSION):
        self.target_cls = target_cls
        self.version = int(version)
        self.tracker = Tracker() if self.version >= 2 else None
        self.reset()

    def reset(self) -> None:
        self.last_seen_t: float | None = None
        self._prev: tuple[float, float] | None = None       # (t, yaw)
        if self.tracker is not None:
            self.tracker.reset()

    def __call__(self, s: Senses, last_action: np.ndarray) -> np.ndarray:
        yaw_rate = 0.0
        if s.odom is not None:
            if self._prev is not None and s.t > self._prev[0]:
                d = math.atan2(math.sin(s.odom[2] - self._prev[1]), math.cos(s.odom[2] - self._prev[1]))
                yaw_rate = d / (s.t - self._prev[0])
            self._prev = (s.t, s.odom[2])
        o, self.last_seen_t = senses_to_obs(s, self.target_cls, last_action, self.last_seen_t,
                                            self.tracker, yaw_rate)
        return o


def onnx_infer(path: Path) -> Callable[[np.ndarray], np.ndarray]:
    """A batch-1 ONNX policy on ONE thread: a dozen training workers each
    spinning a full thread pool for a 61-float MLP measured 170 decisions/s
    where a single env alone does ~390."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=opts)
    name = sess.get_inputs()[0].name
    return lambda obs: sess.run(None, {name: obs[None]})[0][0].astype(np.float32)


@dataclass
class FollowTask:
    target_cls: str = "person"
    distance: float = 0.7          # centre of the reward band
    band: float = 0.25             # ± half-width where the distance reward is full
    room: tuple[float, float] = (5.0, 4.0)
    person_speed: tuple[float, float] = (0.15, 0.35)
    episode_s: float = 20.0
    lose_penalty: float = 0.5
    bump_penalty: float = 1.0
    jerk_penalty: float = 0.05


class BrainEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task: FollowTask = FollowTask(), walker: str | Path | None = None,
                 decide_every: int = 5, sense_dr: bool = True, seed: int | None = None,
                 fixed_preset: str | None = None, obs_version: int = BRAIN_OBS_VERSION):
        super().__init__()
        self.task = task
        self.obs_version = int(obs_version)
        self._builder = ObsBuilder(task.target_cls, self.obs_version)
        self.decide_every = int(decide_every)
        self.sense_dr = sense_dr
        self.fixed_preset = fixed_preset
        self.rng = np.random.default_rng(seed)
        walker = Path(walker) if walker else POLICIES_DIR / "alpha_walking.onnx"
        if not walker.exists():
            raise FileNotFoundError(f"walker policy {walker} not found (clone pollen-robotics/microduck)")
        self._infer = onnx_infer(walker)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (BRAIN_OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(ACT_LOW, ACT_HIGH, dtype=np.float32)
        self.world: World | None = None
        self._last_action = np.zeros(3, np.float32)
        self._last_seen_t: float | None = None
        self._steps = 0
        self.max_decisions = int(round(task.episode_s / C.CTRL_DT / self.decide_every))
        self._build_world()

    # -- world ---------------------------------------------------------------
    def _random_path(self) -> list[tuple[float, float]]:
        hx, hy = self.task.room[0] / 2 - 0.5, self.task.room[1] / 2 - 0.5
        n = int(self.rng.integers(3, 6))
        return [(float(self.rng.uniform(-hx, hx)), float(self.rng.uniform(-hy, hy))) for _ in range(n)]

    def _build_world(self) -> None:
        preset = self.fixed_preset or "datasheet"
        sc = Scenario(name="brain-follow", floor=self.task.room,
                      ducks=[Duck("d0", (0.0, 0.0, 0.0), None, preset, preset)],
                      persons=[Person("p0", (1.0, 0.0), 0.0, path=self._random_path(),
                                      speed=float(self.rng.uniform(*self.task.person_speed)))])
        self.world = World(sc, infer_for={"d0": self._infer}, max_episode_s=1e9,
                           seed=int(self.rng.integers(0, 2**31 - 1)))
        self.duck = self.world.ducks["d0"]
        self.person = self.world.persons["p0"]

    def _randomize_episode(self) -> None:
        p = self.person
        p.spec.path[:] = self._random_path()
        p.spec.speed = float(self.rng.uniform(*self.task.person_speed))
        # Spawn the person ahead-ish, at a random range and bearing, so the
        # first frames usually contain it (an episode that starts lost is a
        # search task, which is fine sometimes).
        r = float(self.rng.uniform(0.6, 1.6))
        a = float(self.rng.uniform(-0.6, 0.6))
        p.spec.pos = (r * math.cos(a), r * math.sin(a))
        p.spec.yaw = float(self.rng.uniform(-math.pi, math.pi))
        if self.sense_dr and self.fixed_preset is None:
            name = str(self.rng.choice(TOF_PRESETS))
            self.duck.tof.noise = TofNoise.preset(name)
            self.duck.detector.noise = DetectorNoise.preset(str(self.rng.choice(TOF_PRESETS)))

    # -- gym -----------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._randomize_episode()
        self.world.reset()
        # Let the first sensor frames land before the first decision.
        for _ in range(self.decide_every):
            self.world.step()
        self._last_action[:] = 0.0
        self._last_seen_t = None
        self._builder.reset()
        self._steps = 0
        return self._obs(), {}

    def senses(self) -> Senses:
        w, d = self.world, self.duck
        tof = d.tof.last if d.tof is not None else None
        det = d.detector.last if d.detector is not None else None
        return Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                      det=det, det_age=None if det is None else w.t - det.t,
                      speed=d.heading_speed(w.data), odom=w.odom(d))

    def _obs(self) -> np.ndarray:
        o = self._builder(self.senses(), self._last_action)
        self._last_seen_t = self._builder.last_seen_t
        return o

    def _truth(self) -> tuple[float, float]:
        """(distance, bearing) to the person, from the sim — for REWARD only."""
        d = self.duck
        pos = d.trunk_pos(self.world.data)
        dx, dy = self.person.x - pos[0], self.person.y - pos[1]
        dist = float(math.hypot(dx, dy))
        yaw = d.yaw(self.world.data)
        bearing = math.atan2(math.sin(math.atan2(dy, dx) - yaw), math.cos(math.atan2(dy, dx) - yaw))
        return dist, float(bearing)

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, np.float32), ACT_LOW, ACT_HIGH)
        w, d, t = self.world, self.duck, self.task
        d.set_cmd(w.data, a)
        bumped = False
        falls0 = d.falls
        for _ in range(self.decide_every):
            w.step()
            tof = d.tof.last
            if tof is not None:
                mid = tof.depth_mm[3:6, 3:5]
                if (mid[(mid > 0)] < 220).any():
                    bumped = True
        dist, bearing = self._truth()
        # Reward: in the band and facing it is worth 1 per decision; the
        # bearing term keeps the person in the frame; losing sight costs;
        # bumping (ToF says something is right there) costs; jerk costs.
        in_band = max(0.0, 1.0 - max(0.0, abs(dist - t.distance) - t.band) / 0.6)
        facing = max(0.0, 1.0 - abs(bearing) / 0.7)
        reward = 0.6 * in_band + 0.4 * facing * in_band
        seen = self._last_seen_t is not None and (w.t - self._last_seen_t) < 0.5
        if not seen:
            reward -= t.lose_penalty * 0.1
        if bumped:
            reward -= t.bump_penalty * 0.1
        reward -= t.jerk_penalty * float(np.abs(a - self._last_action).sum())
        self._last_action = a
        self._steps += 1
        fell = d.falls > falls0
        terminated = bool(fell)
        truncated = self._steps >= self.max_decisions
        if fell:
            reward -= 1.0
        info = {"dist": dist, "bearing": bearing, "seen": seen, "bumped": bumped}
        return self._obs(), float(reward), terminated, truncated, info

    def close(self) -> None:
        pass


__all__ = ["ACT_HIGH", "ACT_LOW", "BRAIN_ACT_DIM", "BRAIN_OBS_DIM", "BRAIN_OBS_VERSION", "BrainEnv", "FollowTask",
           "ObsBuilder", "onnx_infer", "senses_to_obs"]
