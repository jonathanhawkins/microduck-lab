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
from ..world import Box, Duck, Person, Scenario, World
from ..world.scenario import TOF_PRESETS
from .runtime import Senses
from .controllers import ClosingWatch
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
        # `pos` places each hit in the ODOMETRY frame as well (Tracker._place),
        # which is what lets a brain dead-reckon to a target that has gone
        # under the camera — the striker's last 0.3 m onto the ball. It adds
        # `Track.xy`/`vel` and touches no field any of these 80 floats read,
        # so a follow brain's observation is bit-for-bit what it was.
        tracker.update(det, s.t, None if s.odom is None else s.odom[2],
                       None if s.odom is None else (s.odom[0], s.odom[1]))
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
    # The reflex tier under the brain (both on the robot, neither learned):
    # the head yaws toward the tracked target so the 62° camera keeps it
    # while the body catches up (`gaze_gain` × body bearing, clipped to
    # `gaze_max`), and a forward command is refused with something inside
    # `bump_stop` ahead (0 = off). Version-1 observations (no tracker)
    # get neither, so a version-1 brain sees the world it was trained in.
    gaze_gain: float = 0.8
    gaze_max: float = 0.6
    bump_stop: float = 0.25
    # Variety: `furniture` free boxes re-scattered each episode, and a
    # second duck walking a slow circle as a moving non-target.
    furniture: int = 0
    distractor: bool = False
    # The person turns and walks straight at the duck every `charge` s (0:
    # never) - through where it stands, not to it. The case the sidestep
    # reflex is for: `avoid` gives the brain the same ClosingWatch the
    # scripted Follow carries (a sidestep out of the path of whatever
    # closes on the duck faster than its own walk), version-2 only.
    charge: float = 0.0
    avoid: bool = False
    # A polite person stops this far short of the duck in its way (centre
    # to centre; its surface 0.35 m from the duck at 0.55) and steps on to
    # its next waypoint after 2.5 s, instead of walking through it as a
    # mocap capsule does. The default, because real people stop; 0 is the
    # capsule that walks through, measured to cap every brain's band for
    # a reason that has nothing to do with following.
    polite: float = 0.55


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
        self._closing = ClosingWatch()
        self._last_charge = 0.0
        self.charges = 0
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
        t = self.task
        ducks = [Duck("d0", (0.0, 0.0, 0.0), None, preset, preset)]
        if t.distractor:
            ducks.append(Duck("d1", (-1.5, 1.2, 0.0), None, None, None))
        boxes = [Box((2.0 + 0.5 * i, -1.5, 0.15), (0.3, 0.3, 0.3), mass=3.0) for i in range(t.furniture)]
        sc = Scenario(name="brain-follow", floor=t.room, ducks=ducks, boxes=boxes,
                      persons=[Person("p0", (1.0, 0.0), 0.0, path=self._random_path(),
                                      speed=float(self.rng.uniform(*t.person_speed)), yield_m=t.polite)])
        infer = {"d0": self._infer}
        if t.distractor:
            infer["d1"] = self._infer
        self.world = World(sc, infer_for=infer, max_episode_s=1e9,
                           seed=int(self.rng.integers(0, 2**31 - 1)))
        self.duck = self.world.ducks["d0"]
        self.person = self.world.persons["p0"]
        self.distractor = self.world.ducks.get("d1")
        self._box_joints = [j for j in range(self.world.model.njnt)
                            if self.world.model.jnt_type[j] == 0 and "/" not in self.world.model.body(
                                int(self.world.model.jnt_bodyid[j])).name
                            and self.world.model.body(int(self.world.model.jnt_bodyid[j])).name.startswith("box")]

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
        # Furniture lands anywhere but on the duck's start and the person's.
        hx, hy = self.task.room[0] / 2 - 0.4, self.task.room[1] / 2 - 0.4
        for j in self._box_joints:
            q = int(self.world.model.jnt_qposadr[j])
            for _ in range(20):
                bx, by = float(self.rng.uniform(-hx, hx)), float(self.rng.uniform(-hy, hy))
                if math.hypot(bx, by) > 0.7 and math.hypot(bx - p.spec.pos[0], by - p.spec.pos[1]) > 0.6:
                    break
            self._box_pending = getattr(self, "_box_pending", [])
            self._box_pending.append((q, bx, by))
        if self.sense_dr and self.fixed_preset is None:
            name = str(self.rng.choice(TOF_PRESETS))
            self.duck.tof.noise = TofNoise.preset(name)
            self.duck.detector.noise = DetectorNoise.preset(str(self.rng.choice(TOF_PRESETS)))

    # -- gym -----------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._box_pending = []
        self._randomize_episode()
        self.world.reset()
        # An episode is a pure function of (seed, ep): nothing rides in from
        # the episode before it. Three generators outlive world.reset(), so
        # the episode's own rng re-seeds all three here.
        #   - the sensors': seeded once in World.__init__, and their reset()
        #     clears the frames but not the stream, so episode k used to
        #     continue episode k-1's noise;
        #   - the world's: dormant on this task (the follow duck's odom
        #     preset is "ideal", and there is nothing to grasp or kick off),
        #     but re-seeded so that stays true if the task ever turns those
        #     on rather than silently losing the guarantee.
        # _respawn clears last_action, the skill and the gains but not the
        # commanded twist, so the warm-up steps below used to be driven by
        # whatever the last episode ended up asking for - zero it too.
        for _gen in (self.duck.tof, self.duck.detector, self.world):
            if _gen is not None:
                _gen.rng = np.random.default_rng(int(self.rng.integers(0, 2**31 - 1)))
        self.duck.twist_cmd[:] = 0.0
        self.duck.head_cmd[:] = 0.0
        for q, bx, by in self._box_pending:              # after the reset, which re-poses every free body
            self.world.data.qpos[q:q + 3] = [bx, by, 0.15]
            self.world.data.qpos[q + 3:q + 7] = [1.0, 0.0, 0.0, 0.0]
            v = int(self.world.model.jnt_dofadr[[j for j in self._box_joints if int(self.world.model.jnt_qposadr[j]) == q][0]])
            self.world.data.qvel[v:v + 6] = 0.0
        # Let the first sensor frames land before the first decision.
        for _ in range(self.decide_every):
            self.world.step()
        self._last_action[:] = 0.0
        self._last_seen_t = None
        self._builder.reset()
        self._closing.reset()
        self._last_charge = self.world.t
        self._steps = 0
        return self._obs(), {}

    def _charge(self) -> None:
        """The person's next waypoint: 0.4 m past the duck on the line from
        where the person is - it walks through the duck's spot."""
        w, d, p = self.world, self.duck, self.person
        pos = d.trunk_pos(w.data)
        dx, dy = float(pos[0] - p.x), float(pos[1] - p.y)
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return
        p.spec.path.insert(p.wp, (float(pos[0] + 0.4 * dx / n), float(pos[1] + 0.4 * dy / n)))
        self.charges += 1

    def senses(self) -> Senses:
        w, d = self.world, self.duck
        tof = d.tof.last if d.tof is not None else None
        det = d.detector.last if d.detector is not None else None
        return Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                      det=det, det_age=None if det is None else w.t - det.t,
                      speed=d.heading_speed(w.data), odom=w.odom(d), bumped=w.bumped(d))

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

    def gaze(self) -> float:
        """The reflex tier's head yaw: toward the tracked target (version 2+)."""
        tr = self._builder.tracker
        if tr is None or not self.task.gaze_gain:
            return 0.0
        best = tr.best(self.task.target_cls, self.world.t, min_hits=1)
        if best is None or best.age(self.world.t) > 1.0:
            return 0.0
        return float(np.clip(self.task.gaze_gain * best.bearing, -self.task.gaze_max, self.task.gaze_max))

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, np.float32), ACT_LOW, ACT_HIGH)
        w, d, t = self.world, self.duck, self.task
        if t.bump_stop and d.tof is not None and d.tof.last is not None and self._builder.tracker is not None:
            mid = d.tof.last.depth_mm[2:6, 3:5]
            if (mid[(mid > 0)] < t.bump_stop * 1000).any() and a[0] > 0:
                a = a.copy()
                a[0] = 0.0                                  # the reflex tier refuses to walk into it
        if t.avoid and d.tof is not None and self._builder.tracker is not None:
            dodge = self._closing.step(d.tof.last, w.t, d.heading_speed(w.data))
            if dodge is not None:
                a = np.array(dodge, np.float32)             # the reflex tier gets out of its path
        d.set_cmd(w.data, a, (0.0, 0.0, self.gaze(), 0.0))
        if t.charge and w.t - self._last_charge >= t.charge:
            self._last_charge = w.t
            self._charge()
        if self.distractor is not None:
            self.distractor.set_cmd(w.data, (0.25, 0.0, 0.6))    # a slow circle (below 0.2 the walker stands)
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
        # Contact (truth, for the benchmark): the person's capsule against the body.
        contact = dist < self.person.spec.radius + 0.10
        info = {"dist": dist, "bearing": bearing, "seen": seen, "bumped": bumped, "contact": contact,
                "dodges": self._closing.count}
        return self._obs(), float(reward), terminated, truncated, info

    def close(self) -> None:
        pass


__all__ = ["ACT_HIGH", "ACT_LOW", "BRAIN_ACT_DIM", "BRAIN_OBS_DIM", "BRAIN_OBS_VERSION", "BrainEnv", "FollowTask",
           "ObsBuilder", "onnx_infer", "senses_to_obs"]
