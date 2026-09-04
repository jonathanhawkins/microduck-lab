"""`StrikerTask`/`StrikerEnv`: a LEARNED chase brain (roadmap 4.4, with 3.5's
hierarchical head in its simplest form).

The first learned brain in this repo that plays soccer. Same shape as
`FollowTask`: features in, an intent out at 10 Hz over the frozen reflex
walker. Two things make it a striker rather than a follower:

**It can kick.** The action is the twist PLUS two kick logits, one per
foot. Over threshold, the brain asks the reflex tier for the shipped
`ball_kick_left` / `ball_kick_right` skill exactly as the scripted `Chase`
does — a discrete option on top of the continuous twist. A brain that can
only walk cannot compete with one that kicks: measured here on a standing
duck, a ball at body-frame (+0.08, +0.06) flies 2.35 m off the left foot
(+26.5 deg off the body heading) and (+0.08, -0.06) flies 2.37 m off the
right (-21.8 deg); the ball sits on the SAME side as the kicking foot,
and the other side does not move it at all.

**It can see the goal.** The observation is the 80-float brain contract
(`brain_env.senses_to_obs`, version 2 — ToF, the ball TRACK, the last
intent, the odometry yaw rate) with eight floats appended: where the goal
is, where the ball is, and which way the ball has to go, all in the body
frame and all derived from the two things a real robot has — odometry and
detections. The scripted `Chase` reads exactly this and nothing more.
That matters for the reward: the AGENTS.md rule is never to pay for what
the policy cannot observe, and "the ball moved toward the goal" is a
world-anchored quantity that a brain WITHOUT slots 80-86 could only
experience as noise.

The reward IS the benchmark's metric (`world/metrics.py`): the signed
displacement of the ball toward the goal this duck attacks, per decision.
Signed, not the positive part, precisely because `ballAdvance` is
inflatable by churn (see `eval_pitch`'s docstring) and a reward that can
be farmed will be. On top of it: a telescoping approach term (a potential,
so it cannot be cycled for pay), the fall penalty, and a jerk penalty.

Nothing else is priced. What makes the kick appear in rollouts is the
SPAWN LADDER, not the pay: a quarter of episodes start with the ball
already on a kicking foot's sweet spot, so the state where a kick is worth
2.3 m of progress is sampled from the first minute instead of never
(AGENTS.md "Reward design rules": if the skill is not in the rollouts,
change the world, not the reward). Every family is scored by the identical
terms.

MEASURED, AND IT LOSES. `striker-v1` (600k decisions, 8 envs, ~30 min on a
4-core Linux box) against the scripted `Chase` on identical seeds, via
`eval_striker`:

  1v0 solo pitch, 24 paired seeds x 300 s
                    advance      progress      adv/kick   poss   kicks  goals  falls
    scripted chase  0.86+-0.39   +0.42+-0.44     0.451     11.8    229    35      0
    striker-v1      0.31+-0.20   +0.14+-0.28     0.016      5.9   2350    19     14
    paired striker-chase: advance -0.550 m/min (t=-5.74, 22 of 24 seeds DOWN)

  1v1 against a scripted chase, 36 paired seeds (24 + 12 FRESH) x 300 s,
  reading the SWAPPED side: advance -0.068 (t=-2.57), signed progress
  -0.26 m/min (t=-4.63) - and the sign is the finding: the scripted brain
  carries the ball toward the goal it attacks (+0.10 m/min) and this one
  carries it toward its own (-0.16). Both halves agree (-0.067 fresh,
  -0.068 discovery), which is what makes it a result and not a seed.

What it does, from the contact sheets (`render_striker`): it does not reach
the ball. It drifts at ~0.04 m/s - a fifth of the walker's pace - and fires
the kick option at whatever rate the tier allows (3612 kicks against the
scripted brain's 116, advance per kick 0.012 m against 0.512, i.e. 43x less
ball moved per touch). The ball simply never moves in most episodes. The
per-kick number is the one that says so: `ballAdvance` alone would read as
merely "a bit lower", because advance is inflated by churn and this brain is
nothing but churn.

Two failures were diagnosed and fixed along the way and both were in the
WORLD, not the pay (the constants above carry the numbers): a kick fired
mid-stride is the fall mode, and a kick that can re-fire immediately is a
LOCK that stops the duck walking at all. What is left is not a reward
problem either - the reward pays the scripted brain +6.3 an episode and this
one about -0.5 - it is that 600k decisions of PPO over an 88-float
observation did not find "walk to the ball, line up, kick toward the goal",
which the scripted brain spends 900 lines being told. The next lever is the
approach: a striker that reached the ball as often as `Chase` does
(possession 11.8 s/min against 5.9) would be worth measuring; this one is
not.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from .. import contract as C
from ..sensors import DetectorNoise, TofNoise
from ..world import Ball, Duck, Scenario, Wall, World
from ..world.scenario import TOF_PRESETS
from .brain_env import BRAIN_OBS_DIM, POLICIES_DIR, ObsBuilder, onnx_infer
from .controllers import tof_clearance_bearings
from .runtime import Intent, Senses

# --- the striker observation -------------------------------------------------
# [0:80]   the brain contract, version 2 (brain_env.senses_to_obs), target
#          class "ball": ToF, the ball track, last intent, yaw rate.
# [80]     sin(bearing to the goal we attack), body frame
# [81]     cos(bearing to the goal we attack)
# [82]     range to the goal, m / 4, clipped 1
# [83]     sin(bearing of the ball -> goal line), body frame
# [84]     cos(bearing of the ball -> goal line)
# [85]     ball x in the body frame, m (dead-reckoned from the track's
#          odometry position, so it stays right through the last 0.3 m when
#          the ball is under the camera and the track is coasting)
# [86]     ball y in the body frame, m
# [87]     the reflex tier will not take a kick now (a cycle is running, one
#          is latched, or the kick cooldown has not expired)
# 83-86 are zero with no ball track at all.
STRIKER_OBS_DIM = 88
STRIKER_OBS_VERSION = 1
STRIKER_ACT_DIM = 5        # vx, vy, wz, kick_left, kick_right
S_ACT_LOW = np.array([-0.2, -0.3, -1.0, -1.0, -1.0], np.float32)
S_ACT_HIGH = np.array([0.6, 0.3, 1.0, 1.0, 1.0], np.float32)
KICK_ON = 0.5              # a logit over this asks for that foot's kick
# The reflex tier refuses a kick while the body is moving. `robotd` runs the
# kick policy at the STANDING tuning, and the scripted brain stands `settle_s`
# on the spot before it fires for exactly this reason ("a kick started
# mid-stride fell 4 times in 7 here", ChaseParams). Measured on this env with
# a random policy: 4 of 25 episodes ended in a fall and ALL FOUR were inside
# 0.6 s of a kick. That is a property of the reflex tier, not of the brain, so
# it belongs here rather than in the reward - a fall penalty would only be
# paying the brain to work around a swap the robot would never allow.
#
# The gate is on the COMMAND, held for `KICK_SETTLE_S`, not on the measured
# speed alone: a walking duck's heading speed passes through zero inside every
# gait cycle and a turn in place has no heading speed at all, so a speed-only
# gate let kicks fire mid-stride and mid-turn and made falls WORSE (11 of 25
# random episodes against 4). This is the scripted brain's `settle_s` (0.4 s)
# as a reflex, with the measured speed kept as a backstop.
KICK_SETTLE_S = 0.3        # seconds of a standing command before the tier will kick
KICK_STILL_VX = 0.05       # what counts as a standing command
KICK_STILL_WZ = 0.15
KICK_MAX_SPEED = 0.06      # m/s of heading speed (a standing walker sways ~0.01)
KICK_GIVE_UP_S = 1.5       # a latched kick that cannot settle in this long is dropped
# …and the tier will not swap again straight away. Measured (striker-v1 at
# 75k decisions, probed on the 1v0 pitch): without a cooldown the brain
# collapsed onto ONE kick every 0.93 s - 514 kicks over 4 x 120 s - which is
# the stop-and-swap running back to back with no gap. That is a LOCK, not a
# preference: the tier owns the body for 0.3 s of settle plus 0.5 s of kick,
# so a duck that re-latches immediately never walks, never reaches the ball
# (possession 0.0 s/min, advance 0.00) and therefore never samples the state
# where anything else pays. An unsampled state's value is never learned, and
# AGENTS.md says the fix for that is the world and not the reward - so the
# reflex tier, which owns skills on the robot anyway, refuses to re-swap
# inside this. The scripted brain kicks 229 times in 24 x 300 s (0.03/s),
# two orders of magnitude under the gate: it costs the baseline nothing.
KICK_COOLDOWN_S = 2.0

# The measured sweet spot (kickprobe, this file's docstring): body-frame
# offsets of a ball the shipped kick sends 2.3 m.
KICK_AHEAD = 0.08
KICK_SIDE = 0.06

BALL_CLIP = 3.0
GOAL_CLIP = 4.0


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _gaze_pitch(rng: float, cam_z: float = 0.21, cam_level: float = 0.197,
                gain: float = 0.75, head_down: float = 0.6) -> float:
    """The reflex tier's head pitch that puts a floor ball at `rng` on the
    camera's axis — `ChaseParams`' law and constants, so the learned brain
    and the scripted one look down the same way. Level, the camera loses a
    floor ball ~0.3 m out; this keeps it to ~0.2 m."""
    want = math.atan2(cam_z - 0.035, max(rng, 0.05))
    return float(np.clip((want - cam_level) / gain, 0.0, head_down))


def striker_obs(base: np.ndarray, odom: tuple[float, float, float] | None,
                goal: tuple[float, float], ball_xy: tuple[float, float] | None,
                skill_running: bool) -> np.ndarray:
    """The 88-float striker observation from the 80-float brain one plus the
    geometry a brain reads off odometry: `ball_xy` is the TRACK's odometry
    position (not the truth), `goal` the mouth it attacks."""
    o = np.zeros(STRIKER_OBS_DIM, np.float32)
    o[:BRAIN_OBS_DIM] = base
    x, y, yaw = odom if odom is not None else (0.0, 0.0, 0.0)
    gb = _wrap(math.atan2(goal[1] - y, goal[0] - x) - yaw)
    o[80], o[81] = math.sin(gb), math.cos(gb)
    o[82] = min(math.hypot(goal[0] - x, goal[1] - y), GOAL_CLIP) / GOAL_CLIP
    if ball_xy is not None:
        bx, by = ball_xy
        line = _wrap(math.atan2(goal[1] - by, goal[0] - bx) - yaw)
        o[83], o[84] = math.sin(line), math.cos(line)
        dx, dy = bx - x, by - y
        o[85] = float(np.clip(dx * math.cos(yaw) + dy * math.sin(yaw), -BALL_CLIP, BALL_CLIP))
        o[86] = float(np.clip(-dx * math.sin(yaw) + dy * math.cos(yaw), -BALL_CLIP, BALL_CLIP))
    o[87] = 1.0 if skill_running else 0.0      # the reflex tier owns the body
    return o


class StrikerSenses:
    """The per-brain state the striker observation needs across calls: the
    version-2 `ObsBuilder` (its tracker IS the ball tracker) and the head
    the reflex tier holds. One in the env, one in `LearnedStriker` — the
    same code path, so training and deployment cannot drift."""

    def __init__(self, goal: tuple[float, float], head_range: float = 0.9,
                 bump_stop: float = 0.30, gaze: bool = True):
        self.goal = (float(goal[0]), float(goal[1]))
        self.head_range = float(head_range)
        self.bump_stop = float(bump_stop)
        self.gaze = bool(gaze)
        self.builder = ObsBuilder("ball", version=2)
        self.reset()

    def reset(self) -> None:
        self.builder.reset()
        self.head = (0.0, 0.0, 0.0, 0.0)
        self.ball_xy: tuple[float, float] | None = None
        self.still_since: float | None = None
        self.pending: str | None = None            # a latched kick request
        self.pending_t = 0.0
        self.last_kick_t = -1e9

    @property
    def tracker(self):
        return self.builder.tracker

    def observe(self, s: Senses, last_twist: np.ndarray) -> np.ndarray:
        base = self.builder(s, last_twist)          # folds the frame into the tracker
        tr = self.tracker.best("ball", s.t, min_hits=1)
        if tr is not None and tr.xy is not None:
            self.ball_xy = tr.xy
        elif tr is None:
            self.ball_xy = None
        # The reflex head: pitched at the tracked ball while it is near.
        rng = None
        if tr is not None and self.ball_xy is not None and s.odom is not None:
            rng = math.hypot(self.ball_xy[0] - s.odom[0], self.ball_xy[1] - s.odom[1])
        elif tr is not None:
            rng = tr.range
        self.head = (0.0, _gaze_pitch(rng), 0.0, 0.0) if (
            self.gaze and rng is not None and rng < self.head_range) else (0.0, 0.0, 0.0, 0.0)
        # Slot 87 is "the reflex tier will not take a kick request now":
        # a cycle running, a request latched, or the cooldown. The brain has
        # to be able to see the gate it is asking through.
        busy = (s.skill is not None or self.pending is not None
                or s.t - self.last_kick_t < KICK_COOLDOWN_S)
        return striker_obs(base, s.odom, self.goal, self.ball_xy, busy)

    def act(self, action: np.ndarray, s: Senses) -> tuple[tuple[float, float, float], str | None]:
        """The reflex tier, between the brain's action and the robot:
        returns the twist to command and the skill to start, if any.

        Two vetoes and one OPTION. The vetoes are the scripted brain's own:
        no forward command with something body-height inside `bump_stop`
        ahead (`ChaseParams.tof_stop`; every remaining fall it had was a
        walk into a board), and no kick swap while the body is moving.

        The option is what makes a kick reachable by exploration at all.
        A kick request LATCHES: the tier stops the body itself, and fires
        the moment the command has stood for `KICK_SETTLE_S` and the body
        has actually stopped (`KICK_MAX_SPEED`), or drops the request after
        `KICK_GIVE_UP_S`. Without the latch the brain had to produce a
        near-zero twist for three consecutive decisions BY CHANCE before a
        kick could ever fire, and it never did: 25 random episodes
        contained 0 kicks, against 32 an episode with no settle gate at all
        (and 4 falls in 25, every one of them inside 0.6 s of a kick). That
        is an exploration gap, and AGENTS.md says to fix the world rather
        than the pay — so the tier, which owns skills on the robot anyway,
        executes "kick" as the stop-and-swap it really is. The brain picks
        the option; the tier runs it."""
        vx, vy, wz = float(action[0]), float(action[1]), float(action[2])
        skill = None
        if s.skill is not None:
            self.pending, self.still_since = None, None
            self.last_kick_t = s.t                   # the cooldown runs from the END of the cycle
            return (0.0, 0.0, 0.0), None
        if self.pending is None and s.t - self.last_kick_t >= KICK_COOLDOWN_S:
            want = kick_from(action)
            if want is not None:
                self.pending, self.pending_t = want, s.t
        if self.pending is not None:
            vx = vy = wz = 0.0                       # the tier stops the body to swap
            if s.t - self.pending_t > KICK_GIVE_UP_S:
                self.pending, self.still_since = None, None
        if self.bump_stop > 0 and vx > 0:
            fr = s.fresh_tof(0.25)
            if fr is not None and tof_clearance_bearings(fr)[0] < self.bump_stop:
                vx = 0.0
        still = (abs(vx) <= KICK_STILL_VX and abs(vy) <= KICK_STILL_VX and abs(wz) <= KICK_STILL_WZ)
        if not still:
            self.still_since = None
        elif self.still_since is None:
            self.still_since = s.t
        if (self.pending is not None and self.still_since is not None
                and s.t - self.still_since >= KICK_SETTLE_S
                and abs(s.speed or 0.0) <= KICK_MAX_SPEED):
            skill, self.pending, self.still_since = self.pending, None, None
            self.last_kick_t = s.t
        return (vx, vy, wz), skill


def kick_from(action) -> str | None:
    """The hierarchical head: which foot the brain asked for, if any."""
    kl, kr = float(action[3]), float(action[4])
    if max(kl, kr) < KICK_ON:
        return None
    return "kick_left" if kl >= kr else "kick_right"


# --- the task ----------------------------------------------------------------
@dataclass
class StrikerTask:
    """1v0 first: a pitch, a ball, one duck, and the goal at +x (roadmap 4.4
    asks for the dribble toward the goal before any opponent)."""
    pitch: tuple[float, float] = (3.0, 2.5)     # make_pitch(per_side=1)'s size
    goal_width: float = 0.7
    episode_s: float = 25.0
    # Reward (the same terms in every spawn family — a stage may ladder the
    # world, never the pay).
    w_progress: float = 8.0      # per metre of ball displacement toward the goal, SIGNED
    w_approach: float = 1.0      # a potential on the duck-ball distance (telescopes; not farmable)
    approach_ball_v: float = 0.2  # …paid only while the ball is slower than this (a kicked ball
                                  #    rolling away is not a retreat, and must not be charged as one)
    fall_penalty: float = 1.0
    jerk_penalty: float = 0.02
    # The spawn ladder (physics/spawns only): the fraction of episodes that
    # start with the ball already on a foot's sweet spot, and the fraction
    # that start with it a step or two ahead. The rest are open pitch.
    spot_prob: float = 0.25
    near_prob: float = 0.35
    # The reflex tier under the brain, the scripted Chase's own.
    gaze: bool = True
    head_range: float = 0.9
    bump_stop: float = 0.30
    sense_dr: bool = True


class StrikerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task: StrikerTask = StrikerTask(), walker: str | Path | None = None,
                 decide_every: int = 5, seed: int | None = None, fixed_preset: str | None = None):
        super().__init__()
        self.task = task
        self.decide_every = int(decide_every)
        self.fixed_preset = fixed_preset
        self.rng = np.random.default_rng(seed)
        walker = Path(walker) if walker else POLICIES_DIR / "alpha_walking.onnx"
        if not walker.exists():
            raise FileNotFoundError(f"walker policy {walker} not found (clone pollen-robotics/microduck)")
        self._infer = onnx_infer(walker)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (STRIKER_OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(S_ACT_LOW, S_ACT_HIGH, dtype=np.float32)
        self.max_decisions = int(round(task.episode_s / C.CTRL_DT / self.decide_every))
        self._build_world()
        self.sense = StrikerSenses(self.goal, task.head_range, task.bump_stop, task.gaze)
        self._last_twist = np.zeros(3, np.float32)
        self.kicks = 0

    # -- world ---------------------------------------------------------------
    def _build_world(self) -> None:
        t = self.task
        hx, hy = t.pitch[0] / 2, t.pitch[1] / 2
        corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        walls = [Wall(corners[i], corners[(i + 1) % 4], 0.3, 0.02) for i in range(4)]
        preset = self.fixed_preset or "datasheet"
        sc = Scenario(name="striker", floor=(t.pitch[0] + 0.5, t.pitch[1] + 0.5), walls=walls,
                      balls=[Ball((0.0, 0.0))],
                      ducks=[Duck("d0", (-0.9, 0.0, 0.0), None, preset, preset, None, team="left")],
                      goal_width=t.goal_width)
        self.world = World(sc, infer_for={"d0": self._infer}, max_episode_s=1e9,
                           seed=int(self.rng.integers(0, 2**31 - 1)))
        self.duck = self.world.ducks["d0"]
        # The mouth this duck attacks and the boards, exactly as the World
        # and `team.brain_kwargs` compute them for the scripted brain.
        self.goal = (sc.floor[0] / 2 - 0.25, 0.0)
        self.bounds = (sc.floor[0] / 2 - 0.25, sc.floor[1] / 2 - 0.25)

    def _draw_spawn(self) -> tuple[tuple[float, float, float], tuple[float, float]]:
        """(duck spawn, ball xy) for one episode. Three families, one reward."""
        t, r = self.task, self.rng
        bx_lim, by_lim = self.bounds[0] - 0.15, self.bounds[1] - 0.15
        dx_lim, dy_lim = self.bounds[0] - 0.40, self.bounds[1] - 0.40
        u = float(r.random())
        x = float(r.uniform(-dx_lim, min(dx_lim, 0.7)))
        y = float(r.uniform(-dy_lim, dy_lim))
        to_goal = math.atan2(self.goal[1] - y, self.goal[0] - x)
        if u < t.spot_prob:
            # The drill: the ball already on a foot's sweet spot, the body
            # pointed somewhere near the goal. The kick pays 2.3 m of
            # progress here, so the value of a kick is learnable at all.
            yaw = _wrap(to_goal + float(r.uniform(-0.6, 0.6)))
            side = KICK_SIDE if r.random() < 0.5 else -KICK_SIDE
            ball = (x + KICK_AHEAD * math.cos(yaw) - side * math.sin(yaw),
                    y + KICK_AHEAD * math.sin(yaw) + side * math.cos(yaw))
        elif u < t.spot_prob + t.near_prob:
            yaw = _wrap(to_goal + float(r.uniform(-1.0, 1.0)))
            rng_ = float(r.uniform(0.3, 0.9))
            a = yaw + float(r.uniform(-0.6, 0.6))
            ball = (x + rng_ * math.cos(a), y + rng_ * math.sin(a))
        else:
            yaw = float(r.uniform(-math.pi, math.pi))
            for _ in range(20):
                ball = (float(r.uniform(-bx_lim, bx_lim)), float(r.uniform(-by_lim, by_lim)))
                if math.hypot(ball[0] - x, ball[1] - y) > 0.35:
                    break
        ball = (float(np.clip(ball[0], -bx_lim, bx_lim)), float(np.clip(ball[1], -by_lim, by_lim)))
        return (x, y, yaw), ball

    def _place_ball(self, xy: tuple[float, float]) -> None:
        w = self.world
        j = w._ball_joint
        q, v = int(w.model.jnt_qposadr[j]), int(w.model.jnt_dofadr[j])
        w.data.qpos[q:q + 7] = [xy[0], xy[1], w.scenario.balls[0].radius + 0.005, 1.0, 0.0, 0.0, 0.0]
        w.data.qvel[v:v + 6] = 0.0

    # -- gym -----------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        spawn, ball = self._draw_spawn()
        self.duck.spawn = spawn
        if self.task.sense_dr and self.fixed_preset is None:
            self.duck.tof.noise = TofNoise.preset(str(self.rng.choice(TOF_PRESETS)))
            self.duck.detector.noise = DetectorNoise.preset(str(self.rng.choice(TOF_PRESETS)))
        self.world.reset()
        # An episode is a pure function of (seed, ep): the three generators
        # that outlive world.reset() are re-seeded here, as BrainEnv does —
        # a battery you can shard is a battery you can trust.
        for gen in (self.duck.tof, self.duck.detector, self.world):
            if gen is not None:
                gen.rng = np.random.default_rng(int(self.rng.integers(0, 2**31 - 1)))
        self.duck.twist_cmd[:] = 0.0
        self.duck.head_cmd[:] = 0.0
        self._place_ball(ball)
        mujoco.mj_forward(self.world.model, self.world.data)
        for _ in range(self.decide_every):          # let the first sensor frames land
            self.world.step()
        self._last_twist[:] = 0.0
        self.sense.reset()
        self.kicks = 0
        self._steps = 0
        self._goal_seq = self.world.goal_seq
        self.goals = 0
        self._prev_ball = self.world.ball_xy()
        self._prev_dist = self._ball_dist()
        return self._obs(), {}

    def senses(self) -> Senses:
        w, d = self.world, self.duck
        tof = d.tof.last if d.tof is not None else None
        det = d.detector.last if d.detector is not None else None
        return Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                      det=det, det_age=None if det is None else w.t - det.t,
                      speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill,
                      bumped=w.bumped(d))

    def _obs(self) -> np.ndarray:
        return self.sense.observe(self.senses(), self._last_twist)

    def _ball_dist(self) -> float:
        p = self.duck.trunk_pos(self.world.data)
        b = self.world.ball_xy()
        return math.hypot(float(p[0]) - b[0], float(p[1]) - b[1])

    def _ball_speed(self) -> float:
        w = self.world
        v = int(w.model.jnt_dofadr[w._ball_joint])
        return float(math.hypot(w.data.qvel[v], w.data.qvel[v + 1]))

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, np.float32), S_ACT_LOW, S_ACT_HIGH)
        w, d, t = self.world, self.duck, self.task
        s = self.senses()
        twist, want = self.sense.act(a, s)
        fired = None
        if want is not None and w.start_skill(d, want):
            fired = want
            self.kicks += 1
        if d.skill is None:
            d.set_cmd(w.data, twist, self.sense.head)
        falls0 = d.falls
        for _ in range(self.decide_every):
            w.step()
        # Reward. Progress first, credited exactly as `PitchMetrics` does:
        # the jump the World makes when it re-centres the ball after a goal
        # is nobody's progress.
        ball = w.ball_xy()
        if w.goal_seq != self._goal_seq:
            self._goal_seq = w.goal_seq
            self.goals += 1
            progress = 0.0
        else:
            progress = ball[0] - self._prev_ball[0]     # this duck attacks +x
        self._prev_ball = ball
        reward = t.w_progress * progress
        dist = self._ball_dist()
        if self._ball_speed() < t.approach_ball_v:
            reward += t.w_approach * (self._prev_dist - dist)
        self._prev_dist = dist
        reward -= t.jerk_penalty * float(np.abs(np.asarray(twist, np.float32) - self._last_twist).sum())
        self._last_twist[:] = twist
        fell = d.falls > falls0
        if fell:
            reward -= t.fall_penalty
        self._steps += 1
        info = {"progress": progress, "ball_dist": dist, "kick": fired, "kicks": self.kicks,
                "goals": self.goals, "fell": fell,
                "ball_goal_dist": math.hypot(self.goal[0] - ball[0], self.goal[1] - ball[1])}
        if fell:
            self._prev_ball, self._prev_dist = ball, self._ball_dist()
        return self._obs(), float(reward), bool(fell), self._steps >= self.max_decisions, info

    def close(self) -> None:
        pass


# --- the trained brain, in the world -----------------------------------------
class LearnedStriker:
    """A trained striker running on a pitch: `brains/<name>/brain.onnx` at
    the decision rate it trained at, holding the last intent between
    decisions, with the same reflex tier the env gave it (the gaze and the
    ToF bump stop). Emits `Intent.skill` — the hierarchical head — so the
    World's reflex tier runs the shipped kick.

    Not registered under `learned:` because that loader is the follow
    contract's (80 floats, 3 actions, no skill); this is 88 and 5.
    """

    kind = "striker"
    wants_head = True

    def __init__(self, name: str, goal: tuple[float, float] | None = None,
                 duck_id: str = "", bounds: tuple[float, float] | None = None,
                 goal_w: float = 0.0, team=None, decide_every: int | None = None,
                 brains_root: Path | None = None):
        from .learned import brains_dir
        d = (brains_root or brains_dir()) / name
        onnx = d / "brain.onnx"
        if not onnx.exists():
            raise ValueError(f"no exported brain at {onnx}")
        meta = json.loads((d / "brain.json").read_text()) if (d / "brain.json").exists() else {}
        if meta.get("task", "striker") != "striker":
            raise ValueError(f"{name} is a {meta.get('task')!r} brain, not a striker")
        self.name = name
        self.kind = f"striker:{name}"
        self.duck_id = duck_id
        self.team = team
        self.goal = (1.5, 0.0) if goal is None else (float(goal[0]), float(goal[1]))
        self.bounds = bounds
        self.decide_every = int(decide_every or meta.get("decide_every", 5))
        self.sense = StrikerSenses(self.goal, float(meta.get("head_range", 0.9)),
                                   float(meta.get("bump_stop", 0.30)), bool(meta.get("gaze", True)))
        self.infer = onnx_infer(onnx)
        self.reset()

    def reset(self) -> None:
        self.sense.reset()
        self.last_twist = np.zeros(3, np.float32)
        self.last_action = np.zeros(STRIKER_ACT_DIM, np.float32)
        self.kicks = 0
        self.pushes = 0
        self.state = "striker"
        self.role = "attack"
        self._tick = 0
        self._senses: Senses | None = None

    def kickoff(self) -> None:
        """Play restarts: forget the track, keep the tally (brain/team.py's
        contract for `kickoff_brains`)."""
        self.sense.reset()
        self.last_twist[:] = 0.0
        self.last_action[:] = 0.0
        self._tick = 0
        self.state = "striker"

    def inputs(self) -> dict:
        if self._senses is None:
            return {}
        from .runtime import age_inputs
        out = age_inputs(self._senses, 0.25, 0.4)
        out["tracks"] = self.sense.tracker.payload(self._senses.t)
        out["chase"] = {"kicks": self.kicks, "pushes": 0, "role": self.role,
                        "ball": None if self.sense.ball_xy is None else
                        [round(self.sense.ball_xy[0], 2), round(self.sense.ball_xy[1], 2)]}
        return out

    def step(self, senses: Senses) -> Intent:
        self._senses = senses
        if self._tick % self.decide_every == 0:
            obs = self.sense.observe(senses, self.last_twist)
            a = np.clip(self.infer(obs.astype(np.float32)), S_ACT_LOW, S_ACT_HIGH).astype(np.float32)
            self.last_action = a
            self.state = "ball" if obs[65] > 0.5 else "lost"
        self._tick += 1
        twist, skill = self.sense.act(self.last_action, senses)
        self.last_twist[:] = twist
        if skill is not None:
            self.kicks += 1
        return Intent(twist=tuple(float(v) for v in twist), head=self.sense.head,
                      note=("kick" if self.sense.pending else self.state), skill=skill)


__all__ = ["KICK_AHEAD", "KICK_ON", "KICK_SIDE", "LearnedStriker", "STRIKER_ACT_DIM",
           "STRIKER_OBS_DIM", "STRIKER_OBS_VERSION", "S_ACT_HIGH", "S_ACT_LOW",
           "StrikerEnv", "StrikerSenses", "StrikerTask", "kick_from", "striker_obs"]
