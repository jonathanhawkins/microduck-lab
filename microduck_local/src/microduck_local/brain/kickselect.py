"""Choose the kick by simulating its outcomes (roadmap Track 4 §6 A.3).

Mellmann, Schlotter & Blum (Berlin United, RoboCup 2016): before every kick,
each candidate action is forward-simulated a few dozen times, sampling the
kick's speed and direction from distributions fitted to real kicks; each
sample rolls out under a simple deceleration model until it stops, crosses a
goal line or hits the boards; each is labelled; actions with any own-goal
sample are discarded, the rest are ranked by how many samples score and
then by a potential field over the pitch. On labelled video of real games
it cut kicks out at the opponent goal line 5x and raised strategically-good
kicks from 67% to 78%.

This repo already owns every number the model needs, all measured in play
rather than assumed: a kick leaves the foot at ~1.4 m/s (`ChaseParams.
kick_speed`) and slows at `ball_decel`; the ball leaves at +23.6 deg off the
body for the left foot and -28.7 for the right (`kick_exit_*`, 237 kicks,
4b) with 33-49 deg of scatter about that (`dir_sd`); the pitch is walled,
so there is no OUT — a ball into the boards simply stops there — and the
two mouths are lines the World scores. `aim_mode="clamp"` is a one-line
deterministic version of the potential field with no notion of risk; this
is the version that can say "not that way, it goes in our own net" and
"that way, it might score".

Pure functions over numbers; the brain (`Chase._plan`) builds the candidate
fan and hands it here. Deterministic under a seeded generator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GOALOPP, GOALOWN, INFIELD = "GOALOPP", "GOALOWN", "INFIELD"
BLOCKED = "BLOCKED"                              # stopped by a duck in the way: the ball is at its feet (C.4)
WHIFF = "WHIFF"                                  # the swing missed: the ball stays put - never a goal, never a pass
EDGE = 0.02                                      # m: a ball estimate is rolled out from just inside the boards (`inside`)
BLOCK_COST = 0.5                                 # potential units a blocked ball is marked down by - it is theirs now
PUSH = "push"                                    # the third action beside kick_left / kick_right


@dataclass(frozen=True)
class KickModel:
    speed: float = 1.4                           # m/s off the foot (measured in play)
    speed_sd: float = 0.3                        # the tracker saw 1.16x truth on speed; a third of it either way
    dir_sd: float = 0.6                          # rad: the in-play scatter about the exit line (33-49 deg, 4b)
    decel: float = 0.3                           # m/s^2, the constant-deceleration stand-in for this floor
    exit_left: float = math.radians(23.6)        # where the ball really leaves, off the body heading
    exit_right: float = math.radians(-28.7)
    max_roll: float = 10.0                       # m: a cap so a wild speed sample cannot cross the world
    # The share of swings that move the ball less than 10 cm - a WHIFF - in
    # which case the ball stays where it is. Measured 50-61% on the
    # rolling-resistance floor (18-23% on the old one). A model that assumes
    # every kick connects overrates the kick against a push that always
    # does (benched: 50 of 50 walks touched the ball).
    p_whiff: float = 0.0

    def exit(self, action: str) -> float:
        if action == "kick_left":
            return self.exit_left
        if action == "kick_right":
            return self.exit_right
        return 0.0                               # a push leaves along the walk (the body's offset is in dir_sd)


def push_model(roll: float = 0.64, dir_sd: float = 0.5, decel: float = 0.3) -> KickModel:
    """The PUSH as an action: a walk through the ball. Benched on this floor
    (10 walks at 0.45 m/s per side offset, deterministic per offset): the
    ball rolls 0.56-0.71 m and leaves at +17 deg dead ahead, +-12 deg at
    4 cm off, +-45 deg at 8 cm off - a 30 deg spread across offsets, a
    fifth of a kick's reach, and every walk touched. The speed is whatever
    rolls `roll` under `decel`; the spread is the side offset the line-up
    happens to leave, which nothing measures at the moment of the walk."""
    return KickModel(speed=math.sqrt(2.0 * decel * roll), speed_sd=0.05, dir_sd=dir_sd, decel=decel,
                     exit_left=0.0, exit_right=0.0, p_whiff=0.0)


@dataclass(frozen=True)
class Pitch:
    half_x: float                                # inside the boards (Chase.bounds)
    half_y: float
    goal_w: float                                # the mouth's full width
    attack_sign: float                           # +1: we attack the +x mouth

    @property
    def opp_x(self) -> float:
        return self.attack_sign * self.half_x

    @property
    def own_x(self) -> float:
        return -self.attack_sign * self.half_x


def roll_out(ball: tuple[float, float], heading: float, model: KickModel, pitch: Pitch,
             rng: np.random.Generator, n: int, obstacles=None,
             obs_r: float = 0.15) -> list[tuple[str, tuple[float, float]]]:
    """`n` sampled outcomes of a kick from `ball` whose OUTCOME line (the exit
    angle already applied) is `heading`: (label, where the ball ends up).
    `obstacles` (roadmap C.4): ducks in the way - a sample whose path
    passes within `obs_r` of one stops at the closest approach, BLOCKED."""
    out: list[tuple[str, tuple[float, float]]] = []
    v = np.clip(rng.normal(model.speed, model.speed_sd, n), 0.05, None)
    a = rng.normal(heading, model.dir_sd, n)
    d = np.minimum(v * v / (2.0 * max(model.decel, 1e-6)), model.max_roll)
    whiff = rng.random(n) < model.p_whiff if model.p_whiff > 0 else np.zeros(n, bool)
    hw = pitch.goal_w / 2.0
    for k in range(n):
        if whiff[k]:
            out.append((WHIFF, (ball[0], ball[1])))            # the swing missed: the ball stays put
            continue
        dx, dy = float(d[k] * math.cos(a[k])), float(d[k] * math.sin(a[k]))
        # First board this segment reaches, as a fraction of its length.
        hit, label = 1.0, INFIELD
        for wall_x, goal_label in ((pitch.opp_x, GOALOPP), (pitch.own_x, GOALOWN)):
            if abs(dx) > 1e-9:
                f = (wall_x - ball[0]) / dx
                if 0.0 < f < hit:
                    y_at = ball[1] + f * dy
                    hit, label = f, (goal_label if abs(y_at) <= hw else INFIELD)
        for wall_y in (pitch.half_y, -pitch.half_y):
            if abs(dy) > 1e-9:
                f = (wall_y - ball[1]) / dy
                if 0.0 < f < hit:
                    hit, label = f, INFIELD
        # A duck in the way, before any board: the ball stops at its feet.
        l2 = dx * dx + dy * dy
        for ox, oy in obstacles or ():
            if l2 < 1e-12:
                break
            f = ((ox - ball[0]) * dx + (oy - ball[1]) * dy) / l2
            if 0.0 < f < hit and math.hypot(ox - (ball[0] + f * dx), oy - (ball[1] + f * dy)) <= obs_r:
                hit, label = f, BLOCKED
        out.append((label, (ball[0] + hit * dx, ball[1] + hit * dy)))
    return out


def potential(x: float, y: float, pitch: Pitch) -> float:
    """The strategic value of a ball at (x, y): Mellmann's field scaled to
    this pitch — a linear slope from our goal to theirs, a Gaussian
    attractor in front of theirs, a Gaussian repulsor in front of ours."""
    ax = pitch.attack_sign * x / max(pitch.half_x, 1e-6)          # -1 at our mouth, +1 at theirs
    sx_own, sx_opp, sy = 0.75 * pitch.half_x, 0.5 * pitch.half_x, 0.27 * pitch.half_x
    own = math.exp(-0.5 * (((x - pitch.own_x) / sx_own) ** 2 + (y / sy) ** 2))
    opp = math.exp(-0.5 * (((x - pitch.opp_x) / sx_opp) ** 2 + (y / sy) ** 2))
    return ax - own + opp


@dataclass(frozen=True)
class Verdict:
    heading: float                               # the kick line `u` the brain should lay its spot on
    foot: str
    p_goal: float
    p_own: float
    value: float
    n: int
    p_pass: float = 0.0                          # share of samples that stop within reach of a teammate
    p_block: float = 0.0                         # share of samples a duck in the way stopped (C.4)


def inside(ball, pitch: Pitch) -> tuple[float, float]:
    """The ball estimate pulled just inside the boards. An estimate ON or past
    a line - a 2-5 cm error on a ball at the boards, the dominant dead-ball
    state - made that wall invisible to `roll_out`'s `0 < f` test: a swing
    into our own mouth from x = -1.50 read 0% own goal against 99% from
    -1.49 (code review, 2026-09-08)."""
    return (min(max(float(ball[0]), -pitch.half_x + EDGE), pitch.half_x - EDGE),
            min(max(float(ball[1]), -pitch.half_y + EDGE), pitch.half_y - EDGE))


def evaluate(ball, u: float, action: str, model: KickModel, pitch: Pitch,
             rng: np.random.Generator, n: int,
             mates: list[tuple[float, float]] | None = None, pass_reach: float = 0.4,
             pass_bonus: float = 0.0, obstacles=None, obs_r: float = 0.15) -> Verdict:
    """`action` is kick_left / kick_right / push; `model` is that action's.
    With `mates` (teammates' positions in the ball's frame) a sample that
    stops within `pass_reach` of one is RECEIVED, and each such sample adds
    `pass_bonus` to the potential it is valued at (roadmap Track 4 s6 D.1:
    Mellmann's own next step was a teammate attractor in the field). With
    `obstacles` (C.4) a sample a duck stops is valued where it stops, less
    `BLOCK_COST`: a ball at an opponent's feet is not a pass."""
    ball = inside(ball, pitch)
    samples = roll_out(ball, u + model.exit(action), model, pitch, rng, n, obstacles, obs_r)
    labels = [s[0] for s in samples]
    received = 0
    vals = []
    for label, (px, py) in samples:
        if label not in (INFIELD, BLOCKED, WHIFF):
            continue
        v = potential(px, py, pitch)
        if label == BLOCKED:
            v -= BLOCK_COST
        elif label == INFIELD and mates and any(math.hypot(px - mx, py - my) <= pass_reach for mx, my in mates):
            received += 1
            v += pass_bonus
        vals.append(v)
    value = float(np.mean(vals)) if vals else 0.0
    return Verdict(u, action, labels.count(GOALOPP) / n, labels.count(GOALOWN) / n, value, n, received / n,
                   labels.count(BLOCKED) / n)


def select(ball, candidates: list[tuple[float, str]], model: KickModel, pitch: Pitch,
           rng: np.random.Generator, n: int = 30, t_own: float = 0.0,
           models: dict[str, KickModel] | None = None, shoot: float = 0.0,
           mates: list[tuple[float, float]] | None = None, pass_reach: float = 0.4,
           pass_bonus: float = 0.0, obstacles=None, obs_r: float = 0.15,
           chooser=None, safest: bool = False) -> Verdict | None:
    """The best of `candidates` (line, action). Mellmann's two-step rule:
    discard anything with more than `t_own` of its samples in our own net,
    then take the most likely to score, ties (within one sample) broken by
    the potential of where the rest of the samples stop. `model` is the
    kicks'; `models` may override per action (a `push` needs its own).

    `shoot` > 0 changes the priority where a push is on offer: a SAFE push
    is preferred unless some kick scores in at least that share of its
    samples. A one-shot roll-out cannot see what the push is for — tempo:
    a reliable 0.64 m every approach with no settle and no whiff, which
    measured +3.2 s/min of possession against the kick (roadmap A.4) —
    so with `shoot` = 0 a kick with ANY scoring chance outranks it, and on
    a 3 m pitch that is nearly everywhere. None when every candidate is
    too risky — the caller keeps what it had.

    `chooser` (roadmap E.2) replaces the RANKING and nothing else: the same
    fan, the same roll-outs, the same own-goal veto, and then
    `chooser(ball, pitch, safe)` picks instead of Mellmann's rule. None
    (the default) is the shipped path, bit for bit — this function does not
    evaluate the argument otherwise. `brain/kickchoice.py` builds one from
    a weights file fitted on the gym's realised outcomes."""
    if not candidates:
        return None
    per = dict(models or {})
    verdicts = [evaluate(ball, u, act, per.get(act, model), pitch, rng, n, mates, pass_reach, pass_bonus,
                         obstacles, obs_r)
                for u, act in candidates]
    safe = [v for v in verdicts if v.p_own <= t_own]
    if not safe:
        if not safest:
            return None
        # NOTHING PASSES THE FILTER. Returning None hands the decision back to
        # the caller, which keeps `aim_mode`'s clamp line - a line that was
        # never own-goal-checked at all. Near our own mouth, which is the only
        # place this happens, that is the worst moment to stop checking. With
        # `safest` the fan's LEAST BAD line is taken instead: still risky, but
        # it is the minimum of a set the clamp's line is a member of.
        floor = min(v.p_own for v in verdicts)
        return _best([v for v in verdicts if v.p_own <= floor + 1e-12], n, pitch)
    if chooser is not None:
        return chooser(ball, pitch, safe)
    pushes = [v for v in safe if v.foot == PUSH]
    kicks = [v for v in safe if v.foot != PUSH]
    if shoot > 0 and pushes and not any(v.p_goal >= shoot for v in kicks):
        return _best(pushes, n, pitch)
    return _best(safe, n, pitch)


def _best(vs: list[Verdict], n: int, pitch: Pitch) -> Verdict:
    """Mellmann's rule over `vs`: the most likely to score, ties (within one
    sample) broken by the potential of where the rest of the samples stop,
    then by the line nearest the attack direction. `value` excludes the
    samples that SCORE, so ranking the pushes by it alone preferred a push
    that stood well over one that scored (code review, 2026-09-08)."""
    best_goal = max(v.p_goal for v in vs)
    top = [v for v in vs if v.p_goal >= best_goal - 1.0 / n - 1e-12]
    attack = 0.0 if pitch.attack_sign >= 0 else math.pi
    return max(top, key=lambda v: (v.value, -abs(math.atan2(math.sin(v.heading - attack), math.cos(v.heading - attack)))))
