"""Hand-written controllers over the ToF matrix (roadmap 2.3).

`wander_from_tof` is the first brain: cruise forward, slow down as the
middle of the depth matrix closes in, turn toward whichever side has more
room, and spin in place when nothing ahead is far enough. It reads only
what the sensor reports (dropped zones count as unknown, never as "far"),
and it emits only a twist, so it runs unchanged against the real robot's
`tof.stream` and `robot.move` once the bridge exists.

It is deliberately dumb: it is the baseline a learned brain has to beat,
and the lesson page shows exactly which zones it looked at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .gait import GaitWatch, turn
from .runtime import REGISTRY, Intent, Senses, age_inputs
from .tracker import Tracker, TrackerParams


@dataclass(frozen=True)
class WanderParams:
    cruise: float = 0.3        # m/s when the way is clear
    slow_at: float = 0.7       # start slowing when the centre is nearer than this
    stop_at: float = 0.3       # stop and spin when nearer than this
    turn: float = 0.8          # rad/s while steering around something
    spin: float = 1.0          # rad/s when boxed in
    rows: tuple[int, int] = (2, 7)   # zone rows that count: skip the sky, keep the floor edge
    max_range_m: float = 4.0


def _column_clearance(depth_mm: np.ndarray, valid: np.ndarray | None,
                      p: WanderParams) -> np.ndarray:
    """Nearest reported target per column over the counted rows; +inf where
    no zone in the column reported anything."""
    d = depth_mm.astype(np.float64) / 1000.0
    ok = (depth_mm > 0) if valid is None else (valid & (depth_mm > 0))
    r0, r1 = p.rows
    d, ok = d[r0:r1], ok[r0:r1]
    d = np.where(ok, d, np.inf)
    return d.min(axis=0)


def wander_from_tof(depth_mm: np.ndarray, valid: np.ndarray | None = None,
                    p: WanderParams = WanderParams(),
                    prefer_left: bool | None = None) -> tuple[float, float, float]:
    """One decision from one frame. Returns (vx, vy, wz).

    `prefer_left` breaks a tie (and keeps a turn going) — a stateless
    controller re-deciding every frame would dither between the two sides.
    """
    cols = _column_clearance(depth_mm, valid, p)
    centre = float(cols[2:6].min())
    left = float(np.mean(np.minimum(cols[:4], p.max_range_m)))
    right = float(np.mean(np.minimum(cols[4:], p.max_range_m)))
    if prefer_left is None:
        prefer_left = left >= right
    elif abs(left - right) > 0.15:
        prefer_left = left > right
    sign = 1.0 if prefer_left else -1.0        # +wz turns left (toward +y, column 0)
    if centre < p.stop_at:
        return 0.0, 0.0, sign * p.spin
    if centre < p.slow_at:
        frac = (centre - p.stop_at) / (p.slow_at - p.stop_at)
        return p.cruise * frac, 0.0, sign * p.turn
    return p.cruise, 0.0, 0.0


class Wander:
    """Stateful wrapper: remembers the turn direction and, when the duck has
    made no progress for a while under a forward command, spins to unstick.
    Also a `Brain` (runtime.py): `step(senses)` gates the ToF on age."""

    kind = "wander"
    TOF_MAX_AGE = 0.25       # ~3 frames at 15 Hz: older than that, stand

    def __init__(self, p: WanderParams = WanderParams(), stuck_s: float = 2.0,
                 unstick_s: float = 1.2):
        self.p = p
        self._senses: Senses | None = None
        self.prefer_left: bool | None = None
        self.stuck_s, self.unstick_s = stuck_s, unstick_s
        self._still_since: float | None = None
        self._unstick_until = -1.0
        self.last: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.state = "cruise"

    def reset(self) -> None:
        self.prefer_left = None
        self._still_since = None
        self._unstick_until = -1.0
        self.state = "cruise"

    def step(self, senses: Senses) -> Intent:          # the Brain interface
        self._senses = senses
        f = senses.fresh_tof(self.TOF_MAX_AGE)
        tw = self.decide(None if f is None else f.depth_mm, None if f is None else f.valid,
                         senses.t, senses.speed)
        return Intent(twist=tw, note=self.state)

    def inputs(self) -> dict:
        if self._senses is None:
            return {}
        return age_inputs(self._senses, self.TOF_MAX_AGE, 9e9)

    def decide(self, depth_mm: np.ndarray | None, valid: np.ndarray | None,
               t: float, speed: float | None = None) -> tuple[float, float, float]:
        if t < self._unstick_until:
            self.state = "unstick"
            self.last = (0.0, 0.0, (1.0 if self.prefer_left else -1.0) * self.p.spin)
            return self.last
        if depth_mm is None:
            self.state = "blind"
            self.last = (0.0, 0.0, 0.0)      # no frame yet: stand, do not guess
            return self.last
        vx, vy, wz = wander_from_tof(depth_mm, valid, self.p, self.prefer_left)
        self.prefer_left = wz > 0 if wz else self.prefer_left
        self.state = "spin" if vx == 0.0 and wz != 0.0 else ("steer" if wz else "cruise")
        # Stuck detector: asking for forward motion and getting none.
        if speed is not None and vx > 0.1 and abs(speed) < 0.03:
            if self._still_since is None:
                self._still_since = t
            elif t - self._still_since > self.stuck_s:
                self._still_since = None
                self._unstick_until = t + self.unstick_s
                if self.prefer_left is None:
                    self.prefer_left = True
                self.state = "unstick"
                self.last = (0.0, 0.0, (1.0 if self.prefer_left else -1.0) * self.p.spin)
                return self.last
        else:
            self._still_since = None
        self.last = (vx, vy, wz)
        return self.last


@dataclass(frozen=True)
class FollowParams:
    target_cls: str = "person"     # what to follow ("person" or "duck")
    distance: float = 0.7          # hold this far behind, m
    k_turn: float = 3.0            # wz per rad of bearing   (swept on 8 episodes: 3.0 loses sight less)
    k_speed: float = 1.2           # vx per m of distance error (0.6 could not close on a 0.35 m/s walker)
    max_speed: float = 0.5
    min_speed: float = 0.12        # alpha_walking treats slower asks as "stand"
    lost_s: float = 2.0            # keep the last bearing this long, then search
    search_wz: float = 1.0             # the shipped walker barely turns in place below 1.0
    tof_stop: float = 0.35         # never walk into what the ToF says is right there
    head_yaw_gain: float = 0.8     # look toward the target (the robot's own gaze intent)


class Follow:
    """Keep the nearest target of a class ahead at a fixed distance, from a
    TRACK over the detector's frames (bearing, width-derived range; brain/
    tracker.py), with the ToF as a bumper. Loses it: the track coasts —
    its bearing turning with the body — for `lost_s`, then turn to search.

    Deliberately simple — it is the baseline the learned brain (3.1/3.2) is
    measured against, and every number it uses is one the real robot can
    produce today or after one detector retrain. Turns go through
    brain/gait.py: the walker does not start a right turn from a standstill.
    """

    kind = "follow"
    DET_MAX_AGE = 0.4
    TOF_MAX_AGE = 0.25

    def __init__(self, p: FollowParams = FollowParams(), tracker: TrackerParams = TrackerParams()):
        self.p = p
        self.tracker = Tracker(tracker)
        self.gait = GaitWatch()
        self.reset()

    def reset(self) -> None:
        self.state = "search"
        self.last_bearing = 0.0
        self.last_seen_t: float | None = None
        self.last_range: float | None = None
        self.track_id: int | None = None
        self._senses: Senses | None = None
        self.last = (0.0, 0.0, 0.0)
        self.tracker.reset()
        self.gait.reset()

    def inputs(self) -> dict:
        if self._senses is None:
            return {}
        out = age_inputs(self._senses, self.TOF_MAX_AGE, self.DET_MAX_AGE)
        out["target"] = None if self.last_seen_t is None else {
            "bearing": round(self.last_bearing, 3), "range": _r(self.last_range),
            "since": round(self._senses.t - self.last_seen_t, 2), "track": self.track_id}
        out["tracks"] = self.tracker.payload(self._senses.t)
        return out

    def step(self, senses: Senses) -> Intent:
        self._senses = senses
        p = self.p
        cold = self.gait.update(senses)
        yaw = None if senses.odom is None else senses.odom[2]
        self.tracker.update(senses.fresh_det(self.DET_MAX_AGE), senses.t, yaw)
        # Stay on the track we have while it lives; otherwise the best one.
        target = None
        if self.track_id is not None:
            target = next((tr for tr in self.tracker.tracks if tr.id == self.track_id), None)
        if target is None:
            target = self.tracker.best(p.target_cls, senses.t)
            self.track_id = None if target is None else target.id
        # ToF bumper: the nearest thing in the middle columns.
        tof = senses.fresh_tof(self.TOF_MAX_AGE)
        ahead = np.inf
        if tof is not None:
            cols = _column_clearance(tof.depth_mm, tof.valid, WanderParams())
            ahead = float(cols[3:5].min())
        fresh = target is not None and target.age(senses.t) <= self.DET_MAX_AGE
        if target is not None and (fresh or target.age(senses.t) < p.lost_s):
            self.last_bearing = target.bearing
            self.last_range = target.range
            if fresh:
                self.last_seen_t = senses.t
            wz = float(np.clip(p.k_turn * target.bearing, -1.0, 1.0))
            err = target.range - p.distance
            vx = float(np.clip(p.k_speed * err, 0.0, p.max_speed))
            if abs(target.bearing) > 0.6:
                vx, _, wz = turn(target.bearing, cold)      # turn first, walk after
            elif 0.0 < vx < p.min_speed:
                vx = 0.0 if err < 0.1 else p.min_speed
            if not fresh:
                # Coasting: face where the track says it went, but do not
                # walk at a range nobody has measured lately (measured:
                # walking on a coasted track bumped the person 50% more).
                vx = 0.0
                if abs(target.bearing) > 0.15:
                    vx, _, wz = turn(target.bearing, cold)
                else:
                    wz = 0.0
            self.state = ("hold" if vx == 0.0 and abs(wz) < 0.2 else "approach") if fresh else "coast"
        else:
            self.track_id = None
            vx, _, wz = turn(1.0 if self.last_bearing >= 0 else -1.0, cold)
            self.state = "search"
        if ahead < p.tof_stop and vx > 0:
            # Never walk into what is right there — a cold-turn kick included:
            # blocked, the search turns LEFT without the kick (the turn that
            # does start from a standstill, see brain/gait.py).
            vx = 0.0
            if self.state == "search":
                wz = 1.0
            else:
                self.state = "blocked"
        head_yaw = float(np.clip(p.head_yaw_gain * self.last_bearing, -0.6, 0.6)) if self.last_seen_t else 0.0
        self.last = (vx, 0.0, wz)
        return Intent(twist=self.last, head=(0.0, 0.0, head_yaw, 0.0), note=self.state)


@dataclass(frozen=True)
class ChaseParams:
    target_cls: str = "ball"
    speed: float = 0.45            # walk at the ball
    k_turn: float = 3.0
    turn_first: float = 0.6        # rad off the nose: turn in place before walking
    lost_s: float = 2.0
    tof_stop: float = 0.3          # walls and ducks in the upper ToF rows; the ball is too low to count
    # The shipped kicks (measured, `walker-facts`-style, on the walker): a
    # ball 0.08 m ahead of the trunk and 0.06 m to the kicking foot's side
    # flies 1.6 m; 0.10 m dead ahead barely moves; the other side, nothing.
    kick_ahead: float = 0.08
    kick_side: float = 0.06
    lineup_range: float = 0.6      # a ball seen inside this is worth lining up on (the last 0.3 m are blind)
    lineup_tol: float = 0.03       # trunk within this of the kicking spot: kick
    lineup_s: float = 4.0          # give up a line-up after this long
    settle_s: float = 0.4          # stand this long on the spot before the kick (robotd kicks at standing tuning)
    kick_clear: float = 0.35       # no kick with anything closer than this ahead (upper ToF rows)


class Chase:
    """Walk at the nearest ball, line up, and KICK it with the shipped kick
    policy (roadmap soccer, first form). Tracks the ball (brain/tracker.py)
    while it is in view; a floor ball leaves both the camera and the ToF
    in the last ~0.3 m, so the line-up is dead reckoning in odometry to a
    spot `kick_ahead` behind the ball and `kick_side` to the foot's side,
    then a 0.5 s kick window. Searches turning left. The ToF bumper reads
    only the upper rows so the ball itself does not stop the chase."""

    kind = "chase"
    DET_MAX_AGE = 0.4
    TOF_MAX_AGE = 0.25

    def __init__(self, p: ChaseParams = ChaseParams()):
        self.p = p
        self.tracker = Tracker()
        self.gait = GaitWatch()
        self.reset()

    def reset(self) -> None:
        self.state = "search"
        self.last_bearing = 0.0
        self.last_seen_t: float | None = None
        self._senses: Senses | None = None
        self.last = (0.0, 0.0, 0.0)
        self.kicks = 0
        self.spot: tuple[float, float, str] | None = None     # odom-frame kicking spot + foot
        self.t_state = 0.0
        self.tracker.reset()
        self.gait.reset()

    def inputs(self) -> dict:
        if self._senses is None:
            return {}
        out = age_inputs(self._senses, self.TOF_MAX_AGE, self.DET_MAX_AGE)
        out["target"] = None if self.last_seen_t is None else {
            "bearing": round(self.last_bearing, 3), "range": None,
            "since": round(self._senses.t - self.last_seen_t, 2)}
        out["tracks"] = self.tracker.payload(self._senses.t)
        out["chase"] = {"kicks": self.kicks, "spot": None if self.spot is None else
                        [round(self.spot[0], 3), round(self.spot[1], 3), self.spot[2]]}
        return out

    def _lineup_spot(self, odom, ball) -> tuple[float, float, str]:
        """Where to stand to kick a ball seen at (bearing, range): behind it
        along the current line of sight, offset to the far side so the
        nearer foot meets it. Left foot kicks a ball to its LEFT."""
        p = self.p
        x, y, yaw = odom
        a = yaw + ball.bearing
        bx, by = x + ball.range * math.cos(a), y + ball.range * math.sin(a)
        foot = "kick_left" if ball.bearing >= 0 else "kick_right"
        side = -p.kick_side if foot == "kick_left" else p.kick_side     # stand to the ball's other side
        sx = bx - p.kick_ahead * math.cos(a) - side * math.sin(a)
        sy = by - p.kick_ahead * math.sin(a) + side * math.cos(a)
        return sx, sy, foot

    def step(self, senses: Senses) -> Intent:
        self._senses = senses
        p = self.p
        t = senses.t
        cold = self.gait.update(senses)
        odom = senses.odom or (0.0, 0.0, 0.0)
        self.tracker.update(senses.fresh_det(self.DET_MAX_AGE), t, odom[2])
        ball = self.tracker.best(p.target_cls, t, min_hits=1)
        fresh = ball is not None and ball.age(t) <= self.DET_MAX_AGE
        tof = senses.fresh_tof(self.TOF_MAX_AGE)
        ahead = np.inf
        if tof is not None:
            ahead = float(_column_clearance(tof.depth_mm, tof.valid, WanderParams(rows=(2, 5)))[3:5].min())
        skill = None
        if senses.skill is not None:
            vx, wz = 0.0, 0.0                                   # the kick owns the reflex tier
            self.state = "kick"
        elif self.state in ("lineup", "settle") and self.spot is not None:
            # Refresh the spot while the ball is still in view, then walk it blind.
            if fresh and ball.range < p.lineup_range:
                self.spot = self._lineup_spot(odom, ball)
            sx, sy, foot = self.spot
            dx, dy = sx - odom[0], sy - odom[1]
            dist = math.hypot(dx, dy)
            bearing = math.atan2(math.sin(math.atan2(dy, dx) - odom[2]), math.cos(math.atan2(dy, dx) - odom[2]))
            if dist <= p.lineup_tol:
                # Stand first: robotd runs a kick at the standing tuning, and a
                # kick started mid-stride fell 4 times in 7 here. Something
                # within `kick_clear` ahead (a wall, the other duck) means the
                # swing lands on it: let it go and look again.
                vx, wz = 0.0, 0.0
                if ahead < p.kick_clear:
                    self.spot = None
                    self.state = "search"
                elif self.state != "settle":
                    self.state = "settle"
                    self.t_state = t
                elif t - self.t_state >= p.settle_s:
                    skill = foot
                    self.kicks += 1
                    self.spot = None
                    self.state = "kick"
            elif self.state == "lineup" and t - self.t_state > p.lineup_s:
                self.spot = None
                self.state = "search"
                vx, _, wz = turn(1.0, cold)
            elif abs(bearing) > 0.5 and dist > 0.08:
                vx, _, wz = turn(bearing, cold)
                self.state = "lineup"
            else:
                vx = 0.25 if dist < 0.2 else p.speed
                wz = float(np.clip(p.k_turn * bearing, -1.0, 1.0))
                self.state = "lineup"
        elif ball is not None and ball.age(t) < p.lost_s:
            self.last_bearing = ball.bearing
            if fresh:
                self.last_seen_t = t
            if fresh and ball.range < p.lineup_range and abs(ball.bearing) < 0.5:
                self.spot = self._lineup_spot(odom, ball)
                self.state = "lineup"
                self.t_state = t
                vx, wz = p.speed, float(np.clip(p.k_turn * ball.bearing, -1.0, 1.0))
            elif abs(ball.bearing) > p.turn_first:
                vx, _, wz = turn(ball.bearing, cold)
                self.state = "turn"
            else:
                vx, wz = p.speed, float(np.clip(p.k_turn * ball.bearing, -1.0, 1.0))
                self.state = "chase"
        else:
            vx, _, wz = turn(1.0, cold)
            self.state = "search"
        if ahead < p.tof_stop and vx > 0:
            # A wall or the other duck right there: no walking, no cold-turn
            # creep (measured: every remaining fall was a line-up walking
            # into a wall or a kicked turn creeping into one). Turning still
            # happens — the left turn that starts from a standstill.
            vx = 0.0
            if self.state in ("lineup", "settle"):
                wz = 1.0 if wz > 0 else -1.0 if wz < 0 else 0.0
            else:
                wz = 1.0
                self.state = "blocked"
        self.last = (vx, 0.0, wz)
        return Intent(twist=self.last, note=self.state, skill=skill)


def _r(v) -> float | None:
    return None if v is None else round(float(v), 3)


class Script:
    """No brain: the world's drive script / manual command steers."""

    kind = "script"

    def __init__(self):
        self.state = "script"

    def step(self, senses: Senses) -> Intent:
        return Intent()

    def reset(self) -> None:
        pass

    def inputs(self) -> dict:
        return {}


REGISTRY.register("wander", Wander)
REGISTRY.register("follow", Follow)
REGISTRY.register("chase", Chase)
REGISTRY.register("script", Script)
