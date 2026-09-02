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

from .gait import TURN_KICK, GaitWatch, turn
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


def tof_clearance_3d(frame, zmin: float = 0.08, zmax: float = 0.6) -> np.ndarray:
    """Nearest return per column that is a BODY-height thing — a wall, a
    duck, furniture — whatever the head is doing: each zone's hit is placed
    in the body's heading frame from the mount pose the frame carries, and
    hits on the floor (below `zmin`; a ball is 7 cm tall) or above `zmax`
    do not count. Falls back to the level-head rows when a frame carries no
    mount pose (synthetic frames)."""
    if frame.mount_pos is None or frame.dirs_local is None:
        return _column_clearance(frame.depth_mm, frame.valid, WanderParams(rows=(2, 5)))
    d = frame.depth_mm.astype(np.float64) / 1000.0
    z = frame.mount_pos[2] + frame.dirs_local[..., 2] * d
    ok = frame.valid & (frame.depth_mm > 0) & (z > zmin) & (z < zmax)
    return np.where(ok, d, np.inf).min(axis=0)


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
    k_turn: float = 8.0            # wz per rad of bearing (swept on 8 episodes: 3.0 kept sight 0.36 of the time, 8.0 with the idle sidestep 0.51)
    k_speed: float = 1.2           # vx per m of distance error (0.6 could not close on a 0.35 m/s walker)
    max_speed: float = 0.5
    min_speed: float = 0.12        # alpha_walking treats slower asks as "stand"
    turn_first: float = 0.6        # rad off the nose beyond which it turns in place before walking
    k_lead: float = 0.0            # wz per rad/s of bearing RATE: turn toward where the target is going
    # Keep the gait WARM: standing still, the walker cannot start a right
    # turn and starts a left one slowly, so the person walks out of the
    # frame before the body follows (measured: the learned brain sidesteps
    # ±0.23 the whole time and holds the bearing at 0.13 rad; the scripted
    # one stood, went cold and averaged 0.82). A sidestep toward the
    # target's side whenever it would otherwise stand keeps the legs going.
    idle_vy: float = 0.25
    idle_coast: bool = True        # …also while coasting on a lost track
    coast_speed: float = 0.0       # walking speed on a coasted track (0: stand and turn, measured safer)
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
        self._prev_track: tuple[float, float] | None = None     # (t, bearing) for the lead term
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
        vy = 0.0
        if target is not None and (fresh or target.age(senses.t) < p.lost_s):
            self.last_bearing = target.bearing
            self.last_range = target.range
            if fresh:
                self.last_seen_t = senses.t
            rate = 0.0
            if p.k_lead and self._prev_track is not None and senses.t > self._prev_track[0]:
                rate = (target.bearing - self._prev_track[1]) / (senses.t - self._prev_track[0])
            self._prev_track = (senses.t, target.bearing)
            wz = float(np.clip(p.k_turn * target.bearing + p.k_lead * float(np.clip(rate, -2.0, 2.0)), -1.0, 1.0))
            err = target.range - p.distance
            vx = float(np.clip(p.k_speed * err, 0.0, p.max_speed))
            if abs(target.bearing) > p.turn_first:
                vx, _, wz = turn(target.bearing, cold)      # turn first, walk after
            elif 0.0 < vx < p.min_speed:
                vx = 0.0 if err < 0.1 else p.min_speed
            if not fresh:
                # Coasting: face where the track says it went, but do not
                # walk at a range nobody has measured lately (measured:
                # walking on a coasted track bumped the person 50% more).
                vx = p.coast_speed if err > 0.2 else 0.0
                if abs(target.bearing) > 0.15:
                    vx, _, wz = turn(target.bearing, cold)
                else:
                    wz = 0.0
            self.state = ("hold" if vx == 0.0 and abs(wz) < 0.2 else "approach") if fresh else "coast"
            if p.idle_vy and vx == 0.0 and abs(wz) < 1.0 and (fresh or p.idle_coast):
                vy = p.idle_vy * (1.0 if target.bearing >= 0.0 else -1.0)
        else:
            self.track_id = None
            self._prev_track = None
            vx, _, wz = turn(1.0 if self.last_bearing >= 0 else -1.0, cold)
            self.state = "search"
        if ahead < p.tof_stop and (vx > 0 or vy != 0.0):
            # Never walk into what is right there — a cold-turn kick included,
            # the idle sidestep too: blocked, the search turns LEFT without
            # the kick (the turn that does start from a standstill, see
            # brain/gait.py).
            vx, vy = 0.0, 0.0
            if self.state == "search":
                wz = 1.0
            else:
                self.state = "blocked"
        head_yaw = float(np.clip(p.head_yaw_gain * self.last_bearing, -0.6, 0.6)) if self.last_seen_t else 0.0
        self.last = (vx, vy, wz)
        return Intent(twist=self.last, head=(0.0, 0.0, head_yaw, 0.0), note=self.state)


@dataclass(frozen=True)
class ChaseParams:
    target_cls: str = "ball"
    speed: float = 0.45            # walk at the ball
    k_turn: float = 3.0
    turn_first: float = 0.6        # rad off the nose: turn in place before walking
    lost_s: float = 2.0
    tof_stop: float = 0.3          # walls and ducks (body-height ToF returns); the ball and the floor do not count
    side_stop: float = 0.22        # a wall this close in the side columns: no turn in place toward it
    # The shipped kicks (measured, `walker-facts`-style, on the walker): a
    # ball 0.08 m ahead of the trunk and 0.06 m to the kicking foot's side
    # flies 1.6 m; 0.10 m dead ahead barely moves; the other side, nothing.
    kick_ahead: float = 0.08
    kick_side: float = 0.06
    lineup_range: float = 0.6      # a ball seen inside this is worth lining up on
    refresh_min: float = 0.35      # …and the spot is re-planned from sightings down to this range, then walked blind
    lineup_tol: float = 0.03       # trunk within this of the kicking spot: kick
    lineup_s: float = 4.0          # give up a line-up after this long
    settle_s: float = 0.4          # stand this long on the spot before the kick (robotd kicks at standing tuning)
    kick_clear: float = 0.35       # no kick with anything closer than this ahead
    aim_tol: float = 0.25          # face the kick direction within this before kicking (rad)
    aim_max: float = 1.05          # aim at the goal only within this of the line of sight (rad)
    # The head. Level, the camera loses a floor ball ~0.3 m out. Pitched
    # by `_gaze` (a law that puts the ball on the camera's axis: measured
    # 0.6 of command = 0.647 rad of camera, 0.20 m up) while WALKING at a
    # ball inside `head_range`, it keeps the ball in view to ~0.2 m, so the
    # line-up spot is refreshed to `refresh_min` instead of 0.6 m. Only
    # while walking: the walker cannot turn in place with its head down
    # (measured, tidy.py). Re-planning the spot from sightings INSIDE
    # refresh_min was measured and dropped: at 0.2 m the bearing noise is
    # centimetres and the foot choice flipped, the spot dithered for 8 s.
    head_down: float = 0.6
    head_range: float = 0.9
    head_gain: float = 0.75
    cam_level: float = 0.197
    cam_z: float = 0.21
    # After a kick the ball is ahead and low: stand and look down `look_s`
    # before searching (measured: a 9 s search spin with the ball 0.17 m
    # ahead). A search dips the head every `search_dip_every`.
    look_s: float = 0.8
    look_range: float = 0.3
    search_dip_every: float = 1.5
    search_dip_s: float = 0.6
    dip_range: float = 0.22
    # Dribbling: OFF (inf). Measured — a ball pushed at 0.3 m/s for half a
    # second rolls on at about the walking speed on this floor and the duck
    # walks behind it without ever lining up; the kick wins. ~1.4 to try.
    push_beyond: float = math.inf
    push_behind: float = 0.16
    push_speed: float = 0.3
    push_s: float = 0.5
    # The other duck's BODY (measured over 4 traced runs: 5 of 7 falls had the
    # other duck 3–9 cm away and this one turning in place — search, blocked
    # or lining up — the walker tips over when it turns against a body it
    # cannot see below its ToF rows). A tracked duck inside `duck_keepout`
    # and ahead: nothing walks or turns toward it; inside `duck_touch` it is
    # against us: stand until it moves.
    duck_keepout: float = 0.4
    duck_touch: float = 0.22
    duck_bearing: float = 1.2      # rad off the nose that counts as "ahead"
    # Standing against something (avoid, blocked) longer than `stuck_s`:
    # two ducks meeting at the ball otherwise stand and wait for each
    # other (traced: 8 s nose to nose). Retreat: turn toward the freer
    # side, then walk clear.
    stuck_s: float = 1.5
    retreat_turn_s: float = 1.0
    retreat_walk_s: float = 1.2
    # Team play (brain/team.py): a supporter stands `support_back` from the
    # ball toward its own goal, `support_side` to the side per rank, facing
    # the ball, and never inside `support_min` of it.
    support_back: float = 0.7
    support_side: float = 0.45
    support_min: float = 0.45
    # Yielding to a duck that clearly has the ball: OFF by default. Measured
    # over 8 seeds × 300 s: off 1.50 goals / 8.5 kicks / 2.12 falls a run,
    # on (0.5 m) 1.12 / 7.0 / 2.12 — it costs play and saves nothing.
    yield_range: float = 0.0
    yield_ratio: float = 0.7
    yield_s: float = 1.5
    yield_cooldown_s: float = 3.0


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class Chase:
    """Walk at the nearest ball, line up behind it on the line to the goal,
    and KICK it with the shipped kick policy (roadmap soccer). Tracks the
    ball (brain/tracker.py) with the head pitched down on the way in; a
    floor ball leaves the camera ~0.2 m out, so the last leg is dead
    reckoning in odometry to a spot `kick_ahead` behind the ball and
    `kick_side` to the foot's side, then a 0.5 s kick window. Keeps off
    the other ducks and the walls, retreats when stood against something,
    and in a team (brain/team.py) takes the attacker's or a supporter's
    role. Searches turning left, dipping the head for a near ball."""

    kind = "chase"
    wants_head = True
    DET_MAX_AGE = 0.4
    TOF_MAX_AGE = 0.25

    def __init__(self, p: ChaseParams = ChaseParams(), goal: tuple[float, float] | None = None,
                 team=None, duck_id: str = ""):
        self.p = p
        self.goal = None if goal is None else (float(goal[0]), float(goal[1]))
        self.team = team
        self.duck_id = duck_id
        self.tracker = Tracker()
        self.gait = GaitWatch()
        self.reset()

    def reset(self) -> None:
        self.state = "search"
        self.role = "attack"
        self.last_bearing = 0.0
        self.last_seen_t: float | None = None
        self._senses: Senses | None = None
        self.last = (0.0, 0.0, 0.0)
        self.kicks = 0
        self.pushes = 0
        self.spot: tuple[float, float, str | None, float, str] | None = None   # x, y, foot, heading, "kick"|"push"
        self.attack: float | None = None                            # heading of the goal it attacks (first odom yaw)
        self.t_state = 0.0
        self._yield_t0 = -9.0
        self._yield_end = -9.0
        self._poses: list[tuple[float, float, float, float]] = []      # (t, x, y, yaw) over the stuck window
        self._retreat_t0 = -9.0
        self._retreat_sign = 1.0
        self._look_t0 = -9.0
        self._search_t0 = -9.0
        self._prev_skill = None
        self.tracker.reset()
        self.gait.reset()

    def _gaze(self, rng: float) -> float:
        """head_pitch that puts a floor ball at `rng` on the camera's axis."""
        p = self.p
        want = math.atan2(p.cam_z - 0.035, max(rng, 0.05))
        return float(np.clip((want - p.cam_level) / p.head_gain, 0.0, p.head_down))

    def inputs(self) -> dict:
        if self._senses is None:
            return {}
        out = age_inputs(self._senses, self.TOF_MAX_AGE, self.DET_MAX_AGE)
        out["target"] = None if self.last_seen_t is None else {
            "bearing": round(self.last_bearing, 3), "range": None,
            "since": round(self._senses.t - self.last_seen_t, 2)}
        out["tracks"] = self.tracker.payload(self._senses.t)
        out["chase"] = {"kicks": self.kicks, "pushes": self.pushes, "role": self.role,
                        "spot": None if self.spot is None else
                        [round(self.spot[0], 3), round(self.spot[1], 3), self.spot[2] or self.spot[4]]}
        if self.team is not None:
            out["team"] = self.team.payload(self._senses.t)
        return out

    # -- geometry -------------------------------------------------------------
    def _ball_xy(self, odom, ball) -> tuple[float, float]:
        x, y, yaw = odom
        a = yaw + ball.bearing
        return x + ball.range * math.cos(a), y + ball.range * math.sin(a)

    def _own_goal(self, odom) -> tuple[float, float]:
        if self.goal is not None:
            return -self.goal[0], self.goal[1]                  # the pitch is centred on the origin
        a = self.attack if self.attack is not None else odom[2]
        return odom[0] - 2.0 * math.cos(a), odom[1] - 2.0 * math.sin(a)

    def _plan(self, odom, ball) -> tuple[float, float, str | None, float, str]:
        """Where to stand to kick a ball seen at (bearing, range): behind it
        on the line the kick should go — toward the goal (`goal`, in the
        odometry frame; without one, the heading the duck was placed with)
        when that costs under `aim_max` of detour, else along the line of
        sight (a walk-round crossed walls and the other duck — measured) —
        offset sideways so the nearer foot meets it. The left foot kicks a
        ball to its LEFT. A far goal (`push_beyond`) makes it a push spot
        squarely behind the ball. Returns (x, y, foot, heading, mode)."""
        p = self.p
        x, y, yaw = odom
        bx, by = self._ball_xy(odom, ball)
        los = yaw + ball.bearing
        if self.goal is not None:
            u = math.atan2(self.goal[1] - by, self.goal[0] - bx)
            far = math.hypot(self.goal[0] - bx, self.goal[1] - by) > p.push_beyond
        else:
            u, far = (self.attack if self.attack is not None else los), False
        if abs(_wrap(u - los)) > p.aim_max:
            u, far = los, False
        if far:
            return bx - p.push_behind * math.cos(u), by - p.push_behind * math.sin(u), None, u, "push"
        rel = _wrap(math.atan2(by - y, bx - x) - u)
        foot = "kick_left" if rel >= 0 else "kick_right"
        if self.spot is not None and self.spot[2] in ("kick_left", "kick_right") and abs(rel) < 0.3:
            foot = self.spot[2]                                   # hysteresis: nearly on the line, keep the foot
        side = -p.kick_side if foot == "kick_left" else p.kick_side     # stand to the ball's other side
        return (bx - p.kick_ahead * math.cos(u) - side * math.sin(u),
                by - p.kick_ahead * math.sin(u) + side * math.cos(u), foot, u, "kick")

    def _servo(self, odom, target, cold, stop: float, slow_in: float = 0.2) -> tuple[float, float, float, float]:
        """(vx, wz, dist, bearing) toward a point: turn in place first when
        it is well off the nose, walk with steering otherwise, stop inside."""
        p = self.p
        dx, dy = target[0] - odom[0], target[1] - odom[1]
        dist = math.hypot(dx, dy)
        bearing = _wrap(math.atan2(dy, dx) - odom[2])
        if dist <= stop:
            return 0.0, 0.0, dist, bearing
        if abs(bearing) > 0.5 and dist > 0.08:
            vx, _, wz = turn(bearing, cold)
            return vx, wz, dist, bearing
        return (0.25 if dist < slow_in else p.speed), float(np.clip(p.k_turn * bearing, -1.0, 1.0)), dist, bearing

    # -- the machine ----------------------------------------------------------
    def step(self, senses: Senses) -> Intent:
        self._senses = senses
        p = self.p
        t = senses.t
        cold = self.gait.update(senses)
        odom = senses.odom or (0.0, 0.0, 0.0)
        if self.attack is None and senses.odom is not None:
            self.attack = odom[2]                  # placed facing the goal it attacks (make_pitch does)
        self.tracker.update(senses.fresh_det(self.DET_MAX_AGE), t, odom[2])
        ball = self.tracker.best(p.target_cls, t, min_hits=1)
        fresh = ball is not None and ball.age(t) <= self.DET_MAX_AGE
        seen = ball is not None and ball.age(t) < p.lost_s
        if self.team is not None:
            self.team.claim(self.duck_id, t, ball.range if seen else math.inf,
                            self._ball_xy(odom, ball) if seen else None)
            self.role = self.team.role(self.duck_id, t)
        other = self.tracker.best("duck", t, min_hits=1)
        near_duck = (other is not None and other.age(t) <= 0.6 and other.range < p.duck_keepout
                     and abs(other.bearing) < p.duck_bearing)
        clearly_nearer = (other is not None and other.age(t) <= p.lost_s and other.range < p.yield_range
                          and ball is not None and other.range < p.yield_ratio * ball.range
                          and abs(_wrap(other.bearing - ball.bearing)) < 0.8)
        if clearly_nearer and self.state != "yield" and t - self._yield_end > p.yield_cooldown_s:
            self._yield_t0 = t
        yielding = clearly_nearer and t - self._yield_t0 < p.yield_s
        if self.state == "yield" and not yielding:
            self._yield_end = t
        tof = senses.fresh_tof(self.TOF_MAX_AGE)
        ahead = left_near = right_near = np.inf
        if tof is not None:
            cols = tof_clearance_3d(tof)          # body-height things only: not the floor the head looks at, not the ball
            ahead, left_near, right_near = float(cols[3:5].min()), float(cols[0:3].min()), float(cols[5:8].min())
        skill = None
        head = (0.0, 0.0, 0.0, 0.0)
        gaze_at: float | None = None
        retreating = t - self._retreat_t0 < p.retreat_turn_s + p.retreat_walk_s
        if self._prev_skill is not None and senses.skill is None:
            self._look_t0 = t                                   # the kick window just ended: look for the ball ahead
        self._prev_skill = senses.skill
        looking = t - self._look_t0 < p.look_s and not fresh
        if senses.skill is not None:
            vx, wz = 0.0, 0.0                                   # the kick owns the reflex tier
            self.state = "kick"
        elif looking:
            vx, wz = 0.0, 0.0
            gaze_at = p.look_range
            self.state = "look"
        elif retreating:
            if t - self._retreat_t0 < p.retreat_turn_s:
                vx, _, wz = turn(self._retreat_sign, cold)
            else:
                vx, wz = p.speed, 0.0
            self.spot = None
            self.state = "retreat"
        elif near_duck:
            # Turn AWAY from it (the side that puts it behind us), never
            # into it, and not at all while it is touching: a stand is the
            # one thing the walker does safely against another body. The
            # cold-gait kick creeps forward, so no kick with it near the nose.
            self.spot = None
            if other.range < p.duck_touch:
                vx, wz = 0.0, 0.0
            else:
                vx, _, wz = turn(-1.0 if other.bearing >= 0.0 else 1.0, cold)
                if abs(other.bearing) < 0.5:
                    vx = 0.0
            self.state = "avoid"
        elif self.role == "support":
            vx, wz = self._support(odom, ball, seen, cold)
        elif yielding and self.state not in ("settle",):
            vx, wz = 0.0, 0.0
            self.spot = None
            self.state = "yield"
        elif self.state == "push":
            _, _, _, u, _ = self.spot
            vx, wz = p.push_speed, float(np.clip(p.k_turn * _wrap(u - odom[2]), -1.0, 1.0))
            if t - self.t_state >= p.push_s:
                self.spot = None
                self.state = "search"
                self._look_t0 = t
        elif self.state in ("lineup", "settle") and self.spot is not None:
            # Refresh the spot while the ball is in view and not too close
            # (see refresh_min), then walk the rest blind.
            if fresh and self.state != "settle" and p.refresh_min <= ball.range < p.head_range:
                self.spot = self._plan(odom, ball)
            sx, sy, foot, u, mode = self.spot
            vx, wz, dist, bearing = self._servo(odom, (sx, sy), cold, p.lineup_tol)
            heading_err = _wrap(u - odom[2])
            # Hysteresis on both: a settling duck wobbles a centimetre and a
            # few hundredths of a radian, which flipped it between the
            # square-up and the settle at the tolerance (measured: 22 s
            # standing at the spot, no kick).
            settling = self.state == "settle"
            on_spot = dist <= p.lineup_tol + (0.03 if settling else 0.0)
            squared = abs(heading_err) <= p.aim_tol + (0.15 if settling else 0.0)
            if on_spot and not squared:
                vx, _, wz = turn(heading_err, cold)            # on the spot but not facing the goal: square up
                self.state = "lineup"
            elif on_spot:
                # Stand first: robotd runs a kick at the standing tuning, and
                # a kick started mid-stride fell 4 times in 7 here. Something
                # within `kick_clear` ahead (a wall, a duck) means the swing
                # lands on it: let it go and look again.
                vx, wz = 0.0, 0.0
                if ahead < p.kick_clear:
                    self.spot = None
                    self.state = "search"
                elif not settling:
                    self.state = "settle"
                    self.t_state = t
                elif t - self.t_state >= p.settle_s:
                    if mode == "push":
                        self.pushes += 1
                        self.state = "push"
                        self.t_state = t
                        vx, wz = p.push_speed, 0.0
                    else:
                        skill = foot
                        self.kicks += 1
                        self.spot = None
                        self.state = "kick"
            elif self.state == "lineup" and t - self.t_state > p.lineup_s:
                self.spot = None
                self.state = "search"
                vx, _, wz = turn(1.0, cold)
            else:
                self.state = "lineup"
                if vx > 0 and fresh and ball.range < p.head_range and abs(ball.bearing) < 0.6:
                    gaze_at = ball.range
        elif seen:
            self.last_bearing = ball.bearing
            if fresh:
                self.last_seen_t = t
            if fresh and ball.range < p.lineup_range and abs(ball.bearing) < 0.5:
                self.spot = self._plan(odom, ball)
                self.state = "lineup"
                self.t_state = t
                vx, wz = p.speed, float(np.clip(p.k_turn * ball.bearing, -1.0, 1.0))
                gaze_at = ball.range
            elif abs(ball.bearing) > p.turn_first:
                vx, _, wz = turn(ball.bearing, cold)
                self.state = "turn"
            else:
                vx, wz = p.speed, float(np.clip(p.k_turn * ball.bearing, -1.0, 1.0))
                self.state = "chase"
                if fresh and ball.range < p.head_range:
                    gaze_at = ball.range
        else:
            if self.state != "search":
                self._search_t0 = t
            vx, _, wz = turn(1.0, cold)
            self.state = "search"
            if (t - self._search_t0) % p.search_dip_every < p.search_dip_s:
                vx, wz = 0.0, 0.0                               # a standing look down: a near ball is below the level camera
                gaze_at = p.dip_range
        # A wall beside us: no turn in place toward it (measured: a line-up
        # turning against the boards tipped over). Turn toward the side
        # with more room — in a corner that is still a turn, the one move
        # that gets out of a corner (standing there measured as a deadlock).
        if vx <= TURN_KICK and wz != 0.0 and self.state != "retreat":
            if wz > 0 and left_near < p.side_stop and right_near > left_near:
                wz = -1.0
            elif wz < 0 and right_near < p.side_stop and left_near > right_near:
                wz = 1.0
        if ahead < p.tof_stop and vx > 0 and self.state != "push":
            # A wall or the other duck right there: no walking, no cold-turn
            # creep (measured: every remaining fall was a line-up walking
            # into a wall or a kicked turn creeping into one). Turning still
            # happens — the left turn that starts from a standstill.
            vx = 0.0
            if self.state in ("lineup", "settle", "support"):
                wz = 1.0 if wz > 0 else -1.0 if wz < 0 else 0.0
            else:
                wz = 1.0
                self.state = "blocked"
        # Not moving while stood against something (avoid, blocked) for
        # `stuck_s`, whatever the state labels say frame to frame: retreat.
        self._poses.append((t, odom[0], odom[1], odom[2]))
        while self._poses and t - self._poses[0][0] > p.stuck_s:
            self._poses.pop(0)
        if self.state in ("avoid", "blocked", "yield") and len(self._poses) > 1 \
                and t - self._poses[0][0] >= p.stuck_s - 0.05:
            _, x0, y0, yaw0 = self._poses[0]
            if math.hypot(odom[0] - x0, odom[1] - y0) < 0.05 and abs(_wrap(odom[2] - yaw0)) < 0.3:
                self._poses = []
                self._retreat_t0 = t
                self._retreat_sign = 1.0 if left_near >= right_near else -1.0
        if gaze_at is not None and (vx > 0 or self.state in ("look", "search")):
            head = (0.0, self._gaze(gaze_at), 0.0, 0.0)
        self.last = (vx, 0.0, wz)
        return Intent(twist=self.last, head=head, note=self.role if self.role != "attack" else self.state, skill=skill)

    def _support(self, odom, ball, seen: bool, cold: bool) -> tuple[float, float]:
        """A supporter: stand back from the ball toward our own goal, offset
        sideways by rank, facing the ball. The ball's position comes from
        my own track when I see it, else from a teammate's claim."""
        p = self.p
        t = self._senses.t
        bxy = self._ball_xy(odom, ball) if seen else (self.team.ball(t) if self.team is not None else None)
        self.spot = None
        if bxy is None:
            self.state = "support"
            vx, _, wz = turn(1.0, cold)                    # nobody has it: look for it
            return vx, wz
        og = self._own_goal(odom)
        gx, gy = og[0] - bxy[0], og[1] - bxy[1]
        gn = math.hypot(gx, gy)
        ux, uy = (gx / gn, gy / gn) if gn > 1e-6 else (-math.cos(odom[2]), -math.sin(odom[2]))
        rank = self.team.rank(self.duck_id, t) if self.team is not None else 0
        side = p.support_side * ((rank + 1) // 2) * (1 if rank % 2 == 0 else -1)
        target = (bxy[0] + p.support_back * ux - side * uy, bxy[1] + p.support_back * uy + side * ux)
        vx, wz, dist, _ = self._servo(odom, target, cold, 0.12)
        if math.hypot(bxy[0] - odom[0], bxy[1] - odom[1]) < p.support_min and vx > 0:
            vx = 0.0                                        # the attacker's room
        if dist <= 0.12:
            b = _wrap(math.atan2(bxy[1] - odom[1], bxy[0] - odom[0]) - odom[2])
            vx, wz = (0.0, 0.0) if abs(b) < 0.3 else turn(b, cold)[::2]
        self.state = "support"
        return vx, wz


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
