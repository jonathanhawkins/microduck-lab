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


def tof_hits_3d(frame) -> tuple[np.ndarray, np.ndarray] | None:
    """Every zone's hit in the body's HEADING frame, relative to the trunk
    (the frame the mount pose is given in): (rows, cols, 3) points and the
    (rows, cols) depths; None for a frame without a mount pose."""
    if frame.mount_pos is None or frame.mount_rot is None or frame.dirs_local is None:
        return None
    d = frame.depth_mm.astype(np.float64) / 1000.0
    dirs = frame.dirs_local @ frame.mount_rot.T
    return frame.mount_pos[None, None, :] + dirs * d[..., None], d


TRUNK_Z = 0.117                    # the standing trunk height (the ToF mount pose is trunk-relative)


def tof_clearance_bearings(frame, ahead_half: float = 0.10, side_half: float = 0.40,
                           zmin: float = -0.03, zmax: float = 0.5) -> tuple[float, float, float]:
    """(ahead, left, right) body-height clearance selected by BEARING off the
    body's nose, not by sensor column - because the sensor is IN THE HEAD.
    A column is "ahead" only while the head looks along the walking line;
    yaw the head and the middle columns report whatever is off to the side,
    which the brain then stops for (measured: every head-gaze variant lost
    kicks that way). Placed in the heading frame, a hit's bearing says where
    it really is, so a head turned off the line simply returns +inf ahead -
    honestly blind, rather than confidently wrong. `ahead_half` (5.7 deg)
    matches the two middle columns of a 45 deg sensor, `side_half` (23 deg)
    the outer three. Ranges are horizontal, which is what the walking
    thresholds mean. Falls back to the level-head columns for a synthetic
    frame with no mount pose."""
    hits = tof_hits_3d(frame)
    if hits is None:
        cols = _column_clearance(frame.depth_mm, frame.valid, WanderParams(rows=(2, 5)))
        return float(cols[3:5].min()), float(cols[0:3].min()), float(cols[5:8].min())
    pts, _ = hits
    z = pts[..., 2]
    ok = frame.valid & (frame.depth_mm > 0) & (z > zmin) & (z < zmax)
    v = pts - frame.mount_pos[None, None, :]                 # from the APERTURE, as the column version measured
    bear = np.arctan2(v[..., 1], v[..., 0])                  # +left, the brain's convention
    rng = np.where(ok, np.hypot(v[..., 0], v[..., 1]), np.inf)
    ahead = rng[np.abs(bear) <= ahead_half]
    left = rng[(bear > ahead_half) & (bear <= side_half)]
    right = rng[(bear < -ahead_half) & (bear >= -side_half)]
    return (float(ahead.min()) if ahead.size else np.inf,
            float(left.min()) if left.size else np.inf,
            float(right.min()) if right.size else np.inf)


def tof_floor_ball(frame, r_max: float = 0.5, z_lo: float = -0.09, z_hi: float = -0.02) -> tuple[float, float] | None:
    """A ball-sized thing on the floor inside `r_max`, seen by the ToF:
    hits above the floor plane but below 10 cm (trunk-relative z between
    `z_lo`, 2.7 cm up - floor hits scatter to 1.8 cm with datasheet noise -
    and `z_hi`, 9.7 cm up; a ball is 7 cm tall), at least two adjacent
    zones of them, in columns with NOTHING taller near (a wall or a duck
    has hits above the band in the same columns; a ball has the floor
    behind it) - returned as (bearing, horizontal range) of the nearest
    such cluster, heading frame. The camera loses a floor ball inside
    0.3 m unless the head dips; the ToF at 45 deg shows one at 0.3 m as a
    3-6 zone blob (measured). None for a frame without a mount pose."""
    hits = tof_hits_3d(frame)
    if hits is None:
        return None
    pts, _ = hits
    z = pts[..., 2]
    rng = np.hypot(pts[..., 0], pts[..., 1])
    live = frame.valid & (frame.depth_mm > 0)
    ok = live & (z > z_lo) & (z < z_hi) & (rng < r_max)
    if ok.sum() < 2:
        return None
    r, c = np.unravel_index(np.argmin(np.where(ok, rng, np.inf)), ok.shape)
    win = np.zeros_like(ok)
    win[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = True
    sel = ok & win
    if sel.sum() < 2:
        return None
    cols = sel.any(axis=0)
    tall = live & (z >= z_hi) & (rng < r_max + 0.15) & cols[None, :]
    if tall.any():
        return None
    x, y = float(pts[..., 0][sel].mean()), float(pts[..., 1][sel].mean())
    return float(math.atan2(y, x)), float(math.hypot(x, y))


def tof_clearance_3d(frame, zmin: float = -0.03, zmax: float = 0.5) -> np.ndarray:
    """Nearest return per column that is a BODY-height thing — a wall, a
    duck, furniture — whatever the head is doing: each zone's hit is placed
    in the body's heading frame from the mount pose the frame carries
    (rotated by the head's pose: an unrotated placement read the FLOOR as a
    wall 0.35 m ahead whenever the head dipped 0.6 rad, measured), and hits
    on the floor (below `zmin`, trunk-relative: 8.7 cm above the floor, a
    ball is 7 cm tall) or above `zmax` do not count. Falls back to the
    level-head rows when a frame carries no mount pose (synthetic frames)."""
    hits = tof_hits_3d(frame)
    if hits is None:
        return _column_clearance(frame.depth_mm, frame.valid, WanderParams(rows=(2, 5)))
    pts, d = hits
    z = pts[..., 2]
    ok = frame.valid & (frame.depth_mm > 0) & (z > zmin) & (z < zmax)
    return np.where(ok, d, np.inf).min(axis=0)


@dataclass(frozen=True)
class ClosingParams:
    range_m: float = 1.2           # something inside this, ahead...
    rate: float = 0.12             # ...closing on me faster than my own walk by this (m/s)
    window_s: float = 0.4          # over this much ToF history (6 frames at 15 Hz)
    # The manoeuvre. Measured from a standstill: a pure sidestep moves the
    # walker 1 cm in its first second (6 cm in two); a turn toward the
    # freer side then a walk moves it off the line - and out of the
    # oncoming path, which a stop alone is not.
    turn_s: float = 1.0
    walk_s: float = 1.5
    speed: float = 0.3
    cooldown_s: float = 1.5        # after one, look again this much later


class ClosingWatch:
    """Something walking at me. The ToF's clearance ahead (the middle
    columns, body-height returns only) shrinks at my own speed when I walk
    at a wall and faster when a person or a duck comes at me; the
    difference, fitted over a short window, is the closing rate. Past
    `rate` inside `range_m`, the answer is a manoeuvre out of its path: a
    turn toward the freer side (the columns with more clearance), then a
    walk. Nothing here needs more than the robot's ToF and its own
    commanded speed."""

    def __init__(self, p: ClosingParams = ClosingParams()):
        self.p = p
        self.reset()

    def reset(self) -> None:
        self._hist: list[tuple[float, float]] = []      # (t, clearance ahead)
        self._last_t: float | None = None
        self.closing = 0.0                               # m/s toward me beyond my own walk (last estimate)
        self.side = 0.0
        self.t0 = -1e9
        self.until = -1e9
        self.count = 0

    def step(self, frame, t: float, speed: float, cold: bool = True) -> tuple[float, float, float] | None:
        """Fold the newest ToF frame; returns the twist to hold now, or
        None when there is nothing to get out of the way of."""
        p = self.p
        if frame is not None and frame.t != self._last_t:
            self._last_t = frame.t
            cols = tof_clearance_3d(frame)
            ahead = float(cols[3:5].min())
            if np.isfinite(ahead):
                self._hist.append((frame.t, ahead))
            else:
                self._hist.clear()
            self._hist = [h for h in self._hist if frame.t - h[0] <= p.window_s]
            if len(self._hist) >= 3 and self._hist[-1][0] > self._hist[0][0]:
                ts = np.array([h[0] for h in self._hist])
                ds = np.array([h[1] for h in self._hist])
                slope = float(np.polyfit(ts - ts[0], ds, 1)[0])          # m/s, negative = shrinking
                self.closing = -slope - max(float(speed), 0.0)
            else:
                self.closing = 0.0
            if ahead < p.range_m and self.closing > p.rate and t >= self.until + p.cooldown_s:
                left = float(np.mean(np.minimum(cols[0:3], 4.0)))
                right = float(np.mean(np.minimum(cols[5:8], 4.0)))
                self.side = 1.0 if left >= right else -1.0
                self.t0 = t
                self.until = t + p.turn_s + p.walk_s
                self.count += 1
        if t >= self.until:
            return None
        if t < self.t0 + p.turn_s:
            return turn(self.side, cold)
        return (p.speed, 0.0, 0.0)


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
    # Get out of the path of whatever walks at me (ClosingWatch): OFF.
    # Measured over 12 episodes: the walker cannot clear a person walking
    # at it (it sidesteps 1 cm in its first second; a turn-and-walk moves
    # it 0.1 m off the line in 1.8 s, and the person arrives in 2-5 s), so
    # the charge case's contact time is the same either way (4.9 vs 5.3
    # s/ep) while in ordinary following the dodge fires on a person who is
    # merely walking toward the duck and loses it (in band 0.49 -> 0.39,
    # in sight 0.86 -> 0.57). It halves falls in the charge case (0.17 ->
    # 0.08/ep), within the noise. `eval-brain --avoid` measures it.
    avoid: bool = False


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
        self.closing = ClosingWatch()
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
        self.closing.reset()

    def inputs(self) -> dict:
        if self._senses is None:
            return {}
        out = age_inputs(self._senses, self.TOF_MAX_AGE, self.DET_MAX_AGE)
        out["target"] = None if self.last_seen_t is None else {
            "bearing": round(self.last_bearing, 3), "range": _r(self.last_range),
            "since": round(self._senses.t - self.last_seen_t, 2), "track": self.track_id}
        out["tracks"] = self.tracker.payload(self._senses.t)
        out["closing"] = round(self.closing.closing, 2)
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
        # Something walking at me - the person turning back, another duck -
        # gets a sidestep out of its path (after the bumper: this one is
        # meant to move with something right there).
        dodge = self.closing.step(tof, senses.t, senses.speed, cold) if p.avoid else None
        if dodge is not None:
            vx, vy, wz = dodge
            self.state = "dodge"
        head_yaw = float(np.clip(p.head_yaw_gain * self.last_bearing, -0.6, 0.6)) if self.last_seen_t else 0.0
        self.last = (vx, vy, wz)
        return Intent(twist=self.last, head=(0.0, 0.0, head_yaw, 0.0), note=self.state)


@dataclass(frozen=True)
class ChaseParams:
    """The chase brain's constants. Most were measured INTO their value; a
    good few were measured OFF and keep their number in the comment beside
    them, because "we tried it and it was worse" is the expensive part and
    deleting it invites the next person to spend the afternoon again.

    Shipping OFF (0 / False), with their measurements below: `two_stage`
    and `search_walk_after` (line-up precision), `kick_deflect_*` (the kick
    map in the stance), `kick_cone` (shoot only from close), `predict_s`,
    `head_yaw_when`, `predict_steer`, `look_aim`, `search_sweep` (what the
    head does about the ball), `tof_ball_m` (the ToF seeing a ball at the
    feet), `seek_s` (a ball memory), `push_beyond` (deliberate bumping),
    `search_sided`, `mate_keepout` (teammates' poses as obstacles),
    `support_turn_vx`, `support_mode="ahead"` (a poacher supporter, found
    then killed by fresh seeds). Read the numbers before re-trying one -
    and read the README's note on what 8 seeds can and cannot resolve
    first, because several of those "worse" verdicts are inside the noise
    and say only "not shown to help".
    """
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
    # The kick map (a standing duck, the ball swept over (ahead, side) of
    # the trunk, kick_left; the right kick checked mirrored): the ball
    # leaves at an angle to the BODY heading that depends on the side
    # offset - 15 deg/cm near 2 cm, 4.5 deg/cm around 4-8 cm, where the
    # shipped spot sits - and at the spot it is +21.6 deg for the left foot
    # (2.1 m) and -11 deg for the right (1.9 m), the same whichever way the
    # body is yawed. Set, the line-up stands the body rotated by this so a
    # kick from the sweet spot flies along the line to the goal. OFF
    # (measured): in play the ball is 2-3 cm off the sweet spot when the
    # kick fires - the line-up, not the map, is what scatters shots - and
    # the rotated stance scored 1.38 goals a run against 2.00 without it
    # (8 seeds x 300 s, 10.4 vs 8.4 kicks); on a 12-spot lone-shot probe
    # the direction error was 28 vs 35 deg mean absolute, noise-dominated
    # either way. The map's lesson that ships: line-up precision is the
    # next lever for goals, and the sweet spot is 6-10 cm ahead, 4-8 cm
    # to the side.
    kick_deflect_left: float = 0.0
    kick_deflect_right: float = 0.0
    lineup_range: float = 0.6      # a ball seen inside this is worth lining up on
    refresh_min: float = 0.35      # …and the spot is re-planned from sightings down to this range, then walked blind
    # The line-up is two stages (traced: with the spot 8 cm behind the
    # ball, the walk-in's last steering steps and the square-up's turn in
    # place pushed the ball 5-50 cm before the kick - the ball moved 15 cm
    # on average between the last sighting and the kick, more than the
    # 7 cm the sighting was off by). Stage one goes to a pre-spot
    # `approach_back` behind the kick spot on the kick line and squares up
    # THERE, the ball 30 cm from the feet; stage two walks straight in
    # along the line at `approach_speed` and stops on the spot.
    # OFF (measured over 8 seeds x 300 s of 1v1, goals attributed to a
    # kick within 4 s or to a bump): the two-stage line-up puts the ball
    # on the sweet spot (side error 3 cm, heading 4 deg, the ball moving
    # 2 cm before the kick, 4 goals from 11 lone shots against 1 from
    # 12) but kicks 3.5 times a run against 8.4, and those kicks scored
    # 0.00 against 0.75 kicked goals a run; bumped goals were 1.25 either
    # way. With this walker the kick that happens beats the kick that is
    # placed. `two_stage` switches it on.
    two_stage: bool = False
    approach_back: float = 0.22
    approach_speed: float = 0.25
    approach_tol: float = 0.04
    # Stage two follows the LINE, not the heading: on a pure forward
    # command the walker holds its yaw but crabs sideways (measured 9 cm
    # of side error over the 22 cm walk-in, and shots got worse), so a
    # gentle cross-track law steers it back onto the line - `k_lat` per m
    # off the line, `k_head` per rad off the heading, capped at
    # `approach_wz` so no step is a turn against the ball.
    k_lat: float = 4.0
    k_head: float = 1.5
    approach_wz: float = 0.5
    # Traced: a re-plan with the ball already inside `backoff_range` puts
    # the pre-spot behind the duck, and the turn in place toward it is a
    # turn against the ball. Back off instead (turn away, walk clear - the
    # retreat manoeuvre) and line up again from further out. And a search
    # begun with the ball at the feet turns in place without ever seeing
    # it (a standing turn barely turns): after `search_walk_after` with
    # no sighting, walk `search_walk_s` to change the view.
    backoff_range: float = 0.35
    search_walk_after: float = 0.0     # 0: off (measured with the two-stage line-up, see above)
    search_walk_s: float = 1.0
    lineup_tol: float = 0.03       # trunk within this of the kicking spot: kick
    lineup_s: float = 4.0          # give up a line-up after this long
    settle_s: float = 0.4          # stand this long on the spot before the kick (robotd kicks at standing tuning)
    kick_clear: float = 0.35       # no kick with anything closer than this ahead
    aim_tol: float = 0.25          # face the kick direction within this before kicking (rad)
    # Shoot only from inside the goal's cone. Measured on the shipped brain:
    # 7.4 kicks a run for 0.25 kicked goals - one shot in four - and a lone
    # shot's direction error is 28-35 deg, so a kick from far out is a
    # lottery whatever the line-up does. With `kick_cone` > 0 a ball whose
    # goal mouth subtends less than that half-angle is DRIBBLED instead
    # (the push spot, walked through toward the goal), which carries it
    # closer until the cone opens. 0.35 rad is about a metre out on a 0.7 m
    # goal. 0 = off (kick from anywhere).
    kick_cone: float = 0.0
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
    # The kick map says where the ball goes BEFORE it moves: +21.6 deg for
    # the left foot, -11 for the right, off the body heading. `look_aim`
    # yaws the look after a kick to that angle, at `look_aim_range` (near
    # the horizon: the ball is a metre or two out by then) instead of the
    # 0.3 m dip - so the ball, which leaves the level camera at once
    # 30-55 deg off the nose, stays in the frustum long enough to track.
    # Measured OFF (8 seeds x 300 s of 1v1, against 2.38 goals / 7.4 kicks /
    # 0.38 falls a run): 2.25 / 6.4 / 0.50; with the gaze on the track while
    # searching (predict_s 3) 1.88 / 4.6 / 0.25; with the search sweep
    # 1.88 / 6.8 / 0.38. A head turned off the walking line leaves the ToF
    # bumper looking sideways, and the brain stops for what it then sees.
    look_aim: bool = False
    look_aim_range: float = 1.5
    kick_exit_left: float = math.radians(21.6)
    kick_exit_right: float = math.radians(-11.0)
    # A searching head sweeps +-`search_sweep` rad (period `search_sweep_s`)
    # while the body circles, when it has no track to look at.
    search_sweep: float = 0.0
    search_sweep_s: float = 4.0
    # Hunt: a ball lost right after a kick, or after being walked into
    # inside `hunt_lost_range` (the histograms: half a run is search, and
    # the ball leaves the view rolling off along a known line - the kick's,
    # or the duck's own heading), is looked for by WALKING that line for
    # `hunt_s` with the head level (a floor ball is in view from 0.3 m out
    # to the camera's range) before the standing search begins.
    # ON, with its stops (below). Without them it walked into things (1.50
    # falls a run); with them, over 8 seeds x 300 s of 1v1: 8.6 kicks a
    # run against 8.4, 0.12 falls against 0.50, goals within the noise of
    # eight seeds (1.50 against 2.00, 0.25 kicked against 0.75).
    hunt_s: float = 3.0
    hunt_lost_range: float = 0.6
    # The hunt's own stops (traced: it walked at 0.45 m/s turning at full
    # rate into the boards, where the ToF returns nothing inside 3 cm, and
    # into the other duck beside it, outside the camera's cone and the
    # ToF's middle columns). Slower, a capped turn, and it ends - not
    # alternates with "blocked" - when anything is inside `hunt_stop`
    # ahead, a duck track is beside it, or the boards are `hunt_margin`
    # ahead in odometry.
    hunt_speed: float = 0.3
    hunt_wz: float = 0.5
    hunt_stop: float = 0.45
    hunt_margin: float = 0.35
    # A ball memory in odometry: the last sighting, the end of a hunted
    # line, the centre spot at a kickoff. A search with a memory further
    # than `seek_min` away WALKS there first (the hunt's speed and stops)
    # instead of circling on the spot - with the hunt and the circle,
    # 107 s of a 300 s run were still search. Forgotten after `seek_s` or
    # once there with nothing seen. OFF (0): measured over 8 seeds x 300 s
    # of 1v1 at 2.38 goals, 7.8 kicks, 0.75 falls a run against 2.25 / 9.4
    # / 0.38 without - the goals did not move and the blind walks fell.
    seek_s: float = 0.0
    seek_min: float = 0.4
    seek_tol: float = 0.25
    # The ball's trajectory (tracker: an odometry-frame position and
    # velocity from consecutive hits). Measured: a kicked ball leaves at
    # 1.4 m/s and slows at 0.04 m/s^2 on this floor - it rolls to the
    # boards - and leaves the level camera at once, 30-55 deg off the
    # nose, so the track coasted with a stale range for two seconds and a
    # new track was born when it was found again. With `predict_s` > 0
    # the head YAWS toward the predicted bearing (`head_yaw_gain`, to
    # `head_yaw_max`; always, or only while searching / looking), and
    # with `predict_steer` the search opens toward the predicted side and
    # the hunt walks to the predicted point (clamped to the pitch).
    # Measured OFF (8 seeds x 300 s of 1v1, against 2.25 goals / 9.4 kicks
    # / 0.38 falls a run with it off): yaw always + steer 1.12 / 10.5 /
    # 1.62; yaw off + steer 2.12 / 9.1 / 1.12; yaw in search + steer 2.12
    # / 10.0 / 0.75; yaw in search, no steer 2.12 / 6.5 / 0.75. The head
    # yaws 34 deg at most and the ball sits 90-120 deg off the nose while
    # searching, so the gaze cannot reach it; the steering walked blind
    # lines into things. The prediction is still tracked and drawn on the
    # /sim page; the defaults below are the best of the variants when on.
    ball_decel: float = 0.04
    predict_s: float = 0.0         # how long a prediction is worth acting on after the last hit (0: off)
    head_yaw_gain: float = 0.9
    head_yaw_max: float = 1.4          # the walker's trained head-yaw range (upstream curriculum: +-1.40 rad)
    head_yaw_when: str = "search"  # or "always": yaw the head on the ball too, not only searching / looking
    predict_steer: bool = False    # the hunt bends and the search opens toward the prediction
    search_dip_every: float = 1.5
    search_dip_s: float = 0.6
    dip_range: float = 0.22
    # The search is a slow WALKING circle (`search_vx` forward with the
    # turn), not a turn in place: instrumented over 300 s, during search
    # the ball was inside the camera's frustum 1% of the time and detected
    # 0% - it sat 90-120 degrees off the nose, and a standing turn barely
    # turns the walker (the cold-turn kick fires once, the next dip stands
    # it still again). Walking, the body rotates and the camera sweeps.
    search_vx: float = TURN_KICK
    # Turning toward the side the ball was last on (a right turn when it
    # went right) probed 4 s against 10 s for a ball to the right, and
    # measured OFF over 8 seeds: 1.62 goals, 8.9 kicks, 0.62 falls a run
    # against 2.25 / 9.4 / 0.38 always turning left - within the noise of
    # eight seeds, with the falls on the walker's weak right turn.
    search_sided: bool = False
    # Dribbling: OFF (inf). Measured — a ball pushed at 0.3 m/s for half a
    # second rolls on at about the walking speed on this floor and the duck
    # walks behind it without ever lining up; the kick wins. Re-measured
    # at 1.4 with goals attributed (8 seeds x 300 s): 2.25 goals a run
    # either way (0.25 kicked, 2.00 bumped), 6.4 kicks and 1.8 pushes
    # against 9.4 kicks, and 0.75 falls against 0.38 - the deliberate
    # bump scores no more than the accidental one and falls twice as often.
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
    # Where a supporter stands relative to the ball: "back" (toward our own
    # goal - it defends, and 3v3 scores 0.75 goals a run) or "ahead" (toward
    # the goal we attack: a poacher, in position to walk a loose ball in,
    # which is how most goals are actually scored here).
    support_mode: str = "back"
    support_side: float = 0.45
    support_min: float = 0.45
    # Traced over 3 seeds x 300 s of 3v3: 10 of 14 falls were supporters
    # turning in place with a teammate 5-28 cm away or against the boards
    # - a body beside the duck is outside the camera's 62 deg and the ToF's
    # 45 deg, so neither the avoid rule nor the wall rule saw it. Two
    # answers: the support spot stays `support_margin` inside the pitch
    # (`bounds`, from make_pitch), and a supporter with any duck track
    # inside `beside_m` (however stale within `beside_s`: the track's
    # bearing turns with the body, so a duck seen a second ago still says
    # where it is) stands instead of turning in place.
    support_margin: float = 0.35
    beside_m: float = 0.3
    beside_s: float = 1.5
    # The ToF sees the ball at the feet (tof_floor_ball): inside `tof_ball_m`
    # with the head dipped, a floor blob feeds the tracker as a ball sighting
    # when the camera has none - the level camera loses a floor ball inside
    # 0.3 m, which is where the line-up and the kick live. Measured OFF at
    # 0.5 m (8 seeds x 300 s of 1v1): 1.62 goals, 8.1 kicks, 0.75 falls a
    # run against 2.38 / 7.4 / 0.38 - a blob at the feet is as often the
    # other duck's foot as the ball, and a line-up on a foot is a fall.
    tof_ball_m: float = 0.0
    # A bump (Senses.bumped: the body is touching another body - contacts in
    # the sim, the IMU / servo loads on the robot): no turn in place for
    # `bump_stand_s` after the contact STARTED. 12 of 13 traced 3v3 falls
    # were standing turns beside an unseen opponent, and the rule halves the
    # crowd's falls (4 seeds x 300 s: 3v3 5.00 -> 1.75 a run at 0.5 s, 2v2
    # 4.00 -> 2.25) - but the first version of it also cost 1v1 1.50 goals
    # and 1.00 falls against 2.38 / 0.38, and a trace of 838 bumps said
    # exactly why, refuting the obvious guess on the way:
    #   * NOT possession. The feet meet a median 0.66 m from the ball; both
    #     ducks are inside 0.35 m of it in 18% of bumps; and two seconds
    #     later the ball is further from BOTH ducks by the same +0.074 m.
    #     Nobody is walked over - so the "exempt the duck at the ball"
    #     knob this once carried is GONE, not merely defaulted off: a knob
    #     on a premise the data refuted only invites someone to try it.
    #   * It cancelled the ESCAPE. 70% of its firing was in `blocked`, a
    #     state that is 12.6% of the run, where the walk is already zeroed
    #     and the turn is the only command left. 6 of 8 falls were a stand
    #     pressed against the other duck: the walker leans on it.
    #   * It fed itself. Standing on a body keeps touching it, which
    #     refreshed the timer: bumps went 44 -> 105 a run and one freeze
    #     ran 74 s. So the window is edge-triggered now (`bump_gap_s`).
    # 0 for a lone attacker; a roster with teammates gets
    # `team_bump_stand_s` through brain/team.py's brain_kwargs.
    bump_stand_s: float = 0.0
    team_bump_stand_s: float = 0.5
    # Only where a standing turn beside a body is the danger. Never in
    # `blocked` / `avoid` / `retreat` (the turn IS the escape) and never in
    # `search` (its circle WALKS at `search_vx`, and freezing that stops the
    # one behaviour that finds the ball - 5 of 8 traced falls were there).
    bump_stand_states: tuple[str, ...] = ("support", "lineup", "settle", "turn")
    # A contact episode ends after this long without one; the freeze runs
    # from its onset and is never extended by staying in contact.
    bump_gap_s: float = 1.0
    # Teammates' poses off the team board (brain/team.py): a teammate
    # inside `mate_keepout` counts as a duck beside me (no turn in place,
    # no hunt) and, ahead, as a duck to avoid - the camera and the ToF
    # cannot see one beside or behind me. Measured OFF (3v3, 4 seeds x
    # 300 s: 1.50 goals, 4.5 kicks, 5.25 falls a run with it at 0.4
    # against 1.50 / 5.0 / 4.50 without): a fresh trace put 12 of 13
    # falls beside an OPPONENT, which no board carries, turning in place.
    mate_keepout: float = 0.0
    # A supporter at its spot turns to face the ball WALKING (`support_turn_vx`
    # > 0, like the search circle) instead of standing - the traced 3v3
    # falls were standing turns beside a body nothing had seen. Measured
    # OFF at 0.2 (4 seeds x 300 s): 2v2 1.50 goals / 8.8 kicks / 4.00
    # falls a run against 1.50 / 9.8 / 3.50; 3v3 1.00 / 4.2 / 5.25
    # against 1.50 / 5.0 / 4.50 - the walking turn bumps what it cannot see.
    support_turn_vx: float = 0.0
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
                 team=None, duck_id: str = "", bounds: tuple[float, float] | None = None,
                 goal_w: float = 0.0):
        self.p = p
        self.goal = None if goal is None else (float(goal[0]), float(goal[1]))
        self.goal_w = float(goal_w)        # the mouth's width: how wide a target the ball has (kick_cone)
        self.bounds = None if bounds is None else (float(bounds[0]), float(bounds[1]))   # the pitch's half-extents inside the boards
        self.team = team
        self.duck_id = duck_id
        self.tracker = Tracker()
        self.gait = GaitWatch()
        self.reset()

    def reset(self) -> None:
        self.kicks = 0
        self.pushes = 0
        self.attack: float | None = None                            # heading of the goal it attacks (first odom yaw)
        self.kickoff()

    def kickoff(self) -> None:
        """Play restarts (a goal; World.kickoff put the duck back on its
        spawn): forget the ball, the spot and whatever manoeuvre was under
        way; keep the tally and the goal. On a pitch the ball is on the
        centre spot: remember that."""
        self.state = "search"
        self.role = "attack"
        self.last_bearing = 0.0
        self.last_seen_t: float | None = None
        self._senses: Senses | None = None
        self._mates: list[tuple[float, float]] = []          # (range, bearing) of live teammates, off the team board
        self._last_foot: str | None = None                   # the foot of the last kick (the look aims by it)
        self._bump_t = -1e9                                  # last contact
        self._bump_t0 = -1e9                                 # onset of the current contact episode
        self.last = (0.0, 0.0, 0.0)
        self.spot: tuple[float, float, str | None, float, str] | None = None   # x, y, foot, heading, "kick"|"push"
        self.lined = False                      # stage two of the line-up: on the line, walking straight in
        self.t_state = 0.0
        self._yield_t0 = -9.0
        self._yield_end = -9.0
        self._poses: list[tuple[float, float, float, float]] = []      # (t, x, y, yaw) over the stuck window
        self._retreat_t0 = -9.0
        self._retreat_sign = 1.0
        self._look_t0 = -9.0
        self._hunt_t0 = -9.0
        self._hunt_u: float | None = None           # the line to walk (odometry heading)
        self.memory: tuple[float, float, float] | None = None   # (x, y, t) where the ball was, odometry frame
        if self.goal is not None:
            self.memory = (0.0, 0.0, 0.0)           # a pitch: play starts from the centre spot
        self._last_range: float | None = None
        self._search_t0: float | None = None
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
                        "bumped": round(max(0.0, self._senses.t - self._bump_t), 2) if self._bump_t > -1e8 else None,
                        "tofBall": None if getattr(self, "tof_ball", None) is None else
                        [round(self.tof_ball[0], 2), round(self.tof_ball[1], 2)],
                        "memory": None if self.memory is None else [round(self.memory[0], 2), round(self.memory[1], 2)],
                        "predicted": None if getattr(self, "predicted", None) is None else [round(self.predicted[0], 2), round(self.predicted[1], 2)],
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
            if p.kick_cone > 0 and self.goal_cone(bx, by) < p.kick_cone:
                far = True                          # too fine a target from here: dribble it closer
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
        # The body heading that sends the kick along u (the map's deflection
        # is in the body frame, so the spot is laid out in that heading too).
        h = _wrap(u - (p.kick_deflect_left if foot == "kick_left" else p.kick_deflect_right))
        return (bx - p.kick_ahead * math.cos(h) - side * math.sin(h),
                by - p.kick_ahead * math.sin(h) + side * math.cos(h), foot, h, "kick")

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
        det_in = senses.fresh_det(self.DET_MAX_AGE)
        self.tof_ball: tuple[float, float] | None = None
        if p.tof_ball_m > 0 and (det_in is None or not any(d.cls == "ball" for d in det_in.detections)):
            tof_fr = senses.fresh_tof(self.TOF_MAX_AGE)
            blob = None if tof_fr is None else tof_floor_ball(tof_fr, r_max=p.tof_ball_m)
            if blob is not None:
                self.tof_ball = blob
                from ..sensors.detector import Detection, DetectionFrame
                det_in = DetectionFrame(t=tof_fr.t, detections=[Detection("ball", "", blob[0], -0.6, 0.2, blob[1], 0.8)])
        self.tracker.update(det_in, t, odom[2], (odom[0], odom[1]) if senses.odom is not None else None)
        ball = self.tracker.best(p.target_cls, t, min_hits=1)
        fresh = ball is not None and ball.age(t) <= self.DET_MAX_AGE
        seen = ball is not None and ball.age(t) < p.lost_s
        # Where the ball is going: its predicted position, and the bearing
        # to it from here (the head looks there; the search opens there).
        self.predicted: tuple[float, float] | None = None
        pred_bearing: float | None = None
        if ball is not None and ball.xy is not None and ball.age(t) <= p.predict_s:
            px, py = ball.predict(t, p.ball_decel)
            if self.bounds is not None:
                px = float(np.clip(px, -self.bounds[0] + 0.1, self.bounds[0] - 0.1))
                py = float(np.clip(py, -self.bounds[1] + 0.1, self.bounds[1] - 0.1))
            self.predicted = (px, py)
            pred_bearing = _wrap(math.atan2(py - odom[1], px - odom[0]) - odom[2])
        self._mates = []
        if senses.bumped:
            if t - self._bump_t > p.bump_gap_s:
                self._bump_t0 = t                            # a NEW contact, not the same one continuing
            self._bump_t = t
        if self.team is not None:
            self.team.claim(self.duck_id, t, ball.range if seen else math.inf,
                            self._ball_xy(odom, ball) if seen else None, (odom[0], odom[1], odom[2]))
            self.role = self.team.role(self.duck_id, t)
            for _, (mx, my, _) in self.team.mates(self.duck_id, t):
                self._mates.append((math.hypot(mx - odom[0], my - odom[1]),
                                    _wrap(math.atan2(my - odom[1], mx - odom[0]) - odom[2])))
        other = self.tracker.best("duck", t, min_hits=1)
        # The nearest duck ahead to avoid: a seen one, or a teammate by the board.
        threats = [(r, b) for r, b in self._mates if r < p.mate_keepout and abs(b) < p.duck_bearing]
        if other is not None and other.age(t) <= 0.6 and other.range < p.duck_keepout and abs(other.bearing) < p.duck_bearing:
            threats.append((other.range, other.bearing))
        duck_rb = min(threats) if threats else None
        near_duck = duck_rb is not None
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
            # Body-height things only (not the floor the head looks at, not
            # the ball), selected by bearing so a turned head cannot report
            # a wall that is really off to the side.
            ahead, left_near, right_near = tof_clearance_bearings(tof)
        skill = None
        head = (0.0, 0.0, 0.0, 0.0)
        gaze_at: float | None = None
        retreating = t - self._retreat_t0 < p.retreat_turn_s + p.retreat_walk_s
        if self._prev_skill is not None and senses.skill is None:
            self._look_t0 = t                                   # the kick window just ended: look for the ball ahead
        self._prev_skill = senses.skill
        looking = t - self._look_t0 < p.look_s and not fresh
        if seen:
            self._last_range = ball.range
            self.last_bearing = ball.bearing
            if fresh:
                bx_, by_ = self._ball_xy(odom, ball)
                self.memory = (bx_, by_, t)
        elif self._last_range is not None and self._last_range < p.hunt_lost_range and self.state in ("chase", "lineup", "turn"):
            self._hunt_u = odom[2]                              # walked into it: it rolled off ahead
            self._last_range = None
        if self.state == "look" and not looking and not fresh and self._hunt_u is not None:
            self._hunt_t0 = t                                   # the look after the kick found nothing: hunt the line
        hunting = p.hunt_s > 0 and t - self._hunt_t0 < p.hunt_s and not seen and self._hunt_u is not None
        if p.hunt_s > 0 and self._hunt_u is not None and not seen and t - self._hunt_t0 >= p.hunt_s \
                and self.state == "hunt":
            # The hunt ran its course without a sighting: the ball is further
            # along the line - remember a point there and forget the line.
            self.memory = (odom[0] + 0.6 * math.cos(self._hunt_u), odom[1] + 0.6 * math.sin(self._hunt_u), t)
            self._hunt_u = None
        if hunting and (ahead < p.hunt_stop or self._beside(t) or self._boards_ahead(odom, p.hunt_margin)):
            hunting = False                                     # the hunt ends here; the search takes over
            self._hunt_t0 = -1e9
            self._hunt_u = None
        if (not seen and not hunting and self.memory is not None
                and (t - self.memory[2] > p.seek_s
                     or math.hypot(self.memory[0] - odom[0], self.memory[1] - odom[1]) <= p.seek_min)):
            self.memory = None                                  # stale, or here with nothing seen: forget it
        seeking = p.seek_s > 0 and not seen and not hunting and self.memory is not None
        if seeking and (ahead < p.hunt_stop or self._beside(t)):
            seeking = False                                     # something in the way: circle here instead
            self.memory = None
        if senses.skill is not None:
            vx, wz = 0.0, 0.0                                   # the kick owns the reflex tier
            self.state = "kick"
        elif looking:
            vx, wz = 0.0, 0.0
            gaze_at = p.look_aim_range if p.look_aim else p.look_range
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
            if duck_rb[0] < p.duck_touch:
                vx, wz = 0.0, 0.0
            else:
                vx, _, wz = turn(-1.0 if duck_rb[1] >= 0.0 else 1.0, cold)
                if abs(duck_rb[1]) < 0.5:
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
                new = self._plan(odom, ball)
                if self.lined and math.hypot(new[0] - self.spot[0], new[1] - self.spot[1]) > 0.05:
                    self.lined = False                          # the ball is not where the line was laid: lay it again
                self.spot = new
            sx, sy, foot, u, mode = self.spot
            heading_err = _wrap(u - odom[2])
            if mode == "kick" and p.two_stage and not self.lined:
                # Stage one: the pre-spot behind the kick spot on the line;
                # square up there, where a turn in place cannot touch the ball.
                px, py = sx - p.approach_back * math.cos(u), sy - p.approach_back * math.sin(u)
                vx, wz, pdist, bearing = self._servo(odom, (px, py), cold, p.approach_tol)
                ball_rng = math.hypot(sx + p.kick_ahead * math.cos(u) - odom[0], sy + p.kick_ahead * math.sin(u) - odom[1])
                if abs(bearing) > 1.8 and ball_rng < p.backoff_range and pdist > p.approach_tol + 0.02:
                    # The pre-spot is behind us with the ball at our feet:
                    # turning to it is a turn against the ball. Back off.
                    self.spot = None
                    self._retreat_t0 = t
                    self._retreat_sign = -1.0 if _wrap(math.atan2(sy - odom[1], sx - odom[0]) - odom[2]) >= 0 else 1.0
                    vx, _, wz = turn(self._retreat_sign, cold)
                    self.state = "retreat"
                    dist = 9.0
                elif pdist <= p.approach_tol + 0.02 and abs(heading_err) <= p.aim_tol:
                    self.lined = True
                    self.t_state = t                            # stage two gets its own clock
                elif pdist <= p.approach_tol + 0.02:
                    vx, _, wz = turn(heading_err, cold)
                dist = pdist + p.approach_back if self.spot is not None else 9.0   # nowhere near the spot yet
            else:
                vx, wz, dist, bearing = self._servo(odom, (sx, sy), cold, p.lineup_tol)
                if mode == "kick" and p.two_stage and self.state != "settle":
                    # Stage two: in along the line, steering onto it (the
                    # walker crabs on a pure forward command), stop on the
                    # spot by the distance left along the line.
                    along = (sx - odom[0]) * math.cos(u) + (sy - odom[1]) * math.sin(u)
                    lat = -(odom[0] - sx) * math.sin(u) + (odom[1] - sy) * math.cos(u)   # +: left of the line
                    if along > p.lineup_tol:
                        vx = p.approach_speed
                        wz = float(np.clip(-p.k_lat * lat + p.k_head * heading_err, -p.approach_wz, p.approach_wz))
                        dist = along
                    else:
                        vx, wz, dist = 0.0, 0.0, 0.0
            # Hysteresis on both: a settling duck wobbles a centimetre and a
            # few hundredths of a radian, which flipped it between the
            # square-up and the settle at the tolerance (measured: 22 s
            # standing at the spot, no kick).
            settling = self.state == "settle"
            on_spot = dist <= p.lineup_tol + (0.03 if settling else 0.0)
            squared = abs(heading_err) <= p.aim_tol + (0.15 if settling else 0.0)
            if self.spot is None:
                pass                                            # backing off (above)
            elif on_spot and not squared and mode == "kick" and p.two_stage:
                # On the spot but off the heading: a turn in place here is a
                # turn against the ball (traced: 14 s of it). Back off and
                # lay the line again from further out.
                self.spot = None
                self._retreat_t0 = t
                self._retreat_sign = -1.0 if heading_err >= 0 else 1.0
                vx, _, wz = turn(self._retreat_sign, cold)
                self.state = "retreat"
            elif on_spot and not squared:
                vx, _, wz = turn(heading_err, cold)            # a push spot: square up
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
                        self._last_foot = foot
                        self.kicks += 1
                        self.spot = None
                        self.state = "kick"
                        self._hunt_u = u                        # where the ball is going
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
                self.lined = False
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
        elif hunting:
            if self.predicted is not None and p.predict_steer:  # the line bends to where the ball is going
                self._hunt_u = math.atan2(self.predicted[1] - odom[1], self.predicted[0] - odom[0])
            vx, wz = p.hunt_speed, float(np.clip(p.k_turn * _wrap(self._hunt_u - odom[2]), -p.hunt_wz, p.hunt_wz))
            self.state = "hunt"
        elif seeking and self.state not in ("look",):
            # Walk to where the ball was, head level, at the hunt's pace.
            vx, wz, sdist, sbear = self._servo(odom, self.memory[:2], cold, p.seek_tol)
            vx = min(vx, p.hunt_speed) if abs(sbear) <= 0.5 else vx
            if sdist <= p.seek_tol:
                self.memory = None                              # here, and nothing seen: forget it
            self.state = "seek"
        else:
            if p.hunt_s > 0 and self._hunt_u is not None and self.state not in ("search", "hunt", "look") \
                    and self._last_range is None and t - self._hunt_t0 >= p.hunt_s \
                    and ahead >= p.hunt_stop and not self._beside(t) and not self._boards_ahead(odom, p.hunt_margin):
                self._hunt_t0 = t                               # lost while walking into it: hunt before searching
                vx, wz = p.hunt_speed, float(np.clip(p.k_turn * _wrap(self._hunt_u - odom[2]), -p.hunt_wz, p.hunt_wz))
                self.state = "hunt"
            else:
                if self.state != "search" or self._search_t0 is None:
                    self._search_t0 = t
                # Toward the side the ball was last on (probed: the sweep runs
                # at ~24 deg/s, so a ball to the right found by a left turn
                # takes 10 s, by a right turn 4); the cold-turn kick starts
                # the right turn the standing walker cannot.
                side = pred_bearing if pred_bearing is not None and p.predict_steer else (self.last_bearing if p.search_sided else 1.0)
                vx, _, wz = turn(1.0 if side >= 0.0 else -1.0, cold)
                vx = max(vx, p.search_vx)                       # a walking circle: the body actually turns
                self.state = "search"
                since = t - self._search_t0
                if since % p.search_dip_every < p.search_dip_s:
                    vx, wz = 0.0, 0.0                           # a standing look down: a near ball is below the level camera
                    gaze_at = p.dip_range
                elif p.search_walk_after and since > p.search_walk_after and (since - p.search_walk_after) % (p.search_walk_after) < p.search_walk_s:
                    vx, wz = p.speed, 0.0                       # a standing turn barely turns: move to see from elsewhere
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
        look_at = pred_bearing if pred_bearing is not None else (
            ball.bearing if p.predict_s > 0 and ball is not None and ball.age(t) <= p.predict_s else None)
        if look_at is None and self.state == "look" and p.look_aim and self._last_foot is not None:
            look_at = p.kick_exit_left if self._last_foot == "kick_left" else p.kick_exit_right
        elif look_at is None and self.state == "search" and p.search_sweep > 0 and self._search_t0 is not None:
            look_at = p.search_sweep * math.sin(2.0 * math.pi * (t - self._search_t0) / p.search_sweep_s) / p.head_yaw_gain
        if look_at is not None and senses.skill is None and (p.head_yaw_when == "always" or self.state in ("search", "look")):
            head = (head[0], head[1], float(np.clip(p.head_yaw_gain * look_at, -p.head_yaw_max, p.head_yaw_max)), head[3])
        if p.bump_stand_s > 0 and t - self._bump_t0 < p.bump_stand_s and vx <= TURN_KICK and wz != 0.0 \
                and self.state in p.bump_stand_states:
            vx, wz = 0.0, 0.0                                   # touching a body: stand, do not turn in place
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
        og = self._own_goal(odom) if p.support_mode == "back" else (
            self.goal if self.goal is not None else self._own_goal(odom))
        gx, gy = og[0] - bxy[0], og[1] - bxy[1]
        gn = math.hypot(gx, gy)
        ux, uy = (gx / gn, gy / gn) if gn > 1e-6 else (-math.cos(odom[2]), -math.sin(odom[2]))
        rank = self.team.rank(self.duck_id, t) if self.team is not None else 0
        side = p.support_side * ((rank + 1) // 2) * (1 if rank % 2 == 0 else -1)
        target = (bxy[0] + p.support_back * ux - side * uy, bxy[1] + p.support_back * uy + side * ux)
        if self.bounds is not None:                         # never a spot in the boards
            m = p.support_margin
            target = (float(np.clip(target[0], -self.bounds[0] + m, self.bounds[0] - m)),
                      float(np.clip(target[1], -self.bounds[1] + m, self.bounds[1] - m)))
        vx, wz, dist, _ = self._servo(odom, target, cold, 0.12)
        if math.hypot(bxy[0] - odom[0], bxy[1] - odom[1]) < p.support_min and vx > 0:
            vx = 0.0                                        # the attacker's room
        if dist <= 0.12:
            b = _wrap(math.atan2(bxy[1] - odom[1], bxy[0] - odom[0]) - odom[2])
            vx, wz = (0.0, 0.0) if abs(b) < 0.3 else turn(b, cold)[::2]
            if wz != 0.0:
                vx = max(vx, p.support_turn_vx)
        if vx <= TURN_KICK and wz != 0.0 and self._beside(t):
            vx, wz = 0.0, 0.0                               # a body beside us: no turning in place
        self.state = "support"
        return vx, wz

    def goal_cone(self, bx: float, by: float) -> float:
        """Half the angle the goal mouth subtends from a ball at (bx, by),
        in the odometry frame: how fine a target this shot is. +inf off a
        pitch or without a mouth width."""
        if self.goal is None or self.goal_w <= 0:
            return math.inf
        gx = self.goal[0]
        a1 = math.atan2(-self.goal_w / 2 - by, gx - bx)
        a2 = math.atan2(self.goal_w / 2 - by, gx - bx)
        return abs(_wrap(a2 - a1)) / 2.0

    def _boards_ahead(self, odom, margin: float) -> bool:
        """The point `margin` ahead in odometry lies outside the pitch's bounds (None off a pitch: never)."""
        if self.bounds is None:
            return False
        x, y = odom[0] + margin * math.cos(odom[2]), odom[1] + margin * math.sin(odom[2])
        return abs(x) > self.bounds[0] or abs(y) > self.bounds[1]

    def _beside(self, t: float) -> bool:
        """Any duck track inside `beside_m`, at any bearing, within `beside_s`;
        or a teammate inside `mate_keepout` by the team board."""
        p = self.p
        return any(tr.cls == "duck" and tr.range < p.beside_m and tr.age(t) <= p.beside_s
                   for tr in self.tracker.tracks) or any(r < p.mate_keepout for r, _ in self._mates)


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
