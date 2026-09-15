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
import os
import warnings
from dataclasses import dataclass, fields, replace
from typing import ClassVar

import numpy as np

from ..contract import CTRL_DT
from .gait import TURN_KICK, GaitWatch, back_up, clip_wz, max_wz, turn
from .intercept import Interceptor
from .runtime import REGISTRY, Intent, Senses, age_inputs
from .tracker import Tracker, TrackerParams

# The camera's horizontal HALF-field (62° full, sensors/detector.py) plus a
# little. The shipped default of `ChaseParams.gaze_bearing_max`, and it was
# WRONG about why: "past this a target is not in the picture at any head
# pitch" is true of a LEVEL camera only. Pitch the camera down and the
# azimuth stops being the bearing the lens sees - measured on the composed
# model (2026-09-06): a ball 37° off the nose on the kick spot is at 16.5°
# camera bearing, -19.7° elevation, inside the frustum, with the head at
# 60° down, and nothing on the duck occludes it. The neck-carried gaze
# reaches ~52° at `head_down` (0.75 + 0.43 per unit, 11° at rest).
GAZE_MAX_BEARING = 0.6


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
    # DEAD KNOB - declared, never read. The search turn goes through
    # `gait.turn()`, which takes its magnitude from `gait.max_wz()`. Left in
    # place with this note rather than deleted because deleting it silently
    # would let the next person re-add it; sweeping it would measure exactly
    # nothing, which is the failure mode this file documents elsewhere.
    # (Below 1.0 a COLD walker does not turn at all - exactly 0, both ways.)
    search_wz: float = 1.0
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
            wz = clip_wz(p.k_turn * target.bearing + p.k_lead * float(np.clip(rate, -2.0, 2.0)))
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
                wz = max_wz()
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


class ReentrantStepWarning(RuntimeWarning):
    """A brain was stepped twice for the same tick.

    Raised for INSTRUMENTS, not harnesses: `Chase.step` mutates state from its
    first line, so a probe that re-enters it to compare two variants advances
    the gait, the localiser and every counter twice and measures neither. One
    such probe reported a 7.56% firing rate that a direct sweep of the same
    knob showed was zero (roadmap 12o). Read the flags AFTER a single step.
    """


@dataclass(frozen=True)
class ChaseParams:
    """The chase brain's constants. Most were measured INTO their value; a
    good few were measured OFF and keep their number in the comment beside
    them, because "we tried it and it was worse" is the expensive part and
    deleting it invites the next person to spend the afternoon again.

    Shipping OFF (0 / False), with their measurements below: `two_stage`,
    `lineup_lat` (its speed-up) and `search_walk_after` (line-up
    precision), `kick_deflect_*` (the kick
    map in the stance), `kick_cone` (shoot only from close),
    `predict_steer` (a walked prediction line), `look_aim`, `search_sweep`,
    `gaze_still`/`gaze_neck`/`gaze_yaw` (what the
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
    # THE FRESHNESS GATE, IN THE SENSOR'S OWN PERIODS (roadmap 12av
    # follow-up (1)). `Chase.DET_MAX_AGE` is a constant 0.4 s, which is four
    # detector periods at the lab's 10 Hz default and SHORTER than one period
    # at the robot's documented, thermally-bound 2 Hz (0.5 s). At that rate
    # the shipped brain calls its own newest detection stale on 26.7 % of its
    # ticks before anything has been missed (measured; 12av's "a sixth" aged
    # a frame from its ARRIVAL, and `Senses.det_age` ages it from its CAPTURE,
    # which is the number the gate compares) — a mis-specification of the
    # brain, not a property of the hardware, and it is what the 2 Hz arm in
    # 12av was measuring. This expresses the same gate in periods of whatever
    # camera `MICRODUCK_CAMERA` builds, read once at construction:
    #
    #     gate = max(DET_MAX_AGE, det_max_periods * detector period)
    #
    # It can only ever LENGTHEN the gate, and at 10 Hz (period 0.1 s) and
    # 5 Hz (0.2 s) every setting up to 4 and 2 periods respectively is exactly
    # the shipped 0.4 s, so the arm is byte-identical at every rate the lab
    # has measured. Ships OFF (0.0 = the bare constant) so every row on disk
    # still reproduces at every rate, 2 Hz included; 1.0 is "one frame late is
    # not stale", 1.5 adds the frame's arrival jitter on top.
    det_max_periods: float = 0.0
    tof_stop: float = 0.3          # walls and ducks (body-height ToF returns); the ball and the floor do not count
    side_stop: float = 0.22        # a wall this close in the side columns: no turn in place toward it
    # THE LINE-UP'S OWN STOP (roadmap 12am, built after 12al). `tof_stop`
    # halts every walk 0.30 m from anything body-height ahead, and that is
    # what kills a line-up at the boards: the spot is body-reachable
    # (`spot_reach` checked it, 0.13-0.20 m from the wall) but a servoed
    # approach faces the wall until its last step, so the bumper fires
    # ~0.3 m out and the duck stands 21 cm short of its spot until
    # `lineup_s` (probe_board_states: 123 timeouts of 144 line-ups, 6 % ever
    # within 5 cm of the spot). Here the spot geometry, not the bumper, is
    # what keeps the body out of the wall: within `lineup_tof_within` of a
    # kick spot the selector has passed as clear of every board by the body
    # extent, the walk stops at `lineup_tof_stop` instead. 0 = off (the
    # shipped bumper everywhere). Ducks are the risk - the ToF cannot tell
    # a duck from a wall - which is why it only applies inside the window
    # and why the 2v2 ledger's falls are its veto.
    #
    # SHIPS ON at 0.12 (roadmap 12am). Kick gym, the ball at a board, two
    # seed blocks: swings 34 -> 66 and 38 -> 83, connected kicks 66 -> 122
    # pooled (+85 %), falls 1 -> 2 and 0 -> 0; whiff there doubles, 8 ->
    # 18 % pooled (each block unresolved), the swings it buys being worse
    # swings than the few the bumper let through, and the connected travel
    # at the boards drops 1.15 -> 0.79 m. Open play: whiff a null both
    # blocks (8 -> 11, 11 -> 10 %), connected +5 %, falls 0 -> 1 and 1 -> 2
    # in ~400 swings. Corners: nobody swings either way. 2v2 ledger (24 x
    # 300 s): possession 40.0 -> 39.9 null, spread / crowd / depth null,
    # goals 21 = 21, kicks 148 -> 169, falls 5 -> 3, own goals 4 = 4,
    # back-kicks 19 -> 22 % (events, unresolved). At 0.18 half the gain
    # (connected 66 -> 89) for the same whiff. What it does NOT fix: with
    # the stop the duck reaches its spot (7 cm at the timeouts, 21 before)
    # and still times out with a 66 deg heading error - at 5 cm it is
    # outside `lineup_tol` 0.03, servos along the wall instead of turning,
    # and the clock runs out. The square-up at a reached spot is the next
    # lever (probe_board_states attributes every line-up's end).
    lineup_tof_stop: float = 0.12
    lineup_tof_within: float = 0.45
    # The shipped kicks (measured, `walker-facts`-style, on the walker): a
    # ball 0.08 m ahead of the trunk and 0.06 m to the kicking foot's side
    # flies 1.6 m; 0.10 m dead ahead barely moves; the other side, nothing.
    # 0.08 is the WALK-IN'S MINIMUM, not the kick's optimum. Measured on the
    # rolling-resistance floor (2026-09-06, 24 seeds, runs/ahead/): at 0.06
    # and 0.05 the duck still reaches its spot to 1.5 cm, but the ball has
    # moved since the plan - drift 0.042 -> 0.108 m, +5.4 cm p=0.002 and
    # +7.0 cm p=0.031, the only significant effect - and sits further out
    # (spot-to-ball 0.135 -> 0.165). The feet reach a ball 5-6 cm ahead
    # before the settle does, and bump it. Whiff 39 -> 44/50%, on-spot
    # 14 -> 7/9%, all the wrong way. Roadmap Track 4 item 7.
    kick_ahead: float = 0.08
    # THE BOARDS (roadmap Track 4 item 11b). A kick spot laid closer than
    # this to the boards - inside them, or inside the walker's own
    # `tof_stop` of them - is never reached: the line-up stands against
    # the wall for `lineup_s` and times out (measured: the ball is at the
    # boards 72% of a 3v3 run and 0 of 36 kicks were taken there, every
    # boards line-up a timeout). With this the spot is laid on the line
    # ALONG the wall instead (up the pitch on a side board, toward the
    # middle on an end board), the foot chosen so the body stands on the
    # open side. The margin is what the body can stand at: a ball AGAINST
    # the wall (radius 0.035) with the kick_side offset (0.06) puts the
    # trunk 0.095 m from it, so a margin above that never fires for the
    # ball that matters (0.12 was measured first: kicks 2.9 -> 4.2 a run
    # at p = 0.056, firing only for balls 8 cm or more off the wall).
    # MEASURED OFF on the ball-out floor (World.ball_out_s, the lab's
    # pitches): 12 seeds of 3v3, kicks 7.7 -> 7.9 at 0.08 (p = 0.79) and
    # -> 8.7 at 0.12 (p = 0.38), dead ball, possession, progress all flat -
    # the referee's placement already takes the ball off the wall, and a
    # kick along it adds nothing. Kept, off, for a pitch without the rule
    # (eval-pitch's baseline), where it is the only thing that ever kicks
    # a ball at the boards. Roadmap Track 4 item 11b.
    board_margin: float = 0.0
    # REACHABILITY AS A CONSTRAINT IN THE SELECTOR (roadmap 12v / 12aa, built
    # 2026-09-10). `board_margin` above was measured a structural no-op: it
    # rejects a spot AFTER the line is chosen and then escapes along the
    # wall, one number doing two jobs. The census (12aa) says two in five
    # kick plans laid against a board put the spot where the walking body
    # (0.129 m of trunk extent) cannot stand, two in three in a corner - and
    # every such line-up stands against the wall for `lineup_s` and times
    # out. This is the other design: when `kick_select` lays its fan, a
    # candidate whose STAND spot is closer than this to a board is not
    # offered at all, so the selector ranks only spots the body can occupy
    # and picks the best of those. When NO candidate is reachable (a tight
    # corner) the fan is left whole and the plan is what it always was: the
    # constraint never makes a plan worse than the shipped one, it only
    # removes the choice of standing in a wall where another choice exists.
    # 0 = off (the pre-2026-09-10 fan, to the bit). The body extent is the
    # number.
    #
    # SHIPS ON at 0.129 (roadmap 12al). Kick gym, two seed blocks each: open
    # play whiff 9 -> 8 % and 9 -> 11 % (null both ways), connected kicks
    # and falls flat, plans the body cannot occupy 3 % -> 0 %; the ball at
    # a board, swings 16 -> 34 and 25 -> 38 (connected 36 -> 66 pooled,
    # +83 %), unreachable plans 50 -> 15 % and 45 -> 12 %; corners 68 ->
    # 32 % unreachable and nobody swings either way. 2v2 ledger (24 x 300 s)
    # flat: possession 39.2 -> 40.0 (null), spread / crowd / depth null,
    # kicks 147 -> 148, goals 24 -> 21, falls 9 -> 5, own goals 6 -> 4,
    # back-kicks 24 -> 19 % (events, unresolved). What it does NOT fix: 93 %
    # of board line-ups still time out with a reachable spot, because the
    # ToF bumper (`tof_stop` 0.30) halts a servoed approach ~0.3 m from the
    # wall, 21 cm short of the spot (scripts/probe_board_states.py) - body-
    # reachable is not walker-reachable, and that predicate is the
    # approach's, not the spot's. With `board_margin` also on, this acts
    # FIRST: a ball 4 cm off a wall keeps its one body-reachable line (the
    # line of sight) and the rescue never sees the scoring line whose spot
    # was in the wall (tests/test_ball_out.py measures the rescue at 0).
    spot_reach: float = 0.129
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
    #
    # RE-MEASURED 2026-09-06 on the quantity it moves, because the verdict
    # above was 8 seeds judged on GOALS (which need 136). It is refuted
    # again, far more strongly, and now with a mechanism. 24 seeds x 300 s
    # of 2v2 (`runs/deflect/`), paired, against 178 kicks / 32.0 deg mean
    # absolute error:
    #
    #   compensating by the measured error (+13.7 / -6.0 deg)
    #       120 kicks (-2.42 a seed, p=0.004)   |err| 44.7 (+13.1, p=0.0005)
    #   compensating by the in-play map (+23.6 / -28.7 deg)
    #        80 kicks (-3.82 a seed, p<1e-4)    |err| 43.0 (+11.4, p=0.037)
    #
    # It does not halve the error, it DOUBLES it, and it costs a third to a
    # half of the touches. WHY: the spot is laid out in the rotated heading,
    # so rotating the stance does not pre-aim the shot - it moves where the
    # duck stands. Measured: rotating the left stance +23.6 deg shifts the
    # ball's departure off the body by +24.0 deg, essentially 1:1, and the
    # ball's SIDE offset grows +8.7 cm (p=0.0006). At the measured
    # +1.90 deg of aim error per cm of side offset that predicts +16.5 deg
    # of extra error; +13.1 and +11.4 were observed. The coefficient
    # predicts its own failure.
    #
    # THE GENERAL LESSON, which kills a family of ideas and not just this
    # knob: you cannot fix this kick by ROTATING anything. The kick spot is
    # defined relative to the body heading, so every rotation moves the
    # ball's side offset, and the side offset IS the error. That includes
    # rotating the intended line by the PREDICTED offset, which this
    # roadmap proposed earlier the same day and this refutes. The only
    # levers left are the offset itself, or declining the shot when it is bad.
    kick_deflect_left: float = 0.0
    kick_deflect_right: float = 0.0
    # DECLINE the shot when the ball is too far to the side to be worth
    # swinging at. This is the only lever the coefficient leaves open: the
    # side offset IS the aim error (+1.90 deg/cm) and no rotation can remove
    # it, because every rotation moves the offset (see `kick_deflect_*`).
    # What is left is not to swing. Metres of |side| offset at the swing
    # above which the duck drops its spot and re-approaches; 0 = off.
    # The sweet spot is 0.04-0.08 m and the observed median is 0.133 m.
    #
    # It refuses only on a FRESH estimate (`Chase.predicted`, which needs
    # `predict_s` > 0 — on since 2026-09-06). A stale ball gets its swing:
    # that is what stops the gate turning into a duck that never kicks.
    #
    # MEASURED OFF the day it was built. 24 paired seeds x 300 s of 2v2
    # (`runs/decline/`), against 178 kicks / 137 effective / 23.0% whiffs:
    #
    #   decline > 0.12 m   152 kicks, 124 effective, 18.4% whiff
    #   decline > 0.09 m   133 kicks, 107 effective, 19.5% whiff
    #
    # The per-foot bias LOOKS like it is coming out (left +13.7 -> +7.1 ->
    # +4.1, right -6.0 -> -0.6 -> +1.6), and that is the trap: every one of
    # those intervals spans zero, the baseline's included. Paired per seed
    # nothing improves — whiff rate -4.8 points at 0.12 is p = 0.059, mean
    # absolute error moves the WRONG way (+2.7, +5.3), and kicks fall 1.88 a
    # seed at 0.09 (p = 0.002). Rule 6 again: a better rate on fewer touches.
    #
    # THE DIAGNOSTIC: the side offset of the kicks that survived did not
    # change (-0.002 m, p = 0.83; +0.004 m, p = 0.75). The obvious reading is
    # that the estimate is junk and the gate refuses at random. That reading
    # was written here first and it is WRONG — `scripts/probe_shot_gate.py`
    # measures the signal directly, over 162 kicks on 24 seeds:
    #
    #   |predicted side| vs |actual side|   r = +0.48 (p = 7e-5)
    #   the estimate's own error            median 0.023 m, 90th 0.083
    #   as a gate at 0.12 m                 refuses 31% of the swings it can
    #                                       see, and 88% of those really were
    #                                       wide
    #
    # The estimate is GOOD. What it is not is AVAILABLE: the brain has a
    # fresh fix on **34% of swings**. So the gate can only assess a third of
    # the population, refuses about a tenth of all swings, and cannot move a
    # pooled statistic that the other two thirds still dominate — while
    # paying the full price in touches for the ones it does refuse.
    #
    # Kept, off, with the numbers. The mechanism is sound and the precision
    # is there; COVERAGE is what is missing, and coverage is the camera's
    # blind radius (4c), not a threshold to retune. Re-run this the day the
    # brain can see the ball inside 0.35 m — and not before.
    kick_side_max: float = 0.0
    # The AHEAD gate (roadmap item 12a, 2026-09-08): the bench replay of 93
    # play swings found the whiffs are the swings taken at a ball that is no
    # longer in front of the foot - inside a 0.15 x 0.12 m box the swing
    # connects 92% of the time, outside it 24% (465 swings) - and those far
    # balls ARE visible at the line-up gaze (the floor from 0.12 m out), so
    # unlike the side gate this one has coverage. A predicted ball further
    # ahead than this refuses the swing and lays the line again from a fresh
    # sighting; 0 = off. Measured with `gaze_still` (the numbers are on it):
    # alone it fires on 5% of swings, since the track is stale at the swing
    # without the held gaze (whiff 44 -> 38% on seeds 0-23).
    #
    # COVERAGE, MEASURED 2026-09-10 (roadmap 12aj): the belief this reads
    # (`self.predicted`) expires at `predict_s` 1.0 s and the ball track is a
    # median 1.54 s old at the swing (the ball is under the chin through the
    # settle), so the belief exists on 12 % of swings and on NONE of the 41
    # swings with the ball truly > 0.15 m ahead (whiff 44 %). Extending the
    # horizon (`predict_s` 2.5) puts a belief on 81 % of swings and the gate
    # still fires on none of the far ones: the belief says 9.2 cm where the
    # truth is 17.7 - the ball moved during the settle and nothing saw it.
    # A gate on the belief cannot refuse a swing the belief is wrong about.
    kick_ahead_max: float = 0.15
    # Plan the kick spot for where the ball WILL be when the duck gets there,
    # not where it was last seen: at most this many seconds of lead, from the
    # track's own velocity and `ball_decel`. 0 = off.
    #
    # MEASURED WHY THIS EXISTS. Over 191 kicks the duck reaches its spot to
    # 1.4 cm — the walk-in is not the problem — and the SPOT is a median
    # 0.349 m from the ball when it should be `kick_ahead` = 0.08 m, because
    # the ball drifts 0.22-0.27 m during a line-up that is 3.3 s old by the
    # time the swing fires. Not one of those 191 kicks had the ball on the
    # sweet spot. So the plan is right when it is made and stale when it is
    # used, and a lead is the only one of the three ways out that addresses
    # the drift rather than avoiding it (`scripts/probe_kick_line.py`).
    #
    # SHIPS OFF: MEASURED AND IT DOES NOT WORK. Swept 0 / 0.5 / 1.0 / 2.0 s
    # over 24 seeds x 300 s of 2v2 each, judged on the on-spot fraction:
    #
    #     lead   kicks   on the sweet spot   whiffed   spot-to-ball   plan age
    #     0.0     191          0%              18%        0.285 m      3.26 s
    #     0.5     175          0%              22%        0.264 m      2.92 s
    #     1.0     125          0%              21%        0.261 m      2.98 s
    #     2.0     130          0%              24%        0.287 m      3.22 s
    #
    # Zero of every arm, and the whiff rate rises. The reason is the same
    # blindness that causes the staleness: the track's velocity is
    # differenced from SIGHTINGS, and the sightings stop at `refresh_min`
    # (0.35 m) — so the prediction is extrapolated from data that is exactly
    # as old as the plan it is meant to rescue. You cannot predict your way
    # out of not looking. Three aim-side fixes have now died on this
    # (`kick_deflect_*`, `two_stage`/`lineup_lat`, and this), which is what
    # points at the head and the blind radius instead.
    spot_lead: float = 0.0
    lineup_range: float = 0.6      # a ball seen inside this is worth lining up on
    # …and the spot is re-planned from sightings down to this range (the
    # detector's own slant `range_est`), then walked blind.
    #
    # 0.35 IS THE LEVEL CAMERA'S BLIND RADIUS, measured: a floor ball inside
    # 0.37 m of ground distance — 0.35 of slant range — is not reported at
    # all (`scripts/probe_head_pitch.py`). So the constant is honest about
    # what a level head can do, and the gaze can do better: pitched by
    # `_gaze` the same camera holds the ball to 0.18 m at the shipped clamp
    # and 0.08 m at the joint stop.
    #
    # MEASURED OFF ANYWAY, twice, for the same reason. The first time: at
    # 0.2 m the bearing noise is centimetres, the foot choice flipped, and
    # the spot dithered for 8 s. Re-measured over 24 seeds x 300 s of 2v2
    # with the probe that can see the placement (`scripts/probe_kick_line.py`),
    # `refresh_min` 0.35 -> 0.20 does exactly what it promises to the PLAN —
    # ball ahead of the trunk 0.238 -> 0.184 m, side 0.141 -> 0.083,
    # spot-to-ball 0.285 -> 0.220, plan age 3.26 -> 2.11 s, whiffs 18% ->
    # 10% — and costs 58% of the touches: 191 kicks -> 80 (p < 1e-11). The
    # whiff RATE improving while the kick COUNT halves is the `two_stage`
    # shape again (AGENTS.md rule 6): in absolute terms it is 156 effective
    # kicks against 72. Still 0 of 80 on the sweet spot.
    #
    # CONFIRMED ON 24 FRESH SEEDS, and this is the only arm of the whole
    # head/blindness investigation that survived one. Pooled over 48 paired
    # seeds, 150 kicks against the baseline's 360:
    #     ball ahead of the trunk   0.244 -> 0.182 m   (p = 7e-11)
    #     spot-to-ball              0.297 -> 0.215 m   (p = 1e-12)
    #     plan age                  3.32  -> 2.30 s    (p = 2e-19)
    #     ball drift since the plan 0.224 -> 0.151 m   (p = 7e-8)
    #     near the sweet spot       1/360 -> 9/150     (p = 0.0001)
    #     ON the sweet spot         0/360 -> 1/150     (p = 0.29)
    # and on the play ledger (24 seeds of 2v2) possession 21.6 -> 26.5 s/min
    # (p = 0.0003, better on 19 of 24 seeds) with goals 39 -> 47 (p = 0.34),
    # signed `ballProgress` FLAT (-0.050, p = 0.66) and the whiff rate flat
    # pooled (19.2% -> 17.3%).
    #
    # So it is a real fix to the STALENESS and it is not a win: the duck
    # keeps re-planning instead of swinging, which is possession bought with
    # touches, and the ball ends up no further forward. Ships at 0.35, with
    # the numbers, because the next person to reach for the blind radius
    # should start from here and not from the head.
    #
    # RE-MEASURED 2026-09-10 in the kick gym (roadmap 12aj), on the calibrated
    # camera, the 12ac detector gates and the weight-12 kicks: 0.35 -> 0.20 is
    # no longer a touch-for-precision trade - whiff 9 -> 19 % (p < 0.001,
    # worse on 10 of 12 seeds), swings 371 -> 250, far swings 11 -> 28 %,
    # connected travel 0.86 -> 0.54 m. The spot re-planned on centimetre
    # bearing noise dithers; the ball moves anyway during the blind settle.
    refresh_min: float = 0.35
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
    # AND READ ITS ADVANCE-PER-KICK WITH THE ADVANCE ITSELF. 0.163 against
    # the shipped 0.091 is a RATIO whose denominator is the thing this
    # arm changes. Re-measured over 12 paired seeds x 300 s (seeds
    # 100-111): shipped 87 kicks / advance +0.82 m/min, two_stage 35
    # kicks / +0.76, the `lineup_lat` variant 44 kicks / +0.71 - the
    # advance is FLAT to under 1 sigma across arms whose kick counts
    # differ by 5.8 - so advance-per-kick here is 1/kicks, and any change
    # that kicks LESS scores higher on it while moving the ball no
    # further. Measured directly instead, the ball's travel in the 2 s
    # after the swing: 17.7 +- 3.9 cm a kick shipped (34 kicks), 14.4 +-
    # 3.4 two-stage (17), 13.6 +- 3.0 with the faster line-up (20). The
    # placed kick is not worth more. Traced, it is not placed either: the
    # swing fires with the ball a median 21 cm (shipped) / 25 cm
    # (two_stage) ahead of the trunk where the sweet spot is 6-10, in
    # 3 of 17 two-stage kicks and 7 of 34 shipped ones inside a generous
    # box round it - the spot is planned at `refresh_min` or further and
    # the ball moves 20-28 cm during the attempt, so what scatters the
    # shot is the plan going stale, which a LONGER line-up makes worse.
    # The cost in the same paired block: goals 36 -> 21 (-2.3 sigma,
    # better on 11 of 12 seeds) and possession 18.6 -> 13.1 s/min
    # (-4.4 sigma).
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
    # Stage one's whole job is to put the duck ON the kick line, squared up,
    # behind the spot - so a duck that is already there has nothing to walk
    # back for. `lineup_lat` is how near the line counts as on it (stage
    # two's cross-track law closes the rest); 0 makes every line-up go via
    # the pre-spot, which is how the two-stage line-up was first measured.
    # Traced over 12 duck-runs of 300 s of 1v1 with `two_stage`: 27 s of
    # every 300 is the walk to the pre-spot and 4 s the square-up on it,
    # against 12 s of walk-in, and 55 attempts a run end in the back-off
    # below - the pre-spot BEHIND the duck. (That back-off turns away and
    # walks; `gait.back_up` would reach it without turning at all, and is
    # the obvious thing to measure here now that the walker is known to
    # reverse at 0.23 m/s. Untried.)
    # MEASURED at 0.06, `two_stage` on, against `two_stage` alone: the
    # line-up really does get faster - a kicking attempt 5.63 s -> 4.43 s
    # (its walk to the pre-spot 1.79 -> 1.19 s, the square-up on it 0.89 ->
    # 0.68), the pre-spot back-off 55 attempts -> 35 a run, 40% -> 42% of
    # attempts reaching the walk-in and 6.7% -> 8.7% of them firing - and
    # the play does not move: over 24 PAIRED seeds x 300 s of 1v1 (two
    # blocks of 12, the second fresh) kicks 72 -> 83, ballAdvance -0.036
    # +- 0.050, signed progress -0.105 +- 0.061, goals 45 -> 34 (sign
    # p = 0.24), falls 16 -> 11. Ships at 0: a real speed-up that buys
    # nothing the benchmark can see. Note the first 12 seeds promised +26%
    # kicks and a THIRD of the falls and the fresh 12 gave neither, which
    # is what 16 fall events and 35 kicks a block are worth.
    lineup_lat: float = 0.0

    # Traced: a re-plan with the ball already inside `backoff_range` puts
    # the pre-spot behind the duck, and the turn in place toward it is a
    # turn against the ball. Back off instead (turn away, walk clear - the
    # retreat manoeuvre) and line up again from further out. And a search
    # begun with the ball at the feet turns in place without ever seeing
    # it (a cold standing turn is exactly 0 rad/s): after `search_walk_after` with
    # no sighting, walk `search_walk_s` to change the view.
    backoff_range: float = 0.35
    search_walk_after: float = 0.0     # 0: off (measured with the two-stage line-up, see above)
    search_walk_s: float = 1.0
    # Trunk within this of the kicking spot: kick. 0.03 until 2026-09-10;
    # 0.05 SHIPS ON (roadmap 12ao). 12a's funnel priced a 3-6 cm spot error
    # at 8 % whiff, and the gym says the opposite, two seed blocks, both
    # populations: the ball at a board on the cove, connected kicks +21 %
    # pooled (129 -> 169, 146 -> 164) with whiff 38 -> 33 and 37 -> 33 %;
    # open play, whiff a null both blocks (11 -> 10, 10 -> 8), the sweet-spot
    # rate 16-17 -> 22-23 %, connected travel 0.82-0.89 -> 1.00-1.02 m, the
    # ball's advance per episode +25 %; falls flat. The reason is in the
    # line-up: a 3 cm target makes the servo creep and turn around the spot,
    # and that creeping nudges the ball off it; settling at 5 cm settles
    # sooner with the ball where the plan put it. 2v2 ledger, two blocks of
    # 24: possession null both, spread / crowd / depth null, kicks 305 -> 309,
    # goals 40 -> 45, own goals 9 = 9 - and two trends the wrong way at sizes
    # the battery cannot resolve, falls 6 -> 10 and back-kicks 24 -> 29 %
    # (p 0.14 on 600 events); the gym's per-swing falls are flat, so the
    # ledger's are not the swing's. Recorded on the knob; `lineup_tol=0.03`
    # is the old brain to the bit if the next ledger says otherwise.
    lineup_tol: float = 0.05
    lineup_s: float = 4.0          # give up a line-up after this long
    # THE SQUARE-UP AT A REACHED SPOT (roadmap 12ao, built 2026-09-10). On
    # the lab's boards a line-up that has been within 1 cm of its spot
    # still times out 5 cm from it, 64 deg off heading, with the wall 29 cm
    # away and the bumper silent (probe_board_states on the coved gym: 41
    # of 76 line-ups): outside `lineup_tol` the servo walks AT the spot,
    # whose bearing flips sign at close range, so it creeps and turns
    # toward the spot instead of the heading and is never both on it and
    # squared before `lineup_s`. Inside this distance of a kick spot the
    # order changes: turn in place to the heading first, then close the
    # last centimetres straight along it with the two-stage's lateral law
    # (`k_lat` / `k_head`, capped at `approach_wz`). 0 = off (the servo to
    # the spot as before).
    #
    # BUILT, MEASURED OFF in three cuts on the lab's boards (12ao, 40
    # episodes each): turn then walk straight (2 swings, 76 timeouts - after
    # a turn in place the spot is beside the duck), turn then a proportional
    # holonomic close (4 / 74 - a 0.1 m/s ask moves the walker nothing), turn
    # then a fixed 0.25 m/s vector at the spot (14 / 59 - squared to 24 deg
    # and still 5-7 cm off). Within 8 cm no command law here puts the trunk
    # within 3 cm of a point; the walker's positioning is the floor. The
    # crab it plumbed (the twist's lateral component) stays wired and tested.
    lineup_square: float = 0.0
    square_speed: float = 0.25     # the close's speed (m/s) along the body-frame vector to the spot
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
    # What to do when the goal is FURTHER round the ball than `aim_max`, i.e.
    # when kicking at it means walking round to the far side. Three answers,
    # and the shipped one is the reason 30 of 53 kicks in the Track 4 baseline
    # sent the ball back toward the kicker's own goal:
    #   "los"    give up on the goal and kick along the line of sight — which
    #            is straight at our own goal whenever the duck reached the
    #            ball from the goal side, and the support geometry puts it
    #            there (a supporter stands `support_back` goal-side of the
    #            ball and walks in from there when it becomes the attacker).
    #   "clamp"  kick at the edge of the cone on the goal's side: the same
    #            walk-round `aim_max` already allows, and never worse than
    #            `aim_max` off the best available line.
    #   "goal"   always at the goal, whatever the walk-round costs.
    #
    # MEASURED, 24 paired seeds x 300 s of 2v2 and then 24 FRESH ones
    # (docs/roadmap.md Track 4.3.1). `clamp` ships: back-kicks 211 of 419
    # (50%) -> 109 of 324 (34%) pooled over the 48, p < 0.0001, and it
    # replicates on its own in each block (p = 0.011 then p = 0.0001) —
    # while goals, possession, advance, signed progress and crowd are all
    # flat over the 48. It costs 23% of the touches (419 -> 324): aiming
    # better means walking further round. Falls do not resolve pooled
    # (+0.29, p = 0.37) though the fresh block alone looked bad (p = 0.052),
    # which is what a block on its own is worth.
    # `goal` ships OFF with its numbers: it aims best (27% back, p = 0.0003)
    # and plays worst — half the kicks and signed progress -0.26 (p = 0.012,
    # worse on 17 of 24), the ball ending up nearer the ducks' own goals,
    # because a duck arcing round the ball is in possession the whole way
    # and shoves it backwards as it goes. That re-earns the first form's
    # verdict (4 kicks and 2 falls a run for 1.0 goals against 1.75 for the
    # cone rule) with an instrument that can see the mechanism.
    aim_mode: str = "clamp"
    # CHOOSE THE KICK BY SIMULATING ITS OUTCOMES (brain/kickselect.py;
    # roadmap Track 4 s6 A.3, after Mellmann et al., RoboCup 2016). With this
    # on, `_plan` does not take the clamp's line as given: it lays a fan of
    # candidate lines inside the same `aim_max` window, rolls each out
    # `kick_select_n` times under the measured kick model (kick_speed,
    # ball_decel, the per-foot exit angles and their scatter), throws away
    # any line with a sample in our own net, and takes the one most likely
    # to score, ties broken by a potential field over the pitch. The clamp's
    # own line is always one of the candidates, so with nothing to choose
    # between them this reproduces the clamp.
    #
    # SHIPS ON (2026-09-07). Measured with scripts/probe_kick_line.py, 24
    # discovery seeds + 24 fresh, both arms of each block forked on one tree
    # state, scoring every kick that moved the ball by the line it actually
    # travelled (proportions over kick events, the power table's rule):
    #
    #                                 discovery        fresh          pooled 48
    #   aimed AWAY from their goal    25% -> 5%        39% -> 20%     31% -> 13%   z=-2.19 p=0.029
    #   on a line through THEIR mouth  9% -> 18%        4% -> 24%      7% -> 21%   z=+2.05 p=0.040
    #   on a line through OUR mouth    6% -> 0%         4% -> 4%       5% -> 2%    p=0.39 (too rare)
    #   whiffs                        52% -> 66%       50% -> 55%     51% -> 61%   p=0.14
    #   effective kicks a seed                                        -0.17        p=0.41
    #   back-kicks a seed, paired                                     -0.23        p=0.016
    #
    # Same direction on both blocks for the two numbers it was built to
    # move, no significant cost, the whiff gap shrinking on fresh seeds.
    # The second brain-tier change to ship on a fresh-seed confirmation
    # after the aim clamp, and it explains why kicksBack stalled at 34%
    # after that clamp: see `_select_kick_line` - the planner's foot rule
    # let the exit angle bend every clamped kick back toward the ball's
    # own heading. `kick_select=0` is the pre-2026-09-07 brain.
    kick_select: bool = True
    kick_select_n: int = 30
    kick_select_fan: float = 0.35     # rad between candidate lines across the aim window
    kick_select_dir_sd: float = 0.6   # rad of scatter about the exit line (measured 33-49 deg, 4b)
    kick_select_v_sd: float = 0.3     # m/s of scatter about kick_speed
    # The own-goal tolerance. Mellmann refused any line with ONE own-goal
    # sample, on kicks repeatable to a few degrees; ours scatter 35 deg, so
    # that rule refuses everything near our own mouth and the selector would
    # only ever fall back to the clamp. Measured over the fan (200 samples
    # a line, this floor's model), the share of kicks that end in OUR net:
    #
    #   ball 0.4 m from our mouth, facing it:  +-60 deg off the line of sight  0.07-0.09
    #                                          +-40  0.20-0.27   +-20  0.44-0.51   straight  0.66
    #   ball 1.3 m out, facing it:             +-60  0.02-0.04   +-40  0.03-0.07   straight  0.28
    #   mid-pitch or nearer, facing theirs:    every line <= 0.01
    #
    # 0.10 admits the edge lines the clamp itself would take and refuses
    # everything that puts a fifth or more of kicks in our own net.
    kick_select_t_own: float = 0.10
    # ...and what to do when NOTHING passes that filter. Today `select`
    # returns None and `_plan` keeps the clamp's line - a line the own-goal
    # filter never looked at. That happens only near our own mouth, which is
    # precisely where it matters: the table above says a ball 0.4 m off our
    # own line puts 44-51% of kicks in our net at +-20 deg and 66% straight
    # on. With this the fan's LEAST BAD line is taken instead - the minimum
    # of a set the clamp's own line belongs to, so it is never worse than
    # the fallback it replaces.
    kick_select_safest: bool = False
    # The kick's own WHIFF rate in the model (a whiffed sample leaves the
    # ball where it is): 50-61% measured on this floor, 18-23% on the old
    # one. Without it the roll-out assumes every swing connects and rates
    # the kick against a push that always does. And the PUSH as a selector
    # action (A.4): a walk through the ball, no settle, no exit angle, a
    # 0.6-0.7 m roll with a 30 deg spread (benched; kickselect.push_model).
    # Push-only measured +3.2 s/min possession and +0.07 progress against
    # the kick, and +0.21 own goals a run because it has no aim; offering
    # it to the selector gives it the aim.
    #
    # MEASURED 2026-09-07 (probe_search, 24 discovery + 24 fresh seeds,
    # every arm forked with shipped on one tree state; roadmap A.4):
    #   whiff term alone (p_whiff 0.5)      possession 14.2 -> 12.9, progress +0.016: nothing
    #   push under Mellmann's rule           the shipped brain again (the push is never chosen)
    #   push FIRST unless a kick can shoot   pooled 48: progress +0.068 p=0.013 (30/48 better),
    #     (p_whiff 0.5, push 1, shoot 0.3)   possession +1.97 s/min p=0.020 (31/48), falls / crowd /
    #                                        spread flat, own goals +0.10 a run p=0.13 (2 -> 7; push-
    #                                        only was 2 -> 12), kicks 2.8 -> 0.5 a run
    #   3v3 with roles (defender/mid/striker)  possession +3.68 p<0.001 (18/24), progress +0.074
    #                                        p=0.002 (19/24), own goals 0 -> 1; and the cost a
    #                                        formation shows: crowd 0.19 -> 0.30, spread 1.51 ->
    #                                        1.23 m, depth 0.57 -> 0.77 m, all p<0.001 - a defender
    #                                        that gets the ball walks it up-pitch (roadmap D.2)
    # Three blocks in agreement on the ball, no significant cost in 2v2, a
    # shape cost in 3v3. NOT SHIPPED for a physical reason: the push's whole
    # worth is the 0.64 m a walked ball rolls on the parallel session's
    # uncommitted floor - on the old floor it rolled to the boards like a
    # kick. When that floor is committed, one fresh block on it; if it
    # agrees, flip p_whiff to 0.5 and push to True.
    #
    # THAT BLOCK WAS RUN AND IT DISAGREED - the knob stays off for good
    # (roadmap Track 4 item 13a, 2026-09-08). Every number above was
    # measured on a pitch that was 85% a STATIONARY ball; the ball-out
    # rule (item 11b) took dead time 247 -> 176 s and kicks 2.9 -> 7.8 a
    # run, and on a pitch where the ball travels, push-first reverses on
    # both blocks (3v3 with roles, --ball-out-s 5, 24 + 24 seeds paired):
    # dead ball +35.5 s a run (p<0.001, worse on 45 of 48), kicks 8.60 ->
    # 0.31 a run, ballAdvance -0.336 (p<0.001), ballProgress -0.168
    # (p=0.019), the ball carried a kick 0.99 -> 0.02 m, crowd +0.068
    # (p<0.001). Possession still reads +4.40 s/min (p<0.001) and that is
    # the whole lesson: the pusher stands ON the ball by construction, so
    # walking it 0.64 m books possession while the ball goes nowhere -
    # playbook rule 5, "ask what would inflate your metric". On the dead
    # pitch possession was the only instrument that could move, which is
    # why this measured as a win three times. Rendered both arms and
    # looked (record-world pitch-3v3, seed 3, 90 s): shipped, the ball
    # crosses the pitch and there are six kicks; push-first, it crawls
    # 1.15 m inside a six-duck scrum and there are none.
    kick_select_p_whiff: float = 0.0
    kick_select_push: bool = False
    # With the push on offer: prefer a SAFE push unless some kick scores in
    # at least this share of its samples (0 = Mellmann's rule, scoring
    # first - measured to choose the push almost never, because on a 3 m
    # pitch a kick has some scoring chance nearly everywhere, and the arm
    # came back as the shipped brain: kicks 2.79 -> 2.12 a run, possession
    # 14.2 -> 14.0, progress 0.046 -> 0.024). What the push is worth is
    # tempo, which a one-shot roll-out cannot see; this is the knob that
    # lets the measured push-only result (+3.2 s/min possession, +0.07
    # progress) keep its aim.
    kick_select_shoot: float = 0.3
    # The back line CLEARS rather than carries (roadmap Track 4 s6 D.2's
    # first job): with the push on offer, a defender or keeper is not
    # offered it - the selector picks its kick line as usual, the ball
    # leaves the defensive third, and the duck returns to its post. Measured
    # reason: under push-first in 3v3 a defender that got the ball walked it
    # up the pitch and left its post (depth 0.57 -> 0.77 m, spread 1.51 ->
    # 1.23 m, both p<0.001), because a post says where to stand WITHOUT the
    # ball and nothing said what to do with it. Only means anything with
    # `kick_select_push`; the shipped brain is untouched by it.
    #
    # MEASURED, and it is not the fix (3v3 with roles, push-first with and
    # without, 24 seeds, one tree state): depth 0.769 -> 0.723 (p=0.24),
    # spread 1.23 -> 1.31 (p=0.10), crowd flat - a quarter of the depth and
    # a third of the spread back, neither resolving - while the defender's
    # kicks (+0.71 a run, p=0.006) bring back-kicks (0 -> 0.21 a run,
    # p=0.045) and own goals 1 -> 4 (p=0.17): a clearing kick from our own
    # third under the exit-angle geometry (4b) goes the wrong way some of
    # the time. The carrying defender was a minor part of the shape cost; a
    # team compresses around a WALKED ball because every post is laid out
    # relative to the ball. Ships off; the lever is a supporter position
    # that anticipates the carrier, not a rule for the carrier.
    # And now INERT: it gates only the push offer above, and push-first is
    # measured off for good (item 13a). Not re-measured on the ball-out
    # floor because there is nothing left for it to gate.
    defender_clears: bool = False
    # THE GAME STATE (roadmap Track 4 s6 B.3): with `kickoff_wait` on, after
    # a goal the side that SCORED stands off the restart - every duck of it
    # is a supporter, its post clipped into its own half and out of a
    # centre circle of `kickoff_circle` - until the ball leaves the spot
    # (my own sighting, else the board's) or the World's window runs out;
    # the side that conceded plays. The World is the GameController
    # (`World.kickoff_team`, `game_state`); the board carries its message
    # (`Team.waits`). The first kickoff is contested, as it always was.
    # MEASURED (2026-09-07, 24 seeds x 300 s, rule off and on forked on one
    # package copy): it fires and it costs nothing. 1v1: 12 restarts a
    # side, the scorer waits 1.7 s a duck a run, possession 8.62 -> 8.58
    # s/min (p=0.94), goals 12 -> 12, own goals 3 -> 2, back-kicks 1.17 ->
    # 0.96 a run (p=0.16). 3v3 with roles: 3 -> 2 restarts in 24 runs,
    # 0.3 s of waiting, 22 of 24 seeds bit-identical, every ledger number
    # flat (p > 0.24). The ball leaves the spot within a few seconds of
    # the restart, so the wait is short; what it buys is a game whose
    # restart is the conceding side's, as every league's is. Ships ON.
    kickoff_wait: bool = True
    kickoff_circle: float = 0.3
    # PASSING (roadmap Track 4 s6 D.1). With this on, every live teammate
    # the board places at least `pass_min_ahead` metres UP-PITCH of the
    # ball adds a candidate line straight at it (both feet), and every
    # candidate - passes, shots, the push - is valued with a bonus of
    # `pass_bonus` (potential units; the field spans about -1..+2) for each
    # sample that stops within `pass_reach` of ANY teammate. So a pass is
    # chosen by the same rule as everything else: never into our own net,
    # a shot first when one can score, else the line with the best expected
    # place for the ball - and a ball at a teammate's feet is a better
    # place than the same spot with nobody there. The teammates' positions
    # are the board's, in the shared frame (localised when odometry drifts).
    #
    # MEASURED OFF 2026-09-07 (24 seeds, one tree state; roadmap D.1): kicks
    # received within 0.4 m of a mate 1 of 22 -> 2 of 21 (p=0.52), every
    # ledger number flat. Two reasons, both in the data: a teammate is
    # up-pitch of the ball on 26% of swings (the striker IS the duck on the
    # ball; the defender holds behind), and a 1.4 m/s kick with 35 deg of
    # scatter rolls 3.3 m to the boards - it cannot deliver to a point a
    # metre away; only 6 of 61 verdicts had any received sample. The
    # passing instrument on this pitch is the PUSH (0.64 m, 30 deg), which
    # is waiting on the floor with kick_select_push. Re-run WITH a receiver
    # (3v3 with roles, push-first with and without this): possession -0.07
    # (p=0.94), progress -0.005 (p=0.88), spread +0.11 (p=0.011) - nothing
    # on the ball. The push-first selector already carries it; the push IS
    # the pass. Stays off.
    kick_select_pass: bool = False
    pass_min_ahead: float = 0.3
    pass_reach: float = 0.4
    pass_bonus: float = 0.6
    # OPPONENTS IN THE ROLL-OUT (roadmap Track 4 s6 C.4, the duel's first
    # half): with `kick_select_opps` on, every duck track the board does not
    # own (not a teammate by position, not our colour when the colour sense
    # is on) is an obstacle in the selector's roll-out - a sample whose path
    # passes within `kick_select_obs_r` of one stops at its feet, BLOCKED,
    # and is valued there less BLOCK_COST. So a line through an opponent
    # scores as what it is, and the selector turns the kick (or the push)
    # away from the body in the way.
    # MEASURED OFF (3v3 with roles, discovery block 0-23 then fresh block
    # 100-123, controls forked with each): the discovery block's progress
    # gain (+0.068, p=0.027) did not replicate (fresh -0.050, p=0.13;
    # pooled 48 +0.009, p=0.70); pooled, spread -0.086 (p=0.054), depth
    # +0.067 (p=0.026), falls 5 -> 12, and the kicks it takes - more of
    # them, 2.0 -> 2.5 a run - carry LESS (0.225 -> 0.159 m a kick):
    # a line that misses the body is a shorter, wider line. Ships off.
    # RE-MEASURED on the ball-out floor and unchanged (roadmap item 13d,
    # 3v3 with roles, 24 seeds, --ball-out-s 5): dead ball +9.6 s a run
    # (p=0.096), kicks -0.92 (p=0.079), ballAdvance -0.065 (p=0.52),
    # possession -0.70 (p=0.48), spread +0.112 (p=0.017), search +2.6 s a
    # duck (p=0.021). The flowing pitch did not rescue it. Stays off.
    kick_select_opps: bool = False
    kick_select_obs_r: float = 0.15
    # THE LEARNED SCORER (roadmap E.2): a path to the weights fitted by
    # `scripts/kick_choice_data.py` on the kick gym's own realised
    # outcomes. "" (the default) is the shipped roll-out ranking, bit for
    # bit — `kickselect.select` does not look at its `chooser` argument
    # when it is None. When set, the candidate fan, the roll-outs and the
    # own-goal veto are all unchanged and only the choice among the SAFE
    # candidates is the model's (brain/kickchoice.py).
    kick_select_learned: str = ""
    # THE SHARED BALL (roadmap Track 4 s6 C.3): with `fuse_ball` on, the
    # team board's ball is the inverse-variance mean of every live
    # sighting, each weighed by its sender's sigma (C.1) grown by its age
    # (`Team.fuse`; brain_kwargs sets it on the board from this knob). The
    # estimate measured closer (odometry probe, 8 seeds x 300 s of 2v2:
    # when two ducks saw the ball the fused one was nearer the truth 70%
    # of the time at ideal odometry, 61% at datasheet; 95th percentile
    # 0.253 -> 0.236 m and 0.883 -> 0.685 m; the median unchanged, since
    # three samples in four have one sighting). THE PLAY MINDS (3v3 with
    # roles, 24 seeds, freshest v fused forked together): goals at both
    # mouths 2 -> 10 (p=0.004), own goals 0 -> 6 (p=0.006, six seeds to
    # none), crowd +0.04, spread -0.09, falls 3 -> 8. A more accurate
    # point, a worse game: the fusion keeps claims up to 3 x stale_s, so
    # when the ball MOVES the board's ball lags toward where teammates
    # last saw it, and a supporter walks to a point it has left. Off.
    # RE-MEASURED on the ball-out floor (roadmap item 13c, 3v3 with roles,
    # 24 seeds, --ball-out-s 5) and the catastrophe does not reproduce -
    # nor does anything else: dead ball +7.6 s (p=0.20), kicks -0.83
    # (p=0.33), ballAdvance -0.138 (p=0.19), ballProgress -0.105 (p=0.34),
    # possession +1.64 (p=0.14), shape flat, kicksBack 23% -> 22% of kick
    # events (p=0.90). Own goals 4 -> 0 (p=0.032) is a coin at 24 seeds on
    # a metric that needs 347. Stays off, but the honest statement is now
    # "no measured effect", not "worse in play".
    fuse_ball: bool = False
    fuse_window: float = 0.5         # s: only claims this close to the freshest are fused (Team.fuse_window, measured)
    push_roll: float = 0.64          # m a walked-into ball rolls on this floor (benched 0.56-0.71)
    push_dir_sd: float = 0.5         # rad of spread across the side offsets the walk meets the ball at
    # THE FIELD (roadmap Track 4 s6 D.2, brain/field.py): with `support_field`
    # on, a striker, a midfielder or a plain supporter stands at the minimum
    # of a potential field instead of at its post - `ahead` of the ball along
    # the carrier's lane (the role's own number), OUT of that lane
    # (`field_lane` half-width), `field_wide` beside it, away from
    # teammates and from opponents that stand between it and the ball, and
    # pulled `field_goal_pull` x the pitch potential toward the goal. The
    # zone, the boards' margin and the attacker's room stay hard. A
    # defender's and a keeper's posts measured well and are untouched.
    # Why: the 3v3 push-first measurement - the team compresses around a
    # WALKED ball because every post is a distance from the ball.
    #
    # MEASURED (2026-09-07, 3v3 with roles, 24 seeds, four arms forked on
    # one package copy). Under push-first it takes back about half of the
    # shape cost with nothing lost on the ball: crowd 0.304 -> 0.251
    # (p=0.023, better on 17 of 24; the kick baseline is 0.186), spread
    # 1.233 -> 1.374 (p=0.003), depth 0.769 -> 0.708 (p=0.06), possession
    # 17.5 -> 17.4 and progress 0.102 -> 0.111 (flat). Under the SHIPPED
    # kicks it is the wrong shape: crowd 0.186 -> 0.226 (p=0.039) and depth
    # 0.571 -> 0.650 (p=0.003) against a possession hint of +1.8 s/min
    # (p=0.07) - a kicked ball flies 3 m, and a midfielder held level with
    # it and 0.5 m wide is nearer a flying ball than its post between the
    # ball and the centre spot. Ships OFF; it is part of the push-first
    # configuration (kick_select_push=1,kick_select_p_whiff=0.5,
    # support_field=1) for when the floor is committed. `field_mid_ahead`
    # is the midfielder's `ahead` (0 = level with the ball; negative =
    # behind it), the one number the kick case turns on.
    #
    # THE MIDFIELDER BEHIND THE BALL (field_mid_ahead=-0.5), measured
    # across rosters: 3v3 with roles, possession +3.0 s/min (p=0.001) on
    # the discovery block and +1.3 (p=0.26) on the fresh one, pooled 48
    # +2.1 (+15%, p=0.004), shape flat, back line +0.08 m (the OTHER
    # side's defender drawn off its post); 2v2 with roles (defender +
    # striker), ball progress 0.148 -> 0.061 (p=0.002, worse on 18 of
    # 24) with more kicks that go nowhere; the role-less roster null,
    # leaning the wrong way. So the knob ships OFF here and brain_kwargs
    # turns it on for a roster WITH A MIDFIELDER - the one it measured a
    # win on - unless the command line names it.
    #
    # RE-MEASURED ON THE BALL-OUT FLOOR (roadmap item 13b, 2026-09-08):
    # it survives, on a different number than the one it was sold on. The
    # arm is the knob turned OFF against the shipped ON (3v3 with roles,
    # --ball-out-s 5, discovery 0-23 then fresh 100-123, paired). What
    # replicated, same direction on both blocks: ballAdvance -0.199 with
    # the field off (p=0.003 pooled 48) and the ball carried a kick
    # -0.424 m (p=0.002) - the field's kicks move the ball further. What
    # did NOT replicate and is withdrawn: the discovery block's dead-ball
    # (-11.6 s, p=0.035) and kick-count (+1.21, p=0.032) gains, both flat
    # on the fresh block (+1.8 p=0.74, +0.04 p=0.96). Possession is flat
    # on both. The cost is falls, 0.17 -> 0.38 a run with the field on
    # (p=0.043 over 48 runs): the supporter stands nearer the play.
    # And `field_mid_ahead` -0.5 vs 0 is a NULL on the new floor (24
    # seeds: advance -0.134 p=0.14, dead ball +4.7 p=0.51, kicks -0.13
    # p=0.88, possession +0.14 p=0.90, shape flat) - the 2026-09-07
    # possession win for holding the midfielder behind the ball does not
    # reproduce once the ball moves. -0.5 is kept because nothing argues
    # against it, not because it earns its keep.
    support_field: bool = False
    field_lane: float = 0.2
    field_wide: float = 0.5
    field_goal_pull: float = 0.3
    field_hyst: float = 0.1
    field_mid_ahead: float = 0.0
    # The ROLE-LESS supporter (eval-pitch's roster) measured null under the
    # field (2v2, 24 seeds: advance -0.04 p=0.19, ball in view -1.9%
    # p=0.09, nothing else moved), so it keeps its post unless this says
    # otherwise; `support_field` is for the roles it was measured on.
    field_plain: bool = False
    # THE SUPPORTER LOOKS AT THE BALL (2026-09-07, the owner watching /sim:
    # "their necks are still straight"). The gaze law only ever ran while a
    # duck was chasing or lining up; a supporter FACES the ball with its
    # head level, and a duck is a supporter for ~60% of a match - which is
    # what the straight necks are. With this on, a supporter (and a duck
    # standing off a kickoff) that has a fresh sighting inside `head_range`
    # and `gaze_bearing_max` gazes at it, standing or walking, head level
    # again for a turn in place (the walker cannot turn head-down). Off
    # until measured: the ledger and the ball-in-view fraction.
    support_gaze: bool = False
    # THE HEAD. `_gaze` is a law that puts a floor ball at range `rng` on the
    # camera's axis; `head_down` clamps the command it may ask for, and the
    # gaze is applied while WALKING at a ball inside `head_range`.
    #
    # MEASURED END TO END (`scripts/probe_head_pitch.py`: the command swept on
    # the shipped walker, and the blind radius read off the REAL `Detector` on
    # a real composed `World` at each pose). Three facts, none of which was
    # known when 0.6 was chosen:
    #
    # 1. WHERE THE COMMAND SATURATES. Standing, the head-pitch slot buys
    #    0.79 rad of camera depression per unit of command, linearly, until
    #    cmd 1.25 — where the `head_pitch` JOINT reaches its +1.571 MJCF
    #    limit and the camera stops at 1.17 rad (67°). Past that the command
    #    does nothing. So the limit is the JOINT, not the policy (the walker
    #    tracks the command all the way to it and stays upright), and 0.6
    #    (0.65 rad, 37°) was half of what is there.
    # 2. WHAT EACH POSE SEES. Nearest floor ball the detector still reports,
    #    standing, ground distance trunk→ball (and the far edge, which is the
    #    price — pitching down trades the horizon for the feet):
    #        level        0.37 m … 0.90+   ("the level camera loses a floor
    #                                        ball inside ~0.3 m" — measured
    #                                        at 0.37, or 0.35 of the slant
    #                                        `range_est` the brain compares
    #                                        `refresh_min` against)
    #        head 0.6     0.18 m … 0.77
    #        head 0.9     0.12 m … 0.36
    #        head 1.25    0.08 m … 0.21
    #    so the deep end is only worth asking for when the ball really is
    #    that close — which is exactly what `_gaze` decides, since the
    #    command it asks for is a function of the range.
    #
    #    AND THAT IS WHY RAISING `head_down` DOES NOTHING. Read off the
    #    brain that is running (3 seeds x 90 s of 2v2, 3656 gaze frames in
    #    `lineup`/`settle`), the gaze COMMAND is a median 0.259 and the
    #    camera depression it reaches a median 0.245 rad — 14°, with the
    #    90th percentile at 0.566. The clamp binds in under a tenth of the
    #    frames, because `_gaze` aims the axis AT the ball and most of a
    #    line-up happens at 0.3-0.6 m where that asks for a quarter of a
    #    radian. Raising the ceiling of a limit that is not being hit is
    #    not a change; measured, `head_down` 0.6 -> 1.0 moved the median
    #    depression only 0.245 -> 0.339 rad and no kick metric at all.
    # 3. THE NECK SLOT IS FREE AND THE HEAD SLOT IS NOT. `head_pose_cmd[0]`
    #    is `neck_pitch` and this brain never commanded it: `Chase.step`
    #    emitted `(0.0, gaze, 0.0, 0.0)`. Swept (walking at 0.30, 4 headings,
    #    steady over seconds 2-6), depression is additive and linear in the
    #    two slots — +head looks DOWN at 0.79 rad/unit standing, +neck looks
    #    UP, so a downward gaze is a NEGATIVE neck command, worth 0.43
    #    rad/unit — and the same depression costs completely different
    #    amounts of forward speed:
    #        59°  head +1.00 alone      -19.6 %      neck -0.30 head +0.60   +4.4 %
    #        69°  head +1.25 alone      -28.2 %      neck -0.60 head +0.60   -4.9 %
    #        72°  (not reachable)                    neck -0.40 head +0.80   -1.1 %
    #    `gaze_neck` is the fraction of the gaze command mirrored onto the
    #    neck slot, and `neck_gain` its rad-per-command, so `_gaze` divides
    #    the wanted depression by what the two slots together deliver. At 0
    #    it is bit-for-bit the old single-slot law.
    head_down: float = 0.6
    head_range: float = 0.9
    head_gain: float = 0.75        # camera rad per unit of head-pitch command (standing; measured 0.789)
    neck_gain: float = 0.43        # …and per unit of NECK command (measured 0.43, same sweep)
    # Fraction of the gaze routed to the neck slot (see 3 above). Ships at 0
    # because the gaze itself is off; if `gaze_still` is ever turned on it
    # must be 1, since the head slot alone raises the whiff rate 19.2% ->
    # 26.4% (p = 0.027, replicated) and the split does not.
    #
    # It also looks DEEPER for less command, because the neck's gain is much
    # closer to the head's while WALKING (0.78 against 0.93) than standing
    # (0.43 against 0.79), and the law above is calibrated standing. Read off
    # the running brain in `lineup`/`settle`: the shipped gaze reaches a
    # median 0.245 rad with a 90th percentile of 0.566; `head_down` 1.0
    # through the head slot reaches 0.339 / 0.637; the split at `head_down`
    # 0.64 reaches 0.309 / **0.812** on a median command of 0.199. The tail
    # is where a gaze earns its keep (it is the close-in frames), which is
    # why the split arm went blind for 0.14 s at the swing and the
    # head-slot arm for 1.20 s.
    gaze_neck: float = 0.0
    # The azimuth past which a line-up gaze is not attempted (rad). Ships at
    # `GAZE_MAX_BEARING` = 0.6 (34°), which refuses the gaze at exactly the
    # kick spot - the ball there is 37° off the nose - on the level-camera
    # reasoning corrected beside that constant. A pitched camera CAN see
    # that ball. Raise it toward `head_yaw_max` (1.4) to let the gaze try;
    # what it buys is measured with `scripts/probe_shot_gate.py` ("swings
    # it can see at all": 34% shipped, 65% with gaze_still+gaze_neck at the
    # shipped cap).
    gaze_bearing_max: float = GAZE_MAX_BEARING
    cam_level: float = 0.197
    cam_z: float = 0.21
    # Hold the gaze while the duck is STANDING STILL, instead of dropping it
    # the moment it stops. The gaze used to be gated on `vx > 0`, so the
    # settle in front of every swing was taken with the head level — and the
    # level camera cannot see a ball inside 0.37 m, which is precisely where
    # the ball is by then. Measured over 195 kicks (24 seeds x 300 s of 2v2,
    # `scripts/probe_gaze.py`): the head-pitch command at the swing is 0.000
    # at the median, the 3.5 s run-up is head-UP 55% of the time and standing
    # still 49%, the duck has not seen the ball for 1.48 s and 0.175 m of
    # walking when it fires, and 81% of the duck-steps spent with the ball
    # truly inside 0.40 m are steps in which the detector is reporting
    # nothing. That is the owner's "they walk around looking even when the
    # ball is under their feet", and "they keep scaling back up", as numbers.
    # STILL, NOT SLOW: a turn in place keeps the head level, because the
    # walker cannot turn in place with its head down (0.2 rad in 5 s against
    # 3.1 level — measured in tidy.py, and the reason the old gate existed).
    #
    # SHIPS OFF. It does exactly what it says and the kick does not care.
    # The mechanism, on the same 24 seeds x 300 s of 2v2 as the numbers
    # above (with `gaze_neck` = 1, `head_down` = 0.64): the run-up is
    # head-UP 55% -> 37% of the time, the duck has not seen the ball for
    # 1.48 s / 0.175 m at the swing -> **0.14 s / 0.006 m**, and kicks taken
    # having never seen the ball in the whole 3.5 s run-up fall from 16 of
    # 195 to 1 of 189. The blindness is real, it is a choice, and this
    # un-chooses it.
    #
    # And then it buys nothing. Over 48 PAIRED seeds (two blocks of 24, the
    # second fresh — `scripts/probe_kick_line.py`), against 360 baseline
    # kicks and 339 with the gaze held:
    #     on the sweet spot   0/360      ->  4/339 (1.2%, p = 0.055) — and
    #                                        all four are in the first block,
    #                                        none in the fresh one
    #     whiffed             69/360 19.2% ->  64/339 18.9%  (p = 1.00)
    #     ball ahead / spot-to-ball / plan age / drift: all flat
    # The play ledger over 24 seeds of 2v2 agrees: goals, falls, possession,
    # signed progress, spread, crowd and depth all flat, with `ballAdvance`
    # +0.115 (p = 0.048) while signed `ballProgress` is -0.003 (p = 0.98) —
    # churn, which is exactly what rule 5 says advance measures on its own.
    #
    # WORSE THROUGH THE HEAD SLOT ALONE, and this one replicates: at
    # `gaze_neck` = 0 the whiff rate goes 19.2% -> **26.4%** (84 of 318,
    # p = 0.027), in BOTH blocks (27.9%, 25.1%). With the neck carrying the
    # gaze the cost disappears (18.9%). That is the bench measurement
    # showing up in play — the head slot costs 12-28% of forward speed at
    # these depressions and the split costs about nothing — and it is the
    # reason `gaze_neck` exists at all.
    #
    # "WHY IT CANNOT WIN" used to sit here: that on the kick spot the ball is
    # 37° off the nose against a 31° horizontal half-field, unseeable at any
    # pitch. That was a level-camera rule and it is wrong for a pitched one
    # - see `GAZE_MAX_BEARING`. Re-measured 2026-09-06 on the one real
    # camera (`scripts/probe_shot_gate.py`, 24 seeds): with gaze_still +
    # gaze_neck the brain has a fresh ball estimate on 65% of swings against
    # 34% shipped, r to the true side offset 0.95 against 0.48, and the ball
    # at the swing moves onto the sweet spot - side 0.133 -> 0.06 m, on-spot
    # 1.7% -> ~20%. The held gaze DOES win on placement.
    #
    # And then the kick whiffs 82-85% (against 23%): benched, the shipped
    # kick skill whiffs 12 of 12 when the swing starts with the head joint
    # at +0.97 rad (cmd 0.6) and 0 of 12 level (+0.39), same ball, same
    # spot. The arena zeroes the kick's head COMMAND, not its joints, and
    # `ball_kick_*.onnx` were trained from a level head. So this ships OFF
    # for a different reason than before: it puts the ball where the kick
    # cannot swing. The fix is in the kick policy's training distribution,
    # or a settle that raises the head in its last ~0.3 s (not built).
    # Roadmap Track 4 item 7, the correction.
    #
    # ON since 2026-09-08 (roadmap item 12c): the kicks are the local ones
    # now, which do not whiff head-down, and the swing replay (12a) showed
    # the whiffs are swings at a ball that drifted out of reach while the
    # track went stale (median 1.8 s old at the swing, in 100% of the
    # far-ball swings). Held with `kick_ahead_max` the gaze is what gives
    # the gate its coverage: whiff 44 -> 31% (seeds 0-23) and 51 -> 41%
    # (100-123), connected kicks a run 3.6 -> 3.6 on the fresh block; the
    # 2v2 ledger over 24 seeds: possession +0.2 s/min (p 0.70), progress
    # +0.03 (p 0.59), goals 19 -> 25 (p 0.27), back-kicks 1.9 -> 1.25
    # (p 0.04), swings 7.8 -> 6.0 a run (p 0.01: the blind ones), falls
    # flat. Alone (no gate) it sees the ball on 54% of swings and changes
    # nothing (45% whiff): seeing is only worth what the gate does with it.
    gaze_still: bool = True
    # The gaze law reads its range as the SLANT it is (asin) rather than as
    # ground distance (atan2): deeper by ~7 deg at 0.27 m and ~20 deg at
    # 0.20 m. Recorded, not built, in 12ae because it moves the shipped
    # line-up gaze; built 2026-09-10 as a knob to A/B (roadmap 12ap).
    # MEASURED OFF: kick gym, two seed blocks, both populations. Open play
    # whiff 10 -> 7 % on the discovery block (better 9/12, null at MDE 4)
    # and 8 -> 8 % on the fresh one (5/12); the ball at a board on the cove
    # 33 -> 32 % and 33 -> 33 %; connected kicks and falls flat; the track at
    # the swing no fresher (1.43 -> 1.42 s). A 7-20 deg deeper walking gaze
    # changes nothing the kick can use: the ball leaves the frame under the
    # chin either way (12k).
    gaze_slant: bool = False
    # The settle that raises the head: with `gaze_still` the line-up gaze
    # puts the ball on the sweet spot and leaves the head pitched where the
    # kick skill cannot swing (benched: 12 of 12 whiffs from a head joint at
    # +0.97 rad, 0 of 12 level). For the LAST this many seconds of the
    # settle the gaze is dropped and the head commanded level, so the swing
    # starts from the pose the kicks were trained in. It costs the sighting
    # for that long - but the duck is standing and the ball is still, so
    # 0.3 s of a stationary ball is not 3 s of a rolling one. 0 = off (the
    # gaze holds through the swing, as measured off above). Only means
    # anything with `gaze_still`.
    #
    # BENCHED: 0.2 s of level command brings the head joint from +0.93 to
    # +0.45 and the whiff from 12/12 to 0/12; 0.3 s gives +0.41. IN PLAY on
    # the rolling-resistance floor (24 seeds, four arms on one tree): with
    # 0.3 the joint at the swing is +0.41 on all 42 kicks - the mechanism
    # works - and the gaze's whiff falls 76% -> 52% (44% with settle_s 0.6).
    # It still does not beat the shipped brain (34% whiff, 23% on-spot):
    # on that floor a stopped ball no longer drifts off the plan, so the
    # gaze's placement advantage is gone, and a residual whiff the head does
    # not explain remains (ball inside 10 cm, head level: shipped 0/14,
    # this 4/12 - the neck, or the posture after a head-down walk-in; not
    # measured). Ships at 0 with the lead known, for the day the kick skill
    # is retrained head-down. Roadmap Track 4 item 7.
    #
    # THE RESIDUAL IS THE NECK (measured after the above, neck and trunk
    # recorded at every swing): with this at 0.3 the head joint is level
    # (+0.41) but the neck is still low - +0.17 against the shipped +0.21,
    # lower quartile +0.14 - and pooled over both arms with the ball inside
    # 15 cm, a neck at +0.00..0.15 whiffs 83% and at +0.15..0.30 whiffs 43%
    # (corr -0.26); trunk pitch is flat (-0.004 vs -0.006) and is not it.
    # `neck_gain` (0.43) is what carries the gaze and it returns slower than
    # the head. So the kick skill needs BOTH neck_pitch and head_pitch near
    # HOME, which is what the upstream port randomises
    # (docs/patches/microduck_rl-kick-head-down.patch). A longer raise
    # (0.5 s) is the brain-side alternative, not measured.
    settle_head_level: float = 0.0
    # THE SETTLE LOOKS DOWN (roadmap 12d, built after 12aj, 2026-09-10). The
    # ball track is a median 1.54 s old at the swing because the line-up gaze
    # is clipped at `head_down` 0.6 with no neck: a 38 deg depression whose
    # floor window starts 0.19 m from the root (12k), and the ball on the
    # spot is 0.08 m out - the far balls that whiff (truth > 0.15 m, 11 % of
    # swings, 44 % whiff) sit at a median 0.177 m, just inside that blind
    # edge. `gaze_neck` while WALKING lost in play because the window
    # slides (0.12-0.36 m at half the neck) and the far edge is where the
    # ball is during the walk-in. The settle is different: the duck stands
    # on the spot, 12k measured zero falls standing at any pose, and the
    # weight-12 kicks connect from the neck-split pose (bench 0 % whiff).
    # So, in `settle` ONLY: route this fraction of the gaze to the neck and
    # clip the head at `settle_head_down` instead of `head_down`, so the
    # existing far-ball gate (`kick_ahead_max`) reads a sighting instead of
    # a belief that expired. 0 = off (the shipped gaze, to the bit).
    #
    # BUILT, MEASURED, SHIPS OFF (kick gym, seeds 0-11 x 40, paired; roadmap
    # 12ak). It delivers the sighting: the track at the swing 1.54 -> 0.14 s
    # old, fresh (<= 0.5 s) on 58 % of swings from 2 %, the belief present on
    # 62 % from 12 % and right to 2.4 cm. And whiff goes UP: 9 -> 13 % (null,
    # worse 8/12), because the deep neck at the swing costs the kick (sweet
    # spot 15 -> 10 %, connected travel 0.86 -> 0.67 m) and the far balls it
    # reveals sit at the gate's edge (truth 16.5 cm, belief 14.3). Levelling
    # the head for the last 0.2 s (`settle_head_level`) gets the spot rate
    # back (21 %) and loses the sighting (23 % fresh): 15 %, an effect the
    # wrong way. A 0.6 s settle with 0.15 s level: 13 %. The gate at 0.13:
    # 16 % on fewer swings. The far gate OFF with the look: 32 % - the gate
    # does its work once it can see. Neither the settle nor the head touches
    # the ball (scripts/probe_settle_contact.py: floor contacts only; the ball
    # rolls in from the walk-in 4.3 cm during the shipped settle, 2.1 with
    # the look). A fresh sighting the brain can only act on by declining and
    # re-laying is not worth its pose; the lever that would use it is a kick
    # that adapts to the ball (A.2 / 12h), not a gate.
    settle_gaze_neck: float = 0.0
    settle_head_down: float = 0.0
    # …and yaw the head at it too while standing. The pitch alone cannot
    # reach the endpoint: on the kick spot the ball is 0.08 m ahead and
    # 0.06 m to the kicking foot's side, which is 37° off the nose, and the
    # camera's horizontal HALF-field is 31°. A yaw would cover it (the walker
    # tracks a head-yaw command to 1.42 rad, and standing there is no forward
    # speed to lose) — but the ToF sits on the HEAD, so a yawed head points
    # the bumper sideways, which is exactly how `look_aim` was measured off
    # ("the brain stops for what it then sees").
    #
    # TWO THINGS TO KNOW BEFORE TRYING IT.
    #
    # 1. It is a DEAD KNOB ALONE, the same way `head_yaw_when="always"` was
    #    dead without `predict_s` (roadmap 4e). The yaw is only ever non-zero
    #    inside the `gaze_still` branch, and `_gaze_range`, the only thing
    #    this widens, is only called from there. Measured, not read: three
    #    seeds x 40 s of 2v2, 24 000 duck-ticks, 4 157 of them in
    #    lineup/settle where it would apply — **0 ticks differ** with it on.
    #    Turning it on with `gaze_still` off measures nothing. The arm is
    #    `gaze_still=1, gaze_neck=1, gaze_yaw=1`: held, carried on the neck
    #    so it does not cost forward speed, and able to reach the endpoint.
    #
    # 2. The ToF risk it was parked on now has a fix. `yaw_clear` gates this
    #    yaw too, on the same signal, and head-yaw tracking measured FREE
    #    once gated (roadmap 4e). That is what makes the arm worth running:
    #    `gaze_still` was judged with the hazard still in it.
    #
    # MEASURED OFF, 2026-09-06 — no longer "unknown". Three arms x 24 seeds
    # x 300 s of 2v2 on `scripts/probe_kick_line.py` (`runs/gazeyaw/`),
    # against the shipped brain's 178 kicks / 23.0% whiffs / 1.7% on-spot:
    #
    #   gaze_still+gaze_neck      204 kicks  23.5% whiff (p=0.91)  0.5% on-spot
    #   +gaze_yaw                 211 kicks  19.9% whiff (p=0.45)  0.5% on-spot
    #
    # It reaches the endpoint and it does not help. Nothing it was meant to
    # fix moves: on-spot does not rise, spot-to-ball gets WORSE (0.285 ->
    # 0.307 m), plan age is flat (3.03 -> 2.96 s), and paired per seed the
    # absolute aim error is flat too (p=0.93). What it does move is the
    # SYSTEMATIC aim: +12.4 deg, 95% CI [+6.2, +18.6], against a shipped
    # brain whose interval spans zero.
    #
    # WHY, and this is the useful part. The bias looks mechanical — pooled
    # over 462 kicks the aim error tracks the head yaw held at the swing
    # (r = +0.34, p = 5e-15; past +0.40 rad the mean error is +40.7 deg).
    # It is not. Control for where the ball actually was and the head-yaw
    # term collapses to +0.03 deg per deg (p = 0.66) while the ball's SIDE
    # offset carries everything: **+1.90 deg of aim error per cm**, t = 21,
    # R^2 = 0.56. The head yaw was a proxy for a ball off to the side.
    # `gaze_yaw` widens the bearing the gaze will accept, so it lets a duck
    # swing at balls it should not have swung at. Seeing the endpoint was
    # never the problem; standing in the right place is.
    gaze_yaw: bool = False
    # After a kick the ball is ahead and low: stand and look down `look_s`
    # before searching (measured: a 9 s search spin with the ball 0.17 m
    # ahead). A search dips the head every `search_dip_every`.
    look_s: float = 0.8
    look_range: float = 0.3
    # The look that LOOKS (roadmap item 12i, 2026-09-08): with `look_sweep`
    # > 0 rad the standing look after a kick gazes at `look_sweep_range`
    # (a kicked ball is 0.5-1.3 m out, not the 0.3 m above, which is a
    # whiffed ball's) and sweeps the head yaw +- that amplitude around the
    # predicted exit line over `look_s`, so a ball that left on the other
    # side of the line (40-48% do; scatter 40-70 deg a foot) is seen before
    # the hunt walks away from it. The ToF-sideways objection to `look_aim`
    # was a yawed head while WALKING; here the duck stands. 0 = off.
    # MEASURED (with gaze_still + kick_ahead_max, 24 seeds x 300 s of 2v2 a
    # block): connected kicks seen again within 2 s 37 -> 46% (seeds 0-23),
    # whiff 31 -> 26% there and 41 -> 38% on the fresh block (100-123),
    # connected kicks a run 3.8 -> 4.2 / 3.6 -> 3.8. Ships at 0.8 rad.
    look_sweep: float = 0.8
    look_sweep_range: float = 1.0
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
    # Where the ball ACTUALLY leaves, relative to the body heading — measured
    # in play rather than on a bench: 237 kicks over 24 seeds x 300 s of 1v1
    # and 2v2 (`scripts/probe_kick_line.py`), taking each kick's line from the
    # ball's travel over the next CARRY_S.
    #
    #     left foot    +23.6 deg   95% CI [+13.1, +34.1]   (bench said +21.6)
    #     right foot   -28.7 deg   95% CI [-33.8, -23.6]   (bench said -11.0)
    #
    # The bench was right about the left foot and 18 deg wrong about the
    # right, which is worth knowing: the bench swept a ball across a STANDING
    # duck's foot at the sweet spot, and in play the ball is 2-3 cm off it.
    # These are the in-play numbers.
    kick_exit_left: float = math.radians(23.6)
    kick_exit_right: float = math.radians(-28.7)
    # After a kick, hunt along where the ball REALLY went (`u` plus the exit
    # angle above) instead of along the line the kick was aimed at, and
    # publish THAT line to the team board. Pure knowledge: it changes where
    # the duck looks and what it tells its teammates, never how it stands.
    #
    # MEASURED NEUTRAL, and shipping on anyway — with the reason stated so
    # nobody mistakes it for a win. 24 paired seeds x 300 s of 2v2 and then
    # 24 fresh ones, exit line against aim line, nothing resolves on either
    # block or pooled over the 48: goals +0.125 (p = 0.49), falls -0.083
    # (p = 0.77), possession -0.514 (p = 0.32), ballAdvance +0.023
    # (p = 0.48), signed progress -0.026 (p = 0.54), crowd -0.003 (p = 0.64),
    # back-kicks 34% -> 37% (p = 0.45). It is a real null and not a dead
    # path — the arms differ seed by seed (68 falls against 57 on one block,
    # 57 against 64 on the other), which is what rule 0 asks you to check.
    #
    # It ships on because the alternative is knowingly hunting along a line
    # the ball does not take: the hunt is a fallback that fires only when a
    # kicked ball is lost, and the search behind it finds the ball anyway,
    # so being right costs nothing and buys nothing measurable. No
    # performance claim is made for it.
    #
    # The other way of using the same measurement — rotating the STANCE so the
    # kick flies along `u` (`kick_deflect_*`) — is refuted, and this time with
    # the mechanism. Set to the measured values, over 24 paired seeds of 2v2:
    # goals 2.08 -> 1.38 (p = 0.045), ballAdvance 0.839 -> 0.655 (p = 0.023),
    # kicks 151 -> 78, and the aim error it was supposed to remove got WORSE,
    # 45.4 -> 67.3 deg mean absolute. The reason is in the map's own shape:
    # the deflection is a function of where the ball sits relative to the
    # foot (15 deg/cm near 2 cm, 4.5 deg/cm at 4-8 cm), so rotating the stance
    # moves the ball to a different part of that function and produces a
    # DIFFERENT deflection — the right foot's went from -27.3 to -48.6 deg off
    # the body. A fixed rotation cannot cancel an offset that its own rotation
    # changes. The lever stays what the map said first: line-up precision.
    hunt_exit: bool = True
    # Measured: a kick leaves at ~1.4 m/s. Published to the team board as
    # the ball's velocity; below `Team.vel_use` (0.7) a coasting track is
    # ignored so this does not re-open intercept-on-claim.
    kick_speed: float = 1.4
    # A searching head sweeps +-`search_sweep` rad (period `search_sweep_s`)
    # while the body circles, when it has no track to look at.
    #
    # RE-SCREENED 2026-09-05 AND ITS OLD VERDICT WAS STALE. It shipped off on
    # "a searching head sweep makes the body turn MORE, 5/5 seeds", and the
    # mechanism recorded beside that was the ToF: the sensor is on the HEAD,
    # so a turned head reported walls that were not ahead and the brain
    # stopped for them. The clearance rule has since moved from sensor
    # COLUMNS to BEARINGS ("Chase: clearance by bearing, not by sensor
    # column"), and that coupling is now dead - measured by recomputing both
    # rules on the same 1 439 904 frames and binning by head yaw: past
    # 0.70 rad the old rule stops on 13.8% of frames and the shipped one on
    # 0.3%.
    #
    # So the arm was re-run on 24 seeds x 300 s of 2v2, and every field is
    # inside the noise: spinFrac +0.007 (p = 0.55), falls 57 -> 56
    # (p = 0.93), blocked seconds -0.18 (p = 0.78), possession +0.93
    # (p = 0.45), ball-in-view -0.005 (p = 0.71). It is a CLEAN NULL now,
    # not a knob with a reason - free to leave off, and free to turn on if
    # something else wants a swept head.
    #
    # (The bearing rule did not remove the coupling so much as make it
    # honest: past 0.70 rad the duck now has essentially no forward obstacle
    # sense at all rather than a wrong one. That is where the head-tracking
    # arm's falls come from - see `head_yaw_when`.)
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
    # 1.4 m/s and, on the floor before 2026-09-06, slowed at 0.04 m/s^2 -
    # that floor had NO rolling resistance (`Ball.rolling`), the 0.04 was
    # the tracker's own noise, and the ball rolled to the boards. On the
    # floor since, the sim's rolling resistance is speed-proportional (a
    # kick's speed halves about every 1.5 s: 1.4, 1.08, 0.75, 0.47, 0.28
    # m/s second by second); `ball_decel` is the constant-deceleration
    # stand-in over the first two seconds of a kick, which is where
    # `predict_s` looks. The kick leaves the level camera at once, 30-55 deg off the
    # nose, so the track coasted with a stale range for two seconds and a
    # new track was born when it was found again. With `predict_s` > 0
    # the head YAWS toward the predicted bearing (`head_yaw_gain`, to
    # `head_yaw_max`; always, or only while searching / looking), and
    # with `predict_steer` the search opens toward the predicted side and
    # the hunt walks to the predicted point (clamped to the pitch).
    # The head half SHIPS ON (`predict_s` 1.0 + `head_yaw_when` "always" +
    # `yaw_clear` 0.45); the STEERING half stays off. That split is measured,
    # and it reverses an earlier reading taken on 8 seeds of 1v1 judged
    # mostly on goals - a metric that needs 136 seeds to move (Track 4.1.5),
    # so the old table below could not have seen this either way. Keep it as
    # the record of what the steering costs:
    #   8 seeds x 300 s of 1v1, against 2.25 goals / 9.4 kicks / 0.38 falls
    #   with everything off - yaw always + steer 1.12 / 10.5 / 1.62; yaw off
    #   + steer 2.12 / 9.1 / 1.12; yaw in search + steer 2.12 / 10.0 / 0.75;
    #   yaw in search, no steer 2.12 / 6.5 / 0.75.
    # Every arm there that walks a predicted line loses falls, and the
    # steering is what walks blind lines into things, so `predict_steer`
    # stays off. The head does not walk anywhere. Re-measured properly on
    # 48 paired seeds of 2v2 (24 discovery + 24 fresh, agreeing): the head
    # bundle buys +8.0 points of ball-in-view (p<0.0001, 42/48 seeds) and
    # cuts the median time the ball is lost by 0.27 s (p=0.0002), for no
    # measured cost in falls, kicks, possession, spread, crowd, depth or
    # ball progress. The gate is what makes it free - see `yaw_clear`.
    ball_decel: float = 0.3
    predict_s: float = 1.0         # how long a prediction is worth acting on after the last hit (0: off)
    # …and how long a ball MEASURED AT REST is worth acting on (roadmap Track
    # 4 item 12f). The floor has had rolling resistance since 2026-09-06, so
    # a ball that stops stays stopped, but `predict_s` treats a 1.1 s old
    # sighting of a motionless ball exactly like a 1.1 s old sighting of one
    # that was rolling. That is the coverage problem under `kick_ahead_max`:
    # the plan is a median 3.6 s old at the swing, so the gate has nothing to
    # judge on and fails OPEN. A resting ball is the one case where an old
    # fix is still a good fix. `TrackerParams.rest_coast_s` has to be on too,
    # or the track is gone before this can use it. 0 = off.
    rest_predict_s: float = 0.0
    # …and the two `TrackerParams` halves of the same memory, mirrored here so
    # a battery sets the whole thing through one `MICRODUCK_CHASE`. Without
    # `rest_coast_s` the track expires on the 2.5 s clock and there is nothing
    # left for `rest_predict_s` to act on.
    rest_coast_s: float = 0.0
    rest_vel: float = 0.05
    # A body this close to the remembered ball may have moved it, so the
    # memory stops counting (`Tracker.disturb`). Our own kick and push always
    # do. Without this the prior is not a memory, it is a lie with a clock.
    rest_clear_m: float = 0.30
    head_yaw_gain: float = 0.9
    head_yaw_max: float = 1.4          # the walker's trained head-yaw range (upstream curriculum: +-1.40 rad)
    # Only let the head leave the walking line while the ToF says the line is
    # empty: a look target is dropped when the forward clearance is inside
    # this (`hunt_stop` = 0.45 is the natural scale; 0 = off).
    #
    # This exists because head-yaw ball tracking was a confirmed TRADE: on 48
    # paired seeds it bought +7.8 points of ball-in-view and cost +1.54 falls
    # a run (p<0.0001 both ways). The ToF is ON THE HEAD, so yawing it points
    # the bumper off the walking line, and since the clearance rule became
    # bearing-based it reports `+inf` honestly rather than a false wall - so a
    # yawed duck walks with no forward obstacle sense at all. The brain had
    # that signal and never consulted it before turning the head.
    #
    # Capping the yaw's MAGNITUDE was tried first and removes both halves
    # together, because the visibility and the falls live in the same frames
    # by magnitude. Clearance separates them. Measured over all 48 seeds,
    # counting frames where the head is past 0.35 rad (blind) and the old
    # column rule - which does not care where the head points - says
    # something really is ahead:
    #
    #                      blind frames   of which, obstacle ahead   per 1000
    #   head tracking off          0.0%                          -        0.0
    #   tracking, ungated         16.7%                      13.0%       21.8
    #   tracking, gated 0.45      14.1%                       8.2%       11.6
    #
    # The gate drops total blind frames by only 16% but the DANGEROUS ones by
    # 47%: it is selective, which is exactly why the cap was not the lever.
    # Against the same bundle ungated it removes 1.60 falls a run (p<0.0001,
    # worse on only 8 of 48 seeds) while giving up no visibility at all
    # (+0.002, p=0.80). Both blocks agree. What is LEFT: 8.2% of blind frames
    # still have something ahead, because the gate only consults the clearance
    # the head can currently see - it cannot know about what it has already
    # turned away from.
    yaw_clear: float = 0.45
    head_yaw_when: str = "always"  # or "search": yaw the head only while searching / looking
    predict_steer: bool = False    # the hunt bends and the search opens toward the prediction
    # THE HEAD IN TWO AXES (2026-09-09, `scripts/probe_ball_loss.py`). The
    # yaw law above follows the predicted ball; the pitch never does - the
    # gaze (`_gaze`) is a line-up law, refused past `gaze_bearing_max` and
    # off in every other state. Audited on 12 seeds x 180 s of 2v2, a duck
    # loses the ball 93 times a run, and HALF of those losses (571 of 1113)
    # begin with the ball slipping BELOW the frame at a median 0.27 m, 38 deg
    # off the nose, with the head yawed at it (|cmd| 0.50) and its pitch
    # command exactly 0.00 - after which the duck walks on, the ball ends up
    # behind it, and the loss runs a median 2.2 s (53% over 2 s). Those
    # events are 68% of all blind time. The other half is the ball behind
    # the body (a fix for the legs, not the head), occluded by a duck, or too
    # far to find.
    #
    # `track_pitch` pitches the head, in the same states and under the same
    # clearance gate as the yaw, by the SMALLEST command that keeps the
    # predicted ball `track_pitch_margin` inside the bottom edge of the frame
    # (`_track_pitch`; the gaze centres it, which is 2-3x deeper). Slant
    # geometry: the tracker's range is `radius / tan(width / 2)`, the slant,
    # and `_gaze`'s `atan2(height, range)` reads it as ground distance, which
    # at 0.27 m under-aims by 12 deg. Benched on the shipped walker
    # (scratchpad bench, 4 seeds x 6 s a pose, head slot alone): a cold turn
    # is unaffected at any pitch; the WARM in-place turn runs 0.61 rad/s at
    # 0.10 and 0.00 from 0.20 up - the "cannot turn head-down" rule's
    # threshold; walking at 0.3 costs 4% of speed at 0.20, 6.5% at 0.30, 9%
    # at 0.45, 13% at 0.60, no falls. So `track_pitch_max` caps the ask,
    # `track_pitch_turn` is what a turn in place may still take (0: none,
    # the shipped rule), and `cam_level_walk` is the walking camera's level -
    # the gait holds the head 0.08 rad higher than the standing 0.197
    # (DetectionFrame's note; measured in play, camera pitch 5.7 deg at the
    # median loss).
    #
    # MEASURED, AND IT SHIPS ON WITH `look_hold_s` (roadmap 12ae). Alone,
    # neither moves anything (discovery block, 12 seeds: view 44.0% against
    # 43.2%, median loss 1.38 against 1.37 s); together they do, on the
    # discovery block (median loss 1.37 -> 1.08 s, losses over 2 s 41 -> 35%)
    # and on a FRESH block of 24 seeds x 180 s (median 1.47 -> 1.14 s,
    # -0.34 +- 0.27, p = 0.018, better on 17 of 24; over 2 s 43 -> 34%;
    # falls 3 v 1 in 24 runs; kicks flat; possession +1.0 s/min, p = 0.08).
    # It does NOT stop the ball being lost - loss EVENTS go UP 16% - it makes
    # each loss shorter: the head still cannot see a ball under the chin
    # (0.21 m slant, 70 deg down, the line-up's own doing) and cannot pitch
    # while the body turns, but a head held PITCHED AND YAWED at the
    # remembered ball for 2.5 s instead of 1.0 s has it back in the frame the
    # moment the body moves, where a level one looks over it. Ball-in-view
    # +2.5 pp is unresolved at 24 seeds (MDE 4 pp). `track_pitch_max` 0.45
    # (0.30 measured the same on the discovery block and was not carried to
    # the fresh one). The settle is excluded from the pitch (`gaze_still`
    # and `settle_head_level` own the swing's run-up; tests/test_team.py)
    # after the batteries ran; the confirmation run of the shipping code is
    # in the roadmap item.
    track_pitch: bool = True
    track_pitch_margin: float = 0.15   # rad inside the bottom edge the predicted ball is kept
    track_pitch_max: float = 0.45      # the deepest command the tracking pitch asks for
    track_pitch_turn: float = 0.0      # ...during a turn in place (bench: 0.10 is free, 0.20 stalls the turn)
    cam_level_walk: float = 0.117      # the camera's depression while WALKING, rad (standing: `cam_level`)
    # ...and the same head in TIME. `look_hold_s`: the yaw law stops following
    # the track at `predict_s` (1.0 s) while the tracker keeps it to `coast_s`
    # (2.5 s), so for 1.5 s the brain believes in a bearing the head is not
    # pointed at - 15.8% of all blind frames in the audit above sit in that
    # gap. 0: the `predict_s` horizon; else the head follows the coasting
    # track (its bearing turns with the body, by odometry) this long after the
    # last hit. Ships at 2.5 = `coast_s`, with `track_pitch` (above: alone it
    # is a null, view 40.6% against 43.2%). `head_lead_s`: the head servo is a
    # measured 7 ticks (140 ms) behind its command at gain 1.04 (the audit's
    # cross-correlation over every duck), so aim it at the ball predicted that
    # far ahead. MEASURED OFF: alone view 40.0% against 43.2% (null, MDE 4
    # pp) and every ledger sign the wrong way; in the bundle it added nothing
    # the fresh block could see (median loss 1.10 with it, 1.14 without, on
    # the same seeds). 0: off.
    look_hold_s: float = 2.5
    # THE HEAD ON THE REMEMBERED BALL WHERE THE BLIND TIME IS (roadmap 12af).
    # The loss audit's per-tick trace: support, retreat and avoid own 74% of
    # the long blind stretches and consult the ball in none of them, while
    # every head law above runs off the tracker, which forgets in 2.5 s. With
    # `head_memory_s` > 0, in `head_memory_states` (a "+"-joined list, since
    # "," is MICRODUCK_CHASE's separator) and with no fresher look target,
    # the head yaws (and, under `track_pitch`, pitches) toward the ball the
    # BOARD has - a teammate sees it now - else toward this duck's own last
    # sighting (`self.memory`, odometry frame) while it is younger than
    # `head_memory_s`. The head only: nothing walks anywhere on this memory
    # (`seek_s` stays the walk's own knob), and the `yaw_clear` bumper gate
    # applies as to every other head yaw. 0 = off, the head before 2026-09-10.
    # MEASURED (roadmap 12ag; `probe_ball_loss`, 24 seeds x 180 s of 2v2 in
    # two blocks, paired): ball in view 45.9 -> 50.0% (+4.1 pp +-3.6, p =
    # 0.027, 16/24 seeds), the `behind` blind frames 52 619 -> 29 808, long-loss
    # duck-seconds -20%, falls 0.12 -> 0.08 a run; `eval-pitch` 24 x 300 s:
    # possession null at a 4% MDE, falls 4 -> 4, own goals 9 -> 0, kicks 102
    # -> 112 events. SHIPS ON at 30 s. With `avoid` added the view gain is
    # +4.8 pp but the pitch trends the wrong way (goals 26 -> 19, back-kicks
    # 22 -> 32%, both unresolved at 24 seeds): recorded, off.
    head_memory_s: float = 30.0
    head_memory_states: str = "support+wait+retreat"
    head_lead_s: float = 0.0
    # `track_pitch_turn` = 0.15 (the benched-free head pitch during a turn in
    # place) measured on the fresh block beside the bundle: median loss 1.14
    # (the bundle 1.10), kicks 2.79 against 1.79 (+1.0 +- 0.9, p = 0.034, an
    # instrument that needs ~190 seeds for a 10% change), falls 1 v 3. Not
    # distinguishable from the bundle on anything the block can resolve;
    # recorded as the next arm, not shipped.
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
    # MEASURED 2026-09-07, push-only (0) against the shipped brain, 24
    # discovery + 24 fresh seeds of 2v2 (roadmap Track 4 s6 A.4): pooled and
    # paired, possession +3.15 s/min (p<0.001, better on 38 of 48 seeds),
    # ball advance +0.080 (p=0.003), signed progress +0.073 (p=0.011), falls
    # flat, goals for +0.27 a run (p=0.079) - and OWN GOALS +0.21 a run
    # (p=0.008; 0 -> 10 on the fresh block), crowd +0.06 (p=0.001). A walk
    # through the ball has none of the kick's problems (no settle, no stale
    # plan, no head-down pose, no exit angle) and no aim either: near our
    # own mouth it walks the ball in. Stays inf (always kick) until the
    # selector (kick_select) can choose the push as an action with the same
    # own-goal filter it applies to the kicks.
    push_beyond: float = math.inf
    push_behind: float = 0.16
    push_speed: float = 0.3
    push_s: float = 0.5
    # A push spot's own line-up tolerance (m; 0 = `lineup_tol`). A kick needs
    # its foot within 3 cm of the spot (12a's funnel: 0-3 cm 0 % whiff,
    # 3-6 cm 8 %); a push walks 0.64 m through a ball 0.16 m ahead and does
    # not. At the boards a line-up that has REACHED its spot to 5 cm still
    # times out (12am: 66-72 deg off heading at the timeouts), because
    # outside `lineup_tol` it servos at the spot along the wall instead of
    # turning to the heading; a looser tolerance lets a push square up and
    # go. The kick's tolerance is untouched. A no-op on the shipped brain,
    # which plans no push (`push_beyond` inf, `kick_select_push` off,
    # `board_push` 0); it is what any push arm needs (12an: 0 -> 2 -> 24
    # pushes in 40 board episodes with this and `push_aim_tol`).
    push_tol: float = 0.06
    # ...and its own AIM tolerance (rad; 0 = `aim_tol`). The square-up at a
    # reached spot beside a wall is the line-up's last blocker (12am, 12an:
    # 68 deg off heading at the timeouts); a kick needs `aim_tol` 0.25, a
    # push along a wall the wall guides does not. Measured in the board
    # probe: pushes 13 -> 24 in 40 episodes at 0.5 (with the side stop
    # off in that arm; the side stop alone made them fewer, 13 -> 9).
    push_aim_tol: float = 0.5
    # THE BOARD PUSH (roadmap 12g, built 2026-09-10). A ball at the boards is
    # a ball almost nothing can be done with (12p): the kick's stand spot is
    # in the wall or past the bumper, and every board line-up times out.
    # Dribbling everywhere lost (13a: the pusher stands on the ball and the
    # kicks vanish), so this is the SITUATIONAL rule the ask named: a ball
    # closer than `board_push` to a board is walked through along the wall
    # toward the goal - the same lines `_along_the_boards` lays a kick on,
    # up the pitch on a side board, away from our own mouth on our end
    # board, across the mouth on theirs - from `push_behind` behind it, so
    # the wall keeps it rolling and it comes off the boards into room where
    # a kick has a chance. The spot must be somewhere the body can stand; a
    # ball nearer the wall than that is pushed on a line tilted INTO the
    # wall by the smallest angle up to `board_push_tilt` that moves the spot
    # out to a body-clear gap. 0 = off (the kick plan everywhere).
    #
    # BUILT, MEASURED OFF (roadmap 12an; kick gym, 480 episodes an arm, the
    # ball placed at a board, on flat boards AND on the lab's cove). The
    # push touches the ball more and moves it less than the kick line-up it
    # replaces: a push's travel is 5-12 cm (the 0.5 s walk at 0.3 m/s from
    # 0.16 m behind reaches the ball and nudges it - against a wall, into
    # the wall) against a connected kick's 0.75-0.86 m, so the ball's
    # advance toward the goal per episode is five times worse on flat
    # boards (+0.038 -> +0.007 m), flat on the cove's near band, 40 % worse
    # on its wider one (+0.098 -> +0.059). A push made strong (`push_s` 1.0
    # at `push_speed` 0.45) whiffs 13-26 % instead of 42-69 % and travels
    # 0.26 m, and per episode still trails the kick (+0.083 v +0.098 cove,
    # +0.015 v +0.038 flat). Falls 0-2 everywhere. The shipped brain on the
    # cove touches a board ball 207 times an arm where the flat gym said 66:
    # the board problem the push was for is mostly the flat gym's.
    #
    # AND THEN IT WAS MEASURED ON THE WRONG METRIC ALL ALONG (2026-09-12).
    # Everything above judges the push as an ATTACKING tool - advance a
    # touch, whiff, kicks - where the kick beats it, so it shipped off. It
    # is a DEFENSIVE tool, and on our own line it is the only thing that
    # touches the actual failure.
    #
    # `scripts/owngoal_gym.py` puts the ball just off OUR OWN line with the
    # duck coming from up-pitch - the stance that makes own goals - and asks
    # only whether the ball ends in our net. The shipped brain concedes in
    # 14.4% of episodes, and the composition is the finding:
    #
    #   own goals WALKED in  92%          own goals KICKED in  8%
    #
    # The duck bumps the ball over its own line with its BODY. `aim_mode`'s
    # clamp, `kick_select`'s own-goal filter and `kick_select_t_own` all act
    # on the 8%: tightening the tolerance to 0.02 and to 0.0 is a dead null
    # (+1.2 points, MDE 8.3, 320 episodes an arm), exactly as that split
    # predicts. `board_push` acts on the 92%, because `_board_line` on our
    # own end board clears the ball SIDEWAYS toward the corner rather than
    # letting the walk carry it at the mouth.
    #
    # | arm | own goals | | ball advance |
    # |---|---|---|---|
    # | shipped | 14.4 % | | 0.130 |
    # | **board_push 0.25** | **9.3 %** | **-5.1, MDE 3.6, RESOLVED (-35 %)** | 0.072 |
    #
    # Pooled over 2560 episodes across a discovery and a FRESH block, better
    # on 14 of 16 fresh seeds, and the whole reduction is in the walked-in
    # category. A third block (seeds 200-211) shows a clean dose-response:
    # 0.40 reaches 8.3% (-6.5, RESOLVED) for more of the advance.
    #
    # AND IT IS FREE IN A MATCH, which nothing else in this item managed:
    # 2v2 over 12 paired seeds at 0.25, every metric null and every one
    # trending the right way - ballAdvance +0.086, possession +0.56, crowd
    # +0.024, kickCount +0.08, spread -0.24. At 0.40 they are null too but
    # all trend negative (kickCount -1.17, 3 of 12), so 0.25 is the pick.
    #
    # SHIPS ON at 0.40 with the F.2 pack (`chase_behind`, `approach_keepout`,
    # `chase_behind_upto`). 0.25 was the match-null pick on its own; 0.40 is
    # the dose that, combined with the approach bias gated to our third, cut
    # own goals 14.4% -> 0.9% with no measured match cost.
    board_push: float = 0.40
    board_push_tilt: float = math.radians(45.0)
    # GET BEHIND THE BALL BEFORE TOUCHING IT (roadmap Track 4 s6 F.2).
    #
    # The shipped planner never asks for it. `aim_max` caps how far round the
    # ball a line-up may go, so a duck standing BETWEEN the ball and the goal
    # it attacks is offered only the best line inside a 60 deg window of its
    # own line of sight - measured on a parked ball, a spot 97 deg round
    # (beside the ball, not behind it) for a kick 96 deg off the goal, square
    # across the pitch. "Behind" is 180 deg. The duck then arrives 0.10 m from
    # the ball (inside `duck_touch` 0.22) FROM THE WRONG SIDE, and the arrival
    # itself is the contact: over 6 seeds x 300 s of the lab's pitch-2v2, 84%
    # of all ball-moving touches are bodies rather than kicks, and the 28% of
    # them made from up-pitch send the ball backwards 81% of the time (median
    # -0.091 m against +0.148 m for a touch from behind).
    #
    # `aim_mode="goal"` already walks round (143 deg, kick 24 deg off goal)
    # and ships off because it measured worse - half the kicks, signed
    # progress -0.26 (p=0.012) - and the reason is visible in the same
    # geometry: it is the one arm whose straight servo line CROSSES the ball
    # (closest approach 0.05 m at 82-88% of the way in), so the duck arcs
    # round in contact and shoves the ball backwards as it goes. The target
    # rule and the path have to land together or the first re-earns that.
    #
    # `behind_ball` is the target half: metres behind the ball, on the
    # ball-to-goal line, to STAGE at when the best kick this stance allows
    # would lose ground. A staging spot, not a kick spot - it sits outside
    # `duck_touch`, so arriving there cannot move the ball - in its own mode
    # ("around"), which neither settles nor swings: reaching it drops the
    # spot and the next tick lays a real kick line from the good side.
    # Self-releasing, because the test is on the OUTCOME (`behind_ball_cos`,
    # the cosine of the kick's true exit line against the goal direction,
    # exit angle included) and not on where the duck stands: the moment the
    # clamp can offer a line that gains ground, the plan takes the kick.
    # 0 = off, which is the pre-2026-09-11 brain.
    behind_ball: float = 0.0
    # How bad a kick has to be to be worth walking round for: the cosine of
    # its exit line against the goal direction. The default is cos(`aim_max`)
    # = 0.5, so the rule reads "if the clamp cannot get the ball within its
    # own 60 deg window of the goal, go round instead" - one constant, not
    # two. Measured on a parked ball at 10 deg steps, it stages the 7 stances
    # of 36 spanning -30..+30 deg either side of the goal line and leaves a
    # worst standing kick of 56 deg off; at 0.0 ("never actually lose
    # ground") only the single worst stance stages and an 86 deg kick is
    # still allowed, and at 0.7 it stages 10 of 36 (-40..+50).
    behind_ball_cos: float = 0.5
    # THE COMMITMENT. Without these the rule decides afresh every tick and
    # never actually walks anywhere: measured over 180 s of the lab's
    # pitch-2v2, 32 `around` spells, ZERO arrivals, median spell 0.08 s -
    # four ticks - and the duck never closer than 0.50 m to a spot it must
    # reach within 0.06 m. Two things end a spell and neither is the walk
    # failing: 21 of 32 the plan simply flipped back to "kick", because the
    # trigger reads `kick_select`, a 30-rollout MONTE CARLO sampler whose
    # answer is not the same twice - asked 30 times about one frozen stance
    # it said "go round" 1 time with the selector on and 30 with it off; and
    # 11 of 32 `avoid` fired, which outranks `lineup` in `PRIORITY` and drops
    # the spot, and walking round a ball in a crowd is exactly the manoeuvre
    # that passes near another duck.
    #
    # So the walk-round is LATCHED: the trigger is read once, and from then
    # on the staging spot is re-laid against wherever the ball now is until
    # the duck is behind it (`behind_ball_done`, the angle at the ball
    # between the duck and the goal - 2.1 rad is 120 deg, generously short of
    # squarely behind so the last stretch is not paid for) or the budget runs
    # out (`behind_ball_s`). The latch lives beside `spot` and not in it, so
    # an `avoid` that drops the spot PAUSES the walk instead of cancelling it
    # - which is what `avoid` is for, and it keeps its veto over the body.
    behind_ball_s: float = 6.0
    behind_ball_done: float = 2.1
    # How far the staging spot may SLIDE along the arc when the boards will
    # not take the ideal one, and in what steps. A wall is a reason to stand
    # somewhere else behind the ball, not a reason to stop wanting to be
    # behind it: with the spot fixed, "staging spot in a board" released 6 of
    # 8 walk-rounds at 1v1 and 4 of 10 at 2v2. 1.05 rad is 60 deg either way,
    # so the slid spot is never worse than 120 deg round - which is exactly
    # `behind_ball_done`, so a duck that reaches a slid spot has satisfied
    # the release test by construction.
    #
    # MEASURED AND IT SHIPS OFF (0 = the ideal spot or nothing), which is the
    # arm every number below was taken on. At 1.05 rad in 0.26 rad steps the
    # slide makes the duck COMMIT to balls it then cannot finish with:
    # walk-rounds ending behind the ball 5 of 10 -> 4 of 15 at 2v2 and 1 of 2
    # -> 0 of 8 at 3v3, attempts up (10 -> 15, 2 -> 8) because the trigger no
    # longer declines a ball whose ideal spot is unreachable. Giving up on ONE
    # POINT is not the same mistake as giving up on the OBJECTIVE, and this
    # knob was built by confusing them: a staging spot inside a board is often
    # a ball that should not be walked round at all.
    behind_ball_arc: float = 0.0
    behind_ball_step: float = 0.26
    # THE COST GATE. Walking round is cheap FAR from the ball and dear near
    # it: at 1.5 m out, aiming at the far side instead of the near one is a
    # few degrees of approach; at 0.4 m it is a 180 deg orbit at walking pace
    # with the ball sitting dead throughout. The first cut had no notion of
    # this and committed to orbits it could not afford - which is what put a
    # third of the run on a dead ball. Only TAKE the commitment while the
    # ball is at least this far away (0 = at any range); the latch is what
    # carries it through the close-in part, so a walk begun at 1.2 m still
    # finishes.
    behind_ball_far: float = 0.0
    # APPROACH FROM THE RIGHT SIDE INSTEAD OF ORBITING WHEN YOU GET THERE.
    # Measured: `_plan` is only ever called at a ball range of 0.39-0.62 m
    # (median 0.47, max 0.80), because `lineup_range` is 0.6 and outside it
    # the `chase` branch steers at `ball.bearing` - straight at the ball. So
    # the duck runs at the ball and only starts thinking about which side to
    # be on once it is already there, at which point every fix is a 180 deg
    # orbit at walking pace with the ball sitting dead. Nothing downstream of
    # `_plan` can be cheap, because the decision is made too late.
    #
    # This moves it earlier and makes it free: while CHASING (outside
    # `lineup_range`, where no kick is planned yet) steer at a point this far
    # behind the ball on the ball-to-goal line instead of at the ball. At
    # 1.5 m that is about 11 deg of extra turn and no detour worth the name;
    # by the time the duck is inside `lineup_range` it is already on the side
    # it wanted, and `_plan`'s ordinary near-side spot IS the good one. No
    # orbit, no commitment, no dead ball - the geometry is paid for during a
    # walk the duck was making anyway. 0 = off (steer at the ball).
    #
    # MEASURED 2026-09-12 and IT WORKS, on the case it is for. The instrument
    # is `wrongside_gym.py`'s: one duck, one still ball, the duck started on
    # the GOAL SIDE of it, one episode per placement — the stance every other
    # arm in this item was built for, at 60 episodes a minute instead of the
    # nine wrong-side touches a 240 s match yields. With
    # `chase_behind=0.40, approach_keepout=0.20`:
    #
    # | block                | ball advance a 10 s episode |        |
    # |----------------------|------------------|--------------------|
    # | discovery, seeds 0-7 | 0.025 -> 0.123 m | +0.098 (MDE 0.094) |
    # | FRESH, seeds 100-107 | 0.044 -> 0.164 m | +0.121 RESOLVED    |
    # | pooled, 960 episodes | 0.034 -> 0.143 m | +0.109 RESOLVED    |
    #
    # ...a 4.2x improvement that REPLICATES ON A FRESH BLOCK, which nothing
    # else tried for this problem has done. Backward touches 40.0 -> 33.3 %
    # and kicks an episode 0.281 -> 0.335 move with it, neither resolved.
    #
    # AND IT IS FREE WHERE IT DOES NOT APPLY, but only because the offset is
    # SCALED by how wrong the side is (see the chase branch). Unscaled it
    # fired on good approaches too and measured harm on `kick_gym`'s own
    # right-side placement: swings 211 -> 166 and whiff 5 -> 12 % (p=0.016).
    # Scaled, the same battery is swings 211 -> 211 and whiff 5 -> 9 %,
    # verdict null (p=0.178).
    #
    # AT MATCH LEVEL IT IS A WASH, AND THAT IS A DILUTION, NOT A REFUTATION:
    # only about a quarter of match touches are made from the wrong side, so
    # a 4x on that quarter is inside the noise of 12 paired seeds. 2v2:
    # total ball advance a run 4.25 -> 4.46 (flat), advance a touch 0.117 ->
    # 0.153 (7/12), backward 31.3 -> 21.2 % (7/12), and TOUCHES 36.0 -> 29.1,
    # the one resolved delta and a cost. The cost is a crowd effect: at 1v1
    # it falls to -4.75 and stops resolving, with every other column positive
    # (advance a run +0.53, backward -7.5, 6 of 8 seeds).
    #
    # SHIPS ON at 0.40 with the F.2 pack. Alone it is a wash in a match
    # (dilution: only a quarter of touches are from the wrong side); gated
    # to our third by `chase_behind_upto=-0.5` it is the approach half of
    # the own-goal cut, and general play never pays.
    chase_behind: float = 0.40
    # ...and only for the duck the board has made the ATTACKER. The match
    # cost of the chase bias is crowd (2v2, +0.160, resolved) and possession
    # (3v3, -4.6 s/min, resolved) while the gym win is large and replicated,
    # and the obvious suspect is the SUPPORTERS: a duck that is not going for
    # the ball has no business doing approach geometry on it, and three ducks
    # all arcing toward the same point is a pile-up by construction. With
    # this on, a supporter chases the ball exactly as it always did.
    #
    # DEAD KNOB - it changes NOTHING, and the reason is in `PRIORITY`:
    # `support` sits ahead of `seen`, so a duck in the support role takes the
    # support branch and never reaches the chase branch at all. Only the
    # attacker ever runs it, gate or no gate. Proven rather than argued: on
    # 12 paired 2v2 seeds the gated and ungated arms return byte-identical
    # crowd, ballAdvance, possession, kickCount, kicksBack and spread.
    #
    # Kept, documented, and DEFAULT OFF so nobody measures it again. It also
    # retracts a claim made earlier the same day - that gating fixed the 3v3
    # crowd cost. That comparison was a different roster AND a different
    # config, not a test of this gate.
    chase_behind_attacker: bool = False
    # ...and the part of the pitch the bias applies on, in ATTACK coordinates
    # (-1 our own goal line, +1 theirs). The own goals it is for happen in
    # OUR half only, but the bias fires everywhere, so general play pays for
    # a defensive fix: with `board_push` the pair buys crowd +0.100 at 0.25
    # and possession -2.94 at 0.40, both resolved over 12 paired 2v2 seeds.
    # +1.0 = everywhere, which is what every arm above was measured with;
    # 0.0 = our own half only; -0.5 = our defensive THIRD, and that is the
    # one that works.
    #
    # THE FINAL CONFIGURATION (2026-09-12). With
    #   board_push=0.40, chase_behind=0.40, approach_keepout=0.20,
    #   chase_behind_upto=-0.5
    # own goals in the stance that makes them (`scripts/owngoal_gym.py`) go
    # 14.4% -> 0.9%: -13.5, MDE 3.2, RESOLVED, -94% relative, pooled over
    # 1000 episodes across TWO fresh blocks (seeds 900-909 -95%, seeds
    # 1000-1011 -93%, neither a discovery block). The walked-in category -
    # which is 92% of all own goals - goes 13.5% -> 0.9%, and kicked-in goes
    # to zero.
    #
    # AND IT COSTS NOTHING MEASURABLE. 12 paired 2v2 seeds, every metric
    # null: possession -0.98, crowd -0.028 (better), ballAdvance +0.007,
    # kickCount -1.25, kicksBack -0.167, spread -0.166.
    #
    # The gate is what makes it free, and the dose-response says why. The
    # same pair with the bias applied EVERYWHERE costs possession -2.94
    # (resolved); applied over our own HALF it costs kickCount -2.33
    # (resolved); over our defensive THIRD it costs nothing, because the own
    # goals all happen there and general play never sees the bias at all.
    # A defensive fix should be paid for in the defensive third.
    # SHIPS ON at -0.5 with the F.2 pack: the bias is a no-op in the
    # attacking two-thirds, which is what keeps match possession flat.
    chase_behind_upto: float = -0.5
    # Carried to the team board (`Team.side_s`, brain/team.py) by
    # `brain_kwargs`, so one MICRODUCK_CHASE spec drives the whole A/B.
    # It is a TEAM knob - it changes who is sent for the ball, not what
    # any duck does - and it is here only so a battery can set it.
    team_side_s: float = 0.0
    # COST THE WALK-ROUND WITH THE SAME MACHINERY THAT COSTS THE KICKS
    # (roadmap F.2). `kick_select` already forward-simulates every candidate
    # line 30 times and ranks them by what actually happens to the ball. The
    # walk-round was the one action nobody costed: `behind_ball_cos` asks
    # only whether the kick available NOW is bad, never what going round is
    # worth or what it costs, so the duck committed to orbits it could not
    # afford. With this > 0 the trigger instead scores both:
    #
    #   now   = the selector's own best verdict from where the duck stands
    #   after = the best verdict from BEHIND the ball, on the goal line
    #   score = p_goal * `behind_ball_goal_w` + value   (Mellmann's two keys,
    #           collapsed into one number so they can be traded off)
    #
    # ...and goes round only when `score(after) - score(now)` beats this
    # knob times the SECONDS the walk takes (the arc round to the far side
    # at `behind_ball` radius, at `speed`). So the rule is dear when the
    # duck is near and badly placed and cheap when it is nearly there, which
    # is the economics the first cut had no way to express. 0 = off, and
    # then `behind_ball_cos` is the trigger as before.
    behind_ball_rate: float = 0.0
    behind_ball_goal_w: float = 3.0   # potential units a certain goal is worth (the field spans about -1..+2)
    # ...and the path half: keep this far off the ball while walking to a
    # spot. The straight servo line is the only route today and there is a
    # keep-out for every other duck's body (`duck_keepout`) and none for the
    # ball. Above 0 the walk to a kick, push or staging spot that would pass
    # nearer the ball than its own TARGET does is bent round a tangent of
    # this circle instead, on the side the duck is already on (the shorter
    # way, and `duel_side`'s rule: the far side steers the walk across the
    # ball). The effective radius is `min(approach_keepout, |target - ball|)`
    # so the target is always on or outside the circle and the arrival is
    # never fenced off from itself - a kick spot 0.10 m from the ball keeps
    # its own 0.10 m, which is still enough to stop the line-up cutting the
    # corner through it. 0 = off (the straight line).
    #
    # MEASURED (2026-09-11), 4 arms x 6 PAIRED seeds x 300 s of the lab's
    # pitch-2v2 (formation + cove + get-up), classifying every ball-moving
    # touch as a kick or a body and reading its 1 s outcome. BOTH SHIP OFF:
    # nothing here resolves as a win at this size, and the one delta that
    # does clear the 6-seed MDE is a COST.
    #
    # | arm                          | backward | up-pitch | advance/touch | touches |
    # |------------------------------|---------:|---------:|--------------:|--------:|
    # | shipped                      |   32.5 % |   27.4 % |      0.121 m  |    46.7 |
    # | behind 0.35, keepout 0.30    |   22.9 % |   23.5 % |      0.178 m  |  *37.2  |
    # | behind 0.30, keepout 0.25    |   24.7 % |   23.2 % |      0.174 m  |    41.8 |
    # | behind 0.35, keepout 0.20, cos 0.3 | 28.4 % | 27.7 % |  0.126 m  |    39.3 |
    #
    # (* the only paired delta past the MDE: -9.5 touches a run, worse on 5
    # of 6 seeds. Advance per touch is +0.058 m, better on 5 of 6 — the right
    # sign, consistently, but just UNDER its own MDE of 0.067, so suggestive
    # and not a result.)
    #
    # Three readings worth keeping. (1) The walk-round buys the better touch
    # by taking FEWER touches, which is the same bill `aim_mode="goal"` paid
    # — the path fix reduces it, it does not remove it. (2) The THRESHOLD is
    # the lever and the RADIUS is the cost: slackening `behind_ball_cos` to
    # 0.3 throws the whole gain away (+0.005 m) while a smaller keep-out
    # keeps it (+0.054) and halves the touch bill (-4.8, no longer resolved),
    # so this would be the arm to take further. (3) Own goals over the arms
    # were 4 / 6 / 1 / 5 and MEAN NOTHING: item 1.5 puts `ownGoals` at ~347
    # seeds to resolve.
    #
    # AND THEN THE ARM TURNED OUT NOT TO DO THE THING. Instrumented over
    # 180 s of the same pitch: 32 `around` spells, **0 arrivals**, and the
    # closest the duck ever came to a staging spot it must reach within
    # 0.06 m was 0.50 m. A spell lasts about a quarter of a second - the mode
    # flips on, the duck takes a step, the plan reverts to "kick" - because
    # the trigger is read fresh every tick against a stance that is changing
    # under it and a tracked ball that jitters across the threshold. The
    # walk-round never happens, so "+0.058 m a touch" is SELECTION (bad
    # touches declined) and not correction (good touches won), and the dead
    # ball says the same: >20 s spells 10 -> 19, 248 s -> 538 s of the 1800
    # played. Two hypotheses refuted on the way: `lineup_s` is not the
    # blocker (timeouts 7 -> 6, none from `around`) and the boards are not
    # (the body-clear fallback fires as designed).
    #
    # THE LATCH WAS BUILT (`behind_ball_s`, `behind_ball_done`) AND IT FIXED
    # THE MECHANISM: counted over the LATCH's lifetime, which is the right
    # unit because `avoid` chops one walk into several spot-spells, 10
    # walk-rounds of which 5 END BEHIND THE BALL, median final angle at the
    # ball 125 deg, closest approach to the staging spot 0.06 m - against
    # 0 of 32, 56 deg and 0.50 m without it. The duck really does walk round
    # now, about half the times it tries.
    #
    # AND IT STILL DOES NOT PAY. Same 6 paired seeds, shipped / unlatched /
    # latched: backward touches 32.1 / 25.1 / 27.8 %, touches from up-pitch
    # 27.5 / 23.5 / **30.6** %, advance a touch 0.123 / 0.167 / 0.148 m,
    # touches a run 46.7 / 41.8 / 42.0, dead spells over 20 s 10 / 19 / 21
    # totalling 248 / 538 / **599** s of the 1800 played. NOTHING clears the
    # 6-seed MDE, and the latched arm is WORSE than the unlatched one on
    # every column it was supposed to win: making the walk-round actually
    # happen raised the share of touches made from up-pitch ABOVE the
    # shipped brain's and put a third of the run on a dead ball.
    #
    # The likely mechanism, stated as the hypothesis it is: a duck walking
    # round the ball TRANSITS the up-pitch region, and a 0.25 m keep-out is
    # not wide enough to stop the body clipping the ball on the way past, so
    # the manoeuvre manufactures the very touches it exists to prevent. If
    # this is picked up again, test that first (a wider keep-out, or a
    # walk-round that goes the way that never crosses up-pitch) - do not
    # start from the outcome battery.
    #
    # AND THEN THE CHEAP INSTRUMENT KILLED IT OUTRIGHT. `wrongside_gym.py`
    # (one duck, one still ball, the duck started on the goal side) ran 400
    # episodes of `behind_ball=0.30, approach_keepout=0.25`: the duck moved
    # the ball in **0 of 400**, took **0 kicks**, and advanced it 0.000 m,
    # against 92 %, 0.250 and 0.012 m for the shipped brain - three resolved
    # deltas, all catastrophic. Left alone with a ball it cannot lose, the
    # walk-round does not kick at all. Nothing in a 240 s match makes that
    # visible, because there a teammate or an opponent always takes over.
    #
    # `behind_ball` stays 0 and should not be revived: the rule is correct,
    # it is committed to, it completes about half the time it starts, and
    # given a ball to itself it never gets round to kicking it. What DID
    # work is upstream of all of it - `chase_behind`, which fixes the side
    # during the walk-in the duck was making anyway.
    # `approach_keepout` SHIPS ON at 0.20 with the F.2 pack: the collinear
    # run-in `chase_behind` cannot bend on its own (the behind-point sits
    # on the ball-to-goal line) is the case the keep-out exists for.
    approach_keepout: float = 0.20
    # The other duck's BODY (measured over 4 traced runs: 5 of 7 falls had the
    # other duck 3–9 cm away and this one turning in place — search, blocked
    # or lining up — the walker tips over when it turns against a body it
    # cannot see below its ToF rows). A tracked duck inside `duck_keepout`
    # and ahead: nothing walks or turns toward it; inside `duck_touch` it is
    # against us: stand until it moves.
    duck_keepout: float = 0.4
    duck_touch: float = 0.22
    # THE DUEL, first form (roadmap Track 4 item 11c): on a line-up with the
    # ball nearer than the other duck, the keep-out shrinks to this so the
    # attacker finishes its line-up unless the other is about to touch.
    # In the open 263 of 508 3v3 line-ups die to `avoid` at 0.4 s with the
    # ball 0.43 m away and an OPPONENT 0.33 m ahead. MEASURED OFF, 12 seeds
    # of 3v3 on the ball-out floor at 0.25: kicks 8.7 -> 8.9 (p = 0.85),
    # dead ball -5 s (p = 0.66), progress +0.08 (p = 0.51), falls 1 -> 2 -
    # a null; the two attackers meet at the ball whatever the radius, and
    # the survey's colour sense (C.4) is still what the duel needs. 0 = off.
    lineup_keepout: float = 0.0
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
    # THE CORNER TRAP (roadmap 12aa). `_support`'s no-ball branch is
    # `turn(1.0, cold)` and nothing else — a search that works in open field
    # and cannot work against the boards, where the whole view IS board. A
    # supporter that loses the ball near a corner therefore turns there for as
    # long as the ball stays lost: measured at 123.4 s in one corner (seed 30,
    # d3, 0.10 m off both boards, `support` 92% of those ticks, the ball 0.53 m
    # away and unseen throughout). It is self-reinforcing — it cannot see
    # because it is in the corner and it stays in the corner because it cannot
    # see — and the anti-stuck rule cannot save it: `stuck_s` runs only in
    # `avoid`/`blocked`/`yield` and additionally wants under 0.3 rad of yaw
    # change, so a duck SPINNING in `support` fails both of its conditions.
    #
    # `support_unstick_s` seconds of supporting without getting anywhere
    # TRIGGERS THE RETREAT — the escape the
    # brain already has and already tunes (`retreat_turn_s`, `retreat_walk_s`),
    # reached by a trigger it was missing rather than by a second escape
    # beside it. Measured on the seed that produced the 123.4 s visit: 0.0 s
    # at 1 s, 4.6 s at 2 s, 8.8 s at 4 s.
    #
    # TWO EARLIER VERSIONS, both kept here because each failed in a way worth
    # not repeating. Walking to the post in `_support` instead of retreating
    # only got out in 28 s, because a supporter facing a board has `tof_stop`
    # zero its forward command and — in `support` specifically — no turn
    # either, so it servos into a wall it cannot walk through; the retreat
    # turns FIRST, which is why it is the right escape. And doing it whenever
    # the ball is unknown, with no place or time gate, cut the trap just as
    # well but moved every supporter on the pitch: 4.8 s a run more time with
    # no ball belief (t = 4.2) for no measurable reduction in board time — a
    # policy change wearing a bug fix's clothes.
    #
    # The clock is on the DUCK'S POSITION, not on its belief: the belief
    # flickers (the trapped duck had one on 29% of ticks, interleaved), so a
    # clock any sighting resets never reaches a threshold at all — the first
    # version of this rule was bit-identical to shipped for exactly that
    # reason. 0 disables the rule.
    # 4.0 and not 2.0, which was the surprise: firing LATER is better on every
    # axis measured. 48 seeds x 180 s of 2v2 — corner time a duck a run 2.1 s
    # (off) / 0.9 (at 2.0) / **0.5 (at 4.0)**, visits over 30 s 5 / 0 / 0, and
    # falls over 96 seeds 5 / 13 (p 0.045) / 8 (p 0.32, not separable). Each
    # firing at 4 s is an escape the duck actually needed; at 2 s a share of
    # them interrupt a search that was about to succeed on its own, which costs
    # the escape's own risk for nothing. 0 disables the rule.
    support_unstick_s: float = 4.0
    # THE GATE IS MOTION, NOT PLACE (roadmap 12ac). This rule first shipped
    # gated on "within `support_unstick_m` 0.30 m of a board", because the
    # corner was where the freeze was SEEN. Auditing every freeze the brain
    # produces — 15 s with under 0.15 m of travel, `scripts/probe_freeze_audit.py`
    # — says the corner was one instance of a general fault: of 25 such stands
    # in 1440 duck-seconds, 2 were a supporter legitimately AT its post and
    # **20 were a supporter with no ball belief at all, in OPEN PLAY, nowhere
    # near a board**, which the place gate cannot reach. A supporter that loses
    # the ball stops playing wherever it is standing.
    #
    # So the trigger is displacement: under `support_unstick_move` of travel
    # across `support_unstick_s`. That is far more selective than the version
    # this replaces AND than the ungated one rejected earlier ("walk home
    # whenever the ball is unknown", which moved every supporter for +4.8 s a
    # run of no-ball time and no measurable gain) — a supporter walking to its
    # post, or turning and re-acquiring, never trips it.
    #
    # `support_hold_tol` is the exemption that makes it safe, and it is the
    # whole subtlety: a supporter standing still ON its post is doing its job,
    # not freezing. It matches `_support`'s own servo stop, so the rule and the
    # controller agree on what "arrived" means by construction.
    # THE PLACE GATE WAS TRIED BESIDE THIS ONE AND IS INERT. OR-ing the old
    # "within 0.30 m of a board for `support_unstick_s`" term on top produced
    # SIX OF SIX bit-identical seed files: a supporter that has sat within
    # 0.30 m of a board for four seconds has by then also failed the
    # displacement test, so the place term never fires on a tick the motion
    # term has not already claimed. Not kept — a knob that changes nothing is
    # dead code, and this repo has been caught shipping one before (12o).
    support_unstick_move: float = 0.15
    support_hold_tol: float = 0.12
    # Where a duck with a static ROLE stands when it is not the one on the
    # ball (roadmap Track 4.3). All three are a spot to hold, not a new state
    # machine: the same `_support` servo walks to them and faces the ball.
    #   defender  — on the line from the ball to its own goal, this far out
    #               from the goal line, and never over the halfway line. Its
    #               job is to be BETWEEN, which is the one thing the shipped
    #               roster never does: the deepest duck of a 2v2 averages
    #               1.35 m up a 1.7 m half.
    #   striker   — ahead of the ball toward the goal it attacks, offset to
    #               the side the ball is NOT on. The offset is the whole
    #               difference from the poacher that was measured off (that
    #               one stood ON the line and reversed on fresh seeds), and
    #               it is what keeps a striker from being a second duck on
    #               the ball.
    #   mid       — between the ball and the centre spot, on the ball's side,
    #               kept inside the middle third.
    defend_depth: float = 0.5
    strike_ahead: float = 0.8
    strike_side: float = 0.4
    mid_side: float = 0.5
    # THE KEEPER (roadmap Track 4 s6 B.2): a static role that holds the line
    # `keeper_depth` in front of its own mouth, on the ball-to-goal line and
    # never outside the posts, takes the ball only inside its own fifth of
    # the pitch (Team.ROLE_ZONES) and never covers for a field player. The
    # interception work (4d) found the lever was "earlier than the block":
    # 40 of 70 threats start with the ball already inside 0.3 m of the line,
    # where a field player cannot arrive in time - a keeper is already
    # there. So a keeper turns `intercept_eta` on at `keeper_intercept_eta`
    # unless a battery speaks, and otherwise uses the shipped block geometry.
    #
    # MEASURED IN 2v2 AND IT DOES NOT PAY THERE (2026-09-07, rolling-
    # resistance floor, scripts/probe_threat.py --roles, 24 seeds, both arms
    # forked on one tree state; keeper+striker against defender+striker):
    #   conceded threats        10/30 = 33%  ->  11/32 = 34%     p = 0.93
    #   danger clock < 0.9 m    9.09 -> 17.5 s/min   +8.4 +/- 3.9  p = 0.031, worse on 17 of 24 seeds
    #   danger clock < 0.45 m   1.11 -> 5.68 s/min   +4.6 +/- 2.8  p = 0.10
    #   nearest the ball got to the mouth   0.416 -> 0.403 m       p = 0.84
    # The keeper does exactly what it was built to do - on the sheet it holds
    # its post for the whole run while the striker plays the field - and that
    # is the cost: a side of TWO cannot spare a duck to stand in goal, so the
    # ball lives in the keeper's box twice as long, and a keeper's clearing
    # kick on this floor travels too little to get it out (the 4b/item-7
    # kick). Same shape as the interception result: works, does not pay.
    # Ships as a role a scenario may declare (the editor lists it), NOT in
    # `formation_roles`. Unmeasured: 3v3 with a defender in front of it,
    # which is where a keeper would earn its place.
    keeper_depth: float = 0.25
    keeper_intercept_eta: float = 3.0
    beside_m: float = 0.3
    beside_s: float = 1.5
    # Use the colour classifier to tell a teammate from an opponent
    # (roadmap Track 4.4.2): with it on, a duck gives a STRANGER
    # `opp_keepout` of room and keeps the standard `duck_keepout` for a
    # teammate, since the team board already coordinates teammates and
    # nothing coordinates an opponent.
    #
    # SHIPS OFF, MEASURED. At 0.55 m on 3v3 it looked like the crowding fix
    # the 13-fall trace asked for — crowd 26.5% -> 19.7%, p = 0.002 over 24
    # seeds — and then did NOT replicate: 21.4% -> 20.5%, p = 0.593 on 24
    # fresh ones, and better on 25 of 48 pooled, which is a coin. Falls
    # (181 -> 160, p = 0.27) and everything else are unresolved; the only
    # replicated effect is the cost, possession -2.16 s/min (p = 0.032).
    # The same shape as the poacher and the bump-stand rule, caught by the
    # same rule: confirm on seeds the effect was not found on.
    #
    # The SENSE is not what failed, and it stays: a colour-aware rule with a
    # better idea than "stand further off" can use it. For scale, the static
    # roles (Track 4.3) take the same metric from 22.4% to 3.5% and DID
    # replicate — standing somewhere useful beats standing further away.
    use_color: bool = False
    opp_keepout: float = 0.0       # an opponent this near and ahead: treat it as a duck to avoid
    # …and how much nearer the BALL this duck must be than that opponent
    # before it declines to turn away at all (the contest; see the note where
    # `contesting` is computed). Needs `use_color`, since turning this on
    # against a TEAMMATE is how two of ours shoulder each other. 0 = off.
    contest_margin: float = 0.0
    # THE DUEL (roadmap C.4, second half; B-Human's Zweikampf). `contesting`
    # above is the case where THIS duck is nearer the ball. This is the other
    # one - the opponent is nearer - and it is not the same question. The
    # three answers already measured (`lineup_keepout`, `opp_keepout`,
    # `contest_margin`) all asked "do I turn away from that body", and all
    # three were null; the population probe that followed them
    # (scripts/probe_contest.py) said why: the contest rule can act on 0.07%
    # of duck-ticks, so no whole-match metric could have moved.
    #
    # `scripts/probe_duel.py` takes the same measurement for THIS case first,
    # on the shipped brain. The situation - an opponent inside `duel_near` of
    # the ball, nearer to it than this duck, this duck going for the ball -
    # is 9.35% of duck-ticks by the world's own geometry and **7.44% as the
    # brain can actually see it** (2v2, 4 seeds x 120 s), a hundred times the
    # contest rule's population. And it is where the ball is lost: over the
    # next 2 s the other side holds the ball 47-65% of the time against this
    # duck's team's 12-27%, where in the same states with no opponent nearer
    # it is 60% ours to 6% theirs. What the duck does today is flinch
    # (`retreat` 6.2% of ticks, `avoid` 4.7%) or line up on a ball it will
    # not get (2.0%).
    #
    # So this is a POSITIONING rule, which is the one kind the three nulls
    # leave open. A shield - a body between the opponent and the ball - is
    # not available when the opponent is the nearer of the two; what is left
    # is the BLOCK: stand `duel` metres goal-side of the ball on the line to
    # our OWN goal, facing the ball, so the opponent's next touch has to come
    # through this duck. The value is the standoff in metres; 0 = off (the
    # shipped chain, to the bit). It never fires for a supporter (the role
    # branch is ahead of it, so a post is never abandoned) nor in `settle`
    # (that is a swing about to happen), and `avoid` still owns a touch.
    duel: float = 0.0
    duel_near: float = 0.35        # an opponent this close to the ball is contesting it (probe_duel's number)
    # ...and the lateral offset of that spot off the ball-to-own-goal line, in
    # metres, 0 = the line itself (the measured `duel` arms, to the bit).
    #
    # `kick_gym.py --duel` found the block at 0.15 m nearly DOUBLES first touch
    # (9.4/7.9 -> 14.8/17.1%) and hands the metres straight back: our advance
    # falls 0.029 m an episode, equal and opposite to the 0.031 the opponent's
    # loses. The rows say where those metres go, and it is not where the item
    # assumed. A duck standing ON the block spot touches the ball AWAY from our
    # goal - forward, by construction. But at the moment of our first touch the
    # duck is a median 0.317 m from its own spot and has NEVER reached it (0 of
    # 42 within `intercept_tol`): every touch we win is made EN ROUTE. And the
    # route is a straight servo line from wherever the duck stands to a spot on
    # the far side of the ball, so when the duck starts up-pitch the line runs
    # THROUGH the ball and the contact drives it home - the duck is up-pitch of
    # the ball on 50% of its own first touches, the ball leaves with a negative
    # x-velocity on 64%, and that velocity predicts the 2 s advance at r=0.87.
    #
    # So the metres are a PATH problem, not a stance problem, and this is the
    # path fix: put the spot `duel_side` off the line, on the side of it the
    # duck is already on, so the walk goes round the ball rather than across
    # it. The side is the duck's own and not the nearer board's on purpose - a
    # board-side rule is a coin flip against this mechanism, helping on the
    # half of draws where the duck is already board-side and steering the walk
    # further across the ball on the other half.
    #
    # MEASURED AND IT SHIPS OFF (`kick_gym.py --duel`, 40 episodes x 12 seeds a
    # block, seeds 0-11 and 100-111). It does not trade the metres for the
    # touch: it throws the touch away. We touch first 14.8/17.1% at
    # `duel_side` 0 -> 4.6/6.5% at 0.15 -> 1.0/0.8% at 0.25, BELOW the shipped
    # 9.4/7.9%, and the opponent's advance climbs back with it (+.002/+.005 ->
    # +.018/+.030 -> +.020/+.041). The net is flat on every arm and not
    # resolvable at this size anyway (MDE 76-99% of baseline).
    #
    # Why, in one column: our duck touches the ball AT ALL on 10.6/11.5% of
    # shipped episodes, 21.0/21.9% at `duel` 0.15, and 8.1/10.6% with the
    # offset. The block's first touch is not a stance at all - it is the
    # COLLISION on the way in, and it exists only because the spot sits inside
    # `duck_touch` (0.22). An offset of 0.15 puts the spot 0.21 m off the ball
    # and 0.25 puts it 0.29, so the walk that used to end in the ball now ends
    # beside it. Any variant that keeps the touch has to keep the spot inside
    # `duck_touch`, which leaves the contact direction set by where the duck
    # started, not by where the spot is - so the touch's direction is not a
    # spot knob's to fix.
    duel_side: float = 0.0
    # The ToF sees the ball at the feet (tof_floor_ball): inside `tof_ball_m`
    # with the head dipped, a floor blob feeds the tracker as a ball sighting
    # when the camera has none - the level camera loses a floor ball inside
    # 0.3 m, which is where the line-up and the kick live. Measured OFF at
    # 0.5 m (8 seeds x 300 s of 1v1): 1.62 goals, 8.1 kicks, 0.75 falls a
    # run against 2.38 / 7.4 / 0.38 - a blob at the feet is as often the
    # other duck's foot as the ball, and a line-up on a foot is a fall.
    #
    # RE-OPENED at the replacement module's real 60 deg vertical FOV (the
    # original was measured at 48) and it closes harder. 1v1 goals are a
    # weak instrument, so this counted the blob's own EVENTS instead - 6
    # seeds x 180 s, every tick the blob fired, against the ball's true
    # position:
    #
    #   V FOV   blob ticks   camera already had the ball   blind-case ticks   of those, the ball
    #    48 deg      4785                   87.5%                  599              30.1%
    #    60 deg      5809                   93.6%                  374              36.9%
    #
    # Two independent reasons it stays off, both worse at 60 than at 48:
    #  1. It is almost always REDUNDANT - the camera already has the ball on
    #     88-94% of the ticks the blob fires.
    #  2. In the case it exists for (camera blind) it is WRONG about two
    #     times in three, and that is the case that ends in a line-up on a
    #     foot.
    # The wider lens does raise blind-case precision (30 -> 37%) but cuts
    # the opportunity by 38% (599 -> 374 ticks), because it reaches 5.5 cm
    # further into the blind zone itself (blind radius 28.5 -> 23.0 cm,
    # docs/camera-hardware.md 3d). Better optics shrink this feature's job
    # faster than they improve it.
    tof_ball_m: float = 0.0
    # WHERE the blob is allowed to speak (roadmap item 12e, 2026-09-08). The
    # two measurements above pooled EVERY tick it fired, and that population
    # is what made it look useless: it fires in `search`, `avoid`, `retreat`
    # and `support` too, where the duck is anywhere on the pitch and a
    # ball-height thing 0.3 m away is usually a foot. Split by the state the
    # decision would actually use it in (`scripts/probe_tof_ball.py`, 6 seeds
    # x 180 s of 2v2, 1830 events):
    #
    #   population                              events  camera had it  IS THE BALL
    #   every tick it fires                       1830           70%          85%
    #   lineup / settle                           1379           82%          97%
    #   lineup / settle, nobody beside            1354           82%          97%
    #   …and the camera blind (the case for it)    241            0%          85%
    #
    # 85% in the blind line-up is a different sensor from the 30% the pooled
    # number reported, and it clears the bar the probe names (four times in
    # five, because the cost of the other case is a line-up on a foot). With
    # this True the blob is offered only in `lineup`/`settle` with no body
    # beside; False reproduces the old always-on behaviour those two
    # measurements killed.
    tof_ball_lineup: bool = True
    # SELF-LOCALISATION from the goal posts (brain/localize.py; roadmap Track
    # 4 s6 C.2 and item 10). The brain's odometry is dead reckoning that
    # drifts (`OdomNoise`): at `datasheet` two teammates' frames wander
    # 0.456 m apart over a run, and the goal is "where it was at spawn".
    # On, a particle filter over the four posts (a `post` detection class)
    # corrects (x, y, yaw) every tick before anything reads it: the tracker's
    # placements, the spot, the board's ball and pose all move with it.
    # MEASURED (scripts/probe_odom_goal.py, 3 seeds x 300 s of 2v2, medians
    # over the run, raw odometry -> localised):
    #
    #   preset      pos err          yaw err        miss at goal line   over half-width   mates disagree
    #   datasheet   0.215 -> 0.072 m  20.2 -> 2.3 deg  0.663 -> 0.075 m   68% -> 11%       0.295 -> 0.078 m
    #   hostile     0.706 -> 0.089 m  66.3 -> 5.4 deg  1.034 -> 0.158 m   80% -> 30%       0.958 -> 0.146 m
    #
    # "Miss at goal line" is what the heading error costs a shot laid out in
    # this frame; "mates disagree" is how far apart two teammates put the
    # SAME ball, which the board (brain/team.py) shares as a bare (x, y).
    # Off = the shipped brain, bit for bit. The roster default turns it ON
    # for any duck whose odometry preset is not `ideal` (team.brain_kwargs):
    # at `ideal` the frames already agree exactly and every soccer number
    # was measured there.
    localize: bool = False
    # A bump (Senses.bumped: the body is touching another body - contacts in
    # the sim, the IMU / servo loads on the robot): no turn in place for
    # `bump_stand_s` after the contact STARTED. 12 of 13 traced 3v3 falls
    # were standing turns beside an unseen opponent. MEASURED against no
    # rule at all, and THE FALL REDUCTION DID NOT REPLICATE:
    #   seeds 24-35        falls 4.83 -> 3.25   -1.58 +/- 0.92  p = 0.14
    #   seeds 24-35 again  falls 6.17 -> 4.00   -2.17 +/- 1.01  p = 0.060
    #   seeds 200-211      falls 4.08 -> 4.33   +0.25 +/- 1.04  p = 0.88
    #   all 24 DISTINCT layouts             -0.81 +/- 0.69  p = 0.264
    # The first two are the same twelve layouts measured twice (per layout
    # -1.88 +/- 0.84, p = 0.055; pooling them as 24 says p = 0.012, which
    # is repeated measures, not replication, and is what this comment said
    # first). On twelve layouts nobody had run the effect is absent and
    # slightly reversed. "A third fewer falls" is WITHDRAWN - the
    # poacher's shape exactly, caught by AGENTS.md's third rule.
    #
    # It stays on for rosters anyway, as a default nobody has earned in
    # either direction: the pooled point estimate still favours it
    # (better on 15/24) and nothing it was suspected of costing moved
    # (kicks +0.83 p = 0.51, goals -0.58 p = 0.50, advance and progress
    # flat on the fresh block), so flipping it off would be reading noise
    # the other way. Falls want ~376 seeds for a 25% shift; this is 24.
    # `bump_back` below is the arm worth measuring against it next. Its first form
    # cost 1v1 1.50 goals and 1.00 falls against 2.38 / 0.38, and a trace
    # of 838 bumps said why, refuting the obvious guess on the way:
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
    # ("block" is inert unless `intercept_eta` is on, and it belongs here for
    # the same reason `support` does: standing on the line is a turn in place
    # with a body possibly beside it.)
    bump_stand_states: tuple[str, ...] = ("support", "lineup", "settle", "turn", "block")
    # A contact episode ends after this long without one; the freeze runs
    # from its onset and is never extended by staying in contact.
    bump_gap_s: float = 1.0
    # Back up instead of standing, for this long. UNTRIED, ships at 0.
    #
    # Standing is what the rule does today, and standing does not END the
    # contact: measured from 0.10 m of separation, 16 trials, a standing
    # duck was still at 0.099 m four seconds later and cleared 0.30 m in
    # 0 of 16. A straight reverse cleared it in a median 1.6 s (14/16),
    # beating turn-90-and-walk (2.7 s) and turn-180-and-walk (3.2 s) - and
    # unlike either it keeps the ball in frame, so no `search` follows.
    # The walker reverses at 0.23 m/s, faster than it walks forwards; the
    # "it cannot" above this was a dead-band reading (see `gait.back_up`).
    #
    # Why it is the obvious next thing to measure here: the two failure
    # modes the trace found are both a duck that cannot separate. "6 of 8
    # falls were a stand pressed against the other duck: the walker leans
    # on it" is a stand that had somewhere to go. "Standing on a body keeps
    # touching it - 44 -> 105 bumps a run" is the same. And the states this
    # rule is kept OUT of are excluded because "the turn IS the escape" -
    # which was true only while a reverse was believed impossible.
    #
    # MEASURED, and it is the most closed null in this file. Three arms on
    # the same twelve fresh layouts (3v3, seeds 200-211, 300 s a seed):
    #
    #            falls  kicks  goals   possession  advance
    #   no rule   4.08   6.50   2.17     11.83      0.40
    #   stand     4.33   7.33   1.58     13.00      0.42
    #   back      4.33   6.67   2.00     11.97      0.44
    #
    # stand -> back on falls is EXACTLY 0.00 +/- 1.13, p = 1.000 (52 events
    # against 52); nothing else resolves either. And the knob is NOT inert:
    # instrumented over one 3v3 run it issues 690 reverse commands (13.8 s
    # a run) and cuts the ticks spent touching another body from 4473 to
    # 2529 of 90000 - a 43% drop, exactly the self-feeding the 838-bump
    # trace found ("standing on a body keeps touching it").
    #
    # So: the mechanism is real, it operates, and it does not matter. Time
    # in contact is not what makes a 3v3 duck fall. That agrees with the
    # probe it was built on - a turn beside a STATIC body fell 0 times in
    # 98 trials down to 8 cm; the falls need a duck that is MOVING into
    # you - and it means the remaining lever is the closing duck, not the
    # contact. Ships at 0, as a measured null rather than an untried idea.
    bump_back: float = 0.0
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
    # Blocking a ball that is rolling into our own goal (brain/intercept.py):
    # leave the play, walk onto the ball-to-goal line ahead of it, and let the
    # body stop it. The repo owner's idea, watching a 2v2 — "come in from the
    # side and deflect it" rather than line up a kick on a ball that is
    # already past.
    #
    # `intercept_eta` is the master knob and 0 is OFF: a ball predicted to
    # reach our own goal within this many seconds is worth leaving the play
    # for. THE BASELINE it is aimed at (`scripts/probe_threat.py`, 48 seeds
    # x 300 s of 2v2, `runs/thr-base48b.jsonl`): 54 of 70 threats conceded,
    # with the best-placed defender playing on (support 24%, line-up 21%,
    # chase 13%) or searching (18%) through them, and a defender 0.15-0.50 m
    # off the ball's path conceding 31 of 31.
    #
    # SHIPS OFF: IT DOES WHAT IT SAYS AND THE THING IT WAS BUILT TO MOVE DOES
    # NOT MOVE. Measured at 8.0 over 48 paired seeds x 300 s of 2v2 and then
    # 48 FRESH ones (seeds 100-147), plus a 24-seed ledger:
    #
    #   * It engages, and exactly as designed: through a threat the
    #     best-placed defender is in `block` 21% of the ticks, and `lineup`
    #     falls 21% -> 6% and `chase` 13% -> 5%. It is not a dead path.
    #   * The conceded fraction does NOT resolve: 108/137 (79%) -> 102/138
    #     (74%) pooled over the 96 seeds, z = -0.96, p = 0.34 (77->73% and
    #     81->75% on the two blocks; the direction is consistent and the
    #     size is not).
    #   * What DOES replicate is where the ball spends its time: seconds a
    #     minute inside 0.9 m of some mouth, 15.80 -> 14.26, -1.53 +/- 0.60,
    #     p = 0.010 pooled, better on 60 of 96 seeds, same sign on both
    #     blocks (p = 0.093 then 0.054). The 0.45 m clock did not
    #     (p = 0.016 -> 0.558).
    #   * The ledger is quiet: goals 39 -> 36 (p = 0.71), falls 57 -> 46
    #     (p = 0.19), possession +1.5 s/min (p = 0.16) - the feared cost of
    #     abandoning the attack does not appear - advance, crowd, spread,
    #     depth and back-kicks all flat. The one big number, signed
    #     ballProgress -0.369 (p < 0.001), is attribution and not harm: a
    #     duck standing in front of a goalward-rolling ball BECOMES the
    #     possessing duck, and over 6 seeds 3804 of those ticks carry the
    #     ball toward that duck's own mouth at 4.65 m/min against 0.75 in
    #     every other state. `ballAdvance` - the forward half of the same
    #     accumulator, credited by the same rule - is flat, which a duck
    #     genuinely shoving the ball goalward could not manage.
    #   * The cost that is real: kicks 193 -> 149, a fifth of the touches.
    #
    # The ceiling was in the baseline all along, and it is the reason to
    # leave this off rather than tune it: 40 of the 70 threats are declared
    # with the ball ALREADY inside 0.3 m of the goal line (38 of those 40
    # conceded), and a block was geometrically available - perfect knowledge,
    # a generous walk model - in only 20 of the 54 conceded ones. The lever
    # is earlier than the block.
    intercept_eta: float = 0.0
    intercept_vmin: float = 0.12   # …closing this fast on our goal (m/s, a scalar rate, not a velocity)
    intercept_dt: float = 0.6      # …differenced over this window of SIGHTINGS (never a coasted track)
    intercept_age: float = 0.8     # …the newest of which is no older than this
    intercept_ahead: float = 0.25  # stand at least this far goal-side of the ball
    intercept_keep: float = 0.45   # …and no nearer than this to our own goal (below ahead+keep there is no block: see intercept.py)
    intercept_tol: float = 0.12    # on the line within this: stand and face the ball
    intercept_hold: float = 1.0    # keep blocking this long after the trigger drops (the estimate is jittery)
    intercept_clear: float = 0.0   # > 0: with the ball this near, sweep ACROSS the line instead of standing in it

    @staticmethod
    def env_names(spec: str | None = None) -> set[str]:
        """The knob NAMES a battery set through `MICRODUCK_CHASE`.

        A default cannot be told from a caller's explicit value by comparing
        them — `bump_stand_s=0` on the command line and the shipped 0.0 are
        the same number — so the roster default in `brain/team.py` asks which
        names were spoken rather than guessing from the values."""
        if spec is None:
            spec = os.environ.get("MICRODUCK_CHASE", "")
        return {item.partition("=")[0].strip() for item in spec.split(",") if item.strip()}

    # The string knobs' legal values: `from_env` refuses anything else, so a
    # typo (aim_mode=clmap) cannot silently run the other arm.
    CHOICES: ClassVar[dict[str, tuple[str, ...]]] = {
        "aim_mode": ("clamp", "los"),
        "head_yaw_when": ("always", "search"),
        "support_mode": ("back", "ahead"),
    }

    @staticmethod
    def from_env(spec: str | None = None) -> "ChaseParams":
        """The defaults with `MICRODUCK_CHASE` applied — how a battery says
        which variant it is measuring:

            MICRODUCK_CHASE="two_stage=1,approach_speed=0.4" uv run eval-pitch …

        Every knob above ships on a measurement, and until now the only way
        to measure one was to edit its default, run, and edit it back — a
        step that is invisible in the battery's own record and was done
        wrong at least once. `--tag` says which variant a row belongs to;
        this says what the variant IS, from the same command line.

        An unknown name or an unreadable value RAISES: a typo that silently
        measured the default would be the expensive kind of mistake here.
        Tuple-valued knobs are not settable this way."""
        p = ChaseParams()
        if spec is None:
            spec = os.environ.get("MICRODUCK_CHASE", "")
        if not spec.strip():
            return p
        kinds = {f.name: getattr(p, f.name) for f in fields(p)}
        over: dict = {}
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            k, sep, v = item.partition("=")
            k, v = k.strip(), v.strip()
            if not sep or k not in kinds:
                raise ValueError(f"MICRODUCK_CHASE: {item!r} is not <ChaseParams field>=<value>")
            cur = kinds[k]
            if isinstance(cur, bool):
                if v.lower() not in ("0", "1", "true", "false", "on", "off"):
                    raise ValueError(f"MICRODUCK_CHASE: {k}={v!r} is not a boolean")
                over[k] = v.lower() in ("1", "true", "on")
            elif isinstance(cur, int):
                over[k] = int(v)                                   # "20.0" or "n": unreadable, and it raises
            elif isinstance(cur, str):
                allowed = ChaseParams.CHOICES.get(k)
                if allowed is not None and v not in allowed:
                    raise ValueError(f"MICRODUCK_CHASE: {k}={v!r} is not one of {allowed}")
                over[k] = v
            elif isinstance(cur, tuple):
                raise ValueError(f"MICRODUCK_CHASE: {k} is a tuple; set it in code")
            else:
                over[k] = float(v)
        return replace(p, **over)


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def det_gate(base_s: float, periods: float, period_s: float) -> float:
    """The detection-freshness gate to use against a sensor of THIS cadence.

    A gate in seconds is only meaningful beside the period of the camera it
    judges: 0.4 s is four frames at 10 Hz and less than one frame at the
    robot's 2 Hz, where it goes stale on a QUARTER of the ticks with nothing
    missed (26.7 % measured on a 1v1 pitch; roadmap 12av follow-up (1) and its
    own follow-up (1)). `max` rather than a replacement, so
    the knob can only lengthen the gate and every rate at or above
    `periods / base_s` Hz keeps the shipped number exactly.

    Used by `Chase` (through `ChaseParams.det_max_periods`) and by
    `scripts/probe_ball_loss.py`, whose loss EVENT rule is the same question
    asked of the same sensor."""
    return max(float(base_s), float(periods) * float(period_s))


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

    def __init__(self, p: ChaseParams | None = None, goal: tuple[float, float] | None = None,
                 team=None, duck_id: str = "", bounds: tuple[float, float] | None = None,
                 goal_w: float = 0.0, role: str | None = None, det_noise: str | None = "datasheet"):
        # No params given (the lab, the benchmark, the /sim page): the
        # shipped defaults, with `MICRODUCK_CHASE` applied so a battery can
        # name its variant on the command line. A caller that passes `p`
        # (brain/team.py's roster kwargs, a test) is never overridden.
        self.p = ChaseParams.from_env() if p is None else p
        self.goal = None if goal is None else (float(goal[0]), float(goal[1]))
        self.goal_w = float(goal_w)        # the mouth's width: how wide a target the ball has (kick_cone)
        self.bounds = None if bounds is None else (float(bounds[0]), float(bounds[1]))   # the pitch's half-extents inside the boards
        self.loc = None                    # the goal-post particle filter, when `localize` is on (brain/localize.py)
        if self.p.localize and self.bounds is not None and self.goal_w > 0:
            from .localize import (  # noqa: PLC0415  (only a pitch pays for it)
                Localizer,
                pitch_posts,
            )
            self.loc = Localizer(pitch_posts(self.bounds, self.goal_w))
        self.team = team
        self.duck_id = duck_id
        # The STATIC role off the scenario ("defender" / "midfielder" /
        # "striker" / None), which is a different thing from `self.role` — the
        # dynamic attack/support the board hands out every tick. This one says
        # which third of the pitch this duck may take the ball on and where it
        # stands when it does not have it; that one says whether it has it now.
        self.job = role
        if role == "keeper" and self.p.intercept_eta <= 0 and "intercept_eta" not in ChaseParams.env_names():
            from dataclasses import replace  # noqa: PLC0415
            self.p = replace(self.p, intercept_eta=self.p.keeper_intercept_eta)   # a keeper blocks by default
        # The tracker's uncertainty model is the detector's datasheet
        # (roadmap C.1): `det_noise` is the duck's detector preset, which
        # brain_kwargs passes from the scenario.
        # Re-entry detection (roadmap 12o). Cheap: two attributes and one
        # float compare a tick, warned at most once per brain.
        self._last_step_t: float | None = None
        self._reentry_warned = False
        self.tracker = Tracker(TrackerParams.for_detector(
            det_noise, rest_coast_s=self.p.rest_coast_s, rest_vel=self.p.rest_vel))
        # Half the camera's vertical field of view: what `_track_pitch` keeps
        # the ball inside. The camera the sim runs (`MICRODUCK_CAMERA`), read
        # once here the way the tracker reads the detector's datasheet.
        from ..sensors.detector import DetectorSpec  # noqa: PLC0415
        _spec = DetectorSpec.from_env()
        self._half_v = math.radians(_spec.fov_v_deg) / 2
        # …and the same spec's CADENCE. `DET_MAX_AGE` is a class constant in
        # seconds; this SHADOWS it per instance with the period-derived gate
        # (`det_max_periods`, default 0.0 → the constant, unchanged), so every
        # `self.DET_MAX_AGE` read below — and every probe that reads
        # `brain.DET_MAX_AGE` off a constructed brain — asks its question of
        # the camera this duck actually has. `Follow` has the same constant
        # and is deliberately left alone: it looks at people, at the lab rate,
        # and nothing in this item measured it.
        self.det_period = 1.0 / _spec.rate_hz
        self.DET_MAX_AGE = det_gate(Chase.DET_MAX_AGE, self.p.det_max_periods, self.det_period)
        self.gait = GaitWatch()
        self.blocker = Interceptor()
        self._kick_rng = None              # kick_select's generator, seeded from the duck id on first use
        self._kick_choice = None           # kick_select_learned's Chooser (E.2), built on first use; survives reset()
        self._field = None                 # the supporter's potential field, built on first use (support_field)
        self.last_select = None            # kick_select's last Verdict, for probes and the /sim page
        self.reset()

    def reset(self) -> None:
        # A new episode may legitimately restart the clock, so the re-entry
        # comparison starts over with it rather than firing on the boundary.
        self._last_step_t = None
        self.kicks = 0
        self.pushes = 0
        self.declines = 0          # swings refused by `kick_side_max`
        self.unreach_dropped = 0   # candidate lines dropped by `spot_reach` (their spot in a board)
        self.unreach_corners = 0   # plans where NO candidate was reachable (the fan left whole)
        self.attack: float | None = None                            # heading of the goal it attacks (first odom yaw)
        if self.loc is not None:                                    # the goal-post filter starts over with the episode
            from .localize import Localizer, pitch_posts  # noqa: PLC0415
            self.loc = Localizer(pitch_posts(self.bounds, self.goal_w))
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
        self._field_prev: tuple[float, float] | None = None  # the field's last spot (support_field hysteresis)
        self._kickoff_wait = False                           # standing off the other side's kickoff (kickoff_wait)
        self.post: tuple[float, float] | None = None         # where a supporter is holding, for probes and tests
        self.contesting = False                              # the contest fired this tick (contest_margin)
        self.dueling = False                                 # the duel fired this tick (duel)
        self.duel_spot: tuple[float, float] | None = None    # ...and where it is standing, for probes and tests
        self._sup_poses: list[tuple[float, float, float]] = []   # (t, x, y) while supporting: the freeze window (support_unstick_s)
        self._bump_t = -1e9                                  # last contact
        self._bump_t0 = -1e9                                 # onset of the current contact episode
        self.last = (0.0, 0.0, 0.0)
        self.spot: tuple[float, float, str | None, float, str] | None = None   # x, y, foot, heading, "kick"|"push"|"around"
        self._spot_ball: tuple[float, float] | None = None      # the ball that spot was laid against (approach_keepout)
        self._around: tuple[float, float, float] | None = None  # (ball x, y, t) the walk-round is committed to
        self._around_cool = -1e9   # no re-commit before this time: a budget that was SPENT
        self._keep_side = 0.0      # remembered way round the ball (see _keep_off)
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
        self.blocker.reset()

    def _gaze(self, rng: float, neck: float | None = None, down: float | None = None) -> float:
        """The gaze COMMAND that puts a floor ball at `rng` on the camera's
        axis. With `gaze_neck` > 0 the same command drives both slots, so the
        divisor is what the two of them deliver together (measured additive,
        `head_gain` + `neck_gain` per unit); at 0 this is the old law.
        `neck` / `down` override `gaze_neck` / `head_down` for one call (the
        settle's own look, `settle_gaze_neck` / `settle_head_down`)."""
        p = self.p
        k = p.gaze_neck if neck is None else neck
        cap = p.head_down if down is None else down
        h = p.cam_z - 0.035
        if p.gaze_slant:
            # `rng` is a SLANT range (the detector's range_est, and the tracker
            # places its xy that far along the bearing), so the depression that
            # centres the ball is asin(h / slant), as `_track_pitch` already
            # says; atan2(h, slant) treats it as ground distance and under-aims
            # (roadmap 12ae: ~12 deg at 0.27 m). Off, the law `gaze_still` was
            # measured with, to the bit.
            want = math.asin(float(np.clip(h / max(rng, h), -1.0, 1.0)))
        else:
            want = math.atan2(h, max(rng, 0.05))
        gain = p.head_gain + p.neck_gain * k
        return float(np.clip((want - p.cam_level) / max(gain, 1e-6), 0.0, cap))

    def _head_pose(self, cmd: float, yaw: float = 0.0, neck: float | None = None) -> tuple[float, float, float, float]:
        """The 4-slot head command for a gaze of `cmd`. The neck looks UP on a
        positive command, so a downward gaze mirrors it negative. With both
        extras off the slots are plain 0.0 and not -0.0 — the old tuple, to
        the bit. `neck` overrides `gaze_neck` for one call."""
        k = self.p.gaze_neck if neck is None else neck
        return (-k * cmd if k else 0.0, cmd, yaw, 0.0)

    def _track_pitch(self, rng: float, cam_z: float, walking: bool) -> float:
        """The SMALLEST head-pitch command that keeps a floor ball at slant
        range `rng` inside the frame, `track_pitch_margin` above its bottom
        edge (`track_pitch`; `_gaze` centres it instead, 2-3x deeper). `rng`
        is the tracker's range, which is the SLANT: the ball's depression is
        asin(height / slant), not atan2(height / ground). `cam_z` is the
        camera's height (the last frame's, or `cam_z`) and the level it rests
        at is `cam_level_walk` walking, `cam_level` standing. Clipped to
        `track_pitch_max`."""
        p = self.p
        h = cam_z - 0.035                                   # the lens above the ball's centre
        dep = math.asin(float(np.clip(h / max(rng, 0.05), -1.0, 1.0)))
        want = dep - (self._half_v - p.track_pitch_margin)   # the axis depression that puts the ball at the margin
        level = p.cam_level_walk if walking else p.cam_level
        gain = p.head_gain + p.neck_gain * p.gaze_neck
        return float(np.clip((want - level) / max(gain, 1e-6), 0.0, p.track_pitch_max))

    def _gaze_range(self, odom, ball) -> tuple[float, float] | None:
        """(range, bearing) to aim the gaze at during a line-up when the ball
        is not being seen right now: where the tracker last PLACED it
        (odometry frame, so it survives the duck walking on — `Track.range`
        does not, it only moves on a hit), else the ball this line-up was
        planned around. None when neither exists, or when the target is
        further off the nose than `gaze_bearing_max` (0.6 rad shipped, or
        `head_yaw_max` when `gaze_yaw` is on).

        The old text here said a ball past the 31° horizontal half-field is
        "unseeable with a fixed head at any pitch". That is a level-camera
        rule and it is wrong for a pitched one (2026-09-06, measured on the
        composed model): ON the kick spot the ball is 0.08 m forward and
        0.06 m to the side, 37° off the nose, and with the head 60° down it
        sits at 16.5° camera bearing and -19.7° elevation - inside the
        frustum, unoccluded. The gaze was being refused by this very test
        at the one place it was needed."""
        p = self.p
        tgt = None
        if ball is not None and ball.xy is not None:
            tgt = ball.xy
        elif self.spot is not None:
            sx, sy, _, u, _ = self.spot
            tgt = (sx + p.kick_ahead * math.cos(u), sy + p.kick_ahead * math.sin(u))
        if tgt is None:
            return None
        dx, dy = tgt[0] - odom[0], tgt[1] - odom[1]
        bearing = _wrap(math.atan2(dy, dx) - odom[2])
        if abs(bearing) > (p.head_yaw_max if p.gaze_yaw else p.gaze_bearing_max):
            return None
        return math.hypot(dx, dy), bearing

    def inputs(self) -> dict:
        if self._senses is None:
            return {}
        out = age_inputs(self._senses, self.TOF_MAX_AGE, self.DET_MAX_AGE)
        out["target"] = None if self.last_seen_t is None else {
            "bearing": round(self.last_bearing, 3), "range": None,
            "since": round(self._senses.t - self.last_seen_t, 2)}
        out["tracks"] = self.tracker.payload(self._senses.t)
        out["chase"] = {"kicks": self.kicks, "pushes": self.pushes, "role": self.role,
                        **({"declines": self.declines} if self.declines else {}),
                        **({"job": self.job} if self.job else {}),
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
        when that costs under `aim_max` of detour, else whatever `aim_mode`
        says (the shipped "los" kicks along the line of sight, because a full
        walk-round crossed walls and the other duck — measured) —
        offset sideways so the nearer foot meets it. The left foot kicks a
        ball to its LEFT. A far goal (`push_beyond`) makes it a push spot
        squarely behind the ball. Returns (x, y, foot, heading, mode)."""
        p = self.p
        x, y, yaw = odom
        bx, by = self._ball_xy(odom, ball)
        los = yaw + ball.bearing
        self._spot_ball = (bx, by)
        if p.spot_lead > 0 and self._senses is not None and ball.vel_hits >= 2:
            # Where it will be when we arrive: the walk at `speed`, capped so
            # a bad velocity cannot throw the spot across the pitch.
            eta = min(p.spot_lead, math.hypot(bx - x, by - y) / max(p.speed, 1e-3))
            pred = ball.predict(self._senses.t + eta, p.ball_decel)
            if pred is not None:
                bx, by = pred
                self._spot_ball = (bx, by)
        if p.board_push > 0.0 and self.bounds is not None \
                and min(self.bounds[0] - abs(bx), self.bounds[1] - abs(by)) < p.board_push:
            bp = self._board_push(bx, by)                    # the board push (12g): a walk along the wall, not a swing
            if bp is not None:
                return bp
        if self.goal is not None:
            u = math.atan2(self.goal[1] - by, self.goal[0] - bx)
            far = math.hypot(self.goal[0] - bx, self.goal[1] - by) > p.push_beyond
            if p.kick_cone > 0 and self.goal_cone(bx, by) < p.kick_cone:
                far = True                          # too fine a target from here: dribble it closer
        else:
            u, far = (self.attack if self.attack is not None else los), False
        # The detour: how far round the ball this duck must get to send it
        # along `u`. Past `aim_max`, `aim_mode` says what to do instead.
        detour = _wrap(u - los)
        if abs(detour) > p.aim_max:
            if p.aim_mode == "clamp":
                u = _wrap(los + math.copysign(p.aim_max, detour))
            elif p.aim_mode != "goal":
                u, far = los, False
        foot_sel: str | None = None
        if p.kick_select and not far and self.goal is not None and self.bounds is not None and self.goal_w > 0:
            chosen = self._select_kick_line(odom, (bx, by), los, u)
            if chosen is not None:
                u, foot_sel = chosen
                if foot_sel == "push":
                    far = True                                    # the selector chose to walk the ball
        if far:
            return bx - p.push_behind * math.cos(u), by - p.push_behind * math.sin(u), None, u, "push"
        rel = _wrap(math.atan2(by - y, bx - x) - u)
        foot = "kick_left" if rel >= 0 else "kick_right"
        if self.spot is not None and self.spot[2] in ("kick_left", "kick_right") and abs(rel) < 0.3:
            foot = self.spot[2]                                   # hysteresis: nearly on the line, keep the foot
        if foot_sel is not None:
            foot = foot_sel                                       # kick_select chose the foot for its exit angle
        if p.behind_ball > 0.0 and self.goal is not None and self._around is not None:
            # ALREADY COMMITTED. Do not ask the trigger again - it is a
            # sampler, and re-asking is what stopped the duck ever arriving.
            # Re-lay the staging spot against the ball's CURRENT place, so a
            # ball that drifts is still walked behind, and discharge the
            # commitment on the geometry (we are behind it now) or the clock.
            t_now = self._senses.t if self._senses is not None else self._around[2]
            gu = math.atan2(self.goal[1] - by, self.goal[0] - bx)
            behind = abs(_wrap(math.atan2(y - by, x - bx) - gu)) >= p.behind_ball_done
            if behind:
                self._around = None          # arrived: the trigger is free to fire again
            elif t_now - self._around[2] > p.behind_ball_s:
                # SPENT. Dropping the latch here used to fall straight into
                # the trigger block below, which re-took it with a fresh
                # timestamp on the same tick — so the budget never expired and
                # the walk-round it was written to bound was still unbounded
                # (probed: mode "around" at t=0, 1.5 and 60 s alike). Hold the
                # trigger off for as long again, so the duck plays the ball
                # where it lies before it may commit to another walk-round.
                self._around = None
                self._around_cool = t_now + p.behind_ball_s
            else:
                spot = self._behind_spot(bx, by, gu)
                if spot is not None:
                    return spot[0], spot[1], None, gu, "around"
                self._around = None          # the whole arc is blocked: a real corner
        if (p.behind_ball > 0.0 and self.goal is not None and self._around is None
                and (self._senses.t if self._senses is not None else 0.0) >= self._around_cool
                and ball.range >= p.behind_ball_far):
            # Would the best kick this stance allows actually gain ground?
            # Judged on where the ball LEAVES (`_kick_heading`: the aim line
            # plus the measured foot exit angle), not on the aim line, because
            # the exit angle is worth 24-29 deg and the clamp's edge line is
            # exactly where that decides the sign.
            gu = math.atan2(self.goal[1] - by, self.goal[0] - bx)
            want = (self._worth_going_round(odom, bx, by, gu) if p.behind_ball_rate > 0.0
                    else math.cos(_wrap(self._kick_heading(foot, u) - gu)) < p.behind_ball_cos)
            if want:
                spot = self._behind_spot(bx, by, gu)
                if spot is not None:
                    # Go round first, and COMMIT to it. Only a corner where
                    # the WHOLE arc is blocked falls through to the old plan
                    # rather than standing still: a bad touch beats no touch
                    # there, and nothing is committed to.
                    self._around = (bx, by, self._senses.t if self._senses is not None else 0.0)
                    return spot[0], spot[1], None, gu, "around"
        side = -p.kick_side if foot == "kick_left" else p.kick_side     # stand to the ball's other side
        # The body heading that sends the kick along u (the map's deflection
        # is in the body frame, so the spot is laid out in that heading too).
        h = _wrap(u - (p.kick_deflect_left if foot == "kick_left" else p.kick_deflect_right))
        sx, sy = (bx - p.kick_ahead * math.cos(h) - side * math.sin(h),
                  by - p.kick_ahead * math.sin(h) + side * math.cos(h))
        if p.board_margin > 0 and self.bounds is not None and not self._clear_of_boards(sx, sy):
            along = self._along_the_boards(bx, by)
            if along is not None:
                return along
        return sx, sy, foot, h, "kick"

    def _spot_clear(self, ball_xy, u: float, action: str, margin: float) -> bool:
        """Can the body stand where this candidate's spot would be laid?
        The spot is `_plan`'s own geometry for a kick (behind the ball on
        the line, `kick_side` to the foot's side, in the deflected heading)
        and a push's (`push_behind` squarely behind); True when it is at
        least `margin` inside every board, and always True off a pitch."""
        if self.bounds is None:
            return True
        p = self.p
        bx, by = ball_xy
        if action == "push":
            sx, sy = bx - p.push_behind * math.cos(u), by - p.push_behind * math.sin(u)
        else:
            side = -p.kick_side if action == "kick_left" else p.kick_side
            h = _wrap(u - (p.kick_deflect_left if action == "kick_left" else p.kick_deflect_right))
            sx, sy = (bx - p.kick_ahead * math.cos(h) - side * math.sin(h),
                      by - p.kick_ahead * math.sin(h) + side * math.cos(h))
        return self.bounds[0] - abs(sx) >= margin and self.bounds[1] - abs(sy) >= margin

    def _spot_body_clear(self, x: float, y: float) -> bool:
        """Is a laid spot clear of every board by the walking body's extent
        (`spot_reach`, or the measured 0.129 m when that is off)? What the
        line-up's own stop (`lineup_tof_stop`) is conditioned on."""
        if self.bounds is None:
            return True
        m = self.p.spot_reach if self.p.spot_reach > 0.0 else 0.129
        return self.bounds[0] - abs(x) >= m and self.bounds[1] - abs(y) >= m

    def _clear_of_boards(self, x: float, y: float) -> bool:
        """Is a spot far enough off the boards to stand on? True off a pitch
        (`bounds` is None on every world that is not one) - the guard lives
        here and not only in the caller, so a second caller cannot inherit a
        `NoneType is not subscriptable` from this one's homework."""
        if self.bounds is None:
            return True
        return (self.bounds[0] - abs(x) >= self.p.board_margin
                and self.bounds[1] - abs(y) >= self.p.board_margin)

    def _board_line(self, bx: float, by: float) -> float:
        """The line along the nearer wall a ball at the boards is played on:
        up the pitch on a side board, away from our own mouth on our end
        board, across the mouth on theirs. Shared by the kick rescue
        (`_along_the_boards`) and the board push (`_board_push`)."""
        att = 1.0 if (self.goal is None or self.goal[0] >= 0) else -1.0
        if self.bounds[1] - abs(by) < self.bounds[0] - abs(bx):
            return 0.0 if att > 0 else math.pi                        # the side board: up the pitch
        if bx * att < 0.0:
            # OUR end board. Sideways is the only line the body can reach
            # here, and toward the middle is an OWN GOAL: a ball on our own
            # goal line is already inside the scoring band (`_check_goal`
            # scores at |x| > hx - 0.08), so the moment it reaches the mouth
            # in y it is in our net. Measured on a 1v1 pitch: a ball at
            # (-1.45, +0.40) aimed at -90 deg is a goal against us after
            # 0.10 m of travel. Clear it AWAY from the mouth, toward the
            # corner - which is what a defender does with it.
            return math.pi / 2 if by >= 0 else -math.pi / 2
        return -math.pi / 2 if by >= 0 else math.pi / 2               # their end board: across the mouth is a chance

    def _board_push(self, bx: float, by: float) -> tuple[float, float, None, float, str] | None:
        """A PUSH for a ball at the boards (`board_push`, roadmap 12g): walk
        through it along the nearer wall (`_board_line`) from `push_behind`
        behind it, from a spot the body can stand on. A ball nearer the wall
        than the body's extent gets the line tilted INTO the wall, by the
        smallest angle up to `board_push_tilt` that moves the spot out to a
        body-clear gap - the wall keeps the ball rolling along itself. None
        when no tilt does (the caller lays its kick plan as before)."""
        p = self.p
        u = self._board_line(bx, by)
        # The unit normal INTO the nearer wall.
        if self.bounds[1] - abs(by) < self.bounds[0] - abs(bx):
            nx, ny = 0.0, (1.0 if by >= 0 else -1.0)
        else:
            nx, ny = (1.0 if bx >= 0 else -1.0), 0.0
        steps = max(1, int(round(math.degrees(p.board_push_tilt) / 5.0)))
        for k in range(steps + 1):
            th = p.board_push_tilt * k / steps
            dx = math.cos(u) * math.cos(th) + nx * math.sin(th)
            dy = math.sin(u) * math.cos(th) + ny * math.sin(th)
            sx, sy = bx - p.push_behind * dx, by - p.push_behind * dy
            if self._spot_body_clear(sx, sy):
                return sx, sy, None, math.atan2(dy, dx), "push"
        return None

    def _along_the_boards(self, bx: float, by: float) -> tuple[float, float, str, float, str] | None:
        """A kick spot for a ball at the boards (`board_margin`): the line
        along the nearer wall - up the pitch on a side board, toward the
        middle on an end board - with whichever foot puts the body on the
        open side. None if neither foot's spot is clear (a tight corner)."""
        p = self.p
        u = self._board_line(bx, by)
        for foot in ("kick_left", "kick_right"):
            side = -p.kick_side if foot == "kick_left" else p.kick_side
            h = _wrap(u - (p.kick_deflect_left if foot == "kick_left" else p.kick_deflect_right))
            x = bx - p.kick_ahead * math.cos(h) - side * math.sin(h)
            y = by - p.kick_ahead * math.sin(h) + side * math.cos(h)
            if self._clear_of_boards(x, y):
                return x, y, foot, h, "kick"
        return None

    def _board_ball(self, t: float) -> tuple[tuple[float, float] | None, float]:
        """The freshest ball sighting on the team board, and its age. A duck
        that cannot see the ball itself is usually standing off while a
        teammate is on it, and the board is the only way it learns the ball
        is coming at all — the camera loses a floor ball at 0.3 m and, over
        the threat battery, the best-placed defender had a fresh detection on
        22% of the ticks and any live track on 43%."""
        if self.team is None:
            return None, math.inf
        best = None
        for c in self.team.claims.values():
            if c.ball is not None and (best is None or c.t > best.t):
                best = c
        return (None, math.inf) if best is None else (best.ball, t - best.t)

    def _block_target(self, odom, ball, seen: bool, fresh: bool, t: float) -> tuple[float, float] | None:
        """Where to stand to block a ball rolling at our own goal, or None to
        carry on playing. The decision is `brain/intercept.py`'s; this only
        assembles what it needs out of what this duck actually has."""
        p = self.p
        if p.intercept_eta <= 0 or self.goal is None:
            return None
        mine = self._ball_xy(odom, ball) if seen else None
        board, age = self._board_ball(t)
        # The trigger is differenced from SIGHTINGS: my own when it is fresh,
        # else the board's, which is a teammate's. (The board cannot say how
        # old the sighting UNDER a claim is — a claim is re-stamped every tick
        # — so a teammate coasting a track can feed one stale point in. The
        # 0.6 s difference window is what stops a single one from firing it.)
        sight = mine if fresh else (board if age <= p.intercept_age else None)
        return self.blocker.update(t, p, mine if mine is not None else board, sight,
                                   self._own_goal(odom), odom, self.duck_id,
                                   self.team.mates(self.duck_id, t) if self.team is not None else [])

    def _worth_going_round(self, odom, bx: float, by: float, gu: float) -> bool:
        """Is walking round worth what it costs? `kick_select`'s own rollouts
        on both sides of the trade — the verdict it just gave from here, and
        the verdict from behind the ball on the goal line — with the walk's
        seconds priced at `behind_ball_rate`. See the knob for the algebra.

        False (do not go round) whenever the selector was not consulted or
        gave nothing: without a verdict there is no trade to evaluate, and
        the old `behind_ball_cos` test has already had its say."""
        from .kickselect import KickModel, Pitch, evaluate  # noqa: PLC0415

        p = self.p
        now = self.last_select
        if now is None or self.bounds is None or self.goal is None:
            return False
        if self._kick_rng is None:
            import zlib  # noqa: PLC0415
            self._kick_rng = np.random.default_rng(zlib.crc32(self.duck_id.encode() or b"duck"))
        model = KickModel(speed=p.kick_speed, speed_sd=p.kick_select_v_sd, dir_sd=p.kick_select_dir_sd,
                          decel=max(p.ball_decel, 0.02), exit_left=p.kick_exit_left,
                          exit_right=p.kick_exit_right, p_whiff=p.kick_select_p_whiff)
        pitch = Pitch(self.bounds[0], self.bounds[1], self.goal_w, 1.0 if self.goal[0] >= 0 else -1.0)
        after = [evaluate((bx, by), gu, foot, model, pitch, self._kick_rng, p.kick_select_n)
                 for foot in ("kick_left", "kick_right")]
        after = [v for v in after if v.p_own <= p.kick_select_t_own]
        if not after:
            return False                       # from behind it is STILL an own-goal risk: not worth the walk
        def score(v) -> float:
            return v.p_goal * p.behind_ball_goal_w + v.value
        gain = max(score(v) for v in after) - score(now)
        # The walk: the arc from where the duck stands round to the far side.
        arc = abs(_wrap(math.atan2(odom[1] - by, odom[0] - bx) - _wrap(gu + math.pi)))
        secs = arc * p.behind_ball / max(p.speed, 1e-3)
        return gain > p.behind_ball_rate * secs

    def _behind_spot(self, bx: float, by: float, gu: float) -> tuple[float, float, float] | None:
        """Where to stage for a walk-round: `behind_ball` from the ball, as
        near the far side of the ball-to-goal line as the BOARDS allow.

        The ideal is squarely behind (180 deg round from `gu`). When the body
        cannot stand there the spot SLIDES along the arc — +-`behind_ball_arc`
        in `behind_ball_step` steps, nearer offsets first — instead of the
        objective being abandoned, which is what the first cut did: measured
        over 180 s a side, "staging spot in a board" released 6 of 8
        walk-rounds on the 1v1 pitch (1.50 x 1.25 m playable) and 4 of 10 on
        the 2v2, and every one of those was a duck giving up on getting
        behind the ball because ONE point happened to be unreachable.

        Returns (x, y, angle round from `gu`) or None when the whole arc is
        blocked — a genuine corner, where the old plan is the better answer.
        """
        p = self.p
        lim = p.behind_ball_arc
        step = max(p.behind_ball_step, 1e-3)
        offs = [0.0]
        k = 1
        while k * step <= lim + 1e-9:
            offs += [k * step, -k * step]
            k += 1
        for off in offs:
            a = _wrap(gu + math.pi + off)                  # 180 deg round, then slid
            x, y = bx + p.behind_ball * math.cos(a), by + p.behind_ball * math.sin(a)
            if self._spot_body_clear(x, y):
                return x, y, abs(_wrap(a - gu))
        return None

    def _keep_off(self, odom, target, ball, r: float) -> tuple[float, float]:
        """`target`, bent round `ball` when the straight line to it would pass
        NEARER the ball than the target itself does (`approach_keepout`).

        The effective radius is `min(r, |target - ball|)`: the target is then
        always on or outside the circle, so the keep-out can never fence the
        duck off from where it is going - a kick spot 0.10 m from the ball
        keeps its own 0.10 m, which still stops the walk cutting the corner
        through the ball to reach it.

        The waypoint is the TANGENT from where the duck stands to that
        circle, taken on the side the target already lies - the shorter way
        round, and the one `duel_side` measured: a rule that sends the duck
        the far way steers the walk across the ball instead of round it.
        Inside the circle already, the waypoint is straight out."""
        px, py = float(odom[0]), float(odom[1])
        bx, by = float(ball[0]), float(ball[1])
        dx, dy = target[0] - px, target[1] - py
        r = min(r, math.hypot(target[0] - bx, target[1] - by))
        if r <= 1e-6:
            return target
        l2 = dx * dx + dy * dy
        if l2 < 1e-9:
            return target
        f = min(1.0, max(0.0, ((bx - px) * dx + (by - py) * dy) / l2))
        if math.hypot(bx - (px + f * dx), by - (py + f * dy)) >= r - 1e-9:
            return target                                   # the line is already clear of it
        d = math.hypot(bx - px, by - py)
        if d <= r:
            a = math.atan2(py - by, px - bx) if d > 1e-6 else math.atan2(dy, dx)
            return bx + r * math.cos(a), by + r * math.sin(a)
        base = math.atan2(by - py, bx - px)
        # Which way round. The cross product says which side the target lies,
        # but it is ZERO when duck, ball and target are collinear - which is
        # exactly the stance `behind_ball` stages for, so a 2 mm jitter in the
        # tracked ball flipped the side (and swung the servo ~60 deg) every
        # tick and the duck weaved in place instead of walking round. Inside
        # that band both ways are equal in length, so break the tie on the way
        # the duck is ALREADY turned, which does not jitter.
        cross = (bx - px) * dy - (by - py) * dx
        scale = math.hypot(bx - px, by - py) * math.hypot(dx, dy)
        if abs(cross) > 1e-2 * max(scale, 1e-9):
            side = 1.0 if cross > 0.0 else -1.0     # the target is plainly one side
        else:
            # Collinear, so both ways round are the same length and no
            # instantaneous rule can be stable: every quantity here (the
            # bearing to the ball included) turns on the ball's own lateral
            # position, which is what jitters. So HOLD the side already being
            # walked, and let a clear cross be the only thing that re-decides.
            side = self._keep_side or (1.0 if _wrap(base - float(odom[2])) < 0.0 else -1.0)
        self._keep_side = side
        a = base + side * math.asin(min(r / d, 1.0))
        t = math.sqrt(max(d * d - r * r, 0.0))
        return px + t * math.cos(a), py + t * math.sin(a)

    def _servo(self, odom, target, cold, stop: float, slow_in: float = 0.2,
               avoid: tuple[float, float] | None = None, avoid_r: float = 0.0) -> tuple[float, float, float, float]:
        """(vx, wz, dist, bearing) toward a point: turn in place first when
        it is well off the nose, walk with steering otherwise, stop inside.

        `dist` AND the returned `bearing` are always to the TARGET, never to a
        keep-out waypoint - the caller stops and settles on those numbers, and
        a duck that thought it had arrived because a waypoint was close would
        swing at a ball it is still half a metre from. The waypoint steers the
        wheels (`steer` below) and nothing else: the two-stage back-off asks
        "is the pre-spot behind me with the ball at my feet?", and handed a
        waypoint bearing it answered about the detour instead - skipping the
        back-off on a stance that needed it, and firing one on a line-up that
        was fine, every tick, which is a retreat loop rather than a one-off."""
        p = self.p
        dx, dy = target[0] - odom[0], target[1] - odom[1]
        dist = math.hypot(dx, dy)
        bearing = _wrap(math.atan2(dy, dx) - odom[2])
        sx, sy = dx, dy
        if avoid is not None and avoid_r > 0.0:
            way = self._keep_off(odom, target, avoid, avoid_r)
            sx, sy = way[0] - odom[0], way[1] - odom[1]
        steer = _wrap(math.atan2(sy, sx) - odom[2])
        if dist <= stop:
            return 0.0, 0.0, dist, bearing
        if abs(steer) > 0.5 and dist > 0.08:
            vx, _, wz = turn(steer, cold)
            return vx, wz, dist, bearing
        return (0.25 if dist < slow_in else p.speed), clip_wz(p.k_turn * steer), dist, bearing

    def chase_aim(self, odom, bx: float, by: float, bearing: float) -> float:
        """Where to AIM the chase: the ball, or a point `chase_behind` behind
        it on the ball-to-goal line, so the run-in arrives on the side the
        kick wants. The gaze and every range test stay on the BALL - only the
        steering moves. Returns `bearing` unchanged when the bias is off or
        does not apply to this duck.

        Public and taking a ball POSITION so a test can call the running code
        instead of restating it: the mirror that used to stand in for this in
        `tests/test_behind_ball.py` had already drifted, missing both the
        attacker gate and the `cos` scaling below - the very terms whose
        absence is recorded here as measured harm.
        """
        p = self.p
        if not (p.chase_behind > 0.0 and self.goal is not None
                and not (p.chase_behind_attacker and self.role != "attack")):
            return bearing
        if p.chase_behind_upto < 1.0 and self._attack_x(bx) > p.chase_behind_upto:
            return bearing                     # up-pitch of the gate: ordinary chase
        gu = math.atan2(self.goal[1] - by, self.goal[0] - bx)
        # SCALED BY HOW WRONG THE SIDE ACTUALLY IS: the full offset when the
        # duck is between the ball and the goal, nothing at all when it is
        # already behind it. Unscaled, the bias fired on every approach
        # including the good ones and MEASURED HARM there - one duck, one
        # ball, the kick gym's own right-side placement, 8 seeds x 30
        # episodes: swings 211 -> 166 and whiff 5% -> 12% (p = 0.016). A duck
        # already on the right side is steered off a line-up it had.
        ang = abs(_wrap(math.atan2(odom[1] - by, odom[0] - bx) - gu))
        back = p.chase_behind * 0.5 * (1.0 + math.cos(ang))
        ax = bx - back * math.cos(gu)
        ay = by - back * math.sin(gu)
        if p.approach_keepout > 0.0:
            # A duck that is already up-pitch aims at a point on the FAR side
            # of the ball, so the run-in crosses it. Same tangent router as
            # the line-up's.
            ax, ay = self._keep_off(odom, (ax, ay), (bx, by), p.approach_keepout)
        return _wrap(math.atan2(ay - odom[1], ax - odom[0]) - odom[2])

    def _lineup_wrong_side(self, odom, ball) -> bool:
        """Would a kick line-up from here be a run-in from the goal side of
        the ball, in the third `chase_behind` is allowed to act?

        That is the stance that walks own goals in: the duck is between the
        ball and the goal it attacks, close enough that `_plan` is about to
        lay a kick spot 0.10 m from the ball, and the arrival is the contact.
        `chase_behind` is supposed to have already put the duck on the far
        side before line-up starts; this keeps line-up from firing first.

        False (line up as usual) when the bias is off, off a pitch, up-pitch
        of `chase_behind_upto`, or the duck is already behind the ball.
        """
        p = self.p
        if not (p.chase_behind > 0.0 and self.goal is not None and self.bounds is not None):
            return False
        bx, by = self._ball_xy(odom, ball)
        if p.chase_behind_upto < 1.0 and self._attack_x(bx) > p.chase_behind_upto:
            return False
        gu = math.atan2(self.goal[1] - by, self.goal[0] - bx)
        ang = abs(_wrap(math.atan2(odom[1] - by, odom[0] - bx) - gu))
        return math.cos(ang) > 0.0

    def _on_the_line(self, odom, spot, u: float, heading_err: float) -> bool:
        """Is the duck already where stage one is trying to put it — on the
        kick line, squared up, still short of the spot? Then stage two can
        start here (`lineup_lat` > 0; 0 sends every line-up via the pre-spot,
        which is how `two_stage` was first measured).

        Only INSIDE the pre-spot's own distance: further back the walk to the
        pre-spot runs at `speed`, which beats the walk-in's `approach_speed`,
        and the ball is far enough that the square-up there is free."""
        p = self.p
        if p.lineup_lat <= 0.0:
            return False
        along = (spot[0] - odom[0]) * math.cos(u) + (spot[1] - odom[1]) * math.sin(u)
        lat = -(odom[0] - spot[0]) * math.sin(u) + (odom[1] - spot[1]) * math.cos(u)
        return (p.lineup_tol < along <= p.approach_back and abs(lat) <= p.lineup_lat
                and abs(heading_err) <= p.aim_tol)

    # -- the machine ----------------------------------------------------------
    # THE PRIORITY (roadmap Track 4 s6 F.1). `step` is one flat machine, and
    # its `elif` chain IS the behaviour's priority scheme: the first branch
    # whose condition holds owns the tick. Written down here, in the order
    # the chain runs, and locked by a test that reads the chain back out of
    # the source - so a branch moved by accident is a failing test, not a
    # battery three weeks later.
    #   kick     a kick skill is running: the reflex tier owns the body
    #   look     the look after a kick, for the ball ahead
    #   retreat  backing out of a contact (stuck_s)
    #   avoid    a duck too near and ahead: turn away, never into it
    #   block    the ball is heading for OUR mouth and I am the one to stand in it
    #   support  the board says a teammate has the ball (a kickoff wait counts)
    #   yield    a clearly nearer duck is on the ball: stand off it
    #   push     a walk through the ball, until push_s runs out
    #   lineup / settle   on the line-up with a spot: the two-stage line-up and the swing
    #   seen     the ball is in the track: lineup / turn / chase toward it
    #   hunt     the ball rolled off: follow the kick line
    #   seek     walk to where it last was
    #   search   nothing seen: circle, sweep the head
    PRIORITY = ("kick", "look", "retreat", "avoid", "block", "support", "yield", "push",
                "lineup", "seen", "hunt", "seek", "search")

    def step(self, senses: Senses) -> Intent:
        # This method mutates from its next few lines on -- `gait.update`, the
        # localiser, every counter below -- so being called twice for one tick
        # double-advances all of it. A probe that did exactly that (wrapping
        # `_hold_target` and calling the real method again to compare) reported
        # a 7.56% firing rate a direct sweep showed was zero. The harnesses get
        # this right; only an instrument gets it wrong, which is why it warns
        # rather than raises, and once per brain rather than once a tick.
        if (self._last_step_t is not None and senses.t == self._last_step_t
                and not self._reentry_warned):
            self._reentry_warned = True
            warnings.warn(
                f"{type(self).__name__}.step() called twice at t={senses.t:.4f}. It mutates "
                "state on entry, so the second call double-advances the gait, the localiser "
                "and every counter -- whatever it returns is not a measurement of either "
                "variant. Step once and read the flags afterwards.",
                ReentrantStepWarning, stacklevel=2)
        self._last_step_t = senses.t
        self._senses = senses
        p = self.p
        t = senses.t
        cold = self.gait.update(senses)
        odom = senses.odom or (0.0, 0.0, 0.0)
        if self.loc is not None and senses.odom is not None:
            odom = self.loc.update(senses.odom, senses.det)     # localised: the pose everything below steers by
        if self.attack is None and senses.odom is not None:
            self.attack = odom[2]                  # placed facing the goal it attacks (make_pitch does)
        det_in = senses.fresh_det(self.DET_MAX_AGE)
        self.tof_ball: tuple[float, float] | None = None
        blob_ok = not p.tof_ball_lineup or (self.state in ("lineup", "settle") and not self._beside(t))
        if p.tof_ball_m > 0 and blob_ok and (det_in is None or not any(d.cls == "ball" for d in det_in.detections)):
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
        self.predicted_sigma: float | None = None            # its 1-sigma error (roadmap C.1)
        pred_bearing: float | None = None
        self.ball_resting = (p.rest_predict_s > 0.0 and ball is not None
                             and ball.at_rest(self.tracker.p.rest_vel))
        horizon = max(p.predict_s, p.rest_predict_s) if self.ball_resting else p.predict_s
        if ball is not None and ball.xy is not None and ball.age(t) <= horizon:
            px, py = ball.predict(t, p.ball_decel)
            # The sigma needs NOTHING special for a resting ball, and the
            # first draft here that gave it a smaller velocity prior was a
            # no-op dressed as a fix. `Track.sigma` already switches from the
            # generic `vel_prior` to the track's OWN measured velocity scatter
            # once the fix is older than `vel_sig_after_s` (1 s) - and a ball
            # that has been seen twice standing still has a tiny scatter, so
            # its uncertainty stays usable at four seconds by construction.
            # `at_rest` requires those same two hits, so every resting ball is
            # already on that branch. Measured: at dt = 4 s the two priors
            # give the identical sigma.
            self.predicted_sigma = ball.sigma(t, self.tracker.p.vel_prior, self.tracker.p.vel_sig_after_s)
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
                            self._ball_xy(odom, ball) if seen else None, (odom[0], odom[1], odom[2]),
                            ball.sigma(t, self.tracker.p.vel_prior, self.tracker.p.vel_sig_after_s)
                            if seen else math.nan)                                      # how surely (C.3)
            self.role = self.team.role(self.duck_id, t)
            # The GameController (roadmap B.3): the other side's kickoff -
            # we all support, in our own half, until the ball leaves the spot.
            self._kickoff_wait = bool(p.kickoff_wait and self.team.waits(
                t, self._ball_xy(odom, ball) if seen else None))
            if self._kickoff_wait:
                self.role = "support"
            for _, (mx, my, _) in self.team.mates(self.duck_id, t):
                self._mates.append((math.hypot(mx - odom[0], my - odom[1]),
                                    _wrap(math.atan2(my - odom[1], mx - odom[0]) - odom[2])))
        other = self.tracker.best("duck", t, min_hits=1)
        if p.rest_predict_s > 0.0 and ball is not None and ball.xy is not None:
            # Anything we know the position of, that is standing on the ball we
            # are remembering, voids the memory until we look again: our own
            # bump, a teammate off the board, a duck we can see.
            near = [c for _, c in (self.team.mates(self.duck_id, t) if self.team is not None else [])]
            if other is not None and other.xy is not None and other.age(t) <= p.lost_s:
                near.append((other.xy[0], other.xy[1], 0.0))
            if senses.bumped:
                near.append((odom[0], odom[1], 0.0))
            for mx, my, _ in near:
                if math.dist((mx, my), ball.xy) <= p.rest_clear_m:
                    self.tracker.disturb(p.target_cls, ball.xy, p.rest_clear_m)
                    break
        # The nearest duck ahead to avoid: a seen one, or a teammate by the board.
        threats = [(r, b) for r, b in self._mates if r < p.mate_keepout and abs(b) < p.duck_bearing]
        if other is not None and other.age(t) <= 0.6 and abs(other.bearing) < p.duck_bearing:
            # With the colour sense on, an OPPONENT gets its own keep-out:
            # a teammate is on the board (which knows who is quicker) and a
            # stranger is not, so they are not the same obstacle.
            keep = (p.opp_keepout if (p.use_color and p.opp_keepout > 0 and not self._is_mate(other))
                    else p.duck_keepout)
            if p.lineup_keepout > 0 and self.spot is not None and self.spot[4] == "kick" \
                    and seen and ball is not None and ball.range < other.range:
                # The duel's first form (measured off). Gated on the SPOT - the
                # kick line-up this tick is executing - and not on `self.state`,
                # which at this point in `step` is still the label the PREVIOUS
                # tick's elif chain wrote, and so answers for the wrong tick.
                keep = min(keep, p.lineup_keepout)
            if other.range < keep:
                threats.append((other.range, other.bearing))
        duck_rb = min(threats) if threats else None
        near_duck = duck_rb is not None
        # THE CONTEST (survey C.4's second half, bead mdl-23b). 263 of 508
        # 3v3 line-ups die in `avoid` inside 0.4 s with an opponent 0.33 m
        # ahead: both ducks turn away from each other, both drop their spot,
        # and neither gets the ball. Two GEOMETRIC answers to that are already
        # measured null - `lineup_keepout` (shrink the radius on a line-up)
        # and `opp_keepout` (widen it for an opponent) - because a radius
        # cannot break a symmetry. This one is not a radius: it asks WHO
        # SHOULD HAVE THE BALL, and only the duck that is nearer holds its
        # line. The other still avoids, so the deadlock breaks instead of
        # becoming a shoving match.
        #
        # It needs to know the other duck is an OPPONENT, and that is ENFORCED
        # on `use_color` and not merely documented: with the colour sense off,
        # `_is_mate` answers False for everybody (unknown counts as a
        # stranger, which is the safe reading everywhere else), so without the
        # gate this would contest its own teammates. A test caught exactly
        # that. It is why the rule was parked until the colour vote was
        # measured at contact range (95%
        # right inside 0.50 m, 100% coverage, the dangerous error on 1.4% of
        # ticks - `scripts/probe_duck_color.py`). Touching still stands the
        # duck up safely: this only declines to TURN AWAY, it never walks
        # into anybody.
        self.contesting = False
        if (p.contest_margin > 0.0 and p.use_color and near_duck and other is not None
                and other.xy is not None and other.age(t) <= p.lost_s
                and not self._is_mate(other) and ball is not None and ball.xy is not None):
            mine = math.dist((odom[0], odom[1]), ball.xy)
            theirs = math.dist(other.xy, ball.xy)
            self.contesting = mine + p.contest_margin < theirs
        if self.contesting and duck_rb is not None and duck_rb[0] >= p.duck_touch:
            near_duck = False                      # hold the line; `avoid` still owns a touch
        # THE DUEL (`duel`, roadmap C.4's second half): the OTHER side of the
        # same geometry - an opponent inside `duel_near` of the ball and
        # NEARER to it than this duck. Read off `_opponents`, which is the
        # honest sense with the colour vote off (a duck track the board does
        # not own), and off the ball this duck can actually see: the
        # reachable set is what the brain perceives, not what the world
        # knows (probe_duel: 7.44% against the world's 9.35%). Computed
        # here, acted on in the elif chain below.
        self.dueling = False
        self.duel_spot: tuple[float, float] | None = None
        if p.duel > 0.0 and seen and ball is not None and ball.xy is not None:
            bxy = self._ball_xy(odom, ball)
            mine = math.hypot(bxy[0] - odom[0], bxy[1] - odom[1])
            theirs = min((math.hypot(bxy[0] - ox, bxy[1] - oy)
                          for ox, oy in self._opponents(t)), default=math.inf)
            if theirs <= p.duel_near and theirs < mine:
                gx, gy = self._own_goal(odom)
                u = math.atan2(gy - bxy[1], gx - bxy[0])       # from the ball toward OUR goal
                sx, sy = bxy[0] + p.duel * math.cos(u), bxy[1] + p.duel * math.sin(u)
                if p.duel_side > 0.0:
                    # ...offset off that line, on the side of it the duck is
                    # ALREADY on, so the walk to the spot goes ROUND the ball
                    # instead of through it (see the knob).
                    lat = -(odom[0] - bxy[0]) * math.sin(u) + (odom[1] - bxy[1]) * math.cos(u)
                    s = p.duel_side if lat >= 0.0 else -p.duel_side
                    sx, sy = sx - s * math.sin(u), sy + s * math.cos(u)
                if self.bounds is not None:                     # never a spot in the boards
                    m = p.support_margin
                    sx = float(np.clip(sx, -self.bounds[0] + m, self.bounds[0] - m))
                    sy = float(np.clip(sy, -self.bounds[1] + m, self.bounds[1] - m))
                self.dueling, self.duel_spot = True, (sx, sy)
        if self.dueling and duck_rb is not None and duck_rb[0] >= p.duck_touch:
            near_duck = False                      # walking to the block spot is not a flinch; `avoid` still owns a touch
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
        vy = 0.0                                             # the crab: only the last centimetres of a line-up use it
        gaze_yaw = 0.0
        look_yaw: float | None = None                       # the post-kick sweep (look_sweep)
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
                and (t - self.memory[2] > max(p.seek_s, p.head_memory_s)
                     or (p.seek_s > 0.0
                         and math.hypot(self.memory[0] - odom[0], self.memory[1] - odom[1]) <= p.seek_min))):
            self.memory = None                                  # stale, or here with nothing seen: forget it
        seeking = p.seek_s > 0 and not seen and not hunting and self.memory is not None
        if seeking and (ahead < p.hunt_stop or self._beside(t)):
            seeking = False                                     # something in the way: circle here instead
            self.memory = None
        # Is the ball rolling into OUR goal, and am I the one to stand in
        # front of it? (brain/intercept.py; `intercept_eta` = 0 ships it off.)
        block_at = self._block_target(odom, ball, seen, fresh, t)
        if senses.skill is not None:
            vx, wz = 0.0, 0.0                                   # the kick owns the reflex tier
            self.state = "kick"
        elif looking:
            vx, wz = 0.0, 0.0
            gaze_at = p.look_aim_range if p.look_aim else p.look_range
            if p.look_sweep > 0.0:
                gaze_at = p.look_sweep_range
                centre = _wrap(self._hunt_u - odom[2]) if self._hunt_u is not None else 0.0
                phase = (t - self._look_t0) / max(p.look_s, 1e-6)
                look_yaw = centre + p.look_sweep * math.sin(2.0 * math.pi * phase)
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
            if self._around is not None:
                # An `avoid` PAUSES a committed walk-round, it does not spend
                # it: the budget is for walking, and 7 of the first 14
                # walk-rounds died because standing off a teammate ate it.
                # The veto over the body is untouched - only the clock moves.
                self._around = (self._around[0], self._around[1], self._around[2] + CTRL_DT)
        elif block_at is not None:
            # Leave the play and get in the way. Not a line-up: the servo
            # faces where it WALKS, and only once it is on the line does the
            # duck square up on the ball — so the body ends across the path
            # with the camera on the thing it is stopping, and no square-up
            # ever happens next to the ball (which is what turns a line-up
            # into a shove).
            self.spot = None
            vx, wz, bdist, _ = self._servo(odom, block_at, cold, p.intercept_tol)
            if bdist <= p.intercept_tol:
                b = self.blocker.ball
                bb = 0.0 if b is None else _wrap(math.atan2(b[1] - odom[1], b[0] - odom[0]) - odom[2])
                vx, wz = (0.0, 0.0) if abs(bb) < 0.3 else turn(bb, cold)[::2]
                if wz != 0.0 and self._beside(t):
                    vx, wz = 0.0, 0.0                  # a body beside us: never a turn in place
            self.state = "block"
        elif self.role == "support":
            vx, wz = self._support(odom, ball, seen, cold)
            if p.support_gaze and fresh and ball is not None and ball.range < p.head_range \
                    and abs(ball.bearing) < p.gaze_bearing_max:
                gaze_at = ball.range                        # a supporter that looks at the ball (support_gaze)
        elif self.dueling and self.state != "settle":
            # Stand `duel` metres goal-side of the ball, facing it. Not a
            # line-up: the servo faces where it WALKS, and only on arrival
            # does the duck square up on the ball, so the body ends between
            # the ball and our goal with the camera on it — and no square-up
            # ever happens next to the ball, which is what turns a line-up
            # into a shove (the `block` branch above, same reasoning).
            self.spot = None
            vx, wz, ddist, _ = self._servo(odom, self.duel_spot, cold, p.intercept_tol)
            if ddist <= p.intercept_tol:
                bb = _wrap(math.atan2(self._ball_xy(odom, ball)[1] - odom[1],
                                      self._ball_xy(odom, ball)[0] - odom[0]) - odom[2])
                vx, wz = (0.0, 0.0) if abs(bb) < 0.3 else turn(bb, cold)[::2]
                if wz != 0.0 and self._beside(t):
                    vx, wz = 0.0, 0.0              # a body beside us: never a turn in place
            if fresh and ball.range < p.head_range and abs(ball.bearing) < p.gaze_bearing_max:
                gaze_at = ball.range
            self.state = "duel"
        elif yielding and self.state not in ("settle",):
            vx, wz = 0.0, 0.0
            self.spot = None
            self.state = "yield"
        elif self.state == "push":
            _, _, _, u, _ = self.spot
            vx, wz = p.push_speed, clip_wz(p.k_turn * _wrap(u - odom[2]))
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
            if mode == "kick" and p.two_stage and not self.lined \
                    and self._on_the_line(odom, (sx, sy), u, heading_err):
                self.lined = True                               # nothing to go back for
                self.t_state = t                                # stage two gets its own clock
            if mode == "kick" and p.two_stage and not self.lined:
                # Stage one: the pre-spot behind the kick spot on the line;
                # square up there, where a turn in place cannot touch the ball.
                px, py = sx - p.approach_back * math.cos(u), sy - p.approach_back * math.sin(u)
                vx, wz, pdist, bearing = self._servo(odom, (px, py), cold, p.approach_tol,
                                                     avoid=self._spot_ball, avoid_r=p.approach_keepout)
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
                # A staging spot ("around") is reached, not settled on: it
                # borrows the push's looser tolerance rather than a kick's.
                tol = p.push_tol if (mode in ("push", "around") and p.push_tol > 0.0) else p.lineup_tol
                vx, wz, dist, bearing = self._servo(odom, (sx, sy), cold, tol,
                                                    avoid=self._spot_ball, avoid_r=p.approach_keepout)
                if p.lineup_square > 0.0 and mode == "kick" and not p.two_stage and self.state != "settle" \
                        and tol < dist <= p.lineup_square:
                    # The last centimetres (`lineup_square`): square to the
                    # heading first, then walk straight along it onto the spot.
                    if abs(heading_err) > p.aim_tol:
                        vx, _, wz = turn(heading_err, cold)
                    else:
                        # Squared: close the spot as a HOLONOMIC error in the
                        # heading frame - forward along it, a crab across it
                        # (the walker trained on lateral commands; the omni
                        # bucket), the heading held by `k_head`. A straight
                        # walk cannot reach a spot that is beside the duck,
                        # which after a turn in place it usually is (12ao's
                        # first cut: squared to 19 deg, stuck 7 cm off, 76
                        # timeouts of 97).
                        along = (sx - odom[0]) * math.cos(odom[2]) + (sy - odom[1]) * math.sin(odom[2])
                        lat = -(sx - odom[0]) * math.sin(odom[2]) + (sy - odom[1]) * math.cos(odom[2])   # +: the spot is to the LEFT
                        # A FIXED-SPEED vector at the spot, not a proportional
                        # one: the walker trained on forward commands clamped
                        # at 0.3 and a 0.1 m/s ask moves nothing (12ao's second
                        # cut: squared to 23 deg, still 5 cm off, 74 timeouts).
                        n = math.hypot(along, lat)
                        vx = float(p.square_speed * along / n) if n > 1e-6 else 0.0
                        vy = float(p.square_speed * lat / n) if n > 1e-6 else 0.0
                        wz = float(np.clip(p.k_head * heading_err, -p.approach_wz, p.approach_wz))
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
            tol = p.push_tol if (mode in ("push", "around") and p.push_tol > 0.0) else p.lineup_tol
            on_spot = dist <= tol + (0.03 if settling else 0.0)
            aim = p.push_aim_tol if (mode == "push" and p.push_aim_tol > 0.0) else p.aim_tol   # a push's own aim
            squared = abs(heading_err) <= aim + (0.15 if settling else 0.0)
            if self.spot is None:
                pass                                            # backing off (above)
            elif mode == "around":
                # In position behind the ball: drop the staging spot so the
                # next tick lays a real kick line from the good side. Before
                # that it is a plain walk - no square-up (the heading it will
                # want is not known until the line is laid) and no settle.
                if on_spot:
                    self.spot = None
                    self._around = None                     # the walk-round is done with
                    self.state = "chase"
                    self.t_state = t
                elif self.state == "lineup" and t - self.t_state > p.lineup_s:
                    # A staging spot gets the SAME give-up as a kick line-up.
                    # This arm sits above the general `lineup_s` branch in the
                    # chain, so without its own copy an `around` walk owned
                    # the tick and nothing timed it out: a spot the duck could
                    # not close on (a teammate standing there, a spot inside
                    # the board clearance) held it until an unrelated branch
                    # happened to break in. The latch goes with the spot, on
                    # cooldown, so the next tick does not simply re-commit.
                    self.spot = None
                    self._around = None
                    self._around_cool = t + p.behind_ball_s
                    self.state = "search"
                    vx, _, wz = turn(1.0, cold)
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
                        self.tracker.disturb(p.target_cls)     # walking through it moves it too
                        self.state = "push"
                        self.t_state = t
                        vx, wz = p.push_speed, 0.0
                    elif self._too_wide(odom) or self._too_far(odom):
                        # The geometry says this one misses. Drop the spot
                        # and walk it again rather than spend a touch on a
                        # shot already 20-plus degrees wide.
                        self.declines += 1
                        self.spot = None
                        self.state = "chase"
                        self.t_state = t
                    else:
                        skill = foot
                        self._last_foot = foot
                        self.kicks += 1
                        self.tracker.disturb(p.target_cls)     # we just hit it: the memory is void
                        self.spot = None
                        self.state = "kick"
                        # Where the ball is going — which is NOT `u`, the line
                        # it was aimed along: the kick leaves the foot at an
                        # angle to the body (`kick_exit_*`, measured in play).
                        heading = self._kick_heading(foot, u)
                        self._hunt_u = heading
                        if self.team is not None:
                            origin = (self._ball_xy(odom, ball) if seen
                                      else (sx + p.kick_ahead * math.cos(u),
                                            sy + p.kick_ahead * math.sin(u)))
                            self.team.publish_kick(t, origin, heading, p.kick_speed)
            elif self.state == "lineup" and t - self.t_state > p.lineup_s:
                self.spot = None
                self.state = "search"
                vx, _, wz = turn(1.0, cold)
            else:
                self.state = "lineup"
                if vx > 0 and fresh and ball.range < p.head_range and abs(ball.bearing) < p.gaze_bearing_max:
                    gaze_at = ball.range                  # the walking gaze, inside the same bearing window as the line-up's
            # …and keep looking at it through the settle and the square-up,
            # which is where the swing is decided and where the old gate
            # (`vx > 0`, below) dropped the head. Aimed at the ball's last
            # PLACE rather than its last range, because the duck has walked
            # since. The application gate still refuses a turn in place.
            raising = (p.settle_head_level > 0.0 and self.state == "settle"
                       and t - self.t_state >= p.settle_s - p.settle_head_level)
            if p.gaze_still and gaze_at is None and self.state in ("lineup", "settle") and not raising:
                got = self._gaze_range(odom, ball)
                if got is not None:
                    gaze_at, gaze_yaw = got
        elif seen:
            self.last_bearing = ball.bearing
            if fresh:
                self.last_seen_t = t
            take_lineup = False
            if fresh and ball.range < p.lineup_range and abs(ball.bearing) < 0.5:
                planned = self._plan(odom, ball)
                # A kick from the wrong side in our third is the own-goal
                # stance: refuse it and keep the chase-behind / keep-out
                # approach. A board push (or any non-kick plan) still lines
                # up — `_plan` already chose that over a swing.
                if planned[4] != "kick" or not self._lineup_wrong_side(odom, ball):
                    take_lineup = True
                    self.spot = planned
                    self.lined = False
                    self.state = "lineup"
                    self.t_state = t
                    vx, wz = p.speed, clip_wz(p.k_turn * ball.bearing)
                    gaze_at = ball.range
            if not take_lineup:
                # Where to AIM the chase: the ball, or a point `chase_behind`
                # behind it on the ball-to-goal line, so the run-in arrives on
                # the side the kick wants. The gaze and every range test stay
                # on the BALL - only the steering moves.
                aim = ball.bearing
                if p.chase_behind > 0.0 and self.goal is not None:
                    bx, by = self._ball_xy(odom, ball)
                    aim = self.chase_aim(odom, bx, by, ball.bearing)
                if abs(aim) > p.turn_first:
                    vx, _, wz = turn(aim, cold)
                    self.state = "turn"
                else:
                    vx, wz = p.speed, clip_wz(p.k_turn * aim)
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
                if p.search_dip_every > 0 and since % p.search_dip_every < p.search_dip_s:
                    # A standing pause with the gaze down. The comment here
                    # used to say it was for seeing a near ball below the
                    # level camera; MEASURED, it does not do that. Over 24
                    # seeds x 300 s of 2v2, detector frames binned by the
                    # state that COMMANDED the head pose: frozen in the dip,
                    # 11 balls found in 15 646 frames (0.07%); walking the
                    # same search circle, 505 in 11 116 (4.54%). The dip
                    # takes 58% of search frames and returns 2.1% of the
                    # search's sightings - 65x worse than simply walking on
                    # with the head level (z = 26.2).
                    #
                    # It stays because REMOVING it is much worse, and that
                    # confirmed on fresh seeds: `search_dip_s` = 0 gives back
                    # 19 s a run of standing still and costs falls 121 -> 195
                    # over 48 seeds (+61%) and 30% of the kicks, for no
                    # visibility gain at all (+0.008, p = 0.60). `_search_t0`
                    # resets on every ENTRY, so this is not a duty cycle in a
                    # long hunt: the median search is 0.60 s, exactly the
                    # dip, and 48% of searches are nothing but this pause. It
                    # is a flinch every time the ball leaves view (~33 a run
                    # a duck) - and standing is the one thing this walker
                    # does safely against another body, which is why taking
                    # it away costs falls. Mis-commented, not mis-designed.
                    vx, wz = 0.0, 0.0
                    gaze_at = p.dip_range
                elif p.search_walk_after and since > p.search_walk_after and (since - p.search_walk_after) % (p.search_walk_after) < p.search_walk_s:
                    vx, wz = p.speed, 0.0                       # a cold standing turn is exactly 0 rad/s: move to see from elsewhere
        # A wall beside us: no turn in place toward it (measured: a line-up
        # turning against the boards tipped over). Turn toward the side
        # with more room — in a corner that is still a turn, the one move
        # that gets out of a corner (standing there measured as a deadlock).
        if vx <= TURN_KICK and wz != 0.0 and self.state != "retreat":
            if wz > 0 and left_near < p.side_stop and right_near > left_near:
                wz = -1.0
            elif wz < 0 and right_near < p.side_stop and left_near > right_near:
                wz = max_wz()
        stop = p.tof_stop
        # (a body-clear PUSH spot walks past the bumper too - 12an; the shipped brain plans none)
        if p.lineup_tof_stop > 0.0 and self.state in ("lineup", "settle") and self.spot is not None \
                and self.spot[4] in ("kick", "push") and self.bounds is not None \
                and math.hypot(self.spot[0] - odom[0], self.spot[1] - odom[1]) <= p.lineup_tof_within \
                and self._spot_body_clear(self.spot[0], self.spot[1]):
            stop = p.lineup_tof_stop                 # the spot keeps the body out of the wall (`lineup_tof_stop`)
        if ahead < stop and vx > 0 and self.state != "push":
            # A wall or the other duck right there: no walking, no cold-turn
            # creep (measured: every remaining fall was a line-up walking
            # into a wall or a kicked turn creeping into one). Turning still
            # happens — the left turn that starts from a standstill.
            vx = 0.0
            if self.state in ("lineup", "settle", "support", "wait"):
                wz = max_wz() if wz > 0 else -max_wz() if wz < 0 else 0.0
            else:
                wz = max_wz()
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
        # THE SAME REMEDY, ON THE TRIGGER THE RULE ABOVE CANNOT HAVE (roadmap
        # 12aa). That one wants a still body AND a still head, in `avoid` /
        # `blocked` / `yield`. The corner trap is neither: a supporter that
        # cannot see the ball TURNS ON THE SPOT (`_support`'s no-ball branch is
        # `turn(1.0, cold)` and nothing else), in `support`, forever — 123.4 s
        # in one corner on seed 30, 0.10 m off both boards, the ball 0.53 m
        # away and unseen. Turning is a search in open field and cannot be one
        # against the boards, where the whole view IS board, so the duck is
        # blind BECAUSE it is in the corner and stays there because it is
        # blind. Escaping it is the retreat's own job; only the trigger was
        # missing.
        #
        # The clock is on DISPLACEMENT, not on the ball: the belief flickers
        # (the trapped duck had one on 29% of ticks, interleaved), so a clock
        # any momentary sighting resets never reaches a threshold at all — the
        # first version of this rule was bit-identical to shipped for exactly
        # that reason. What does not flicker is whether the duck has got
        # anywhere. A supporter ON its post is exempt (`support_hold_tol`):
        # standing still there is the job, and the audit found those are 2 of
        # every 25 long stands.
        if p.support_unstick_s > 0 and self.state in ("support", "wait") and not self._kickoff_wait:
            self._sup_poses.append((t, odom[0], odom[1]))
            while self._sup_poses and t - self._sup_poses[0][0] > p.support_unstick_s:
                self._sup_poses.pop(0)
            held = self.post is not None and math.hypot(self.post[0] - odom[0],
                                                        self.post[1] - odom[1]) <= p.support_hold_tol
            if held:
                self._sup_poses = []              # on its post: standing still IS the job
            elif len(self._sup_poses) > 1 and t - self._sup_poses[0][0] >= p.support_unstick_s - 0.05 \
                    and max(math.hypot(px - odom[0], py - odom[1])
                            for _, px, py in self._sup_poses) < p.support_unstick_move:
                self._sup_poses = []
                self._retreat_t0 = t
                self._retreat_sign = 1.0 if left_near >= right_near else -1.0
        elif p.support_unstick_s > 0:
            self._sup_poses = []
        # A turn in place keeps the head level whatever the state asked for:
        # the walker cannot turn in place with its head down (0.2 rad in 5 s
        # against 3.1 level, measured in tidy.py). A COLD turn carries
        # `TURN_KICK` of forward command to start the gait, so "turning in
        # place" is not "vx == 0" — which is why the old `vx > 0` gate let
        # the head down during exactly the manoeuvre that cannot take it.
        turning = wz != 0.0 and vx <= TURN_KICK
        if gaze_at is not None and not (p.gaze_still and turning) \
                and (vx > 0 or self.state in ("look", "search")
                     or (p.gaze_still and wz == 0.0)
                     or (p.support_gaze and self.state in ("support", "wait") and not turning)):
            # The gaze YAW is gated on forward clearance exactly like the look
            # yaw below, and for the same measured reason: the ToF is on the
            # head, so a yawed head is honestly blind ahead. The gaze PITCH is
            # not gated — the dip re-screened as a clean null (roadmap 4e) and
            # gating it would change the shipped brain, which this does not:
            # with `gaze_yaw` off the yaw is 0.0 either way.
            gyaw = float(np.clip(p.head_yaw_gain * gaze_yaw,
                                 -p.head_yaw_max, p.head_yaw_max)) if p.gaze_yaw else 0.0
            if p.yaw_clear > 0.0 and ahead < p.yaw_clear:
                gyaw = 0.0
            if self.state == "settle" and (p.settle_gaze_neck > 0.0 or p.settle_head_down > 0.0):
                # The settle's own look (`settle_gaze_neck` / `settle_head_down`):
                # standing on the spot, the gaze may go where the walking one
                # cannot. Off, this branch is not taken and the tuple is the
                # shipped one to the bit.
                nk = p.settle_gaze_neck if p.settle_gaze_neck > 0.0 else p.gaze_neck
                dn = p.settle_head_down if p.settle_head_down > 0.0 else p.head_down
                head = self._head_pose(self._gaze(gaze_at, nk, dn), gyaw, nk)
            else:
                head = self._head_pose(self._gaze(gaze_at), gyaw)
        if look_yaw is not None and self.state == "look":
            head = self._head_pose(self._gaze(gaze_at), float(np.clip(look_yaw, -p.head_yaw_max, p.head_yaw_max)))
        # Where the head looks, and the range that goes with it (the tracking
        # pitch, `track_pitch`). The prediction while it is fresh; then, with
        # `look_hold_s`, the coasting track - whose bearing the tracker turns
        # with the body - until that runs out; `head_lead_s` aims at the ball
        # where it will be when the servo gets there.
        look_rng: float | None = None
        if pred_bearing is not None:
            look_at = pred_bearing
            look_rng = math.hypot(self.predicted[0] - odom[0], self.predicted[1] - odom[1])
            if p.head_lead_s > 0.0:
                lx, ly = ball.predict(t + p.head_lead_s, p.ball_decel)
                look_at = _wrap(math.atan2(ly - odom[1], lx - odom[0]) - odom[2])
        elif p.predict_s > 0 and ball is not None and ball.age(t) <= max(p.predict_s, p.look_hold_s):
            look_at = ball.bearing
            look_rng = ball.range
        else:
            look_at = None
        if look_at is None and self.state == "look" and p.look_aim and self._last_foot is not None:
            look_at = p.kick_exit_left if self._last_foot == "kick_left" else p.kick_exit_right
        elif look_at is None and self.state == "search" and p.search_sweep > 0 and self._search_t0 is not None:
            look_at = p.search_sweep * math.sin(2.0 * math.pi * (t - self._search_t0) / p.search_sweep_s) / p.head_yaw_gain
        elif look_at is None and p.head_memory_s > 0.0 and self.state in p.head_memory_states.split("+"):
            # The head on the remembered ball (`head_memory_s`): the board's
            # ball if a teammate has it, else our own last sighting.
            mem = self.team.ball(t) if self.team is not None else None
            if mem is None and self.memory is not None and t - self.memory[2] <= p.head_memory_s:
                mem = (self.memory[0], self.memory[1])
            if mem is not None:
                look_at = _wrap(math.atan2(mem[1] - odom[1], mem[0] - odom[0]) - odom[2])
                look_rng = math.hypot(mem[0] - odom[0], mem[1] - odom[1])
        if look_at is not None and senses.skill is None and (p.head_yaw_when == "always" or self.state in ("search", "look")) \
                and not (p.yaw_clear > 0.0 and ahead < p.yaw_clear):
            # …unless the way ahead is not clear. The ToF is ON THE HEAD, so
            # yawing it points the bumper off the walking line: since the
            # clearance rule became bearing-based it reports `+inf` honestly
            # instead of a false wall, which means a yawed duck walks with no
            # forward obstacle sense at all (measured: past 0.70 rad it stops
            # on 0.3% of frames where the old column rule stopped on 13.8%).
            # That is where head-tracking's falls come from. The brain has the
            # signal and did not consult it; this consults it, keeping the
            # head on the line whenever something is inside `yaw_clear`.
            yaw_cmd = float(np.clip(p.head_yaw_gain * look_at, -p.head_yaw_max, p.head_yaw_max))
            pitch = 0.0
            if p.track_pitch and look_rng is not None and head[1] == 0.0 and self.state != "settle" \
                    and (not turning or p.track_pitch_turn > 0.0):
                # ...and the PITCH that keeps that ball inside the frame, only
                # where no gaze law has already put the head somewhere (a
                # line-up gaze centres the ball, deeper; the look and the
                # search dip aim at their own ranges), never in the settle
                # (the swing's run-up belongs to `gaze_still` and
                # `settle_head_level`, both measured against the kick), and
                # never more than a turn in place can take (`track_pitch_turn`).
                cam_z = senses.det.cam_z if senses.det is not None and senses.det.cam_z > 0.0 else p.cam_z
                pitch = self._track_pitch(look_rng, cam_z, walking=vx > TURN_KICK)
                if turning:
                    pitch = min(pitch, p.track_pitch_turn)
            head = self._head_pose(pitch, yaw_cmd) if pitch > 0.0 else (head[0], head[1], yaw_cmd, head[3])
        # The two arms of the same rule, on the SAME gate - a turn in place,
        # beside a body, in a state where that turn is not itself the escape
        # - so an A/B between them measures the action and nothing else.
        if t - self._bump_t0 < max(p.bump_stand_s, p.bump_back) and vx <= TURN_KICK and wz != 0.0 \
                and self.state in p.bump_stand_states:
            if p.bump_back > 0 and t - self._bump_t0 < p.bump_back:
                vx, _, wz = back_up()                           # back out of it, which standing never does
            elif p.bump_stand_s > 0:
                vx, wz = 0.0, 0.0                               # touching a body: stand, do not turn in place
        self.last = (vx, vy, wz)
        return Intent(twist=self.last, head=head, note=self.role if self.role != "attack" else self.state, skill=skill)

    def _attack_x(self, x: float) -> float:
        """A point's position along the pitch in ATTACK coordinates: −1 at the
        goal we defend, +1 at the one we attack. Both teams read the same
        numbers, so a third is a third whichever way a duck is pointing."""
        if self.bounds is None or self.bounds[0] <= 0:
            return 0.0
        sign = 1.0 if (self.goal is None or self.goal[0] >= 0) else -1.0
        return sign * x / self.bounds[0]

    def _from_attack_x(self, a: float) -> float:
        sign = 1.0 if (self.goal is None or self.goal[0] >= 0) else -1.0
        return sign * a * (self.bounds[0] if self.bounds else 0.0)

    def _too_wide(self, odom) -> bool:
        """Is the ball too far to the SIDE for this swing to be worth a
        touch? (`kick_side_max`; False when the knob is off.)

        The side offset is the kick error — +1.90 deg of aim error per cm,
        measured over 462 kicks — and it cannot be aimed out, because the
        spot is laid out in the body heading so every rotation moves the
        offset itself. Declining is the only remaining lever.

        The estimate is `self.predicted`, the track's position propagated by
        its own velocity, which exists only while the sighting is inside
        `predict_s`. With no fresh estimate this returns False and the duck
        swings: the plan's own assumed ball position sits ON the sweet spot
        by construction, so gating on it would refuse nothing, and gating on
        nothing at all would be a duck that never kicks."""
        if self.p.kick_side_max <= 0.0 or self.predicted is None:
            return False
        dx, dy = self.predicted[0] - odom[0], self.predicted[1] - odom[1]
        side = -dx * math.sin(odom[2]) + dy * math.cos(odom[2])
        return abs(side) > self.p.kick_side_max

    def _too_far(self, odom) -> bool:
        """Is the ball too far AHEAD for this swing to reach it?
        (`kick_ahead_max`; False when the knob is off or nothing fresh has
        been seen.) The plan's own ball sits `kick_ahead` = 0.08 m out by
        construction; a predicted ball beyond the gate means the plan went
        stale while the duck walked in (median plan age 3.6 s)."""
        if self.p.kick_ahead_max <= 0.0 or self.predicted is None:
            return False
        dx, dy = self.predicted[0] - odom[0], self.predicted[1] - odom[1]
        ahead = dx * math.cos(odom[2]) + dy * math.sin(odom[2])
        return ahead > self.p.kick_ahead_max

    def _kick_heading(self, foot: str, u: float) -> float:
        """The line the ball actually leaves on: aim heading plus the in-play
        foot exit angle when `hunt_exit` is on (the shipped default)."""
        p = self.p
        if not p.hunt_exit:
            return u
        return u + (p.kick_exit_left if foot == "kick_left" else p.kick_exit_right)

    def _hold_target(self, bxy, odom) -> tuple[float, float]:
        """Where this duck stands while a teammate has the ball.

        With no static role it is the shipped supporter's spot — back from the
        ball toward our own goal, spread sideways by rank. With one it is that
        role's post (`ChaseParams.defend_depth` / `strike_ahead` / `mid_side`),
        kept inside the third the role owns so that holding a post and being
        allowed to take the ball are the same geometry (`Team.zone_ok`)."""
        p = self.p
        t = self._senses.t
        og = self._own_goal(odom)
        if p.support_field and (self.job in ("striker", "midfielder") or (self.job is None and p.field_plain)):
            spot = self._field_spot(bxy, odom)          # the FIELD: already inside the zone and the boards
            if spot is not None:
                return spot
        if self.job == "defender":
            dx, dy = bxy[0] - og[0], bxy[1] - og[1]
            n = math.hypot(dx, dy)
            if n < 1e-6:
                target = og
            else:
                # On the line from our goal to the ball: between, which is the job.
                target = (og[0] + p.defend_depth * dx / n, og[1] + p.defend_depth * dy / n)
        elif self.job == "striker":
            g = self.goal if self.goal is not None else (og[0] + 2.0, og[1])
            gx, gy = g[0] - bxy[0], g[1] - bxy[1]
            n = math.hypot(gx, gy)
            ux, uy = (gx / n, gy / n) if n > 1e-6 else (1.0, 0.0)
            # Off the kick line, on the side the ball is NOT on: a striker
            # standing ON the line is the poacher that reversed on fresh
            # seeds, and it is a second duck on the ball.
            # The offset rides the lane's left normal (-uy, ux), whose sense
            # flips with the attack direction, so the side is chosen in ATTACK
            # coordinates: until 2026-09-08 the team attacking -x posted its
            # striker on the SAME side as the ball (a code review caught it).
            sign = 1.0 if (self.goal is None or self.goal[0] >= 0) else -1.0
            side = (-p.strike_side if bxy[1] >= 0 else p.strike_side) * sign
            target = (bxy[0] + p.strike_ahead * ux - side * uy, bxy[1] + p.strike_ahead * uy + side * ux)
        elif self.job == "midfielder":
            a = float(np.clip(self._attack_x(bxy[0]) * 0.5, -1.0 / 3.0, 1.0 / 3.0))
            target = (self._from_attack_x(a), p.mid_side * (1.0 if bxy[1] >= 0 else -1.0))
        elif self.job == "keeper":
            # On the ball-to-goal line, `keeper_depth` in front of the mouth's
            # centre, never outside the posts: the shot it has to be in the
            # way of runs from the ball to the mouth, and the mouth is where
            # it is. A ball beside the goal pins it to the near post.
            sgn = 1.0 if (self.goal is None or self.goal[0] >= 0) else -1.0    # from OUR mouth toward the pitch
            dx = bxy[0] - og[0]
            frac = p.keeper_depth / max(abs(dx), p.keeper_depth)
            span = max(self.goal_w / 2.0 - 0.05, 0.05)
            target = (og[0] + sgn * p.keeper_depth,
                      float(np.clip(og[1] + (bxy[1] - og[1]) * frac, og[1] - span, og[1] + span)))
        else:
            anchor = og if p.support_mode == "back" else (self.goal if self.goal is not None else og)
            gx, gy = anchor[0] - bxy[0], anchor[1] - bxy[1]
            gn = math.hypot(gx, gy)
            ux, uy = (gx / gn, gy / gn) if gn > 1e-6 else (-math.cos(odom[2]), -math.sin(odom[2]))
            rank = self.team.rank(self.duck_id, t) if self.team is not None else 0
            side = p.support_side * ((rank + 1) // 2) * (1 if rank % 2 == 0 else -1)
            return (bxy[0] + p.support_back * ux - side * uy, bxy[1] + p.support_back * uy + side * ux)
        # Holding a post and being allowed to take the ball are the same
        # geometry: clip to the zone the board uses (halfway without a mid,
        # thirds with one).
        z = self.team.zone_of(self.duck_id) if self.team is not None else None
        if z is not None:
            a = float(np.clip(self._attack_x(target[0]), z[0], z[1]))
            target = (self._from_attack_x(a), target[1])
        return target

    def _field_spot(self, bxy, odom) -> tuple[float, float] | None:
        """Where the potential field (brain/field.py) puts this supporter:
        `ahead` of the ball along the carrier's lane by the role's number,
        beside the lane and not in it, clear of teammates (the board's
        positions) and of opponents (duck tracks that are not on the board
        as a teammate, and not our colour when the colour sense is on).
        None off a pitch, or when the zone leaves no spot."""
        p = self.p
        if self.goal is None or self.bounds is None:
            return None
        if self._field is None:
            from .field import Field, FieldParams  # noqa: PLC0415  (only a pitch pays for it)
            self._field = Field(self.bounds, p.support_margin, FieldParams(
                lane_w=p.field_lane, wide=p.field_wide, goal_pull=p.field_goal_pull, hysteresis=p.field_hyst))
        from .field import lane_unit  # noqa: PLC0415
        from .kickselect import Pitch  # noqa: PLC0415
        t = self._senses.t
        u = lane_unit(bxy, self.goal, self.attack if self.attack is not None else odom[2])
        if self.job == "striker":
            ahead = p.strike_ahead
        elif self.job == "midfielder":
            ahead = p.field_mid_ahead
        else:
            ahead = p.support_back if p.support_mode == "ahead" else -p.support_back
        # Teammates repel - except the carrier, which is AT the ball: the
        # lane and the attacker's room already say where not to stand for
        # it, and a repulsor on it too pushes the spot out of a push's reach.
        att = self.team.attacker(t) if self.team is not None else None
        mates = ([(mx, my) for k, (mx, my, _) in self.team.mates(self.duck_id, t) if k != att]
                 if self.team is not None else [])
        opps = self._opponents(t)
        pitch = Pitch(self.bounds[0], self.bounds[1], self.goal_w, 1.0 if self.goal[0] >= 0 else -1.0)
        zone = self.team.zone_of(self.duck_id) if self.team is not None else None
        spot = self._field.spot(bxy, u, ahead, mates, opps, pitch, zone=zone,
                                keep_out=p.support_min, prev=self._field_prev, me=(odom[0], odom[1]))
        self._field_prev = spot
        return spot

    def _opponents(self, t: float) -> list[tuple[float, float]]:
        """Where the OTHER side is, as well as this duck can tell (roadmap
        C.4): every duck track seen within `lost_s` that the board does not
        own - not within 0.35 m of a teammate's own claim of its position,
        and not our colour when the colour sense is on (`_is_mate`)."""
        p = self.p
        mates = ([(mx, my) for _, (mx, my, _) in self.team.mates(self.duck_id, t)]
                 if self.team is not None else [])
        out: list[tuple[float, float]] = []
        for tr in self.tracker.tracks:
            if tr.cls != "duck" or tr.xy is None or tr.age(t) > p.lost_s or self._is_mate(tr):
                continue
            if any(math.hypot(tr.xy[0] - mx, tr.xy[1] - my) < 0.35 for mx, my in mates):
                continue                                    # the board says that one is ours
            out.append((float(tr.xy[0]), float(tr.xy[1])))
        return out

    def _support(self, odom, ball, seen: bool, cold: bool) -> tuple[float, float]:
        """A supporter: hold the post its role gives it (`_hold_target`) —
        without a role, back from the ball toward our own goal, offset
        sideways by rank — facing the ball. The ball's position comes from
        my own track when I see it, else from a teammate's claim."""
        p = self.p
        t = self._senses.t
        bxy = self._ball_xy(odom, ball) if seen else (
            self.team.led_ball(t) if self.team is not None else None)
        self.spot = None
        self.post = None
        if bxy is None:
            self.state = "wait" if self._kickoff_wait else "support"
            vx, _, wz = turn(1.0, cold)                    # nobody has it: look for it
            return vx, wz
        target = self._hold_target(bxy, odom)
        if self.bounds is not None:                         # never a spot in the boards
            m = p.support_margin
            target = (float(np.clip(target[0], -self.bounds[0] + m, self.bounds[0] - m)),
                      float(np.clip(target[1], -self.bounds[1] + m, self.bounds[1] - m)))
        if self._kickoff_wait and self.bounds is not None and self.bounds[0] > 0:
            # Standing off the other side's kickoff: own half, out of the circle.
            a = min(self._attack_x(target[0]), -p.kickoff_circle / self.bounds[0])
            target = (self._from_attack_x(a), target[1])
        self.post = target
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
        self.state = "wait" if self._kickoff_wait else "support"
        return vx, wz

    def _select_kick_line(self, odom, ball_xy, los: float, u_clamp: float) -> tuple[float, str] | None:
        """The kick line and FOOT `_plan` should lay its spot for, chosen by
        simulated outcomes (brain/kickselect.py) from a fan of lines inside
        the aim window `los +- aim_max` - the same walk-round the clamp
        permits, so nothing here costs a longer line-up - with BOTH feet
        offered on every line. The foot matters more than it looks: the
        planner's own rule takes the foot on the ball's side of the line,
        and that foot's exit angle (+23.6 / -28.7 deg) bends the kick back
        toward the line of sight, so a 60 deg clamp turn leaves an outcome
        only ~30 deg off the ball's own heading. Measured (this scenario is
        locked in tests/test_kickselect.py): facing our own mouth from
        0.4 m, every line with the planner's foot puts 50-83% of kicks in
        our own net; the other foot on the same edge line puts in 7%. None
        when every candidate is too risky: the caller keeps the clamp's
        line and its own foot."""
        from .kickselect import (  # noqa: PLC0415  (only a pitch pays for it)
            KickModel,
            Pitch,
            select,
        )
        p = self.p
        if self._kick_rng is None:
            import zlib  # noqa: PLC0415
            self._kick_rng = np.random.default_rng(zlib.crc32(self.duck_id.encode() or b"duck"))
        x, y, _ = odom
        bx, by = ball_xy
        lines: list[tuple[float, str]] = []
        k = max(1, int(round(p.aim_max / max(p.kick_select_fan, 1e-3))))
        for i in range(-k, k + 1):
            u = _wrap(los + i * p.kick_select_fan)
            if abs(_wrap(u - los)) > p.aim_max + 1e-9:
                continue
            lines += [(u, "kick_left"), (u, "kick_right")]
        lines += [(u_clamp, "kick_left"), (u_clamp, "kick_right")]   # the clamp's own line is always a candidate
        mates_xy: list[tuple[float, float]] | None = None
        if p.kick_select_pass and self.team is not None and self._senses is not None:
            t_now = self._senses.t
            sign = 1.0 if self.goal[0] >= 0 else -1.0
            mates_xy = []
            for _, (mx, my, _) in self.team.mates(self.duck_id, t_now):
                mates_xy.append((float(mx), float(my)))
                if sign * (mx - bx) >= p.pass_min_ahead:              # up-pitch of the ball: a pass, not a back-pass
                    u_m = math.atan2(my - by, mx - bx)
                    if abs(_wrap(u_m - los)) <= p.aim_max + 1e-9:    # inside the same walk-round the clamp permits
                        lines += [(u_m, "kick_left"), (u_m, "kick_right")]
        models = None
        if p.kick_select_push and not (p.defender_clears and self.job in ("defender", "keeper")):
            from .kickselect import push_model  # noqa: PLC0415
            lines += [(u_, "push") for u_, act in lines if act == "kick_left"]   # one push per line
            models = {"push": push_model(p.push_roll, p.push_dir_sd, max(p.ball_decel, 0.02))}
        if p.spot_reach > 0.0:
            # Only spots the body can occupy are offered (`spot_reach`); a
            # fan with none left is a corner, and it is left whole.
            reachable = [(u_, act) for u_, act in lines if self._spot_clear((bx, by), u_, act, p.spot_reach)]
            if reachable:
                self.unreach_dropped += len(lines) - len(reachable)
                lines = reachable
            else:
                self.unreach_corners += 1
        model = KickModel(speed=p.kick_speed, speed_sd=p.kick_select_v_sd, dir_sd=p.kick_select_dir_sd,
                          decel=max(p.ball_decel, 0.02), exit_left=p.kick_exit_left, exit_right=p.kick_exit_right,
                          p_whiff=p.kick_select_p_whiff)
        pitch = Pitch(self.bounds[0], self.bounds[1], self.goal_w, 1.0 if self.goal[0] >= 0 else -1.0)
        opps = self._opponents(self._senses.t) if (p.kick_select_opps and self._senses is not None) else None
        # THE LEARNED RANKING (E.2). `_kick_choice` may also be installed
        # directly by a probe (scripts/kick_choice_data.py), which is why the
        # knob only builds it when it is still None.
        if self._kick_choice is None and p.kick_select_learned:
            from .kickchoice import Chooser  # noqa: PLC0415  (only a learned arm pays for it)
            self._kick_choice = Chooser.load(p.kick_select_learned)
        chooser = None if self._kick_choice is None else self._kick_choice.bind(odom, los, self.goal)
        v = select((bx, by), lines, model, pitch, self._kick_rng, n=p.kick_select_n, t_own=p.kick_select_t_own,
                   safest=p.kick_select_safest,
                   models=models, shoot=p.kick_select_shoot if p.kick_select_push else 0.0,
                   mates=mates_xy, pass_reach=p.pass_reach, pass_bonus=p.pass_bonus if p.kick_select_pass else 0.0,
                   obstacles=opps, obs_r=p.kick_select_obs_r, chooser=chooser)
        self.last_select = v
        return None if v is None else (v.heading, v.foot)

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

    def _is_mate(self, tr) -> bool:
        """Is this duck track one of ours? Only with the colour sense on, and
        only on the track's VOTE — one frame of the classifier is a coin at
        the hostile preset. Unknown counts as an opponent: the cost of
        treating a teammate as a stranger is a wasted metre, and the cost of
        the reverse is walking into one."""
        return bool(self.p.use_color and self.team is not None
                    and getattr(tr, "color", None) == self.team.name)

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
