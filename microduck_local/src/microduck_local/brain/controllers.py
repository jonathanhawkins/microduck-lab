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

from dataclasses import dataclass

import numpy as np

from .runtime import REGISTRY, Intent, Senses, age_inputs


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
    k_turn: float = 2.0            # wz per rad of bearing
    k_speed: float = 0.6           # vx per m of distance error
    max_speed: float = 0.45
    min_speed: float = 0.12        # alpha_walking treats slower asks as "stand"
    lost_s: float = 2.0            # keep the last bearing this long, then search
    search_wz: float = 1.0             # the shipped walker barely turns in place below 1.0
    tof_stop: float = 0.35         # never walk into what the ToF says is right there
    head_yaw_gain: float = 0.8     # look toward the target (the robot's own gaze intent)


class Follow:
    """Keep the nearest target of a class ahead at a fixed distance, from the
    detector's bearing and width-derived range, with the ToF as a bumper.
    Loses it: coast toward where it was for `lost_s`, then turn to search.

    Deliberately simple — it is the baseline the learned brain (3.1/3.2) is
    measured against, and every number it uses is one the real robot can
    produce today or after one detector retrain.
    """

    kind = "follow"
    DET_MAX_AGE = 0.4
    TOF_MAX_AGE = 0.25

    def __init__(self, p: FollowParams = FollowParams()):
        self.p = p
        self.state = "search"
        self.last_bearing = 0.0
        self.last_seen_t: float | None = None
        self.last_range: float | None = None
        self._senses: Senses | None = None
        self.last: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self.state = "search"
        self.last_bearing = 0.0
        self.last_seen_t = None
        self.last_range = None
        self._senses = None
        self.last = (0.0, 0.0, 0.0)

    def inputs(self) -> dict:
        if self._senses is None:
            return {}
        out = age_inputs(self._senses, self.TOF_MAX_AGE, self.DET_MAX_AGE)
        out["target"] = None if self.last_seen_t is None else {
            "bearing": round(self.last_bearing, 3), "range": _r(self.last_range),
            "since": round(self._senses.t - self.last_seen_t, 2)}
        return out

    def step(self, senses: Senses) -> Intent:
        self._senses = senses
        p = self.p
        det = senses.fresh_det(p.DET_MAX_AGE if hasattr(p, "DET_MAX_AGE") else self.DET_MAX_AGE)
        target = None
        if det is not None:
            cands = [d for d in det.detections if d.cls == p.target_cls]
            if cands:
                target = min(cands, key=lambda d: d.range_est)
        # ToF bumper: the nearest thing in the middle columns.
        tof = senses.fresh_tof(self.TOF_MAX_AGE)
        ahead = np.inf
        if tof is not None:
            cols = _column_clearance(tof.depth_mm, tof.valid, WanderParams())
            ahead = float(cols[3:5].min())
        if target is not None:
            self.last_bearing = target.bearing
            self.last_range = target.range_est
            self.last_seen_t = senses.t
            wz = float(np.clip(p.k_turn * target.bearing, -1.0, 1.0))
            err = target.range_est - p.distance
            vx = float(np.clip(p.k_speed * err, 0.0, p.max_speed))
            if abs(target.bearing) > 0.6:
                vx = 0.0                                # turn first, walk after…
                wz = 1.0 if target.bearing > 0 else -1.0   # …at the rate the walker honours
            elif 0.0 < vx < p.min_speed:
                vx = 0.0 if err < 0.1 else p.min_speed
            self.state = "hold" if vx == 0.0 and abs(wz) < 0.2 else "approach"
        elif self.last_seen_t is not None and senses.t - self.last_seen_t < p.lost_s:
            wz = (1.0 if self.last_bearing > 0 else -1.0) if abs(self.last_bearing) > 0.15 else 0.0
            vx = 0.0
            self.state = "coast"
        else:
            wz = p.search_wz if self.last_bearing >= 0 else -p.search_wz
            vx = 0.0
            self.state = "search"
        if ahead < p.tof_stop and vx > 0:
            vx = 0.0
            self.state = "blocked"
        head_yaw = float(np.clip(p.head_yaw_gain * self.last_bearing, -0.6, 0.6)) if self.last_seen_t else 0.0
        self.last = (vx, 0.0, wz)
        return Intent(twist=self.last, head=(0.0, 0.0, head_yaw, 0.0), note=self.state)


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
REGISTRY.register("script", Script)
