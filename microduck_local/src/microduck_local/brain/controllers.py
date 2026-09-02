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
    made no progress for a while under a forward command, spins to unstick."""

    def __init__(self, p: WanderParams = WanderParams(), stuck_s: float = 2.0,
                 unstick_s: float = 1.2):
        self.p = p
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

    def step(self, depth_mm: np.ndarray | None, valid: np.ndarray | None,
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
