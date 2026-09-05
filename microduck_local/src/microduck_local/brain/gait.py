"""What every brain needs to know about the reflex walker underneath it —
measured on the shipped `alpha_walking` (`uv run walker-facts`):

**Every one of these is a DEAD BAND, not a limit** — which is how this file
asserted for months that the walker cannot reverse. A locomotion fact read
off a single command value is a reading of the dead band.

- it walks backwards **faster than it walks forwards**: -0.35 and -0.40
  back up at 0.20 and 0.23 m/s, against 0.13 and 0.19 at +0.30 and +0.40.
  Everything from -0.30 in is dead (4 mm in 6 s) — and -0.30 is exactly
  where this file used to measure. `back_up()` hands out the command;
- from a standstill NO turn starts below |wz| = 1.0 — not a weak turn, an
  exactly zero one, in both directions — and a cold RIGHT turn is zero even
  at -1.0. Only wz = +1 breaks through cold, at 0.57 rad/s;
- with the gait going the rate is roughly linear in the command and tops
  out at what the command range allows: 0.15 / 0.28 / 0.47 / 0.61 rad/s at
  wz +0.25…+1, and -0.00 / -0.42 / -0.57 / -0.78 at -0.25…-1. The walker's
  ~0.6–0.8 rad/s ceiling and `ANG_VEL_Z_RANGE`'s ±1.0 are the same wall:
  at full command there is nothing left to ask for;
- a small forward command (0.2 m/s) starts the gait, so a cold turn with
  that kick turns at ~0.7 rad/s. It does NOT move the robot: +0.2 is inside
  the forward dead band too (9 mm in 6 s), so `TURN_KICK` is a gait starter
  and never a way to travel — a brain that walks at 0.2 stands still.

`GaitWatch` tells a brain whether the gait is going, from what the robot
itself can measure (odometry yaw rate and speed), and `turn` hands out the
right in-place turn command for it. Shared by every scripted brain so the
fact is written once.
"""

from __future__ import annotations

import math
import os

from .. import contract as C
from .runtime import Senses

TURN_KICK = 0.2            # forward command that starts the gait for a cold turn (it does NOT travel)
COLD_AFTER_S = 0.4         # standing this long counts as cold
BACK_MIN = -0.35           # the reverse dead band's edge: -0.30 moves 4 mm in 6 s, -0.35 backs up at 0.20 m/s
BACK_SPEED = -0.40         # and -0.40 at 0.23 m/s, the fastest this walker moves in any direction


class GaitWatch:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._prev_yaw: float | None = None
        self._moving_t = -9.0
        self.cold = True

    def update(self, senses: Senses) -> bool:
        """Call once per brain step, before deciding. Returns `cold`."""
        odom = senses.odom
        if odom is not None and self._prev_yaw is not None:
            dyaw = math.atan2(math.sin(odom[2] - self._prev_yaw), math.cos(odom[2] - self._prev_yaw))
            if abs(senses.speed or 0.0) > 0.05 or abs(dyaw) > 0.3 * C.CTRL_DT:
                self._moving_t = senses.t
        elif odom is None and abs(senses.speed or 0.0) > 0.05:
            self._moving_t = senses.t
        self._prev_yaw = None if odom is None else odom[2]
        self.cold = senses.t - self._moving_t > COLD_AFTER_S
        return self.cold


def max_wz() -> float:
    """The largest yaw rate any brain will ask for, as a rate in rad/s.

    ONE constant so a faster walker can actually be COMMANDED. Every
    scripted brain here caps its turn at 1.0 - `turn()` below, five
    `np.clip(..., -1.0, 1.0)` in `controllers.py`, four literal `wz = 1.0`,
    plus `TidyParams.scan_wz` (`FollowParams.search_wz` is DEAD -
    declared and never read; see its comment). Train a walker
    that does 1.5 rad/s, leave those alone, and every brain still asks for
    1.0: the battery would compare two walkers that are never commanded
    differently and conclude a faster turn does not help.

    It reads `MICRODUCK_MAX_WZ` so an experiment can set it per arm beside
    the walker it is testing, without a code edit that has to be undone.
    Default 1.0 = `C.ANG_VEL_Z_RANGE`'s edge, which is what the shipped
    walker delivers, so nothing changes until someone opts in.

    Raising this ALONE does nothing good: the shipped walker saturates at
    0.6-0.8 rad/s however hard it is asked (`walker-facts`). It is half of
    a pair - widen the training range too (roadmap 3.7,
    docs/turn-rate-experiment.md)."""
    try:
        v = float(os.environ.get("MICRODUCK_MAX_WZ", "") or C.ANG_VEL_Z_RANGE[1])
    except ValueError:
        return float(C.ANG_VEL_Z_RANGE[1])
    return max(0.0, v)


def clip_wz(wz: float) -> float:
    """A yaw command clipped to `max_wz()` — the one place that cap lives."""
    m = max_wz()
    return float(min(m, max(-m, wz)))


def turn(sign: float, cold: bool, kick: float = TURN_KICK) -> tuple[float, float, float]:
    """An in-place turn the walker will actually perform."""
    wz = max_wz() if sign > 0 else -max_wz()
    return (kick, 0.0, wz) if cold else (0.0, 0.0, wz)


def back_up(wz: float = 0.0, speed: float = BACK_SPEED) -> tuple[float, float, float]:
    """A reverse the walker will actually perform — for backing off a rim, a
    wall or another duck's feet without turning away from what you were
    looking at.

    `speed` is CLAMPED past the dead band's edge, because a brain that asks
    politely for -0.3 gets 4 mm in 6 s and no error. Combines with a turn.
    Measured escape from contact at 0.10 m: a reverse clears 0.30 m in a
    median 1.6 s, against 2.7 s for turn-90-and-walk and 3.2 s for
    turn-180-and-walk — and unlike either, it keeps the target in frame."""
    return (max(C.LIN_VEL_X_RANGE[0], min(float(speed), BACK_MIN)), 0.0, float(wz))


__all__ = ["BACK_MIN", "BACK_SPEED", "COLD_AFTER_S", "TURN_KICK", "GaitWatch",
           "back_up", "clip_wz", "max_wz", "turn"]
