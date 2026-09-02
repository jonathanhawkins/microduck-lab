"""What every brain needs to know about the reflex walker underneath it —
measured on the shipped `alpha_walking` (`uv run walker-facts`):

- it does not walk backwards (a -0.3 m/s command moves 4 mm in 2 s);
- from a standstill a RIGHT turn never starts (0.05 rad/s) and a LEFT one
  usually does within a second, but not always — with the gait already
  going, either turns at 0.6–0.8 rad/s;
- a small forward command (0.2 m/s) starts the gait, so a cold turn with
  that kick turns at ~0.7 rad/s at the cost of creeping forward a few
  centimetres a second.

`GaitWatch` tells a brain whether the gait is going, from what the robot
itself can measure (odometry yaw rate and speed), and `turn` hands out the
right in-place turn command for it. Shared by every scripted brain so the
fact is written once.
"""

from __future__ import annotations

import math

from .. import contract as C
from .runtime import Senses

TURN_KICK = 0.2            # forward command that starts the gait for a cold turn
COLD_AFTER_S = 0.4         # standing this long counts as cold


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


def turn(sign: float, cold: bool, kick: float = TURN_KICK) -> tuple[float, float, float]:
    """An in-place turn the walker will actually perform."""
    wz = 1.0 if sign > 0 else -1.0
    return (kick, 0.0, wz) if cold else (0.0, 0.0, wz)


__all__ = ["COLD_AFTER_S", "TURN_KICK", "GaitWatch", "turn"]
