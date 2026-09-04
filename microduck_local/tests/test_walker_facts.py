"""The locomotion facts every scripted brain is built on, locked down.

`brain/gait.py` asserted for months that the walker cannot walk backwards.
It can — faster than it walks forwards. The claim came from measuring ONE
command, -0.3 m/s, which is the inside of a dead band, and three brains
grew workarounds for a limit that was not there. These tests measure the
dead band's EDGE as well as its inside, so the next walker that changes
this breaks a test instead of a benchmark six months later.

They run the real `alpha_walking.onnx` on an empty floor: `make_room`'s
3.0 x 2.5 m room with four boxes cannot hold the 1.3 m a 6 s reverse
covers, and measuring the gait in it measures the boxes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.brain.brain_env import POLICIES_DIR
from microduck_local.brain.gait import BACK_MIN, BACK_SPEED, TURN_KICK, back_up, turn
from microduck_local.walker_facts import _flat_world, _settle

walker = pytest.mark.skipif(not (POLICIES_DIR / "alpha_walking.onnx").exists(),
                            reason="upstream policies not checked out")


def _drive(vx: float, secs: float = 6.0) -> float:
    """Steady speed along the start heading over the last 2/3 of the command."""
    w, d = _flat_world()
    _settle(w, d)
    p0 = d.trunk_pos(w.data).copy()
    y0 = d.yaw(w.data)
    c, s = math.cos(y0), math.sin(y0)
    ahead = []
    for _ in range(int(secs / 0.02)):
        d.set_cmd(w.data, (vx, 0, 0), (0, 0, 0, 0))
        w.step()
        p = d.trunk_pos(w.data)
        ahead.append(c * (p[0] - p0[0]) + s * (p[1] - p0[1]))
    i2 = int(2 / 0.02)
    return (ahead[-1] - ahead[i2]) / (secs - 2.0)


def _spin(wz: float, warm: bool, secs: float = 4.0) -> float:
    w, d = _flat_world()
    _settle(w, d)
    if warm:
        for _ in range(100):
            d.set_cmd(w.data, (0.3, 0, 0), (0, 0, 0, 0))
            w.step()
    ys = []
    for _ in range(int(secs / 0.02)):
        d.set_cmd(w.data, (0, 0, wz), (0, 0, 0, 0))
        w.step()
        ys.append(d.yaw(w.data))
    u = np.unwrap(np.array(ys))
    return float((u[-1] - u[int(2 / 0.02)]) / (secs - 2.0))


@walker
def test_the_walker_reverses_and_minus_zero_point_three_is_the_dead_band():
    """The fact `gait.py` had backwards. -0.30 really does move ~nothing —
    but it is the dead band's inside, not the walker's limit, and one notch
    past it the robot backs up faster than it ever goes forwards."""
    assert abs(_drive(-0.30)) < 0.01                    # what the old fact measured
    assert _drive(BACK_MIN) < -0.15                     # -0.35: ~0.20 m/s
    back = _drive(BACK_SPEED)                           # -0.40: ~0.23 m/s
    assert back < -0.19
    assert back < -_drive(0.40)                         # ...faster than forwards (~0.19)
    assert abs(_drive(TURN_KICK)) < 0.01                # +0.2 starts the gait, it does not travel


@walker
def test_a_cold_turn_below_full_command_is_exactly_zero():
    """Not "weak" — zero, both ways, which is why `gait.turn` only ever
    commands |wz| = 1 and adds `TURN_KICK` when the gait is cold."""
    for wz in (0.25, 0.5, 0.75, -0.25, -0.5, -0.75, -1.0):
        assert abs(_spin(wz, warm=False)) < 0.05, wz
    assert _spin(1.0, warm=False) > 0.4                 # only a full LEFT command breaks through cold
    assert _spin(1.0, warm=True) > 0.4 and _spin(-1.0, warm=True) < -0.4
    assert abs(_spin(-0.5, warm=True)) > 0.2            # warm, the rate follows the command


def test_back_up_clamps_past_the_dead_band_and_into_the_command_range():
    """A brain that politely asks for -0.3 gets 4 mm and no error, so the
    helper refuses to hand out a command inside the dead band — or one
    outside what the walker's contract accepts."""
    assert back_up() == (BACK_SPEED, 0.0, 0.0)
    assert back_up(speed=-0.2) == (BACK_MIN, 0.0, 0.0)          # clamped past the dead band
    assert back_up(speed=-9.0) == (C.LIN_VEL_X_RANGE[0], 0.0, 0.0)   # clamped into the range
    assert back_up(0.5)[2] == 0.5 and back_up(0.5)[0] < 0        # combines with a turn
    assert turn(+1, cold=True) == (TURN_KICK, 0.0, 1.0)
    assert turn(-1, cold=False) == (0.0, 0.0, -1.0)


@walker
def test_the_reverse_composes_with_a_turn_and_the_arc_is_not_a_failure():
    """Speed along the BODY axis is what says whether a reverse is working.
    Net displacement in the start frame is not: a duck reversing while it
    turns drives an arc that curves back toward where it began, and reading
    that as "the reverse fails under a turn" is a reading of the coordinate
    frame. It was written into `gait.back_up`'s docstring until the rollout
    was rendered and looked at."""
    straight = _body_axis_speed(-0.40, 0.0)
    assert straight < -0.19
    for wz in (0.5, 1.0, -1.0):
        turning = _body_axis_speed(-0.40, wz)
        assert turning < -0.19, wz                         # the reverse is unimpaired...
        assert abs(turning - straight) < 0.05, wz          # ...to within 5 cm/s of straight


def _body_axis_speed(vx: float, wz: float, secs: float = 6.0) -> float:
    """Metres a second along the duck's OWN heading, integrated step by step
    against the live yaw — the frame-independent reading."""
    w, d = _flat_world()
    _settle(w, d)
    prev = d.trunk_pos(w.data).copy()
    along = 0.0
    for _ in range(int(secs / 0.02)):
        d.set_cmd(w.data, (vx, 0, wz), (0, 0, 0, 0))
        w.step()
        p, y = d.trunk_pos(w.data), d.yaw(w.data)
        along += (p[0] - prev[0]) * math.cos(y) + (p[1] - prev[1]) * math.sin(y)
        prev = p.copy()
    return along / secs
