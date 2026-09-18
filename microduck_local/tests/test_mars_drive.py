"""MARS's base drive: does the ported controller actually move the robot?

`robots/mars_drive.py` is a port of Innate's own driver, and a port is a
claim — that the constants mean here what they mean there. This file is the
measurement behind that claim, in MARS's own standalone scene, and every
positive case has a planted break beside it that shows the assertion has
teeth (`microduck_local/AGENTS.md`, "A/B new tests against planted
regressions"):

    positive                          planted break                 caught by
    ------------------------------    --------------------------    ---------
    0.3 m/s x 5 s -> 1.5 m            KP_FORWARD = 0                x < 1 mm
    1.0 rad/s x 3 s -> 3.0 rad        the yaw torque sign flipped   yaw < 0
    1.0 rad/s tracks at 5 ms          GAIN_LIMIT = 100 (no guard)   |wz| > 10
    the watchdog stops the base       CMD_VEL_TIMEOUT_S = 1e9       keeps going
    a push comes back to the hold     station keeping returns 0     it sticks

The numbers, MEASURED on this Mac, are in `robots/mars_drive.py`'s module
docstring; the bands below are the task's, not the measurements' — a band
tight enough to be worth asserting and loose enough that a solver revision
is not a failure.

Nothing here touches `world/`: a driver is testable on one robot in one
scene, and Phase 3b is what wires it to a room.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.robots import mars
from microduck_local.robots import mars_drive as MD
from microduck_local.robots.mars_drive import MarsDriver

needs_mars = pytest.mark.skipif(
    not mars.mars_ready(), reason="MARS assets missing — uv run fetch-robot mars")

DT = C.PHYSICS_DT                 # 5 ms — the lab's world step
CONTROL_DT = 1.0 / 50.0           # the world loop's own tick, what refreshes a command


def world_with(*, prefixes: tuple[str, ...] = ("",), boxes=()) -> mujoco.MjModel:
    """A compiled scene: N MARSes, a floor, a light, and any boxes.

    Built here from `mars.robot_spec()` rather than by editing
    `mars._scene_spec`: that function is the SHIPPED standalone scene, and a
    test that grew a wall into it would change what every other body case and
    the conformance suite compiles. `attach` keeps the PARENT's options, so
    the timestep is set on the world, exactly as `world/compose.py` does it.
    """
    world = mujoco.MjSpec()
    world.option.timestep = DT
    world.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                             size=[20.0, 20.0, 0.05])
    light = world.worldbody.add_light()
    light.pos = [0.0, 0.0, 3.0]
    light.dir = [0.0, 0.0, -1.0]
    for name, pos, size in boxes:
        body = world.worldbody.add_body(name=name, pos=list(pos))
        body.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=list(size))
    for prefix in prefixes:
        frame = world.worldbody.add_frame(pos=[0.0, 0.0, 0.0])
        mars.MARS.attach(world, prefix=prefix, frame=frame)
    return world.compile()


def drive_for(drv: MarsDriver, model: mujoco.MjModel, data: mujoco.MjData,
              seconds: float, vx: float, wz: float, *,
              refresh_s: float | None = CONTROL_DT) -> dict:
    """Hold a command for `seconds`, refreshing it every `refresh_s`.

    `refresh_s=None` commands ONCE and never again, which is how the watchdog
    and the station-keeping cases are set up: the difference between those and
    a driving robot is nothing but whether the lease is renewed.
    """
    every = None if refresh_s is None else max(1, int(round(refresh_s / DT)))
    worst_arm = 0.0
    for k in range(int(round(seconds / DT))):
        if every is not None and k % every == 0:
            drv.set_cmd(vx, wz, data.time)
        elif k == 0:
            drv.set_cmd(vx, wz, data.time)
        drv.step(data)
        mujoco.mj_step(model, data)
        worst_arm = max(worst_arm, max(
            abs(float(data.qpos[drv.adr[j][0]]) - target)
            for j, target in mars.ARM_HOME.items()))
    x, y, yaw = drv.pose(data)
    v_forward, v_lateral, wz_now = drv.velocity(data)
    return {"x": x, "y": y, "yaw": yaw, "v_forward": v_forward,
            "v_lateral": v_lateral, "wz": wz_now, "arm_err": worst_arm,
            "finite": bool(np.isfinite(data.qpos).all()
                           and np.isfinite(data.qvel).all())}


def fresh(prefix: str = "", model: mujoco.MjModel | None = None,
          *, x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
    """A model, a zeroed MjData and a spawned driver."""
    model = mars.model() if model is None else model
    data = mujoco.MjData(model)
    drv = MarsDriver(model, prefix)
    drv.spawn(data, x, y, yaw)
    return model, data, drv


# ------------------------------------------------------------ the constants

def test_every_constant_is_innates_and_the_envelope_is_the_mad_mode():
    """The ported numbers, against literals.

    Pinned against typed-out values rather than against the module's own
    names: `tests/test_mars.py` learned that comparing a model to the
    constant it was built from is a tautology, and a planted `KP_YAW = 0.0`
    passed the first version of that file. These are Innate's
    `world.py` / `core.py` / `drive_limits.py` values, read off the pinned
    innate-os revision.
    """
    assert (MD.KP_FORWARD, MD.KP_LATERAL, MD.KP_YAW) == (200.0, 40.0, 3.0)
    assert (MD.KP_HOLD_LINEAR, MD.KP_HOLD_YAW) == (300.0, 6.0)
    assert MD.HOLD_SETTLE_S == 0.4
    assert (MD.MAX_BASE_LINEAR_SPEED, MD.MAX_BASE_ANGULAR_SPEED) == (2.0, 6.0)
    assert MD.CMD_VEL_TIMEOUT_S == 0.5
    assert (MD.MAX_CMD_LINEAR, MD.MAX_CMD_YAW) == (0.8, 2.0)
    # The one constant that is NOT theirs, and the only one this repo added.
    assert MD.GAIN_LIMIT == 1.0


def test_the_command_envelope_clamps_each_axis_on_its_own():
    """A full-speed arc is both limits at once, not a vector of one length.

    Per-axis because the base is differential-drive: asking for 0.8 m/s
    while turning at 2.0 rad/s is a legal command on the real robot, and a
    norm-style clamp would quietly slow every turn.
    """
    assert MD.clamp_cmd(2.0, 5.0) == (0.8, 2.0)
    assert MD.clamp_cmd(-2.0, -5.0) == (-0.8, -2.0)
    assert MD.clamp_cmd(0.8, 2.0) == (0.8, 2.0)
    assert MD.clamp_cmd(0.3, -1.0) == (0.3, -1.0)


# ---------------------------------------------------------- the step guard

@needs_mars
def test_the_gains_are_innates_except_the_yaw_the_timestep_cannot_carry():
    """The step-size guard, and the inertia it is computed against.

    MEASURED on this model: the apparent inertia at the base's three DoFs,
    arm folded at HOME, is 1.315 kg / 1.297 kg / 0.005761 kg*m^2 — the yaw
    figure is small because mars.urdf carries PLACEHOLDER inertias
    (ixx=iyy=izz=0.001 on every link), so the robot's yaw inertia is almost
    entirely its mass offsets. `KP_YAW = 3` against that at 5 ms is a loop
    gain of 2.60, past the 2.0 bound of an explicit first-order velocity
    loop, and the next case shows what that does. Forward and lateral are
    0.76 and 0.15, so they are Innate's verbatim.
    """
    _m, _d, drv = fresh()
    g = drv.gains()
    assert g["dt"] == pytest.approx(DT)
    assert g["mass_x"] == pytest.approx(1.315, abs=0.02)
    assert g["mass_y"] == pytest.approx(1.297, abs=0.02)
    assert g["inertia_yaw"] == pytest.approx(0.005761, rel=0.02)
    # Untouched: the step can integrate them.
    assert drv.kp_forward == MD.KP_FORWARD
    assert drv.kp_lateral == MD.KP_LATERAL
    assert MD.KP_FORWARD * DT / g["mass_x"] < 2.0
    # Clamped: it cannot integrate this one.
    assert drv.kp_yaw < MD.KP_YAW
    assert drv.kp_yaw == pytest.approx(MD.GAIN_LIMIT * g["inertia_yaw"] / DT,
                                       rel=1e-9)
    assert drv.kp_yaw == pytest.approx(1.152, abs=0.02)


@needs_mars
def test_at_innates_own_two_millisecond_step_the_yaw_gain_is_nearly_theirs():
    """The guard is about the STEP, not about the robot.

    The same driver on the same robot at Innate's 2 ms keeps 96 % of their
    yaw gain (2.88 of 3.0), which is the check that the clamp is not quietly
    re-tuning their controller: it only removes what the integrator cannot
    carry. `mars._scene_spec()` is re-compiled at a different timestep here
    rather than mutated, so the shipped scene is untouched.
    """
    spec = mars._scene_spec()
    spec.option.timestep = 0.002
    model = spec.compile()
    drv = MarsDriver(model)
    assert drv.dt == pytest.approx(0.002)
    assert drv.kp_yaw == pytest.approx(2.88, abs=0.05)
    assert drv.kp_yaw / MD.KP_YAW > 0.95


# --------------------------------------------------------------- the drive

@needs_mars
def test_it_drives_a_straight_line_at_the_commanded_speed():
    """0.3 m/s for 5 s: 1.5 m, and nothing sideways.

    The whole point of the lateral and yaw terms — a velocity-only drive on
    FRICTIONLESS wheels (`mars.tune_contacts`) has nothing but those two
    terms to stop the base from crabbing or spinning, because the wheels
    cannot resist a sideways push themselves.

    MEASURED: x = 1.4994 m, y = -6.65 mm, yaw = -0.0045 rad. The bands are
    the task's 10 % / 5 cm / 0.05 rad.
    """
    model, data, drv = fresh()
    r = drive_for(drv, model, data, 5.0, 0.3, 0.0)
    assert r["finite"]
    assert r["x"] == pytest.approx(1.5, rel=0.10)
    assert abs(r["y"]) < 0.05, f"crabbed {r['y'] * 1000:.1f} mm sideways"
    assert abs(r["yaw"]) < 0.05, f"spun {r['yaw']:.4f} rad while driving straight"
    assert r["v_forward"] == pytest.approx(0.3, abs=0.01)


@needs_mars
def test_a_dead_forward_gain_leaves_it_where_it_started():
    """PLANTED: `KP_FORWARD = 0`.

    The break the straight-line case must catch, and the reason it asserts a
    distance rather than a speed: with no forward term the base never moves
    at all, and a test that only looked at `velocity()` would read 0.0 and
    have nothing to compare it against.
    """
    kp = MD.KP_FORWARD
    MD.KP_FORWARD = 0.0
    try:
        model, data, drv = fresh()
        assert drv.kp_forward == 0.0
        r = drive_for(drv, model, data, 5.0, 0.3, 0.0)
    finally:
        MD.KP_FORWARD = kp
    assert abs(r["x"]) < 0.001, "a zero forward gain still moved the robot"
    assert r["x"] != pytest.approx(1.5, rel=0.10)


@needs_mars
def test_it_turns_at_the_commanded_rate_and_counter_clockwise():
    """1.0 rad/s for 3 s: 3.0 rad, and in the right direction.

    The sign is asserted separately from the magnitude because a flipped
    yaw torque is the one drive bug that a magnitude band on |yaw| would
    wave through, and because CCW-positive is what every other bearing in
    this repo (the lidar's rays, `Detection.bearing`, the brains' turn
    commands) already means.
    """
    model, data, drv = fresh()
    r = drive_for(drv, model, data, 3.0, 0.0, 1.0)
    assert r["finite"]
    assert r["yaw"] == pytest.approx(3.0, rel=0.10)
    assert r["yaw"] > 0.0, "+wz must turn counter-clockwise"
    assert r["wz"] == pytest.approx(1.0, abs=0.05)
    # Turning in place: the base holds its position while it spins.
    assert math.hypot(r["x"], r["y"]) < 0.01


@needs_mars
def test_a_flipped_yaw_torque_turns_the_wrong_way():
    """PLANTED: the sign of the yaw torque written into `xfrc_applied`.

    Subclassed rather than monkeypatched on a constant, because negating
    `KP_YAW` is a different bug (positive feedback on the gain) and this is
    the one worth guarding: the mapping from a body-frame command to a
    world-frame torque, which is the half of `drive()` a reader has to trust.
    """
    class FlippedYaw(MarsDriver):
        def drive(self, data):
            super().drive(data)
            data.xfrc_applied[self.base_id][5] *= -1.0

    model = mars.model()
    data = mujoco.MjData(model)
    drv = FlippedYaw(model)
    drv.spawn(data, 0.0, 0.0, 0.0)
    r = drive_for(drv, model, data, 3.0, 0.0, 1.0)
    assert r["yaw"] < 0.0, "a flipped yaw torque still turned counter-clockwise"
    assert r["yaw"] != pytest.approx(3.0, rel=0.10)


@needs_mars
def test_without_the_step_guard_the_yaw_loop_diverges():
    """PLANTED: `GAIN_LIMIT = 100`, i.e. Innate's KP_YAW verbatim at 5 ms.

    This is the case that justifies the one constant in `mars_drive.py` that
    is not Innate's. MEASURED with the guard off: 1.0 rad/s commanded for 3 s
    ends at yaw 7.13 rad with |wz| peaking at 13.3 rad/s — a base that
    oscillates itself across the room rather than turning. The governor is
    what keeps it finite (6 rad/s), which is exactly the "recoverable thump
    instead of a NaN" it exists for, and why a NaN check alone would not
    have found this.
    """
    limit = MD.GAIN_LIMIT
    MD.GAIN_LIMIT = 100.0
    try:
        model, data, drv = fresh()
        assert drv.kp_yaw == MD.KP_YAW, "the plant did not take"
        peak = 0.0
        for k in range(int(round(3.0 / DT))):
            if k % 10 == 0:
                drv.set_cmd(0.0, 1.0, data.time)
            drv.step(data)
            mujoco.mj_step(model, data)
            peak = max(peak, abs(drv.velocity(data)[2]))
    finally:
        MD.GAIN_LIMIT = limit
    assert peak > 10.0, f"peak |wz| was only {peak:.2f} rad/s — did it diverge?"
    assert drv.pose(data)[2] != pytest.approx(3.0, rel=0.10)


@needs_mars
def test_a_command_past_the_envelope_is_clamped_not_obeyed():
    """cmd 2.0 m/s -> 0.800 m/s, measured on the robot and not just in the
    accessor.

    Two different limits could have produced a slower robot — the command
    envelope (0.8) and the safety governor (2.0) — so the speed is measured
    against the envelope and the governor is shown NOT to be what bit: at
    0.8 m/s the base is nowhere near the 2.0 m/s clamp.
    """
    model, data, drv = fresh()
    drv.set_cmd(2.0, 0.0, 0.0)
    assert drv.cmd() == (0.8, 0.0)
    r = drive_for(drv, model, data, 3.0, 2.0, 0.0)
    assert r["v_forward"] == pytest.approx(0.8, abs=0.01)
    assert r["v_forward"] < MD.MAX_BASE_LINEAR_SPEED
    assert r["x"] == pytest.approx(0.8 * 3.0, rel=0.05)


@needs_mars
def test_the_governor_clamps_a_velocity_nothing_commanded():
    """The safety net on the STATE — innate core._apply_control's first block.

    A contact impulse, a bad hull seam or a test writing into `qvel` can put
    a speed on the base that no command asked for; the governor is what makes
    that a thump instead of a robot that leaves the map. Injected at 10 m/s,
    five times the clamp, and read back on the next step.
    """
    model, data, drv = fresh()
    data.qvel[drv.base_dadr[0]] = 10.0
    drv.drive(data)                      # the clamp is applied before the PD
    assert abs(float(data.qvel[drv.base_dadr[0]])) == pytest.approx(
        MD.MAX_BASE_LINEAR_SPEED, abs=1e-9)
    data.qvel[drv.base_dadr[2]] = -30.0
    drv.drive(data)
    assert float(data.qvel[drv.base_dadr[2]]) == pytest.approx(
        -MD.MAX_BASE_ANGULAR_SPEED, abs=1e-9)


# ------------------------------------------------------------ the watchdog

@needs_mars
def test_a_stale_command_stops_the_base():
    """One command, then 1.5 s of silence: |v| < 0.02 m/s.

    A command is a lease. MEASURED: the base is under 0.02 m/s 0.52 s after
    the last `set_cmd` — the 0.5 s timeout plus the ~20 ms the forward PD
    needs to shed 0.3 m/s — and at 1.5 s it is 0.0 to six decimals.
    """
    model, data, drv = fresh()
    drive_for(drv, model, data, 2.0, 0.3, 0.0)
    assert drv.velocity(data)[0] == pytest.approx(0.3, abs=0.01)
    drv.set_cmd(0.3, 0.0, data.time)
    t_cmd = float(data.time)
    stopped_at = None
    for _ in range(int(round(1.5 / DT))):
        drv.step(data)
        mujoco.mj_step(model, data)
        if stopped_at is None and abs(drv.velocity(data)[0]) < 0.02:
            stopped_at = float(data.time) - t_cmd
    assert abs(drv.velocity(data)[0]) < 0.02
    assert stopped_at is not None
    assert MD.CMD_VEL_TIMEOUT_S <= stopped_at < MD.CMD_VEL_TIMEOUT_S + 0.1, (
        f"stopped {stopped_at:.3f} s after the command — the watchdog should "
        f"fire at {MD.CMD_VEL_TIMEOUT_S} s and the PD shed the speed in ~20 ms")


@needs_mars
def test_without_the_watchdog_a_stale_command_keeps_driving():
    """PLANTED: `CMD_VEL_TIMEOUT_S = 1e9`.

    The break the case above must catch. It also shows what the watchdog is
    protecting against in a room: 1.5 s of silence at 0.3 m/s is 45 cm of
    unattended travel, which in the playroom is the distance from the mat to
    the wall.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(MD, "CMD_VEL_TIMEOUT_S", 1e9)
        model, data, drv = fresh()
        r = drive_for(drv, model, data, 2.0, 0.3, 0.0, refresh_s=None)
    assert r["v_forward"] == pytest.approx(0.3, abs=0.01), (
        "with the watchdog disabled the base should still be driving")
    assert r["x"] > 0.5


@needs_mars
def test_an_uncommanded_driver_never_drives():
    """A driver nobody has spoken to is already expired.

    `_cmd_t` starts at -inf rather than 0.0 so that a MARS attached into a
    room and stepped before any brain claims it holds still, instead of
    waiting out half a second of watchdog with a zero command (which is the
    same thing here, and would not be if the command block were ever
    pre-filled).
    """
    model, data, drv = fresh()
    for _ in range(int(round(1.0 / DT))):
        drv.step(data)
        mujoco.mj_step(model, data)
    x, y, yaw = drv.pose(data)
    assert math.hypot(x, y) < 0.002 and abs(yaw) < 0.01
    assert drv.hold_pose() is not None, "it should have latched a hold pose"


# ------------------------------------------------------ station keeping

def _settle(drv, model, data, seconds: float = 1.0):
    """One zero command, then quiet: the hold latches after HOLD_SETTLE_S."""
    drv.set_cmd(0.0, 0.0, data.time)
    for _ in range(int(round(seconds / DT))):
        drv.step(data)
        mujoco.mj_step(model, data)
    hold = drv.hold_pose()
    assert hold is not None, "the hold pose never latched"
    return hold


def _shove(drv, model, data, hold, *, speed: float = 0.5,
           push_s: float = 0.0, wait_s: float = 1.0) -> tuple[float, float]:
    """Push the base along +x, then let go. Returns (peak, final) error, m."""
    dx = drv.base_dadr[0]
    n_push = int(round(push_s / DT))
    peak = 0.0
    for k in range(n_push + int(round(wait_s / DT))):
        if k <= n_push:
            data.qvel[dx] = speed
        drv.step(data)
        mujoco.mj_step(model, data)
        peak = max(peak, math.dist(drv.pose(data)[:2], hold[:2]))
    return peak, math.dist(drv.pose(data)[:2], hold[:2])


@needs_mars
def test_a_stopped_base_holds_the_pose_it_stopped_in():
    """The hold latches where the robot stopped, once, after 0.4 s of quiet.

    Innate's reason: the drive is velocity-only, so with a zero command
    nothing pulls the base back and the arm's reaction torque walks the
    robot. The latch is also why a disturbed base returns to where it
    STOPPED rather than to where the disturbance left it.
    """
    model, data, drv = fresh(x=0.25, y=-0.5, yaw=0.3)
    # Nothing is latched while the command is still live and non-zero.
    drv.set_cmd(0.1, 0.0, data.time)
    drv.step(data)
    assert drv.hold_pose() is None
    hold = _settle(drv, model, data)
    assert hold[0] == pytest.approx(0.25, abs=0.01)
    assert hold[1] == pytest.approx(-0.5, abs=0.01)
    assert hold[2] == pytest.approx(0.3, abs=0.01)
    # A live command releases it again.
    drv.set_cmd(0.2, 0.0, data.time)
    drv.step(data)
    assert drv.hold_pose() is None


@needs_mars
def test_the_hold_does_not_latch_before_the_settle_time():
    """HOLD_SETTLE_S outlasts a skill's per-camera-frame `cmd_vel` gaps.

    Their reason for the delay, and the thing it buys: a brain that publishes
    at 6 Hz (the lidar's rate) leaves 167 ms holes in its command stream, and
    a hold that latched on the first quiet step would anchor a base that is
    still meant to be driving.
    """
    model, data, drv = fresh()
    drv.set_cmd(0.0, 0.0, data.time)
    # 79 and 80 written out, NOT `HOLD_SETTLE_S / DT`: a loop sized by the
    # constant under test cannot fail when that constant is the bug — a
    # planted `HOLD_SETTLE_S = 0` made the range empty and this case passed.
    # (0.4 s at the lab's 5 ms step is 80 steps; `test_every_constant_...`
    # is what pins the 0.4 itself.)
    for k in range(79):
        drv.step(data)
        mujoco.mj_step(model, data)
        assert drv.hold_pose() is None, (
            f"latched {(k + 1) * DT:.3f} s in, before the 0.4 s settle")
    for _ in range(3):
        drv.step(data)
        mujoco.mj_step(model, data)
    assert drv.hold_pose() is not None, "never latched after HOLD_SETTLE_S"


@needs_mars
def test_a_shoved_base_comes_back_to_within_two_centimetres():
    """The task's case: stop, wait 1 s, inject 0.5 m/s once, wait 1 s.

    MEASURED: 0.89 mm peak, 0.20 mm after a second. HONEST ABOUT WHAT THAT
    MEASURES — a one-shot velocity injection is shed by the drive's own
    forward damper in ~7 ms (KP_FORWARD against 1.315 kg), so this case
    passes at 0.91 mm with station keeping switched OFF too. The
    discriminating version is the next one; this one is here because a 2 cm
    band on a single shove is what a room will actually do to a parked robot,
    and it should stay true.
    """
    model, data, drv = fresh()
    hold = _settle(drv, model, data)
    peak, final = _shove(drv, model, data, hold)
    assert peak < 0.02, f"a 0.5 m/s shove moved it {peak * 1000:.1f} mm"
    assert final < 0.02


@needs_mars
def test_a_sustained_push_is_undone_by_the_hold_and_not_by_the_damper():
    """The discriminating case: 200 ms of push, then let go.

    A shove the velocity damper CANNOT undo, because the damper only removes
    speed — it has no memory of where the robot was. MEASURED: 200 ms at
    0.5 m/s displaces the base 51 mm, and after 1 s the hold has pulled it
    back to 5.6 mm while the same push with `_station_keeping` stubbed out
    leaves all 51 mm exactly where it landed. That 9x is the measurement that
    station keeping does anything at all.
    """
    model, data, drv = fresh()
    hold = _settle(drv, model, data)
    _peak, with_hold = _shove(drv, model, data, hold, push_s=0.2)
    assert with_hold < 0.02, f"the hold left it {with_hold * 1000:.1f} mm out"

    model, data, drv = fresh()
    hold = _settle(drv, model, data)
    drv._station_keeping = lambda data, vx, wz: (0.0, 0.0, 0.0)   # PLANTED
    _peak, without = _shove(drv, model, data, hold, push_s=0.2)
    assert without > 0.04, (
        f"with station keeping off the push should stick; it ended "
        f"{without * 1000:.1f} mm out")
    assert without > 5 * with_hold


@needs_mars
def test_the_hold_also_puts_the_heading_back():
    """The yaw half of the hold — KP_HOLD_YAW, through an `atan2` wrap.

    MEASURED: 100 ms of 2.0 rad/s spin leaves the heading 0.00005 rad from
    the latched pose with the hold on, and 0.0177 rad — 350x more — with it
    off. The wrap is what makes this work at any heading rather than only
    near zero, and `pose()` deliberately does NOT wrap, so a robot that has
    turned three times still holds the heading it stopped at.
    """
    out = {}
    for label, on in (("on", True), ("off", False)):
        model, data, drv = fresh(yaw=3.0)
        hold = _settle(drv, model, data)
        if not on:
            drv._station_keeping = lambda data, vx, wz: (0.0, 0.0, 0.0)
        for k in range(int(round(1.1 / DT))):
            if k <= int(round(0.1 / DT)):
                data.qvel[drv.base_dadr[2]] = 2.0
            drv.step(data)
            mujoco.mj_step(model, data)
        out[label] = abs(drv.pose(data)[2] - hold[2])
    assert out["on"] < 0.005, f"the heading drifted {out['on']:.5f} rad"
    assert out["off"] > 10 * max(out["on"], 1e-6)


# ----------------------------------------------------------------- the arm

@needs_mars
def test_the_arm_stays_at_home_while_the_base_drives():
    """A driving base must not shake the arm off its targets.

    0.02 rad is the task's band; MEASURED 0.0034 rad, which is the same
    number `tests/test_body_conformance.py` measures for a STATIONARY MARS —
    so the drive adds nothing the servo cannot hold. It matters because the
    arm's pose is what Phase 4's observation reads, and because a folded arm
    that sags into the chassis is a contact, not a pose.
    """
    model, data, drv = fresh()
    r = drive_for(drv, model, data, 5.0, 0.3, 0.0)
    assert r["arm_err"] < 0.02, f"the arm drifted {r['arm_err']:.4f} rad"
    mimic, source, mult = mars.MIMIC_JOINT
    assert float(data.qpos[drv.adr[mimic][0]]) == pytest.approx(
        mult * mars.ARM_HOME[source], abs=0.02)


@needs_mars
def test_set_arm_holds_the_joints_it_is_given_and_homes_the_rest():
    """One key moves one joint; everything absent holds ARM_HOME.

    The contract a brain's `arm` intent channel will speak in Phase 3b — a
    gripper command should not require re-sending six joint angles, because
    a caller that has to restate the whole pose will eventually restate a
    stale one.
    """
    model, data, drv = fresh()
    drv.set_arm({"joint_head": -0.3})
    targets = drv.arm_targets()
    assert targets["joint_head"] == pytest.approx(-0.3)
    for name in mars.ARM_JOINTS:
        assert targets[name] == pytest.approx(mars.ARM_HOME[name])
    for _ in range(int(round(1.5 / DT))):
        drv.step(data)
        mujoco.mj_step(model, data)
    assert float(data.qpos[drv.adr["joint_head"][0]]) == pytest.approx(
        -0.3, abs=0.02)
    assert float(data.qpos[drv.adr["joint1"][0]]) == pytest.approx(
        mars.ARM_HOME["joint1"], abs=0.02)


@needs_mars
def test_set_arm_clamps_to_the_joint_range_and_refuses_a_name():
    """innate core.set_joint_target's clamp, and one deliberate difference.

    The clamp is theirs: the real gripper close is commanded past the
    mechanical stop, and unclamped in sim that target scissors the blades
    through each other. The `KeyError` is not — their node drops an unknown
    name silently, and here the caller is a policy action block where a typo
    that quietly stopped driving the gripper would read as a broken grasp.
    """
    model, _data, drv = fresh()
    lo, hi = model.joint("joint6").range
    drv.set_arm({"joint6": -0.6})
    assert drv.arm_targets()["joint6"] == pytest.approx(float(lo))
    assert float(lo) == pytest.approx(mars.GRIPPER_CLOSED_ON_AIR_RAD)
    drv.set_arm({"joint6": 99.0})
    assert drv.arm_targets()["joint6"] == pytest.approx(float(hi))
    with pytest.raises(KeyError, match="joint6M"):
        drv.set_arm({"joint6M": 0.0})
    with pytest.raises(KeyError, match="not MARS joints"):
        drv.set_arm({"elbow": 0.0})


# ------------------------------------------------------------- the odometry

@needs_mars
def test_pose_and_velocity_are_the_state_transformed_by_the_yaw():
    """`pose()` and `velocity()` against `data.qpos`/`qvel` done by hand.

    The accessors are what a brain's odometry channel and Phase 4's
    observation will read, so they are checked against the arithmetic rather
    than against themselves: a heading rotation applied the wrong way round
    is the `mj_objectVelocity` trap this repo already paid for once
    (AGENTS.md, "MuJoCo velocity-frame trap"), and it is invisible at yaw 0.
    """
    model, data, drv = fresh(x=0.4, y=-0.2, yaw=0.9)
    # A velocity that is neither along nor across the heading.
    data.qvel[drv.base_dadr[0]] = 0.7
    data.qvel[drv.base_dadr[1]] = -0.25
    data.qvel[drv.base_dadr[2]] = 0.4
    mujoco.mj_forward(model, data)

    qx, qy, qyaw = drv.base_qadr
    assert drv.pose(data) == (pytest.approx(float(data.qpos[qx])),
                              pytest.approx(float(data.qpos[qy])),
                              pytest.approx(float(data.qpos[qyaw])))
    yaw = float(data.qpos[qyaw])
    cos, sin = math.cos(yaw), math.sin(yaw)
    vx, vy = 0.7, -0.25
    v_forward, v_lateral, wz = drv.velocity(data)
    assert v_forward == pytest.approx(vx * cos + vy * sin)
    assert v_lateral == pytest.approx(-vx * sin + vy * cos)
    assert wz == pytest.approx(0.4)
    # And the sanity check the arithmetic exists for: driving forward at a
    # heading puts the DISTANCE on the heading, not on world x.
    model, data, drv = fresh(yaw=math.pi / 2)
    r = drive_for(drv, model, data, 3.0, 0.3, 0.0)
    assert r["y"] == pytest.approx(0.9, rel=0.10), "it did not drive along +y"
    assert abs(r["x"]) < 0.05
    assert r["v_forward"] == pytest.approx(0.3, abs=0.01)


# --------------------------------------------------------------- the spawn

@needs_mars
def test_spawn_puts_it_at_home_at_rest_with_no_force_left_over():
    """`spawn` writes state and clears the force rows it owns.

    The rows matter: they are MuJoCo's to keep until something overwrites
    them, so a spawn that left the last step's drive force in
    `xfrc_applied` would shove a freshly placed robot on the next `mj_step`
    even with no command. Checked by stepping the physics WITHOUT the driver
    afterwards, which is the only way to see a leftover force.
    """
    model, data, drv = fresh()
    drive_for(drv, model, data, 1.0, 0.8, 1.5)
    assert float(np.abs(data.xfrc_applied[drv.base_id]).max()) > 0.0

    drv.spawn(data, 1.0, -2.0, 0.5)
    assert drv.pose(data) == (pytest.approx(1.0), pytest.approx(-2.0),
                              pytest.approx(0.5))
    assert float(np.abs(data.qvel[drv._my_dofs]).max()) == 0.0
    assert float(np.abs(data.xfrc_applied[drv.base_id]).max()) == 0.0
    assert float(np.abs(data.qfrc_applied[drv._my_dofs]).max()) == 0.0
    assert drv.hold_pose() is None
    assert drv.cmd() == (0.0, 0.0)
    for name, target in mars.ARM_HOME.items():
        assert float(data.qpos[drv.adr[name][0]]) == pytest.approx(target,
                                                                   abs=1e-9)
    for _ in range(20):
        mujoco.mj_step(model, data)           # no driver: nothing should push
    assert math.hypot(*drv.velocity(data)[:2]) < 0.01


@needs_mars
def test_a_spawn_in_a_shared_world_touches_only_its_own_robot():
    """Two MARSes, one model: spawning one must not move the other.

    The reason `spawn` indexes `mars._home_qpos`'s output by this robot's own
    addresses instead of assigning the vector: that vector is full-width, and
    writing it whole would zero every other body in a composed world — which
    is precisely the kind of out-of-scope write that took 1381 passing tests
    to notice last time (AGENTS.md, "Out-parameter registries strand
    silently").
    """
    model = world_with(prefixes=("m0/", "m1/"))
    data = mujoco.MjData(model)
    a, b = MarsDriver(model, "m0/"), MarsDriver(model, "m1/")
    a.spawn(data, 0.0, 1.0, 0.0)
    b.spawn(data, 2.0, -1.0, 1.0)
    before = data.qpos.copy()
    a.spawn(data, 0.5, 1.0, 0.0)              # re-spawn ONLY m0
    assert b.pose(data) == (pytest.approx(2.0), pytest.approx(-1.0),
                            pytest.approx(1.0))
    changed = np.nonzero(np.abs(data.qpos - before) > 1e-12)[0]
    assert set(changed) <= set(a._home_adr.tolist()), (
        "a spawn wrote outside its own robot's qpos addresses")


@needs_mars
def test_two_mars_in_one_model_drive_independently():
    """A roster is N bodies in ONE model, and each driver owns its own rows.

    MEASURED: m0 commanded 0.3 m/s for 3 s ends at x = 0.899 m on its own
    row, while m1 commanded 1.0 rad/s turns to 2.9995 rad without leaving
    (0.001, -1.000). Two drivers writing the same `xfrc_applied` row, or
    sharing one `_hold`, would show up here as either robot doing the other's
    job — the failure mode `world/compose.py`'s prefixes exist to prevent.
    """
    model = world_with(prefixes=("m0/", "m1/"))
    data = mujoco.MjData(model)
    a, b = MarsDriver(model, "m0/"), MarsDriver(model, "m1/")
    assert a.base_id != b.base_id
    a.spawn(data, 0.0, 1.0, 0.0)
    b.spawn(data, 0.0, -1.0, 0.0)
    for k in range(int(round(3.0 / DT))):
        if k % 10 == 0:
            a.set_cmd(0.3, 0.0, data.time)
            b.set_cmd(0.0, 1.0, data.time)
        a.step(data)
        b.step(data)
        mujoco.mj_step(model, data)
    ax, ay, ayaw = a.pose(data)
    bx, by, byaw = b.pose(data)
    assert ax == pytest.approx(0.9, rel=0.10) and abs(ayaw) < 0.05
    assert ay == pytest.approx(1.0, abs=0.05)
    assert byaw == pytest.approx(3.0, rel=0.10)
    assert math.hypot(bx, by + 1.0) < 0.02, "the turning MARS translated"
    assert np.isfinite(data.qpos).all()


@needs_mars
def test_a_driver_for_a_prefix_that_is_not_in_the_model_is_refused():
    """A missing robot is a `KeyError` at construction, not a silent no-op.

    `world/compose.py` names its slots, and a typo'd prefix that produced a
    driver writing into body 0's force row would be a world where a ghost
    pushes the floor.
    """
    model = world_with(prefixes=("m0/",))
    MarsDriver(model, "m0/")                  # fine
    with pytest.raises(KeyError, match="attached under this prefix"):
        MarsDriver(model, "m7/")
    with pytest.raises(KeyError, match="attached under this prefix"):
        MarsDriver(model, "")


# --------------------------------------------------------------- the cadence

@needs_mars
def test_the_drive_cannot_be_decimated_to_the_control_tick():
    """PLANTED (as a design question, and answered): `drive()` every 4th step.

    Phase 3b has to know whether the base PD can ride the 50 Hz control tick
    like a policy does. It cannot, and this is the measurement: at a 20 ms
    tick the forward loop's gain is 200 * 0.02 / 1.315 = 3.04, past the 2.0
    bound, and the base ends up moving BACKWARDS at 2.6 m/s having gone
    nowhere. At 10 ms it rings (the position is right, the velocity is not).
    The arm servo is free to decimate; the drive is not.
    """
    model, data, drv = fresh()
    for k in range(int(round(3.0 / DT))):
        if k % 10 == 0:
            drv.set_cmd(0.3, 0.0, data.time)
        if k % 4 == 0:                        # a 20 ms drive tick
            drv.drive(data)
        drv.servo(data)
        mujoco.mj_step(model, data)
    x = drv.pose(data)[0]
    v = drv.velocity(data)[0]
    assert abs(v) > 1.0, (
        f"a 20 ms drive tick settled at {v:.3f} m/s — if this is stable now, "
        "re-measure the gain and rewrite this case and the module docstring")
    assert x != pytest.approx(0.9, rel=0.10)
    # And the same run at every physics step is the passing one.
    model, data, drv = fresh()
    r = drive_for(drv, model, data, 3.0, 0.3, 0.0)
    assert r["x"] == pytest.approx(0.9, rel=0.10)
    assert r["v_forward"] == pytest.approx(0.3, abs=0.01)
