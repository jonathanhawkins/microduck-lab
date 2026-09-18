"""MARS's 360-degree planar LiDAR: does it read what is actually there?

A range sensor is only as good as its geometry, and every bug this file
guards is a geometry bug that LOOKS like a working sensor: a fan authored
around the wrong axis still returns 360 plausible numbers, a scan that
includes its own housing still returns 360 plausible numbers, and an offset
mount is a scan that is right about the wall and wrong about where the robot
is. So the cases are built around a box at a KNOWN distance and each one has
a planted break beside it (`microduck_local/AGENTS.md`, "A/B new tests
against planted regressions"):

    positive                            planted break              caught by
    --------------------------------    -----------------------    ----------
    the 0 deg ray reads the wall        the box removed            max_range
    the 0 deg ray points at +x          the fan authored +y        the wall is
                                        (`ccw=False`)              at ray 180
    the chassis is not in the scan      exclude the MOUNT only     360/360 hits
    a 0.10 m return is unresolvable     min_range = 0              it is valid
    `datasheet` perturbs the frame      `ideal`                    nothing moves

The measured numbers live in `sensors/lidar.py`'s module docstring, including
the one thing the sensor CAN see of its own robot: the folded arm.

Scenes are built here from `mars.robot_spec()` — `mars._scene_spec` is the
shipped standalone scene that the conformance suite and every other MARS
case compile, and a wall grown into it would change all of them.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.robots import mars
from microduck_local.robots.mars_drive import MarsDriver
from microduck_local.sensors import lidar as LD
from microduck_local.sensors.lidar import LidarNoise, LidarSensor
from microduck_local.sensors.ray import planar_fan

needs_mars = pytest.mark.skipif(
    not mars.mars_ready(), reason="MARS assets missing — uv run fetch-robot mars")

#: A wall: 0.1 m thick in x, 2 m wide in y, 0.5 m tall, centred 2 m ahead of
#: the spawn. Its NEAR FACE is therefore at x = 1.95, and the 0 deg ray's
#: reading is that minus the laser's own x offset — which is the arithmetic
#: `test_the_zero_degree_ray_reads_the_wall_from_the_laser_not_the_base`
#: exists to pin, because the two differ by 76 mm.
WALL = ("wall", (2.0, 0.0, 0.25), (0.05, 1.0, 0.25))
#: Something inside the device's 0.15 m minimum. Placed from the LASER, not
#: the base: 0.10 m ahead of the lens is inside the chassis box, which is
#: fine for a ray test (nothing is stepped) and is the only way to put a
#: return in that band at all on a robot 0.41 m long.
NEAR_BOX_M = 0.10


def scene(*, boxes=(), prefixes: tuple[str, ...] = ("",)) -> mujoco.MjModel:
    world = mujoco.MjSpec()
    world.option.timestep = C.PHYSICS_DT
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


def spawned(*, boxes=(), prefix: str = "", yaw: float = 0.0,
            x: float = 0.0, y: float = 0.0):
    """A scene with one MARS at HOME, forward-kinematics resolved."""
    model = scene(boxes=boxes, prefixes=(prefix,))
    data = mujoco.MjData(model)
    drv = MarsDriver(model, prefix)
    drv.spawn(data, x, y, yaw)
    return model, data, drv


def laser_offset_m(model: mujoco.MjModel, data: mujoco.MjData,
                   prefix: str = "") -> float:
    """The laser's x offset from the base origin, from the model itself."""
    laser = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                              prefix + mars.LIDAR_SITE)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                             prefix + mars.BASE_BODY)
    return float(data.xpos[laser][0] - data.xpos[base][0])


# ------------------------------------------------------------- the geometry

def test_the_fan_is_a_full_turn_of_equal_steps_starting_on_plus_x():
    """Innate's `lidar_scan` convention: `arange(n) * 2*pi/n`, CCW from +x.

    Asserted on `planar_fan(n, 360, ccw=True)` directly, with no model in the
    way, because this is the one property that makes every bearing in a frame
    mean something: ray 0 on +x, one ray per degree, no ray repeated, and the
    90th ray on +y rather than on -y.
    """
    d = planar_fan(360, 360.0, ccw=True)
    assert d.shape == (360, 3)
    assert np.allclose(d[:, 2], 0.0), "a planar scan must stay in the x-y plane"
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert np.allclose(d[0], [1.0, 0.0, 0.0])
    assert np.allclose(d[90], [0.0, 1.0, 0.0], atol=1e-12)     # +y is 90 deg
    assert np.allclose(d[180], [-1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(d[270], [0.0, -1.0, 0.0], atol=1e-12)
    ang = np.arctan2(d[:, 1], d[:, 0])
    assert np.allclose(np.diff(np.unwrap(ang)), math.radians(1.0))
    # No direction appears twice — a closed circle's two ends are one ray.
    assert len({tuple(np.round(v, 9)) for v in d}) == 360


def test_left_first_cannot_author_a_full_turn():
    """Why `ccw=` had to be added to `planar_fan` rather than reused.

    The left-first convention spans a FOV between two endpoints, which is
    right for an aperture and wrong for a turn: at 360 degrees it puts ray 0
    on -x, repeats +-180 as two rays, and leaves no ray on +x at all — the
    one bearing every obstacle question is asked about. This is the planted
    break for the case above, and the reason the flag exists.
    """
    d = planar_fan(360, 360.0)                      # the old behaviour
    assert np.allclose(d[0], [-1.0, 0.0, 0.0], atol=1e-12)
    ang = np.arctan2(d[:, 1], d[:, 0])
    assert np.min(np.abs(ang)) > math.radians(0.4), "a ray landed on +x"
    assert np.allclose(d[0], d[-1], atol=1e-9), "+-180 deg is two rays"
    # And the sector case it IS right for is unchanged.
    five = planar_fan(5, 90.0)
    assert np.allclose(np.arctan2(five[:, 1], five[:, 0]),
                       np.radians([45, 22.5, 0, -22.5, -45]))


@needs_mars
def test_the_sensor_mounts_on_the_laser_body_and_is_blind_to_the_chassis():
    """The mount, the exclusion, and where the exclusion comes from.

    `base_laser` is a jointless FRAME (it survives only because of the
    `fusestatic="false"` rewrite), so the geometry around the scanner —
    including `base_turret`, the box that models its own housing — belongs to
    `base_link`. The exclusion therefore defaults to the mount's PARENT,
    resolved from the model so a prefixed robot needs no id table.
    """
    model, data, _drv = spawned(boxes=(WALL,))
    lidar = LidarSensor(model)
    assert lidar.mount == "base_laser"
    assert lidar.exclude_body == "base_link"
    # The sensor package names the frame as a literal so it never imports the
    # robot package; this is what stops the two drifting apart.
    assert LD.DEFAULT_MOUNT == mars.LIDAR_SITE
    assert (LD.DEFAULT_MAX_RANGE_M, LD.DEFAULT_MIN_RANGE_M,
            LD.DEFAULT_RATE_HZ, LD.DEFAULT_N_RAYS) == (6.0, 0.15, 6.0, 360)
    assert lidar.fan.body_id == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "base_laser")
    assert lidar.fan.exclude_id == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert (lidar.n_rays, lidar.max_range, lidar.min_range, lidar.rate_hz) == (
        360, 6.0, 0.15, 6.0)
    with pytest.raises(KeyError, match="mount body"):
        LidarSensor(model, "no_such_body")


@needs_mars
def test_excluding_only_the_mount_returns_the_scanners_own_housing():
    """PLANTED: `exclude_body="base_laser"`, i.e. `RayFan`'s own default.

    MEASURED, and the whole reason `exclude_body=` was added to `RayFan`:
    the laser origin sits INSIDE `base_turret` (centre z 0.1828, half-height
    0.0152; the laser is at z 0.1717), so every one of the 360 rays leaves
    through that box and reports it at 0.039-0.101 m. A brain reading that
    scan sees a wall 4 cm away in every direction and never moves again — a
    failure that returns a full, plausible frame, which is why it is pinned.
    """
    model, data, _drv = spawned(boxes=(WALL,))
    frame = LidarSensor(model, exclude_body="base_laser").scan(data, 0.0)
    assert int((frame.truth_m < 6.0 - 1e-9).sum()) == 360
    assert frame.truth_m.max() < 0.11
    assert frame.truth_m[0] < 0.11, "the 0 deg ray should be eaten by the turret"


@needs_mars
def test_the_zero_degree_ray_reads_the_wall_from_the_laser_not_the_base():
    """The 0 deg ray: 2.0 m - 0.05 m of box - the laser's own x offset.

    MEASURED 2.0264 m against a predicted 2.0264 m. The offset is the point:
    the laser sits 76.4 mm BEHIND the base origin, so a consumer that
    assumed the scan came from the base frame would put every obstacle 76 mm
    nearer than it is — a third of the robot's own length, and enough to
    plan a path through a doorway it does not fit. `mount_pos` is how a
    brain gets the offset instead of assuming it away.
    """
    model, data, _drv = spawned(boxes=(WALL,))
    lidar = LidarSensor(model, base_body=mars.BASE_BODY)
    frame = lidar.scan(data, 0.0)
    offset = laser_offset_m(model, data)
    assert offset == pytest.approx(-0.0764, abs=0.001)
    want = 2.0 - 0.05 - offset
    assert frame.ranges[0] == pytest.approx(want, abs=0.02)
    assert frame.valid[0]
    assert frame.angles[0] == pytest.approx(0.0)
    # The frame carries the offset, in the base's heading frame.
    assert frame.mount_pos is not None
    assert frame.mount_pos[0] == pytest.approx(offset, abs=1e-6)
    assert frame.mount_pos[2] == pytest.approx(0.1716, abs=0.001)


@needs_mars
def test_the_sides_and_the_back_read_max_range():
    """+-90 deg and 180 deg with only a wall ahead: nothing, at 6 m.

    A miss reports `max_range` rather than a NaN or a zero, which is the
    LaserScan convention Innate's driver publishes: "nothing within 6 m" is
    information, and a consumer that averages a bin should get the far wall
    and not a hole.
    """
    model, data, _drv = spawned(boxes=(WALL,))
    frame = LidarSensor(model).scan(data, 0.0)
    for deg in (90, 180, 270):
        assert frame.ranges[deg] == pytest.approx(6.0), f"{deg} deg"
        assert frame.truth_m[deg] == pytest.approx(6.0)
        assert frame.valid[deg], "a miss is a valid reading of empty space"
    # The wall subtends a known arc and nothing outside it returns.
    hit = frame.truth_m < 6.0 - 1e-9
    assert not hit[27:334].any(), "something returned where there is nothing"


@needs_mars
def test_with_the_box_removed_every_ray_reads_max_range():
    """PLANTED: no box. The floor is below the scan plane, so an empty room
    is an empty scan — which is also the check that the 2.03 m reading above
    is the WALL and not a chassis geom at a coincidental distance."""
    model, data, _drv = spawned()
    frame = LidarSensor(model).scan(data, 0.0)
    arm = frame.truth_m < 6.0 - 1e-9
    assert int(arm.sum()) == 4, (
        "the only thing an empty-room scan may see is the folded arm")
    assert frame.ranges[0] == pytest.approx(6.0)
    assert frame.ranges[0] != pytest.approx(2.0264, abs=0.02)


@needs_mars
def test_the_folded_arm_is_the_one_part_of_the_robot_the_scan_can_see():
    """MEASURED: 4 of 360 rays, at 7-10 deg, from 0.1562 m off `link5`.

    Documented rather than hidden. At `ARM_HOME` the arm is folded over the
    chassis and its `link5` crosses the scan plane, exactly as it would
    occlude the real scanner — so a brain that treats any short return as an
    obstacle will brake for MARS's own elbow. The shadow MOVES with the arm,
    which is why the fix belongs to the consumer (ignore returns inside the
    footprint) and not to a hard-coded angle mask here.

    Pinned because both directions are bugs: more rays means something else
    started occluding the scanner, and zero rays means the robot stopped
    being visible to its own sensor and a real obstacle at 0.16 m might not
    be either.
    """
    model, data, _drv = spawned(boxes=(WALL,))
    lidar = LidarSensor(model)
    frame = lidar.scan(data, 0.0)
    near = np.nonzero(frame.truth_m < 0.5)[0]
    assert near.tolist() == [7, 8, 9, 10]
    assert frame.truth_m[near].min() == pytest.approx(0.1562, abs=0.01)
    assert frame.truth_m[near].max() == pytest.approx(0.1883, abs=0.01)
    link5 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link5")
    for i in near:
        geom = int(lidar._last_hits.geomid[i])
        assert int(model.geom_bodyid[geom]) == link5, (
            f"ray {i} deg is blocked by "
            f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom]))}")
    # It is a RETURN, not a dropout: above min_range, so it reads as valid.
    assert frame.valid[near].all()


@needs_mars
def test_a_return_inside_the_minimum_range_is_clipped_and_marked():
    """A box 0.10 m from the lens: reported 0.15 m, `valid` False.

    The device cannot resolve nearer than 0.15 m. Clipping errs toward
    "something is very close", which is the safe direction for anything that
    brakes, and `valid` is what says the number is a floor rather than a
    measurement. Marked from the TRUTH, before noise, so a sigma cannot argue
    a real 0.10 m return up into the valid band.
    """
    model, data, _drv = spawned()
    offset = laser_offset_m(model, data)
    # A thin box whose near face is NEAR_BOX_M ahead of the laser.
    half = 0.02
    face = offset + NEAR_BOX_M
    model, data, _drv = spawned(boxes=(("near", (face + half, 0.0, 0.25),
                                       (half, 0.3, 0.25)),))
    lidar = LidarSensor(model)
    frame = lidar.scan(data, 0.0)
    assert frame.truth_m[0] == pytest.approx(NEAR_BOX_M, abs=0.005)
    assert frame.ranges[0] == pytest.approx(lidar.min_range)
    assert not frame.valid[0], "a sub-minimum return must not read as valid"
    assert frame.as_payload()["mm"][0] == 0, "invalid is 0 on the wire"
    # PLANTED: no minimum at all, and the same return is reported verbatim.
    loose = LidarSensor(model, min_range=0.0).scan(data, 0.0)
    assert loose.valid[0]
    assert loose.ranges[0] == pytest.approx(NEAR_BOX_M, abs=0.005)


@needs_mars
def test_the_fan_turns_with_the_robot():
    """Spawned facing +y, the wall at world +x moves to ray 270 deg.

    The bearings are MOUNT-frame, so a robot that turns does not renumber
    its rays — the world moves through them. 270 deg and not 90: the wall is
    clockwise of a robot facing +y, and CCW numbering puts clockwise at the
    top of the range. MEASURED 1.9500 m, which is the wall's near face with
    the laser's offset now pointing along -y and contributing nothing to x.
    """
    model, data, _drv = spawned(boxes=(WALL,), yaw=math.pi / 2)
    frame = LidarSensor(model).scan(data, 0.0)
    nearest = int(np.argmin(frame.truth_m + 10.0 * (frame.truth_m < 0.5)))
    assert nearest == 270
    assert frame.truth_m[270] == pytest.approx(1.95, abs=0.02)
    assert frame.truth_m[0] == pytest.approx(6.0), "ahead is now empty"
    assert frame.angles[270] == pytest.approx(math.radians(270) - 2 * math.pi,
                                              abs=1e-6)


@needs_mars
def test_the_fan_authored_around_plus_y_misses_the_wall_ahead():
    """PLANTED: the sensor built on the left-first fan.

    The break that the 0 deg case must catch, and the one that is invisible
    without a known box: the scan still returns 360 numbers, still sees the
    wall, and is simply wrong about where it is — reported at ray 180 instead
    of ray 0, which to a brain is an obstacle BEHIND the robot.
    """
    model, data, _drv = spawned(boxes=(WALL,))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(LD, "planar_fan",
                   lambda n, fov_deg, *, ccw=False: planar_fan(n, fov_deg))
        frame = LidarSensor(model).scan(data, 0.0)
    assert frame.ranges[0] == pytest.approx(6.0), "ray 0 should now face -x"
    assert frame.ranges[0] != pytest.approx(2.0264, abs=0.02)
    assert frame.truth_m[180] == pytest.approx(2.0264, abs=0.02)


# ---------------------------------------------------------------- the noise

@needs_mars
def test_ideal_returns_the_truth_and_datasheet_does_not():
    """`ideal` is the default, and it is the identity.

    `datasheet` perturbs every RETURN and leaves the misses alone: a miss is
    not a measurement, so it has no measurement error — jittering it would
    be inventing a wall to be uncertain about. MEASURED at seed 3: all 53
    returns moved, 0 misses moved, rms 0.0164 m against a predicted sigma of
    0.0152 m at 2.03 m.
    """
    model, data, _drv = spawned(boxes=(WALL,))
    ideal = LidarSensor(model).scan(data, 0.0)
    assert np.allclose(ideal.ranges, ideal.truth_m)
    assert ideal.valid.all()

    noisy = LidarSensor(model, noise=LidarNoise.datasheet(), seed=3)
    frame = noisy.scan(data, 0.0)
    got = frame.truth_m < noisy.max_range - 1e-9
    moved = frame.ranges != frame.truth_m
    assert moved[got].all(), "every return should carry noise"
    assert not moved[~got].any(), "a miss is not a measurement"
    rms = float(np.sqrt(((frame.ranges - frame.truth_m)[got] ** 2).mean()))
    assert 0.005 < rms < 0.05, f"sigma is {rms:.4f} m at ~2 m"
    assert LidarNoise.preset("ideal") == LidarNoise()
    assert LidarNoise.preset("hostile").sigma_m > LidarNoise.datasheet().sigma_m
    with pytest.raises(ValueError, match="unknown lidar noise preset"):
        LidarNoise.preset("optimistic")


@needs_mars
def test_a_noisy_scan_is_reproducible_from_its_seed():
    """Two sensors, one seed, one answer — and a different seed differs.

    A world is replayed from a seed in this repo (`record-world`, the ring
    replay), so a sensor that drew from the global RNG would make two runs of
    the same scenario disagree for reasons nobody changed.
    """
    model, data, _drv = spawned(boxes=(WALL,))
    a = LidarSensor(model, noise=LidarNoise.hostile(), seed=11).scan(data, 0.0)
    b = LidarSensor(model, noise=LidarNoise.hostile(), seed=11).scan(data, 0.0)
    c = LidarSensor(model, noise=LidarNoise.hostile(), seed=12).scan(data, 0.0)
    assert np.array_equal(a.ranges, b.ranges)
    assert not np.array_equal(a.ranges, c.ranges)


# ----------------------------------------------------------------- the rate

@needs_mars
def test_two_polls_a_hundred_milliseconds_apart_are_one_scan():
    """6 Hz is a 167 ms period: the second poll gets nothing new.

    The rate is the device's, not the loop's — the world steps at 200 Hz and
    polls every step, and a sensor that answered every poll would hand a
    brain 33x the information the robot has. `last` still holds the newest
    scan, and `age()` is how a consumer finds out it is looking at a frame
    from 167 ms and 13 cm ago.
    """
    model, data, _drv = spawned(boxes=(WALL,))
    lidar = LidarSensor(model)
    first = lidar.maybe_scan(data, 0.0)
    assert first is not None
    assert lidar.maybe_scan(data, 0.1) is None
    assert lidar.last is first
    assert lidar.age(0.1) == pytest.approx(0.1)
    second = lidar.maybe_scan(data, 0.2)
    assert second is not None and second is not first
    assert second.t == pytest.approx(0.2)
    assert lidar.period == pytest.approx(1.0 / 6.0)


@needs_mars
def test_the_rate_is_scheduled_from_the_grid_and_not_from_the_poll():
    """Six scans in a second, and no drift.

    Scheduling `next = t + period` instead of from the grid would turn a
    6 Hz sensor into a 5.9 Hz one over a long run, because a poll always
    arrives a little after the frame is due. MEASURED: scans at 0, 0.17,
    0.335, 0.5, 0.67, 0.835 s — each within one physics step of k/6.
    """
    model, data, _drv = spawned(boxes=(WALL,))
    lidar = LidarSensor(model)
    times = []
    for k in range(int(round(1.0 / C.PHYSICS_DT))):
        t = k * C.PHYSICS_DT
        if lidar.maybe_scan(data, t) is not None:
            times.append(t)
    assert len(times) == 6
    for k, t in enumerate(times):
        assert t == pytest.approx(k / 6.0, abs=C.PHYSICS_DT)
    lidar.reset()
    assert lidar.last is None and lidar.age(1.0) is None
    assert lidar.maybe_scan(data, 0.0) is not None, "reset re-arms the sensor"


# -------------------------------------------------------------- the payload

@needs_mars
def test_the_payload_is_millimetre_integers_and_a_rebuildable_axis():
    """The wire shape: `t`, `a0`, `da`, and 360 uint16 millimetres.

    `TofFrame.as_payload`'s units and its convention that 0 means NO reading,
    so the two range sensors do not disagree on the wire. `a0`/`da` instead
    of 360 angles because the bearings never change — a consumer rebuilds
    them, and a 6 Hz stream carries 0.7 kB instead of 3 kB.
    """
    model, data, _drv = spawned(boxes=(WALL,))
    frame = LidarSensor(model).scan(data, 0.123456)
    p = frame.as_payload()
    assert sorted(p) == ["a0", "da", "mm", "t"]
    assert p["t"] == pytest.approx(0.1235, abs=1e-4)
    assert p["a0"] == 0.0
    assert p["da"] == pytest.approx(math.radians(1.0), abs=1e-5)
    assert len(p["mm"]) == 360
    assert all(isinstance(v, int) for v in p["mm"])
    assert p["mm"][0] == pytest.approx(round(frame.ranges[0] * 1000), abs=1)
    assert p["mm"][90] == 6000
    rebuilt = np.array([p["a0"] + k * p["da"] for k in range(360)])
    assert np.allclose(rebuilt[:180], frame.angles[:180], atol=1e-4)
    import json
    assert json.loads(json.dumps(p))["mm"][0] == p["mm"][0]


# --------------------------------------------------------------- the prefix

@needs_mars
def test_a_prefixed_mars_scans_with_no_id_table():
    """`m0/base_laser` mounts, and `m0/base_link` is excluded automatically.

    What a `/sim` room needs, and all it needs: one string. A roster of two
    MARSes gets two sensors that see each other's chassis (group 0 collision
    geoms) and not their own.
    """
    model = scene(boxes=(WALL,), prefixes=("m0/", "m1/"))
    data = mujoco.MjData(model)
    a, b = MarsDriver(model, "m0/"), MarsDriver(model, "m1/")
    a.spawn(data, 0.0, 0.0, 0.0)
    # m1 sits 0.8 m to m0's left — still in front of the 2 m wall, and close
    # enough that its chassis is inside m0's 6 m range.
    b.spawn(data, 0.0, 0.8, 0.0)
    la = LidarSensor(model, "m0/base_laser", base_body="m0/base_link")
    lb = LidarSensor(model, "m1/base_laser")
    assert la.exclude_body == "m0/base_link"
    assert lb.exclude_body == "m1/base_link"
    fa, fb = la.scan(data, 0.0), lb.scan(data, 0.0)
    offset = laser_offset_m(model, data, "m0/")
    assert fa.ranges[0] == pytest.approx(2.0 - 0.05 - offset, abs=0.02)
    assert fb.ranges[0] == pytest.approx(2.0 - 0.05 - offset, abs=0.02)
    # m0 looking left sees m1's chassis — 0.8 m of centres less m1's 0.091 m
    # half-width — and not its own.
    assert fa.truth_m[90] == pytest.approx(0.709, abs=0.05), (
        "the neighbouring MARS should be in the scan")
    assert fa.truth_m[270] == pytest.approx(6.0), "nothing is to the right"


@needs_mars
def test_the_sensor_refuses_a_geometry_it_cannot_measure():
    """Constructor guards, because a sensor is built once and read forever.

    A zero-ray fan, a minimum past the maximum or a zero rate would each
    produce a sensor that returns something — an empty frame, an all-invalid
    frame, a division by zero on the first poll — and the first two would do
    it silently.
    """
    model = scene()
    with pytest.raises(ValueError, match="n_rays"):
        LidarSensor(model, n_rays=0)
    with pytest.raises(ValueError, match="min_range < max_range"):
        LidarSensor(model, min_range=7.0)
    with pytest.raises(ValueError, match="rate_hz"):
        LidarSensor(model, rate_hz=0.0)
    with pytest.raises(KeyError, match="base_body"):
        LidarSensor(model, base_body="pelvis")
