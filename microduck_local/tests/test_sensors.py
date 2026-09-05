"""Contract locks for the simulated ToF (docs/sim-roadmap.md 1.2 / 1.9):
zone geometry, units, rate, range limits, and that the noise presets do
what their names promise. The world is a real duck from the upstream MJCF
standing in front of a wall, composed the way /sim composes rooms."""

import math

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.sensors import TofNoise, TofSensor, TofSpec, planar_fan, tof_fan
from microduck_local.sensors.ray import RayFan
from microduck_local.world import Box, Duck, Scenario, Wall, compose
from microduck_local.world.compose import DuckAddress, spawn_duck

pytestmark = pytest.mark.skipif(
    not C.SCENE_WALK_XML.exists(), reason="microduck_rl checkout not found")


def wall_world(dist: float = 1.0, height: float = 0.6, extra_boxes=()):
    """One duck at the origin facing +x, a wall `dist` m ahead."""
    sc = Scenario(name="tof-test", floor=(6.0, 6.0),
                  walls=[Wall((dist, -2.0), (dist, 2.0), height, 0.02)],
                  boxes=list(extra_boxes),
                  ducks=[Duck("d0", (0.0, 0.0, 0.0), None, "ideal")])
    m = compose(sc)
    d = mujoco.MjData(m)
    adr = DuckAddress.resolve(m, "d0")
    spawn_duck(m, d, adr, 0.0, 0.0, 0.0)
    mujoco.mj_forward(m, d)
    return m, d, adr


def test_tof_fan_is_unit_and_image_ordered():
    dirs = tof_fan(8, 8, 45.0)
    assert dirs.shape == (64, 3)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-12)
    # Row 0 tilts UP (+z), column 0 tilts LEFT (+y); the extreme rays sit
    # half a zone inside the 45° edge.
    assert dirs[0, 2] > 0 and dirs[-1, 2] < 0
    assert dirs[0, 1] > 0 and dirs[7, 1] < 0
    edge = math.degrees(math.atan2(dirs[0, 2], dirs[0, 0]))
    assert 22.5 - 45 / 8 < edge < 22.5


def test_planar_fan_spans_fov_left_first():
    d = planar_fan(5, 90.0)
    np.testing.assert_allclose(d[2], [1, 0, 0], atol=1e-12)
    assert d[0, 1] > 0 and d[-1, 1] < 0
    assert abs(math.degrees(math.atan2(d[0, 1], d[0, 0])) - 45.0) < 1e-9


def test_tof_site_frame_is_x_forward_in_stand_pose():
    m, d, adr = wall_world()
    R = d.site_xmat[adr.tof_site].reshape(3, 3)
    fwd = d.xmat[adr.trunk_body].reshape(3, 3)[:, 0]
    assert R[:, 0] @ fwd > 0.99      # site +x is the duck's forward
    assert R[:, 2] @ [0, 0, 1] > 0.99  # site +z is up


def test_tof_frame_shape_units_and_a_wall_at_one_metre():
    m, d, adr = wall_world(dist=1.0)
    tof = TofSensor(m, site="d0/tof", noise=TofNoise.ideal())
    f = tof.measure(d, t=0.0)
    assert f.depth_mm.shape == (8, 8) and f.depth_mm.dtype == np.uint16
    assert f.valid.shape == (8, 8) and f.valid.dtype == bool
    assert f.valid.all()
    # The aperture sits ~6.4 cm ahead of the trunk origin and ~25 cm up, so
    # the wall is ~0.94 m away on-axis; perspective adds up to sec(21°) ≈ 6 %
    # at the corners. The BOTTOM row looks down 20° and meets the floor
    # (0.25 / tan 20° ≈ 0.68 m) before the wall.
    centre = f.depth_mm[3:5, 3:5]
    assert np.all((centre > 920) & (centre < 950)), centre
    assert f.depth_mm[:7].min() > 900 and f.depth_mm[:7].max() < 1050
    assert 650 < f.depth_mm[7].min() and f.depth_mm[7].max() < 800
    np.testing.assert_array_equal(f.depth_mm, f.depth_mm[:, ::-1])   # symmetric
    assert (f.depth_mm == 0).sum() == 0


def test_tof_low_wall_top_rows_miss_bottom_rows_see_floor():
    # 15 cm wall: rays above it fly off to nothing within 4 m; the bottom
    # row looks down at the floor, closer than the wall.
    m, d, adr = wall_world(dist=1.0, height=0.15)
    f = TofSensor(m, site="d0/tof").measure(d, t=0.0)
    assert not f.valid[0].any() and not f.valid[1].any()
    assert f.valid[7].all() and f.depth_mm[7].max() < 900


def test_tof_min_range_saturates_and_max_range_misses():
    # A slab whose face sits ~6 mm ahead of the aperture (which is at
    # x ≈ 0.064): inside the 2 cm minimum range, so the zones saturate.
    close = Box((0.08, 0.0, 0.25), (0.02, 0.6, 0.5))
    m, d, adr = wall_world(dist=5.0, extra_boxes=[close])   # wall beyond 4 m
    f = TofSensor(m, site="d0/tof").measure(d, t=0.0)
    assert not f.valid.any()
    assert (f.depth_mm == 0).all()
    assert (f.truth_m[2:6] > 0).all() and (f.truth_m[2:6] < 0.02).all()
    m2, d2, _ = wall_world(dist=5.0)
    f2 = TofSensor(m2, site="d0/tof").measure(d2, t=0.0)
    assert not f2.valid[0:4].any()


def test_tof_rate_is_15hz_on_a_fixed_grid():
    m, d, adr = wall_world()
    tof = TofSensor(m, site="d0/tof", spec=TofSpec(rate_hz=15.0))
    assert tof.sample(d, 0.0) is not None
    assert tof.sample(d, 0.02) is None
    assert tof.sample(d, 0.05) is None
    got = tof.sample(d, 0.0667)
    assert got is not None and abs(tof.age(0.0667)) < 1e-9
    assert tof.sample(d, 0.10) is None
    # A late poll doesn't shift the grid: the next frame is still due at 2/15 s.
    assert tof.sample(d, 0.1334) is not None
    assert tof.age(0.20) == pytest.approx(0.20 - 0.1334)
    tof.reset()
    assert tof.last is None and tof.sample(d, 5.0) is not None


def test_tof_noise_presets_ideal_datasheet_hostile():
    m, d, adr = wall_world()
    truth = TofSensor(m, site="d0/tof").measure(d).truth_m
    assert truth.min() > 0
    ds = TofSensor(m, site="d0/tof", noise=TofNoise.datasheet(), seed=1)
    ho = TofSensor(m, site="d0/tof", noise=TofNoise.hostile(), seed=1)
    ds.measure(d, 0.0)
    ho.measure(d, 0.0)          # start both clocks so warm-up is over by t=10
    errs, drops_ds, drops_ho = [], [], []
    for k in range(200):
        t = 10.0 + k / 15
        f = ds.measure(d, t)
        e = f.depth_mm[f.valid] / 1000.0 - truth[f.valid]
        errs.append(e)
        drops_ds.append(1 - f.valid.mean())
        drops_ho.append(1 - ho.measure(d, t).valid.mean())
    e = np.concatenate(errs)
    core = e[np.abs(e) < 0.1]                  # outliers are their own knob
    assert len(core) > 0.99 * len(e)
    assert abs(core.mean()) < 0.005            # unbiased
    assert 0.005 < core.std() < 0.03           # datasheet-ish spread at ~1 m
    assert (np.abs(e) >= 0.1).any()            # …but outliers do happen
    assert 0.0 < np.mean(drops_ds) < 0.05      # a few dropouts
    assert np.mean(drops_ho) > np.mean(drops_ds) + 0.02
    # Warm-up: the first half second of a datasheet sensor is all invalid.
    warm = TofSensor(m, site="d0/tof", noise=TofNoise.datasheet())
    assert not warm.measure(d, 0.0).valid.any()
    assert warm.measure(d, 0.6).valid.any()


def test_ray_fan_hit_points_and_self_exclusion():
    m, d, adr = wall_world(dist=1.0)
    fan = RayFan(m, [[1, 0, 0], [0, 0, -1], [-1, 0, 0]], site="d0/tof", max_range=4.0)
    hits = fan.cast(d)
    assert hits.hit[0] and m.geom(int(hits.geomid[0])).name == "wall0"
    assert hits.hit[1] and m.geom(int(hits.geomid[1])).name == "floor"
    pts = fan.hit_points(d, hits)
    assert pts[0, 0] == pytest.approx(1.0, abs=0.015)
    assert pts[1, 2] == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(pts[2]).all() or hits.hit[2]   # backward ray: sky, or the duck itself


def test_two_ducks_see_each_other():
    sc = Scenario(name="pair", floor=(6.0, 6.0),
                  ducks=[Duck("a", (0.0, 0.0, 0.0), None, "ideal"),
                         Duck("b", (0.5, 0.0, math.pi), None, "ideal")])
    m = compose(sc)
    d = mujoco.MjData(m)
    for did, (x, y, yaw) in (("a", (0.0, 0.0, 0.0)), ("b", (0.5, 0.0, math.pi))):
        spawn_duck(m, d, DuckAddress.resolve(m, did), x, y, yaw)
    mujoco.mj_forward(m, d)
    f = TofSensor(m, site="a/tof").measure(d)
    # Duck b's head is ~0.4 m ahead, its top level with the sensor: the zones
    # at and below the optical axis see it, the ones above look over it. The
    # walk model's head is visual meshes with gaps, so ask for most, not all.
    block = f.valid[3:6, 2:6]
    seen = f.depth_mm[3:6, 2:6][block]
    on_duck = (seen > 300) & (seen < 460)
    # A duck seen head-on at half a metre is narrow: one or two zones per row.
    assert on_duck.sum() >= 4, f.depth_mm
    assert (f.depth_mm[3:6, 3:5] > 0).any()   # …and it is in the middle columns
    # …and the zones that look past its head land on the floor, not nowhere.
    assert ((seen >= 450) | on_duck).all()
