"""Locks for the geometric detector (roadmap 1.3): frustum, occlusion, size
gating, class pass-through, noise presets, and the capture-to-availability
latency — on real ducks composed into one world."""

import math

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.sensors import Detector, DetectorNoise, DetectorSpec, Target
from microduck_local.world import Ball, Duck, Scenario, Wall, compose
from microduck_local.world.compose import DuckAddress, spawn_duck

pytestmark = pytest.mark.skipif(
    not C.SCENE_WALK_XML.exists(), reason="microduck_rl checkout not found")


def world(ducks, walls=(), balls=()):
    sc = Scenario(name="det", floor=(8, 8), walls=list(walls), balls=list(balls),
                  ducks=[Duck(i, s, None, None) for i, s in ducks])
    m = compose(sc)
    d = mujoco.MjData(m)
    for i, s in ducks:
        spawn_duck(m, d, DuckAddress.resolve(m, i), *s)
    mujoco.mj_forward(m, d)
    return m, d


def targets(m, ducks=(), balls=()):
    out = [Target(i, "duck", mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{i}/trunk_base"), 0.10) for i in ducks]
    out += [Target(f"ball{k}", "ball", mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ball{k}"), 0.035) for k in balls]
    return out


def test_camera_site_is_x_forward_and_sees_a_duck_ahead_not_behind():
    m, d = world([("a", (0, 0, 0)), ("b", (0.6, 0.0, math.pi)), ("c", (-0.6, 0.0, 0.0))])
    det = Detector(m, site="a/head_camera", targets=targets(m, ducks=("a", "b", "c")))
    f = det.capture(d, 0.0)
    names = {x.name: x for x in f.detections}
    assert "b" in names and "c" not in names and "a" not in names   # ahead yes, behind no, self never
    b = names["b"]
    assert abs(b.bearing) < 0.05 and -0.4 < b.elevation < 0     # its trunk sits below the camera
    assert 0.35 < b.range_est < 0.8 and b.conf == 1.0


def test_bearing_sign_is_left_positive_and_fov_limits():
    m, d = world([("a", (0, 0, 0)), ("l", (0.5, 0.25, 0)), ("r", (0.5, -0.25, 0)), ("far", (0.3, 0.8, 0))])
    det = Detector(m, site="a/head_camera", targets=targets(m, ducks=("l", "r", "far")))
    names = {x.name: x for x in det.capture(d, 0.0).detections}
    assert names["l"].bearing > 0.3 and names["r"].bearing < -0.3
    assert "far" not in names          # ~70° off-axis: outside a 62° FOV


def test_wall_occludes_and_size_gates_at_range():
    m, d = world([("a", (0, 0, 0)), ("b", (1.2, 0.0, math.pi))], walls=[Wall((0.6, -1), (0.6, 1), 0.6)])
    det = Detector(m, site="a/head_camera", targets=targets(m, ducks=("b",)))
    assert det.capture(d, 0.0).detections == []
    m, d = world([("a", (0, 0, 0)), ("b", (3.9, 0.0, math.pi))])
    det = Detector(m, site="a/head_camera", targets=targets(m, ducks=("b",)), seed=0)
    # A 10 cm radius at 3.9 m is ~2.9° wide: found only sometimes, with low confidence.
    found = [det.capture(d, 0.0).detections for _ in range(200)]
    hits = [f[0] for f in found if f]
    assert 0 < len(hits) < 200
    # conf = p_find × U(floor, 1) with p_find ≈ 0.66 at 2.9°: found about two
    # times in three, never with more confidence than that.
    assert 0.4 < len(hits) / 200 < 0.9 and all(h.conf < 0.7 for h in hits)


def test_ball_class_and_range_from_width():
    # A floor ball half a metre out is BELOW a 48° vertical field of view from
    # a camera 25 cm up (−26°): the head has to look down for it. At 0.8 m
    # it just makes it in.
    m, d = world([("a", (0, 0, 0))], balls=[Ball((0.5, 0.0))])
    det = Detector(m, site="a/head_camera", targets=targets(m, balls=(0,)))
    assert det.capture(d, 0.0).detections == []
    m, d = world([("a", (0, 0, 0))], balls=[Ball((0.8, 0.0))])
    det = Detector(m, site="a/head_camera", targets=targets(m, balls=(0,)))
    f = det.capture(d, 0.0).detections
    assert len(f) == 1 and f[0].cls == "ball"
    assert -0.4 < f[0].elevation < -0.2
    assert abs(f[0].range_est - math.hypot(0.8 - 0.064, 0.21)) < 0.08


def test_latency_and_rate():
    m, d = world([("a", (0, 0, 0)), ("b", (0.6, 0.0, math.pi))])
    det = Detector(m, site="a/head_camera", targets=targets(m, ducks=("b",)),
                   noise=DetectorNoise(latency_s=0.05), spec=DetectorSpec(rate_hz=10.0))
    assert det.sample(d, 0.00) is None                  # captured at 0, not available yet
    assert det.sample(d, 0.02) is None and det.last is None
    got = det.sample(d, 0.06)
    assert got is not None and got.t == 0.0 and det.age(0.06) == pytest.approx(0.06)
    assert det.sample(d, 0.08) is None                  # next capture is due at 0.1 …
    assert det.sample(d, 0.12) is None                  # … taken on this first tick after, available at 0.17
    assert det.sample(d, 0.16) is None
    assert det.sample(d, 0.18).t == pytest.approx(0.12)


def test_noise_presets_bearing_spread_and_ghosts():
    m, d = world([("a", (0, 0, 0)), ("b", (0.6, 0.0, math.pi))])
    tg = targets(m, ducks=("b",))
    ideal = Detector(m, site="a/head_camera", targets=tg).capture(d, 0.0).detections[0]
    ds = Detector(m, site="a/head_camera", targets=tg, noise=DetectorNoise.datasheet(), seed=2)
    ho = Detector(m, site="a/head_camera", targets=tg, noise=DetectorNoise.hostile(), seed=2)
    b_ds, ghosts_ds, miss_ds, miss_ho = [], 0, 0, 0
    for _ in range(300):
        f = ds.capture(d, 0.0).detections
        real = [x for x in f if x.name == "b"]
        ghosts_ds += len(f) - len(real)
        miss_ds += not real
        b_ds += [x.bearing for x in real]
        miss_ho += not [x for x in ho.capture(d, 0.0).detections if x.name == "b"]
    assert abs(np.mean(b_ds) - ideal.bearing) < 0.01
    assert 0.01 < np.std(b_ds) < 0.03                   # ~1° bearing noise
    assert 0 < miss_ds < 30 and ghosts_ds >= 1
    assert miss_ho > miss_ds
    with pytest.raises(ValueError):
        DetectorNoise.preset("lidar")


def test_frames_carry_the_camera_pose_and_toys_do_not_occlude():
    """A detection frame says where the camera was (height, depression) —
    the brain ranges floor objects from it — and a toy in the beak does
    not hide the basket behind it (toys are geom group 4, see compose)."""
    from microduck_local.world import Basket, Duck, Pickable, Scenario, World
    sc = Scenario(name="occl", floor=(4, 4), ducks=[Duck("d0", (0, 0, 0), None, "ideal", "ideal")],
                  pickables=[Pickable("t0", "block", (0.6, 0.0))], basket=Basket((1.5, 0.0), (0.3, 0.3), 0.06))
    w = World(sc)
    d = w.ducks["d0"]
    fr = d.detector.capture(w.data, 0.0)
    assert 0.15 < fr.cam_z < 0.3 and -0.1 < fr.cam_pitch < 0.5     # head height; level at the spawn pose
    classes = set()
    for _ in range(20):                                             # a 2 cm toy at 0.6 m is found ~9 frames in 10
        classes |= {x.cls for x in d.detector.capture(w.data, 0.0).detections}
    assert "toy" in classes and "basket" in classes
    # Park the toy right on the line of sight to the basket marker: still seen.
    import mujoco
    q = w.model.jnt_qposadr[mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_JOINT, "t0_free")]
    cam = w.data.site_xpos[d.detector.site_id]
    mk = w.data.xpos[mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, "basket_marker")]
    w.data.qpos[q:q + 3] = cam + 0.3 * (mk - cam)
    mujoco.mj_forward(w.model, w.data)
    assert "basket" in {x.cls for x in d.detector.capture(w.data, 0.0).detections}
