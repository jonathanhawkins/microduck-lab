"""Odometry drift (roadmap 1.7): the pose a brain gets is dead reckoning
with a per-run scale, a gyro bias and step noise — exact under `ideal`,
drifting under the presets, re-zeroed on a respawn."""

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.world import Duck, Scenario, World, validate_scenario
from microduck_local.world.arena import OdomNoise
from microduck_local.world.scenario import ScenarioError

pytestmark = pytest.mark.skipif(
    not C.SCENE_WALK_XML.exists(), reason="microduck_rl checkout not found")


def _walk(preset: str, seed: int = 0):
    sc = Scenario(name="odom", floor=(6, 6), ducks=[Duck("d0", (0, 0, 0), None, None, None, None, preset)])
    w = World(sc, infer_for={"d0": onnx_infer(POLICIES_DIR / "alpha_walking.onnx")}, seed=seed)
    d = w.ducks["d0"]
    for _ in range(int(6.0 / C.CTRL_DT)):
        d.set_cmd(w.data, [0.3, 0, 0.3])
        w.step()
    true = (*d.trunk_pos(w.data)[:2], d.yaw(w.data))
    est = w.odom(d)
    return w, d, true, est


def test_ideal_is_the_truth_and_presets_drift():
    _, _, true, est = _walk("ideal")
    assert np.hypot(est[0] - true[0], est[1] - true[1]) < 1e-6 and abs(est[2] - true[2]) < 1e-9
    w, d, true, est = _walk("hostile")
    err = np.hypot(est[0] - true[0], est[1] - true[1])
    assert 0.01 < err < 1.0                     # drifted, but still an estimate of the same walk
    assert d.odom_preset == "hostile"
    # A respawn re-zeroes the estimate to the true spawn pose.
    w.reset_duck("d0")
    assert np.allclose(w.odom(d), d.spawn, atol=1e-6)
    # Switching presets live re-anchors at the current truth.
    w.set_odom_preset(d, "ideal")
    assert isinstance(d.odom_noise, OdomNoise) and d.odom_noise.scale_sigma == 0.0


def test_scenario_odom_field_round_trips_and_validates():
    sc = Scenario(name="o", floor=(4, 4), ducks=[Duck("d0", (0, 0, 0), None, "datasheet", "datasheet", "tidy", "datasheet")])
    raw = sc.to_dict()
    assert raw["ducks"][0]["odom"] == "datasheet" and validate_scenario(raw) == sc
    raw["ducks"][0]["odom"] = "wobbly"
    with pytest.raises(ScenarioError):
        validate_scenario(raw)
    raw["ducks"][0].pop("odom")
    assert validate_scenario(raw).ducks[0].odom == "ideal"
