"""Room mapping in the odometry frame (brain/mapping.py): a duck standing in
a walled room fills free cells between itself and the walls and marks the
walls occupied, using only the ToF frames' mount pose and its odometry."""

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.brain.mapping import GridSpec, OccupancyGrid
from microduck_local.world import Duck, Scenario, Wall, World

pytestmark = pytest.mark.skipif(
    not C.SCENE_WALK_XML.exists(), reason="microduck_rl checkout not found")


def test_frames_carry_the_mount_pose_and_a_wall_ahead_is_mapped():
    sc = Scenario(name="map", floor=(4, 4), ducks=[Duck("d0", (0, 0, 0), None, "ideal", None)],
                  walls=[Wall((1.0, -1.0), (1.0, 1.0), 0.4, 0.02)])
    w = World(sc)
    d = w.ducks["d0"]
    grid = OccupancyGrid(GridSpec(size=(4.0, 4.0), res=0.05))
    used = 0
    for _ in range(int(0.5 / C.CTRL_DT)):
        w.step()
        fr = d.tof.last
        if fr is not None:
            assert fr.mount_pos is not None and fr.mount_rot.shape == (3, 3) and fr.dirs_local.shape == (8, 8, 3)
            assert 0.08 < fr.mount_pos[2] < 0.3                  # the head, above the trunk origin
            used += grid.update(fr, w.odom(d))
    assert used >= 3 and grid.frames == used
    occ = grid.occupied()
    ys, xs = np.nonzero(occ)
    assert len(xs) > 0
    wall_x = (xs + 0.5) * grid.spec.res - 2.0
    assert np.all(np.abs(wall_x - 1.0) < 0.12)                    # only the wall, where the wall is
    # Cells between the duck and the wall were traced free.
    free = grid.logodds <= -1.0
    i, j = grid.cell(0.5, 0.0)
    assert free[j, i]
    pl = grid.payload()
    assert pl["nx"] == 80 and len(pl["cells"]) == 80 * 80 and "2" in pl["cells"] and "1" in pl["cells"]


def _drift_run(match: bool, bias_deg: float, seconds: float = 14.0):
    """A duck walking at a corner under a KNOWN odometry yaw bias, mapping
    as it goes: (pose error at the end, fraction of occupied cells within
    10 cm of a true wall line, the grid)."""
    import math

    from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
    from microduck_local.world.arena import OdomNoise
    sc = Scenario(name="corner", floor=(4, 4), ducks=[Duck("d0", (-0.8, -0.6, 0.3), None, "datasheet", None)],
                  walls=[Wall((1.0, -1.5), (1.0, 1.5), 0.4, 0.02), Wall((-1.5, 1.0), (1.5, 1.0), 0.4, 0.02)])
    w = World(sc, infer_for={"d0": onnx_infer(POLICIES_DIR / "alpha_walking.onnx")}, seed=0)
    d = w.ducks["d0"]
    d.odom_noise = OdomNoise(yaw_bias_sigma=1.0)      # the preset only has to be non-trivial: the bias is set below
    d._odom_yaw_bias = math.radians(bias_deg)
    grid = OccupancyGrid(GridSpec(size=(4.0, 4.0), res=0.05, match=match))
    for _ in range(int(seconds / C.CTRL_DT)):
        d.set_cmd(w.data, [0.25, 0, 0] if w.t < 5.0 else ([0.0, 0, 1.0] if w.t < 7.2 else [0.25, 0, 0]))
        w.step()
        grid.update(d.tof.last, w.odom(d))
    assert d.falls == 0
    ys, xs = np.nonzero(grid.occupied())
    px, py = (xs + 0.5) * grid.spec.res - 2.0, (ys + 0.5) * grid.spec.res - 2.0
    err = np.minimum(np.abs(px - 1.0), np.abs(py - 1.0))
    pose = grid.pose if match else grid.correct(w.odom(d))
    truth = d.trunk_pos(w.data)
    return float(np.hypot(pose[0] - truth[0], pose[1] - truth[1])), float(np.mean(err < 0.10)), grid


def test_wall_line_matching_closes_the_loop_under_yaw_drift():
    """A 1.5°/s gyro bias (1.5× the hostile preset's σ) fans the raw map out;
    matched against its own walls the map stays tight and the pose error
    shrinks by half — and ideal odometry is left alone (a matcher that
    chased line-fit noise turned it into a 12° random walk)."""
    raw_err, raw_ok, _ = _drift_run(match=False, bias_deg=1.5)
    fix_err, fix_ok, grid = _drift_run(match=True, bias_deg=1.5)
    assert grid.corrections > 5 and grid.pose is not None
    assert fix_err < 0.7 * raw_err, (raw_err, fix_err)                 # measured 0.21 → 0.12 m
    assert fix_ok > raw_ok + 0.1 and fix_ok > 0.75, (raw_ok, fix_ok)   # measured 0.64 → 0.85
    pl = grid.payload()
    assert len(pl["offset"]) == 3 and pl["corrections"] == grid.corrections and len(pl["pose"]) == 3
    assert abs(pl["offset"][2]) > 0.05                    # it found the bias
    id_err, id_ok, ideal = _drift_run(match=True, bias_deg=0.0)
    assert id_err < 0.05 and id_ok > 0.95 and abs(ideal.offset[2]) < np.deg2rad(2.0), (id_err, ideal.offset)
