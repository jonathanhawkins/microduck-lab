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
