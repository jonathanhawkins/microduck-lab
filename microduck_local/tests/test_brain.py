"""The wander controller (roadmap 2.3): decisions from synthetic ToF frames,
dropouts treated as unknown, turn memory, the stuck spin, and one end-to-end
check that it steers a real duck away from a wall it would otherwise walk into."""

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.brain import Wander, WanderParams, wander_from_tof


def frame(fill_mm: int, **cols) -> np.ndarray:
    f = np.full((8, 8), fill_mm, np.uint16)
    for k, v in cols.items():          # c3=400 → column 3 reads 400 mm in the counted rows
        f[2:7, int(k[1:])] = v
    return f


def test_clear_ahead_cruises():
    assert wander_from_tof(frame(3000)) == (0.3, 0.0, 0.0)
    assert wander_from_tof(np.zeros((8, 8), np.uint16)) == (0.3, 0.0, 0.0)   # nothing reported = nothing known


def test_wall_ahead_slows_and_turns_toward_the_open_side():
    # Right half near, left half far: turn LEFT (+wz), slower.
    f = frame(3000, c4=500, c5=500, c6=500, c7=500)
    vx, vy, wz = wander_from_tof(f)
    assert 0 < vx < 0.3 and wz == pytest.approx(0.8)
    f = frame(3000, c0=500, c1=500, c2=500, c3=500)
    vx, vy, wz = wander_from_tof(f)
    assert 0 < vx < 0.3 and wz == pytest.approx(-0.8)


def test_boxed_in_stops_and_spins():
    vx, vy, wz = wander_from_tof(frame(200))
    assert vx == 0.0 and abs(wz) == 1.0


def test_sky_rows_and_dropouts_are_ignored():
    f = frame(3000)
    f[0:2, :] = 150                 # something "near" only in the top rows: ignored
    assert wander_from_tof(f) == (0.3, 0.0, 0.0)
    f = frame(3000, c3=250)
    valid = np.ones((8, 8), bool)
    valid[:, 3] = False             # that column is all dropouts: unknown, not near
    assert wander_from_tof(f, valid) == (0.3, 0.0, 0.0)
    assert wander_from_tof(f)[0] == 0.0    # …but trusted, it stops


def test_turn_memory_and_stuck_spin():
    w = Wander(WanderParams(), stuck_s=1.0, unstick_s=0.5)
    assert w.step(None, None, 0.0) == (0.0, 0.0, 0.0) and w.state == "blind"
    f = frame(3000, c4=500, c5=500, c6=500, c7=500)
    assert w.step(f, None, 0.1)[2] > 0 and w.state == "steer"
    # A nearly symmetric frame keeps the remembered direction.
    sym = frame(3000, c2=500, c3=500, c4=500, c5=500)
    assert w.step(sym, None, 0.2)[2] > 0
    # Forward command, no progress for > stuck_s → an unstick spin.
    clear = frame(3000)
    for k in range(12):
        out = w.step(clear, None, 1.0 + k * 0.1, speed=0.0)
    assert w.state == "unstick" and out[0] == 0.0 and abs(out[2]) == 1.0
    assert w.step(clear, None, 2.7, speed=0.0)[0] == 0.3 and w.state == "cruise"


@pytest.mark.skipif(not C.SCENE_WALK_XML.exists(), reason="microduck_rl checkout not found")
def test_wander_keeps_a_walking_duck_off_the_wall():
    import onnxruntime as ort

    from microduck_local.world import Duck, Scenario, Wall, World
    path = C.MICRODUCK_RL_DIR.parent / "microduck" / "policies" / "alpha_walking.onnx"
    if not path.exists():
        pytest.skip("upstream policies not checked out")
    sess = ort.InferenceSession(str(path))
    nm = sess.get_inputs()[0].name
    sc = Scenario(name="wall", floor=(6, 6),
                  walls=[Wall((1.0, -2.0), (1.0, 2.0), 0.6)],
                  ducks=[Duck("d0", (0.0, 0.0, 0.0), None, "ideal")])
    world = World(sc, infer_for={"d0": lambda o: sess.run(None, {nm: o[None]})[0][0].astype(np.float32)})
    d = world.ducks["d0"]
    w = Wander()
    min_gap = 9.0
    for _ in range(int(6.0 / C.CTRL_DT)):
        tof = d.tof.last
        cmd = w.step(None if tof is None else tof.depth_mm, None if tof is None else tof.valid,
                     world.t, d.heading_speed(world.data))
        d.set_cmd(world.data, cmd)
        world.step()
        min_gap = min(min_gap, 1.0 - float(d.trunk_pos(world.data)[0]))
    assert d.falls == 0
    assert min_gap > 0.15, min_gap            # never got its beak on the wall
    assert w.state in ("steer", "cruise", "spin")
