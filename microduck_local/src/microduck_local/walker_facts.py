"""`walker-facts`: measure the shipped walker instead of assuming it.

    uv run walker-facts            # ~1 min on CPU; prints the table below

Every number a brain relies on — how far the beak reaches, where the feet
end, how the head moves while walking, how the walker stops, turns and
(does not) reverse — is measured here on the real `alpha_walking.onnx` in
the World, so a change of walker (or of MJCF) re-measures the brain's
constants in a minute instead of breaking it silently. `brain/tidy.py`
quotes these figures next to the constants they justify.

What it measures (2026-09, alpha_walking):
  beak tip vs head pitch      level: 0.080 m ahead, z 0.211; pitched 0.6: 0.073 / 0.180 (down pulls it BACK)
  feet                        reach 0.04 m ahead of the trunk (contact with a 6 cm rim from 0.185 m to its centre)
  camera depression           standing 0.19 rad; walking 0.11 ± 0.02 (the gait holds the head 0.08 rad higher)
  stop                        1 cm coast from 0.3 m/s, no yaw drift
  reverse                     -0.35 / -0.40 back up at 0.20 / 0.23 m/s — FASTER than the walker goes forward
                              (0.13 / 0.18 at +0.30 / +0.40). Everything from -0.30 up is a DEAD BAND: 4 mm in 6 s.
  turn in place               a dead band too, and a wide one: cold, every |wz| < 1.0 is EXACTLY zero, and so is
                              wz=-1; only wz=+1 breaks through, at 0.57 rad/s. Warm (after 2 s of walking) the rate
                              is roughly linear in the command: 0.15 / 0.28 / 0.47 / 0.61 at +0.25…+1, and
                              -0.00 / -0.42 / -0.57 / -0.78 at -0.25…-1. So the ceiling is ~0.6-0.8 rad/s AND the
                              command range (±1.0) is already at it — there is nothing left to ask for.
  sidestep                    vy=±0.3 for 2 s: 8 / 12 cm, with a 0.1 / 0.34 rad yaw drift; vy=±0.15 moves 1 mm in 3 s

The reverse and turn rows above replace two facts this file asserted for
months and that three brains were built around — "the walker cannot
reverse" and "a standing turn barely turns". Both were measured at ONE
command value that happens to sit inside a dead band (-0.3, and a cold
wz below 1.0). `command_deadbands()` sweeps the range instead, on an EMPTY
floor: `make_room`'s 3.0 x 2.5 m room with four boxes cannot hold the
1.3 m a 6 s reverse covers, so measuring the gait in it measures the
boxes. The lesson generalises past this file — a locomotion limit read off
a single command is a reading of the dead band, not of the walker.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .brain.brain_env import POLICIES_DIR, onnx_infer
from .world import World, make_room


def _world():
    w = World(make_room(seed=0), infer_for={"d0": onnx_infer(POLICIES_DIR / "alpha_walking.onnx")}, seed=0)
    return w, w.ducks["d0"]


def _settle(w, d, secs=1.0):
    for _ in range(int(secs / 0.02)):
        d.set_cmd(w.data, (0, 0, 0), (0, 0, 0, 0))
        w.step()


def _frame(w, d):
    pos = d.trunk_pos(w.data).copy()
    yaw = d.yaw(w.data)
    return pos, yaw, np.cos(yaw), np.sin(yaw)


def beak_and_feet() -> None:
    w, d = _world()
    m = w.model
    feet = [g for g in range(m.ngeom)
            if "ankle" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")]
    for hp in (0.0, 0.6, 0.9, 1.2):
        for _ in range(75):
            d.set_cmd(w.data, (0, 0, 0), (0.0, hp, 0.0, 0.0))
            w.step()
        pos, yaw, c, s = _frame(w, d)
        tip = w.mouth_tip(d) - pos
        feet_ahead = max(c * (w.data.geom_xpos[g][0] - pos[0]) + s * (w.data.geom_xpos[g][1] - pos[1]) + m.geom_size[g].max()
                         for g in feet)
        print(f"  head_pitch {hp:.1f}: beak tip {c * tip[0] + s * tip[1]:.3f} m ahead, z {tip[2] + pos[2]:.3f} · "
              f"feet reach {feet_ahead:.3f} m ahead")


def camera_pitch() -> None:
    w, d = _world()
    cam = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_SITE, "d0/head_camera")
    for hp, vx, tag in ((0.0, 0.0, "standing, head level"), (0.6, 0.0, "standing, head 0.6"),
                        (0.0, 0.3, "walking, head level"), (0.6, 0.3, "walking, head 0.6")):
        deps, zs = [], []
        for i in range(200):
            d.set_cmd(w.data, (vx, 0, 0), (0.0, hp, 0.0, 0.0))
            w.step()
            if i > 60:
                R = w.data.site_xmat[cam].reshape(3, 3)
                deps.append(-np.arcsin(np.clip(R[2, 0], -1, 1)))
                zs.append(w.data.site_xpos[cam][2])
        print(f"  {tag:22s}: camera depression {np.mean(deps):.3f} ± {np.std(deps):.3f} rad, height {np.mean(zs):.3f} m")


def _flat_world(yaw: float = 0.0):
    """An empty 12 x 12 m floor with the duck at the origin. `_world()`'s room
    is 3.0 x 2.5 m with four boxes in it — fine for a 2 s probe, useless for
    anything that travels, which is how the reverse fact came out wrong."""
    from .world.scenario import Duck, Scenario, Wall
    h = 6.0
    cs = [(-h, -h), (h, -h), (h, h), (-h, h)]
    sc = Scenario(name="flat", seed=0, floor=(13.0, 13.0),
                  walls=[Wall(cs[i], cs[(i + 1) % 4], 0.3, 0.02) for i in range(4)],
                  boxes=[], ducks=[Duck("d0", (0.0, 0.0, float(yaw)))])
    w = World(sc, infer_for={"d0": onnx_infer(POLICIES_DIR / "alpha_walking.onnx")}, seed=0)
    return w, w.ducks["d0"]


def command_deadbands() -> None:
    """Sweep vx and wz across their command ranges instead of sampling one
    value each. Steady rate over the last 2 s of the command, averaged over
    four start headings (the only free variable once the floor is empty —
    the gait is deterministic in the model)."""
    yaws = (0.0, 1.6, 3.1, 4.7)

    def drive(vx: float, yaw: float, secs: float = 6.0) -> float:
        w, d = _flat_world(yaw)
        _settle(w, d)
        p0, _, c, s = _frame(w, d)
        ahead = []
        for _ in range(int(secs / 0.02)):
            d.set_cmd(w.data, (vx, 0, 0), (0, 0, 0, 0))
            w.step()
            p = d.trunk_pos(w.data)
            ahead.append(c * (p[0] - p0[0]) + s * (p[1] - p0[1]))
        i2 = int(2 / 0.02)
        return (ahead[-1] - ahead[i2]) / (secs - 2.0)

    def spin(wz: float, yaw: float, warm: bool, secs: float = 4.0) -> float:
        w, d = _flat_world(yaw)
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
        i2 = int(2 / 0.02)
        return float((u[-1] - u[i2]) / (secs - 2.0))

    print("  forward/reverse (steady m/s over sec 2-6, empty floor, 4 headings):")
    for vx in (-0.40, -0.35, -0.30, -0.25, -0.20, 0.20, 0.25, 0.30, 0.40):
        v = float(np.mean([drive(vx, y) for y in yaws]))
        print(f"    vx {vx:+.2f} -> {v:+.3f} m/s" + ("   (dead band)" if abs(v) < 0.01 else ""))
    print("  turn in place (steady rad/s over sec 2-4; warm = after 2 s of walking):")
    for wz in (1.0, 0.75, 0.5, 0.25, -0.25, -0.5, -0.75, -1.0):
        cold = float(np.mean([spin(wz, y, False) for y in yaws]))
        warm = float(np.mean([spin(wz, y, True) for y in yaws]))
        print(f"    wz {wz:+.2f} -> cold {cold:+.3f}, warm {warm:+.3f} rad/s"
              + ("   (cold: dead band)" if abs(cold) < 0.01 else ""))


def stop_reverse_turn() -> None:
    w, d = _world()
    for _ in range(200):
        d.set_cmd(w.data, (0.3, 0, 0), (0, 0, 0, 0))
        w.step()
    p0, y0, c, s = _frame(w, d)
    _settle(w, d, 2.0)
    p, yaw, _, _ = _frame(w, d)
    print(f"  stop from 0.3 m/s: coasts {c * (p[0] - p0[0]) + s * (p[1] - p0[1]):.3f} m, yaw drift {yaw - y0:+.3f}")
    for cmd, tag in (((-0.3, 0, 0), "reverse -0.3 m/s"), ((0, 0.3, 0), "sidestep vy +0.3"), ((0, -0.3, 0), "sidestep vy -0.3")):
        w, d = _world()
        _settle(w, d)
        p0, y0, c, s = _frame(w, d)
        for _ in range(100):
            d.set_cmd(w.data, cmd, (0, 0, 0, 0))
            w.step()
        p, yaw, _, _ = _frame(w, d)
        dy = np.arctan2(np.sin(yaw - y0), np.cos(yaw - y0))
        print(f"  {tag} for 2 s: {c * (p[0] - p0[0]) + s * (p[1] - p0[1]):+.3f} m ahead, "
              f"{-s * (p[0] - p0[0]) + c * (p[1] - p0[1]):+.3f} m left, yaw {dy:+.2f}")
    for seq, tag in (([((0, 0, 1.0), 5.0)], "turn wz=+1 from a standstill"),
                     ([((0, 0, -1.0), 5.0)], "turn wz=-1 from a standstill"),
                     ([((0.3, 0, 0), 2.0), ((0, 0, -1.0), 3.0)], "walk 2 s, then wz=-1"),
                     ([((0.2, 0, -1.0), 3.0)], "vx=0.2 with wz=-1")):
        w, d = _world()
        _settle(w, d)
        parts = []
        for cmd, secs in seq:
            y0 = d.yaw(w.data)
            per_s = []
            for i in range(int(secs / 0.02)):
                d.set_cmd(w.data, cmd, (0, 0, 0, 0))
                w.step()
                if (i + 1) % 50 == 0:
                    dy = d.yaw(w.data) - y0
                    per_s.append(round(float(np.arctan2(np.sin(dy), np.cos(dy))), 2))
            parts.append(f"{cmd}: yaw per second {per_s}")
        print(f"  {tag}: " + " ; ".join(parts))


def main() -> None:
    print("beak and feet (standing):")
    beak_and_feet()
    print("camera pose (what a detection frame is stamped with):")
    camera_pitch()
    print("stopping, reversing, sidestepping, turning:")
    stop_reverse_turn()
    print("where the command ranges are actually dead (empty floor):")
    command_deadbands()


if __name__ == "__main__":
    main()
