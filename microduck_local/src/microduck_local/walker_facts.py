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
  reverse                     none: -0.3 m/s commanded moves 4 mm in 2 s
  turn in place               wz=+1: 0.25 rad in the first second, ~0.6 rad/s after; wz=-1 from a standstill: 0.05 rad/s
                              — unless the gait is going (after walking, or with vx=0.2: ~0.7 rad/s)
  sidestep                    vy=±0.3 for 2 s: 8 / 12 cm, with a 0.1 / 0.34 rad yaw drift
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


if __name__ == "__main__":
    main()
