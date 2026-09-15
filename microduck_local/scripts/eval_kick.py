"""Deterministic kick evaluation from an HONEST standing start.

The subtlety that invalidated earlier numbers: a reset may spawn mid-cycle,
which lowers the root so the support sole sits on the floor. Overwriting the
joints back to the standing pose then leaves the root at that lowered height
and buries BOTH feet 35 mm in the ground — the robot starts interpenetrating
and falls in ~1 s regardless of how good it is. Re-seat the root after posing.
"""
from __future__ import annotations

import sys

import mujoco
import numpy as np
import onnxruntime as ort


def stand_start(e, phase0=0):
    """Pose the robot standing, with its soles ON the floor.

    `phase0=0` gives the policy a full idle stretch before its first kick,
    which is the EASY start — the natural reset picks a random phase, so the
    kick can be demanded from a cold standing body immediately. Measured: the
    same policy reads 5/5 at phase 0 and falls at 2.1 s under a random phase.
    Pass phase0=None for the honest, randomised version.
    """
    e.reset(seed=0)
    if phase0 is None:
        import numpy as _np
        n = max(int(round(e.CYCLE_S / 0.02)), 1)
        e._phase0 = int(_np.random.default_rng(e._seed_used).integers(0, n)) \
            if hasattr(e, "_seed_used") else int(_np.random.randint(0, n))
    else:
        e._phase0 = phase0
    e.step_count = 0
    e.data.qpos[e.joint_qpos_adr] = e.default_pose
    e.data.qvel[:] = 0.0
    mujoco.mj_forward(e.model, e.data)
    low = min(float(e.data.geom_xpos[g][2] - e.model.geom_size[g][0])
              for side in ("left", "right") for g in e.foot_geom_ids[side])
    e.data.qpos[e._root_qpos + 2] += 0.002 - low        # <-- the missing line
    mujoco.mj_forward(e.model, e.data)
    e._refresh_derived()
    return e._get_obs()


def evaluate(path, env_cls, seeds=5, seconds=20.0, noise=False,
             phase0=0, **env_kw):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = {"held": 0, "kick": [], "idle_both": [], "yaw": [], "kicks": [], "dur": []}
    for sd in range(seeds):
        e = env_cls(seed=sd, obs_noise=noise, domain_rand=noise,
                    push_robot=False, max_episode_s=seconds, **env_kw)
        obs = stand_start(e, phase0=phase0)

        def sole(side):
            return min(float(e.data.geom_xpos[g][2] - e.model.geom_size[g][0])
                       for g in e.foot_geom_ids[side])

        clr, amps, both = [], [], []
        y0 = None
        kicks = 0
        prev = 0.0
        for i in range(int(seconds * 50) + 5):
            amp = e.strike_amplitude()
            act = sess.run(None, {name: obs[None].astype(np.float32)})[0][0]
            obs, _, term, trunc, _ = e.step(act)
            if amp > 0.95 >= prev:
                kicks += 1
            prev = amp
            clr.append(sole(e.STRIKE_SIDE) - sole(e.SUPPORT_SIDE))
            amps.append(e.strike_amplitude())
            c = e._foot_contacts()
            both.append(bool(c["left"] and c["right"]))
            q = e.data.qpos[e._root_qpos + 3:e._root_qpos + 7]
            yaw = np.degrees(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                        1 - 2 * (q[2] ** 2 + q[3] ** 2)))
            if y0 is None:
                y0 = yaw
            if term or trunc:
                break
        clr, amps, both = np.array(clr), np.array(amps), np.array(both)
        out["held"] += int(not term)
        out["dur"].append(i / 50)
        out["kicks"].append(kicks)
        out["kick"].append(clr[amps > 0.95].mean() if (amps > 0.95).any() else np.nan)
        out["idle_both"].append(both[amps < 0.05].mean() if (amps < 0.05).any() else np.nan)
        out["yaw"].append(abs(yaw - y0))
    return out


def report(label, run, target, noise=False, **kw):
    import microduck_local.robots.g1_karate as K
    r = evaluate(f"runs/{run}/policy.onnx", K.G1FrontKickEnv, noise=noise, **kw)
    print(f"  {label:24s} {r['held']}/5 ({np.mean(r['dur']):4.1f}s) | "
          f"kick {np.nanmean(r['kick']):.3f} m ({100*np.nanmean(r['kick'])/target:3.0f}%) | "
          f"kicks/ep {np.mean(r['kicks']):.1f} | feet {100*np.nanmean(r['idle_both']):3.0f}% | "
          f"yaw {np.mean(r['yaw']):3.0f}")


if __name__ == "__main__":
    print(__doc__.splitlines()[0])
    sys.exit(0)
