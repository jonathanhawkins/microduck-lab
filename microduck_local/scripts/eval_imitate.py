"""Deterministic evaluation of a G1 IMITATION policy against its clip.

    uv run python scripts/eval_imitate.py runs/<run>/policy.onnx g1-front-kick [--seeds 8] [--seconds 20] [--noise]

Every seed starts STANDING (soles re-seated on the floor — the buried-sole
trap from scripts/eval_kick.py) at a random point of the clip, so the policy
gets no free idle stretch; the clock then runs. Per seed: survived the whole
episode or not, apex clearance of the lifted foot per clip cycle against the
clip's own apex, how many cycles reached a real lift, the fraction of
"both feet planted" frames in which both feet were down, and yaw drift.
The numbers are the ones to quote — the training chart measures the
exploration-noised policy (memory: verify policies deterministically).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort


def stand_start(e, rng):
    e.reset(seed=int(rng.integers(0, 2**31)))
    e._phase0 = int(rng.integers(0, e.clip.steps))
    e.step_count = 0
    e.last_spawn = "standing"
    e.data.qpos[e.joint_qpos_adr] = e.default_pose
    e.data.qpos[e._root_qpos + 3:e._root_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    e.data.qvel[:] = 0.0
    e._write_ctrl(e.default_pose)
    mujoco.mj_forward(e.model, e.data)
    low = min(float(e.data.geom_xpos[g][2] - e.model.geom_size[g][0])
              for side in ("left", "right") for g in e.foot_geom_ids[side])
    e.data.qpos[e._root_qpos + 2] += 0.002 - low
    mujoco.mj_forward(e.model, e.data)
    e._refresh_derived()
    e._spawn_xy = np.asarray(e._trunk_xpos[:2], float).copy()
    return e._get_obs()


def evaluate(path, clip, seeds=8, seconds=20.0, noise=False):
    from microduck_local.robots.g1_imitate import G1ImitateEnv
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = {"held": 0, "dur": [], "apex": [], "lifts": [], "cycles": [],
           "planted": [], "yaw": [], "track": []}
    rng = np.random.default_rng(0)
    for sd in range(seeds):
        e = G1ImitateEnv(clip_name=clip, seed=sd, obs_noise=noise, domain_rand=noise,
                         push_robot=False, max_episode_s=seconds)
        obs = stand_start(e, rng)
        lifted_side = max(("left", "right"), key=lambda s: e.ref_foot_air[s].max())
        other = "left" if lifted_side == "right" else "right"
        ref_apex = float(e.ref_foot_air[lifted_side].max())

        def sole(side):
            return min(float(e.data.geom_xpos[g][2] - e.model.geom_size[g][0])
                       for g in e.foot_geom_ids[side])

        clr, ref_air, both_ref, both, track = [], [], [], [], []
        y0 = None
        term = trunc = False
        for i in range(int(seconds * 50) + 5):
            act = sess.run(None, {name: obs[None].astype(np.float32)})[0][0]
            obs, _, term, trunc, _ = e.step(act)
            j = e.ref_index()
            clr.append(sole(lifted_side) - sole(other))
            ref_air.append(float(e.ref_foot_air[lifted_side][j]))
            both_ref.append(bool(e.ref_planted["left"][j] and e.ref_planted["right"][j]))
            c = e._foot_contacts()
            both.append(bool(c["left"] and c["right"]))
            _, terms = e._compute_reward()
            track.append(terms["foot_track"] / e.W_FOOT_TRACK)
            q = e.data.qpos[e._root_qpos + 3:e._root_qpos + 7]
            yaw = np.degrees(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                        1 - 2 * (q[2] ** 2 + q[3] ** 2)))
            if y0 is None:
                y0 = yaw
            if term or trunc:
                break
        clr, ref_air = np.array(clr), np.array(ref_air)
        both_ref, both = np.array(both_ref), np.array(both)
        # One apex reading per clip cycle the episode reached, taken where
        # the clip's own lift is within 20% of its apex.
        window = ref_air > 0.8 * ref_apex
        cycles = 0
        apexes = []
        in_win = False
        cur = -np.inf
        for k in range(len(clr)):
            if window[k]:
                in_win = True
                cur = max(cur, clr[k])
            elif in_win:
                cycles += 1
                apexes.append(cur)
                in_win, cur = False, -np.inf
        out["held"] += int(not term)
        out["dur"].append((i + 1) / 50)
        out["cycles"].append(cycles)
        out["apex"].append(np.mean(apexes) if apexes else np.nan)
        out["lifts"].append(sum(a > 0.5 * ref_apex for a in apexes))
        out["planted"].append(both[both_ref].mean() if both_ref.any() else np.nan)
        out["yaw"].append(abs(yaw - y0))
        out["track"].append(float(np.mean(track)))
    out["ref_apex"] = ref_apex
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("policy")
    ap.add_argument("clip")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--record", action="store_true",
                    help="write the numbers into the run's record.json, so the "
                         "lab's policy palette shows them instead of a hash")
    ap.add_argument("--pick", action="store_true",
                    help="with --record: mark this the stage of its chain "
                         "worth using (clears the flag on its siblings)")
    a = ap.parse_args(argv)
    r = evaluate(a.policy, a.clip, seeds=a.seeds, seconds=a.seconds, noise=a.noise)
    apex = np.nanmean(r["apex"]) if not np.all(np.isnan(r["apex"])) else float("nan")
    print(f"{os.path.relpath(a.policy)}  clip={a.clip}  {'noise+DR' if a.noise else 'clean'}")
    print(f"  held {r['held']}/{a.seeds} for {a.seconds:.0f} s (mean {np.mean(r['dur']):.1f} s)")
    print(f"  apex clearance {apex:.3f} m of the clip's {r['ref_apex']:.3f} m "
          f"({100 * apex / r['ref_apex']:.0f}%)  |  cycles/ep {np.mean(r['cycles']):.1f}, "
          f"real lifts (>50% apex) {np.mean(r['lifts']):.1f}")
    print(f"  both feet down when the clip plants both: {100 * np.nanmean(r['planted']):.0f}%  |  "
          f"foot_track {100 * np.mean(r['track']):.0f}% of full  |  yaw drift {np.mean(r['yaw']):.0f} deg")
    print("  per seed dur:", [round(d, 1) for d in r["dur"]],
          "apex:", [None if np.isnan(x) else round(x, 2) for x in r["apex"]])
    if a.record:
        from microduck_local.describe_run import _chain_siblings
        from microduck_local.run_record import read_record, write_measurement, write_record
        run_dir = Path(a.policy).resolve().parent
        held = f"{r['held']}/{a.seeds}"
        note = (f"{held} seeds hold {a.seconds:.0f} s"
                f"{' under noise+DR' if a.noise else ''}, apex {apex:.2f} m "
                f"({100 * apex / r['ref_apex']:.0f}% of the clip), "
                f"{np.mean(r['lifts']):.1f} real lifts/episode")
        if a.pick:
            for sib in _chain_siblings(run_dir):
                if read_record(sib).get("pick"):
                    write_record(sib, pick=False)
        write_measurement(
            run_dir,
            tool=f"eval_imitate.py --seeds {a.seeds} --seconds {a.seconds:g}"
                 + (" --noise" if a.noise else ""),
            numbers={"held": held, "seconds": a.seconds,
                     "apex_m": None if np.isnan(apex) else round(float(apex), 3),
                     "clip_apex_m": round(float(r["ref_apex"]), 3),
                     "cycles_per_ep": round(float(np.mean(r["cycles"])), 1),
                     "real_lifts_per_ep": round(float(np.mean(r["lifts"])), 1),
                     "both_feet_when_planted": round(float(np.nanmean(r["planted"])), 2),
                     "yaw_deg": round(float(np.mean(r["yaw"])), 0)},
            note=note, pick=True if a.pick else None)
        print(f"  recorded in {run_dir.name}/record.json" + (" ★ pick" if a.pick else ""))


if __name__ == "__main__":
    sys.exit(main())
