"""Headless eval of an exported ONNX policy in the local env.

    uv run eval-walk runs/<run>/policy.onnx [--episodes 20]
    uv run eval-run  runs/<run>/policy.onnx [--cmd 0.4 --episodes 20]

Reports what rollouts actually show (fall rate, tracking error, episode length)
— the numbers to look at before claiming anything works. Also runs the shipped
alpha policy fine, e.g.:  uv run eval-walk ../microduck/policies/alpha_walking.onnx

`--behavior run` (the eval-run alias) uses the run env: BAM, no height-kill,
pinned forward command. Evaluating a run policy with the default walk env
asks it to turn and stand still, which it was never trained for.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnxruntime as ort


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--behavior", default=None,
                    help="BehaviorEnv id (eval-run sets this to 'run')")
    ap.add_argument("--cmd", type=float, default=None,
                    help="Pin twist vx (run eval). Default 0.4 for --behavior run")
    ap.add_argument("--actuator", default=None, choices=("xml", "bam"),
                    help="Override actuator; run defaults to bam")
    args = ap.parse_args()

    # `eval-run` is the same entry point with run defaults.
    if os.path.basename(sys.argv[0]) == "eval-run" and not args.behavior:
        args.behavior = "run"

    if args.behavior == "run" and args.cmd is None:
        args.cmd = 0.4
    if args.behavior == "run" and args.cmd is not None:
        os.environ["MICRODUCK_RUN_CMD"] = str(args.cmd)

    if args.behavior:
        from .behaviors import BEHAVIORS, BehaviorEnv
        b = BEHAVIORS[args.behavior]
        kw = dict(obs_noise=True, domain_rand=True, action_delay=True,
                  random_yaw=True, seed=args.seed,
                  max_episode_s=b.episode_s)
        if b.forward_cmd:
            kw["height_termination"] = False
            kw["actuator"] = args.actuator or "bam"
        elif args.actuator:
            kw["actuator"] = args.actuator
        env = BehaviorEnv(args.behavior, **kw)
    else:
        from .walk_env import MicroduckWalkEnv
        kw = dict(seed=args.seed)
        if args.actuator:
            kw["actuator"] = args.actuator
        env = MicroduckWalkEnv(**kw)

    sess = ort.InferenceSession(args.onnx_path)
    in_name = sess.get_inputs()[0].name

    lengths, falls, lin_errs, ang_errs, fwds = [], 0, [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            action = sess.run(None, {in_name: obs[None]})[0][0].astype(np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            # Body-frame xy, same frame as the twist command and the reward.
            body_v = env.body_lin_vel()
            lin_errs.append(float(np.linalg.norm(env.twist_cmd[:2] - body_v[:2])))
            ang_errs.append(abs(float(env.twist_cmd[2] - env.data.sensordata[env.gyro_adr][2])))
            fwds.append(float(body_v[0]))
        lengths.append(env.step_count)
        falls += int(terminated)

    n = args.episodes
    print(f"episodes: {n}   mean length: {np.mean(lengths):.0f}/{env.max_steps} steps")
    print(f"falls: {falls}/{n} ({100 * falls / n:.0f}%)")
    print(f"lin vel tracking err: mean {np.mean(lin_errs):.3f} m/s (p90 {np.percentile(lin_errs, 90):.3f})")
    print(f"ang vel tracking err: mean {np.mean(ang_errs):.3f} rad/s (p90 {np.percentile(ang_errs, 90):.3f})")
    if args.behavior == "run":
        cmd = args.cmd
        print(f"achieved body-x speed: mean {np.mean(fwds):.3f} m/s "
              f"(cmd {cmd:.2f}, tracked {100 * np.mean(fwds) / max(cmd, 1e-6):.0f}%)")


if __name__ == "__main__":
    main()
