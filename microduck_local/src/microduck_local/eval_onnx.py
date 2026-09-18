"""Headless eval of an exported ONNX policy in the local env.

    uv run eval-walk runs/<run>/policy.onnx [--episodes 20]
    uv run eval-run  runs/<run>/policy.onnx [--cmd 0.4 --episodes 20]
    uv run eval-walk runs/g1-walk/policy.onnx --robot g1     # another body

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

from .robots import registry


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
    ap.add_argument("--robot", default=None, choices=registry.ids(),
                    help="which body the policy drives; default: the policy's "
                         "own contract (its ONNX metadata, else the run.json "
                         "next to it). Refused when it contradicts a contract "
                         "the file RECORDED. Behaviors are duck recipes and "
                         "are refused for another body.")
    ap.add_argument("--push", action="store_true",
                    help="Shove the base every 3-6 s (the training env's pushes, "
                         "2026-09-06). Off here by default so eval numbers stay "
                         "comparable with the ones taken before pushes existed.")
    args = ap.parse_args()

    # `eval-run` is the same entry point with run defaults.
    if os.path.basename(sys.argv[0]) == "eval-run" and not args.behavior:
        args.behavior = "run"

    # WHICH BODY, resolved before anything is written: MICRODUCK_RUN_CMD below
    # is process-wide, and an argument error that sets it on the way out leaves
    # the knob armed for whatever runs next (it did, in the test suite).
    from pathlib import Path as _Path

    from .robots.policy_contract import recorded, resolve

    # The POLICY's own contract, from the file itself where it has one:
    # `eval-walk some.onnx` far from its run directory used to default to the
    # duck and had to be told otherwise (robots/policy_contract.resolve).
    onnx_path = _Path(args.onnx_path)
    contract = resolve(onnx_path)
    robot = args.robot or contract.robot
    if args.robot:
        # The flag against the file. Refused only when the file RECORDED a
        # contract — a stamped ONNX, or a run.json that declares one — since
        # then the file cannot be wrong about itself and building the other
        # body's env would feed the policy an observation of the wrong shape
        # (or, one day, the right shape and the wrong meaning, which is the
        # cross a width never catches). A policy that declares nothing is
        # still the caller's to name: that is what --robot was added for.
        declared = recorded(onnx_path)
        want = registry.get(args.robot).contract()
        if declared is not None and not declared.matches(want):
            raise SystemExit(
                f"{args.onnx_path} records contract {declared.id} "
                f"({declared.obs_dim} obs / {declared.act_dim} actions); "
                f"--robot {args.robot} speaks {want.id} ({want.obs_dim} obs / "
                f"{want.act_dim} actions) — one of the two is wrong, and this "
                "is the check that used to be a width")
    if robot != "microduck" and args.behavior:
        raise SystemExit(
            f"--behavior is a Microduck reward recipe; {robot} has none yet")

    # EVERY number this command prints is a walker's: fall rate, linear and
    # angular twist-tracking error against a commanded twist, achieved
    # body-x speed. A wheeled body has no gait to score and `MarsArmEnv` has
    # no `twist_cmd`, no gyro and no fall — so this refuses rather than
    # building `registry.get(robot).env_class(task)` and printing three
    # meaningless columns off it. AGENTS.md's rule 6 in its writing-side
    # form: a measurement that cannot produce the answer it reports is worse
    # than no measurement, because it reads like one.
    #
    # The eye for a MARS policy is `scripts/probe_mars_reach.py`: the
    # deterministic ONNX in its own env over N seeds, the final distance per
    # seed, and a contact sheet to LOOK at.
    if registry.get(robot).kind != "legged":
        raise SystemExit(
            f"eval-walk scores a GAIT — falls, twist tracking, body-x speed — "
            f"and {robot} is a {registry.get(robot).kind} body with none of "
            f"them. For a MARS arm policy use:\n"
            f"    uv run python scripts/probe_mars_reach.py {args.onnx_path} "
            f"--seeds 8 --out /tmp/mars-reach")

    if args.behavior == "run" and args.cmd is None:
        args.cmd = 0.4
    if args.behavior == "run" and args.cmd is not None:
        os.environ["MICRODUCK_RUN_CMD"] = str(args.cmd)

    if robot != "microduck":
        from .train import env_class
        kw = dict(seed=args.seed, push_robot=args.push)
        env = env_class(robot)(**kw)
    elif args.behavior:
        from .behaviors import BEHAVIORS, BehaviorEnv
        b = BEHAVIORS[args.behavior]
        kw = dict(obs_noise=True, domain_rand=True, action_delay=True,
                  random_yaw=True, seed=args.seed,
                  max_episode_s=b.episode_s)
        if b.forward_cmd:
            kw["actuator"] = args.actuator or "bam"
        elif args.actuator:
            kw["actuator"] = args.actuator
        env = BehaviorEnv(args.behavior, **kw)
    else:
        from .walk_env import MicroduckWalkEnv
        kw = dict(seed=args.seed, push_robot=args.push)
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
