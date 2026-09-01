"""Train the local walking policy with SB3 PPO across CPU cores.

    uv run train-walk --envs 16 --steps 3_000_000 --run-name first

Same recipe shape as jenga-stacker's train_rl.py, scaled for locomotion:
multi-process rollout parallelism (vec_env.py — the workers share ONE compiled
mjModel; $MICRODUCK_VEC_ENV picks the backend) + VecNormalize obs
standardization (baked into the ONNX at export, like microduck_rl's
scripts/export.py). device=cpu by default — for MLP policies this size, CPU
beats MPS dispatch overhead.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from .ppo_hparams import N_STEPS, VF_COEF, configure_torch_cpu, ppo_batch_size
from .vec_env import as_sb3_vec_env, make_vec_env
from .walk_env import MicroduckWalkEnv

# Overridable so tests and scratch lab servers write somewhere disposable —
# discover_policies() scans this dir, so stray test runs would otherwise show
# up in the live viewer's palette.
RUNS_DIR = Path(os.environ.get("MICRODUCK_RUNS_DIR")
                or Path(__file__).resolve().parents[2] / "runs")


def _penalty_sign_callback_cls(BaseCallback):
    """AGENTS.md's infallible check: every *_penalty episode sum must be <= 0.

    Built after `make_vec_env` so importing this module does not pull torch
    (forking a torch-initialized parent deadlocks on macOS).
    """

    class PenaltySignCallback(BaseCallback):
        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                sums = info.get("episode_rewards")
                if not sums:
                    continue
                for name, value in sums.items():
                    if name.endswith("_penalty") and value > 1e-6:
                        raise RuntimeError(
                            f"Reward-sign bug: episode sum of '{name}' is {value:+.4f} "
                            "(> 0). A penalty became a reward — fix before training on."
                        )
            return True

    return PenaltySignCallback


def make_env(rank: int, seed: int, **env_kwargs):
    def _init():
        return MicroduckWalkEnv(seed=seed + rank, **env_kwargs)
    return _init


def main() -> None:
    ap = argparse.ArgumentParser()
    # 32: the bench-envs knee — see train_behavior.py's --envs for the numbers.
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--run-name", default=time.strftime("walk-%Y%m%d-%H%M%S"))
    ap.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None,
                    help="warm-start from an existing run dir (fine-tune)")
    ap.add_argument("--no-domain-rand", action="store_true")
    ap.add_argument("--no-obs-noise", action="store_true")
    args = ap.parse_args()

    out = RUNS_DIR / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(
        domain_rand=not args.no_domain_rand,
        obs_noise=not args.no_obs_noise,
    )
    # Fork workers BEFORE importing torch. A torch-initialized parent has
    # OpenMP/Accelerate thread pools; forking them deadlocks on macOS.
    # One compiled mjModel for the whole fleet: see vec_env.py.
    venv = make_vec_env([make_env(i, args.seed, **env_kwargs)
                         for i in range(args.envs)])

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
    from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

    configure_torch_cpu(torch)
    venv = VecMonitor(as_sb3_vec_env(venv))
    batch = ppo_batch_size(N_STEPS, args.envs)

    from .symmetry import FastActorCriticPolicy

    if args.init_from:
        prev = Path(args.init_from)
        venv = VecNormalize.load(str(prev / "vecnormalize.pkl"), venv)
        model = PPO.load(str(prev / "model"), env=venv, device=args.device,
                         custom_objects={"policy_class": FastActorCriticPolicy})
        model.batch_size = batch
        print(f"warm-started from {prev}")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=100.0)
        model = PPO(
            FastActorCriticPolicy, venv,
            # rsl_rl-flavored: big-ish MLP, ELU, PPO with standard locomotion params.
            policy_kwargs=dict(
                net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                activation_fn=torch.nn.ELU,
                log_std_init=0.0,
            ),
            n_steps=N_STEPS, batch_size=batch, n_epochs=5,
            # Matched to the official stack's rsl_rl cfg (entropy 0.01,
            # lr 1e-3). Half the entropy and a third the learning rate, on far
            # fewer samples, collapses to the safest available policy —
            # standing still, which is exactly what first-gait learned.
            learning_rate=1e-3, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01, vf_coef=VF_COEF,
            max_grad_norm=1.0,
            device=args.device, seed=args.seed, verbose=1,
            tensorboard_log=str(out / "tb"),
        )

    try:
        git_sha = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[3] / "microduck_rl"),
             "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
    except OSError:
        git_sha = "unknown"
    (out / "run.json").write_text(json.dumps({
        "run_name": args.run_name, "envs": args.envs, "steps": args.steps,
        "seed": args.seed, "env_kwargs": env_kwargs,
        "microduck_rl_sha": git_sha, "init_from": args.init_from,
    }, indent=2))

    checkpoints = CheckpointCallback(
        save_freq=max(500_000 // args.envs, 1), save_path=str(out / "checkpoints"),
        name_prefix="model", save_vecnormalize=True,
    )
    model.learn(
        total_timesteps=args.steps,
        callback=[checkpoints, _penalty_sign_callback_cls(BaseCallback)()],
        progress_bar=False,
        reset_num_timesteps=args.init_from is None,
    )

    model.save(str(out / "model"))
    venv.save(str(out / "vecnormalize.pkl"))
    print(f"saved {out}/model.zip + vecnormalize.pkl")
    print(f"export with: uv run export-walk {out}")


if __name__ == "__main__":
    main()
