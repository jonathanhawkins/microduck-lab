"""Train a brain (roadmap 3.1/3.2/4.4): PPO over BrainEnv, then export it.

    uv run train-brain --run-name follow-v2 --envs 12 --steps 400000
    uv run eval-brain --brain learned:follow-v2 --preset hostile
    uv run eval-brain --brain follow --preset hostile          # the scripted baseline

    uv run train-brain --task striker --run-name striker-v1 --envs 8 --steps 400000
    uv run python -m microduck_local.eval_striker --brain striker-v1 --seeds 8

`--task striker` (roadmap 4.4) trains the soccer brain in `brain/striker.py`
instead: 88 features (the 80-float brain contract plus the goal geometry),
five actions (the twist plus a kick logit per foot — roadmap 3.5's
hierarchical head), reward = the ball's SIGNED progress toward the goal.
Its baseline is the scripted `Chase`, on the pitch, not the scripted follow.

Each PPO step here is one DECISION (five control steps of the frozen
walker), so 400k steps is 2M control steps of physics. Workers use
SubprocVecEnv with the spawn-safe forkserver start method (the ONNX
walker session must not be forked), so this runs on Windows too.

Artifacts land in brains/<run-name>/: model.zip, vecnormalize.pkl,
progress.jsonl (one line per rollout), brain.onnx (normalizer baked in,
output clamped to the intent bounds) and brain.json (the contract the
world's LearnedBrain reads).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .brain.brain_env import (
    ACT_HIGH,
    ACT_LOW,
    BRAIN_OBS_DIM,
    BRAIN_OBS_VERSION,
    BrainEnv,
    FollowTask,
)
from .brain.striker import (
    S_ACT_HIGH,
    S_ACT_LOW,
    STRIKER_OBS_DIM,
    STRIKER_OBS_VERSION,
    StrikerEnv,
    StrikerTask,
)
from .brain.learned import brains_dir
from .machine import profile, with_phase_callbacks
from .ppo_hparams import configure_torch_cpu


def make_env_fn(seed: int, fixed_preset: str | None, variety: bool = False, polite: float = FollowTask.polite):
    def fn():
        return BrainEnv(FollowTask(furniture=2 if variety else 0, distractor=variety, polite=polite), seed=seed,
                        fixed_preset=fixed_preset)
    return fn


def make_striker_env_fn(seed: int, task: StrikerTask, fixed_preset: str | None):
    def fn():
        return StrikerEnv(task, seed=seed, fixed_preset=fixed_preset)
    return fn


def export_brain(run_dir: Path, obs_dim: int = BRAIN_OBS_DIM,
                 act_low: np.ndarray = ACT_LOW, act_high: np.ndarray = ACT_HIGH) -> Path:
    import pickle

    import torch
    from stable_baselines3 import PPO

    model = PPO.load(str(run_dir / "model"), device="cpu")
    with open(run_dir / "vecnormalize.pkl", "rb") as f:
        vn = pickle.load(f)

    class OnnxBrain(torch.nn.Module):
        def __init__(self, policy, mean, var, clip):
            super().__init__()
            self.policy = policy
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("std", torch.tensor(np.sqrt(var + 1e-8), dtype=torch.float32))
            self.register_buffer("lo", torch.tensor(act_low))
            self.register_buffer("hi", torch.tensor(act_high))
            self.clip = clip

        def forward(self, obs):
            x = torch.clamp((obs - self.mean) / self.std, -self.clip, self.clip)
            feats = self.policy.extract_features(x, self.policy.features_extractor)
            latent = self.policy.mlp_extractor.forward_actor(feats)
            return torch.maximum(torch.minimum(self.policy.action_net(latent), self.hi), self.lo)

    wrapper = OnnxBrain(model.policy, vn.obs_rms.mean, vn.obs_rms.var, vn.clip_obs).eval()
    out = run_dir / "brain.onnx"
    torch.onnx.export(wrapper, (torch.zeros(1, obs_dim),), str(out),
                      input_names=["obs"], output_names=["intent"], opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out))
    rng = np.random.default_rng(0)
    for _ in range(3):
        obs = rng.normal(0, 1, (1, obs_dim)).astype(np.float32)
        with torch.no_grad():
            want = wrapper(torch.tensor(obs)).numpy()
        got = sess.run(["intent"], {"obs": obs})[0]
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=400_000, help="PPO steps = brain decisions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-preset", default=None, help="train on one sensor preset instead of drawing per episode")
    ap.add_argument("--init-from", default=None, help="warm-start from brains/<name>")
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--variety", action="store_true", help="train with furniture and a wandering duck (eval-brain --variety)")
    ap.add_argument("--polite", type=float, default=FollowTask.polite, metavar="M",
                    help="the person stops M m short of the duck in its way (eval-brain --polite); 0: walks through it, as follow-v1..v4 were trained")
    ap.add_argument("--task", default="follow", choices=("follow", "striker"),
                    help="follow (roadmap 3.2, BrainEnv) or striker (roadmap 4.4, StrikerEnv on a pitch)")
    # --- striker only. The reward weights are here so a run RECORDS what it
    # was paid; the spawn probabilities are the ladder (physics and spawns,
    # never the pay — AGENTS.md), and are the knob to reach for when the kick
    # does not appear in rollouts.
    ap.add_argument("--episode-s", type=float, default=StrikerTask.episode_s)
    ap.add_argument("--spot-prob", type=float, default=StrikerTask.spot_prob,
                    help="fraction of episodes that start with the ball on a kicking foot's sweet spot")
    ap.add_argument("--near-prob", type=float, default=StrikerTask.near_prob,
                    help="fraction that start with the ball a step or two ahead")
    ap.add_argument("--w-progress", type=float, default=StrikerTask.w_progress)
    ap.add_argument("--w-approach", type=float, default=StrikerTask.w_approach)
    ap.add_argument("--no-gaze", action="store_true", help="no reflex head pitch onto the tracked ball")
    ap.add_argument("--snap-steps", type=int, default=100_000, metavar="N",
                    help="atomically refresh model.zip / vecnormalize.pkl / brain.onnx every N steps "
                         "(0: only at the end). The exported brain is the one that ships, so a long run "
                         "has to be probeable while it runs — AGENTS.md verification discipline #6.")
    args = ap.parse_args()
    striker = args.task == "striker"
    task = StrikerTask(episode_s=args.episode_s, spot_prob=args.spot_prob, near_prob=args.near_prob,
                       w_progress=args.w_progress, w_approach=args.w_approach, gaze=not args.no_gaze)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

    # This trainer never called it, so it ran on torch's defaults: intra-op AND
    # inter-op both at the machine's core count, every one of those threads
    # spinning through rollouts made of 128-step brain decisions. Same policy
    # as the other two trainers now (machine.py).
    configure_torch_cpu(torch)
    print(profile().describe())

    out = brains_dir() / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    if striker:
        fns = [make_striker_env_fn(args.seed * 1000 + i, task, args.fixed_preset) for i in range(args.envs)]
    else:
        fns = [make_env_fn(args.seed * 1000 + i, args.fixed_preset, args.variety, args.polite) for i in range(args.envs)]
    venv = VecMonitor(SubprocVecEnv(fns, start_method="forkserver"))
    if args.init_from:
        prev = brains_dir() / args.init_from
        venv = VecNormalize.load(str(prev / "vecnormalize.pkl"), venv)
        model = PPO.load(str(prev / "model"), env=venv, device="cpu")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO("MlpPolicy", venv,
                    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]),
                                       activation_fn=torch.nn.ELU, log_std_init=-0.5),
                    n_steps=args.n_steps, batch_size=min(1024, args.n_steps * args.envs),
                    n_epochs=5, learning_rate=args.lr, gamma=0.98, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.003, vf_coef=0.5, max_grad_norm=1.0,
                    device="cpu", seed=args.seed, verbose=0)

    log = (out / "progress.jsonl").open("a")
    t0 = time.time()

    class Progress(BaseCallback):
        next_snap = args.snap_steps or 1 << 62

        def _snapshot(self) -> None:
            """model.zip + vecnormalize.pkl + brain.onnx, via temp names, so a
            reader never sees a half-written file."""
            import os
            # SB3 appends ".zip" only to a suffix-less path, so "model.tmp"
            # lands literally and the replace stays atomic.
            self.model.save(str(out / "model.tmp"))
            os.replace(out / "model.tmp", out / "model.zip")
            venv.save(str(out / "vecnormalize.pkl.tmp"))
            os.replace(out / "vecnormalize.pkl.tmp", out / "vecnormalize.pkl")
            export_brain(out, STRIKER_OBS_DIM, S_ACT_LOW, S_ACT_HIGH) if striker else export_brain(out)

        def _on_rollout_end(self) -> None:
            if self.num_timesteps >= self.next_snap:
                self.next_snap = self.num_timesteps + args.snap_steps
                self._snapshot()
            ep = self.model.ep_info_buffer
            rew = float(np.mean([e["r"] for e in ep])) if ep else float("nan")
            ln = float(np.mean([e["l"] for e in ep])) if ep else float("nan")
            row = {"steps": int(self.num_timesteps), "ep_rew": round(rew, 3), "ep_len": round(ln, 1),
                   "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[train-brain] {row}", flush=True)

        def _on_step(self) -> bool:
            return True

    meta = {"name": args.run_name, "task": args.task, "decide_every": 5,
            "envs": args.envs, "steps": args.steps, "seed": args.seed,
            "fixed_preset": args.fixed_preset}
    if striker:
        meta |= {"target_cls": "ball", "obs_dim": STRIKER_OBS_DIM, "obs_version": STRIKER_OBS_VERSION,
                 "act_low": S_ACT_LOW.tolist(), "act_high": S_ACT_HIGH.tolist(),
                 "episode_s": task.episode_s, "spot_prob": task.spot_prob, "near_prob": task.near_prob,
                 "w_progress": task.w_progress, "w_approach": task.w_approach,
                 "gaze": task.gaze, "head_range": task.head_range, "bump_stop": task.bump_stop,
                 "init_from": args.init_from}
    else:
        meta |= {"target_cls": "person", "obs_dim": BRAIN_OBS_DIM, "obs_version": BRAIN_OBS_VERSION,
                 "act_low": ACT_LOW.tolist(), "act_high": ACT_HIGH.tolist(),
                 "variety": args.variety, "polite": args.polite}
    (out / "brain.json").write_text(json.dumps(meta, indent=2))
    try:
        model.learn(total_timesteps=args.steps,
                    callback=with_phase_callbacks(Progress(), BaseCallback))
    finally:
        model.save(str(out / "model"))
        venv.save(str(out / "vecnormalize.pkl"))
        venv.close()
        log.close()
    path = (export_brain(out, STRIKER_OBS_DIM, S_ACT_LOW, S_ACT_HIGH) if striker else export_brain(out))
    print(f"[train-brain] exported {path}", flush=True)


if __name__ == "__main__":
    main()
