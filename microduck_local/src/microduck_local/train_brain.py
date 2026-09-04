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
import os
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
from .brain.learned import brains_dir
from .brain.striker import (
    S_ACT_HIGH,
    S_ACT_LOW,
    STRIKER_OBS_DIM,
    STRIKER_OBS_VERSION,
    StrikerEnv,
    StrikerTask,
)
from .machine import profile, with_phase_callbacks
from .plateau import PlateauDetector, env_defaults
from .ppo_hparams import configure_torch_cpu, linear_decay, ppo_batch_size
from .vec_env import as_sb3_vec_env, make_vec_env

# Hard cap on the policy's per-dim action log_std (std <= ~0.6), the same cap
# and for the same measured reason as `train_behavior.LOG_STD_MAX`: every
# `--init-from` reloads the previous run's log_std and the entropy bonus
# pushes it up each generation, until the CLIPPED sampling distribution — not
# the mean — carries the behavior. `brain.onnx` exports the MEAN, so a
# ratcheted lineage exports garbage while its reward curve still looks fine.
# 0.0 (std 1.0), not train_behavior's -0.5. The brain's `log_std_init` IS
# -0.5, so a -0.5 cap binds from step 0 and forbids the entropy bonus from
# ever widening exploration. Measured on five 2M runs: every one drove its
# own log_std DOWN to between -0.55 and -1.22, so a cap up here never binds
# on a healthy run — it exists only to catch the warm-start ratchet, which
# reached log_std 3.2 (std 21-26) on a trick chain.
LOG_STD_MAX = 0.0
# Reproduce the pre-fix hyperparameters exactly, for A/B measurement only:
# the truncating `min(1024, n_steps * envs)` batch, a constant learning rate
# and no action-std cap. AGENTS.md requires a seed-matched A/B at matched
# STEP COUNTS before any training change becomes a default, and an A/B needs
# the baseline arm to be runnable from the same code. Same escape-hatch
# pattern as MICRODUCK_OVERLAP / MICRODUCK_UPDATE_DEVICE.
LEGACY = os.environ.get("MICRODUCK_BRAIN_LEGACY", "") not in ("", "0")
# Learning rate. `train_behavior` decays its own because every trick run
# peaked and then came apart, but the brain runs measured here do NOT show
# that shape and a seed-matched 2M A/B of the decay came out inside
# training-seed noise. A tuning change with no measured benefit does not get
# to be a default, so LR_END defaults to LR_START (constant, as before) and
# `--lr-end` turns the decay on.
LR_START = 3e-4
LR_END = LR_START


def git_state(repo: Path | None = None) -> dict:
    """{"git_sha": short sha, "git_dirty": bool} for the workspace this trainer
    runs from — the one fact brain.json could not answer before: "what CODE
    was different between these two runs". Best-effort: a checkout without
    git, or a copy that is not a repo, records nothing rather than failing a
    training run over provenance."""
    import subprocess
    root = repo or Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if sha.returncode != 0:
            return {}
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5)
        return {"git_sha": sha.stdout.strip(),
                "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None}
    except (OSError, subprocess.TimeoutExpired):
        return {}


def make_env_fn(seed: int, fixed_preset: str | None, variety: bool = False,
                polite: float = FollowTask.polite, polite_mix: tuple[float, ...] = ()):
    def fn():
        return BrainEnv(FollowTask(furniture=2 if variety else 0, distractor=variety,
                                   polite=polite, polite_mix=tuple(polite_mix)),
                        seed=seed, fixed_preset=fixed_preset)
    return fn


def make_striker_env_fn(seed: int, task: StrikerTask, fixed_preset: str | None):
    def fn():
        return StrikerEnv(task, seed=seed, fixed_preset=fixed_preset)
    return fn


def export_brain(run_dir: Path, obs_dim: int = BRAIN_OBS_DIM,
                 act_low: np.ndarray = ACT_LOW, act_high: np.ndarray = ACT_HIGH,
                 model_path: Path | None = None,
                 vn_path: Path | None = None, out: Path | None = None) -> Path:
    """Bake the normalizer into an ONNX brain.

    Defaults to the run's final `model.zip` + `vecnormalize.pkl` into
    `brain.onnx`. The explicit paths are what `select-brain` uses to export a
    numbered CHECKPOINT for deterministic scoring without disturbing the
    shipped artifact. The dims default to the follow contract; the striker
    trains on its own (88-feature, five-action) one.
    """
    import pickle

    import torch
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path or (run_dir / "model")), device="cpu")
    with open(vn_path or (run_dir / "vecnormalize.pkl"), "rb") as f:
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
    out = Path(out) if out is not None else run_dir / "brain.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
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
    ap.add_argument("--net-arch", default="128,128", metavar="H,H,...",
                    help="hidden sizes for the policy and value MLPs. NOT a throughput "
                         "knob on this trainer: the PPO update is only 5.3%% of brain "
                         "wall time (94.7%% is rollout collection, which is physics), so "
                         "halving the net saves ~3.5%%. It is a CAPACITY knob — whether "
                         "80 observations and a 3-dim intent need more than 128-128")
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=LR_START,
                    help="initial learning rate; decays linearly to --lr-end")
    ap.add_argument("--lr-end", type=float, default=LR_END,
                    help="final learning rate (pass --lr-end equal to --lr for the old constant rate)")
    pd = env_defaults()
    ap.add_argument("--plateau-patience", type=int, default=int(pd["patience"]), metavar="N",
                    help="stop once the smoothed reward has not improved for N rollouts "
                         "(0: off, the default). follow-v4 trained 2M decisions on a curve "
                         "that was flat from ~900k; the second half bought nothing")
    ap.add_argument("--plateau-min-steps", type=int, default=int(pd["min_steps"]), metavar="STEPS")
    ap.add_argument("--plateau-rel", type=float, default=pd["rel"], metavar="F")
    ap.add_argument("--plateau-window", type=int, default=int(pd["window"]), metavar="N")
    ap.add_argument("--decide-every-start", type=int, default=0, metavar="N",
                    help="start at this decision period (control steps) and switch to the "
                         "deployment period of 5 at --decide-every-switch. NOT a saving: a "
                         "coarser period makes each PPO sample cover MORE physics, so a "
                         "fixed decision budget costs more. What it trades for is a shorter "
                         "horizon in decisions for the same 20 s episode. 0: off")
    ap.add_argument("--decide-every-switch", type=float, default=0.5, metavar="F",
                    help="fraction of --steps at which to drop to the deployment period")
    ap.add_argument("--probe-every", type=int, default=0, metavar="STEPS",
                    help="every STEPS, score the DETERMINISTIC export on a few benchmark "
                         "episodes and log it as `probe`. This is the signal a plateau stop "
                         "should watch: measured on a 2M run, in-band was flat from 250k "
                         "while ep_rew climbed all the way to 2M, so a reward-based stop "
                         "would never have fired. 0: off")
    # 5 seeds x 8 episodes = 40, matching what `select-brain` scores a
    # checkpoint on. MEASURED reason for the size: an 8-episode probe
    # (2 seeds x 4) read 0.903 at the 250k checkpoint of a run whose
    # 40-episode score was 0.938 — about +-0.035 of noise, which walks
    # straight through a 1% improvement bar and kept a plateau stop from
    # ever firing. Cost scales with the pool, not the total: 40 episodes
    # across 5 processes is ~3 s, ~2% of the training between two probes.
    ap.add_argument("--probe-seeds", default="100,200,300,400,500", metavar="S,S,...")
    ap.add_argument("--probe-episodes", type=int, default=8)
    ap.add_argument("--probe-presets", default="datasheet", metavar="P,P,...",
                    help="sensor presets the probe averages over. The follow table is "
                         "reported under BOTH datasheet and hostile noise and brains rank "
                         "differently across them, so a probe on one preset can stop a run "
                         "at a checkpoint that is only good under that one. 'datasheet,"
                         "hostile' costs twice as much and scores what is reported")
    ap.add_argument("--log-std-max", type=float, default=LOG_STD_MAX, metavar="L",
                    help="cap on the policy's per-dim action log_std. NOTE the brain's "
                         "log_std_init is -0.5, so the default cap binds from step 0 and the "
                         "policy can never explore MORE than it started with; pass a large "
                         "value (e.g. 10) to disable and let the entropy bonus work")
    ap.add_argument("--checkpoint-every", type=int, default=250_000, metavar="STEPS",
                    help="save a numbered checkpoint this often so `select-brain` can pick the "
                         "best one deterministically afterwards (0: off)")
    ap.add_argument("--variety", action="store_true", help="train with furniture and a wandering duck (eval-brain --variety)")
    ap.add_argument("--polite-mix", default=None, metavar="M,M,...",
                    help="draw the person's politeness from this list EACH EPISODE instead of "
                         "holding --polite fixed, e.g. --polite-mix 0,0.55. Trained against one "
                         "kind of person a brain is paid for exploiting that kind: follow-v5, "
                         "trained against a person who always stops, learned to stand in its way")
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
    # The ONNX contract this run exports under — the striker trains on its own
    # 88-feature, five-action head, so every export in this file goes through
    # these dims rather than the follow defaults.
    export_dims = ((STRIKER_OBS_DIM, S_ACT_LOW, S_ACT_HIGH) if striker
                   else (BRAIN_OBS_DIM, ACT_LOW, ACT_HIGH))
    if striker and args.probe_every > 0:
        # The probe scores through `select_brain._score_one`, which builds a
        # FollowTask. A striker probe would be measuring the wrong task.
        raise SystemExit("--probe-every scores the follow benchmark; not available for --task striker")

    out = brains_dir() / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    mix = tuple(float(x) for x in args.polite_mix.split(",") if x.strip()) if args.polite_mix else ()
    if striker:
        fns = [make_striker_env_fn(args.seed * 1000 + i, task, args.fixed_preset)
               for i in range(args.envs)]
    else:
        fns = [make_env_fn(args.seed * 1000 + i, args.fixed_preset, args.variety, args.polite, mix)
               for i in range(args.envs)]
    # Fork the workers BEFORE importing torch, as the walk and trick trainers
    # do: a torch-initialized parent holds OpenMP/Accelerate pools and forking
    # those deadlocks on macOS. The default `fork` backend also drops the
    # per-worker MJCF compile and the pickled pipe round-trip per step that
    # `SubprocVecEnv` pays. Measured on this env at 8 workers (variety, one
    # machine, same load): setup 2.77 s -> 0.37 s and 1644 -> 1879
    # decisions/s, with rollouts BIT-IDENTICAL to the forkserver path
    # (tests/test_brain_vec_env.py). `MICRODUCK_VEC_ENV=subproc` restores it.
    venv = make_vec_env(fns)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

    venv = VecMonitor(as_sb3_vec_env(venv))
    # This trainer never called it, so it ran on torch's defaults: intra-op AND
    # inter-op both at the machine's core count, every one of those threads
    # spinning through rollouts made of 128-step brain decisions. Same policy
    # as the other two trainers now (machine.py). Called here, after
    # `make_vec_env` above has already forked the workers.
    configure_torch_cpu(torch)
    print(profile().describe())
    # A minibatch that DIVIDES the rollout buffer. The old `min(1024, n_steps *
    # envs)` did not at the default 12 envs: a 1536-step buffer under a 1024
    # batch made SB3 truncate every update into a 1024 and a 512 minibatch,
    # so half the gradient steps ran at 2/3 the intended batch and the last
    # 512 samples of every rollout carried a noisier gradient than the rest.
    # `ppo_batch_size` is the same rule the walk and trick trainers use.
    arch = [int(x) for x in args.net_arch.split(",") if x.strip()]
    if not arch:
        raise SystemExit("--net-arch needs at least one hidden size")
    if arch != [128, 128] or args.n_epochs != 5:
        print(f"[train-brain] net {arch}, n_epochs {args.n_epochs}", flush=True)
    log_std_max = None if LEGACY else float(args.log_std_max)
    batch = (min(1024, args.n_steps * args.envs) if LEGACY
             else ppo_batch_size(args.n_steps, args.envs))
    lr_sched = args.lr if LEGACY else linear_decay(args.lr, args.lr_end)
    if LEGACY:
        print("[train-brain] MICRODUCK_BRAIN_LEGACY: pre-fix hyperparameters "
              f"(batch {batch}, constant lr {args.lr}, no log_std cap)", flush=True)
    if args.init_from:
        prev = brains_dir() / args.init_from
        venv = VecNormalize.load(str(prev / "vecnormalize.pkl"), venv)
        model = PPO.load(str(prev / "model"), env=venv, device="cpu")
        # THIS launch is the authority over what the checkpoint recorded —
        # the same rule train_behavior applies to its warm starts. A warm
        # start with a different --envs would otherwise reinstate a batch
        # size that no longer divides the new buffer.
        model.batch_size = batch
        model.n_epochs = args.n_epochs
        model.lr_schedule = (lambda _: lr_sched) if LEGACY else lr_sched
        # Cap the INHERITED action std before a single step is taken: see
        # LOG_STD_MAX above. Without it a warm-start chain ratchets the std
        # up until only the clipped sampling noise carries the behavior,
        # and brain.onnx — which exports the mean — ships garbage.
        if log_std_max is not None:
            with torch.no_grad():
                model.policy.log_std.data.clamp_(max=log_std_max)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO("MlpPolicy", venv,
                    policy_kwargs=dict(net_arch=dict(pi=list(arch), vf=list(arch)),
                                       activation_fn=torch.nn.ELU, log_std_init=-0.5),
                    n_steps=args.n_steps, batch_size=batch,
                    n_epochs=args.n_epochs, learning_rate=lr_sched,
                    gamma=0.98, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.003, vf_coef=0.5, max_grad_norm=1.0,
                    device="cpu", seed=args.seed, verbose=0)

    log = (out / "progress.jsonl").open("a")
    t0 = time.time()

    ckpt_dir = out / "checkpoints"
    if args.checkpoint_every > 0:
        ckpt_dir.mkdir(exist_ok=True)
    plateau = PlateauDetector(args.plateau_patience, args.plateau_min_steps,
                              args.plateau_rel, args.plateau_window,
                              label="probe in_band" if args.probe_every > 0 else "ep_rew")
    if plateau.enabled:
        print(f"[train-brain] plateau stop armed: patience {plateau.patience}, "
              f"warmup {plateau.min_steps:,} steps, bar {plateau.rel:.1%}, "
              f"window {plateau.window}", flush=True)

    probe_seeds = [int(x) for x in args.probe_seeds.split(",") if x.strip()]
    probe_presets = [x.strip() for x in args.probe_presets.split(",") if x.strip()]
    probe_dir = out / ".probe" / "live"
    if args.probe_every > 0:
        probe_dir.mkdir(parents=True, exist_ok=True)
        print(f"[train-brain] deterministic probe every {args.probe_every:,} steps: "
              f"{len(probe_seeds)} seeds x {args.probe_episodes} episodes x "
              f"{'/'.join(probe_presets)}", flush=True)

    probe_cost_s = 0.0

    def probe(steps: int) -> float | None:
        """Score the DETERMINISTIC export on a few benchmark episodes.

        This is what the reward curve is not. Measured on a 2M run: in-band
        sat at 0.938 from the 250k checkpoint and 0.939 at 2M while `ep_rew`
        climbed 159 -> 177 the whole way. A stop watching reward never fires;
        one watching this stops at the first checkpoint that stops improving.

        Runs in a `spawn` pool of single-threaded processes. The training
        workers are forked children blocked on their semaphores while this
        runs, so it borrows cores that are idle anyway — measured at ~2 s
        against the ~2 min of training between probes.
        """
        import multiprocessing as mp

        from .select_brain import _score_one
        t_probe = time.time()
        model.save(str(probe_dir / "model"))
        venv.save(str(probe_dir / "vecnormalize.pkl"))
        export_brain(probe_dir, out=probe_dir / "brain.onnx")
        (probe_dir / "brain.json").write_text(
            json.dumps({"obs_version": BRAIN_OBS_VERSION, "decide_every": 5,
                        "target_cls": "person", "name": f"{args.run_name}@probe"}))
        # Fixed conditions on purpose — the named presets and this run's own
        # `--polite`, even under `--polite-mix`. A probe is a series compared
        # against ITSELF over time, so anything drawn per episode would put
        # noise where the signal is meant to be.
        jobs = [(str(probe_dir), "live", sd, args.probe_episodes, args.variety,
                 preset, args.polite)
                for preset in probe_presets for sd in probe_seeds]
        try:
            with mp.get_context("spawn").Pool(len(jobs)) as pool:
                rows = pool.map(_score_one, jobs)
        except Exception as exc:                       # a probe must never kill a run
            print(f"[train-brain] probe failed at {steps}: {exc}", flush=True)
            return None
        nonlocal probe_cost_s
        probe_cost_s += time.time() - t_probe
        return float(np.mean([r["in_band"] for r in rows]))

    switch_at = int(args.steps * args.decide_every_switch) if args.decide_every_start else 0
    if args.decide_every_start:
        venv.env_method("set_decide_every", args.decide_every_start)
        print(f"[train-brain] decision period {args.decide_every_start} control steps, "
              f"dropping to 5 at {switch_at:,}", flush=True)

    class Progress(BaseCallback):
        next_ckpt = args.checkpoint_every
        switched = False
        next_probe = args.probe_every
        next_snap = args.snap_steps or 1 << 62
        stop = False
        last_row: dict | None = None

        def _snapshot(self) -> None:
            """model.zip + vecnormalize.pkl + brain.onnx, via temp names, so a
            reader never sees a half-written file."""
            # SB3 appends ".zip" only to a suffix-less path, so "model.tmp"
            # lands literally and the replace stays atomic.
            self.model.save(str(out / "model.tmp"))
            os.replace(out / "model.tmp", out / "model.zip")
            venv.save(str(out / "vecnormalize.pkl.tmp"))
            os.replace(out / "vecnormalize.pkl.tmp", out / "vecnormalize.pkl")
            export_brain(out, *export_dims)

        def _on_rollout_end(self) -> None:
            if self.num_timesteps >= self.next_snap:
                self.next_snap = self.num_timesteps + args.snap_steps
                self._snapshot()
            # Re-assert the action-std cap every rollout: the entropy bonus
            # pushes log_std back up between clamps, so a load-time clamp
            # alone does not hold (train_behavior's callback does the same).
            if log_std_max is not None:
                with torch.no_grad():
                    self.model.policy.log_std.data.clamp_(max=log_std_max)
            ep = self.model.ep_info_buffer
            rew = float(np.mean([e["r"] for e in ep])) if ep else float("nan")
            ln = float(np.mean([e["l"] for e in ep])) if ep else float("nan")
            row = {"steps": int(self.num_timesteps), "ep_rew": round(rew, 3), "ep_len": round(ln, 1),
                   "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[train-brain] {row}", flush=True)
            self.last_row = row
            if switch_at and not self.switched and self.num_timesteps >= switch_at:
                # The exported brain.json records the DEPLOYMENT period, and
                # LearnedBrain runs at whatever it records — so the schedule
                # must always END at 5, never ship a coarser one.
                self.venv_ref.env_method("set_decide_every", 5)
                self.switched = True
                print(f"[train-brain] decision period -> 5 at {self.num_timesteps:,}",
                      flush=True)
            signal = rew
            have = bool(ep) and np.isfinite(rew)
            if args.probe_every > 0 and self.num_timesteps >= self.next_probe:
                self.next_probe += args.probe_every
                score = probe(int(self.num_timesteps))
                if score is not None:
                    row["probe"] = round(score, 4)
                    log.write(json.dumps({**row, "probe": round(score, 4)}) + "\n")
                    log.flush()
                    print(f"[train-brain] probe @{self.num_timesteps}: in_band {score:.3f} "
                          f"({probe_cost_s:.1f}s of probe so far)", flush=True)
                    signal, have = score, True
                else:
                    have = False
            elif args.probe_every > 0:
                have = False        # between probes there is nothing to judge on
            # Fed only when there is a real measurement: an empty episode
            # buffer reports nan, and feeding that would poison the mean.
            if have and plateau.update(int(self.num_timesteps), signal):
                self.stop = True
                print(f"[train-brain] plateau: {plateau.summary()}", flush=True)
            # Numbered checkpoints so `select-brain` can score the whole run
            # DETERMINISTICALLY afterwards and ship the best one. The reward
            # curve cannot make that call: it measures the noise-crutched
            # stochastic policy, and every run here so far has peaked and
            # then drifted (follow-v4's curve was flat from ~900k of 2M).
            if args.checkpoint_every > 0 and self.num_timesteps >= self.next_ckpt:
                tag = f"{int(self.num_timesteps):09d}"
                self.model.save(str(ckpt_dir / f"model_{tag}"))
                venv.save(str(ckpt_dir / f"vecnormalize_{tag}.pkl"))
                self.next_ckpt += args.checkpoint_every

        def _on_step(self) -> bool:
            # The clean SB3 stop: False ends collect_rollouts and breaks
            # learn(), so the finally-block save and the ONNX export below
            # still run and the run ends COMPLETE, not crashed.
            return not self.stop

    meta = {"name": args.run_name, "task": args.task, "decide_every": 5,
            "envs": args.envs, "steps": args.steps, "seed": args.seed,
            "fixed_preset": args.fixed_preset,
            "decide_every_start": args.decide_every_start or None,
            "probe_presets": probe_presets if args.probe_every > 0 else None,
            "net_arch": args.net_arch, "n_epochs": args.n_epochs,
        "n_steps": args.n_steps, "batch_size": batch, "lr": args.lr, "lr_end": args.lr_end,
            "log_std_max": log_std_max, "legacy_hparams": LEGACY, **git_state()}
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
                 "variety": args.variety, "polite": args.polite, "polite_mix": list(mix)}
    (out / "brain.json").write_text(json.dumps(meta, indent=2))
    cb = Progress()
    cb.venv_ref = venv
    try:
        model.learn(total_timesteps=args.steps,
                    callback=with_phase_callbacks(cb, BaseCallback))
    finally:
        # One last clamp before the artifact is written. The rollout-end
        # clamp leaves the FINAL gradient update free to push log_std back
        # over the cap (measured: -0.49994 against a -0.5 cap), and this
        # checkpoint is what the next `--init-from` inherits and what
        # `export_brain` reads. The invariant belongs to what ships.
        if log_std_max is not None:
            with torch.no_grad():
                model.policy.log_std.data.clamp_(max=log_std_max)
        model.save(str(out / "model"))
        venv.save(str(out / "vecnormalize.pkl"))
        venv.close()
        log.close()
    if plateau.fired:
        # A terminating line so a later reader can tell an early stop from a
        # crash. `steps` is the ORIGINAL target so the lab's /brains progress
        # still reads 100% (the run completed, it just completed early), and
        # the last real reward is repeated so the charted curve keeps a
        # well-formed final point.
        last = cb.last_row or {}
        with (out / "progress.jsonl").open("a") as f:
            f.write(json.dumps({
                "steps": args.steps, "total": args.steps,
                "ep_rew": last.get("ep_rew"), "ep_len": last.get("ep_len"),
                "elapsed_s": round(time.time() - t0, 1), "done": True,
                **plateau.record(int(last.get("steps") or 0))}) + "\n")
    print(f"[train-brain] exported {export_brain(out, *export_dims)}", flush=True)


if __name__ == "__main__":
    main()
