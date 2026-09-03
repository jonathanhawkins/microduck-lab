"""Train a teachable behavior, streaming progress for the web viewer.

    uv run train-behavior one_leg --run-name chat-123 --steps 2_500_000

Writes into runs/<run-name>/:
  progress.jsonl   one line per PPO rollout: steps, ep_rew, ep_len, per-term means
  live.onnx        refreshed every --snap-steps (atomic replace) — the lab
                   hot-loads it onto the trainee duck so you can watch the
                   policy improve
  model.zip + vecnormalize.pkl   refreshed with every snapshot (atomic replace),
                   so a warm restart can pick up mid-run; finals on completion
                   (same artifacts as train-walk) plus policy.onnx

Warm starts: `--init-from <run_dir>` loads model.zip + vecnormalize.pkl. If it
points at THIS run's own dir, training resumes: the step counter continues and
`--steps` stays an ABSOLUTE target (SB3's learn() treats total_timesteps as
*additional* when reset_num_timesteps=False, so the remaining budget is
computed here); progress lines append to the same progress.jsonl. That is how
viz_server rescales the env count mid-run: kill, relaunch with `--init-from`
the run's own dir and a new `--envs`. Pointing at a DIFFERENT run dir is a
fine-tune instead: the weights carry over but the counter resets, so `--steps`
is a full fresh budget (e.g. retrain a finished trick under an edited
`--weights-json` scorecard).

The bilateral mirror loss (symmetry.py) is on by default, but only for
recipes it actually fits: `--symmetry-coef` defaults to the behavior's own
`symmetric` flag (and is forced off for any run carrying a motion clip, whose
phase the mirror map scrambles). Passing the flag overrides that either way.
This matters because viz_server's teach endpoint launches this module WITHOUT
the flag, so the default is what every taught trick trains under.

Early stopping is available but OFF: `--plateau-patience N` (or
$MICRODUCK_PLATEAU_PATIENCE, for lab-launched runs that pass no flags) ends a
run whose smoothed ep_rew has stopped improving — see plateau.py for why it
is opt-in. It stops the run CLEANLY: the final snapshot, the ONNX export and
the terminating "done" line all happen, with `"stopped": "plateau"` on that
line so a short curve is never mistaken for a crash.

Designed to be launched as a subprocess by viz_server's teach endpoint, but
works standalone too.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .behaviors import BEHAVIORS, Behavior, BehaviorEnv, is_symmetric
from .plateau import PlateauDetector
from .plateau import env_defaults as plateau_env_defaults
from .ppo_hparams import (
    DEFAULT_DESIRED_KL,
    DEFAULT_SYMMETRY_COEF,
    N_STEPS,
    UPSTREAM_DESIRED_KL,
    VF_COEF,
    configure_torch_cpu,
    ppo_batch_size,
)
from .train import RUNS_DIR
from .vec_env import as_sb3_vec_env, make_vec_env


# Linear learning-rate decay. EVERY run so far has peaked and then come
# apart — ep_rew 354->223, 177->29, 157->77, and the from-scratch BAM run
# peaked at 2.7M (ep_len 62) and decayed to 43 by 12M. That is the classic
# constant-rate signature: once the policy is near a good solution, a rate
# sized for early exploration overshoots on every update and the policy random-
# walks away from it.
#
# This is NOT rsl_rl's KL controller (see symmetry.py): that one is off by
# default because upstream's 0.01 KL target is unreachable at our batch size
# and pins the rate to its floor. A plain linear decay needs no calibration
# against batch size — it just stops the late thrash.
# Env-overridable (MICRODUCK_LR_START/END): the lab's /teach endpoint has no
# lr parameters, and a warm continuation of an unstable run needs a cooler
# rate than a fresh run — measured: at ~5.5e-4 a near-converged policy
# boom-busted three times in one leg (ep_len 396->10, 42->12).
def _env_lr(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


LR_START, LR_END = _env_lr("MICRODUCK_LR_START", 1e-3), _env_lr("MICRODUCK_LR_END", 1e-4)


def linear_decay(start: float, end: float):
    """SB3 schedule: called with progress_remaining, 1.0 -> 0.0."""
    def f(progress_remaining: float) -> float:
        return end + (start - end) * float(progress_remaining)
    return f


SNAP_STEPS = 150_000  # export a live snapshot roughly every ~15 s of wall time
# Hard cap on the policy's per-dim action log_std (std <= ~0.6). Past ~std 1
# the clipped Gaussian degenerates into bang-bang sampling that the entropy
# bonus loves and the DETERMINISTIC export can't reproduce — see the clamp
# sites for the measured 2fca3a failure. Warm starts inherit log_std, so the
# cap must bind on load AND every rollout.
LOG_STD_MAX = -0.5


def make_env(behavior_id: str, rank: int, seed: int,
             weight_overrides: dict[str, float] | None = None):
    def _init():
        b = BEHAVIORS[behavior_id]
        kw = dict(
            seed=seed + rank,
            max_episode_s=b.episode_s,
            domain_rand=False,
            random_yaw=False,
        )
        # Locomotion is the GPU run task: BAM physics, the velocity DR
        # subset, random yaw. Tricks stay on the cheap XML/no-DR path —
        # they don't saturate the servos the way a run does.
        if b.forward_cmd:
            kw.update(domain_rand=True, random_yaw=True, actuator="bam",
                      height_termination=False)
        return BehaviorEnv(behavior_id, weight_overrides=weight_overrides, **kw)
    return _init


def _progress_callback_cls(BaseCallback):
    """Built after `make_vec_env` so this module can be imported without torch."""

    class ProgressCallback(BaseCallback):
        """One JSONL line per rollout + periodic atomic snapshots.

        A snapshot is live.onnx AND model.zip + vecnormalize.pkl: the latter two
        are what lets viz_server warm-restart this process with --init-from to
        change the env count. Everything lands via temp-name + os.replace so a
        restart never reads a half-written file.
        """

        def __init__(self, out: Path, venv, total_steps: int,
                     snap_steps: int = SNAP_STEPS, start_steps: int = 0,
                     plateau: PlateauDetector | None = None,
                     checkpoint_every: int = 0):
            super().__init__()
            self.out = out
            self.venv = venv
            self.total_steps = total_steps
            self.snap_steps = snap_steps
            # Off unless a detector with patience > 0 was passed: a disabled
            # one is a no-op, so the default path keeps returning True from
            # every _on_step exactly as before.
            self.plateau = plateau if plateau is not None else PlateauDetector()
            self._stop = False
            # Numbered checkpoints (0 = off, the default). `live.onnx` and
            # `model.zip` are OVERWRITTEN every snapshot, so a finished run
            # keeps only its last policy — and this trainer's own comment
            # below says the last one is not reliably the best. `select-run`
            # scores these on ACHIEVED GROUND SPEED, which is the criterion
            # that comment asks a future selector to use.
            self.checkpoint_every = int(checkpoint_every)
            self.next_ckpt = start_steps + self.checkpoint_every
            self.term_sums: dict[str, float] = {}
            self.term_steps = 0
            self.ep_lens: list[int] = []
            # Anchor to the warm-start step so a resumed run doesn't burn its first
            # rollouts re-snapshotting to catch a counter it inherited.
            self.next_snap = start_steps + snap_steps
            self.snapshots = 0
            self.t0 = time.time()
            self._prev_steps = start_steps
            self._prev_t = self.t0

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                sums = info.get("episode_rewards")
                if sums:
                    for k, v in sums.items():
                        self.term_sums[k] = self.term_sums.get(k, 0.0) + v
                    ep = info.get("episode")
                    if ep:
                        self.ep_lens.append(int(ep["l"]))
                        self.term_steps += int(ep["l"])
            # False ends collection and breaks out of learn() — the ONLY
            # clean SB3 stop, and the reason the plateau verdict (taken in
            # _on_rollout_end, where the ep_rew series lives) is parked on a
            # flag instead of raising. main() then runs its normal final
            # path: _snapshot, the ONNX export, the "done" line. Costs one
            # extra step of collection, which is why the flag is read here
            # rather than mid-rollout.
            return not self._stop

        def _snapshot(self) -> None:
            import torch

            from .export_onnx import OnnxWalkPolicy
            wrapper = OnnxWalkPolicy(
                self.model.policy, self.venv.obs_rms.mean, self.venv.obs_rms.var,
                self.venv.clip_obs,
            ).eval()
            tmp = self.out / "live.onnx.tmp"
            dummy = torch.zeros(1, 61, dtype=torch.float32)
            torch.onnx.export(wrapper, (dummy,), str(tmp), input_names=["obs"],
                              output_names=["actions"], opset_version=17,
                              dynamo=False)
            tmp.replace(self.out / "live.onnx")
            # SB3 only appends ".zip" to suffix-less paths, so the ".tmp" names
            # land literally and the os.replace stays atomic.
            self.model.save(str(self.out / "model.tmp"))
            os.replace(self.out / "model.tmp", self.out / "model.zip")
            self.venv.save(str(self.out / "vecnormalize.pkl.tmp"))
            os.replace(self.out / "vecnormalize.pkl.tmp",
                       self.out / "vecnormalize.pkl")
            self.snapshots += 1

        def _on_rollout_end(self) -> None:
            # Re-assert the action-std cap every rollout: the entropy bonus
            # pushes log_std up between clamps, and past ~std 1 the clipped
            # Gaussian goes bang-bang (see the warm-start clamp's comment).
            import torch
            with torch.no_grad():
                self.model.policy.log_std.data.clamp_(max=LOG_STD_MAX)
            ep_rew = self.model.ep_info_buffer
            rew = float(np.mean([e["r"] for e in ep_rew])) if ep_rew else 0.0
            length = float(np.mean([e["l"] for e in ep_rew])) if ep_rew else 0.0
            terms = {
                k: round(v / max(self.term_steps, 1), 4)  # per-step mean
                for k, v in self.term_sums.items()
            }
            # Rollout windows shorter than an episode contain no episode boundary
            # and tally nothing — carry the last real breakdown forward so
            # consumers (the teach panel's live bars) don't blink empty.
            if terms:
                self._last_terms = terms
            else:
                terms = getattr(self, "_last_terms", {})
            # NO best-checkpoint selection. It was tried and it made things
            # WORSE: scoring on `keep_pace * ep_len` is ~90% ep_len (pace
            # spans 19% across a run, length 165%), so it collapsed into
            # "longest-surviving", and _run_speed pays 0.29 at ZERO velocity —
            # a motionless duck scores well. Measured on the three runs it
            # shipped: 0.236 vs 0.359 m/s, 0.046 vs 0.112, 0.320 vs 0.343 —
            # the FINAL policy beat the "best" one every time. The premise was
            # wrong too: a falling ep_rew late in training did not mean the
            # policy got worse at the task, only that it survived less while
            # moving more. Re-introduce only with a criterion scored on
            # ACHIEVED GROUND SPEED, not on a reward term.
            now = time.time()
            dt = now - self._prev_t
            ds = int(self.num_timesteps) - self._prev_steps
            sps = (ds / dt) if dt > 0 else 0.0
            self._prev_steps = int(self.num_timesteps)
            self._prev_t = now
            line = {
                "steps": int(self.num_timesteps),
                "total": self.total_steps,
                "ep_rew": round(rew, 2),
                "ep_len": round(length, 1),
                "terms": terms,
                "snapshots": self.snapshots,
                "elapsed_s": round(now - self.t0, 1),
                "sps": round(sps),
            }
            with open(self.out / "progress.jsonl", "a") as f:
                f.write(json.dumps(line) + "\n")
            self.term_sums, self.term_steps, self.ep_lens = {}, 0, []
            if self.num_timesteps >= self.next_snap:
                self._snapshot()
                self.next_snap += self.snap_steps
            if self.checkpoint_every > 0 and self.num_timesteps >= self.next_ckpt:
                d = self.out / "checkpoints"
                d.mkdir(exist_ok=True)
                tag = f"{int(self.num_timesteps):09d}"
                self.model.save(str(d / f"model_{tag}"))
                self.venv.save(str(d / f"vecnormalize_{tag}.pkl"))
                self.next_ckpt += self.checkpoint_every
            # Same ep_rew series that just went into progress.jsonl. Disabled
            # unless --plateau-patience (or $MICRODUCK_PLATEAU_PATIENCE) is
            # positive; `steps` is the ABSOLUTE counter, so a warm restart's
            # warmup is measured from the run's own start, not this process's.
            # `ep_rew` gates it because an empty buffer reports 0.0, which is
            # not a measurement: on a recipe whose early episodes score
            # negative, those leading zeros would become an unbeatable `best`
            # and stop a run that was climbing from -50 toward -10.
            if ep_rew and self.plateau.update(int(self.num_timesteps), rew):
                self._stop = True
                print(f"plateau: {self.plateau.summary()}", flush=True)

    return ProgressCallback


def symmetry_coef_for(behavior: Behavior, explicit: float | None) -> float:
    """The mirror-loss weight this run trains under.

    `explicit` is the CLI's --symmetry-coef, or None when it was not passed.
    It is authoritative in BOTH directions: 0 turns the prior off for a
    symmetric recipe, and a positive value turns it back ON for an asymmetric
    one (worth having — a one-sided trick with a symmetric warm start may
    still want it early). Only the unspecified case consults the behavior.

    The default matters because viz_server's /teach launches this module
    without the flag, so every taught trick — including the deliberately
    one-sided ones — would otherwise train under the mirror prior.
    """
    if explicit is not None:
        return float(explicit)
    return DEFAULT_SYMMETRY_COEF if is_symmetric(behavior) else 0.0


def build_parser() -> argparse.ArgumentParser:
    """The CLI. Split out of main() so tests can read the real defaults —
    --symmetry-coef's default of None is load-bearing (it is what "the user
    did not say" looks like), and a test asserting it must see argparse's own
    value, not a copy."""
    ap = argparse.ArgumentParser()
    ap.add_argument("behavior", choices=sorted(BEHAVIORS))
    # 32, not this Mac's 18 cores: `uv run bench-envs` measures PPO throughput
    # rising past cpu_count (the rollout and the multi-threaded torch update
    # take turns, so extra workers fill the cores the serial phases leave
    # idle — at 16 envs the workers are only ~11% busy). Idle sweep
    # 2026-08-30: 16 → 14.3k steps/s, 24 → 15.6k, 32 → 16.5k, ~17.1k
    # asymptote; with the lab + browser live the same ordering holds and the
    # margin widens. See viz_server.BASE_ENVS, which mirrors this and records
    # why the old 16 (a confounded live-lab test) was wrong.
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None,
                    help="run dir with model.zip + vecnormalize.pkl: continue that "
                         "training (a different --envs is fine — that's the point); "
                         "--steps stays an absolute target, not extra steps")
    ap.add_argument("--snap-steps", type=int, default=SNAP_STEPS,
                    help="steps between live.onnx snapshots (tests dial this down)")
    ap.add_argument("--weights-json", default=None,
                    help='JSON dict of reward-weight overrides, e.g. '
                         '\'{"spin_fast": 3.0}\' — keys are RewardTerm keys')
    ap.add_argument("--lr-start", type=float, default=None,
                    help="initial learning rate (decays linearly to --lr-end). "
                         "Lower it hard when fine-tuning a policy that already "
                         "works: the default 1e-3 is sized for discovery and "
                         "will destroy a good gait in the first few updates")
    ap.add_argument("--lr-end", type=float, default=None)
    ap.add_argument("--desired-kl", type=float, default=DEFAULT_DESIRED_KL,
                    help="KL target for rsl_rl's adaptive learning rate. OFF "
                         "by default: upstream's "
                         f"{UPSTREAM_DESIRED_KL} is unreachable at our batch "
                         "size and pins the rate to its floor (see symmetry.py "
                         "for the three-arm A/B). Worth revisiting if the "
                         "batch grows a lot")
    ap.add_argument("--overlap", action="store_true",
                    default=os.environ.get("MICRODUCK_OVERLAP", "") not in ("", "0"),
                    help="collect rollout k+1 while the update on rollout k "
                         "runs in a background thread (SymmetryPPO.learn). "
                         "Data trains one update stale — A/B'd, see README. "
                         "Also enabled by MICRODUCK_OVERLAP=1, which is how "
                         "lab-launched runs opt in (viz_server passes no "
                         "flags)")
    ap.add_argument("--update-device",
                    default=os.environ.get("MICRODUCK_UPDATE_DEVICE", "auto"),
                    help="device for the PPO minibatch loop only (rollout "
                         "inference stays on cpu, where batch-32 beats GPU "
                         "dispatch latency). 'auto' (default) = mps when "
                         "available; 'cpu' opts out. Measured 3.9x on the "
                         "update-sized matmuls on this machine")
    ap.add_argument("--symmetry-coef", type=float, default=None,
                    help="weight of the bilateral mirror loss added to the PPO "
                         "loss (microduck_rl's SYMMETRY_CFG value; 0 disables). "
                         "Left/right consistency for free — but it is a WRONG "
                         "prior for a deliberately one-sided trick, so the "
                         f"default is {DEFAULT_SYMMETRY_COEF} for a symmetric "
                         "behavior and 0 for an asymmetric one (Behavior."
                         "symmetric, or any run carrying a motion clip). "
                         "Passing this overrides that, in either direction")
    # Early stopping (plateau.py). OFF by default (patience 0) and staying
    # that way until a seed-matched A/B at matched step counts says otherwise
    # — a flat curve is not proof that nothing is being consolidated, and
    # AGENTS.md #4 is explicit that budget/throughput changes earn their
    # default. Env vars are how the lab's /teach opts in: it launches this
    # module with no flags (same escape hatch as MICRODUCK_LR_START).
    pd = plateau_env_defaults()
    ap.add_argument("--plateau-patience", type=int, default=pd["patience"],
                    help="stop once the smoothed ep_rew has not improved for "
                         "this many consecutive rollouts. 0 (default) = never "
                         "stop early; run the full --steps. Also settable as "
                         "$MICRODUCK_PLATEAU_PATIENCE")
    ap.add_argument("--plateau-min-steps", type=int, default=pd["min_steps"],
                    help="warmup: no plateau may fire before this many steps "
                         "(absolute, so a warm restart counts its whole run). "
                         "Early training legitimately looks flat. "
                         "$MICRODUCK_PLATEAU_MIN_STEPS")
    ap.add_argument("--plateau-rel", type=float, default=pd["rel"],
                    help="relative improvement over the best smoothed ep_rew "
                         "that counts as still learning. "
                         "$MICRODUCK_PLATEAU_REL")
    ap.add_argument("--plateau-window", type=int, default=pd["window"],
                    help="rollouts in the smoothing window — one rollout's "
                         "ep_rew is far too noisy to judge alone. "
                         "$MICRODUCK_PLATEAU_WINDOW")
    ap.add_argument("--net-arch", default="512,256,128", metavar="H,H,...",
                    help="hidden sizes for the policy and value MLPs. The default came "
                         "from the GPU stack, where a big net is free because it runs "
                         "~250x more samples; here the network is 54%% of wall time "
                         "(12%% rollout forward + 42%% update) and 256-128 measured 2.29x "
                         "cheaper on that block, 128-128 3.42x. Never swept for QUALITY, "
                         "so it is still the default")
    ap.add_argument("--shared-trunk", action="store_true",
                    help="share all but the last hidden layer between the actor and the "
                         "critic (symmetry.SharedTrunk), computing the expensive early "
                         "layers once instead of twice. ~40%% off the network. The value "
                         "loss then backpropagates into the actor's features, which "
                         "rsl_rl deliberately avoids — hence off by default")
    ap.add_argument("--n-epochs", type=int, default=5,
                    help="PPO epochs per rollout. A near-linear scale on the update, "
                         "which is 42%% of wall time")
    ap.add_argument("--distill", action="store_true",
                    default=os.environ.get("MICRODUCK_DISTILL", "") not in ("", "0"),
                    help="warm-start a LOCOMOTION recipe by cloning the shipped walker "
                         "(distill.py) instead of starting from scratch. Measured reason: "
                         "under BAM every policy trained here from scratch either falls "
                         "(18/20) or is stable and slow (0.13 m/s), while alpha_walking is "
                         "stable AND 0.21 m/s — this turns 'discover a gait' into 'make an "
                         "existing gait faster'. Ignored when --init-from is given, and for "
                         "non-locomotion tricks (a clone of a walker is a wrong prior for "
                         "a headstand). The clone is CACHED, so it is paid once")
    ap.add_argument("--distill-teacher", default=None,
                    help="ONNX to clone (default: the shipped alpha_walking)")
    ap.add_argument("--distill-no-critic", action="store_true",
                    help="clone only the ACTOR, as the first version did. Its own header "
                         "predicted the consequence — 'the critic starts untrained ... "
                         "expect a dip' — and the measurement was that the dip destroys "
                         "the gait: every checkpoint of four 1M-step arms fell in 100%% "
                         "of episodes. Kept as the A/B baseline")
    ap.add_argument("--symmetry-augment", action="store_true",
                    default=os.environ.get("MICRODUCK_SYMMETRY_AUGMENT", "") not in ("", "0"),
                    help="rsl_rl's OTHER symmetry mode: double every minibatch with its "
                         "mirror image and put the mirrored halves through the surrogate "
                         "and value losses, instead of only penalising asymmetry. The one "
                         "lever that buys sample EFFICIENCY rather than more samples, which "
                         "is the only currency a CPU trainer has — but the mirrored sample "
                         "reuses the original's old_log_prob, so its importance ratio is "
                         "biased. OFF until an A/B says otherwise")
    ap.add_argument("--checkpoint-every", type=int, default=0, metavar="STEPS",
                    help="keep a numbered checkpoint this often so `select-run` can "
                         "score the whole run on ACHIEVED GROUND SPEED afterwards. "
                         "0 (default) = off, and live.onnx/model.zip keep being "
                         "overwritten as before")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    b = BEHAVIORS[args.behavior]
    # Resolved BEFORE anything is written or built: behavior.json, the warm-start
    # path and the fresh-model path must all agree on one number.
    symmetry_coef = symmetry_coef_for(b, args.symmetry_coef)
    if args.symmetry_coef is None and symmetry_coef == 0.0:
        print(f"{b.id}: one-sided recipe — mirror loss off "
              f"(pass --symmetry-coef {DEFAULT_SYMMETRY_COEF} to force it on)")
    steps = args.steps or b.default_steps
    weights: dict[str, float] = json.loads(args.weights_json) if args.weights_json else {}
    run_name = args.run_name or time.strftime(f"{b.id}-%Y%m%d-%H%M%S")
    out = RUNS_DIR / run_name
    out.mkdir(parents=True, exist_ok=True)
    # Weights go in behavior.json so a restarted/inspected run can't silently
    # train under a different scorecard than the one on record.
    (out / "behavior.json").write_text(json.dumps(
        {"behavior": b.id, "steps": steps, "weights": weights,
         "symmetry_coef": symmetry_coef, "desired_kl": args.desired_kl,
         "net_arch": args.net_arch, "shared_trunk": args.shared_trunk,
         "n_epochs": args.n_epochs}))

    # Fork workers BEFORE importing torch. A torch-initialized parent has
    # OpenMP/Accelerate thread pools; forking them deadlocks on macOS.
    # Shares one compiled mjModel across the workers (vec_env.py).
    # MUST be exported before the vec-env workers fork: lifetime-ramped
    # reward terms seed their counter from it (see behaviors._compute_reward).
    # A same-dir --init-from is a warm RESTART of an ongoing run, so the ramp
    # resumes at strength instead of whiplashing the policy; progress.jsonl's
    # last counter is that run's step count. Fresh runs and cross-dir
    # fine-tunes start the ramp from zero, which is the intended curriculum.
    ramp_offset = 0
    if args.init_from:
        prog = Path(args.init_from) / "progress.jsonl"
        if (Path(args.init_from).resolve() == out.resolve()) and prog.exists():
            try:
                last = [json.loads(ln) for ln in prog.read_text().splitlines() if ln.strip()]
                ramp_offset = int(last[-1].get("steps", 0)) // max(args.envs, 1)
            except Exception:
                ramp_offset = 0
    os.environ["MICRODUCK_RAMP_OFFSET"] = str(ramp_offset)

    venv = make_vec_env([make_env(b.id, i, args.seed, weights or None)
                         for i in range(args.envs)])

    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
    from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

    from .symmetry import FastActorCriticPolicy, SymmetryPPO

    # Distillation runs AFTER the fork, not before. It imports torch and
    # builds its own env, and a torch-initialized parent deadlocks the
    # macOS fork — the exact trap this file's own comment above warns about,
    # walked into once (four runs hung at 0% CPU with only behavior.json
    # written). The workers are already alive and blocked on their
    # semaphores while this runs, so it costs wall time, not correctness.
    if args.distill and not args.init_from:
        if not b.forward_cmd:
            print(f"{b.id}: not a locomotion recipe — --distill ignored "
                  "(a cloned walker is the wrong prior for a trick)")
        else:
            from .distill import ensure_distilled
            args.init_from = str(ensure_distilled(
                args.distill_teacher, seed=args.seed,
                critic=not args.distill_no_critic))
            print(f"{b.id}: warm-starting from the distilled walker at {args.init_from}")

    configure_torch_cpu(torch)

    # Resolved here, after torch import. "auto" = mps only when the minibatch
    # is large enough to pay for the device hop — measured on the real recipe
    # (seed 4, quiet machine): at 32 envs / batch 1024 MPS is a slight LOSS
    # (13.5k vs 14.8k steps/s), at 64 envs / batch 4096 a big win (18k vs
    # 12.5k). Quality is device-independent (ep_rew 184.9 vs 185.1 at 0.5M,
    # same seed). None/cpu leaves the whole update on the CPU as before.
    update_device = args.update_device
    if update_device == "auto":
        big_batch = ppo_batch_size(N_STEPS, args.envs) >= 2048
        update_device = ("mps" if big_batch
                         and torch.backends.mps.is_available() else None)
    elif update_device in ("", "cpu", "none"):
        update_device = None

    venv = VecMonitor(as_sb3_vec_env(venv))
    batch = ppo_batch_size(N_STEPS, args.envs)

    arch = [int(x) for x in args.net_arch.split(",") if x.strip()]
    if not arch:
        raise SystemExit("--net-arch needs at least one hidden size")
    if args.shared_trunk:
        if len(arch) < 2:
            raise SystemExit("--shared-trunk needs at least two hidden sizes: all but "
                             "the last are shared, the last is the per-head layer")
        from .symmetry import SharedTrunk
        net_kwargs = dict(
            features_extractor_class=SharedTrunk,
            features_extractor_kwargs=dict(arch=tuple(arch[:-1]),
                                           activation_fn=torch.nn.ELU),
            net_arch=dict(pi=[arch[-1]], vf=[arch[-1]]),
        )
    else:
        net_kwargs = dict(net_arch=dict(pi=list(arch), vf=list(arch)))
    if arch != [512, 256, 128] or args.shared_trunk or args.n_epochs != 5:
        print(f"net: {'shared ' if args.shared_trunk else ''}{arch}, "
              f"n_epochs {args.n_epochs}")
    ProgressCallback = _progress_callback_cls(BaseCallback)
    resume = False
    if args.init_from:
        prev = Path(args.init_from)
        # Decided HERE, before anything reads it: the learning-rate choice
        # below used to run ~30 lines ahead of this assignment, so it always
        # saw the `resume = False` initializer and every same-dir warm RESTART
        # (viz_server's env-count rescale, a live weight edit) silently
        # continued on the cool fine-tune schedule instead of resuming its own.
        resume = prev.resolve() == out.resolve()
        venv = VecNormalize.load(str(prev / "vecnormalize.pkl"), venv)
        # custom_objects: adopt the lean rollout forward even for checkpoints
        # saved before FastActorCriticPolicy existed (identical parameters —
        # the class only overrides forward()).
        model = SymmetryPPO.load(
            str(prev / "model.zip"), env=venv, device="cpu",
            custom_objects={"policy_class": FastActorCriticPolicy})
        # This launch is the authority (same rule as symmetry_coef below); a
        # checkpoint from before the attribute existed carries nothing.
        model.overlap_update = args.overlap
        model.update_device = update_device
        # This launch is the authority, same rule as symmetry_coef below.
        model.symmetry_augment = args.symmetry_augment
        # SB3's load() replays the checkpoint's __dict__, including the
        # previous run's batch_size. Helpers change --envs at runtime, and
        # the new buffer (n_steps * n_envs) may not divide that old size
        # (SB3 then truncates the last minibatch). Recompute from THIS
        # launch's env count.
        model.batch_size = batch
        model.n_epochs = args.n_epochs
        # Same for the symmetry coefficient: a run saved under a different
        # value would silently reinstate its own. This launch is the authority.
        model.symmetry_coef = symmetry_coef
        # Same rule for the KL controller: the CLI is the authority over
        # whatever the checkpoint recorded. A checkpoint written before the
        # controller existed carries neither attribute, hence the explicit set
        # and the _adaptive_lr guard (None = "adopt the optimizer's rate").
        # Warm starts default to a COOL schedule: restarting the decay at
        # the discovery-sized 1e-3 re-shocked converged brains on every leg
        # (one relaunch chain destroyed a banked 0.61 m/s policy, ep_rew
        # -1437). Explicit --lr-start still wins.
        lr0 = args.lr_start if args.lr_start is not None else (
            LR_START if resume else 2e-4)
        lr1 = args.lr_end if args.lr_end is not None else (
            LR_END if resume else 3e-5)
        model.lr_schedule = linear_decay(lr0, lr1)
        model.desired_kl = (args.desired_kl
                            if args.desired_kl and args.desired_kl > 0 else None)
        # Warm starts consolidate too — without this a fine-tune keeps its
        # constant entropy bonus forever and re-learns the noise crutch.
        model.ent_anneal = True
        if getattr(model, "_ent0", None) is None:
            model._ent0 = 0.01
        # Cap the INHERITED action std. Every --init-from reloads the previous
        # run's log_std, and tonight's long headstand chain ratcheted the leg
        # dims to log_std ~3.2 (std 21-26 in an action space clipped far
        # narrower): sampled actions became bang-bang slams whose direction
        # bias — not the mean — carried the behavior. Measured 2026-09-01 on
        # 2fca3a: stochastic episodes survive 6.9 s, the DETERMINISTIC mean
        # falls in 0.5 s on every seed — and the exported .onnx ships the
        # mean. The cap forces the mean to carry the skill.
        with torch.no_grad():
            model.policy.log_std.data.clamp_(max=LOG_STD_MAX)
        if getattr(model, "_adaptive_lr", None) is None:
            model._adaptive_lr = None
        # Same-dir init-from is viz_server's env-count rescale: the step counter
        # and progress.jsonl continue toward the original target. A different
        # dir is a fine-tune warm start (e.g. retrain a finished policy under an
        # edited scorecard): weights carry over, the counter starts fresh so
        # --steps is a full new budget.
        print(f"warm-started from {prev} at {model.num_timesteps} steps, "
              f"{args.envs} envs ({'resuming' if resume else 'fine-tuning'})")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=100.0)
        model = SymmetryPPO(
            FastActorCriticPolicy, venv,
            policy_kwargs=dict(**net_kwargs,
                               activation_fn=torch.nn.ELU, log_std_init=0.0,
                               # Profiled on MPS: torch falls back to
                               # _single_tensor_adam (3.8 ms/step of tiny
                               # kernel launches); fused is the same math in
                               # one kernel (~1 ms). Only set when the update
                               # runs on MPS so the CPU path keeps stock
                               # optimizer numerics.
                               **({"optimizer_kwargs": {"fused": True}}
                                  if update_device else {})),
            # Matched to the official stack's PPO (rsl_rl cfg in
            # microduck_rl): entropy 0.01 and lr 1e-3. Ours had half the
            # entropy and a third the learning rate, which — with far fewer
            # samples — collapses to the safest policy available, i.e.
            # standing still. Locomotion has to survive early exploration.
            n_steps=N_STEPS, batch_size=batch, n_epochs=args.n_epochs,
            learning_rate=linear_decay(
                args.lr_start if args.lr_start is not None else LR_START,
                args.lr_end if args.lr_end is not None else LR_END),
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            vf_coef=VF_COEF, max_grad_norm=1.0, device="cpu", seed=args.seed,
            verbose=1,
            # Bilateral mirror loss (symmetry.py). At coefficient 0 this
            # class is stock PPO; above 0 it also pays for pi(mirror(obs)) ==
            # mirror(pi(obs)), which is the one lever that buys sample
            # EFFICIENCY rather than more samples — the only currency a CPU
            # trainer running ~250x fewer samples than the GPU stack has.
            symmetry_coef=symmetry_coef,
            desired_kl=args.desired_kl,
            ent_anneal=True,
            overlap_update=args.overlap,
            update_device=update_device,
            symmetry_augment=args.symmetry_augment,
        )
    start = int(model.num_timesteps) if resume else 0
    plateau = PlateauDetector(patience=args.plateau_patience,
                              min_steps=args.plateau_min_steps,
                              rel=args.plateau_rel, window=args.plateau_window)
    if plateau.enabled:
        print(f"plateau stop armed: patience {plateau.patience} rollouts, "
              f"window {plateau.window}, rel {plateau.rel}, "
              f"warmup {plateau.min_steps} steps")
    cb = ProgressCallback(out, venv, steps, snap_steps=args.snap_steps,
                          start_steps=start, plateau=plateau,
                          checkpoint_every=args.checkpoint_every)
    # SB3 treats total_timesteps as ADDITIONAL steps when reset_num_timesteps
    # is False (_setup_learn adds num_timesteps back in), so subtract to keep
    # --steps an absolute target across warm restarts. Pinned by
    # tests/test_train_resume.py.
    remaining = max(steps - start, 0)
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=cb, progress_bar=False,
                    reset_num_timesteps=not resume)

    # Only if learn() actually ran: SB3 binds callback.model in init_callback,
    # so with remaining == 0 (an --init-from of a run already at its target,
    # which scale() produces when a helper is added near the end of a stage)
    # cb has no .model and _snapshot raises AttributeError, failing the job and
    # stalling a staged chain.
    if remaining > 0:
        cb._snapshot()  # final live.onnx + model.zip + vecnormalize.pkl
    from .export_onnx import export
    export(out, out / "policy.onnx")
    # A plateau stop is a COMPLETED run, so `steps`/`total`/`done` stay as
    # they are (the lab reads them as 100%, and the per-rollout lines already
    # carry the real counter). What gets added is why the curve is short —
    # without it a later reader cannot tell an early stop from a crash.
    final = {"steps": steps, "total": steps, "done": True}
    if plateau.fired:
        final.update(plateau.record(int(model.num_timesteps)))
    with open(out / "progress.jsonl", "a") as f:
        f.write(json.dumps(final) + "\n")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
