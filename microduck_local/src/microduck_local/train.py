"""Train the local walking policy with SB3 PPO across CPU cores.

    uv run train-walk --envs 16 --steps 3_000_000 --run-name first
    uv run train-walk --actuator xml ...     # the cheap XML servo instead of BAM
    uv run train-walk --robot g1 --envs 16   # the Unitree G1 (fetch-g1 first)

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

from .machine import profile, with_phase_callbacks
from .ppo_hparams import N_STEPS, VF_COEF, configure_torch_cpu, ppo_batch_size
from .robots import registry
from .vec_env import as_sb3_vec_env, make_vec_env

# What a robot can be trained to do. The duck's tasks are reward recipes in
# `behaviors/` (train-behavior); another body has no recipe library, so its
# tasks are env subclasses listed here.
G1_TASKS = ("walk", "stand", "squat", "front_kick", "punch", "imitate")
# Every G1 task that is a HELD POSE rather than locomotion: long
# episodes, and a drive command it is paid to ignore.
HOLD_TASKS = ("stand", "squat", "front_kick", "punch", "imitate")
# MARS does not walk at all: its tasks are ARM tasks (robots/mars_env.py).
#
# Kept as a literal beside G1_TASKS, and not read from the body, on purpose.
# `parse_args` builds the `--task` help before `main` forks its workers, and
# asking a body for its tasks imports the whole `behaviors` package (and
# `walk_env` with it) to print `--help` — while this module's contract is that
# nothing heavy is imported before the fork. The BODY is still the authority
# at runtime (`env_class` -> `MarsBody.env_class` -> `mars_env.TASKS`), and
# `tests/test_mars_env.py` pins this tuple against both `mars_env.TASKS` and
# `behaviors.for_robot("mars")`, so the three cannot drift apart.
MARS_TASKS = ("reach", "pick")
# What a `--robot` value may be besides a registry id. The default and the
# empty string have always meant the duck; the registry itself is strict, so
# these live at the CLI edge instead of loosening `registry.get`.
ROBOT_ALIASES = {None: "microduck", "": "microduck", "duck": "microduck"}


def _body(robot: str | None):
    """The `Body` a `--robot` value names, as a CLI error when it names none.

    `""`/`None`/`"duck"` have always meant the duck here; the registry is
    strict about ids, so the aliases stay at the CLI edge that documented them.
    """
    body_id = ROBOT_ALIASES.get(robot, robot)
    try:
        return registry.get(body_id)
    except KeyError as e:
        # Recomposed rather than passed through: the registry's own text says
        # "unknown robot", and a CLI should name the FLAG. The hint still
        # travels, for the id that is known but not downloaded.
        hint = registry.setup_hint(body_id)
        raise SystemExit(
            f"unknown --robot {robot!r} (have: {', '.join(registry.ids())})"
            + (f"; `{hint}` adds it" if hint else "")) from e


def is_pinned_command(task: str) -> bool:
    """Does a run of `task` produce a policy the lab must NOT drive?

    `viz_server.is_trick_duck` reads `run.json`'s `pinned_command`, and
    without it the lab's demo script asks an idle for 0.9 m/s 27 s out of
    every 30. A function rather than an inline tuple in `main` so that a test
    can ask it: three tasks were added to that tuple over three phases and
    nothing could check any of them without running a trainer.

    `stand` and `imitate` are the G1's held poses — asked for a twist and
    paid to ignore it. `reach` and `pick` are stronger than ignoring:
    `MarsArmEnv` forces the base pair of the action to zero for every task in
    its `ARM_ONLY_TASKS`, so a drive command was never in those policies'
    training distribution at all.
    """
    return task in ("stand", "imitate", "reach", "pick")


def env_class(robot: str, task: str = "walk"):
    """The env class that trains `task` on `robot`.

    One line, because each body now answers for itself
    (`robots/body.Body.env_class`). This was an if-chain per robot, which is
    the pattern `docs/mars-roadmap.md` §1 costs out: a third body would have
    been a third branch here and in five more files.
    """
    return _body(robot).env_class(task)

# Overridable so tests and scratch lab servers write somewhere disposable —
# discover_policies() scans this dir, so stray test runs would otherwise show
# up in the live viewer's palette.
RUNS_DIR = Path(os.environ.get("MICRODUCK_RUNS_DIR")
                or Path(__file__).resolve().parents[2] / "runs")


def _watch_callback_cls(BaseCallback):
    """`progress.jsonl` + `live.onnx`, the contract the lab's TrainingJob tails.

    `train_behavior` has written these since the teach panel existed; without
    them a `train-walk` run is invisible in the viewer — which is exactly how
    the G1's first idle got trained where nobody could watch it. Same schema,
    same atomic replace, dimensions from the robot's own spec.
    """

    class WatchCallback(BaseCallback):
        def __init__(self, out, total_steps: int, snap_steps: int, obs_dim: int,
                     start_steps: int = 0):
            super().__init__()
            self.out = Path(out)
            self.total_steps = int(total_steps)
            self.snap_steps = int(snap_steps)
            self.obs_dim = int(obs_dim)
            # Anchor to the warm-start count, like train_behavior's
            # ProgressCallback. `model.num_timesteps` CONTINUES from the
            # loaded checkpoint when reset_num_timesteps is False, so a bare
            # `next_snap = snap_steps` is already behind on the first rollout
            # of stage 2 and stays behind: every rollout end fires a full
            # onnx export + model.save + venv.save until the counter catches
            # up, and the lab's card reads 700000/500000 from line one.
            self.start_steps = int(start_steps)
            self.next_snap = self.start_steps + self.snap_steps
            self.snapshots = 0
            self.ep_lens: list[int] = []
            self.ep_rews: list[float] = []
            self.term_sums: dict[str, float] = {}
            self.term_eps = 0
            self.t0 = self._prev_t = time.time()
            self._prev_steps = self.start_steps
            self._last_terms: dict[str, float] = {}

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                ep = info.get("episode")
                if ep:
                    self.ep_lens.append(int(ep["l"]))
                    self.ep_rews.append(float(ep["r"]))
                sums = info.get("episode_rewards")
                if sums:
                    for k, v in sums.items():
                        self.term_sums[k] = self.term_sums.get(k, 0.0) + float(v)
                    self.term_eps += 1
            return True

        def _on_rollout_end(self) -> None:
            terms = ({k: round(v / self.term_eps, 3) for k, v in self.term_sums.items()}
                     if self.term_eps else dict(self._last_terms))
            if self.term_eps:
                self._last_terms = terms
            now = time.time()
            dt, ds = now - self._prev_t, int(self.num_timesteps) - self._prev_steps
            self._prev_t, self._prev_steps = now, int(self.num_timesteps)
            line = {
                "steps": int(self.num_timesteps) - self.start_steps,
                "total": self.total_steps,
                "ep_rew": round(sum(self.ep_rews) / len(self.ep_rews), 2) if self.ep_rews else 0.0,
                "ep_len": round(sum(self.ep_lens) / len(self.ep_lens), 1) if self.ep_lens else 0.0,
                "terms": terms,
                "snapshots": self.snapshots,
                "elapsed_s": round(now - self.t0, 1),
                "sps": round(ds / dt) if dt > 0 else 0,
            }
            with open(self.out / "progress.jsonl", "a") as f:
                f.write(json.dumps(line) + "\n")
            self.ep_lens, self.ep_rews = [], []
            self.term_sums, self.term_eps = {}, 0
            if self.num_timesteps >= self.next_snap:
                self._snapshot()
                self.next_snap += self.snap_steps

        def _snapshot(self) -> None:
            """live.onnx + model.zip + vecnormalize.pkl, all atomic — the lab
            loads live.onnx onto the trainee while training continues."""
            import torch

            from .export_onnx import OnnxWalkPolicy
            venv = self.model.get_vec_normalize_env()
            wrapper = OnnxWalkPolicy(self.model.policy, venv.obs_rms.mean,
                                     venv.obs_rms.var, venv.clip_obs).eval()
            tmp = self.out / "live.onnx.tmp"
            dummy = torch.zeros(1, self.obs_dim, dtype=torch.float32)
            torch.onnx.export(wrapper, (dummy,), str(tmp), input_names=["obs"],
                              output_names=["actions"], opset_version=17,
                              dynamo=False)
            tmp.replace(self.out / "live.onnx")
            self.model.save(str(self.out / "model.tmp"))
            os.replace(self.out / "model.tmp", self.out / "model.zip")
            venv.save(str(self.out / "vecnormalize.pkl.tmp"))
            os.replace(self.out / "vecnormalize.pkl.tmp", self.out / "vecnormalize.pkl")
            self.snapshots += 1

    return WatchCallback


def _pass_through_command_dims(venv, spec) -> None:
    """Stop NORMALIZING the twist command: pass it through as it is.

    The command is already a bounded physical quantity (m/s, rad/s) — the
    running normalizer exists for the unbounded ones. Normalizing it makes
    its statistics part of the policy's input contract, and then any stage
    that changes how often a command is nonzero silently redefines every
    other value. Both failures were that:

      * a policy cloned under a PINNED zero command has var 1e-8 there, so
        the first real 0.9 normalizes to ~6400 and clips (ep_len 68 -> 14);
      * "fixing" it with the sampler's true mean/var moved ZERO to -0.2, and
        the policy that had learned to stand at zero fell in 0.4 s.

    mean 0 / var 1 has neither problem, and it is backward compatible: zero
    maps to zero under both, so a policy trained with the old statistics is
    unchanged by the switch.
    """
    start, stop = spec.twist_obs_slice
    venv.obs_rms.mean[start:stop] = 0.0
    venv.obs_rms.var[start:stop] = 1.0
    print(f"twist command passed through un-normalized (obs dims {start}:{stop})")


# Hard cap on the policy's per-dim action log_std (std <= ~0.6), ported from
# train_behavior.py, which has carried it since 2026-09-01. Past ~std 1 the
# clipped Gaussian degenerates into bang-bang sampling that the entropy bonus
# rewards and the DETERMINISTIC export cannot reproduce — and export-walk ships
# the mean. train_behavior's note records stochastic episodes surviving 6.9 s
# against a mean that fell in 0.5 s; measured here on a G1 kick warm-start
# chain, 40 s stochastic against 1.5 s deterministic, with std ratcheted to
# 0.89 mean and 1.57 peak. Warm starts INHERIT log_std, so the cap has to bind
# on load and again every rollout.
LOG_STD_MAX = -0.5


def _log_std_cap_callback_cls(BaseCallback):
    class LogStdCap(BaseCallback):
        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            import torch
            with torch.no_grad():
                self.model.policy.log_std.data.clamp_(max=LOG_STD_MAX)

    return LogStdCap


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


def make_env(rank: int, seed: int, robot: str = "microduck",
             task: str = "walk", **env_kwargs):
    cls = env_class(robot, task)

    def _init():
        return cls(seed=seed + rank, **env_kwargs)
    return _init


# The actuator train-walk trains on when neither --actuator nor
# MICRODUCK_ACTUATOR says otherwise. BAM is the physics the shipped policies
# were optimized against (firmware current limit, real back-EMF, load-
# dependent gearbox friction, bus lag); the XML servo is its small-signal
# linearization and ~30% cheaper per step (README "Actuator model"). Until
# 2026-09-06 this defaulted to xml by omission — the audit's item 8.
DEFAULT_ACTUATOR = "bam"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # 32: the bench-envs knee — see train_behavior.py's --envs for the numbers.
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--head-range", default=None, metavar="nlo,nhi,hlo,hhi,ylo,yhi,rlo,rhi",
                    help="head-pose command ranges (neck, head, yaw, roll; rad) the walker trains under; "
                         "default: the contract's keep-alive +-0.05. The gaze poses: -0.75,0.05,-0.05,0.8,-1.4,1.4,-0.015,0.015")
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--run-name", default=time.strftime("walk-%Y%m%d-%H%M%S"))
    ap.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None,
                    help="warm-start from an existing run dir (fine-tune)")
    ap.add_argument("--no-domain-rand", action="store_true")
    ap.add_argument("--no-obs-noise", action="store_true")
    ap.add_argument("--freeze-obs-norm", action="store_true", default=True,
                    help="warm start: stop updating the observation "
                         "normalizer (default on — see the comment in main)")
    ap.add_argument("--live-obs-norm", dest="freeze_obs_norm",
                    action="store_false",
                    help="the old behaviour: keep adapting the normalizer")
    ap.add_argument("--lr", type=float, default=None,
                    help="PPO learning rate (default 1e-3 from scratch). A "
                         "warm start is POLISHING a policy that already "
                         "works: 1e-3 measured out at destroying a distilled "
                         "G1 idle inside the first updates (held 27-48 s "
                         "before, 0.2-1.0 s after), so --init-from defaults "
                         "to 1e-4 instead.")
    ap.add_argument("--snap-steps", type=int, default=25_000,
                    help="steps between live.onnx snapshots — what the lab's "
                         "teach panel loads onto the 🎓 trainee while the run "
                         "is still going (0 = off)")
    ap.add_argument("--command-mix", type=float, default=0.0,
                    help="g1 --task stand: fraction of episodes that show a "
                         "drive command the idle must ignore (default 0)")
    ap.add_argument("--clip", default=None,
                    help="g1 --task imitate: the clip (clips/<name>.json, "
                         "authored in the 🎬 panel) to track; MICRODUCK_CLIP "
                         "is the lab's channel for the same thing")
    ap.add_argument("--action-mode", default=None,
                    choices=("absolute", "delta", "cubic", "rung"),
                    help="mars: what the six arm actions MEAN. 'delta' "
                         "integrates the commanded target so a=0 holds it; "
                         "'cubic' keeps an absolute target but stretches the "
                         "box's interior; 'absolute' is Phase 4a's linear "
                         "map; 'rung' is 'absolute' in a narrower box "
                         "(mars_env.RUNG_SCALE_RAD — measured unable to reach "
                         "the shell, kept only so its null is reproducible). "
                         "Default: mars_env.DEFAULT_ACTION_MODE.")
    ap.add_argument("--task", default="walk",
                    help="what to train on a non-duck body (g1: "
                         f"{', '.join(G1_TASKS)}; mars: "
                         f"{', '.join(MARS_TASKS)}; a name that body does not "
                         "have lists the ones it does). 'stand' is the IDLE: "
                         "it ignores the drive command and holds its ground, "
                         "the role alpha_stand plays for the duck.")
    ap.add_argument("--robot", default="microduck", choices=registry.ids(),
                    help="which body to train (default microduck). 'g1' needs "
                         "`uv run fetch-robot g1` and trains the "
                         "99-obs/29-action Unitree G1 — a lab contract, not a "
                         "sim2real one. 'mars' needs `uv run fetch-robot mars` "
                         "and trains Innate's MARS ARM (32-obs/8-action, "
                         "25 Hz) — it does not walk, so --task is one of "
                         f"{', '.join(MARS_TASKS)}.")
    ap.add_argument("--actuator", default=None, choices=("xml", "bam"),
                    help=f"servo model (default {DEFAULT_ACTUATOR}; "
                         "MICRODUCK_ACTUATOR overrides the default, an "
                         "explicit flag overrides both)")
    return ap.parse_args(argv)


def env_kwargs_from_args(args: argparse.Namespace) -> dict:
    """The walking-env kwargs a train-walk invocation trains under.

    The knobs every body shares are here; the per-body ones come from the
    body (`Body.train_env_kwargs`), which is where the old `if robot == g1`
    block went — the actuator model it refuses, the episode length its held
    poses need, the clip an imitation task tracks.
    """
    robot = getattr(args, "robot", "microduck")
    kw = dict(
        domain_rand=not args.no_domain_rand,
        obs_noise=not args.no_obs_noise,
    )
    if getattr(args, "head_range", None):
        if robot != "microduck":
            raise SystemExit("--head-range is a Microduck command slot; "
                             f"{robot} has no head-pose command in its obs")
        v = [float(x) for x in args.head_range.split(",")]
        assert len(v) == 8, "--head-range needs 8 numbers"
        kw["head_cmd_ranges"] = tuple((v[i], v[i + 1]) for i in range(0, 8, 2))
    # `--action-mode` is MARS's action map (`robots/mars_env.ACTION_MODES`).
    # Refused for a body that has no such knob rather than dropped, which is
    # `--head-range`'s rule above and AGENTS.md's rule 0: a flag accepted and
    # discarded looks exactly like one that worked, and this one decides what
    # every float the policy emits means.
    if getattr(args, "action_mode", None):
        if robot != "mars":
            raise SystemExit(
                "--action-mode is MARS's arm action map; "
                f"{robot} has one linear joint-target map "
                "(robots/spec.py's action_scale)")
    kw.update(_body(robot).train_env_kwargs(args))
    return kw


def main() -> None:
    args = parse_args()

    out = RUNS_DIR / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    env_kwargs = env_kwargs_from_args(args)
    # Fork workers BEFORE importing torch. A torch-initialized parent has
    # OpenMP/Accelerate thread pools; forking them deadlocks on macOS.
    # One compiled mjModel for the whole fleet: see vec_env.py.
    print(profile().describe())
    venv = make_vec_env([make_env(i, args.seed, robot=args.robot, task=args.task,
                                  **env_kwargs)
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
        if args.freeze_obs_norm:
            # FREEZE the observation statistics of a warm start.
            #
            # VecNormalize keeps updating its running mean/var from whatever
            # the policy is currently doing. A policy cloned from a NARROW
            # distribution (a G1 standing still) starts producing falls the
            # moment PPO explores; those states widen the stats; the same
            # weights then see inputs that mean something else, which makes
            # it fall more. Measured on the G1 idle: the clone holds 27-48 s,
            # and 300-600k steps of fine-tuning at three learning rates left
            # it at 0.2-2.1 s. Frozen stats keep the policy's inputs meaning
            # what they meant when it learned to stand.
            venv.training = False
            # …and the twist command is never normalized at all, so a stage
            # that starts showing one cannot redefine the policy's inputs.
            #
            # A WALKER's command, that is. `twist_obs_slice` is a `RobotSpec`
            # field — the walking env's half of the contract — and MARS is a
            # `Body` without one: its 32 floats carry a MEASURED base twist,
            # not a commanded one, and there is nothing in them to hold at
            # mean 0 / var 1. Asked by capability rather than by id, per
            # AGENTS.md's rule about guarding on `kind`/`hasattr` and never on
            # a robot name.
            from .robots import spec as _spec_mod
            body = _spec_mod.get(args.robot)
            if hasattr(body, "twist_obs_slice"):
                _pass_through_command_dims(venv, body)
        model = PPO.load(str(prev / "model"), env=venv, device=args.device,
                         custom_objects={"policy_class": FastActorCriticPolicy})
        # Warm starts INHERIT log_std, so a chain ratchets it up run after run
        # until the exported mean no longer carries the skill. Bind the cap on
        # load, not only per rollout.
        import torch as _torch
        with _torch.no_grad():
            model.policy.log_std.data.clamp_(max=LOG_STD_MAX)
        model.batch_size = batch
        # A warm start polishes; it does not explore from nothing. The flat
        # 1e-3 that trains a policy from scratch wrecks a cloned one.
        lr = args.lr if args.lr is not None else 1e-4
        model.learning_rate = lr
        model.lr_schedule = lambda _progress: lr
        print(f"warm-started from {prev} (lr {lr:g})")
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
            learning_rate=args.lr if args.lr is not None else 1e-3,
            gamma=0.99, gae_lambda=0.95,
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
    run_meta = {
        "run_name": args.run_name, "envs": args.envs, "steps": args.steps,
        "robot": args.robot,
        # WHAT THIS RUN'S POLICIES SPEAK (robots/policy_contract.py). "robot"
        # stays beside it, untouched, because every reader in the tree and
        # every run already on disk is keyed on the name — the contract is an
        # addition, not a migration. This is the record `export-walk` checks
        # its own export against, and the one that answers for a checkpoint
        # ONNX that was never stamped.
        "contract": _body(args.robot).contract().as_dict(),
        "task": args.task,
        "pinned_command": is_pinned_command(args.task),
        "seed": args.seed, "env_kwargs": env_kwargs,
        "microduck_rl_sha": git_sha, "init_from": args.init_from,
    }
    (out / "run.json").write_text(json.dumps(run_meta, indent=2))
    # ...and the record a PERSON reads (run_record.py). Written before the
    # run starts, not after, so a run that is stopped or crashes still says
    # what it was trying to do — an unlabelled half-run in the palette is
    # the exact problem this file exists to remove.
    from . import run_record
    behavior_title = os.environ.get("MICRODUCK_RUN_TITLE") or None
    stage_env = {k: v for k, v in os.environ.items()
                 if k.startswith("MICRODUCK_G1_") or k == "MICRODUCK_CLIP"}
    run_record.write_record(out, **run_record.default_record(
        args.run_name, run_meta, behavior_title=behavior_title,
        stage_env=stage_env))

    checkpoints = CheckpointCallback(
        save_freq=max(500_000 // args.envs, 1), save_path=str(out / "checkpoints"),
        name_prefix="model", save_vecnormalize=True,
    )
    # On Linux/cloud `with_phase_callbacks` appends the profile's thread
    # policy — the rollout gets one torch thread, the update every core
    # (machine.py). On a Mac the list comes back unchanged. Throughput only:
    # no PPO math moves.
    from .robots import spec as _spec
    callbacks = [checkpoints, _penalty_sign_callback_cls(BaseCallback)(),
                 _log_std_cap_callback_cls(BaseCallback)()]
    if args.snap_steps > 0:
        # Makes the run WATCHABLE: the lab tails progress.jsonl for the card
        # and loads live.onnx onto the 🎓 trainee as it improves.
        callbacks.append(_watch_callback_cls(BaseCallback)(
            out, args.steps, args.snap_steps,
            _spec.get(args.robot).obs_dim,
            # SB3 keeps the loaded count when reset_num_timesteps is False.
            start_steps=int(model.num_timesteps) if args.init_from else 0))
    model.learn(
        total_timesteps=args.steps,
        callback=with_phase_callbacks(callbacks, BaseCallback),
        progress_bar=False,
        reset_num_timesteps=args.init_from is None,
    )

    model.save(str(out / "model"))
    venv.save(str(out / "vecnormalize.pkl"))
    # A TERMINAL RECORD, written last and only on a clean finish. Without it
    # "has this run finished?" had no answer in the artifact, and the only way
    # to ask was to grep the process table — which is how a `pgrep -f
    # "microduck_local.train "` ended up matching the very shell that was
    # doing the grepping, and a wait loop that could never end. train_behavior
    # has written this line since the teach panel existed; train.py did not.
    if args.snap_steps > 0:
        with open(out / "progress.jsonl", "a") as f:
            f.write(json.dumps({"steps": int(model.num_timesteps),
                                "total": int(model.num_timesteps),
                                "done": True}) + "\n")
    print(f"saved {out}/model.zip + vecnormalize.pkl")
    print(f"export with: uv run export-walk {out}")


if __name__ == "__main__":
    main()
