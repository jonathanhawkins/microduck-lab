"""Duck lab: run many policies side by side and stream poses to the web viewer.

    uv run duck-lab --checkpoints runs/first-gait ../microduck/policies/alpha_walking.onnx

Each argument becomes one live duck: a .onnx file, a run dir (uses policy.onnx,
exporting it on the fly if the run finished without one), or with --checkpoints
a run's checkpoints/*.zip lined up oldest→newest.

HTTP (default 127.0.0.1:8788):
  GET  /scene       visual geometry pulled from the compiled MuJoCo model
                    (the jenga-stacker extract_visual_scene trick)
  GET  /policies    everything assignable: shipped Pollen policies, local runs,
                    checkpoints — for the viewer's drag-and-drop palette
  DELETE /runs/{name}  permanently delete a training run's directory (policy,
                    checkpoints, progress log). `?chain=true` treats {name} as
                    a curriculum-chain prefix and deletes every stage of it in
                    one go. Refused (409) for any run of the job that is
                    training right now. Shipped Pollen policies are not
                    deletable — they are not ours to delete.
  GET  /behaviors   the teachable-behavior library (cards for the teach panel)
  POST /teach       {"text": "stand on one leg"} → match a behavior and start a
                    local training run (subprocess); progress streams in frames.
                    A behavior with a curriculum trains as a CHAIN of stage
                    runs (…-s1, -s2, …), each fine-tuning from the previous
                    under its own env knobs — orchestrated here, narrated in
                    the frames. Optional: "weights": {termKey: float}
                    reward-weight overrides (sliders), "stageWeights":
                    {"<1-based stage>": {termKey: float}} per-stage overrides
                    layered over weights (stage wins per key), "startStage":
                    N (1-based) to begin the chain at stage N — earlier
                    stages are skipped and stage N warm-starts from the
                    newest existing …-s{N-1} run (refused with a message when
                    none exists), "initFrom": "<run name>" to fine-tune an
                    existing run's policy under the (possibly edited) recipe
                    — that stays a SINGLE run, using the final stage's env
                    knobs, "steps": N the TOTAL practice budget for the whole
                    job (the panel's "how long should it practice?" control) —
                    a staged chain splits it across its stages in PROPORTION
                    to their declared steps (split_step_budget), so the
                    curriculum keeps its shape instead of the number silently
                    meaning "per stage", "stageSteps": {"<1-based stage>": N}
                    explicit per-stage budgets laid over that split. Both are
                    sticky per behavior, like the weights.
  POST /teach/load  {"policy": "run:<name>"} seat a FINISHED run in the teach
                    panel without training anything: its recipe card/sliders
                    stream in "done" state so ✨ fine-tune targets that run.
                    Accepts palette ids ("run:…", "ckpt:…@Nk") or bare run
                    names; refused while a job is actively training.
  POST /teach/weights  {"stageWeights": {...}} live edit on the active chain:
                    future stages record; a changed ACTIVE stage warm-restarts
  POST /teach/stop  stop the active training run/chain (final policy still saved)

Keyframe animation authoring (the viewer's 🎬 animate panel — pose the robot,
key the poses, save a clip an imitation-RL reward can track):
  GET  /joints      per-joint editing metadata: JOINT_NAMES order, MJCF limits
                    (model.jnt_range), DEFAULT_POSE, and the body/axis/anchor
                    each joint drives — so the editor clamps and picks joints
                    by clicking the 3D duck without hardcoding the model
  POST /pose        {"joints": [14 rad], "rootPitch": rad?, "ground": bool?} →
                    {"bodies": [[x,y,z,qw,qx,qy,qz], …], "joints", "rootPitch"}
                    forward kinematics of an ARBITRARY authored pose, computed
                    on a dedicated scratch model (pose_scratch()) so previewing
                    never disturbs a live duck's episode. Body order matches
                    GET /scene. Joints are clamped to the MJCF limits and the
                    clamped values come back. rootPitch is the right-handed
                    rotation about the trunk's +Y axis: NEGATIVE = lean back
                    (projected gravity acquires -x in the trunk frame),
                    positive = nose-down. ~0.1 ms per call.
  GET  /clips       [{name, duration, loop, keys, modified}] newest first
  GET  /clips/{n}   one clip
  PUT  /clips/{n}   save (validates the clip contract, clamps joints to limits)
  DELETE /clips/{n} remove
Screen captures (the viewer's 🎥 record button — the browser records its
canvas with MediaRecorder and this server makes shareable files of the take):
  POST /captures?name=<duck>  raw video body (any MediaRecorder container) →
                    {name, mp4, gif, mp4Kb, gifKb, dir}: converts with
                    imageio-ffmpeg's bundled binary to captures/<slug>-<ts>.mp4
                    (h264, full resolution) + .gif (480 px palette gif), beside
                    runs/ (MICRODUCK_CAPTURES_DIR relocates it)
  GET  /captures/{file}       download one capture (Content-Disposition set)

  Clips are JSON files in clips/ beside runs/ (MICRODUCK_CLIPS_DIR relocates
  it). Format v1: {version, name, duration, loop, keys: [{t, joints[14],
  rootPitch}]} — t seconds ascending from 0, joints ABSOLUTE radians in
  JOINT_NAMES order, linear interpolation between keys.

WS /ws — ~25 Hz frames:
  {cmd, mode, stats, events,
   ducks: [{id, name, falls, step, rew, speed, cmdSpeed, bodies}],
   training: {runName, status, behavior, progress, weights, stageWeights,
              envs, helpers, restarting,
              stage: {idx, count, label, detail, start} | null} | null}
  progress carries the ACTIVE stage's fields verbatim plus overallSteps /
  overallTotal, cumulative across the stage chain (== steps/total when the
  job is a single run), and overallElapsed: wall-clock seconds since the job
  launched (spans stage handoffs and warm restarts, unlike the per-subprocess
  elapsed_s; frozen at finish, null for adopted runs)
  stats: {cpu, mem: machine-wide %, lab/trainer: {cpu, memMb} per process
          (trainer sums its SubprocVecEnv workers; null when not training),
          trainFps: training steps/s from progress.jsonl | null}
accepts:
  {"cmd": [vx, vy, wz]}                       shared drive command (held 6 s)
  {"reset": true}                             reset every duck
  {"assign": {"duck": "d2", "policy": "pollen:alpha_stand"}}   hot-swap a brain
                           optional "showcase": true (the palette's chain-level
                           "whole trick" chip): rebuild the duck's env as the
                           policy's behavior env under the FINAL curriculum
                           stage's spawn knobs, so spawns rehearse the whole
                           trick arc instead of only a standing start — a
                           no-op for policies without a curriculum behind them
  {"spawn_helper": true}   add a helper duck: another viewer of the same
                           live.onnx snapshot. Helpers do NOT add trainer
                           workers — measured live-lab, 16 envs ran at
                           10.0k steps/s and 26 envs (5 helpers × +2) at
                           6.8k, because the extra processes fight the
                           lab's own sim loop.
  {"remove_duck": {"duck": "d3"}}   remove ANY duck (declutter the roster);
                           the trainee is only removable when no run is active
  {"spawn_duck": {"policy": "pollen:alpha_stand"}}   add a fresh duck running
                           that palette policy (cap 20 ducks); accepts the
                           same optional "showcase" flag as assign

The roster persists to lab-state.json next to runs/ (override the path with
the LAB_STATE_PATH env var) and is restored on startup, at which point the CLI
duck args are ignored — pass --fresh to delete the state file and reseed from
the CLI. Training jobs are NOT resumed across restarts (the subprocess dies
with the server): a restored trainee/helper simply keeps its last live.onnx
snapshot brain, frozen until the next /teach.

Testing knobs (env vars): TEACH_STEPS_OVERRIDE / TEACH_SNAP_OVERRIDE shrink new
jobs' total steps / snapshot interval; MICRODUCK_RUNS_DIR relocates runs/.

The frontend lives in ../duck-viewer (Next.js + react-three-fiber).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from collections import deque
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path

import numpy as np
import psutil

# Top-level on purpose: this module uses `from __future__ import annotations`,
# so FastAPI resolves handler type hints against MODULE globals — a
# function-local `WebSocket` import makes the ws param unresolvable and every
# connection is denied with HTTP 403 (cost an hour; leave these here).
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Imported AS A MODULE so /teach can hot-reload it (behaviors.reload_library /
# motion.reload_self — all-or-nothing, never bare importlib.reload): recipe edits in
# behaviors.py are picked up by the training SUBPROCESS (fresh import) but
# were invisible to this long-running server — the teach panel then showed a
# stale scorecard missing new terms (bit the user twice: head_up, head_up_pull).
from . import behaviors as behaviors_mod
from . import contract as C
from . import motion as motion_mod
from .train import RUNS_DIR
from .walk_env import MicroduckWalkEnv, shared_model_scope

TICK_HZ = 50            # env control rate (real time)
SEND_EVERY = 2          # broadcast at 25 Hz
# Forward-speed readout: mean of the last SPEED_WINDOW control steps.
# 25 @ 50 Hz = 0.5 s, about one stride. Instantaneous forward speed on a
# stepping gait swings by ~100% WITHIN a stride (it peaks at push-off and
# dips through swing), so the raw number is unreadable as text; half a
# second averages the stride out while still following a policy that
# speeds up, stalls or falls over inside a second.
SPEED_WINDOW = 25
EPISODE_RESET_S = 30.0  # periodic reset so wandering ducks regroup
OVERRIDE_HOLD_S = 6.0
POLICIES_DIR = Path(__file__).resolve().parents[3] / "microduck" / "policies"
# Authored keyframe clips (the 🎬 animate panel), beside runs/ — same
# overridable-path convention as RUNS_DIR/LAB_STATE_PATH so tests and scratch
# servers never write into the real workspace.
CLIPS_DIR = Path(os.environ.get("MICRODUCK_CLIPS_DIR")
                 or RUNS_DIR.parent / "clips")
# 🎥 viewer screen captures (the record button), beside runs/ — the browser
# uploads whatever container MediaRecorder produced and this server converts
# it to a shareable mp4 + gif with imageio-ffmpeg's bundled binary.
CAPTURES_DIR = Path(os.environ.get("MICRODUCK_CAPTURES_DIR")
                    or RUNS_DIR.parent / "captures")
# 🤗 BYOK Hugging Face token (the viewer's ⚙ settings), beside runs/ — same
# overridable-path convention. Holds {"token", "username"}; written 0600 and
# gitignored, validated against whoami() before it is ever saved, and NEVER
# returned to the browser (only a mask + the username go back over the wire).
# This is the doorway to the real-GPU step: the Jobs API trains microduck_rl
# on HF hardware under the user's own account.
HF_TOKEN_PATH = Path(os.environ.get("MICRODUCK_HF_TOKEN_PATH")
                     or RUNS_DIR.parent / "hf-token.json")


def _hf_mask(token: str) -> str:
    return f"{token[:7]}…{token[-4:]}" if len(token) > 14 else "•••"


def load_hf_token() -> dict | None:
    """{"token", "username"} or None. Corrupt files read as absent."""
    try:
        d = json.loads(HF_TOKEN_PATH.read_text())
        return d if isinstance(d, dict) and d.get("token") else None
    except (OSError, ValueError):
        return None

# Trainer env count is FIXED at BASE_ENVS for lab-launched jobs.
#
# 32, not 16. The old 16 came from a live-lab test that seemed to invert the
# idle-machine curve (teach-run-be11cc, "26 envs", held 6.8k steps/s vs 10.0k
# at 16) — but that test was CONFOUNDED: its 26 trainer envs arrived as 5
# helper ducks × 2, so it also carried five extra 50 Hz viewer sims in this
# server. Helpers no longer resize the trainer, and an A/B with helpers-as-
# viewers only (2026-08-30, lab + browser + a competing 16-env trainer all
# live) re-agreed with the idle curve: 16 → 4.7k, 24 → 5.4k (+15%), 32 → 6.5k
# (+37%). Idle the same day: 16 → 14.3k, 24 → 15.6k, 32 → 16.5k, ~17.1k
# asymptote. Profiling says why more-than-cores wins: the parent's serial
# per-vec-step work (policy forward + 32 pipe messages) and the serial PPO
# update leave the workers ~11% busy at 16 envs — extra envs amortize the
# parent, they don't fight it. The serial-update growth the old note feared
# is handled by ppo_hparams.ppo_batch_size (minibatch grows, update stays 4
# × n_epochs optimizer steps).
#
# Helpers stay as extra VIEWERS of live.onnx; they do not resize the trainer.
# ENVS_PER_HELPER is kept at 0 so TrainingJob.scale() / payload arithmetic
# cannot quietly grow --envs if a helper spawn ever calls it again.
BASE_ENVS = 32          # train_behavior's own --envs default
ENVS_PER_HELPER = 0
RECOMMENDED_ENVS = BASE_ENVS
MAX_HELPERS = int(os.environ.get("DUCK_MAX_HELPERS", "6"))
RECOMMENDED_HELPERS = MAX_HELPERS

# Bounds on the user-chosen practice budget (the teach panel clamps to the
# same range). The floor is roughly "long enough to produce a snapshot worth
# watching"; the ceiling stops a typo — 40 instead of 4 in the millions
# field — from turning into an overnight run on someone's laptop.
MIN_STEP_BUDGET = 100_000
MAX_STEP_BUDGET = 40_000_000


def split_step_budget(declared: list[int], total: int) -> list[int]:
    """Scale a curriculum's declared per-stage budgets to a chosen TOTAL,
    keeping the stages' RATIOS (backflip's 1/2/1/1.5/1.5 stays 1/2/1/1.5/1.5).

    Reading a user's number as "this many steps per stage" instead would
    silently multiply the real cost by the stage count — exactly the surprise
    the budget control exists to remove. Largest-remainder rounding, so the
    parts sum to EXACTLY `total`: a chain whose stages don't add up to the
    number on screen is that same misreport in miniature. Every stage keeps
    at least one step, so scaling right down can't skip one entirely.
    """
    n = len(declared)
    if n == 0:
        return []
    total = max(int(total), n)
    base = sum(max(0, int(d)) for d in declared)
    exact = ([max(0, int(d)) * total / base for d in declared] if base > 0
             else [total / n] * n)
    out = [max(1, int(x)) for x in exact]
    short = total - sum(out)
    # Remainder goes to the largest fractional parts first (Hamilton).
    order = sorted(range(n), key=lambda i: exact[i] - int(exact[i]),
                   reverse=True)
    for i in range(max(short, 0)):
        out[order[i % n]] += 1
    while short < 0:  # the min-1 clamp overshot — take it back off the top
        j = max(range(n), key=lambda i: out[i])
        if out[j] <= 1:
            break
        out[j] -= 1
        short += 1
    return out


# Auto demo script: (seconds, [vx, vy, wz]) — loops.
DEMO_SCRIPT = [
    # Runway show. The old script demoed every command bucket (turn, sidestep,
    # stop, reverse) — correct behavior that read as "something screwy going
    # on when it restarts": ducks turning aside, stopping, then all sprinting
    # in unison. Now it's mostly the thing being trained: a long straight
    # sprint, with a short walk-up and cooldown. One 30 s episode = one pass.
    # Straight into the sprint — the walk-up was cosmetic staging, not a
    # requirement; the policy handles a standing start at full command
    # (every eval does exactly that, 0/10 falls).
    (27.0, [0.9, 0.0, 0.0]),   # sprint from step one
    (3.0, [0.0, 0.0, 0.0]),    # brief stand so the reset reads as a reset
]


def _zero_infer(obs: np.ndarray) -> np.ndarray:
    return np.zeros(14, dtype=np.float32)


class Duck:
    """One env + one policy (ONNX session or in-process SB3 checkpoint)."""

    def __init__(self, duck_id: str, label: str, infer, seed: int,
                 policy_id: str | None = None, onnx_path: str | None = None,
                 env_kwargs: dict | None = None):
        self.id = duck_id
        self.label = label
        self.infer = infer  # (obs[61]) -> action[14]
        self.env_kwargs = dict(env_kwargs or {})
        # Brain provenance, for lab-state.json: a palette id, an .onnx path,
        # or neither (a zero-infer trainee before its first snapshot).
        self.policy_id = policy_id
        self.onnx_path = onnx_path
        # True while this duck runs a chain-level "whole trick" assign — its
        # env rehearses full-arc spawns (see showcase_env_kwargs). Persisted
        # so a restart doesn't silently demote the duck to standing starts
        # while its ✨ label still promises the whole trick.
        self.showcase = False
        # Policy HANDOFF (the robot's real pattern: tricks hot-swap back to a
        # standing/walking brain when they finish). When set, this duck runs
        # `infer` until the trick completes and both feet are down, then runs
        # `handoff_infer` — measured 11/12 stands, 8.1 s holds, where the
        # trick policy alone managed none.
        self.handoff_infer = None
        self.handoff_label = None
        self.handed = False
        # Kept: rebuild_env re-seeds the env RNG, and every duck must keep
        # drawing its own spawn stream (helpers exist to be independent).
        self.seed = seed
        self.env = self._make_env(seed)
        self.obs, _ = self.env.reset(seed=seed)
        self._hold_yaw = None   # heading-hold anchor (see set_cmd)
        self._settle = 0        # ticks since handoff (see _recenter_wz)
        self.falls = 0
        self.reward_ema = 0.0
        # Rolling window of heading-frame forward speeds (see sample_speed).
        self.speed_hist: deque[float] = deque(maxlen=SPEED_WINDOW)

    def _make_env(self, seed: int, kwargs: dict | None = None):
        """A `behavior_id` in env_kwargs asks for the behavior's OWN env class
        — the walk env is only for policies that actually walk. Every policy
        with a behavior behind it needs this: the walk env resamples a random
        locomotion twist into the observation, which a trick policy trained on
        pinned-zero twist reads as a walk order (measured: an assigned one_leg
        policy fell 106x per 1500 steps there, 0x in its own env).

        `spawn_overrides` then carries the active curriculum stage's knobs per
        instance — that's the trainee preview, where spawn families matter: the
        walk env only ever spawns STANDING, so during "learning to land" the
        user watched stand-then-topple while the real trainer practiced
        mid-roll drops invisibly. `standing_spawns` asks for the opposite
        (keyframe starts under the behavior's own physics) — see
        env_kwargs_for_behavior."""
        # `kwargs` lets a caller build an env BEFORE committing it to
        # self.env_kwargs (see rebuild_env), so a failed build can't poison
        # the memo the rebuild guard compares against.
        kw = dict(self.env_kwargs if kwargs is None else kwargs)
        behavior_id = kw.pop("behavior_id", None)
        standing = kw.pop("standing_spawns", False)  # BehaviorEnv-only knob
        # 30 s episodes, matching EPISODE_RESET_S: the env default of 10 s
        # truncated preview ducks mid-sprint (the user watched a 1.0 m/s run
        # get cut off by the reset).
        common = dict(obs_noise=False, domain_rand=False, action_delay=False,
                      random_yaw=False, seed=seed)
        # DEFAULT only — a behavior/stage env that declares its own episode
        # length (kw) wins; the env's 10 s default truncated preview ducks
        # mid-sprint otherwise.
        kw.setdefault("max_episode_s", 30.0)
        # One compiled mjModel per (scene, actuator) for the whole roster. A
        # model costs ~470 MB as a process's first compile and ~90-140 MB per
        # extra copy, against the ~0.9 MB of mjData that is all a duck actually
        # owns — a 6-duck lab was paying ~1.4 GB to simulate ~5 MB of state.
        # Safe in particular because `common` pins domain_rand=False, so no env
        # here ever writes to the model, and the frame loop steps ducks
        # serially. BAM is the exception: it rewrites dof_frictionloss every
        # physics substep, so a lab launched with MICRODUCK_ACTUATOR=bam
        # (resolved exactly as the env resolves it) keeps private models.
        # actuator_force FIRST — it is what walk_env resolves as the winner, so
        # this mirror has to agree with it or the two disagree in the worst
        # possible way: a stage declaring bam on a lab whose process env does
        # not (uv run duck-lab, no MICRODUCK_ACTUATOR) computed "xml" here,
        # entered the SHARED-model scope, and handed every duck one mjModel
        # that the BAM actuator then rewrites every substep — walk_env refuses it
        # outright, and the raise lands inside the 50 Hz loop.
        actuator = (
            kw.get("actuator_force")
            or os.environ.get("MICRODUCK_ACTUATOR", kw.get("actuator", "xml"))
        ).strip().lower()
        scope = (nullcontext() if actuator == "bam"
                 else shared_model_scope(exclusive=False))
        with scope:
            if behavior_id:
                return behaviors_mod.BehaviorEnv(
                    behavior_id, standing_spawns=standing, **common, **kw)
            return MicroduckWalkEnv(**common, **kw)

    def set_cmd(self, cmd: np.ndarray) -> None:
        tw = np.asarray(cmd, np.float32).copy()
        # Deployment heading-hold, same 3 lines the robot runtime would run:
        # the policy is compass-blind (61 obs carry no yaw), so an unsteered
        # duck MUST drift into circles — which is exactly what the user was
        # shown while the videos (steered) ran straight. The viewer shows
        # DEPLOYED behavior now: driving forward with no explicit turn command
        # closes the loop on measured yaw. An explicit turn command wins.
        if tw[0] > 0.05 and abs(float(tw[2])) < 1e-6:
            q = self.env.data.qpos[3:7]
            yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                   1 - 2 * (q[2] ** 2 + q[3] ** 2)))
            # Hold the heading the duck HAS when the straight segment begins —
            # not world zero. Holding to zero made every post-turn straight
            # segment command a saturated spin-back while also asking full
            # forward speed: the duck whipped around and fell ~3 s in, and
            # softer versions of the same fight read as "veers off slow".
            if self._hold_yaw is None:
                self._hold_yaw = yaw
            err = yaw - self._hold_yaw
            err = float(np.arctan2(np.sin(err), np.cos(err)))
            tw[2] = float(np.clip(-4.0 * err, -1.0, 1.0))
        else:
            self._hold_yaw = None   # turns/sidesteps release the hold
        self.env.twist_cmd[:] = tw
        self.env.head_cmd[:] = 0.0
        self.env.body_cmd[:] = 0.0

    def rebuild_env(self, env_kwargs: dict) -> None:
        """Recreate the sim with different physics (scene/termination) — needed
        when a duck takes on a policy trained on the full-collision scene
        (headstand): in the walk scene its head would clip through the floor
        and fall-termination would reset it every couple of seconds."""
        if env_kwargs == self.env_kwargs:
            return
        # The duck's OWN seed, not a constant: _make_env and env.reset both
        # re-seed the env RNG that picks spawn families and pose noise, so
        # rebuilding every duck with one seed (a stage handoff rebuilds the
        # trainee AND every helper) made the helpers draw identical starts
        # episode after episode — they stop being independent samples, which
        # is the whole point of having them.
        seed = self.seed
        # Build FIRST, commit after. Committing env_kwargs up front meant a
        # construction failure left the duck describing an env it does not
        # have — and because the guard above compares against that memo, every
        # retry with the same kwargs then returned instantly without rebuilding.
        want = dict(env_kwargs)
        env = self._make_env(seed, kwargs=want)
        self.env_kwargs = want
        self.env = env
        self.obs, _ = self.env.reset(seed=seed)
        # Per-episode heading state belongs to the env that just died: the
        # hold anchor is a yaw in the OLD sim's frame, and carrying it into a
        # fresh one commands a saturated turn until the next episode reset.
        self._hold_yaw = None
        self._settle = 0
        # Same reasoning as reset(): speed samples from the dead sim would be
        # averaged into the HUD for the next half second, and a duck that had
        # already handed off would run the stand policy in the fresh episode
        # instead of the trick it was rebuilt to perform.
        self.speed_hist.clear()
        self.handed = False

    def swap_policy(self, label: str, infer, policy_id: str | None = None,
                    onnx_path: str | None = None) -> None:
        self.label = label
        self.infer = infer
        self.policy_id = policy_id
        self.onnx_path = onnx_path
        self.falls = 0
        self.reward_ema = 0.0
        self.speed_hist.clear()  # the old brain's speed is not this one's
        # Handoff state belongs to the OUTGOING brain. Left standing, a duck
        # that had already handed off kept handed=True while do_assign cleared
        # handoff_infer for the incoming plain policy — the next tick called
        # None(obs) and the TypeError killed the whole 50 Hz loop.
        self.handed = False
        self.handoff_infer = None
        self.handoff_label = None
        self._settle = 0

    def _handoff_due(self) -> bool:
        """The trick is finished and the duck is on both feet."""
        env = self.env
        rot = getattr(env, "_bf_rot", None)
        if rot is None or rot < 5.2:
            return False
        c = getattr(env, "foot_contact_state", {})
        if not (c.get("left") and c.get("right")):
            return False
        # ...and the ROTATION IS BRAKED. Handing off while still spinning gave
        # alpha_stand (heading-indifferent by design — yaw is unobservable)
        # the leftover angular momentum, which it absorbed by pivoting: the
        # "it lands then turns to the side" the user kept seeing. The trick
        # policy carries the braking incentives (stick_it, calm_landed), so
        # it keeps the wheel until the spin is actually killed.
        w = env.data.sensordata[env.gyro_adr]
        return float(w[0] ** 2 + w[1] ** 2 + w[2] ** 2) < 2.0

    _walker_infer = None   # class-level lazy alpha_walking for recentering
    _walker_missing = False  # upstream policies/ absent: try once, then skip

    def _recenter_wz(self) -> float | None:
        """After the trick settles, drive yaw back to the spawn heading.

        The policy cannot learn this (yaw is unobservable — the whole
        heading saga), so the COMMANDER owns it, exactly like the run's
        heading-hold: hand the settled duck to alpha_walking with a turn
        command until it faces its spawn vector again, then hand back to the
        stand. Returns the wz command while recentring, else None.
        """
        if not self.handed:
            self._settle = 0
            return None
        q = self.env.data.qpos[3:7]
        yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                               1 - 2 * (q[2] ** 2 + q[3] ** 2)))
        # Read home_yaw LIVE, deliberately. It is not a spawn anchor —
        # BehaviorEnv re-anchors it to the current heading every ~5 s — so this
        # correction only fires between resamples and self-heals afterwards.
        # Latching it at handoff was measurably worse: a mid-flip spawn (~85%
        # of a showcase mix) reports home_yaw ≈ ±180° because the ZYX yaw
        # degenerates at trick pitch, and freezing that spun the duck ~140°
        # on every landing. A real fix needs an upright-measured spawn anchor
        # the env does not currently keep.
        err = yaw - float(getattr(self.env, "home_yaw", 0.0) or 0.0)
        err = float(np.arctan2(np.sin(err), np.cos(err)))
        self._settle = getattr(self, "_settle", 0) + 1
        if self._settle < 50:              # let the landing settle ~1 s first
            return None
        # Wider deadband + a timeout: at 8 deg the walker chased tiny errors
        # and read as "shuffling toward the old vector" (alpha_walking's
        # turn-in-place drifts forward slightly). 20 deg only triggers on
        # genuinely crooked landings, and after ~3 s it stands wherever it is
        # rather than pacing forever.
        if abs(err) < 0.35 or self._settle > 200:
            return None
        return float(np.clip(-2.0 * err, -0.8, 0.8))

    def tick(self) -> None:
        if self.handoff_infer is not None and not self.handed and self._handoff_due():
            self.handed = True

        # handed without a brain to hand off to is the one combination that
        # calls None(obs) and kills the loop; make it unrepresentable here
        # rather than relying on every writer of handoff_infer to pair them.
        if self.handed and self.handoff_infer is None:
            self.handed = False
        wz = self._recenter_wz() if self.handoff_infer is not None else None
        if wz is not None:
            if Duck._walker_infer is None and not Duck._walker_missing:
                # POLICIES_DIR, not a cwd-relative path: this load happens
                # deep inside the duck loop, where an exception kills the
                # whole loop (every duck freezes while HTTP/WS stay green),
                # and it would fire for any lab started outside
                # microduck_local/.
                try:
                    Duck._walker_infer = _onnx_infer(
                        POLICIES_DIR / "alpha_walking.onnx")
                except Exception:
                    # The upstream microduck/ clone is optional (the roster
                    # loader already checks POLICIES_DIR.exists()), and this
                    # runs inside lab_loop — an unguarded raise here stops the
                    # sim for every duck while HTTP/WS stay green. Recentring
                    # is a nicety; losing the whole loop is not.
                    Duck._walker_missing = True
        if wz is not None and Duck._walker_infer is not None:
            self.env.twist_cmd[:] = (0.0, 0.0, wz)
            action = Duck._walker_infer(self.obs)
        else:
            action = (self.handoff_infer if self.handed else self.infer)(self.obs)
        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        self.reward_ema = 0.98 * self.reward_ema + 0.02 * float(reward)
        self.sample_speed()
        if terminated:
            self.falls += 1
            self.reset()
        elif truncated:
            self.reset()

    def reset(self) -> None:
        cmd = self.env.twist_cmd.copy()
        self._hold_yaw = None   # new episode, new heading anchor
        self.handed = False   # each episode starts on the trick's own brain
        # Speed is reported PER EPISODE: carrying the last half second of a
        # run that ended in a faceplant into the fresh episode would show a
        # duck standing still at 0.3 m/s.
        self.speed_hist.clear()
        self.obs, _ = self.env.reset()
        self.set_cmd(cmd)  # resets resample commands; keep the shared one

    def sample_speed(self) -> None:
        """Record this step's forward speed, in the HEADING frame.

        ``behaviors._base_vel`` (→ ``MicroduckWalkEnv.heading_lin_vel``) is the
        only correct source, and this must never be swapped for the obvious
        ``mj_objectVelocity(..., flg_local=1)``: that returns the trunk's
        INERTIAL (principal-axis) frame, so its "forward" component is
        actually sideways, and a body-axis projection also pays for DIVING —
        a duck falling nose-down scores metres per second while covering no
        ground. That trap already cost this project a reward term that
        rewarded a side shuffle for hours; test_lab pins the frame.

        Reached through the module (not a `from … import`) so the hot
        reload in POST /teach keeps this binding live.
        """
        self.speed_hist.append(float(behaviors_mod._base_vel(self.env)[0]))

    def forward_speed(self) -> float | None:
        """Smoothed forward speed in m/s, or None when the window is empty
        (the single tick right after a reset) — the UI shows "—" rather than
        inventing a zero."""
        if not self.speed_hist:
            return None
        mean = sum(self.speed_hist) / len(self.speed_hist)
        # A blown-up sim puts NaN/inf in qvel, and json.dumps would then emit a
        # bare `NaN` into the frame — invalid JSON, so the BROWSER's JSON.parse
        # throws and the viewer loses the whole frame, not just this cell.
        return round(mean, 3) if math.isfinite(mean) else None

    def pose_payload(self) -> list[list[float]]:
        d, m = self.env.data, self.env.model
        out = []
        for b in range(m.nbody):
            p, q = d.xpos[b], d.xquat[b]
            out.append([round(float(v), 4) for v in (*p, *q)])
        return out


# ------------------------------------------------------------ policy loading

def _onnx_infer(path: Path):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path))
    in_name = sess.get_inputs()[0].name

    def infer(obs: np.ndarray) -> np.ndarray:
        return sess.run(None, {in_name: obs[None]})[0][0].astype(np.float32)
    return infer


def _checkpoint_infer(zip_path: Path, vecnorm_path: Path):
    import pickle

    import torch
    from stable_baselines3 import PPO

    from .export_onnx import OnnxWalkPolicy

    model = PPO.load(str(zip_path), device="cpu")
    with open(vecnorm_path, "rb") as f:
        vn = pickle.load(f)
    wrapper = OnnxWalkPolicy(model.policy, vn.obs_rms.mean, vn.obs_rms.var, vn.clip_obs).eval()

    def infer(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return wrapper(torch.tensor(obs[None])).numpy()[0].astype(np.float32)
    return infer


def _run_mtime(run: Path) -> float | None:
    """A run's newest-artifact timestamp (epoch seconds, float): policy.onnx
    is the finished product; live.onnx then progress.jsonl cover runs still
    training or stopped before export. None for a dir with none of them."""
    for name in ("policy.onnx", "live.onnx", "progress.jsonl"):
        f = run / name
        if f.exists():
            return f.stat().st_mtime
    return None


def _run_size(run: Path) -> int:
    """Bytes a run dir occupies, checkpoints included — what deleting it
    frees. The delete confirmation shows this, so a user can tell a 4 MB
    scratch run from the 900 MB chain that is eating the disk. Unreadable
    entries are skipped rather than failing the whole listing."""
    total = 0
    for f in run.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


# Curriculum-chain run names: teach-<behavior>-<hash>-sN. The palette folds
# the stages of one chain into a single family row.
_CHAIN_RE = re.compile(r"^(teach-.+)-s(\d+)$")


def discover_policies() -> list[dict]:
    """Everything assignable, grouped for the palette. Run entries carry
    `mtime` (epoch seconds, see _run_mtime) and are sorted newest-first —
    the user couldn't tell which run was fresh from bare name chips. Stage
    runs additionally carry `chain` (the prefix without -sN) and `stage`
    (1-based) so the panel can group a curriculum chain as one family.
    `sizeBytes` (run entries) is what deleting the run would free — the
    palette's delete confirmation shows it."""
    out: list[dict] = []
    if POLICIES_DIR.exists():
        for p in sorted(POLICIES_DIR.glob("*.onnx")):
            out.append({"id": f"pollen:{p.stem}", "label": p.stem,
                        "group": "pollen", "path": str(p)})
    run_entries: list[dict] = []
    ckpt_entries: list[dict] = []
    if RUNS_DIR.exists():
        runs = [r for r in RUNS_DIR.iterdir() if r.is_dir()]
        # mtime descending, name as the tiebreak so the order is stable.
        runs.sort(key=lambda r: (-(_run_mtime(r) or 0.0), r.name))
        for run in runs:
            if (run / "policy.onnx").exists():
                entry = {"id": f"run:{run.name}", "label": run.name,
                         "group": "runs", "path": str(run / "policy.onnx"),
                         "mtime": _run_mtime(run),
                         "sizeBytes": _run_size(run)}
                m = _CHAIN_RE.match(run.name)
                if m:
                    entry["chain"] = m.group(1)
                    entry["stage"] = int(m.group(2))
                run_entries.append(entry)
            for z in sorted(run.glob("checkpoints/model_*_steps.zip"),
                            key=lambda p: int(p.stem.split("_")[1])):
                steps = z.stem.split("_")[1]
                vn = z.parent / f"model_vecnormalize_{steps}_steps.pkl"
                if vn.exists():
                    label = f"{run.name}@{int(steps) // 1000}k"
                    ckpt_entries.append({"id": f"ckpt:{label}", "label": label,
                                         "group": "checkpoints", "path": str(z)})
    return out + run_entries + ckpt_entries


# Run names as they arrive off the wire on DELETE /runs/{name}. Restrictive on
# purpose (same reasoning as clip_path): the string selects a DIRECTORY TREE to
# erase, so no separators, no leading dot, nothing that could climb out of
# runs/.
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


def run_dir(name: str) -> Path:
    """Path of a run directory inside RUNS_DIR, or ValueError.

    fullmatch, not match: `$` also matches before a trailing newline, so
    `.match()` waved through "my-run\\n" — and this name reaches a spawned
    trainer's --init-from. No real run name ends in one (all 373 on disk
    validate identically either way), so this only ever rejects more."""
    if not isinstance(name, str) or not RUN_NAME_RE.fullmatch(name) or ".." in name:
        raise ValueError("run name must be 1-96 chars of letters, digits, dot, "
                         "dash or underscore, starting alphanumeric")
    return RUNS_DIR / name


def chain_run_names(name: str) -> list[str]:
    """Every EXISTING run dir belonging to chain `name`: the stage runs
    <name>-s1, -s2, … in stage order, plus a bare <name> dir if one exists
    (a non-curriculum job trained under the same base name). Used by
    DELETE /runs/{name}?chain=true so a five-stage trick goes in one act
    instead of five half-confirmed ones."""
    if not RUNS_DIR.exists():
        return []
    staged: list[tuple[int, str]] = []
    bare: list[str] = []
    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        m = _CHAIN_RE.match(d.name)
        if m and m.group(1) == name:
            staged.append((int(m.group(2)), d.name))
        elif d.name == name:
            bare.append(d.name)
    return bare + [n for _, n in sorted(staged)]


def training_run_names(st: "LabState") -> set[str]:
    """Run names the ACTIVE job owns — its current stage plus every other
    stage of the same chain. A chain stage warm-starts from the previous
    stage's dir, so deleting an already-finished stage mid-chain would break
    the launch of the next one; the whole chain is off limits until the job
    stops."""
    job = getattr(st, "job", None)
    if job is None or job.status != "training":
        return set()
    base = job._base_name
    return {job.run_name, base} | {
        d.name for d in ([] if not RUNS_DIR.exists() else RUNS_DIR.iterdir())
        if d.is_dir() and (m := _CHAIN_RE.match(d.name)) and m.group(1) == base
    }


def delete_runs(names: list[str], st: "LabState | None" = None) -> dict:
    """Erase run directories. All-or-nothing on the guards: if ANY target is
    off limits (bad name, missing, still training) nothing is deleted, so a
    chain can't end up half-gone. Returns {deleted, freedBytes}.

    Deleting a run does NOT disturb ducks already running its brain — an
    onnx session is loaded in memory and keeps stepping. They simply drop out
    of the roster on the next lab restart (restore_ducks skips entries whose
    file is gone)."""
    active = training_run_names(st) if st is not None else set()
    targets: list[Path] = []
    for name in names:
        d = run_dir(name)                       # raises ValueError on junk
        if name in active:
            raise PermissionError(
                f"“{name}” belongs to the job training right now — stop the "
                f"training first")
        if not d.is_dir():
            raise FileNotFoundError(name)
        targets.append(d)
    if not targets:
        raise FileNotFoundError(", ".join(names) or "(nothing)")
    freed = 0
    deleted: list[str] = []
    for d in targets:
        freed += _run_size(d)
        shutil.rmtree(d)
        deleted.append(d.name)
        # A cached infer keeps a deleted policy assignable by id — drop it so
        # a stale palette chip fails honestly instead of resurrecting it.
        _infer_cache.pop(f"run:{d.name}", None)
        for key in [k for k in _infer_cache if k.startswith(f"ckpt:{d.name}@")]:
            _infer_cache.pop(key, None)
    return {"deleted": deleted, "freedBytes": freed}


_infer_cache: dict[str, object] = {}


def load_policy_infer(policy_id: str):
    """Resolve a palette id to an infer callable (cached). Runs in a thread."""
    if policy_id in _infer_cache:
        return _infer_cache[policy_id]
    entry = next((p for p in discover_policies() if p["id"] == policy_id), None)
    if entry is None:
        raise KeyError(policy_id)
    path = Path(entry["path"])
    if path.suffix == ".onnx":
        infer = _onnx_infer(path)
    else:
        steps = path.stem.split("_")[1]
        infer = _checkpoint_infer(path, path.parent / f"model_vecnormalize_{steps}_steps.pkl")
    _infer_cache[policy_id] = infer
    return infer


def build_ducks(args) -> list[Duck]:
    ducks: list[Duck] = []

    def add(label, infer, policy_id=None, onnx_path=None):
        ducks.append(Duck(f"d{len(ducks)}", label, infer, seed=len(ducks),
                          policy_id=policy_id, onnx_path=onnx_path))

    if args.checkpoints:
        run = Path(args.checkpoints)
        zips = sorted(run.glob("checkpoints/model_*_steps.zip"),
                      key=lambda p: int(p.stem.split("_")[1]))
        for z in zips:
            steps = z.stem.split("_")[1]
            vn = z.parent / f"model_vecnormalize_{steps}_steps.pkl"
            if vn.exists():
                label = f"{run.name}@{int(steps) // 1000}k"
                add(label, _checkpoint_infer(z, vn), policy_id=f"ckpt:{label}")
        if (run / "policy.onnx").exists():
            add(f"{run.name}@final", _onnx_infer(run / "policy.onnx"),
                onnx_path=str(run / "policy.onnx"))

    for spec in args.policies:
        p = Path(spec)
        if p.suffix == ".onnx":
            add(p.parent.name if p.name == "policy.onnx" else p.stem,
                _onnx_infer(p), onnx_path=str(p))
        elif p.is_dir():
            onnx = p / "policy.onnx"
            if not onnx.exists() and (p / "model.zip").exists():
                from .export_onnx import export
                export(p, onnx)
                print(f"[lab] exported {onnx} on the fly")
            if onnx.exists():
                add(p.name, _onnx_infer(onnx), onnx_path=str(onnx))
            else:
                print(f"[lab] skipping {p}: no policy.onnx / model.zip")
        else:
            print(f"[lab] skipping {spec}: not an .onnx or run dir")
    if not ducks:
        raise SystemExit("no ducks — pass run dirs or .onnx paths")
    return ducks


# ------------------------------------------------------------ training jobs

class TrainingJob:
    """One logical teach job: a `train-behavior` subprocess chain + its
    progress/snapshot state.

    A behavior with a curriculum trains as a SEQUENCE of stages — run names
    `teach-<id>-<hash>-s1`, `-s2`, … — where each stage `--init-from`s the
    previous stage's dir (a cross-dir fine-tune: fresh step budget) under its
    own env knobs. `run_name`/`dir` always point at the ACTIVE stage, so the
    progress tail, live.onnx watcher and helper guards follow the chain
    without knowing about it.

    Practice budget: `steps` is the TEST knob (TEACH_STEPS_OVERRIDE) and
    keeps its own meaning — that many steps for EVERY stage. `budget` is the
    user-facing control: ONE total for the whole job, split across the stages
    in proportion to their declared steps (split_step_budget) so a curriculum
    keeps its shape, with `stage_budgets` ({1-based stage: steps}) replacing
    individual stages outright. `steps` wins over both, so a probe still runs
    a tiny job whatever budget the user last picked.

    Helper scaling: SB3 can't add envs to a live SubprocVecEnv, so scale()
    SIGTERMs the trainer and relaunches it warm — `--init-from` its own run
    dir, same run name (progress.jsonl keeps appending), new `--envs`. The
    relaunch rewinds to the last snapshot, so progress may step back by up to
    one snapshot interval. The `restarting` flag covers the gap for the UI.
    """

    def __init__(self, behavior_id: str, helpers: int = 0,
                 steps: int | None = None, snap_steps: int | None = None,
                 weights: dict[str, float] | None = None,
                 init_from: Path | None = None,
                 stage_weights: dict | None = None,
                 start_stage: int = 1,
                 stage_init_from: Path | None = None,
                 extra_env: dict[str, str] | None = None,
                 budget: int | None = None,
                 stage_budgets: dict | None = None):
        self.behavior = behaviors_mod.BEHAVIORS[behavior_id]
        # Run-scoped knobs that are not stage knobs — currently the reference
        # CLIP an imitation run tracks, so a user can author several motions
        # in the timeline editor and train whichever one they mean.
        self.extra_env = dict(extra_env or {})
        clip_slug = ""
        if (extra_env or {}).get("MICRODUCK_CLIP"):
            clip_slug = "-" + re.sub(r"[^a-zA-Z0-9]+", "_",
                                     extra_env["MICRODUCK_CLIP"])[:24].strip("_")
        base = f"teach-{behavior_id}{clip_slug}-{uuid.uuid4().hex[:6]}"
        curriculum = tuple(self.behavior.curriculum)
        # An explicit initFrom is the user fine-tuning an EXISTING run —
        # re-running the whole chain would throw away what they chose to
        # keep, so that stays a single run, trained under the FINAL stage's
        # env (the finished trick's spawn window).
        self.stages = curriculum if (curriculum and init_from is None) else ()
        # startStage begins the chain partway (the earlier stages already
        # trained elsewhere) — the caller resolves stage_init_from to the
        # prev-stage run the first launched stage warm-starts from. Overall
        # progress counts only the stages actually being run.
        self._start_idx = (min(max(int(start_stage), 1), len(self.stages)) - 1
                           if self.stages else 0)
        self.stage_idx = self._start_idx
        if self.stages:
            declared = [s.steps for s in self.stages]
            self.run_name = f"{base}-s{self.stage_idx + 1}"
            self._stage_env = dict(self.stages[self.stage_idx].env)
            # Whatever the caller resolved — a prev-stage run for a startStage
            # jump, or a donor brain the whole chain warm-starts from (stage 1
            # included, which is how the chain inherits a skill like standing).
            launch_init = stage_init_from
        else:
            declared = [self.behavior.default_steps]
            self.run_name = base
            self._stage_env = dict(curriculum[-1].env) if curriculum else {}
            launch_init = init_from
        # How long the user asked it to practice, in total, clamped —
        # None means "use what the recipe declares". Kept separate from
        # stage_steps so the sticky value can't drift with a per-stage
        # override folded in.
        self.budget = (None if (budget is None or steps)
                       else max(MIN_STEP_BUDGET,
                                min(int(budget), MAX_STEP_BUDGET)))
        self.stage_budgets = self._clamp_stage_budgets(stage_budgets)
        self.stage_steps = self._resolve_stage_steps(declared, steps)
        self._base_name = base
        # What the CURRENT stage warm-started from — scale() falls back to it
        # when no snapshot exists yet, so a restart can't silently drop a
        # stage's (or fine-tune's, or a startStage jump's) inheritance.
        self._stage_init_from = launch_init
        self.dir = RUNS_DIR / self.run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.helpers = helpers
        self.envs = BASE_ENVS + ENVS_PER_HELPER * helpers
        # Set by stop(); scale() honours it instead of relaunching (see stop()).
        self.stop_requested = False
        self.total_steps = self.stage_steps[self.stage_idx]  # ACTIVE stage's budget
        self.snap_steps = snap_steps
        # Mirror BehaviorEnv's clamp so the payload shows the weights actually
        # in force, not what a client asked for. Weight keys OUTSIDE the recipe
        # adopt that CATALOG term (the "＋ add a term" channel — BehaviorEnv
        # composes them at launch; unknown keys are dropped here).
        self.weights = self._clamp_weights(weights)
        # Per-stage overrides LAYERED over the behavior-level weights (stage
        # wins per key) — {1-based stage: {key: weight}}. Only meaningful for
        # staged chains; a single-run job (incl. initFrom fine-tunes) ignores
        # them so a fine-tune can't half-apply one stage's crank.
        self.stage_weights: dict[int, dict[str, float]] = (
            self._clamp_stage_weights(stage_weights) if self.stages else {})
        self._refresh_extra_keys()
        self.restarting = False
        # Live handles on the trainer's SubprocVecEnv workers, refreshed by
        # poll() while it is alive — the only way to reach them once it is not.
        self._workers: list[psutil.Process] = []
        self.progress: dict = {"steps": 0, "total": self.total_steps}
        self._offset = 0
        self._live_mtime = 0.0
        self._fps_points: list[tuple[float, float]] = []  # (steps, elapsed_s)
        # Job-lifetime wall clock: elapsed_s restarts with every subprocess
        # (stage handoffs, helper warm-restarts), so the "how long has this
        # been training" number lives here instead. Frozen when the job
        # leaves "training" so a finished run doesn't keep counting.
        self._t0: float | None = time.time()
        self._elapsed_final: float | None = None
        self.status = "training"  # training | done | stopped | failed
        # A launched job creates the trainee/helpers, so its card owns them.
        self.owns_preview_ducks = True
        self.proc = self._launch(init_from=launch_init)

    # Does this job own the trainee/helper ducks? A LAUNCHED job creates them,
    # so dismissing its card takes them along. An ADOPTED one (POST /teach/load,
    # fired by merely selecting a duck) created nothing and must never sweep a
    # roster it did not build — including the one restore_ducks just brought
    # back after a restart.
    #
    # Defaults FALSE and is turned on in __init__, not the other way round:
    # this flag guards an irreversible roster delete, so a construction path
    # that forgets it must fail toward leaving ducks alone. (adopt() builds via
    # cls.__new__ and hand-assigns its fields, so it never runs __init__ — it
    # simply inherits this default.)
    owns_preview_ducks = False

    @classmethod
    def adopt(cls, run_name: str) -> "TrainingJob":
        """Seat a FINISHED run as the panel's current job — no subprocess.

        POST /teach/load uses this so clicking a duck (or dropping a chip on
        the teach panel) pulls that run's recipe up for refinement: the
        payload streams in "done" state, which is exactly the state whose
        sliders unlock and whose ✨ fine-tune targets `run_name`. Built from
        the run's behavior.json (the weights actually trained under — written
        so an inspected run can't show a different scorecard than it ran).
        Always a single-run seat, even for a chain stage: fine-tuning `-s3`
        should warm-start from THAT brain, and retrain re-runs the chain via
        the behavior title as usual. `proc` is None; poll()/stop()/sample()
        guard for it.
        """
        run = RUNS_DIR / run_name
        meta = json.loads((run / "behavior.json").read_text())
        behavior_id = meta.get("behavior")
        if behavior_id not in behaviors_mod.BEHAVIORS:
            raise ValueError(
                f"{run_name} trained behavior {behavior_id!r}, which is no "
                "longer in behaviors.py — nothing to refine")
        self = cls.__new__(cls)
        self.behavior = behaviors_mod.BEHAVIORS[behavior_id]
        # Seated, not launched: created no preview ducks (redundant with the
        # class default, stated here so the contract is visible at the site).
        self.owns_preview_ducks = False
        self.extra_env = {}
        self.stages = ()
        self._start_idx = 0
        self.stage_idx = 0
        self.run_name = run_name
        self._base_name = run_name
        curriculum = tuple(self.behavior.curriculum)
        self._stage_env = dict(curriculum[-1].env) if curriculum else {}
        self.budget = None
        self.stage_budgets = {}
        steps = int(meta.get("steps") or self.behavior.default_steps)
        self.stage_steps = [steps]
        self._stage_init_from = None
        self.dir = run
        self.helpers = 0
        self.envs = BASE_ENVS
        self.stop_requested = False
        self.total_steps = steps
        self.snap_steps = None
        self.weights = self._clamp_weights(meta.get("weights") or None)
        self.stage_weights = {}
        self._refresh_extra_keys()
        self.restarting = False
        self._workers = []
        self.progress = {"steps": 0, "total": steps}
        self._offset = 0  # poll() replays progress.jsonl → real final numbers
        # live.onnx here is old news, not a fresh snapshot — don't let the
        # first poll() flag it.
        self._live_mtime = time.time()
        self._fps_points = []
        self._t0 = None  # adopted after the fact — its wall clock is unknown
        self._elapsed_final = None
        self.status = "done"
        self.proc = None
        return self

    def _clamp_stage_budgets(self, stage_budgets: dict | None) -> dict[int, int]:
        """Explicit per-stage step counts, 1-based (JSON keys arrive as
        strings), mirroring _clamp_stage_weights. A single-run job has no
        stages to key, so it drops the whole layer."""
        out: dict[int, int] = {}
        if not self.stages:
            return out
        for i, v in (stage_budgets or {}).items():
            try:
                idx, n = int(i), int(v)
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= len(self.stages) and n > 0:
                out[idx] = min(n, MAX_STEP_BUDGET)
        return out

    def _resolve_stage_steps(self, declared: list[int],
                             steps: int | None) -> list[int]:
        """The per-stage budgets this job actually trains under. Precedence,
        narrowest last: the recipe's declared steps → the user's TOTAL budget
        split proportionally → an explicit per-stage number. The test knob
        (`steps`) short-circuits all of it — TEACH_STEPS_OVERRIDE must keep
        meaning "tiny job", not "tiny job unless a budget is sticky"."""
        if steps:
            return [int(steps)] * len(declared)
        out = (split_step_budget(declared, self.budget)
               if self.budget is not None else list(declared))
        for i, v in self.stage_budgets.items():
            out[i - 1] = v
        return out

    def _clamp_weights(self, weights: dict | None) -> dict[str, float]:
        recipe_keys = {t.key for t in self.behavior.terms}
        return {
            k: max(0.0, float(v)) for k, v in (weights or {}).items()
            if k in recipe_keys or k in behaviors_mod.CATALOG
        }

    def _clamp_stage_weights(self, stage_weights: dict | None
                             ) -> dict[int, dict[str, float]]:
        """Same clamp per stage; wire keys are 1-based stage indices (JSON
        keys arrive as strings). Out-of-range stages and empty dicts drop."""
        out: dict[int, dict[str, float]] = {}
        for i, sw in (stage_weights or {}).items():
            try:
                idx = int(i)
            except (TypeError, ValueError):
                continue
            if not 1 <= idx <= len(self.stages):
                continue
            clamped = self._clamp_weights(sw)
            if clamped:
                out[idx] = clamped
        return out

    def _refresh_extra_keys(self) -> None:
        """Adopted catalog terms, unioned across the behavior-level weights
        and every stage's overrides — the card must show a slider row for a
        term any layer adopted."""
        recipe_keys = {t.key for t in self.behavior.terms}
        self.extra_keys = tuple(dict.fromkeys(
            k for src in (self.weights, *self.stage_weights.values())
            for k in src
            if k not in recipe_keys and k in behaviors_mod.CATALOG
        ))

    def stage_launch_weights(self) -> dict[str, float]:
        """What the ACTIVE stage actually trains under: behavior-level weights
        with this stage's overrides layered on top (stage wins per key)."""
        return {**self.weights, **self.stage_weights.get(self.stage_idx + 1, {})}

    def set_stage_weights(self, stage_weights: dict | None) -> bool:
        """Replace the whole per-stage override map (the panel sends it in
        full). Future stages pick the new values up at their handoff launch
        (_launch re-reads the map). Returns True when the ACTIVE stage's
        merged weights changed — the caller then warm-restarts it via
        scale() with the current helper count (the helper-join pattern:
        terminate, relaunch --init-from the stage's own snapshot)."""
        if not self.stages:
            return False
        before = self.stage_launch_weights()
        self.stage_weights = self._clamp_stage_weights(stage_weights)
        self._refresh_extra_keys()
        return self.stage_launch_weights() != before

    def clip_name(self) -> str | None:
        return self.extra_env.get("MICRODUCK_CLIP")

    def display_title(self) -> str:
        """What the panel and the trainee duck are called. An imitation run is
        about a SPECIFIC authored motion, so it says which one."""
        clip = self.clip_name()
        return f"Perform “{clip}”" if clip else self.behavior.title

    def _behavior_card(self) -> dict:
        card = behaviors_mod.behavior_card(self.behavior, extra_keys=self.extra_keys)
        clip = self.clip_name()
        if clip:
            card["title"] = self.display_title()
            card["clip"] = clip
        return card

    def stage_env(self) -> dict[str, str]:
        """The ACTIVE stage's env knobs — what the lab mirrors onto the
        trainee's preview env. Includes `extra_env` (the run's MICRODUCK_CLIP)
        because the trainer subprocess merges the same two dicts into its
        environment (see _launch): an imitation run whose clip only rode
        extra_env gave the preview duck the recipe's DEFAULT clip, so the
        watched duck tracked a different motion than the one training."""
        return {**self.extra_env, **self._stage_env}

    def _launch(self, init_from: Path | None) -> subprocess.Popen:
        cmd = [sys.executable, "-m", "microduck_local.train_behavior",
               self.behavior.id, "--run-name", self.run_name,
               "--envs", str(self.envs), "--steps", str(self.total_steps)]
        if self.snap_steps:
            cmd += ["--snap-steps", str(self.snap_steps)]
        # Merged per launch, not cached: a handoff (_advance_stage) and a
        # weight-edit restart (scale) both land here with stage_idx /
        # stage_weights already updated.
        weights = self.stage_launch_weights()
        if weights:
            cmd += ["--weights-json", json.dumps(weights)]
        if init_from is not None:
            cmd += ["--init-from", str(init_from)]
        log = open(self.dir / "train.log", "a")
        # Stage env knobs (spawn windows etc.) ride the subprocess
        # environment — the trainer itself stays curriculum-agnostic.
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                cwd=str(Path(__file__).resolve().parents[2]),
                                env={**os.environ, **self.extra_env,
                                     **self._stage_env})

    def _advance_stage(self) -> None:
        """Launch the next curriculum stage: a fresh run fine-tuned from the
        one that just finished (`--init-from` a DIFFERENT dir = full new step
        budget), under the new stage's env knobs. Watcher state resets so
        poll() tails the new run dir from byte 0 and the first snapshot of the
        new stage lands on the trainee like any other."""
        prev_dir = self.dir
        self.stage_idx += 1
        stage = self.stages[self.stage_idx]
        self.run_name = f"{self._base_name}-s{self.stage_idx + 1}"
        self.dir = RUNS_DIR / self.run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.total_steps = self.stage_steps[self.stage_idx]
        self._stage_env = dict(stage.env)
        self._stage_init_from = prev_dir
        self.progress = {"steps": 0, "total": self.total_steps}
        self._offset = 0
        self._live_mtime = 0.0
        self._fps_points.clear()
        self.proc = self._launch(init_from=prev_dir)

    def effective_weights(self) -> dict[str, float]:
        """Recipe defaults with the job's BEHAVIOR-LEVEL overrides folded in —
        what the UI's whole-chain sliders read (per-stage overrides layer on
        top client-side, from the payload's stageWeights). Adopted catalog
        terms ride along at their given weight; one adopted only by a stage
        shows its catalog default at the behavior level."""
        out = {t.key: self.weights.get(t.key, t.weight)
               for t in self.behavior.terms}
        for k in self.extra_keys:
            out[k] = self.weights.get(k, behaviors_mod.CATALOG[k].weight)
        return out

    def _snapshot_workers(self) -> list[psutil.Process]:
        """Remember the trainer's SubprocVecEnv workers WHILE it is alive.

        The moment the trainer exits its workers are reparented to init and
        `children()` can no longer name them, so the list has to be kept fresh
        (poll() refreshes it ~1 Hz) rather than looked up at kill time — that
        is the only way the poll()-found-it-dead path has anything to sweep.

        psutil handles rather than bare pids on purpose: a Process is
        identified by (pid, create_time), so a stale entry whose pid has been
        recycled raises NoSuchProcess instead of signalling whatever process
        now owns that number.
        """
        if self.proc.poll() is None:
            try:
                self._workers = psutil.Process(self.proc.pid).children(
                    recursive=True)
            except psutil.Error:
                pass  # racing its death — the last live snapshot still stands
        return self._workers

    def _sweep_workers(self, workers: list[psutil.Process] | None = None) -> None:
        """SIGTERM every remembered worker.

        Under the `fork` backend a worker never sees EOF on its pipe when the
        trainer dies — it inherited its siblings' copies of the parent end, so
        the read side stays open forever (`forkserver`, the old default, had
        no such inheritance and the fleet drained itself). A worker therefore
        outlives an un-swept trainer at 0% CPU holding ~32 MB, which is how
        one stop stranded ~322 MB for the rest of a session.

        Idempotent: an already-dead handle just raises NoSuchProcess. Every
        psutil error is swallowed per worker so one bad handle cannot strand
        the rest of the fleet.
        """
        for c in (self._workers if workers is None else workers):
            try:
                c.terminate()
            except psutil.Error:
                pass

    def _terminate_tree(self, wait_s: float = 20.0) -> None:
        """SIGTERM the trainer AND every SubprocVecEnv worker beneath it.

        The child list must be snapshotted while the trainer is still alive
        (see _snapshot_workers). Skipping the sweep is how a stop stranded ~10
        workers for the rest of the session — scale() had it right and stop()
        did not, so both now share this.

        wait_s=0 skips reaping the parent, for callers on the event loop
        (/teach/stop is an async handler; blocking it stalls the WebSocket
        stream for every watching duck). The workers still get their own
        SIGTERM either way, which is the part that matters for the leak.
        """
        # Held locally as well as cached: scale() runs on a worker thread, so
        # a concurrent poll() may rebind self._workers between these two
        # lines, and the fleet we are killing is the one we just named.
        if self.proc is None:  # adopted run: no subprocess ever existed
            return
        workers = self._snapshot_workers()
        if self.proc.poll() is None:
            self.proc.terminate()
            if wait_s:
                try:
                    self.proc.wait(timeout=wait_s)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        self._sweep_workers(workers)

    def scale(self, helpers: int) -> None:
        """Warm-restart the trainer. Blocking (SIGTERM + wait) — call it
        from a thread. Env count stays BASE_ENVS: helpers are extra
        viewers, not extra workers (see the module comment on live-lab
        steps/s). Used by /teach/weights, not by spawn_helper."""
        self.restarting = True
        try:
            self._terminate_tree()
            self.helpers = helpers
            self.envs = BASE_ENVS + ENVS_PER_HELPER * helpers
            # Old-process rates would otherwise keep showing as trainFps while
            # the new trainer boots (venv spawn + PPO.load take seconds) —
            # null is the honest reading until two fresh lines land.
            self._fps_points.clear()
            # model.zip appears with the first snapshot; before that, fall
            # back to what this stage warm-started from (the previous stage's
            # dir, or a /teach initFrom) — a cold relaunch here would silently
            # drop that inheritance. None only for a genuinely cold first run.
            init = (self.dir if (self.dir / "model.zip").exists()
                    else self._stage_init_from)
            # A /teach/stop that landed while we were tearing the old trainer
            # down wins. Relaunching here would strand a live trainer (plus its
            # fork workers) behind status == "stopped", which poll() never
            # looks at again.
            if self.stop_requested:
                self.status = "stopped"
                return
            self.proc = self._launch(init_from=init)
        finally:
            self.restarting = False

    def poll(self) -> tuple[bool, bool]:
        """Returns (progress_changed, new_snapshot)."""
        changed = snap = False
        pf = self.dir / "progress.jsonl"
        if pf.exists():
            with open(pf) as f:
                f.seek(self._offset)
                for line in f:
                    if line.endswith("\n"):
                        self._offset += len(line)
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        self.progress = {**self.progress, **rec}
                        changed = True
                        if "elapsed_s" in rec:
                            self._fps_points.append(
                                (float(rec["steps"]), float(rec["elapsed_s"])))
                            del self._fps_points[:-2]
        live = self.dir / "live.onnx"
        if live.exists():
            m = live.stat().st_mtime
            if m > self._live_mtime + 0.5:
                self._live_mtime = m
                snap = True
        if self.proc is None:  # adopted run: nothing running, nothing to reap
            return changed, snap
        # During scale() the old proc is dead on purpose — not a failure.
        if (self.proc.poll() is not None and self.status == "training"
                and not self.restarting):
            # A trainer the lab did not kill itself (OOM, a stray kill -9, a
            # crash by signal) never runs multiprocessing's atexit hook, so it
            # orphans its workers exactly as an un-swept stop did. This is the
            # last moment anything knows their pids — spend the snapshot.
            # A clean exit already reaped them via atexit, making this a
            # no-op, and a stage handoff sweeps the finished stage's fleet
            # before _advance_stage() rebinds self.proc to the next one.
            self._sweep_workers()
            if (self.proc.returncode == 0
                    and self.stage_idx < len(self.stages) - 1):
                # Stage complete → chain the next one. Advancing on process
                # EXIT (not the progress "done" line) guarantees the finished
                # stage's model.zip/vecnormalize.pkl are fully written before
                # the next stage warm-starts from them.
                self._advance_stage()
            else:
                self.status = "done" if self.proc.returncode == 0 else "failed"
                self.finished_clean = self.status == "done"
                self._freeze_elapsed()
            changed = True
        else:
            self._snapshot_workers()
        return changed, snap

    def train_fps(self) -> float | None:
        """Steps/sec from the last two progress lines. None right after a
        restart — elapsed_s starts over in the new subprocess, so the deltas
        only make sense between lines from the same one. Also None once the
        job is no longer training: a frozen "6.4k steps/s" after completion
        reads as a live-but-stuck run (a user hit exactly that)."""
        if self.status != "training":
            return None
        if len(self._fps_points) < 2:
            return None
        (s0, e0), (s1, e1) = self._fps_points
        if s1 <= s0 or e1 <= e0:
            return None
        return (s1 - s0) / (e1 - e0)

    def stop(self) -> None:
        """Stop the whole job: kill the current stage's subprocess, and the
        "stopped" status keeps poll() from ever chaining the next stage."""
        # Non-blocking: this runs on the event loop, so the trainer is not
        # reaped here (poll() collects it later) — but its workers are swept,
        # which is the leak that mattered.
        #
        # stop_requested is checked by scale(), which runs on a worker thread:
        # without it, a stop landing MID-RESCALE killed the old (already dead)
        # process, set status="stopped", and then scale() went on to launch a
        # brand-new trainer and rebind self.proc. poll() is gated on
        # status == "training", so that trainer — plus its 16-32 fork workers —
        # ran unreachable until the lab exited.
        self.stop_requested = True
        self._terminate_tree(wait_s=0)
        self.status = "stopped"
        self._freeze_elapsed()

    def stage_payload(self) -> dict | None:
        """The frame's `training.stage` field — null for single-run jobs.
        `start` (1-based) is where this chain actually began (startStage);
        stages before it were skipped, warm-started from an earlier run."""
        if not self.stages:
            return None
        stage = self.stages[self.stage_idx]
        return {"idx": self.stage_idx + 1, "count": len(self.stages),
                "label": stage.label, "detail": stage.detail,
                "start": self._start_idx + 1}

    def _freeze_elapsed(self) -> None:
        if self._t0 is not None and self._elapsed_final is None:
            self._elapsed_final = time.time() - self._t0

    def overall_elapsed(self) -> float | None:
        """Wall-clock seconds the JOB has been training — across stage
        handoffs and warm restarts, where progress.elapsed_s starts over.
        Frozen at the moment the job stops training; None for adopted runs
        (they finished before this lab ever saw them)."""
        if self._t0 is None:
            return None
        if self._elapsed_final is not None:
            return self._elapsed_final
        return time.time() - self._t0

    def overall_progress(self) -> tuple[int, int]:
        """(steps, total) cumulative across the stages actually being RUN,
        using each stage's declared budget — what long-lived counters show,
        so progress never appears to reset when a stage hands off, and a
        startStage jump doesn't book skipped stages as instant progress."""
        done = sum(self.stage_steps[self._start_idx:self.stage_idx])
        return (done + int(self.progress.get("steps", 0) or 0),
                sum(self.stage_steps[self._start_idx:]))

    def payload(self) -> dict:
        overall_steps, overall_total = self.overall_progress()
        return {
            "runName": self.run_name,
            "status": self.status,
            "behavior": self._behavior_card(),
            "weights": self.effective_weights(),
            # Per-stage fields stay exactly as streamed (existing consumers);
            # overall* are cumulative across the stage chain (== steps/total
            # for single-run jobs).
            "progress": {**self.progress, "overallSteps": overall_steps,
                         "overallTotal": overall_total,
                         "overallElapsed": (
                             None if (el := self.overall_elapsed()) is None
                             else round(el, 1))},
            "stage": self.stage_payload(),
            # Per-stage OVERRIDES only (1-based string keys) — the panel
            # layers them over `weights` to show a stage's merged sliders.
            "stageWeights": {str(i): dict(w)
                             for i, w in sorted(self.stage_weights.items())},
            # The practice budget IN FORCE: per-stage counts in stage order
            # (one entry for a single-run job) and their sum. The panel shows
            # these rather than the recipe's declared numbers, so an
            # inspected run can never advertise a budget it isn't training
            # under. stageBudgets is the explicitly-pinned subset (what the
            # stage inspector marks as overridden), like stageWeights.
            "stageSteps": list(self.stage_steps),
            "stepBudget": sum(self.stage_steps),
            # The TOTAL the user chose, or null when the recipe's own plan is
            # in force. stepBudget above already has per-stage pins folded
            # in, so the panel needs this to re-split around an edit.
            "chosenBudget": self.budget,
            "stageBudgets": {str(i): v
                             for i, v in sorted(self.stage_budgets.items())},
            "envs": self.envs,
            "helpers": self.helpers,
            "maxHelpers": MAX_HELPERS,
            "restarting": self.restarting,
        }


class StatsSampler:
    """~1 Hz psutil sampling for the frame stream.

    cpu_percent(interval=None) measures since the previous call on the SAME
    Process object (first call reads 0.0), so handles are cached per pid —
    fresh objects every poll would read 0.0 forever.
    """

    def __init__(self):
        self.lab = psutil.Process()
        self._tracked: dict[int, psutil.Process] = {}
        psutil.cpu_percent(interval=None)  # prime the machine-wide window
        self.lab.cpu_percent(interval=None)

    def _tree(self, root_pid: int) -> tuple[float, float] | None:
        """(cpu%, rss MB) summed over a process and its live descendants — the
        trainer plus its SubprocVecEnv workers. Workers churn during scale
        restarts, so every psutil touch races process death gracefully."""
        try:
            root = psutil.Process(root_pid)
            procs = [root, *root.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
        cpu = rss = 0.0
        tracked: dict[int, psutil.Process] = {}
        for p in procs:
            p = self._tracked.get(p.pid, p)
            try:
                cpu += p.cpu_percent(interval=None)
                rss += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            tracked[p.pid] = p
        self._tracked = tracked
        return cpu, rss / 2**20

    def sample(self, job: TrainingJob | None) -> dict:
        stats = {
            "cpu": psutil.cpu_percent(interval=None),
            "mem": psutil.virtual_memory().percent,
            "lab": {"cpu": round(self.lab.cpu_percent(interval=None), 1),
                     "memMb": round(self.lab.memory_info().rss / 2**20, 1)},
            "trainer": None,
            "trainFps": None,
        }
        if job is not None:
            if job.proc is not None and job.proc.poll() is None:
                tree = self._tree(job.proc.pid)
                if tree is not None:
                    stats["trainer"] = {"cpu": round(tree[0], 1),
                                        "memMb": round(tree[1], 1)}
            fps = job.train_fps()
            stats["trainFps"] = round(fps, 1) if fps is not None else None
        return stats


# ------------------------------------------------------------ lab persistence

def lab_state_path() -> Path:
    """Roster persistence target. LAB_STATE_PATH lets a scratch server (tests,
    a second port) keep its hands off the real lab's file."""
    env = os.environ.get("LAB_STATE_PATH")
    return Path(env) if env else RUNS_DIR.parent / "lab-state.json"


def teach_weights_path() -> Path:
    return lab_state_path().with_name("teach-weights.json")


# The layers of one behavior's sticky panel settings. Everything a user
# adjusts by hand in the teach panel remembers itself per behavior.
STICKY_KEYS = ("weights", "stageWeights", "steps", "stageSteps")


def empty_sticky() -> dict:
    return {"weights": {}, "stageWeights": {}, "steps": None, "stageSteps": {}}


def load_teach_weights() -> dict[str, dict]:
    """Per-behavior sticky panel settings, normalized to four layers:
    {behavior: {"weights": {key: w}, "stageWeights": {"<1-based>": {key: w}},
                "steps": total practice budget | None,
                "stageSteps": {"<1-based>": steps}}}.
    A user who cranked a term — or who always practices for 4M steps —
    expects the NEXT 'do a backflip' to keep it; before this, any fresh
    /teach without explicit weights silently reset every slider to the recipe
    defaults (and the user lost track of what they had set). Files from
    before per-stage weights stored the flat behavior-level dict
    ({"backflip": {"legs_over": 2.5}}) — still read, as the behavior-level
    layer, so existing files stay valid."""
    try:
        raw = json.loads(teach_weights_path().read_text())
    except (OSError, ValueError):
        return {}
    out: dict[str, dict] = {}
    for bid, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        if any(k in entry for k in STICKY_KEYS):
            out[bid] = {
                "weights": dict(entry.get("weights") or {}),
                "stageWeights": {str(i): dict(sw) for i, sw in
                                 (entry.get("stageWeights") or {}).items()},
                "steps": _sticky_int(entry.get("steps")),
                "stageSteps": {str(i): int(n) for i, n in
                               (entry.get("stageSteps") or {}).items()
                               if _sticky_int(n)},
            }
        else:  # legacy flat shape = behavior-level weights only
            out[bid] = {**empty_sticky(), "weights": dict(entry)}
    return out


def _sticky_int(v) -> int | None:
    """A stored step count, or None for anything unusable — a hand-edited or
    older file must not crash the panel."""
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _atomic_write_json(path: Path, obj, mode: int = 0o644) -> None:
    """The ONE json writer for every file this server persists (roster, sticky
    teach weights, clips, the HF token).

    tmp+replace so a crash mid-write cannot leave a truncated file in place,
    and the tmp name carries pid+uuid because a FIXED one is not per-writer:
    FastAPI runs sync handlers in a threadpool and the sim loop saves the
    roster too, so two overlapping saves interleaved their json into ONE inode
    and renamed the garbled result into place (lab-state.json then fails to
    parse; hf-token.json then reads as "not configured"). Creation mode is
    explicit rather than a chmod afterwards — a secret must never be
    world-readable for even the window between the two calls — and the scratch
    file is removed on any failure, since for the token it holds the secret and
    sits at a path .gitignore does not cover.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_teach_weights(w: dict[str, dict]) -> None:
    """Callers pass the normalized shape load_teach_weights returns."""
    _atomic_write_json(teach_weights_path(), w)


def save_lab_state(ducks: list[Duck]) -> None:
    state = {"version": 1, "ducks": [
        {"id": d.id, "label": d.label, "policy": d.policy_id,
         "onnxPath": d.onnx_path,
         # getattr: tests build rosters from bare namespaces without the flag
         "showcase": bool(getattr(d, "showcase", False))}
        for d in ducks
    ]}
    # ~11 call sites, one of them inside the sim loop — the writer that most
    # needs _atomic_write_json's per-call tmp name.
    _atomic_write_json(lab_state_path(), state)


def restore_ducks(path: Path) -> list[Duck]:
    """Rebuild the roster from lab-state.json. Training jobs die with the
    server, so a trainee/helper comes back frozen at its last live.onnx
    snapshot; entries whose brain can't be loaded any more are skipped."""
    state = json.loads(path.read_text())
    ducks: list[Duck] = []
    for i, entry in enumerate(state.get("ducks", [])):
        try:
            if entry.get("policy"):
                infer = load_policy_infer(entry["policy"])
            elif entry.get("onnxPath"):
                infer = _onnx_infer(Path(entry["onnxPath"]))
            else:
                raise ValueError("no brain recorded")
        except Exception as e:
            print(f"[lab] skipping {entry.get('id')} from {path.name}: "
                  f"{type(e).__name__}: {e}")
            continue
        run_path = entry.get("onnxPath")
        if not run_path and str(entry.get("policy", "")).startswith("run:"):
            e2 = next((p for p in discover_policies()
                       if p["id"] == entry["policy"]), None)
            run_path = e2["path"] if e2 else None
        # A showcase duck comes back showcasing — falling back to the plain
        # preview env when the behavior can't be resolved any more (the flag
        # then quietly drops rather than mislabeling the env).
        skw = showcase_env_kwargs(run_path) if entry.get("showcase") else None
        duck = Duck(str(entry["id"]), str(entry["label"]), infer, seed=i,
                    policy_id=entry.get("policy"),
                    onnx_path=entry.get("onnxPath"),
                    env_kwargs=(skw if skw is not None
                                else env_kwargs_for_policy_path(run_path)))
        duck.showcase = skw is not None
        ho = handoff_for(run_path) if skw is not None else None
        duck.handoff_infer, duck.handoff_label = ho if ho else (None, None)
        ducks.append(duck)
    return ducks


class HfTokenReq(BaseModel):
    """POST /settings/hf body: the user's own Hugging Face access token
    (BYOK). Needs write scope on their namespace for the Jobs API."""
    token: str


class TeachReq(BaseModel):
    text: str
    # Reference motion for an imitation run — a clip saved by the viewer's
    # timeline editor ("⚡ train this" sends the clip it has open).
    clip: str | None = None
    weights: dict[str, float] | None = None   # reward-weight overrides (sliders)
    # Per-stage overrides layered over `weights` (stage wins per key), keyed
    # by 1-based stage index as a string: {"1": {"legs_over": 4.0}}. Ignored
    # for single-run jobs (incl. initFrom fine-tunes).
    stageWeights: dict[str, dict[str, float]] | None = None
    # 1-based stage to START the chain at; earlier stages are skipped and the
    # first launched stage warm-starts from the newest matching prev-stage
    # run (resolve_stage_init). Only valid for staged behaviors, without
    # initFrom.
    startStage: int | None = None
    initFrom: str | None = None               # run name under runs/ to fine-tune
    # TOTAL practice budget for the whole job, in steps — the panel's "how
    # long should it practice?" control. None means "unchanged": the user's
    # last sticky choice for this behavior, or the recipe's declared budgets
    # when they have never set one. For a staged behavior this is the total
    # ACROSS the chain, split proportionally to the declared per-stage steps
    # — never a per-stage number.
    steps: int | None = None
    # Explicit per-stage budgets laid over that split ({"2": 3000000}), keyed
    # by 1-based stage index as a string. Staged behaviors only, like
    # stageWeights.
    stageSteps: dict[str, int] | None = None


class StageWeightsReq(BaseModel):
    """POST /teach/weights — live per-stage weight edits on the ACTIVE job.
    Same shape as TeachReq.stageWeights; sent in full (it replaces the map)."""
    stageWeights: dict[str, dict[str, float]]


class LoadRunReq(BaseModel):
    """POST /teach/load — seat a FINISHED run in the teach panel without
    starting anything: its recipe card, sliders and ✨ fine-tune target become
    that run's (TrainingJob.adopt). `policy` is a palette id ("run:<name>",
    "ckpt:<name>@Nk") or a bare run name under runs/. Refused while a job is
    actively training."""
    policy: str


def resolve_stage_init(behavior_id: str, start_stage: int) -> Path:
    """Warm start for a chain beginning at stage N>1: the most recently
    trained run dir matching teach-<behaviorId>-*-s{N-1} (any chain hash)
    whose snapshot artifacts exist — "most recent" by model.zip mtime, the
    newest actual training rather than newest mkdir. Raises ValueError with
    a client-facing message when the previous stage was never trained."""
    pattern = f"teach-{behavior_id}-*-s{start_stage - 1}"
    candidates = [d for d in RUNS_DIR.glob(pattern)
                  if d.is_dir() and (d / "model.zip").exists()
                  and (d / "vecnormalize.pkl").exists()]
    if not candidates:
        raise ValueError(
            f"starting at stage {start_stage} needs a trained stage "
            f"{start_stage - 1} to build on — no runs/{pattern} with a "
            "model.zip snapshot found; train the earlier stages first")
    return max(candidates, key=lambda d: (d / "model.zip").stat().st_mtime)


def resolve_init_from(name: str) -> Path:
    """Validate a /teach initFrom run name into a warm-startable run dir.
    Raises ValueError with a client-facing message."""
    # run_dir() is the ONE run-name validator (RUN_NAME_RE), shared with
    # DELETE /runs, /teach/load and the .onnx download. The hand-rolled
    # `Path(name).name != name` this used to do is strictly looser — it waved
    # through leading dots, spaces and unbounded length — and this name goes
    # straight onto a spawned trainer's --init-from.
    try:
        run = run_dir(name)
    except ValueError as e:
        raise ValueError(f"initFrom must be a plain run name under runs/, "
                         f"not {name!r} ({e})") from None
    if not run.is_dir():
        raise ValueError(f"initFrom run {name!r} does not exist under runs/")
    missing = [f for f in ("model.zip", "vecnormalize.pkl")
               if not (run / f).exists()]
    if missing:
        raise ValueError(f"initFrom run {name!r} has no {' / '.join(missing)} "
                         "yet — it can't be warm-started")
    return run


class LabState:
    def __init__(self, ducks: list[Duck]):
        self.ducks = ducks
        self.clients: set[WebSocket] = set()
        self.override: np.ndarray | None = None
        self.override_until = 0.0
        self.script_t = 0.0
        self.job: TrainingJob | None = None
        # Bounded: events are drained only when a client is attached, so a
        # headless lab (a long training chain with no browser open) grew
        # this forever while only the last few are ever sent.
        self.events: deque[str] = deque(maxlen=200)  # one-shot toast lines for the UI
        self.scaling = False  # a spawn/remove scale is in flight — hold others
        self.stats: dict = {}

    def duck(self, duck_id: str) -> Duck | None:
        return next((d for d in self.ducks if d.id == duck_id), None)

    def trainee(self) -> Duck | None:
        return self.duck("trainee")


# ------------------------------------------------------------ helper ducks

def helper_ducks(ducks: list[Duck]) -> list[Duck]:
    return [d for d in ducks if d.id.startswith("helper")]


def next_helper_slot(ducks: list[Duck]) -> int:
    """Smallest free helper number — reusing freed slots keeps ids stable for
    the UI when helpers come and go out of order."""
    used = {d.id for d in ducks}
    n = 1
    while f"helper{n}" in used:
        n += 1
    return n


def spawn_helper_error(st: LabState) -> str | None:
    """Why {"spawn_helper": true} can't be honored right now (None = go)."""
    if st.job is None or st.job.status != "training":
        return "no active training for a helper to join"
    if st.scaling:
        return "trainer is mid-restart — try again in a moment"
    if len(helper_ducks(st.ducks)) >= MAX_HELPERS:
        return f"helper cap reached ({MAX_HELPERS})"
    if not (st.job.dir / "model.zip").exists():
        return "helpers can join after the first training snapshot — moments away"
    return None


MAX_DUCKS = 20  # perf guard: each duck is a live CPU-MuJoCo env in the lab loop


def remove_duck_error(st: LabState, duck_id: str) -> str | None:
    """Why {"remove_duck": ...} can't be honored (None = go). ANY duck can be
    removed (declutter: the default checkpoint roster crowds the view during a
    training run) except the trainee mid-training — it's the run's only
    window. Helpers additionally wait out an in-flight trainer restart."""
    if st.duck(duck_id) is None:
        return f"no duck {duck_id}"
    if duck_id == "trainee" and st.job and st.job.status == "training":
        return "can't remove the trainee while it's training — stop the run first"
    if duck_id.startswith("helper") and st.scaling:
        return "trainer is mid-restart — try again in a moment"
    return None


def env_kwargs_for_behavior(b) -> dict:
    """Lab-preview physics matching a behavior's training env.

    `behavior_id` is the load-bearing key: without it `Duck._make_env` builds a
    plain MicroduckWalkEnv, and a trick policy then runs under the WALKING
    contract — the walk env resamples a random locomotion twist into the
    observation (at reset and every resample window), which the lab's
    `set_cmd(zeros)` cannot undo because the obs is already built. Measured on
    an assigned one_leg policy: 106 falls per 1500 steps in the walk env, 0 in
    its own env with the same weights. `standing_spawns` then holds the visible
    contract of a plain assign — a finished trick shows off from its feet, not
    dropped mid-roll (that is what the ✨ showcase assign is for) and not lying
    on the floor (`stand`'s 50% ground-spawn family).
    """
    kw: dict = {"behavior_id": b.id, "standing_spawns": True}
    if getattr(b, "scene", "walk") == "all":
        kw["scene_xml"] = str(C.SCENE_ALL_XML)
    if not getattr(b, "terminate_on_fall", True):
        kw["terminate_on_fall"] = False
    # Preview episodes as long as training ones — a 10 s lab default made
    # long-hold tricks look like they reset mid-pose.
    ep = getattr(b, "episode_s", None)
    if ep and ep > 10.0:
        kw["max_episode_s"] = float(ep)
    if getattr(b, "forward_cmd", 0.0):
        kw["height_termination"] = False
    return kw


def trainee_env_kwargs(b, stage_env: dict[str, str] | None = None) -> dict:
    """The 🎓 trainee's preview physics: the behavior's OWN env class (spawn
    families included) under the active curriculum stage's spawn knobs — the
    in-process mirror of what the trainer subprocess sees via its
    environment. Plain values only, so Duck.rebuild_env's kwargs-equality
    check treats a stage handoff (different knobs) as a real change.

    One deliberate divergence from the trainer: when a stage's spawn mix is
    dominated by one family (a focused rehearsal stage), the PREVIEW leans
    that family up to 85% — the trainer's exact mix includes spawns that are
    visually indistinguishable from a plain standing start, and a watcher
    concluded the mirroring was broken outright. The preview is a viewport;
    the stage inspector states the trainer's true percentages."""
    overrides = dict(stage_env or {})
    probs_env = overrides.get("MICRODUCK_SPAWN_FAMILY_PROBS")
    if probs_env:
        try:
            probs = [float(x) for x in probs_env.split(",")]
        except ValueError:
            probs = []
        # Only ever lean UP: a stage that already declares ~all of one family
        # (the stand-steady stage is 1.0/0.0) must not be watered down to 85%.
        if probs and 0.5 <= max(probs) < 0.85:
            rest = sum(p for p in probs if p != max(probs)) or 1.0
            overrides["MICRODUCK_SPAWN_FAMILY_PROBS"] = ",".join(
                "0.85" if p == max(probs) else f"{0.15 * p / rest:.3f}"
                for p in probs)
    kw = {**env_kwargs_for_behavior(b), "behavior_id": b.id,
          # The trainee/showcase preview mirrors the TRAINER's spawns, so it
          # drops the standing pin a plain assign carries.
          "standing_spawns": False,
          "spawn_overrides": overrides}
    # Trainee preview should see the same plant the trainer subprocess uses.
    # BAM cannot share a model with the rest of the roster, and _make_env
    # already falls back to a private compile when actuator is bam.
    if getattr(b, "forward_cmd", 0.0):
        kw["actuator"] = "bam"
    # PHYSICS knobs mirror per instance too, not just spawns. The lab process
    # runs with MICRODUCK_ACTUATOR=bam and no current scale, so before this a
    # stage declaring xml (the headstand ladder's training wheels) or bam@1.3
    # trained on those servos while the watched preview duck ran honest bam —
    # a policy that holds in training visibly crumples in the viewer, which is
    # exactly the "am I watching the trainer?" confusion this function exists
    # to prevent. actuator_force/bam_current_scale beat the process env.
    act = overrides.get("MICRODUCK_ACTUATOR")
    if act:
        kw["actuator_force"] = act
    cur = overrides.get("MICRODUCK_BAM_CURRENT_SCALE")
    if cur:
        try:
            kw["bam_current_scale"] = float(cur)
        except ValueError:
            pass
    if overrides.get("MICRODUCK_CLIP"):
        kw["clip_name"] = overrides["MICRODUCK_CLIP"]
    return kw


def on_stage_handoff(st: "LabState") -> None:
    """A curriculum stage just advanced: narrate it ("Training …" so the
    teach panel's event filter folds it into the chat) and re-mirror the
    trainee's preview env onto the NEW stage's spawn knobs — the trainer
    subprocess gets them via its environment, but the lab's in-process
    preview needs them per instance or it keeps rehearsing the old stage."""
    sp = st.job.stage_payload()
    st.events.append(
        f"Training stage {sp['idx']}/{sp['count']} — {sp['label']}")
    kw = trainee_env_kwargs(st.job.behavior, st.job.stage_env())
    trainee = st.trainee()
    if trainee is not None:
        trainee.rebuild_env(kw)
    for h in helper_ducks(st.ducks):  # helpers rehearse the same stage
        h.rebuild_env(kw)


def env_kwargs_for_policy_path(path: str | None) -> dict:
    """Same, derived from a run artifact's sibling behavior.json (assigned or
    restored teach-run policies)."""
    if not path:
        return {}
    bj = Path(path).parent / "behavior.json"
    try:
        behavior_id = json.loads(bj.read_text()).get("behavior")
        return env_kwargs_for_behavior(behaviors_mod.BEHAVIORS[behavior_id])
    except (OSError, KeyError, json.JSONDecodeError):
        return {}


def showcase_env_kwargs(path: str | None) -> dict | None:
    """Env kwargs for a showcase assign (the palette's chain-level "whole
    trick" chip). The FINAL stage's policy carries the whole curriculum, but
    from a standing start it attempts little — so rebuild the duck's env as
    the behavior's OWN env under the LAST curriculum stage's knobs, which is
    the "whole trick" stage by definition (behaviors own their knob names;
    the server never hardcodes them). None = no behavior/curriculum behind
    this policy; callers then treat the flag as a plain assign."""
    if not path:
        return None
    bj = Path(path).parent / "behavior.json"
    try:
        b = behaviors_mod.BEHAVIORS[json.loads(bj.read_text()).get("behavior")]
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    if not b.curriculum:
        return None
    if b.spotter_fn is not None:
        # Spotted showcase: START FROM STANDING (what a viewer asks to see —
        # "the walk pose, then the roll") and let the demo assist carry the
        # arc the actuators can't, releasing into the policy's own territory.
        # No mid-trick spawns needed; the trick plays start to finish.
        kw = trainee_env_kwargs(b, {**b.curriculum[-1].env,
                                    "MICRODUCK_SPAWN_FAMILY_PROBS": "0.0,0.0"})
        kw["spotter"] = True
        return kw
    # UNSPOTTED showcase (the headstand ladder is the first such trick: a
    # curriculum with no spotter_fn). This tail used to sit — unreachable —
    # after handoff_for's return, so every unspotted chain fell through to
    # None and do_assign silently downgraded the ✨ chip to a plain standing
    # assign with no handoff.
    env = dict(b.curriculum[-1].env)
    # Lean the spawn mix toward the trick's arc for VIEWING: the final stage
    # trains mostly from plain starts (right for training, but a policy that
    # rationally refuses a blocked entry then just stands there — a 12 s
    # showcase sample showed exactly that). Pre-scaling past 0.5 total also
    # triggers trainee_env_kwargs' dominant-family lean, so the net showcase
    # mix is 85% dominant family / 15% the rest / no idle standing starts —
    # exactly what a viewer should see. Knob names stay behavior-owned.
    probs_env = env.get("MICRODUCK_SPAWN_FAMILY_PROBS")
    if probs_env:
        try:
            probs = [float(x) for x in probs_env.split(",")]
            total = sum(probs)
            if 0 < total < 0.85:
                env["MICRODUCK_SPAWN_FAMILY_PROBS"] = ",".join(
                    f"{p * 0.85 / total:.3f}" for p in probs)
        except ValueError:
            pass
    return trainee_env_kwargs(b, env)


HANDOFF_POLICY = "pollen:alpha_stand"


def handoff_for(path: str | None):
    """The brain a finished trick hands control to — the robot's own pattern
    (policies hot-swap behind the shared 61-obs contract). A trick policy
    lands in a crouch it cannot rise from; alpha_stand rises from that exact
    pose and holds. Returns (infer, label) or None."""
    if not path:
        return None
    bj = Path(path).parent / "behavior.json"
    try:
        b = behaviors_mod.BEHAVIORS[json.loads(bj.read_text()).get("behavior")]
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    if not b.curriculum:
        return None
    # Only offer the hand-off if the duck can ever ASK for it. _handoff_due
    # tests env._bf_rot, the rotation accumulator that only the flip family's
    # state_fn advances — a headstand never sets it, so attaching alpha_stand
    # to that chain advertised "handoff: alpha_stand" in every frame for a
    # swap that could not happen.
    if getattr(b, "state_fn", None) is not behaviors_mod._bf_update:
        return None
    try:
        return load_policy_infer(HANDOFF_POLICY), HANDOFF_POLICY.split(":", 1)[-1]
    except Exception:
        return None


def showcase_label(policy_id: str, spotted: bool = False) -> str:
    """Roster label for a showcase duck: the CHAIN's name rather than one
    stage's run name, marked ✨ so it reads as "the whole trick" next to a
    single-stage assign's plain run-name label. A SPOTTED showcase says so
    permanently — a demo assist carries part of the trick, and a viewer must
    never mistake that for the policy doing it unaided."""
    name = policy_id.split(":", 1)[-1]
    m = _CHAIN_RE.match(name)
    chain = (m.group(1) if m else name).removeprefix("teach-")
    # "spotter-driven", not "spotted": measured honestly, the assist does most
    # of the roll (a LIMP duck reaches 291° under the same torque, and the
    # trick policy RESISTS tipping at lower torque — 59° vs limp's 134°).
    # Calling it "spotted" implied a gymnast doing the work with a hand
    # nearby; it is closer to the hand doing the work.
    return f"{chain} ✨ spotter-driven" if spotted else f"{chain} ✨"


_TRICK_DUCK_CACHE: dict[str, bool] = {}


def is_trick_duck(d: Duck) -> bool:
    """Trick policies trained on zero twist commands — the lab never sends
    them drive commands, and the UI surfaces them as non-steerable (a fully
    decluttered roster of trick ducks once made WASD look broken: every
    command was correctly ignored by everyone)."""
    if d.id == "trainee" or d.id.startswith("helper"):
        # trainee/helpers mirror the ACTIVE job; the loop already sends drive
        # commands to locomotion behaviors via the env.behavior check.
        return True
    pid = d.policy_id or ""
    if not pid.startswith("run:"):
        return False
    # Name-based classification called every teach-run a trick — including
    # LOCOMOTION runs, so a dragged-in run policy stood still at cmd (0,0,0)
    # and the user rightly asked why it didn't move. The run dir records what
    # it trained: behavior.json {"behavior": "run", ...}. Zero commands only
    # for behaviors that actually trained on zero twist (forward_cmd == 0).
    name = pid.split(":", 1)[1]
    # MEMOIZED: this is called once per duck per 50 Hz tick and twice more per
    # broadcast frame, and it used to read+parse behavior.json from disk every
    # time — ~100 synchronous file reads/second per assigned run duck, on the
    # event loop that also runs the sim and the WS broadcast. A run's recorded
    # behavior never changes once written, so the answer is cacheable for the
    # life of the process (a re-trained run keeps the same behavior id).
    cached = _TRICK_DUCK_CACHE.get(name)
    if cached is not None:
        return cached
    verdict = name.startswith("teach-")   # unknown: old conservative rule
    try:
        rec = json.loads((RUNS_DIR / name / "behavior.json").read_text())
        b = behaviors_mod.BEHAVIORS.get(rec.get("behavior", ""))
        if b is not None:
            verdict = not bool(getattr(b, "forward_cmd", 0.0))
        _TRICK_DUCK_CACHE[name] = verdict
    except (OSError, ValueError):
        # Cache the name-based fallback for anything that is not "the run dir
        # isn't there yet". A run that exists without a readable behavior.json
        # (train-walk and distill write none; a kill mid-write can truncate
        # one; a permission or not-a-directory error is permanent) would
        # otherwise re-attempt the failing open() ~100x/second per duck on the
        # event loop that also steps the sim — exactly the traffic this memo
        # exists to remove. The cache is dropped on every recipe reload and on
        # run deletion, which are the events that can change the answer.
        if (RUNS_DIR / name).is_dir():
            _TRICK_DUCK_CACHE[name] = verdict
    return verdict


def next_duck_slot(ducks: list[Duck]) -> int:
    """Smallest free d<n> number (mirrors next_helper_slot)."""
    used = {int(d.id[1:]) for d in ducks
            if d.id.startswith("d") and d.id[1:].isdigit()}
    n = 0
    while n in used:
        n += 1
    return n


def spawn_duck_error(st: LabState, policy_id: str | None) -> str | None:
    """Why {"spawn_duck": ...} can't be honored (None = go)."""
    if not policy_id:
        return "spawn_duck needs a policy id"
    if len(st.ducks) >= MAX_DUCKS:
        return f"lab is full ({MAX_DUCKS} ducks) — remove one first"
    return None


def extract_scene() -> dict:
    """Visual geometry for Three.js, straight from the compiled model
    (jenga-stacker's extract_visual_scene, deduplicated by mesh id)."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(C.SCENE_WALK_XML))
    mesh_ids: dict[int, int] = {}
    meshes: list[dict] = []
    geoms: list[dict] = []
    for i in range(m.ngeom):
        if m.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[i] != 2:
            continue
        mid = int(m.geom_dataid[i])
        if mid not in mesh_ids:
            va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
            mesh_ids[mid] = len(meshes)
            meshes.append({
                "v": np.round(m.mesh_vert[va:va + vn], 4).reshape(-1).tolist(),
                "f": m.mesh_face[fa:fa + fn].reshape(-1).tolist(),
            })
        # Material name + rgba ride along so the viewer can paint per-part
        # colors (eye ring, mouth, shells) instead of guessing per body.
        mat_id = int(m.geom_matid[i])
        rgba = m.mat_rgba[mat_id] if mat_id >= 0 else m.geom_rgba[i]
        geoms.append({
            "mesh": mesh_ids[mid],
            "body": int(m.geom_bodyid[i]),
            "pos": [round(float(x), 5) for x in m.geom_pos[i]],
            "quat": [round(float(x), 5) for x in m.geom_quat[i]],  # wxyz
            "mat": m.material(mat_id).name if mat_id >= 0 else "",
            "rgba": [round(float(x), 4) for x in rgba],
        })
    return {
        "bodies": [m.body(b).name for b in range(m.nbody)],
        "meshes": meshes,
        "geoms": geoms,
    }


# ------------------------------------------------- keyframe animation (🎬)
#
# The authoring half of motion imitation: pose the robot, key the poses, save
# a clip. The RL half (a reward that tracks a saved clip) reads the same JSON,
# so the on-disk format below is a CONTRACT — see the module docstring.
#
# SIGN CONVENTION, stated once: `rootPitch` is the right-handed rotation of the
# trunk about its +Y axis. NEGATIVE = lean BACK — the trunk's projected gravity
# acquires -x, which is the sim's own reading (gravity_x = sin(rootPitch), see
# contract.quat_rotate_inverse). Positive = nose-down / lean forward.

CLIP_VERSION = 1
CLIP_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")
MAX_CLIP_KEYS = 512
MAX_CLIP_DURATION_S = 120.0


class PoseReq(BaseModel):
    joints: list[float]
    rootPitch: float = 0.0
    # Drop the pose so its lowest point sits where the standing pose's does —
    # a crouch then plants its feet instead of floating with the trunk pinned.
    # Presentation only: the clip contract carries joints + rootPitch, and the
    # RL side owns the real height.
    ground: bool = True


class PoseScratch:
    """A model/data pair used ONLY to answer POST /pose.

    Deliberately NOT any live duck's env: the editor previews an arbitrary
    authored pose on every slider tick, and writing qpos into a duck mid-episode
    would corrupt the very rollout the viewer is streaming. One extra MjData on
    a 16-body model costs nothing, and mj_forward here measures ~0.03 ms.
    """

    def __init__(self) -> None:
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(C.SCENE_WALK_XML))
        self.data = mujoco.MjData(self.model)
        self.joint_qpos_adr = np.array(
            [self.model.joint(n).qposadr[0] for n in C.JOINT_NAMES])
        self.limits = np.array(
            [self.model.jnt_range[self.model.joint(n).id] for n in C.JOINT_NAMES],
            dtype=np.float64)
        # Geoms that can touch the floor, for the grounding offset below —
        # world-body geoms (the floor plane itself) are not part of the robot.
        self.robot_geoms = np.array(
            [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] != 0])
        # Root pose of the STAND keyframe: the preview duck stands where the
        # keyframe puts it, so an authored pose reads against the same ground.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("STAND").id)
        mujoco.mj_forward(self.model, self.data)
        self.base_qpos = self.data.qpos.copy()
        # Grounding is measured RELATIVE to the standing pose, so the AABB
        # bound's conservatism cancels exactly at DEFAULT_POSE instead of
        # leaving the preview duck hovering a few mm off the floor.
        self.stand_low_z = self._lowest_z()

    def _lowest_z(self) -> float:
        """World z of the lowest point of the robot's geom AABBs (conservative
        — see stand_low_z for why that is fine)."""
        m, d = self.model, self.data
        g = self.robot_geoms
        rot = d.geom_xmat[g].reshape(-1, 3, 3)
        aabb = m.geom_aabb[g]                       # [cx cy cz hx hy hz], geom frame
        zrow = rot[:, 2, :]                         # world-z row of each geom's basis
        centers = d.geom_xpos[g, 2] + np.einsum("ij,ij->i", zrow, aabb[:, :3])
        extents = np.einsum("ij,ij->i", np.abs(zrow), aabb[:, 3:])
        return float(np.min(centers - extents))

    def clamp(self, joints: np.ndarray) -> np.ndarray:
        return np.clip(joints, self.limits[:, 0], self.limits[:, 1])

    def solve(self, joints: np.ndarray, root_pitch: float = 0.0,
              ground: bool = True) -> list[list[float]]:
        """Forward kinematics for an authored pose → the WS frame's body payload."""
        m, d = self.model, self.data
        d.qpos[:] = self.base_qpos
        d.qvel[:] = 0.0
        d.qpos[self.joint_qpos_adr] = joints
        half = float(root_pitch) / 2.0
        # Rotation about +Y (wxyz) — the sign convention documented above.
        d.qpos[3:7] = [np.cos(half), 0.0, np.sin(half), 0.0]
        self.mj.mj_forward(m, d)
        if ground:
            drop = self._lowest_z() - self.stand_low_z
            if abs(drop) > 1e-6:
                d.qpos[2] -= drop
                self.mj.mj_forward(m, d)
        return [[round(float(v), 4) for v in (*d.xpos[b], *d.xquat[b])]
                for b in range(m.nbody)]

    def meta(self) -> dict:
        """Everything the editor needs to build clamped controls and map a
        clicked body back to the joint that moves it."""
        m = self.model
        groups = ("left leg",) * 5 + ("head + neck",) * 4 + ("right leg",) * 5
        joints = []
        for i, name in enumerate(C.JOINT_NAMES):
            j = m.joint(name)
            body = int(m.jnt_bodyid[j.id])
            joints.append({
                "index": i,
                "name": name,
                "group": groups[i],
                "min": round(float(self.limits[i, 0]), 6),
                "max": round(float(self.limits[i, 1]), 6),
                "default": round(float(C.DEFAULT_POSE[i]), 6),
                "body": body,
                "bodyName": m.body(body).name,
                # Hinge axis and anchor in the BODY frame — the viewer turns a
                # screen drag into a joint delta with these.
                "axis": [round(float(v), 6) for v in m.jnt_axis[j.id]],
                "pos": [round(float(v), 6) for v in m.jnt_pos[j.id]],
            })
        return {
            "joints": joints,
            "bodies": [m.body(b).name for b in range(m.nbody)],
            "trunkBody": int(self.mj.mj_name2id(
                m, self.mj.mjtObj.mjOBJ_BODY, "trunk_base")),
            # An EDITOR hint, not a validation bound: a flip is a continuous
            # rotation, so a clip may legitimately carry a full turn (the
            # backflip recipe runs to ±2π) and the clip contract never clamps
            # rootPitch. Two turns of slider travel covers either direction.
            "rootPitchRange": [-round(float(2 * np.pi), 6), round(float(2 * np.pi), 6)],
            # Restated here so a client never has to guess (docs own the why).
            "rootPitchSign": "negative = lean back (gravity gains -x in trunk frame)",
        }


_pose_scratch: PoseScratch | None = None


def pose_scratch() -> PoseScratch:
    """Lazily built singleton — the editor is optional, so a lab that never
    opens it never pays for the extra model."""
    global _pose_scratch
    if _pose_scratch is None:
        _pose_scratch = PoseScratch()
    return _pose_scratch


def clips_dir() -> Path:
    """Clip store, resolved per call so MICRODUCK_CLIPS_DIR set after import
    (tests, a scratch server on another port) still wins."""
    env = os.environ.get("MICRODUCK_CLIPS_DIR")
    return Path(env) if env else CLIPS_DIR


def _finite(x) -> float:
    """A float that is safe to write into the contract, or ValueError."""
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("not a finite number")
    return v


def clean_joints(raw) -> list[float]:
    """14 finite floats in JOINT_NAMES order, CLAMPED to the MJCF limits.

    Clamping rather than rejecting: an out-of-range angle is unreachable on the
    real servo, so the honest fix is the nearest reachable one — and the caller
    gets the clamped values back so its UI can show what actually happened."""
    if not isinstance(raw, (list, tuple)) or len(raw) != C.NUM_JOINTS:
        raise ValueError(f"joints must be {C.NUM_JOINTS} numbers, "
                         f"got {len(raw) if hasattr(raw, '__len__') else type(raw).__name__}")
    vals = np.array([_finite(v) for v in raw], dtype=np.float64)
    return [round(float(v), 6) for v in pose_scratch().clamp(vals)]


def clean_clip(name: str, raw: dict) -> dict:
    """Validate + normalize a clip against the v1 contract, or ValueError.

    Rejects anything the RL resampler could silently misread (missing t=0,
    out-of-order keys, a duration that would truncate the last key); clamps
    only what has one obvious right answer (joint angles → servo limits)."""
    if not isinstance(raw, dict):
        raise ValueError("clip must be an object")
    keys_raw = raw.get("keys")
    if not isinstance(keys_raw, list) or not keys_raw:
        raise ValueError("clip needs at least one key")
    if len(keys_raw) > MAX_CLIP_KEYS:
        raise ValueError(f"too many keys (max {MAX_CLIP_KEYS})")
    keys: list[dict] = []
    prev_t = None
    for i, k in enumerate(keys_raw):
        if not isinstance(k, dict):
            raise ValueError(f"key {i} must be an object")
        try:
            t = round(_finite(k.get("t", 0.0)), 6)
        except (TypeError, ValueError):
            raise ValueError(f"key {i}: t must be a finite number")
        if t < 0:
            raise ValueError(f"key {i}: t must be >= 0")
        if i == 0 and t != 0.0:
            raise ValueError("the first key must be at t = 0")
        if prev_t is not None and t <= prev_t:
            raise ValueError(f"key {i}: times must ascend (got {t} after {prev_t})")
        prev_t = t
        keys.append({
            "t": t,
            "joints": clean_joints(k.get("joints")),
            "rootPitch": round(_finite(k.get("rootPitch", 0.0) or 0.0), 6),
        })
    duration = round(_finite(raw.get("duration", keys[-1]["t"])), 6)
    if duration <= 0:
        raise ValueError("duration must be > 0")
    if duration > MAX_CLIP_DURATION_S:
        raise ValueError(f"duration must be <= {MAX_CLIP_DURATION_S} s")
    if duration < keys[-1]["t"]:
        raise ValueError(
            f"duration {duration}s would cut off the last key at {keys[-1]['t']}s")
    return {
        "version": CLIP_VERSION,
        "name": name,
        "duration": duration,
        "loop": bool(raw.get("loop", False)),
        "keys": keys,
    }


def clip_path(name: str) -> Path:
    """Path for a clip name, or ValueError. The charset is restrictive on
    purpose — the name comes off the wire and becomes a filename."""
    if not isinstance(name, str) or not CLIP_NAME_RE.match(name):
        raise ValueError("clip name must be 1-64 chars of letters, digits, "
                         "space, dot, dash or underscore, starting alphanumeric")
    return clips_dir() / f"{name}.json"


def save_clip(name: str, clip: dict) -> None:
    _atomic_write_json(clip_path(name), clip)


def load_clips() -> list[dict]:
    """Every readable clip, newest first (the runs/ convention). Unparseable
    files are skipped rather than failing the whole listing."""
    d = clips_dir()
    if not d.exists():
        return []
    out = []
    for f in sorted(d.glob("*.json")):
        try:
            clip = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(clip, dict):
            continue
        clip.setdefault("name", f.stem)
        clip["modified"] = round(f.stat().st_mtime, 3)
        out.append(clip)
    out.sort(key=lambda c: c.get("modified", 0.0), reverse=True)
    return out


# --------------------------------------------------------------------------
# 🎥 screen captures — the viewer records its canvas with MediaRecorder and
# uploads the take here; ffmpeg (imageio-ffmpeg's bundled binary, same as
# render_rollout) turns it into a shareable mp4 + palette gif in captures/.

CAPTURE_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")
CAPTURE_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.(mp4|gif)$")
CAPTURE_MAX_BYTES = 300 * 1024 * 1024
CAPTURE_GIF_WIDTH = 480


def captures_dir() -> Path:
    """Capture store, resolved per call (clips_dir convention) so
    MICRODUCK_CAPTURES_DIR set after import still wins."""
    env = os.environ.get("MICRODUCK_CAPTURES_DIR")
    return Path(env) if env else CAPTURES_DIR


def capture_slug(name: str) -> str:
    """Duck names carry emoji/spaces — reduce to a safe filename stem."""
    slug = CAPTURE_SLUG_RE.sub("-", name).strip("-").lower()[:40].strip("-")
    # CAPTURE_FILE_RE (what GET /captures/{fname} will accept) demands an
    # alphanumeric first character, but the slug charset keeps "_" — a duck
    # named "_experimental" produced a file the panel's own download links
    # then refused to serve with 422.
    slug = slug.lstrip("_-")
    return slug or "duck"


def capture_base(name: str) -> str:
    """Unique stem for a new capture: slug + timestamp (+ -2, -3… on a
    same-second collision, so a quick retake never overwrites the last one)."""
    stem = f"{capture_slug(name)}-{time.strftime('%Y%m%d-%H%M%S')}"
    d = captures_dir()
    base, n = stem, 2
    while (d / f"{base}.mp4").exists() or (d / f"{base}.gif").exists():
        base = f"{stem}-{n}"
        n += 1
    return base


def _ffmpeg(*args: str, timeout: float = 180.0) -> None:
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run([exe, "-y", "-hide_banner", "-loglevel", "error", *args],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError("; ".join(tail) or f"ffmpeg exited {proc.returncode}")


def convert_capture(src: Path, base: str) -> dict:
    """Browser upload → captures/<base>.mp4 + .gif. Blocking — call it in a
    thread. ffmpeg sniffs the container, so whatever MediaRecorder produced
    (webm on Chrome/Firefox, mp4 on Safari) works unchanged."""
    d = captures_dir()
    mp4, gif = d / f"{base}.mp4", d / f"{base}.gif"
    # h264 yuv420p rejects odd dimensions and browser canvases often are —
    # the same trap render_rollout hit; crop-to-even instead of failing.
    _ffmpeg("-i", str(src), "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            "-movflags", "+faststart", str(mp4))
    # gif from the cleaned mp4: two-pass palette at a README-friendly width.
    # (Plain palettegen/paletteuse only — imageio-ffmpeg's bundled binary is
    # an older build without stat_mode/diff_mode.)
    _ffmpeg("-i", str(mp4), "-filter_complex",
            f"[0:v] fps=15,scale={CAPTURE_GIF_WIDTH}:-1:flags=lanczos,"
            "split [a][b];[a] palettegen [p];"
            "[b][p] paletteuse=dither=bayer:bayer_scale=5",
            str(gif))
    return {
        "name": base,
        "mp4": f"/captures/{mp4.name}", "mp4Kb": mp4.stat().st_size // 1024,
        "gif": f"/captures/{gif.name}", "gifKb": gif.stat().st_size // 1024,
        "dir": str(d),
    }


# --------------------------------------------------------------------------
# Front-door origin policy — ONE definition, three enforcement points.
#
# The lab binds 127.0.0.1, so "same machine" is the whole legitimate surface
# and every legitimate browser origin is a loopback one (the viewer is served
# from localhost on a dev port that varies, and may be pointed at this lab via
# ?lab=host:port). Non-browser clients — curl, the CLI, the tests — send no
# Origin at all; that is not a forgeable state, so it passes.
LOCAL_ORIGIN_RE = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")

# CORS-safelisted request content types. A cross-origin POST carrying one of
# these is a "simple" request: the browser dispatches it with NO preflight, so
# CORSMiddleware never gets the chance to refuse it and the side effect lands
# even though the response is opaque to the attacker.
CORS_SIMPLE_CONTENT_TYPES = frozenset({
    "text/plain", "application/x-www-form-urlencoded", "multipart/form-data",
})


def origin_allowed(origin: str | None) -> bool:
    """May a request carrying this Origin act on the lab? Absent = a
    non-browser client (curl/python/the CLI), which keeps working; browsers
    always stamp Origin on a WebSocket handshake and on any cross-origin
    fetch, so an absent header cannot be a forged cross-site request.

    fullmatch, not match, because sharing LOCAL_ORIGIN_RE is not by itself
    enough to keep this in step with CORSMiddleware: Starlette calls
    fullmatch, and Python's `$` also matches just BEFORE a trailing newline,
    so `.match()` here accepted "http://localhost\\n" where the middleware
    refused it. One pattern, two verdicts — the drift the shared constant
    was meant to prevent, hiding in the call instead of the regex."""
    return origin is None or bool(LOCAL_ORIGIN_RE.fullmatch(origin))


def make_app(ducks: list[Duck]):
    scene = extract_scene()
    st = LabState(ducks)
    stats = StatsSampler()
    st.stats = stats.sample(None)  # frames carry the full stats shape from #1

    @asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(lab_loop())
        # Nothing else ever retrieves this task's exception, so before this
        # callback a crash in lab_loop killed every duck SILENTLY: HTTP and
        # the WebSocket handshake kept working, the viewer kept its green
        # "live" badge, and zero frames arrived. That is precisely how a
        # set-mutation race in the broadcast went unnoticed. Log it loudly.
        def _loop_died(t: asyncio.Task) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                traceback.print_exception(type(exc), exc, exc.__traceback__)
                print("[lab] FATAL: the duck loop stopped — no frames will be "
                      "sent. Restart the lab.", flush=True)

        task.add_done_callback(_loop_died)
        yield
        task.cancel()
        if st.job:
            st.job.stop()

    app = FastAPI(title="Duck lab", lifespan=lifespan)
    app.state.lab = st  # the roster/job the handlers close over, for tests
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    # LOCAL origins only. This was allow_origins=["*"], which was harmless
    # while every route was a read or a duck nudge — but the lab now owns
    # DELETE /runs/{name} (rmtree of a whole training chain) and the BYOK
    # Hugging Face token. With a wildcard, ANY page the user browses while
    # the lab runs could enumerate /policies and delete every run, or swap
    # the stored token, from the user's own machine. Both of those are
    # preflighted requests (DELETE; POST with application/json), so an
    # origin allowlist is what actually stops them.
    # The viewer is served from localhost (63317 by default, any port in a
    # dev setup) and may be pointed at this lab via ?lab=host:port, so the
    # allowlist is "any localhost origin", expressed as a regex because the
    # port varies. Non-browser clients (curl, the CLI) are unaffected —
    # CORS only ever constrained browsers.
    app.add_middleware(
        CORSMiddleware,
        # LOOPBACK ONLY. An earlier revision also admitted RFC1918 and
        # *.local origins so the "Network: http://192.168.x.x:63317" URL that
        # `next dev` prints would work — on the reasoning that such an origin
        # is the same machine. That reasoning is wrong: a page served by ANY
        # other host on the LAN (a neighbour's laptop, a router admin UI, a
        # captive portal) is a genuine 192.168 origin, forges nothing, and
        # would have been handed DELETE /runs and the HF token routes. The
        # lab binds 127.0.0.1, so loopback is the whole legitimate surface;
        # open the viewer at http://localhost:63317, not the Network URL.
        # The pattern is LOCAL_ORIGIN_RE, not a copy: /ws and POST /captures
        # enforce the same rule outside CORS, and a second literal here would
        # let the two drift apart on the next port-scheme change.
        allow_origin_regex=LOCAL_ORIGIN_RE.pattern,
        allow_methods=["*"], allow_headers=["*"])

    @app.get("/scene")
    def get_scene() -> dict:
        return scene

    @app.get("/policies")
    def get_policies() -> dict:
        return {"policies": discover_policies()}

    @app.delete("/runs/{name}")
    def delete_run(name: str, chain: bool = False) -> dict:
        """Permanently delete a training run (or a whole curriculum chain with
        ?chain=true): policy.onnx, checkpoints, progress log, the lot. The
        viewer confirms first — this endpoint is the point of no return."""
        names = chain_run_names(name) if chain else [name]
        try:
            result = delete_runs(names, st)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except PermissionError as e:
            raise HTTPException(409, str(e))
        except FileNotFoundError:
            raise HTTPException(404, f"no run named \u201c{name}\u201d")
        except OSError as e:
            raise HTTPException(500, f"could not delete \u201c{name}\u201d: {e}")
        # Drop a teach card that was showing one of these runs. A finished run
        # seated by /teach/load is adopted as status "done", so it is NOT
        # protected by training_run_names (nor should it be \u2014 finished runs are
        # ordinary data): without this the panel kept streaming a card for a
        # deleted dir, and its \u2728 fine-tune launched against nothing.
        for gone in result["deleted"]:
            _TRICK_DUCK_CACHE.pop(gone, None)
        if st.job is not None and st.job.run_name in set(result["deleted"]):
            owned = st.job.owns_preview_ducks
            st.job = None
            # ...and its preview ducks go with it — but ONLY if this card
            # actually built them. A seated card (teach/load, fired by mere
            # duck selection) may be showing run B while the trainee and
            # helpers on the pitch belong to run A and still preview it; those
            # must survive B's deletion. Same ownership rule teach_clear uses.
            if owned:
                kept = [d for d in st.ducks
                        if d.id != "trainee" and not d.id.startswith("helper")]
                if len(kept) != len(st.ducks):
                    st.ducks = kept
                    save_lab_state(st.ducks)
        mb = result["freedBytes"] / 1e6
        st.events.append(
            f"\U0001f5d1 deleted {name}"
            + (f" ({len(result['deleted'])} stages)" if chain and len(result["deleted"]) > 1 else "")
            + f" \u2014 {mb:.0f} MB freed")
        return result

    @app.get("/runs/{name}/policy.onnx")
    def download_policy(name: str) -> FileResponse:
        """The deployable brain for a run, one click from the policies panel.

        Serves policy.onnx (the export with the obs normalizer baked in \u2014
        the only artifact worth handing anyone; the playbook forbids raw
        checkpoints), falling back to the newest live.onnx snapshot while
        the run is still training. Same restrictive name rule as DELETE:
        the string picks a directory, so it must not be able to climb."""
        try:
            d = run_dir(name)
        except ValueError as e:
            raise HTTPException(422, str(e))
        for fname in ("policy.onnx", "live.onnx"):
            p = d / fname
            if p.exists():
                return FileResponse(p, media_type="application/octet-stream",
                                    filename=f"{name}.onnx")
        raise HTTPException(404, f"no exported policy in \u201c{name}\u201d yet")

    # ---- \u2699 settings: BYOK Hugging Face token -----------------------------
    # The browser never sees the token again after POSTing it: GET returns a
    # mask + the username captured at validation time, and the file sits
    # 0600 + gitignored beside runs/. The token's job is the REAL training
    # step \u2014 launching microduck_rl on HF Jobs GPUs under the user's account.

    @app.get("/settings/hf")
    def hf_settings() -> dict:
        d = load_hf_token()
        if not d:
            return {"configured": False}
        return {"configured": True, "username": d.get("username", ""),
                "masked": _hf_mask(d["token"])}

    @app.post("/settings/hf")
    def hf_settings_save(req: HfTokenReq) -> dict:
        tok = req.token.strip()
        if not tok:
            raise HTTPException(422, "empty token")
        # Validate BEFORE persisting: whoami() is the cheapest call that
        # proves the token is real, and its username is worth keeping.
        try:
            from huggingface_hub import HfApi
            who = HfApi(token=tok).whoami()
        except Exception as e:
            raise HTTPException(401, f"Hugging Face rejected that token: {e}")
        username = who.get("name", "") if isinstance(who, dict) else ""
        # 0600 FROM CREATION, not write_text()+chmod: that left the token
        # world-readable under the default umask for the window between the
        # two calls — and permanently if the process died in between. The rest
        # of the write (per-call tmp name, atomic rename, and the unlink that
        # keeps a live token out of a stray scratch file at a path .gitignore
        # does not cover) is _atomic_write_json's job, shared with the roster,
        # the sticky weights and the clips.
        _atomic_write_json(HF_TOKEN_PATH, {"token": tok, "username": username},
                           mode=0o600)
        st.events.append(f"\ud83e\udd17 Hugging Face connected as {username}")
        return {"configured": True, "username": username,
                "masked": _hf_mask(tok)}

    @app.delete("/settings/hf")
    def hf_settings_delete() -> dict:
        HF_TOKEN_PATH.unlink(missing_ok=True)
        # Scratch files from interrupted saves hold the same secret. Two
        # shapes exist: the current "<name>.<pid>.<hex>.tmp" and the fixed
        # "<name>.tmp" an earlier build wrote — a glob with `.*.` misses the
        # latter, and someone upgrading may still have one on disk.
        for pat in (HF_TOKEN_PATH.name + ".tmp", HF_TOKEN_PATH.name + ".*.tmp"):
            for stray in HF_TOKEN_PATH.parent.glob(pat):
                stray.unlink(missing_ok=True)
        return {"configured": False}

    @app.get("/behaviors")
    def get_behaviors() -> dict:
        return {"behaviors": [behaviors_mod.behavior_card(b)
                              for b in behaviors_mod.BEHAVIORS.values()]}

    # ------------------------------------------------ 🎬 animation authoring

    @app.get("/joints")
    def get_joints() -> dict:
        return pose_scratch().meta()

    @app.post("/pose")
    def post_pose(req: PoseReq) -> dict:
        try:
            joints = clean_joints(req.joints)
        except ValueError as e:
            raise HTTPException(422, str(e))
        try:
            pitch = _finite(req.rootPitch)
        except ValueError:
            raise HTTPException(422, "rootPitch must be a finite number")
        return {
            "bodies": pose_scratch().solve(np.array(joints), pitch, req.ground),
            "joints": joints,       # clamped — the editor snaps its sliders to these
            "rootPitch": round(pitch, 6),
        }

    @app.get("/clips")
    def get_clips() -> dict:
        return {"clips": load_clips()}

    @app.get("/clips/{name}")
    def get_clip(name: str) -> dict:
        try:
            path = clip_path(name)
        except ValueError as e:
            raise HTTPException(422, str(e))
        if not path.exists():
            raise HTTPException(404, f"no clip named “{name}”")
        try:
            return json.loads(path.read_text())
        except ValueError as e:
            raise HTTPException(500, f"clip “{name}” is unreadable: {e}")

    @app.put("/clips/{name}")
    def put_clip(name: str, body: dict) -> dict:
        try:
            path = clip_path(name)             # name validity first…
            clip = clean_clip(name, body)      # …then the contract itself
        except ValueError as e:
            raise HTTPException(422, str(e))
        save_clip(name, clip)
        clip = dict(clip)
        clip["modified"] = round(path.stat().st_mtime, 3)
        return clip

    @app.delete("/clips/{name}")
    def delete_clip(name: str) -> dict:
        try:
            path = clip_path(name)
        except ValueError as e:
            raise HTTPException(422, str(e))
        if not path.exists():
            raise HTTPException(404, f"no clip named “{name}”")
        path.unlink()
        return {"deleted": name}

    # ------------------------------------------------ 🎥 screen captures

    @app.post("/captures")
    async def post_capture(req: Request, name: str = "duck") -> dict:
        # This is the one route that reads a raw body with no schema, and it
        # writes up to CAPTURE_MAX_BYTES to disk and spawns ffmpeg. With a
        # `Content-Type: text/plain` body it was a CORS-SIMPLE request: sent
        # cross-origin with no preflight at all, so CORSMiddleware never got a
        # veto and the side effect landed — the attacker cannot read the
        # opaque reply, but the disk fills and the process spawns anyway.
        # Two doors, because neither covers the other: Origin catches a
        # typeless blob (also un-preflighted), and the content-type rule holds
        # even if a client somehow suppresses Origin.
        if not origin_allowed(req.headers.get("origin")):
            raise HTTPException(403, "capture uploads are local-only")
        ctype = req.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype in CORS_SIMPLE_CONTENT_TYPES:
            # The viewer posts the MediaRecorder blob itself, so fetch stamps
            # video/webm (Chrome/Firefox) or video/mp4 (Safari); curl and the
            # tests send no Content-Type. Neither shape is refused here.
            raise HTTPException(415, f"upload the recording as a video blob, not {ctype}")
        data = await req.body()
        if not data:
            raise HTTPException(422, "empty capture upload")
        if len(data) > CAPTURE_MAX_BYTES:
            raise HTTPException(413, "capture too large — keep takes under a "
                                     "few minutes")
        d = captures_dir()
        d.mkdir(parents=True, exist_ok=True)
        base = capture_base(name)
        # Extension is cosmetic — ffmpeg sniffs the actual container.
        src = d / f"{base}.upload"
        src.write_bytes(data)
        try:
            return await asyncio.to_thread(convert_capture, src, base)
        except Exception as e:
            raise HTTPException(500, f"capture conversion failed: {e}")
        finally:
            src.unlink(missing_ok=True)

    @app.get("/captures/{fname}")
    def get_capture(fname: str) -> FileResponse:
        if not CAPTURE_FILE_RE.match(fname):
            raise HTTPException(422, "bad capture filename")
        path = captures_dir() / fname
        if not path.exists():
            raise HTTPException(404, f"no capture named “{fname}”")
        # filename= sets Content-Disposition, so the viewer's ⬇ buttons save
        # straight to the user's downloads with a sensible name.
        return FileResponse(path, filename=fname)

    @app.post("/teach")
    async def teach(req: TeachReq) -> dict:
        if st.job and st.job.status == "training":
            return {"matched": False,
                    "message": f"Already teaching “{st.job.display_title()}” — stop it first.",
                    "busy": True}
        # Pick up recipe edits without a server restart: the training
        # subprocess always imports behaviors.py fresh, so reloading here keeps
        # the card/sliders in step with what the run will actually train.
        # Reload the modules behaviors.py DEPENDS on first: reloading only
        # behaviors leaves it calling into a stale `motion`, which crashed
        # every preview duck the moment a clip method was added (the trainer
        # subprocess was fine — it imports everything fresh — so training ran
        # while the scene sat empty).
        # reload_self(), not importlib.reload: the latter re-executes in the
        # LIVE dict, so a clip edit that compiles but raises part-way left the
        # lab holding half of each version with no error anywhere. Same
        # all-or-nothing machinery reload_library uses.
        try:
            motion_mod.reload_self()
        except Exception as e:
            # Same contract as reload_library below: an edit to the clip
            # machinery that raises must report itself rather than 500 the
            # panel with no message in the chat.
            return {"matched": False,
                    "message": f"motion.py didn't load: {e}"}
        # behaviors is a PACKAGE now (one module per trick):
        # reload()ing it would only re-run __init__ and keep every
        # submodule stale — reload_library() re-imports them all in
        # registration order and re-flattens the namespace.
        try:
            behaviors_mod.reload_library()
        except Exception as e:
            # A recipe edit that compiles but raises at import: the library
            # rolled itself back, so say what happened in the chat instead of
            # handing the panel a bare 500.
            return {"matched": False,
                    "message": f"behaviors.py didn't load: {e}"}
        _TRICK_DUCK_CACHE.clear()   # verdicts derive from the reloaded recipes
        b = behaviors_mod.match_behavior(req.text)
        if b is None:
            return {"matched": False,
                    "message": "I don't know that trick yet. I can teach these — "
                               "new tricks need a reward recipe added to the "
                               "behaviors/ package:",
                    "behaviors": [behaviors_mod.behavior_card(x)
                                  for x in behaviors_mod.BEHAVIORS.values()]}
        # The clip rides MICRODUCK_CLIP into the trainer AND (via stage_env)
        # into the preview envs, where motion.load_clip joins it straight onto
        # clips_dir() — so it gets the same restrictive validation every clip
        # endpoint uses, and must actually exist. Unchecked, "../x" read JSON
        # outside clips/, and a typo'd name reported a healthy job whose
        # workers all died at env construction with FileNotFoundError.
        if req.clip:
            try:
                if not clip_path(req.clip).exists():
                    return {"matched": False,
                            "message": f"I don't have a clip named “{req.clip}” — "
                                       "save it in the 🎬 animate panel first."}
            except ValueError as e:
                return {"matched": False, "message": str(e)}
        init_from = None
        if req.initFrom:
            try:
                init_from = resolve_init_from(req.initFrom)
            except ValueError as e:
                return {"matched": False, "message": str(e)}
        # startStage: begin the chain partway, warm-started from the newest
        # existing prev-stage run. Refused (with the reason) rather than
        # silently reinterpreted when it can't mean anything — the panel
        # surfaces these messages in the chat log.
        start_stage = req.startStage or 1
        stage_init = None
        # initFrom + startStage = run the CHAIN, warm-started from that brain
        # (initFrom alone keeps its fine-tune meaning: one run at the final
        # stage). This is how a trick inherits a skill it needs but never
        # learns on its own — the backflip could not hold a stand after 1M
        # dedicated steps, while the one-leg and crouch policies hold one for
        # a full episode, and they are checkpoints in the same format.
        chain_from = None
        if req.startStage is not None and init_from is not None:
            chain_from, init_from = init_from, None
        if req.startStage is not None:
            if not b.curriculum:
                return {"matched": False,
                        "message": f"“{b.title}” trains as a single run — "
                                   "there are no stages to start from"}
            if not 1 <= start_stage <= len(b.curriculum):
                return {"matched": False,
                        "message": f"startStage must be between 1 and "
                                   f"{len(b.curriculum)} for “{b.title}”"}
        if chain_from is not None:
            stage_init = chain_from
        elif start_stage > 1:
            try:
                stage_init = resolve_stage_init(b.id, start_stage)
            except ValueError as e:
                return {"matched": False, "message": str(e)}
        # Sticky sliders: no weights in the request means "same as I had it",
        # not "back to defaults" — inherit this behavior's last-used settings
        # (both layers: behavior-level and per-stage). Explicit values
        # (retrain/fine-tune, or a scripted call) win and become the new
        # sticky set.
        sticky = load_teach_weights()
        prev_sticky = sticky.get(b.id, empty_sticky())
        weights = (req.weights if req.weights is not None
                   else prev_sticky["weights"] or None)
        stage_weights = (req.stageWeights if req.stageWeights is not None
                         else prev_sticky["stageWeights"] or None)
        # Same "same as I had it" rule for the practice budget.
        budget = req.steps if req.steps is not None else prev_sticky["steps"]
        stage_budgets = (req.stageSteps if req.stageSteps is not None
                         else prev_sticky["stageSteps"] or None)
        # TEACH_STEPS_OVERRIDE / TEACH_SNAP_OVERRIDE: testing knobs so probes
        # can run tiny jobs without touching the behavior library's budgets.
        steps = os.environ.get("TEACH_STEPS_OVERRIDE")
        snap = os.environ.get("TEACH_SNAP_OVERRIDE")
        st.job = TrainingJob(
            b.id,
            # Helpers already on the lab pitch in from step one.
            helpers=len(helper_ducks(st.ducks)),
            steps=int(steps) if steps else None,
            snap_steps=int(snap) if snap else None,
            weights=weights,
            init_from=init_from,
            stage_weights=stage_weights,
            start_stage=start_stage,
            stage_init_from=stage_init,
            extra_env=({"MICRODUCK_CLIP": req.clip} if req.clip else None),
            budget=budget,
            stage_budgets=stage_budgets,
        )
        entry = {"weights": st.job.weights,
                 "stageWeights": prev_sticky["stageWeights"],
                 # job.budget is None while TEACH_STEPS_OVERRIDE shrinks the
                 # job — a probe's 1k must never become the user's saved
                 # budget, so the previous choice stands.
                 "steps": st.job.budget or prev_sticky["steps"],
                 "stageSteps": prev_sticky["stageSteps"]}
        if st.job.stages:
            # Only staged jobs own the stage layer — a fine-tune (single run)
            # must not wipe the user's per-stage settings just because the
            # job ignored them.
            entry["stageWeights"] = {str(i): dict(w) for i, w in
                                     sorted(st.job.stage_weights.items())}
            entry["stageSteps"] = {str(i): v for i, v in
                                   sorted(st.job.stage_budgets.items())}
        if entry != prev_sticky:
            sticky[b.id] = entry
            save_teach_weights(sticky)
        live = str(st.job.dir / "live.onnx")
        label = f"🎓 {st.job.display_title()} (untrained)"
        # The trainee previews what training practices: the behavior env with
        # the ACTIVE stage's spawn knobs (see requirement A / _spawn_knob).
        ekw = trainee_env_kwargs(b, st.job.stage_env())
        if st.trainee() is None:
            st.ducks.append(Duck("trainee", label, _zero_infer, seed=97,
                                 onnx_path=live, env_kwargs=ekw))
        else:
            st.trainee().rebuild_env(ekw)
            st.trainee().swap_policy(label, _zero_infer, onnx_path=live)
        for h in helper_ducks(st.ducks):
            h.rebuild_env(ekw)  # helpers mirror the stage from step one too
            # ...and the BRAIN too, like the trainee above. Rebuilding only
            # the env left every helper stepping the PREVIOUS job's ONNX
            # inside the new behavior's sim (out of distribution — they fall
            # repeatedly beside a correctly-zeroed trainee), and lab-state
            # persisted them pointing at the old run's live.onnx.
            # h.label, not the trainee's: apply_snapshot preserves a helper's
            # label forever, so passing the trainee's would rename every helper
            # to "🎓 <trick> (untrained)" permanently and erase the 🤝 identity
            # that tells them apart in the roster.
            h.swap_policy(h.label, _zero_infer, onnx_path=live)
        st.events.append(f"Training started: {b.emoji} {b.title}")
        sp = st.job.stage_payload()
        if sp:  # curriculum chain: name the opening stage right away
            st.events.append(
                f"Training stage {sp['idx']}/{sp['count']} — {sp['label']}")
        save_lab_state(st.ducks)
        return {"matched": True, "job": st.job.payload()}

    @app.post("/teach/stop")
    async def teach_stop() -> dict:
        if st.job:
            st.job.stop()
            st.events.append("Training stopped")
        return {"ok": True}

    @app.post("/teach/clear")
    async def teach_clear() -> dict:
        """Dismiss a FINISHED (or stopped/failed) training card.

        The teach panel's 🗑 cleared only the chat; the finished-run card kept
        coming back because every frame carries the job payload for as long as
        st.job exists ("when I hit the trash can it should clear out the
        spin"). A running job is deliberately NOT cleared — stop it first.
        """
        if st.job is None:
            # Deliberately does NOT sweep trainee/helper ducks here: after a
            # lab restart st.job is always None while restore_ducks has legit-
            # imately brought those ducks back, and 🗑 is always rendered — so
            # purging on this path silently deleted a restored roster. The
            # genuine orphan case (DELETE /runs clearing a seated card) is
            # handled where it happens, in the delete route.
            return {"ok": True, "cleared": False}
        if st.job.status in ("training", "restarting"):
            return {"ok": False, "cleared": False,
                    "message": "still training — stop it first"}
        owned = st.job.owns_preview_ducks
        st.job = None
        if owned:
            # The trainee/helper ducks are this job's artifacts; they go with
            # it. A merely SEATED run (teach/load, which the panel fires on
            # duck selection) owns nothing and leaves the roster alone.
            st.ducks = [d for d in st.ducks
                        if d.id != "trainee" and not d.id.startswith("helper")]
            # Persist, like every other roster mutation (assign/spawn/remove).
            save_lab_state(st.ducks)
        st.events.append("Training card cleared")
        return {"ok": True, "cleared": True}

    @app.post("/teach/load")
    async def teach_load(req: LoadRunReq) -> dict:
        """Pull a finished run's recipe up in the teach panel (see LoadRunReq).
        The viewer calls this when a duck running a teach-run policy is
        selected, or a policy chip is dropped on the teach panel."""
        if st.job and st.job.status == "training":
            return {"ok": False,
                    "message": f"Already teaching “{st.job.display_title()}” — "
                               "stop it first."}
        # "run:<name>" / "ckpt:<name>@123k" / bare name → the run dir name.
        name = req.policy.split(":", 1)[-1].split("@", 1)[0]
        # run_dir() is the ONE run-name validator (RUN_NAME_RE), shared with
        # DELETE /runs and the .onnx download — this used to hand-roll a looser
        # check of its own.
        try:
            run = run_dir(name)
        except ValueError:
            run = None
        if run is None or not run.is_dir():
            return {"ok": False,
                    "message": f"{name or req.policy!r} isn't a training run — "
                               "only runs under runs/ have a recipe to refine"}
        if not (run / "behavior.json").exists():
            return {"ok": False,
                    "message": f"{name} predates recipe records "
                               "(no behavior.json) — it can't be loaded"}
        # Same freshness rule as /teach: the card must reflect behaviors.py
        # as it is NOW, since retrain/fine-tune will train under it.
        # reload_self(), not importlib.reload — see the note in POST /teach.
        try:
            motion_mod.reload_self()
        except Exception as e:
            return {"ok": False, "message": f"motion.py didn't load: {e}"}
        # behaviors is a PACKAGE now (one module per trick):
        # reload()ing it would only re-run __init__ and keep every
        # submodule stale — reload_library() re-imports them all in
        # registration order and re-flattens the namespace.
        try:
            behaviors_mod.reload_library()
        except Exception as e:
            return {"ok": False, "message": f"behaviors.py didn't load: {e}"}
        _TRICK_DUCK_CACHE.clear()
        # Carry ownership across the swap: the OUTGOING job may have created
        # the trainee/helpers still on the roster, and seating a different run
        # (which the panel does on mere duck selection) must not orphan them
        # beyond the reach of 🗑.
        inherited = st.job is not None and st.job.owns_preview_ducks
        try:
            st.job = TrainingJob.adopt(name)
            if inherited:
                st.job.owns_preview_ducks = True
        except ValueError as e:
            return {"ok": False, "message": str(e)}
        except OSError:
            # adopt re-reads behavior.json; the run can be deleted between the
            # exists() check above and that read (a 500 before this).
            return {"ok": False,
                    "message": f"{name} disappeared while loading it"}
        st.events.append(
            f"📋 {name} loaded in the teach panel — tweak the recipe, "
            "then fine-tune or retrain")
        return {"ok": True, "job": st.job.payload()}

    @app.post("/teach/weights")
    async def teach_stage_weights(req: StageWeightsReq) -> dict:
        """Live per-stage weight edits. Future stages just record (their
        launch re-reads the map at handoff); a change to the ACTIVE stage's
        merged weights warm-restarts it — scale() with the unchanged helper
        count is exactly that restart (terminate, relaunch --init-from the
        stage's own snapshot, or its inherited warm start before the first
        snapshot lands)."""
        job = st.job
        if job is None or not job.stages:
            return {"ok": False,
                    "message": "no staged training job to set stage weights on"}
        if st.scaling:
            return {"ok": False,
                    "message": "trainer is mid-restart — try again in a moment"}
        changed = job.set_stage_weights(req.stageWeights)
        sticky = load_teach_weights()
        entry = sticky.get(job.behavior.id, empty_sticky())
        entry["weights"] = job.weights
        entry["stageWeights"] = {str(i): dict(w) for i, w in
                                 sorted(job.stage_weights.items())}
        sticky[job.behavior.id] = entry
        save_teach_weights(sticky)
        restarted = False
        if changed and job.status == "training":
            st.scaling = True
            try:
                await asyncio.to_thread(job.scale, job.helpers)
            finally:
                st.scaling = False
            restarted = True
            sp = job.stage_payload()
            st.events.append(
                f"Training stage {sp['idx']} restarted warm with new weights")
        elif job.status == "training":
            st.events.append(
                "Training stage weights recorded — future stages launch with them")
        return {"ok": True, "restarted": restarted, "job": job.payload()}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        # A WebSocket handshake is NOT a CORS request: no preflight is sent,
        # CORSMiddleware never sees it, and same-origin policy does not apply.
        # So the allowlist above bought this socket nothing, while the socket
        # itself takes `assign`, `spawn_duck`, `remove_duck` and `reset` — any
        # page the user happened to be browsing while the lab runs could open
        # ws://127.0.0.1:8788/ws and wipe the roster. Close BEFORE accept()
        # (Starlette turns that into a rejected handshake, not a live socket).
        if not origin_allowed(sock.headers.get("origin")):
            await sock.close(code=1008)   # 1008 = policy violation
            return
        await sock.accept()
        st.clients.add(sock)
        try:
            while True:
                msg = json.loads(await sock.receive_text())
                print(f"[ws] recv {msg}", flush=True)
                if "cmd" in msg:
                    # vx clamp follows the RUN curriculum ceiling (0.9), not
                    # the walk range (0.4) — the run policy trains to 0.9.
                    st.override = np.clip(np.array(msg["cmd"], np.float32),
                                          [-0.9, -0.3, -1.0], [0.9, 0.3, 1.0])
                    st.override_until = time.monotonic() + OVERRIDE_HOLD_S
                if msg.get("reset"):
                    for d in st.ducks:
                        d.reset()
                        d.falls = 0
                if "assign" in msg:
                    a = msg["assign"]
                    asyncio.create_task(do_assign(str(a.get("duck")), str(a.get("policy")),
                                                  showcase=bool(a.get("showcase"))))
                if msg.get("spawn_helper"):
                    asyncio.create_task(do_spawn_helper())
                if "remove_duck" in msg:
                    rd = msg["remove_duck"]
                    duck_id = rd.get("duck") if isinstance(rd, dict) else rd
                    asyncio.create_task(do_remove_duck(str(duck_id)))
                if "spawn_duck" in msg:
                    sd = msg["spawn_duck"]
                    pid = sd.get("policy") if isinstance(sd, dict) else sd
                    sc = bool(sd.get("showcase")) if isinstance(sd, dict) else False
                    asyncio.create_task(do_spawn_duck(str(pid) if pid else "",
                                                      showcase=sc))
        except WebSocketDisconnect:
            pass
        finally:
            st.clients.discard(sock)

    async def do_assign(duck_id: str, policy_id: str,
                        showcase: bool = False) -> None:
        duck = st.duck(duck_id)
        if duck is None:
            st.events.append(f"assign failed: no duck {duck_id}")
            return
        try:
            infer = await asyncio.to_thread(load_policy_infer, policy_id)
        except Exception as e:
            st.events.append(f"assign failed: {policy_id} ({type(e).__name__})")
            return
        entry = next((p for p in discover_policies() if p["id"] == policy_id), None)
        path = entry["path"] if entry else None
        # Showcase = the "whole trick" assign: rehearse spawns across the
        # whole trick arc (final-stage knobs). Quietly a plain assign when
        # the policy has no curriculum behind it — the flag can't mean
        # anything there, and refusing would make the chip feel broken.
        skw = showcase_env_kwargs(path) if showcase else None
        if skw is None:
            label = policy_id.split(":", 1)[-1]
            duck.rebuild_env(env_kwargs_for_policy_path(path))
        else:
            label = showcase_label(policy_id, bool(skw.get("spotter")))
            duck.rebuild_env(skw)
        duck.showcase = skw is not None
        duck.swap_policy(label, infer, policy_id=policy_id)
        ho = handoff_for(path) if skw is not None else None
        duck.handoff_infer, duck.handoff_label = ho if ho else (None, None)
        duck.handed = False
        st.events.append(f"{duck_id} now runs {label}")
        save_lab_state(st.ducks)

    async def do_spawn_helper() -> None:
        err = spawn_helper_error(st)
        if err:
            st.events.append(f"spawn_helper ignored: {err}")
            return
        st.scaling = True
        try:
            job = st.job
            n = next_helper_slot(st.ducks)
            live = job.dir / "live.onnx"
            try:
                infer = await asyncio.to_thread(_onnx_infer, live)
                onnx_path = str(live)
            except Exception:
                # The guard saw model.zip so live.onnx should exist; if a write
                # races us, the helper idles until the next snapshot lands.
                infer, onnx_path = _zero_infer, None
            # Same stage-mirrored preview env as the trainee: helpers rehearse
            # the same run, and a helper standing calmly next to a trainee
            # dropped mid-roll reads as "the helpers didn't get the fixes"
            # (it did to the user who spotted exactly that).
            st.ducks.append(Duck(f"helper{n}", f"🤝 helper {n}", infer,
                                 seed=100 + n, onnx_path=onnx_path,
                                 env_kwargs=trainee_env_kwargs(
                                     job.behavior, job.stage_env())))
            save_lab_state(st.ducks)
            # Visual only — do NOT job.scale(). A warm restart would stall
            # training for seconds and, with ENVS_PER_HELPER=0, would not
            # even change --envs. job.helpers tracks the roster for the HUD.
            job.helpers = len(helper_ducks(st.ducks))
            st.events.append(
                f"helper {n} joined — watching the same brain "
                f"({job.envs} train envs)")
        finally:
            st.scaling = False

    async def do_remove_duck(duck_id: str) -> None:
        err = remove_duck_error(st, duck_id)
        if err:
            st.events.append(f"remove_duck ignored: {err}")
            return
        was_helper = duck_id.startswith("helper")
        if not was_helper:
            # Plain roster duck: no trainer involvement, just drop and persist.
            st.ducks.remove(st.duck(duck_id))
            save_lab_state(st.ducks)
            st.events.append(f"{duck_id} left the lab")
            return
        st.ducks.remove(st.duck(duck_id))
        save_lab_state(st.ducks)
        if st.job and st.job.status == "training":
            st.job.helpers = len(helper_ducks(st.ducks))
        st.events.append(f"{duck_id} left the lab")

    async def do_spawn_duck(policy_id: str, showcase: bool = False) -> None:
        err = spawn_duck_error(st, policy_id)
        if err:
            st.events.append(f"spawn ignored: {err}")
            return
        try:
            infer = await asyncio.to_thread(load_policy_infer, policy_id)
        except Exception as e:
            st.events.append(f"spawn failed: {policy_id} ({type(e).__name__})")
            return
        n = next_duck_slot(st.ducks)
        entry = next((p for p in discover_policies() if p["id"] == policy_id), None)
        path = entry["path"] if entry else None
        # The "whole trick" chip drops on empty floor like any other chip —
        # the spawned duck showcases too (same no-op fallback as do_assign).
        skw = showcase_env_kwargs(path) if showcase else None
        label = (showcase_label(policy_id, bool(skw.get("spotter"))) if skw is not None
                 else policy_id.split(":", 1)[-1])
        duck = Duck(f"d{n}", label, infer, seed=37 + n,
                    policy_id=policy_id,
                    env_kwargs=(skw if skw is not None
                                else env_kwargs_for_policy_path(path)))
        duck.showcase = skw is not None
        ho = handoff_for(path) if skw is not None else None
        duck.handoff_infer, duck.handoff_label = ho if ho else (None, None)
        st.ducks.append(duck)
        save_lab_state(st.ducks)
        st.events.append(f"spawned d{n} running {label}")

    async def apply_snapshot() -> None:
        job = st.job
        if not job:
            return
        # The trainee and every helper track the same latest snapshot.
        targets = [d for d in st.ducks
                   if d.id == "trainee" or d.id.startswith("helper")]
        if not targets:
            return
        try:
            infer = await asyncio.to_thread(_onnx_infer, job.dir / "live.onnx")
        except Exception:
            return  # snapshot mid-write; the next one will land
        # Overall steps across a curriculum chain — a per-stage counter here
        # would make the trainee's label appear to rewind at each handoff.
        steps, _ = job.overall_progress()
        live = str(job.dir / "live.onnx")
        for d in targets:
            # display_title(), not behavior.title: an imitation run is about a
            # SPECIFIC authored clip, and the launch label already says so —
            # rebuilding from the generic recipe name here quietly renamed the
            # duck back to "Copy the animation" at the first snapshot.
            label = (f"🎓 {job.display_title()} @{steps // 1000}k"
                     if d.id == "trainee" else d.label)
            d.swap_policy(label, infer, onnx_path=live)
        st.events.append(f"Trainee updated to {steps // 1000}k steps")

    def current_cmd(now: float) -> tuple[np.ndarray, str]:
        if st.override is not None and now < st.override_until:
            return st.override, "manual"
        total = sum(s for s, _ in DEMO_SCRIPT)
        t = st.script_t % total
        for dur, cmd in DEMO_SCRIPT:
            if t < dur:
                return np.array(cmd, np.float32), "auto"
            t -= dur
        return np.zeros(3, np.float32), "auto"

    async def lab_loop():
        tick = 0
        next_t = time.monotonic()
        last_regroup = next_t
        while True:
            now = time.monotonic()
            cmd, mode = current_cmd(now)
            st.script_t += 1.0 / TICK_HZ
            if now - last_regroup > EPISODE_RESET_S:
                last_regroup = now
                # Restart the drive SCRIPT with the episode. Its clock was
                # global, so a reset dropped ducks into whatever phase was
                # playing — mid-sprint (spawn already running), mid-stop
                # (stand and shuffle), mid-sidestep (wanders off) — which
                # read as "some bug when it resets". Now every episode tells
                # the same story from the top: walk, turn, sprint, ...
                st.script_t = 0.0
                for d in st.ducks:
                    d.reset()
            training = bool(st.job and st.job.status == "training")
            for d in st.ducks:
                # Trick policies (trainee/helpers, and anything assigned from a
                # teach-* run) trained on zero twist commands — drive commands
                # are out-of-distribution noise to them and cause the "why does
                # this trick policy keep falling in the lab" report.
                # Zero commands are right for TRICKS (they trained on zero
                # twist) but wrong for locomotion behaviors (forward_cmd set):
                # zeroing those commanded the run trainee to STAND in the
                # preview while the real trainer practiced running — the exact
                # viewer-vs-reality split the user kept catching.
                b = getattr(d.env, "behavior", None)
                locomotion = bool(getattr(b, "forward_cmd", 0.0))
                d.set_cmd(cmd if (locomotion or not is_trick_duck(d))
                          else np.zeros(3, np.float32))
                # Helpers are visual clones. While a trainer is running they
                # step at 25 Hz (every other 50 Hz tick) so the lab's BAM
                # loop gives those cores back to the 16 training workers.
                # Broadcast is already 25 Hz, so the viewer sees every pose.
                if (training and d.id.startswith("helper")
                        and (tick & 1)):
                    continue
                d.tick()
            tick += 1
            if tick % 50 == 0:  # ~1 Hz: training poll + system stats
                if st.job:
                    prev_status = st.job.status
                    prev_stage = st.job.stage_idx
                    _, snap = st.job.poll()
                    if snap:
                        asyncio.create_task(apply_snapshot())
                    if st.job.stage_idx != prev_stage:
                        # Narrate the handoff + re-mirror the trainee's
                        # preview env onto the new stage's spawn knobs.
                        on_stage_handoff(st)
                    if st.job.status != prev_status and st.job.status in (
                        "done", "stopped", "failed"
                    ):
                        # 🎓 means "actively training" — a finished trainee
                        # relabels to its run hash so the row matches the
                        # palette entry (a user read the lingering 🎓 label
                        # as a live-but-stuck run).
                        t = st.trainee()
                        if t is not None:
                            short = st.job.run_name.removeprefix("teach-")
                            mark = {"done": "✔", "stopped": "■", "failed": "✗"}[
                                st.job.status]
                            t.label = f"{st.job.behavior.emoji} {short} {mark}"
                            # Preview goes back to ordinary standing spawns
                            # once the run ends — mid-trick drops mirror
                            # TRAINING; a finished trick shows off from its
                            # feet.
                            t.rebuild_env(
                                env_kwargs_for_behavior(st.job.behavior))
                            save_lab_state(st.ducks)
                        st.events.append(
                            f"training {st.job.status} — saved as {st.job.run_name}")
                st.stats = stats.sample(st.job)
            if tick % SEND_EVERY == 0 and st.clients:
                frame = json.dumps({
                    "cmd": [round(float(v), 3) for v in cmd],
                    "mode": mode,
                    "stats": st.stats,
                    "training": st.job.payload() if st.job else None,
                    "events": list(st.events)[-5:],
                    "ducks": [{
                        "id": d.id,
                        "name": d.label,
                        # Brain provenance ("run:<name>", "ckpt:…", "pollen:…",
                        # or null) — lets the viewer load a selected duck's
                        # run into the teach panel (POST /teach/load).
                        "policy": d.policy_id,
                        "falls": d.falls,
                        "steerable": not is_trick_duck(d),
                        "step": d.env.step_count,
                        "rew": round(d.reward_ema, 2),
                        "speed": d.forward_speed(),
                        # What the duck is being ASKED for, to read the
                        # achieved figure against — both our policies and
                        # shipped alpha_walking deliver about HALF their
                        # command, which is the most informative thing on
                        # the row. None for trick ducks: they run a pinned-
                        # zero twist, so "0.00 asked for" under a backflip
                        # is noise, not information.
                        "cmdSpeed": (None if is_trick_duck(d) else
                                     round(float(d.env.twist_cmd[0]), 3)),
                        "spawn": getattr(d.env, "last_spawn", None),
                        "assist": bool(getattr(d.env, "spotter_active", False)),
                        "handed": bool(getattr(d, "handed", False)),
                        "handoff": getattr(d, "handoff_label", None),
                        "bodies": d.pose_payload(),
                    } for d in st.ducks],
                })
                st.events.clear()  # one-shot toasts: deliver once, then drop
                dead = []
                # Snapshot: `await` inside the send suspends this task, and a
                # browser connecting or dropping in that window mutates
                # st.clients — "Set changed size during iteration" then killed
                # lab_loop outright. Nothing retrieves that task's exception
                # (make_app parks it in the lifespan frame), so the lab went
                # on serving HTTP and accepting sockets while every duck froze
                # and the scene sat empty, with no line in the log.
                for c in list(st.clients):
                    try:
                        await c.send_text(frame)
                    except Exception:
                        dead.append(c)
                for c in dead:
                    st.clients.discard(c)
            next_t += 1.0 / TICK_HZ
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("policies", nargs="*", help="run dirs and/or .onnx paths")
    ap.add_argument("--checkpoints", default=None,
                    help="run dir: add one duck per training checkpoint")
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--fresh", action="store_true",
                    help="delete lab-state.json and seed the roster from the "
                         "CLI args instead of restoring it")
    args = ap.parse_args()

    state_path = lab_state_path()
    if args.fresh:
        state_path.unlink(missing_ok=True)
    ducks: list[Duck] = []
    if state_path.exists():
        ducks = restore_ducks(state_path)
        if ducks:
            print(f"[lab] restored {len(ducks)} ducks from {state_path} "
                  "(CLI duck args ignored — --fresh to reseed)")
    if not ducks:
        ducks = build_ducks(args)
    print(f"[lab] {len(ducks)} ducks: {', '.join(d.label for d in ducks)}")

    import uvicorn
    uvicorn.run(make_app(ducks), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
