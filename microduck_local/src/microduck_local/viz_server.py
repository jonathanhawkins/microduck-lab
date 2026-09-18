"""Duck lab: run many policies side by side and stream poses to the web viewer.

    uv run duck-lab --checkpoints runs/first-gait ../microduck/policies/alpha_walking.onnx

Each argument becomes one live duck: a .onnx file, a run dir (uses policy.onnx,
exporting it on the fly if the run finished without one), or with --checkpoints
a run's checkpoints/*.zip lined up oldest→newest.

HTTP (default 127.0.0.1:8788):
  GET  /scene       visual geometry pulled from the compiled MuJoCo model
                    (the jenga-stacker extract_visual_scene trick)
  GET  /policies    everything assignable: shipped Pollen policies, local runs,
                    checkpoints — for the viewer's drag-and-drop palette. Runs
                    carry `trick` (the recipe they practised, see run_trick);
                    `tricks` names those ids and `robots` lists the bodies
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
                    knobs, and its weights MERGE over what that run trained
                    under (its behavior.json), so a one-slider edit can't
                    silently reset the rest of the recipe, "steps": N the TOTAL practice budget for the whole
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
  {"spawn_robot": {"robot": "mars"}}   add a slot for a BODY with no policy —
                           it idles (an arm holds HOME under its zero action,
                           a level-0 body holds its keyframe). The only way
                           onto the stage for a body that ships nothing to
                           assign, which is MARS and every Menagerie model.

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
from collections.abc import Sequence
from contextlib import asynccontextmanager, nullcontext
from functools import lru_cache
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
from . import run_record
from .brain.learned import brains_dir
from .lab import robots as lab_robots
from .pose import (  # noqa: F401 — the private names are re-exports for tests
    IkTarget,
    PoseScratch,
    _convex_hull,
    _over,
    _signed_distance,
    pose_scratch,
)
from .train import RUNS_DIR
from .walk_env import MicroduckWalkEnv, shared_model_scope
from .world_server import mount_world

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
# Re-exported: it moved to `lab/robots.py` so a BODY can read it without
# importing this server (`robots/microduck.MicroduckBody.shipped_policies`,
# which used to, under a `PHASE 1B:` note). The name stays here because
# `Duck.tick` and a handful of tests address it as `viz_server.POLICIES_DIR`.
POLICIES_DIR = lab_robots.POLICIES_DIR
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


def _zero_infer_for(robot: str = "microduck"):
    """A do-nothing brain of the right width — the 🎓 trainee runs this until
    the first snapshot lands. `_zero_infer` emitted 14 floats for every body,
    which a 29-joint robot's env rejects outright."""
    if robot in (None, "", "microduck"):
        return _zero_infer
    from .robots import spec as _spec
    body = _spec.get(robot)
    # A zero action is not always "do nothing", and on MARS it is exactly
    # the right nothing: its default action map is `delta`, so a = 0 holds
    # the commanded arm target where it is (docs/mars-roadmap.md 4a-2 — an
    # absolute map has no fixed point, which is why it could never sit
    # still). That is the idle a MARS slot runs before its first snapshot.
    zeros = np.zeros(body.num_actions, dtype=np.float32)

    def infer(obs: np.ndarray) -> np.ndarray:
        return zeros
    infer.obs_dim = body.obs_dim
    return infer


def _zero_infer(obs: np.ndarray) -> np.ndarray:
    return np.zeros(14, dtype=np.float32)


class Duck:
    """One env + one policy (ONNX session or in-process SB3 checkpoint)."""

    def __init__(self, duck_id: str, label: str, infer, seed: int,
                 policy_id: str | None = None, onnx_path: str | None = None,
                 env_kwargs: dict | None = None, robot: str = "microduck"):
        self.id = duck_id
        self.label = label
        # WHICH BODY. The lab was one robot deep; a roster entry now says
        # which, so a 99-obs G1 policy can never be stepped in a 61-obs duck
        # env (it would load, run, and produce nonsense).
        self.robot = robot or "microduck"
        self.infer = infer  # (obs[obs_dim]) -> action[num_joints]
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
            if self.robot != "microduck":
                # Another body: its own env class (99-obs G1, robots/g1_env;
                # 32-obs MARS, robots/mars_env) — or, for a body at level 0
                # of docs/mars-roadmap.md §7.1, no env at all and a
                # kinematic idle at its keyframe (lab/robots.slot_env).
                #
                # `task` picks WHICH env: the caller's when a teach job named
                # one (the trainee mirrors what the trainer is practising),
                # otherwise the BODY's `default_task` — "walk" is not a
                # question MARS can be asked, and `env_class("walk")` rightly
                # raises rather than handing an arm a walker's env.
                kw.pop("actuator", None)
                kw["actuator_force"] = "xml"
                return lab_robots.slot_env(self.robot, seed,
                                           {**common, **kw})
            if behavior_id:
                return behaviors_mod.BehaviorEnv(
                    behavior_id, standing_spawns=standing, **common, **kw)
            return MicroduckWalkEnv(**common, **kw)

    def set_cmd(self, cmd: np.ndarray) -> None:
        # A body with no drive channel is not steered rather than crashed:
        # `MarsArmEnv` has no `twist_cmd` (its base is disabled for every arm
        # task — rolling the whole robot at the target is the cheapest way to
        # satisfy a reach and not the thing being taught), and a level-0
        # body's `KinematicIdle` has none either. This runs inside the 50 Hz
        # loop, where an AttributeError stops the sim for the WHOLE roster.
        if not self.steers():
            return
        tw = np.asarray(cmd, np.float32).copy()
        # Deployment heading-hold, same 3 lines the robot runtime would run:
        # the policy is compass-blind (61 obs carry no yaw), so an unsteered
        # duck MUST drift into circles — which is exactly what the user was
        # shown while the videos (steered) ran straight. The viewer shows
        # DEPLOYED behavior now: driving forward with no explicit turn command
        # closes the loop on measured yaw. An explicit turn command wins.
        if tw[0] > 0.05 and abs(float(tw[2])) < 1e-6:
            rq = getattr(self.env, "_root_qpos", 0)
            q = self.env.data.qpos[rq + 3:rq + 7]
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
        if self.robot == "microduck":
            self.env.head_cmd[:] = 0.0
            self.env.body_cmd[:] = 0.0

    def steers(self) -> bool:
        """Does this slot's env take a drive command at all?

        Asked of the ENV, not of the robot id: a walking env carries
        `twist_cmd` and an arm env does not, and a level-0 body's
        `KinematicIdle` has no channel to steer. Answering by capability is
        what keeps WASD from raising inside the 50 Hz loop for a body nobody
        thought about — the failure mode a hard-coded id list has here is a
        dead lab, not a dead duck (`lab_loop`'s exception is unretrieved).
        """
        return hasattr(self.env, "twist_cmd")

    def set_robot(self, robot: str) -> None:
        """Move this roster slot to another body.

        Clears the rebuild memo: `rebuild_env` returns early when the kwargs
        are unchanged, and a robot swap usually carries the SAME kwargs — so
        without this the slot kept the old body's env and the new policy's
        observations went into the wrong robot.
        """
        robot = robot or "microduck"
        if robot == self.robot:
            return
        self.robot = robot
        self.env_kwargs = {"__robot_swap__": robot}   # force the next rebuild

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
        want.pop("__robot_swap__", None)   # set_robot's rebuild trigger
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
        """The trick is finished and the duck is on both feet.

        A behavior may own this (Behavior.handoff_fn) — find_ball hands to a
        kick once it is squared up on the ball, which has nothing to do with
        rotation. render_rollout.handoff_due consults the same field, so a
        behavior that brings a predicate gets both callers from one
        implementation instead of two copies kept in step by comment.
        """
        env = self.env
        fn = getattr(getattr(env, "behavior", None), "handoff_fn", None)
        if fn is not None:
            return bool(fn(env))
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
        # Opt-out for a behavior whose heading is the point: find_ball turned
        # to face the ball, and "drive yaw back to the spawn heading" would
        # undo exactly the work it just did (and aim the kick at nothing).
        if not getattr(getattr(self.env, "behavior", None), "handoff_recenter", True):
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
        # The shared drive command survives the episode — for a body that has
        # one. An arm env and a kinematic idle have none (see `steers`), and
        # reading `twist_cmd` off them here would raise inside the 50 Hz loop
        # at the first episode boundary rather than at the first keypress.
        cmd = self.env.twist_cmd.copy() if self.steers() else None
        self._hold_yaw = None   # new episode, new heading anchor
        self.handed = False   # each episode starts on the trick's own brain
        # Speed is reported PER EPISODE: carrying the last half second of a
        # run that ended in a faceplant into the fresh episode would show a
        # duck standing still at 0.3 m/s.
        self.speed_hist.clear()
        self.obs, _ = self.env.reset()
        if cmd is not None:
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

        A body whose env has no heading frame records NOTHING, and
        `forward_speed()` then shows "—": `heading_lin_vel` is
        `MicroduckWalkEnv`'s, an arm env has no trunk to measure and a
        `KinematicIdle` never moves, so a zero here would be a made-up
        reading rather than a missing one.
        """
        if not hasattr(self.env, "heading_lin_vel"):
            return
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
        """One pose per body of `GET /scene`, IN ITS ORDER.

        The viewer zips this list against the scene's bodies positionally, so
        the two have to be the same shape. They stopped being so when the
        hinged `mouth` was inserted mid-list (scene index 11): this env is the
        un-split walk model and has 16 bodies where the scene has 17, which
        drew the bill at the bearing's pose, put the whole right leg one link
        late, and left `ankle_right` at the origin because nothing ever wrote
        it. So emit in SCENE order and look each body up by NAME; a body the
        walk model does not have rides its scene parent, which is where the
        bill actually hangs."""
        d, m = self.env.data, self.env.model
        out = []
        mapping = (_scene_to_env(m) if self.robot == "microduck"
                   else _robot_scene_to_env(m, self.robot))
        for env_b, parent_b in mapping:
            b = env_b if env_b >= 0 else parent_b
            p, q = d.xpos[b], d.xquat[b]
            out.append([round(float(v), 4) for v in (*p, *q)])
        return out


# ------------------------------------------------------------ policy loading

@lru_cache(maxsize=4)
def _scene_to_env(model) -> tuple[tuple[int, int], ...]:
    """For each body of `GET /scene`, in scene order: (index of the body of
    the same name in `model`, or -1; index of its scene PARENT resolved the
    same way). Cached per model — the scene's body list is static."""
    from .world.compose import scene_model

    sm = scene_model()
    by_name = {model.body(b).name: b for b in range(model.nbody)}
    out: list[tuple[int, int]] = []
    for b in range(sm.nbody):
        own = by_name.get(sm.body(b).name, -1)
        par = by_name.get(sm.body(int(sm.body_parentid[b])).name, 0)
        out.append((own, par))
    return tuple(out)


@lru_cache(maxsize=4)
def _robot_scene_to_env(model, robot: str) -> tuple[tuple[int, int], ...]:
    """`_scene_to_env` for a body that is not the duck.

    Same contract: one (own, parent) pair per body of `GET /scene?robot=<id>`,
    IN THAT ORDER, resolved by NAME. The viewer zips the streamed poses
    against the scene's body list positionally, so a robot whose training
    scene and visual scene were compiled from different files (they are —
    the visual one has no floor) must still line up name by name.

    Generic since Phase 1b: the names come from the BODY's own
    `visual_scene()` (`lab/robots.scene_body_names`), so MARS's 18 bodies and
    a Menagerie quadruped's line up exactly as the G1's did, with no branch
    here. A body whose assets are gone raises, which is the same
    `KeyError`-shaped absence the `g1` check used to produce.
    """
    names = lab_robots.scene_body_names(robot)
    by_name = {model.body(b).name: b for b in range(model.nbody)}
    out: list[tuple[int, int]] = []
    for name in names:
        own = by_name.get(name, -1)
        # The parent of an unknown body is the world (index 0): a body the
        # env does not have rides the root rather than sitting at the origin.
        out.append((own, 0))
    return tuple(out)


def _onnx_infer(path: Path):
    import onnxruntime as ort
    # One thread per session, like brain_env's walker. The lab steps its whole
    # roster serially in one frame loop, so a session's own thread pool can
    # never overlap with anything — but ORT's DEFAULT is a thread per core, so
    # an 8-duck roster meant 8 pools of N spinning threads in this one process,
    # competing with the trainer subprocess a teach run just launched.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=opts)
    in_name = sess.get_inputs()[0].name

    def infer(obs: np.ndarray) -> np.ndarray:
        return sess.run(None, {in_name: obs[None]})[0][0].astype(np.float32)
    # What this brain expects to be fed. The roster can hold two bodies now,
    # and a policy stepped in the wrong one runs happily on garbage.
    shape = sess.get_inputs()[0].shape
    if len(shape) == 2 and isinstance(shape[1], int):
        infer.obs_dim = int(shape[1])
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
    infer.obs_dim = int(model.policy.observation_space.shape[0])
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


# --- robots: every answer below is the registry's, via lab/robots.py -------
#
# `docs/mars-roadmap.md` counted 15 `"g1"` literals in this file, and §6.4's
# fix is that a body ANSWERS instead of being branched on. What remains here
# are thin re-exports, kept under the names the rest of this module, the
# tests and the viewer's HTTP surface already address.

#: Which body a policy drives — see `lab/robots.policy_robot` (it resolves the
#: policy's own CONTRACT now, which is what lets a stamped .onnx answer for
#: itself after it has been moved away from its run directory).
policy_robot = lab_robots.policy_robot

#: One-click asset downloads, PER BODY (`POST /robots/{id}/fetch`). It was one
#: module-level dict called `_G1_FETCH`, so a MARS download would have shown
#: up in the palette as a G1 one.
start_fetch = lab_robots.start_fetch

#: What a SENTENCE calls a body ("teach the {noun} a trick").
robot_noun = lab_robots.robot_noun

#: The palette's robot switch. Every `registry.ids()` entry, `ready` per body.
available_robots = lab_robots.available_robots


#: The RECIPE a run practised, as a behaviors id — what the palette groups
#: "Our runs" by. Through `Body.tasks()` now, so a body's recipes are the
#: body's (`lab/robots.run_trick`).
run_trick = lab_robots.run_trick


def trick_names(policies: list[dict]) -> dict[str, dict]:
    """Heading text for every trick id `policies` mentions: the recipe's own
    title and emoji. An id with no recipe behind it (a bare task name, a
    recipe since deleted) is left out and the palette shows the id."""
    out: dict[str, dict] = {}
    for tid in {p["trick"] for p in policies if p.get("trick")}:
        b = behaviors_mod.BEHAVIORS.get(tid)
        if b is not None:
            out[tid] = {"title": b.title, "emoji": b.emoji}
    return out


def discover_policies() -> list[dict]:
    """Everything assignable, grouped for the palette. Run entries carry
    `mtime` (epoch seconds, see _run_mtime) and are sorted newest-first —
    the user couldn't tell which run was fresh from bare name chips. Stage
    runs additionally carry `chain` (the prefix without -sN) and `stage`
    (1-based) so the panel can group a curriculum chain as one family.
    `sizeBytes` (run entries) is what deleting the run would free — the
    palette's delete confirmation shows it."""
    # Every body's SHIPPED drop, from the body (`Body.shipped_policies`).
    # This was two hand-written blocks — a `pollen` glob and a `g1`
    # try/except — and a third body would have been a third. A body that
    # ships nothing contributes nothing and its section disappears from the
    # palette, which is MARS's case: Innate's learned skills are ACT
    # checkpoints trained from demonstrations, not ONNX, and `innate-os`
    # vendors none.
    out: list[dict] = list(lab_robots.shipped_entries())
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
                         "robot": policy_robot(run),
                         "mtime": _run_mtime(run),
                         "sizeBytes": _run_size(run)}
                # What a PERSON reads (run_record.py): the chip's title, the
                # measured sentence behind it, and the ★ on the stage of a
                # chain worth using. Absent for a run nobody has described —
                # the panel falls back to the directory name, as it always
                # did. `label` stays the run NAME whatever the title says:
                # it is the identifier --init-from and the docs address.
                label = run_record.read_label(run)
                if label.get("title"):
                    entry["title"] = str(label["title"])
                if label.get("note"):
                    entry["note"] = str(label["note"])
                if label.get("pick"):
                    entry["pick"] = True
                trick = run_trick(run, entry["robot"])
                if trick:
                    entry["trick"] = trick
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
                                         "group": "checkpoints", "path": str(z),
                                         "robot": policy_robot(run)})
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


def _same_run_dir(p: Path) -> bool:
    """Is `p` the run dir that `run:<p.name>` resolves to?

    is_trick_duck() and the palette both address a run by BARE NAME under
    RUNS_DIR, so tagging a duck `run:<name>` is only honest when the directory
    it came from really is that one. A run dir given by some other path
    (a scratch copy, another checkout, MICRODUCK_RUNS_DIR pointing elsewhere)
    keeps policy_id=None and the old conservative behavior."""
    try:
        return p.resolve() == (RUNS_DIR / p.name).resolve()
    except OSError:
        return False


def build_ducks(args) -> list[Duck]:
    ducks: list[Duck] = []

    def add(label, infer, policy_id=None, onnx_path=None, robot=None):
        ducks.append(Duck(f"d{len(ducks)}", label, infer, seed=len(ducks),
                          policy_id=policy_id, onnx_path=onnx_path,
                          robot=robot or policy_robot(onnx_path)))

    if args.checkpoints:
        run = Path(args.checkpoints)
        zips = sorted(run.glob("checkpoints/model_*_steps.zip"),
                      key=lambda p: int(p.stem.split("_")[1]))
        robot = policy_robot(run)
        for z in zips:
            steps = z.stem.split("_")[1]
            vn = z.parent / f"model_vecnormalize_{steps}_steps.pkl"
            if vn.exists():
                label = f"{run.name}@{int(steps) // 1000}k"
                add(label, _checkpoint_infer(z, vn), policy_id=f"ckpt:{label}",
                    robot=robot)
        if (run / "policy.onnx").exists():
            add(f"{run.name}@final", _onnx_infer(run / "policy.onnx"),
                onnx_path=str(run / "policy.onnx"), robot=robot)

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
                # policy_id, not just a label: is_trick_duck() classifies on a
                # "run:" prefix, so without this a run dir passed on the CLI is
                # not recognised as a trick and the lab drives it with the
                # WASD velocity command. For a ball/trick brain that command is
                # pure out-of-distribution noise — `duck-lab runs/find_ball`
                # measured 1058 falls and r̄ -3.2 in four minutes, while the
                # identical dir dropped from the 🧠 palette (which does set
                # policy_id) stood there quite happily.
                same = _same_run_dir(p)
                add(p.name, _onnx_infer(onnx),
                    policy_id=f"run:{p.name}" if same else None,
                    onnx_path=str(onnx))
            else:
                print(f"[lab] skipping {p}: no policy.onnx / model.zip")
        else:
            print(f"[lab] skipping {spec}: not an .onnx or run dir")
    if not ducks and not getattr(args, "world", None):
        raise SystemExit("no ducks — pass run dirs or .onnx paths (or --world <scenario>)")
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
        # The run's clip comes back with it (train_behavior records it), so
        # the card says `Perform “<clip>”` and a fine-tune trains the SAME
        # motion. Older runs predate the record: they seat clip-less.
        clip = meta.get("clip")
        self.extra_env = {"MICRODUCK_CLIP": clip} if clip else {}
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
        if self.behavior.trainer:
            # Another body's task: its own trainer, same artifacts. train.py
            # writes progress.jsonl + live.onnx exactly as train_behavior
            # does, which is what makes the job watchable in the panel.
            cmd = [sys.executable, *self.behavior.trainer,
                   "--run-name", self.run_name, "--envs", str(self.envs),
                   "--steps", str(self.total_steps)]
            if self.snap_steps:
                cmd += ["--snap-steps", str(self.snap_steps)]
            if init_from is not None:
                cmd += ["--init-from", str(init_from)]
            log = open(self.dir / "train.log", "a")
            # …including the STAGE's knobs: a task curriculum steers its
            # trainer exactly as a trick's does, through the environment.
            return subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).resolve().parents[2]),
                # MICRODUCK_RUN_TITLE: the trainer writes the run's record
                # (run_record.py) and cannot know what the LAB called this
                # job — "Perform “g1-front-kick”" is the name the person
                # watching it saw, so it is the name the palette should keep.
                env={**os.environ, "MICRODUCK_RUN_TITLE": self.display_title(),
                     **self.extra_env, **self._stage_env})
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
         "robot": getattr(d, "robot", "microduck"),
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
            elif entry.get("robot") and entry["robot"] != "microduck":
                # A slot put on the stage with no brain (`spawn_robot`): MARS
                # before anything is trained on it, a Menagerie model at
                # level 0. Its idle IS the zero action, so there is nothing
                # to load and dropping the row would silently empty the
                # stage across a restart.
                infer = _zero_infer_for(str(entry["robot"]))
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
        try:
            duck = Duck(str(entry["id"]), str(entry["label"]), infer, seed=i,
                        policy_id=entry.get("policy"),
                        onnx_path=entry.get("onnxPath"),
                        robot=str(entry.get("robot") or policy_robot(run_path)),
                        env_kwargs=(skw if skw is not None
                                    else env_kwargs_for_policy_path(run_path)))
        except Exception as e:
            # Its BODY can't be built — a G1 saved on a machine whose G1
            # assets are gone. Skip it like an unloadable brain: one missing
            # robot must not stop the whole lab from starting.
            print(f"[lab] skipping {entry.get('id')} from {path.name}: "
                  f"{type(e).__name__}: {e}")
            continue
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
    # Which body to teach, as the panel's robot switch says. None = the old
    # rule (the trainee's body, else a one-robot roster, else the duck).
    robot: str | None = None
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


def run_trained_weights(run: Path) -> dict[str, float]:
    """The reward weights a finished run ACTUALLY trained under — its
    behavior.json (written by train_behavior before the first step, and the
    same record TrainingJob.adopt seats in the panel). {} for a run that
    predates the file or whose record is unreadable: a fine-tune of such a
    run falls back to the recipe defaults, which is all anyone knows."""
    try:
        meta = json.loads((run / "behavior.json").read_text())
    except (OSError, ValueError):
        return {}
    weights = meta.get("weights") if isinstance(meta, dict) else None
    return dict(weights) if isinstance(weights, dict) else {}


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


def run_clip(run: Path) -> str | None:
    """The clip a run trained against, from its behavior.json — None for
    runs without one (non-imitation recipes, or records that predate it)."""
    try:
        clip = json.loads((run / "behavior.json").read_text()).get("clip")
    except (OSError, ValueError):
        return None
    return clip if isinstance(clip, str) and clip else None


# The lab's OWN display title for an imitation job (TrainingJob.display_title).
# A client that echoes it back as the /teach text — the panel did, before it
# learned to send the behavior id — must reach the imitation recipe with that
# clip, never a recipe that happens to own a keyword inside the clip's name.
_DISPLAY_TITLE_RE = re.compile(r"\s*perform\s+[“\"](.+?)[”\"]\s*$", re.IGNORECASE)


def match_teach_text(text: str, robot: str = "microduck"
                     ) -> tuple["behaviors_mod.Behavior | None", str | None]:
    """(behavior, clip-implied-by-the-text) for a /teach message.

    Matching is scoped to the ROBOT the roster is teaching: the duck's trick
    recipes and another body's tasks live in one registry, and neither can be
    trained on the other's geometry."""
    m = _DISPLAY_TITLE_RE.match(text)
    if m:
        b = imitation_behavior(robot)
        if b is not None:
            return b, m.group(1)
    return behaviors_mod.match_behavior(text, robot), None


#: The 🎓 panel's suggestion chips for a body, through `Body.tasks()`
#: (`lab/robots.teach_suggestions`). A body with nothing to teach — a
#: Menagerie model at level 0 — gets an empty list and therefore no chip,
#: rather than the duck's.
teach_suggestions = lab_robots.teach_suggestions

#: The recipe that tracks a saved clip on a body (the duck's `imitate`).
imitation_behavior = lab_robots.imitation_behavior


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
    if not getattr(b, "height_termination", True):
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
    if getattr(b, "robot", "microduck") != "microduck":
        # Another body's task is an ENV, not a spawn-knob recipe: the preview
        # is simply that env, and the trainer's own knobs (command mix) are
        # its defaults. `task` is read by Duck._make_env.
        kw = {"task": b.task,
              "max_episode_s": float(getattr(b, "episode_s", 10.0))}
        # An imitation task tracks the SAME clip the trainer was handed, and
        # is judged at the same strictness rung (robots/g1_imitate).
        if (stage_env or {}).get("MICRODUCK_CLIP"):
            kw["clip_name"] = stage_env["MICRODUCK_CLIP"]
        if (stage_env or {}).get("MICRODUCK_G1_LIFT_MIN"):
            try:
                kw["lift_min_got"] = float(stage_env["MICRODUCK_G1_LIFT_MIN"])
            except ValueError:
                pass
        return kw
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


def env_kwargs_for_task_run(run: Path) -> dict:
    """Lab-preview env for a WALK/TASK run of another body, from its run.json.

    A `train-walk --robot g1 --task …` run writes no behavior.json, so the
    lookup below returned {} and `Duck._make_env` fell through to
    `task="walk"`: the G1's measured-best front kick (`g1_imitate` on the
    "g1-front-kick" clip) was stepped in G1WalkEnv, where the three command
    slots carry a locomotion twist instead of the clip's clock. The policy
    read each held twist as ONE frozen phase — filmed on the lab page it
    kicked once after a reset and then stood in a split stance; measured
    headless, 2 lifts in 12 s against 4 (one per 3 s loop) in its own env.
    The same hole swallowed every task env that owns those slots (stand,
    squat, front_kick, punch). Mirrors `trainee_env_kwargs`' other-body
    branch: `task` (read by Duck._make_env), the clip, the episode length.

    {} for the duck, a walk run, an unknown task or a clip that is gone —
    the slot then previews in the walk env as before, because a raise here
    lands in a spawn/assign/restore path and an unbuildable env there costs
    the whole roster, not one chip."""
    try:
        rj = json.loads((run / "run.json").read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(rj, dict):
        return {}
    robot, task = rj.get("robot") or "microduck", rj.get("task")
    if robot == "microduck" or task in (None, "", "walk"):
        return {}
    from .train import env_class
    try:
        env_class(robot, str(task))
    except (SystemExit, Exception):  # noqa: BLE001  (env_class raises SystemExit on an unknown task)
        return {}
    kw: dict = {"task": str(task)}
    ekw = rj.get("env_kwargs") if isinstance(rj.get("env_kwargs"), dict) else {}
    clip = ekw.get("clip_name")
    if clip:
        from . import motion
        try:
            motion.load_clip(str(clip))
        except Exception:  # noqa: BLE001  (a deleted/renamed clip must not kill a restore)
            return {}
        kw["clip_name"] = str(clip)
    elif str(task) == "imitate":
        return {}                       # G1ImitateEnv raises without a clip
    try:
        ep = float(ekw.get("max_episode_s") or 0.0)
    except (TypeError, ValueError):
        ep = 0.0
    if ep > 0.0:
        kw["max_episode_s"] = ep
    return kw


def env_kwargs_for_policy_path(path: str | None) -> dict:
    """Same, derived from a run artifact's sibling behavior.json (assigned or
    restored teach-run policies) — or, for another body's walk/task run,
    from its run.json (`env_kwargs_for_task_run`)."""
    if not path:
        return {}
    run = Path(path).parent
    try:
        behavior_id = json.loads((run / "behavior.json").read_text()).get("behavior")
        return env_kwargs_for_behavior(behaviors_mod.BEHAVIORS[behavior_id])
    except (OSError, KeyError, json.JSONDecodeError):
        return env_kwargs_for_task_run(run)


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
    # Only offer the hand-off if the duck can ever ASK for it. A behavior
    # that declares handoff_fn answers that itself; otherwise the fallback
    # rule tests env._bf_rot, the rotation accumulator that only the flip
    # family's state_fn advances — a headstand never sets it, so attaching
    # alpha_stand to that chain advertised "handoff: alpha_stand" in every
    # frame for a swap that could not happen.
    if (getattr(b, "handoff_fn", None) is None
            and getattr(b, "state_fn", None) is not behaviors_mod._bf_update):
        return None
    policy = getattr(b, "handoff_policy", None) or HANDOFF_POLICY
    try:
        return load_policy_infer(policy), policy.split(":", 1)[-1]
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


# ----------------------------------------------------------- stage layout
#
# One slot's pitch is a property of its ROBOT, not of the lab. The viewer used
# to lay the grid out itself on a single 0.65 m constant, which is the duck's:
# six G1 helpers plus a trainee (a 1.3 m humanoid is 0.53 m wide) landed 0.12 m
# apart and interpenetrated into one unreadable clump. The layout moved here so
# there is ONE definition and it can read each robot's spec.

_LAB_SPACING: dict[str, float] = {}


def lab_spacing_m(robot: str | None) -> float:
    """`RobotSpec.lab_spacing_m` for a roster row's robot, in metres.

    Falls back to the duck's pitch for a body this process cannot resolve (a
    lab-state row naming a robot whose assets have since been removed): a
    layout number must never be able to stop the 50 Hz loop.
    """
    key = robot or "microduck"
    hit = _LAB_SPACING.get(key)
    if hit is not None:
        return hit
    from .robots import spec as _spec
    try:
        val = float(_spec.get(key).lab_spacing_m)
    except Exception:
        return float(_spec.RobotSpec.lab_spacing_m)   # the dataclass default
    _LAB_SPACING[key] = val                           # cache successes only
    return val


def lab_slot_offsets(robots: Sequence[str | None]) -> list[tuple[float, float]]:
    """Where each roster slot stands on the lab floor, in MuJoCo XY metres.

    The same square grid the viewer has always drawn — columns across x,
    centred on the row the first slots occupy, rows growing toward +y — with
    one change: a column's half-pitch is the widest half-pitch IN that column
    and a row's is the widest in that row, instead of one duck-sized constant
    everywhere. So the gap between two neighbours is always at least what the
    larger of the two asks for, and a MIXED roster spaces a duck from a duck by
    the duck's pitch while the G1 beside them gets its own.

    For a roster of one robot this is arithmetically the old expression, to the
    bit: the duck's layout is unchanged (tests/test_lab_robots.py pins it
    against values taken from the viewer's own function before the change).
    """
    n = len(robots)
    if n == 0:
        return []
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    half = [lab_spacing_m(r) / 2.0 for r in robots]
    col_half = [max(half[i] for i in range(n) if i % cols == c)
                for c in range(cols)]
    row_half = [max(half[i] for i in range(n) if i // cols == r)
                for r in range(rows)]
    xs = [0.0] * cols
    for c in range(1, cols):
        xs[c] = xs[c - 1] + col_half[c - 1] + col_half[c]
    ys = [0.0] * rows
    for r in range(1, rows):
        ys[r] = ys[r - 1] + row_half[r - 1] + row_half[r]
    # Centred on the columns the FIRST row occupies, exactly as the viewer was.
    span = xs[min(n, cols) - 1] - xs[0]
    return [(xs[i % cols] - span / 2.0, ys[i // cols]) for i in range(n)]


def spawn_duck_error(st: LabState, policy_id: str | None) -> str | None:
    """Why {"spawn_duck": ...} can't be honored (None = go)."""
    if not policy_id:
        return "spawn_duck needs a policy id"
    if len(st.ducks) >= MAX_DUCKS:
        return f"lab is full ({MAX_DUCKS} ducks) — remove one first"
    return None


#: The DUCK's mesh dump for the viewer. It moved to `lab/robots.py` so
#: `robots/microduck.MicroduckBody.visual_scene` can call it without
#: importing this server (that method carried a `PHASE 1B:` note saying the
#: import pointed the wrong way). Re-exported under the name `make_app` and
#: two test modules already address.
extract_scene = lab_robots.extract_scene


def spawn_robot_error(st: LabState, robot: str | None) -> str | None:
    """Why {"spawn_robot": ...} can't be honored (None = go).

    A slot with no POLICY, which `spawn_duck` cannot express: it needs a
    palette id, and the two bodies that ship no policy at all are exactly the
    ones a person most needs to be able to put on the stage — MARS before
    anybody has trained it anything, and every Menagerie model (level 0 of
    docs/mars-roadmap.md §7.1 is "you can see it", and there is nothing to
    assign). The slot idles: an arm env holds HOME under a zero action (its
    default map is `delta`, so zero means stay), a level-0 body holds its
    keyframe (`lab/robots.KinematicIdle`).
    """
    if not robot:
        return "spawn_robot needs a robot id"
    if len(st.ducks) >= MAX_DUCKS:
        return f"lab is full ({MAX_DUCKS} ducks) — remove one first"
    if robot not in registry_ids():
        return f"no robot {robot!r} — have {', '.join(registry_ids())}"
    return None


def registry_ids() -> tuple[str, ...]:
    """Every body id this install knows, fetched or not (`registry.ids`)."""
    from .robots import registry as _registry
    return _registry.ids()


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


# The posing engine itself — forward kinematics, the balance read and the IK
# solve — lives in pose.py, one scratch model per ROBOT (the G1 has an editor
# too); it is imported at the top with the other modules, and the helper
# names the tests grew up on are re-exported from there.
SOLE_TOL = C.MICRODUCK.sole_tol
GROUND_TOL = C.MICRODUCK.ground_tol


class IkTargetReq(BaseModel):
    pos: list[float]
    # Hold the effector level (a foot flat on the floor); default for feet.
    level: bool | None = None
    weight: float = 1.0


class IkReq(BaseModel):
    """POST /ik — drag a foot or a hand: joints that put each named effector
    at its target, every OTHER foot pinned where the input pose has it."""
    joints: list[float]
    rootPitch: float = 0.0
    ground: bool = True
    targets: dict[str, IkTargetReq]
    # Effector ids to hold at their current world position while the targets
    # are reached. None = every foot that is not itself a target.
    pins: list[str] | None = None


def robot_of(robot: str | None) -> str:
    """The robot id a `?robot=` query names — the duck when it names none."""
    return "microduck" if not robot or robot in ("duck", "microduck") else robot


def scratch_for(robot: str | None) -> PoseScratch:
    """pose_scratch(), with a missing body reported as a 404 rather than a
    500 (the G1's assets are optional — `uv run fetch-g1`).

    A body the editor cannot pose is a 404 too, and by CAPABILITY rather
    than by id: `PoseScratch` is a walker's tool — effectors, soles, a base
    body — so `pose_scratch("mars")` died with `AttributeError: 'MarsBody'
    object has no attribute 'base_body'`, a 500 with a traceback in the log
    for a question that has a perfectly good answer (`docs/mars-roadmap.md`
    §2a found this the moment a third body was registered). `GET /robots`
    carries the same flag as `animate`, so the panel never offers the body
    in the first place; this is the door being locked as well as unlisted.
    """
    rid = robot_of(robot)
    try:
        if not lab_robots.available_animate(rid):
            raise HTTPException(
                404, f"the 🎬 editor cannot pose a {lab_robots.robot_noun(rid)}"
                     " — it poses walkers (effectors, soles, a base link)")
        return pose_scratch(rid)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, f"{robot_of(robot)} assets missing — {e}")


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


def clean_joints(raw, robot: str = "microduck") -> list[float]:
    """The robot's joint count of finite floats in joint_names order, CLAMPED
    to the MJCF limits (14 for the duck, 29 for the G1).

    Clamping rather than rejecting: an out-of-range angle is unreachable on the
    real servo, so the honest fix is the nearest reachable one — and the caller
    gets the clamped values back so its UI can show what actually happened."""
    scratch = pose_scratch(robot_of(robot))
    n = scratch.spec.num_joints
    if not isinstance(raw, (list, tuple)) or len(raw) != n:
        raise ValueError(f"joints must be {n} numbers for the {scratch.spec.id}, "
                         f"got {len(raw) if hasattr(raw, '__len__') else type(raw).__name__}")
    vals = np.array([_finite(v) for v in raw], dtype=np.float64)
    return [round(float(v), 6) for v in scratch.clamp(vals)]


def clean_clip(name: str, raw: dict) -> dict:
    """Validate + normalize a clip against the v1 contract, or ValueError.

    Rejects anything the RL resampler could silently misread (missing t=0,
    out-of-order keys, a duration that would truncate the last key); clamps
    only what has one obvious right answer (joint angles → servo limits).

    `robot` names the body the clip poses (absent = the duck, so every clip
    saved before there was a second body still reads); the joint count and
    limits are that body's, and the trainer refuses a clip for another."""
    if not isinstance(raw, dict):
        raise ValueError("clip must be an object")
    robot = robot_of(raw.get("robot") or None)
    try:
        pose_scratch(robot)
    except KeyError:
        raise ValueError(f"unknown robot {robot!r}")
    except FileNotFoundError:
        raise ValueError(f"{robot} assets missing — `uv run fetch-g1`")
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
            "joints": clean_joints(k.get("joints"), robot),
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
        "robot": robot,
        "duration": duration,
        "loop": bool(raw.get("loop", False)),
        "keys": keys,
    }


def clip_robot(name: str) -> str:
    """Which body a saved clip poses (the duck when the file predates the
    field). Missing/unreadable clips read as the duck: the caller checks
    existence separately and reports that instead."""
    try:
        return robot_of(json.loads(clip_path(name).read_text()).get("robot") or None)
    except (OSError, ValueError):
        return "microduck"


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
    # …and the ducks' voices, when the take carried them (the /sim page's
    # 🔊 quacks put an opus track in the upload). `-c:a aac` is inert on a
    # silent take, so there is no branch here.
    _ffmpeg("-i", str(src), "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", str(mp4))
    # gif from the cleaned mp4: two-pass palette at a README-friendly width.
    # (Plain palettegen/paletteuse only — imageio-ffmpeg's bundled binary is
    # an older build without stat_mode/diff_mode.)
    # -an: a gif has no soundtrack. The bundled build maps only the filter's
    # output and ignores the mp4's aac either way (checked both ways on this
    # binary) — this says so rather than resting on a default.
    _ffmpeg("-i", str(mp4), "-an", "-filter_complex",
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
    # Ducks apply_snapshot has already refused to re-brain, so the reason is
    # said once instead of at every snapshot. Cleared when a new job starts.
    snapshot_skipped: set[str] = set()

    @asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(lab_loop())
        world.start()   # the /sim world loop, beside the roster loop
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
        world.stop()
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

    # World mode (the /sim page): its own World, routes and /ws/sim socket —
    # see world_server.py. Mounted first so its routes exist before the
    # lifespan starts its loop.
    world = mount_world(app, load_infer=load_policy_infer, origin_allowed=origin_allowed)

    @app.get("/scene")
    def get_scene(robot: str = "microduck") -> dict:
        """Visual meshes for a body in the roster.

        The duck's scene is precompiled at startup; every other body builds
        its own on demand and caches it (`Body.visual_scene` — the G1's mesh
        dump is ~21 MB of millimetre ints, MARS's is 9 meshes of STL, a
        Menagerie model's is whatever its visual group holds).

        The 404 carries the SETUP HINT, because "unknown robot" and "you have
        not downloaded it yet" are different problems and the palette's ⤓
        button is the answer to only one of them."""
        if robot in ("", "microduck", "duck"):
            return scene
        try:
            return lab_robots.robot_scene(robot)
        except FileNotFoundError as e:              # known body, no assets
            raise HTTPException(404, str(e))
        except KeyError:
            raise HTTPException(404, f"unknown robot {robot!r}")

    @app.get("/policies")
    def get_policies() -> dict:
        policies = discover_policies()
        # `shipped` names the palette's shipped SECTIONS and their headings.
        # The viewer used to hold that list itself, so a third body would
        # have meant a fourth literal in a TypeScript file.
        return {"policies": policies, "robots": available_robots(),
                "shipped": lab_robots.shipped_groups(),
                "tricks": trick_names(policies)}

    @app.post("/robots/{robot}/fetch")
    def fetch_robot(robot: str) -> dict:
        """Download a robot's assets from the palette. Returns at once; poll
        GET /policies until the robot reports `ready`.

        Any body with a `fetch()`: the G1's ~140 MB clone, MARS's 7.2 MB of
        STLs, or a Menagerie model — a `menagerie:<name>` id routes to the
        NAMESPACE's own download, because at that moment there is no body in
        the cache to ask (`lab/robots.start_fetch`, and `fetch_robot.py`'s
        CLI has the same branch for the same reason)."""
        try:
            return lab_robots.start_fetch(robot)
        except KeyError as e:
            raise HTTPException(404, str(e))

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

    # ------------------------------------------------ 🧠 brain training runs

    # train-brain is a plain CLI process: nothing about it reaches the lab, so
    # a brain run has always been invisible while a teach job is watchable.
    # This reads the artifacts the trainer already writes — brain.json (the
    # contract) and progress.jsonl (one row per rollout) — so the viewer's
    # /train page can chart a run live without the trainer knowing about it.
    # Read-only and derived entirely from disk: nothing here can steer a run.
    BRAIN_ACTIVE_S = 30.0     # a progress file touched this recently is live
    BRAIN_MAX_POINTS = 400    # downsample the curve; 2M steps is ~1300 rows

    @app.get("/brains")
    def get_brains() -> dict:
        try:
            root = brains_dir()
        except Exception:
            return {"brains": []}
        if not root.is_dir():
            return {"brains": []}
        now = time.time()
        out = []
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            card: dict = {"name": d.name}
            try:
                card.update(json.loads((d / "brain.json").read_text()))
            except Exception:
                pass                      # a run mid-flight may not have one yet
            card["shipped"] = (d / "brain.onnx").exists()
            prog = d / "progress.jsonl"
            rows = []
            if prog.is_file():
                try:
                    for ln in prog.read_text().splitlines():
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            rows.append(json.loads(ln))
                        except json.JSONDecodeError:
                            continue      # a half-written last line while training
                except OSError:
                    rows = []
                card["active"] = (now - prog.stat().st_mtime) < BRAIN_ACTIVE_S
            else:
                card["active"] = False
            card["rollouts"] = len(rows)
            if rows:
                last = rows[-1]
                card["last"] = last
                done, want = last.get("steps") or 0, card.get("steps") or 0
                el = last.get("elapsed_s") or 0
                card["progress"] = min(1.0, done / want) if want else None
                card["steps_per_s"] = round(done / el, 1) if el else None
                card["eta_s"] = (round((want - done) / (done / el))
                                 if card["active"] and el and done and want > done else None)
            # Downsample evenly, always keeping the first and last row.
            if len(rows) > BRAIN_MAX_POINTS:
                step = len(rows) / BRAIN_MAX_POINTS
                idx = sorted({int(i * step) for i in range(BRAIN_MAX_POINTS)} | {len(rows) - 1})
                rows = [rows[i] for i in idx]
            card["curve"] = [{"steps": r.get("steps"), "ep_rew": r.get("ep_rew"),
                              "ep_len": r.get("ep_len"), "elapsed_s": r.get("elapsed_s")}
                             for r in rows]
            out.append(card)
        return {"brains": out}

    # ------------------------------------------------ 🎬 animation authoring

    @app.get("/robots")
    def get_robots() -> dict:
        """Every body, for the 🎬 editor and the 🎓 panel.

        `ready` is each body's own answer now (`Body.ready`). It used to be
        `True if rid == "microduck" else bool(g1_ready())`, so the moment a
        THIRD body was registered it inherited the G1's download state — and
        the endpoint then hand-patched an absent G1 back onto the end of the
        list it had just built, because `registry()` drops a body whose
        assets are missing.

        `animate` says whether the 🎬 pose editor can open it: `PoseScratch`
        is a WALKER's tool (effectors, soles, a base body) and
        `pose_scratch("mars")` raised `AttributeError: no attribute
        'base_body'` — so the panel filters by CAPABILITY, which is what the
        flag carries (`lab/robots._animates`). `kind` lets the viewer pick an
        emoji and a verb without a table of ids."""
        return {"robots": lab_robots.robot_entries()}

    @app.get("/joints")
    def get_joints(robot: str = "microduck") -> dict:
        return scratch_for(robot).meta()

    @app.post("/pose")
    def post_pose(req: PoseReq, robot: str = "microduck") -> dict:
        scratch = scratch_for(robot)
        try:
            joints = clean_joints(req.joints, scratch.spec.id)
        except ValueError as e:
            raise HTTPException(422, str(e))
        try:
            pitch = _finite(req.rootPitch)
        except ValueError:
            raise HTTPException(422, "rootPitch must be a finite number")
        bodies = scratch.solve(np.array(joints), pitch, req.ground)
        return {
            "robot": scratch.spec.id,
            "bodies": bodies,
            "joints": joints,       # clamped — the editor snaps its sliders to these
            "rootPitch": round(pitch, 6),
            # Read straight after solve, off the same posed mjData.
            "balance": scratch.balance(),
            # Where each draggable point is for THIS pose — the IK handles.
            "effectors": scratch.effector_positions(),
        }

    @app.post("/ik")
    def post_ik(req: IkReq, robot: str = "microduck") -> dict:
        """Inverse kinematics for a dragged foot or hand: the /pose answer
        for the joints that put every target where it was asked, with the
        other feet held on the spot. Targets are in the frame /pose reports
        bodies in. `ik.residual` says, per effector, how far short the limb
        fell (metres) — an unreachable target is answered with the closest
        pose, never an error."""
        scratch = scratch_for(robot)
        try:
            joints = clean_joints(req.joints, scratch.spec.id)
            pitch = _finite(req.rootPitch)
        except ValueError as e:
            raise HTTPException(422, str(e))
        targets = {}
        for eid, t in req.targets.items():
            if eid not in scratch.effectors:
                raise HTTPException(422, f"no effector named {eid!r} on the "
                                         f"{scratch.spec.id} (have "
                                         f"{sorted(scratch.effectors)})")
            if len(t.pos) != 3:
                raise HTTPException(422, f"target {eid!r}: pos must be [x, y, z]")
            try:
                pos = tuple(_finite(v) for v in t.pos)
            except ValueError:
                raise HTTPException(422, f"target {eid!r}: pos must be finite")
            level = (scratch.effectors[eid]["kind"] == "foot"
                     if t.level is None else bool(t.level))
            targets[eid] = IkTarget(pos=pos, level=level,
                                    weight=float(np.clip(t.weight, 0.05, 20.0)))
        if req.pins is None:
            pins = tuple(eid for eid, e in scratch.effectors.items()
                         if e["kind"] == "foot" and eid not in targets)
        else:
            bad = [p for p in req.pins if p not in scratch.effectors]
            if bad:
                raise HTTPException(422, f"no effector named {bad[0]!r}")
            pins = tuple(req.pins)
        res = scratch.solve_ik(np.array(joints), pitch, targets, pins=pins,
                               ground=req.ground)
        out_joints = [round(float(v), 6) for v in res.joints]
        bodies = scratch.solve(np.array(out_joints), pitch, req.ground)
        return {
            "robot": scratch.spec.id,
            "bodies": bodies,
            "joints": out_joints,
            "rootPitch": round(pitch, 6),
            "balance": scratch.balance(),
            "effectors": scratch.effector_positions(),
            "ik": {"iterations": res.iterations, "converged": res.converged,
                   "residual": res.residual, "pins": list(pins)},
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
        # WHICH BODY is being taught. The trainee's own robot wins (it is the
        # duck the panel is watching); otherwise the roster decides, so a lab
        # holding only a G1 offers G1 tasks without anyone selecting anything.
        # Duck recipes name duck joints, duck feet and the 61-obs command
        # slots — they are not portable, so the registry is filtered rather
        # than the trainer being asked to cope.
        trainee = st.trainee()
        robots = {getattr(d, "robot", "microduck") for d in st.ducks
                  if d.id != "trainee"}
        robot = (getattr(trainee, "robot", None)
                 or (robots.pop() if len(robots) == 1 else "microduck"))
        # The panel's robot switch is an explicit choice and beats the guess
        # — a lab holding a duck AND a G1 could otherwise only ever teach
        # whichever body the trainee happened to be.
        if req.robot:
            if not behaviors_mod.for_robot(req.robot):
                return {"matched": False,
                        "message": f"I don't have any recipes for {req.robot!r} yet."}
            robot = req.robot
        # A clip names its own body. "⚡ train this" on a G1 clip is a G1
        # task whatever the roster is standing on — the job rebuilds the
        # trainee onto the clip's body, as it does for any other task.
        if req.clip:
            try:
                if clip_path(req.clip).exists():
                    robot = clip_robot(req.clip)
            except ValueError:
                pass
        b, title_clip = match_teach_text(req.text, robot)
        if b is None:
            what = ("trick" if robot == "microduck" else f"{robot} task")
            return {"matched": False,
                    "message": f"I don't know that {what} yet. I can teach these — "
                               "new ones need a recipe added to the "
                               "behaviors/ package:",
                    "behaviors": [behaviors_mod.behavior_card(x)
                                  for x in behaviors_mod.for_robot(robot)]}
        # The clip rides MICRODUCK_CLIP into the trainer AND (via stage_env)
        # into the preview envs, where motion.load_clip joins it straight onto
        # clips_dir() — so it gets the same restrictive validation every clip
        # endpoint uses, and must actually exist. Unchecked, "../x" read JSON
        # outside clips/, and a typo'd name reported a healthy job whose
        # workers all died at env construction with FileNotFoundError.
        init_from = None
        if req.initFrom:
            try:
                init_from = resolve_init_from(req.initFrom)
            except ValueError as e:
                return {"matched": False, "message": str(e)}
        # Which clip: the request's, else the one its title names, else — for
        # a fine-tune — the one the run being refined actually trained on.
        # The recipe's default clip is the last resort, never a silent swap
        # away from the motion a finished run was built around.
        clip = req.clip or title_clip or (run_clip(init_from) if init_from else None)
        if not clip and b is imitation_behavior(robot) and not getattr(b, "clip_name", None):
            return {"matched": False,
                    "message": f"“{b.title}” needs a clip to copy — open one in "
                               "the 🎬 animate panel and press ⚡ train this."}
        if clip:
            try:
                if not clip_path(clip).exists():
                    return {"matched": False,
                            "message": f"I don't have a clip named “{clip}” — "
                                       "save it in the 🎬 animate panel first."}
            except ValueError as e:
                return {"matched": False, "message": str(e)}
            # The recipe and the clip must pose the same body: a duck recipe
            # resamples 14 joints and would raise inside every worker on a
            # 29-joint clip, with nothing in the panel saying why.
            body = clip_robot(clip)
            if body != getattr(b, "robot", "microduck"):
                return {"matched": False,
                        "message": f"“{clip}” is a {body} clip and “{b.title}” "
                                   f"trains the {getattr(b, 'robot', 'microduck')} "
                                   "— pick the clip's own body in the 🎬 panel."}
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
        # (retrain, or a scripted call) win and become the new sticky set —
        # except for a fine-tune, whose base is the run it continues.
        sticky = load_teach_weights()
        prev_sticky = sticky.get(b.id, empty_sticky())
        if init_from is not None:
            # A fine-tune continues THAT run's brain, so its recipe is the
            # base and the request's weights layer over it (per key) — not
            # the sticky set, and not a replacement. Before this the panel's
            # "touched sliders only" dict replaced everything: a run seated
            # from /teach/load with rotation_match at 6.05 was fine-tuned
            # after adding one catalog term, and rotation_match silently
            # dropped back to the recipe's 4.00 with no message.
            weights = {**run_trained_weights(init_from),
                       **(req.weights or {})} or None
        else:
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
            extra_env=({"MICRODUCK_CLIP": clip} if clip else None),
            budget=budget,
            stage_budgets=stage_budgets,
        )
        snapshot_skipped.clear()   # a new run, a new set of bodies to refuse
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
        zero = _zero_infer_for(b.robot)
        if st.trainee() is None:
            st.ducks.append(Duck("trainee", label, zero, seed=97,
                                 onnx_path=live, env_kwargs=ekw, robot=b.robot))
        else:
            st.trainee().set_robot(b.robot)
            st.trainee().rebuild_env(ekw)
            st.trainee().swap_policy(label, zero, onnx_path=live)
        # Helpers are clones of the trainee, so they follow it to the job's
        # BODY exactly as it does — set_robot, rebuild, re-brain. This used
        # to skip every helper unless the job was a duck's ("rebuilding a
        # duck helper into a G1 env would step its 14-action brain in a
        # 29-joint robot"), which is true of a rebuild ALONE; swapping the
        # brain in the same breath is what makes it safe. Skipping left a
        # 61-obs helper on the roster of a 99-obs run — stepping the previous
        # job's ONNX, and first in line for that run's snapshots.
        for h in helper_ducks(st.ducks):
            h.set_robot(b.robot)
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
            h.swap_policy(h.label, zero, onnx_path=live)
        st.events.append(f"Training started: {b.emoji} {b.title}")
        sp = st.job.stage_payload()
        if sp:  # curriculum chain: name the opening stage right away
            st.events.append(
                f"Training stage {sp['idx']}/{sp['count']} — {sp['label']}")
        save_lab_state(st.ducks)
        return {"matched": True, "job": st.job.payload()}

    @app.get("/teach/status")
    def teach_status() -> dict:
        """Is a job running, and how far along? The AUTHORITATIVE answer.

        The lab owns the trainer subprocess, so it is the only thing that
        knows — `TrainingJob._poll` reads `proc.poll()`. Anything outside had
        no way to ask, so scripts (and agents) fell back to grepping the
        process table, which is unreliable in a way that bites silently: a
        `pgrep -f "microduck_local.train "` matches the very shell running the
        pgrep, and `pgrep -f "Python.*microduck_local"` matches THIS server,
        whose venv path contains the string. Both report "a trainer is
        running" forever, and a wait loop built on either never ends.

        Poll this instead: `running` is the one field a waiter needs.
        """
        j = st.job
        if j is None:
            return {"running": False, "status": "idle", "job": None}
        prog = j.progress or {}
        return {
            "running": j.status == "training",
            "status": j.status,          # training | done | stopped | failed
            "job": {
                "behavior": j.behavior.id,
                "title": j.display_title(),
                "runName": j.run_name,
                "stage": j.stage_idx + 1 if j.stages else None,
                "stages": len(j.stages) or None,
                "steps": prog.get("steps"),
                "total": prog.get("total"),
            },
        }

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
                if "spawn_robot" in msg:
                    sr = msg["spawn_robot"]
                    rid = sr.get("robot") if isinstance(sr, dict) else sr
                    asyncio.create_task(
                        do_spawn_robot(str(rid) if rid else ""))
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
        want_robot = (entry or {}).get("robot") or policy_robot(path)
        if want_robot != duck.robot:
            duck.set_robot(want_robot)
            st.events.append(f"{duck_id} is now a {want_robot}")
        # Showcase = the "whole trick" assign: rehearse spawns across the
        # whole trick arc (final-stage knobs). Quietly a plain assign when
        # the policy has no curriculum behind it — the flag can't mean
        # anything there, and refusing would make the chip feel broken.
        skw = showcase_env_kwargs(path) if showcase else None
        run_name = policy_id.split(":", 1)[-1]
        if skw is None:
            # What the ROSTER ROW reads: the run's own title when it has one
            # (run_record.py). That column was a list of
            # `teach-…-<hash>-sN` strings, which is where "which one of these
            # is the good one?" started — and the answer was on disk. The
            # identifier is unchanged (`policy_id`), and the event line below
            # still names it, so the log stays greppable.
            label = (entry or {}).get("title") or run_name
            duck.rebuild_env(env_kwargs_for_policy_path(path))
        else:
            label = showcase_label(policy_id, bool(skw.get("spotter")))
            duck.rebuild_env(skw)
        duck.showcase = skw is not None
        # Does this policy speak this body? The policy's own CONTRACT when
        # the file (or its run) records one, and the graph's width as the
        # last-ditch check on a file that says nothing — one call now, where
        # this was four copies of a width compare (docs/mars-roadmap.md
        # §6.3). A 99-d G1 brain in a 61-d duck env used to load, run and
        # emit nonsense; two bodies of the SAME width would still have
        # crossed silently, which is what the contract id fixes.
        why = lab_robots.policy_refusal(
            path, duck.robot, getattr(infer, "obs_dim", None),
            int(duck.env.observation_space.shape[0]))
        if why:
            st.events.append(
                f"assign failed: {policy_id} {why} ({duck_id})")
            return
        duck.swap_policy(label, infer, policy_id=policy_id)
        ho = handoff_for(path) if skw is not None else None
        duck.handoff_infer, duck.handoff_label = ho if ho else (None, None)
        duck.handed = False
        st.events.append(f"{duck_id} now runs {label}"
                         + (f" ({run_name})" if label != run_name else ""))
        save_lab_state(st.ducks)

    async def do_spawn_helper() -> None:
        err = spawn_helper_error(st)
        if err:
            st.events.append(f"spawn_helper ignored: {err}")
            return
        st.scaling = True
        n = next_helper_slot(st.ducks)
        try:
            job = st.job
            # WHICH BODY — the job's, same as the 🎓 trainee's. A helper is a
            # clone of the trainee, and without this it was built as a duck
            # whatever was being taught: for a G1 task `trainee_env_kwargs`
            # returns the env's `task=`, which only `_make_env`'s non-duck
            # branch pops, so MicroduckWalkEnv got an unexpected keyword and
            # raised — inside an asyncio task nobody awaits. The + button did
            # nothing at all and said nothing, which is how it was reported.
            robot = getattr(job.behavior, "robot", "microduck") or "microduck"
            live = job.dir / "live.onnx"
            try:
                infer = await asyncio.to_thread(_onnx_infer, live)
                onnx_path = str(live)
            except Exception:
                # The guard saw model.zip so live.onnx should exist; if a write
                # races us, the helper idles until the next snapshot lands.
                # _zero_infer_for, not _zero_infer: the latter emits 14 floats
                # for every body and a 29-joint env rejects that outright.
                infer, onnx_path = _zero_infer_for(robot), None
            # Same stage-mirrored preview env as the trainee: helpers rehearse
            # the same run, and a helper standing calmly next to a trainee
            # dropped mid-roll reads as "the helpers didn't get the fixes"
            # (it did to the user who spotted exactly that).
            helper = Duck(f"helper{n}", f"🤝 helper {n}", infer,
                          seed=100 + n, onnx_path=onnx_path, robot=robot,
                          env_kwargs=trainee_env_kwargs(
                              job.behavior, job.stage_env()))
            # The same contract check do_assign/do_spawn_duck carry: a
            # mismatch is a message, not an ONNX raise inside the 50 Hz loop
            # — which stops the loop for every duck, not just this one.
            why = lab_robots.policy_refusal(
                onnx_path, robot, getattr(infer, "obs_dim", None),
                int(helper.env.observation_space.shape[0]))
            if why:
                st.events.append(
                    f"helper {n} not spawned: the run's brain {why}")
                return
            st.ducks.append(helper)
            save_lab_state(st.ducks)
            # Visual only — do NOT job.scale(). A warm restart would stall
            # training for seconds and, with ENVS_PER_HELPER=0, would not
            # even change --envs. job.helpers tracks the roster for the HUD.
            job.helpers = len(helper_ducks(st.ducks))
            st.events.append(
                f"helper {n} joined — watching the same brain "
                f"({job.envs} train envs)")
        except Exception as e:
            # LOUD, always. This coroutine runs under create_task with nobody
            # awaiting it, so anything raised here used to vanish: the user
            # pressed + and the lab did nothing and reported nothing.
            st.events.append(f"helper {n} not spawned: {type(e).__name__}: {e}")
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
        # WHICH BODY. Without this a 99-obs G1 brain landed in a 61-obs duck
        # env and the ONNX session raised inside the 50 Hz loop — which is
        # FATAL: the loop stops and the lab streams no frames at all (the
        # viewer goes blank, not "one bad duck").
        robot = (entry or {}).get("robot") or policy_robot(path)
        duck = Duck(f"d{n}", label, infer, seed=37 + n,
                    policy_id=policy_id, robot=robot,
                    env_kwargs=(skw if skw is not None
                                else env_kwargs_for_policy_path(path)))
        why = lab_robots.policy_refusal(
            path, robot, getattr(infer, "obs_dim", None),
            int(duck.env.observation_space.shape[0]))
        if why:
            st.events.append(f"spawn failed: {policy_id} {why}")
            return
        duck.showcase = skw is not None
        ho = handoff_for(path) if skw is not None else None
        duck.handoff_infer, duck.handoff_label = ho if ho else (None, None)
        st.ducks.append(duck)
        save_lab_state(st.ducks)
        st.events.append(f"spawned d{n} running {label}")

    # The roster mutations the WebSocket drives, reachable without a socket:
    # the same handlers, so a test exercises what the viewer actually calls
    # (see app.state.lab above).
    async def do_spawn_robot(robot: str) -> None:
        """Put one `robot` on the stage with NO policy — it idles.

        The palette's ＋ for a body that has nothing to assign yet: MARS
        before anyone has trained it, and every Menagerie model. Without it
        the only way onto the stage was a policy chip, so the two bodies most
        worth LOOKING at first were the two that could not be shown at all.
        """
        err = spawn_robot_error(st, robot)
        if err:
            st.events.append(f"spawn_robot ignored: {err}")
            return
        n = next_duck_slot(st.ducks)
        try:
            duck = Duck(f"d{n}", f"{lab_robots.robot_title(robot)}",
                        _zero_infer_for(robot), seed=37 + n, robot=robot)
        except Exception as e:
            # Assets gone, an MJCF that will not compile: LOUD, because this
            # coroutine runs under create_task with nobody awaiting it and a
            # silent raise is a button that does nothing and says nothing.
            st.events.append(
                f"{robot} not spawned: {type(e).__name__}: {e}")
            return
        st.ducks.append(duck)
        save_lab_state(st.ducks)
        st.events.append(f"spawned d{n} — a {lab_robots.robot_noun(robot)} "
                         "with no brain yet")

    app.state.do_assign = do_assign
    app.state.do_spawn_duck = do_spawn_duck
    app.state.do_spawn_robot = do_spawn_robot
    app.state.do_spawn_helper = do_spawn_helper

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
        want_obs = getattr(infer, "obs_dim", None)
        for d in targets:
            # WIDTH FIRST. This fanned the run's brain onto every id starting
            # with "helper" unconditionally; a 99-obs snapshot handed to a
            # 61-obs duck raises inside the 50 Hz loop, which is FATAL — the
            # lab streams nothing and the viewer goes blank with no error
            # anywhere. Helpers follow the job's body now (see /teach), so
            # this should never fire; it costs one compare and it is the
            # difference between one parked duck and a dead lab.
            #
            # `live.onnx` IS stamped with its contract (`export()` writes the
            # metadata props, and the trainer's snapshot goes through the
            # same exporter), so the first rung of `policy_refusal` does the
            # work here and the width is the backstop.
            why = lab_robots.policy_refusal(
                live, d.robot, want_obs,
                int(d.env.observation_space.shape[0]))
            if why:
                if d.id not in snapshot_skipped:
                    # Once per duck per job: this runs at every snapshot, and
                    # a line per snapshot would bury the rest of the chat.
                    snapshot_skipped.add(d.id)
                    st.events.append(
                        f"{d.id} ({d.robot}) is not following this run — "
                        f"the brain {why}")
                continue
            # display_title(), not behavior.title: an imitation run is about a
            # SPECIFIC authored clip, and the launch label already says so —
            # rebuilding from the generic recipe name here quietly renamed the
            # duck back to "Copy the animation" at the first snapshot.
            label = (f"🎓 {job.display_title()} @{steps // 1000}k"
                     if d.id == "trainee" else d.label)
            d.swap_policy(label, infer, onnx_path=live)
        st.events.append(f"Trainee updated to {steps // 1000}k steps")

    # Driven by the lab loop rather than the socket, but exposed for the same
    # reason: a test has to be able to run the real snapshot fan-out.
    app.state.apply_snapshot = apply_snapshot

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
            doomed: list[Duck] = []
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
                try:
                    d.tick()
                except Exception as e:
                    # ONE duck must never take the lab down. A brain whose
                    # observation width does not match its body raises here
                    # (ONNX: "Got: 61 Expected: 99"), and before this the
                    # exception propagated out of the 50 Hz loop: the loop
                    # stopped, the socket went quiet and the viewer showed an
                    # empty stage with no explanation anywhere. Park the duck
                    # instead and say so.
                    st.events.append(
                        f"{d.id} stopped: {type(e).__name__}: {e}"[:200])
                    print(f"[lab] {d.id} raised in tick(), removing it: {e}")
                    doomed.append(d)
            for d in doomed:
                if d in st.ducks:
                    st.ducks.remove(d)
            if doomed:
                save_lab_state(st.ducks)
                doomed.clear()
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
                # Stage layout, once per frame — every slot pitched by its own
                # robot. The viewer draws what this says and only falls back to
                # its own duck-pitched grid for a server that predates it.
                slots = lab_slot_offsets(
                    [getattr(d, "robot", "microduck") for d in st.ducks])
                frame = json.dumps({
                    "cmd": [round(float(v), 3) for v in cmd],
                    "mode": mode,
                    "stats": st.stats,
                    "training": st.job.payload() if st.job else None,
                    "events": list(st.events)[-5:],
                    "ducks": [{
                        "id": d.id,
                        "name": d.label,
                        # Where this slot stands on the floor, MuJoCo XY m.
                        "offset": [round(v, 4) for v in slots[i]],
                        # Which body the viewer should draw for this row
                        # (any registry id); the meshes come from
                        # GET /scene?robot=<id>.
                        "robot": getattr(d, "robot", "microduck"),
                        # Brain provenance ("run:<name>", "ckpt:…", "pollen:…",
                        # or null) — lets the viewer load a selected duck's
                        # run into the teach panel (POST /teach/load).
                        "policy": d.policy_id,
                        "falls": d.falls,
                        # `steers()` FIRST: a trick policy is one a duck
                        # ignores commands on, but an arm env and a
                        # kinematic idle have no command channel at all, so
                        # the honest answer for them is the same "no" for a
                        # different reason — and asking the id instead of the
                        # env is what would put `twist_cmd` on a MARS.
                        "steerable": d.steers() and not is_trick_duck(d),
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
                        "cmdSpeed": (round(float(d.env.twist_cmd[0]), 3)
                                     if d.steers() and not is_trick_duck(d)
                                     else None),
                        "spawn": getattr(d.env, "last_spawn", None),
                        "assist": bool(getattr(d.env, "spotter_active", False)),
                        "handed": bool(getattr(d, "handed", False)),
                        "handoff": getattr(d, "handoff_label", None),
                        # Task objects that live in the env, not the
                        # physics: the find_ball ball as [x, y, z, r] so
                        # the viewer can draw what the duck is looking for.
                        "ball": behaviors_mod.ball_marker_payload(d.env),
                        "bodies": d.pose_payload(),
                    } for i, d in enumerate(st.ducks)],
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
    ap.add_argument("--world", default=None, metavar="SCENARIO",
                    help="preload a /sim world (a built-in or scenarios/<name>.json); "
                         "with --world the roster may be empty")
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
    app = make_app(ducks)
    if args.world:
        app.state.world.preload(args.world)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
