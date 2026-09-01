"""Reference motion clips: keyframed poses the policy learns to physically
execute (motion imitation, the DeepMimic idea).

This is the bridge between authoring and reinforcement learning. A clip is a
keyframed animation — exactly what an animator would make — and the training
reward is simply "be in the reference pose for right now". That converts an
open-ended search ("discover a backflip") into a tracking problem ("follow
this"), which is how most impressive robot flips are actually trained: the
choreography is authored, and RL solves the physics of executing it.

Clips are written by the viewer's timeline editor; the JSON contract lives in
`load_clip`. Nothing here touches the 61-obs contract — see `phase_signal`
for how the policy is told where it is in the clip.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

from . import contract as C

CONTROL_HZ = 1.0 / (0.005 * C.DECIMATION)  # env control rate (50 Hz)


def clips_dir() -> Path:
    env = os.environ.get("MICRODUCK_CLIPS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "clips"


@dataclass(frozen=True)
class Clip:
    """A reference motion resampled onto the control grid.

    `joints` is (T, 14) absolute joint angles in JOINT_NAMES order; `pitch`
    is (T,) intended body pitch in radians in the EDITOR/SIM convention:
    right-handed about the trunk's +Y, so NEGATIVE means leaning back and a
    backflip ramps toward -2π. Note this is the opposite sign to the
    backflip rotation accumulator (which counts backward rotation as
    positive) — the reward negates one of them; a first draft of this file
    had them agreeing in sign, which would have trained a FRONT flip.
    """
    name: str
    joints: np.ndarray
    pitch: np.ndarray
    loop: bool = False

    @property
    def steps(self) -> int:
        return int(self.joints.shape[0])

    @property
    def duration(self) -> float:
        return self.steps / CONTROL_HZ

    def at(self, step: int) -> tuple[np.ndarray, float]:
        """Reference pose for a control step. A LOOPING clip (a gait) wraps —
        the cycle just keeps going; a one-shot clip (a trick) holds its final
        pose, because the clip describes the maneuver, not the rest of the
        episode."""
        i = (int(step) % self.steps if self.loop
             else int(np.clip(step, 0, self.steps - 1)))
        return self.joints[i], float(self.pitch[i])

    def phase(self, step: int) -> tuple[float, float]:
        """(sin, cos) of where we are in the clip — wrapping per cycle for a
        loop, running once for a one-shot. See `phase_signal`."""
        return phase_signal(int(step) % self.steps if self.loop else int(step),
                            self.steps)


def load_clip(name: str, directory: Path | None = None) -> Clip:
    """Read a clip authored by the timeline editor and resample it to the
    control rate. Format (version 1):

        {"version": 1, "name": ..., "duration": 1.6, "loop": false,
         "keys": [{"t": 0.0, "joints": [14 floats], "rootPitch": 0.0}, ...]}

    Keys are linearly interpolated in joint space; `t` is seconds from the
    clip start and must ascend from 0.
    """
    path = (directory or clips_dir()) / f"{name}.json"
    data = json.loads(path.read_text())
    keys = sorted(data["keys"], key=lambda k: float(k["t"]))
    if not keys:
        raise ValueError(f"clip {name!r} has no keys")
    times = np.array([float(k["t"]) for k in keys])
    poses = np.array([np.asarray(k["joints"], dtype=np.float64) for k in keys])
    if poses.shape[1] != C.NUM_JOINTS:
        raise ValueError(f"clip {name!r}: keys must carry {C.NUM_JOINTS} joints")
    pitches = np.array([float(k.get("rootPitch", 0.0)) for k in keys])
    duration = float(data.get("duration") or times[-1])
    n = max(int(round(duration * CONTROL_HZ)), 1)
    grid = np.arange(n) / CONTROL_HZ
    joints = np.stack([np.interp(grid, times, poses[:, j])
                       for j in range(C.NUM_JOINTS)], axis=1)
    pitch = np.interp(grid, times, pitches)
    return Clip(name=str(data.get("name", name)),
                joints=joints.astype(np.float64), pitch=pitch.astype(np.float64),
                loop=bool(data.get("loop", False)))


def phase_signal(step: int, total: int) -> tuple[float, float]:
    """Where we are in the clip, as (sin, cos) of the phase angle.

    The policy is a memoryless network reading a FIXED 61-dim observation, so
    it cannot know the time — and a reference pose is meaningless without it
    (the same body pose maps to different targets at different moments). The
    two values go into command slots that trick training otherwise fills with
    keep-alive noise, so the observation contract is untouched: a deployed
    runtime feeds the phase there instead of noise while the trick plays.
    """
    p = 2.0 * np.pi * float(np.clip(step / max(total, 1), 0.0, 1.0))
    return float(np.sin(p)), float(np.cos(p))


# --------------------------------------------------------------- hot reloading
#
# The lab reloads recipe code in-process on every /teach so an edit takes
# effect without restarting the server. `importlib.reload` re-executes a module
# in its LIVE dict, which is exactly wrong for that: an edit that compiles but
# raises part-way leaves the module HALF-NEW — names bound before the raise
# hold the new values, names after it are still the old ones, and a name the
# edit deleted keeps answering. The lab then serves a Frankenstein module that
# the trainer subprocess (which always imports fresh) will never run, and the
# two disagree with no error anywhere.
#
# The generic helper lives in this module rather than in behaviors/ because the
# dependency runs behaviors -> motion (behaviors/core.py does
# `from .. import motion`); importing it the other way would be a cycle.


def reload_modules(modules: list[ModuleType],
                   after: Callable[[], None] | None = None) -> list[ModuleType]:
    """Reload `modules`, in list order, all-or-nothing.

    Three guarantees, each paid for by a failure this harness actually hit:

    1. A SYNTAX ERROR MUTATES NOTHING. Every source is compiled before any
       namespace is touched, so the commonest bad edit costs nothing at all.
    2. NO GHOSTS. Each namespace is emptied before its module re-executes, so
       a name the edit deleted or renamed is really gone. A plain reload leaves
       it behind, and the behaviors modules' `__all__ = [n for n in dir()]`
       footers then re-export the ghost down the whole star-import cascade.
    3. NO TORN STATE, ACROSS THE WHOLE LIST. A module that raises while
       EXECUTING has EVERY module's previous namespace put back before the
       error is re-raised. Restoring only the module that failed is not enough
       for a dependent group: behaviors' core reloads first and rebinds
       BEHAVIORS/CATALOG, so the modules after the failure never re-register
       into those fresh dicts — leaving `match_behavior` denying tricks the
       catalog still lists.

    `after` runs INSIDE that protection. A caller whose reload is only complete
    once it has re-derived something from the fresh modules (the behaviors
    package re-flattens its namespace) needs a failure there to roll the
    modules back too — otherwise it keeps old derived state sitting on top of
    new modules, which is the same split brain guarantee 3 exists to prevent.

    `modules` is updated IN PLACE with whatever reload returned, before `after`
    runs, so an `after` that walks the same list sees the fresh modules.
    """
    # Local aliases: reloading THIS module empties motion's own globals halfway
    # through this call, and only dunders survive that — so `importlib` has to
    # be captured here, while builtins (open/compile/vars/...) keep resolving
    # through the surviving `__builtins__`.
    imp = importlib
    for m in modules:
        path = getattr(m, "__file__", None)
        if path:
            with open(path, encoding="utf-8") as f:
                compile(f.read(), path, "exec")  # SyntaxError mutates nothing
    snapshots = [{k: v for k, v in vars(m).items() if not k.startswith("__")}
                 for m in modules]
    originals = list(modules)
    live = list(modules)   # what each slot holds NOW — the rollback's targets
    try:
        for i, m in enumerate(originals):
            for k in snapshots[i]:
                delattr(m, k)
            new = imp.reload(m)
            # A no-op reload (tests monkeypatch importlib.reload to keep class
            # identity stable across /teach calls) must not gut the module.
            if not any(not k.startswith("__") for k in vars(new)):
                vars(new).update(snapshots[i])
            live[i] = modules[i] = new
        if after is not None:
            after()
    except BaseException:
        # Restoring in list order matters: a module the later ones register
        # into must hold its pre-reload objects again before they go back.
        for m, snap in zip(live, snapshots):
            for k in [k for k in vars(m) if not k.startswith("__")]:
                delattr(m, k)
            vars(m).update(snap)
        raise
    return modules


def reload_self() -> None:
    """Reload THIS module without the torn state a plain reload leaves.

    The lab's /teach reloads motion BEFORE behaviors, because behaviors calls
    into it (reloading only behaviors left it calling a stale clip module,
    which killed every preview duck the moment a Clip method was added). That
    call used `importlib.reload` directly, so a clip edit that raised part-way
    left the lab holding half of each version — see reload_modules.
    """
    reload_modules([sys.modules[__name__]])
