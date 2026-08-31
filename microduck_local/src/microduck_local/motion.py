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

import json
import os
from dataclasses import dataclass
from pathlib import Path

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
