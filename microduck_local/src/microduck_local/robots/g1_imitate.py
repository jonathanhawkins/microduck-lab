"""Motion imitation on the Unitree G1: track a clip authored in the 🎬 panel.

The kick that reward search could not find (docs/roadmap.md, track 13.3: six
formulations, every one converging to a ~0.1 m shin-high kick from a policy
that survives) is a TRACKING problem once someone has drawn the kick. This
env is the G1's `behaviors/imitate.py`: the reward is "be where the clip says
you should be right now", and the policy is told where "now" is through the
three command slots it is otherwise paid to ignore (the same clock the karate
cycle used — a memoryless network cannot time a motion without one).

Everything here is a RETARGETED term of `G1StandEnv`, never a switched-off
one (AGENTS.md: five retrains in one day taught that lesson). Standing still
is the clip that happens to be constant:

    pose        — joints vs the clip's joints this instant, wide     (was: default)
    carriage    — the same, TIGHT, over every joint                  (was: default)
    height      — pelvis vs the clip's pelvis, GROUNDED by FK        (was: stand_z)
    upright     — projected gravity vs the clip's rootPitch          (was: level)
    feet        — a foot the clip plants is down and flat; a foot    (was: planted
                  the clip lifts is paid PROGRESS toward its height        + flat_feet)
    foot_track  — each foot's position in the pelvis frame vs the    (new: Cartesian,
                  clip's — prices the kick's REACH, which 29 joint         the term the
                  angles at std2 4.0 price at nothing)                     kick lacked)
    still / no_spin / stay_home / penalties — as the idle.

The reference is precomputed once per clip: every 50 Hz frame is posed on a
scratch MjData, grounded on its lowest sole, and the pelvis height, the feet's
pelvis-frame positions, each foot's air height and the projected gravity are
read off it. No live term ever solves anything in the root frame — the pelvis
IS the free joint, and a target there measures nothing (the squat solver's
trap, docs/roadmap.md 13.1).
"""

from __future__ import annotations

import math
import os

import numpy as np

from .. import motion
from .g1_env import G1StandEnv


class G1ImitateEnv(G1StandEnv):
    """Track a saved clip. `clip_name` (or MICRODUCK_CLIP, the trainer's
    channel) names a file in clips/ whose `robot` is "g1"."""

    # --- the earners (AGENTS.md: check each pays SOMETHING at the start) ----
    W_POSE = 2.0
    POSE_STD2 = 4.0            # wide: a gradient from wherever the body starts
    W_CARRIAGE = 3.0
    CARRIAGE_STD2 = 0.8        # tight: the pose it settles into is the clip's
    W_HEIGHT = 3.0
    HEIGHT_STD2 = 0.03         # 17 cm of sag still pays a third
    W_UPRIGHT = 2.0
    UPRIGHT_STD2 = 0.05
    W_FEET = 3.0               # contact + flat where planted; lift where lifted
    FLAT_STD2 = 0.02
    W_FOOT_TRACK = 4.0
    # Two layers, like the duck's pose_match: the tight one pays for precision
    # (22 cm off pays 1/e), the wide one keeps a gradient from STANDING when
    # the clip has a foot half a metre away — at that distance the tight
    # layer alone paid 0.001 of 4, i.e. nothing (checked by the tests).
    FOOT_STD2 = 0.05
    FOOT_STD2_WIDE = 0.5
    W_STILL = 0.5              # the clip moves; stillness is not the task
    STILL_STD2 = 0.05
    W_SPIN = 1.0
    W_ANCHOR = 1.0
    ANCHOR_STD2 = 0.01         # 10 cm from the spawn spot costs ~63%
    W_PLANTED = 0.0            # folded into `feet` (retargeted, not dropped)
    W_FLAT_FEET = 0.0          # folded into `feet`
    ACTION_RATE_W = 0.1
    W_JOINT_VEL = 0.0005       # a moving clip has moving joints

    # A foot counts as PLANTED in the reference when its sole is within this
    # of the lowest one (the editor's own ground_tol for this body).
    REF_GROUND_TOL = 0.01
    # EARLY TERMINATION ON NOT TRACKING (DeepMimic's second half). Falling
    # ends the episode, so from a standing body the gradient points AWAY
    # from a kick it might fall out of — the barrier six reward formulations
    # measured (roadmap 13.3), and the first imitation run stood through the
    # kick window at 6/6 survival, apex 0.000 m. The symmetric rule closes
    # it: while the clip has a foot at least LIFT_MIN_REF up and the body
    # keeps it under LIFT_MIN_GOT for LIFT_MISS_STEPS such steps (frames
    # asking for no lift neither count nor forgive; an achieved lift resets
    # the count), the episode ends as if it had fallen. Not lifting is no
    # safer than falling,
    # so the only survivable policy is the one that performs the clip. A
    # strictness knob, not a reward edit (AGENTS.md: stages may ladder
    # strictness); MICRODUCK_G1_IMITATE_LOOSE=1 switches it off.
    LIFT_MIN_REF = 0.30
    LIFT_MIN_GOT = 0.05
    LIFT_MISS_STEPS = 15
    # The strictness ladder: a curriculum stage raises the lift the rule
    # demands through this env var (the lab's stage machinery passes knobs
    # through the trainer's environment), so a policy warm-started from a
    # 0.1 m kick is first kept alive by any lift at all, then held to more.
    LIFT_MIN_ENV = "MICRODUCK_G1_LIFT_MIN"
    # Spawn: this fraction of episodes start at a random point in the clip,
    # posed to match, so every phase is sampled from step one (an unsampled
    # state's value is never learned). The rest start standing at phase 0.
    SPAWN_IN_CLIP_PROB = 0.6
    FALL_HEIGHT = 0.45

    def __init__(self, *args, clip_name: str | None = None,
                 lift_min_got: float | None = None, **kwargs):
        name = clip_name or os.environ.get("MICRODUCK_CLIP") or ""
        if lift_min_got is None:
            env = os.environ.get(self.LIFT_MIN_ENV)
            lift_min_got = float(env) if env else None
        if lift_min_got is not None:
            self.LIFT_MIN_GOT = float(lift_min_got)
        if not name:
            raise ValueError("G1ImitateEnv needs a clip: pass clip_name= or set "
                             "MICRODUCK_CLIP (save one in the 🎬 animate panel)")
        self.clip = motion.load_clip(name)
        if self.clip.robot != "g1":
            raise ValueError(f"clip {name!r} poses the {self.clip.robot}, not the g1 "
                             f"({self.clip.num_joints} joints) — author it with the "
                             "G1 selected in the 🎬 panel")
        kwargs.setdefault("push_robot", False)
        super().__init__(*args, **kwargs)
        self._phase0 = 0
        self.last_spawn = "standing"
        self._miss = 0
        self.track_termination = not os.environ.get("MICRODUCK_G1_IMITATE_LOOSE")
        self._precompute_reference()
        # The clip may go lower than the idle's fall line (a crouching kick
        # chambers deep): keep the line under everything the clip asks for.
        self._fall_height = min(self.FALL_HEIGHT,
                                float(self.ref_height.min()) - 0.10)

    # --- the reference -------------------------------------------------------

    def _precompute_reference(self) -> None:
        import mujoco

        m = self.model
        d = mujoco.MjData(m)
        T = self.clip.steps
        pel = self.trunk_body_id
        foot_body = {side: int(m.geom_bodyid[self.foot_geom_ids[side][0]])
                     for side in ("left", "right")}
        self.ref_height = np.zeros(T)
        self.ref_gravity = np.zeros((T, 3))
        self.ref_foot_rel = {s: np.zeros((T, 3)) for s in foot_body}
        self.ref_foot_air = {s: np.zeros(T) for s in foot_body}
        self.ref_planted = {s: np.zeros(T, dtype=bool) for s in foot_body}
        for i in range(T):
            q, pitch = self.clip.at(i)
            mujoco.mj_resetDataKeyframe(m, d, self.key_stand)
            d.qpos[self.joint_qpos_adr] = q
            half = pitch / 2.0
            d.qpos[self._root_qpos + 3:self._root_qpos + 7] = [
                math.cos(half), 0.0, math.sin(half), 0.0]
            mujoco.mj_forward(m, d)
            low = {s: min(float(d.geom_xpos[g][2] - m.geom_size[g][0])
                          for g in self.foot_geom_ids[s]) for s in foot_body}
            floor = min(low.values())
            self.ref_height[i] = float(d.xpos[pel][2]) - floor
            R = d.xmat[pel].reshape(3, 3)
            self.ref_gravity[i] = R.T @ np.array([0.0, 0.0, -1.0])
            for s, b in foot_body.items():
                self.ref_foot_rel[s][i] = R.T @ (d.xpos[b] - d.xpos[pel])
                self.ref_foot_air[s][i] = low[s] - floor
                self.ref_planted[s][i] = (low[s] - floor) <= self.REF_GROUND_TOL
        self._foot_body = foot_body

    # --- the clock -----------------------------------------------------------

    def clip_step(self) -> int:
        return self.step_count + self._phase0

    def ref_index(self) -> int:
        i = self.clip_step()
        return i % self.clip.steps if self.clip.loop else min(i, self.clip.steps - 1)

    def _get_obs(self) -> np.ndarray:
        s, c = self.clip.phase(self.clip_step())
        # (sin, cos, 0): the third slot stays a zero, as a still idle is
        # commanded — a policy warm-started from one reads nothing new there.
        self.twist_cmd[:] = (s, c, 0.0)
        return super()._get_obs()

    def _sample_commands(self) -> None:
        return                          # the clock owns these slots

    # --- retargeted terms ----------------------------------------------------

    def pose_target_rel(self) -> np.ndarray:
        q, _ = self.clip.at(self.clip_step())
        return (q - self.default_pose)[self._pose_ids]

    def carriage_target_rel(self) -> np.ndarray:
        q, _ = self.clip.at(self.clip_step())
        return (q - self.default_pose)[self._carriage_ids]

    def target_height(self) -> float:
        return float(self.ref_height[self.ref_index()])

    def _compute_reward(self) -> tuple[float, dict[str, float]]:
        # Every joint is the carriage's business here: the clip poses all 29.
        reward, terms = super()._compute_reward()
        i = self.ref_index()
        gravity = self._projected_gravity()
        # upright, RETARGETED: the clip's own lean is the level to hold.
        g_err = gravity[:2] - self.ref_gravity[i][:2]
        upright = self.W_UPRIGHT * float(np.exp(-float((g_err ** 2).sum())
                                                / self.UPRIGHT_STD2))
        reward += upright - terms["upright"]
        terms["upright"] = upright

        # feet: planted where the clip plants, lifted where it lifts.
        contacts = self._foot_contacts()
        pel = self.trunk_body_id
        R = self.data.xmat[pel].reshape(3, 3)
        feet = 0.0
        track = 0.0
        for s, b in self._foot_body.items():
            if self.ref_planted[s][i]:
                up = self.data.xmat[b].reshape(3, 3)[2, 2]
                flat = float(np.exp(-((1.0 - up) ** 2) / self.FLAT_STD2))
                feet += 0.5 * float(contacts[s]) + 0.5 * flat
            else:
                want = float(self.ref_foot_air[s][i])
                got = min(float(self.data.geom_xpos[g][2] - self.model.geom_size[g][0])
                          for g in self.foot_geom_ids[s])
                # Progress pay, monotone: a Gaussian on the height is flat
                # where every attempt starts (roadmap 13.3, measured twice).
                feet += min(max(got / max(want, 1e-3), 0.0), 1.0)
            rel = R.T @ (self.data.xpos[b] - self.data.xpos[pel])
            track += float(((rel - self.ref_foot_rel[s][i]) ** 2).sum())
        feet = self.W_FEET * feet / 2.0
        foot_track = self.W_FOOT_TRACK * (
            0.5 * float(np.exp(-track / self.FOOT_STD2_WIDE))
            + 0.5 * float(np.exp(-track / self.FOOT_STD2)))
        for k in ("feet_planted", "flat_feet"):
            reward -= terms.pop(k, 0.0)
        terms["feet"] = feet
        terms["foot_track"] = foot_track
        reward += feet + foot_track
        return float(reward), terms

    def not_tracking(self) -> bool:
        """True once the body has stood through LIFT_MISS_STEPS steps of a
        lift the clip asks for — see LIFT_MIN_REF."""
        i = self.ref_index()
        asked = False
        missed = False
        for s, b in self._foot_body.items():
            if self.ref_foot_air[s][i] < self.LIFT_MIN_REF:
                continue
            asked = True
            got = min(float(self.data.geom_xpos[g][2] - self.model.geom_size[g][0])
                      for g in self.foot_geom_ids[s])
            other = "left" if s == "right" else "right"
            floor = min(float(self.data.geom_xpos[g][2] - self.model.geom_size[g][0])
                        for g in self.foot_geom_ids[other])
            if got - floor < self.LIFT_MIN_GOT:
                missed = True
        if asked:
            self._miss = self._miss + 1 if missed else 0
        return self._miss >= self.LIFT_MISS_STEPS

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.track_termination and not terminated and self.not_tracking():
            terminated = True
            self.last_termination = "not-tracking"
            if "episode_rewards" not in info:
                info["episode_rewards"] = dict(self.reward_sums)
        return obs, reward, terminated, truncated, info

    # --- spawning ------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._phase0 = 0
        self._miss = 0
        self.last_termination = None
        self.last_spawn = "standing"
        if self._rng.uniform() < self.SPAWN_IN_CLIP_PROB:
            import mujoco
            self._phase0 = int(self._rng.integers(0, self.clip.steps))
            q, pitch = self.clip.at(self._phase0)
            d = self.data
            d.qpos[self.joint_qpos_adr] = q
            half = pitch / 2.0
            d.qpos[self._root_qpos + 3:self._root_qpos + 7] = [
                math.cos(half), 0.0, math.sin(half), 0.0]
            mujoco.mj_forward(self.model, d)
            # Soles on the floor: the pose changes which foot is lowest, and a
            # root left where the standing spawn put it buries them.
            low = min(float(d.geom_xpos[g][2] - self.model.geom_size[g][0])
                      for ids in self.foot_geom_ids.values() for g in ids)
            d.qpos[self._root_qpos + 2] += 0.002 - low
            d.qvel[:] = 0.0
            self._write_ctrl(q)
            mujoco.mj_forward(self.model, d)
            self._refresh_derived()
            self._spawn_xy = np.asarray(self._trunk_xpos[:2], float).copy()
            self.prev_joint_vel = self._joint_vel().copy()
            self.last_spawn = "in-clip"
            obs = self._get_obs()
        return obs, info


def clip_summary(clip: motion.Clip) -> str:
    """One line for a run description: what the clip is."""
    return (f"{clip.name}: {clip.steps} frames at {motion.CONTROL_HZ:.0f} Hz "
            f"({clip.duration:.2f} s), {'looping' if clip.loop else 'one-shot'}, "
            f"{clip.num_joints} joints")


__all__ = ["G1ImitateEnv", "clip_summary"]
