"""Bilateral (left-right) symmetry for the 61-D obs / 14-D action contract.

Ported from the official stack's
``microduck_rl/src/mjlab_microduck/tasks/symmetry.py`` — the permutation and
sign tables below are that module's, re-derived here against our own
``contract.JOINT_NAMES`` and cross-checked against it at import time
(``_UPSTREAM_JOINT_PERM`` / ``_UPSTREAM_JOINT_SIGN``).

Why it exists: a gait is a left-right symmetric skill, so every transition
``(o, a)`` has a physically valid twin ``(M o, M a)``. Telling the policy that
up front — instead of making it rediscover it from samples — is the one gap
against the official stack that buys *sample efficiency* rather than more
samples, which is the only currency this CPU trainer has.

The mirror operator
-------------------
``M`` is a signed permutation: ``M(v)[i] = sign[i] * v[perm[i]]``. It is an
involution (``M∘M = I``), so mirroring twice is the identity.

Joint block (14, the order in ``contract.JOINT_NAMES``)::

    0 left_hip_yaw   5 neck_pitch    9 right_hip_yaw
    1 left_hip_roll  6 head_pitch   10 right_hip_roll
    2 left_hip_pitch 7 head_yaw     11 right_hip_pitch
    3 left_knee      8 head_roll    12 right_knee
    4 left_ankle                    13 right_ankle

- left leg (0-4) swaps with right leg (9-13); midline neck/head (5-8) stay put.
- Every leg joint NEGATES after the swap. This is not an axis-by-axis
  argument: the MJCF gives left and right opposite home-frame conventions, so
  the two sides' relative deviations are sign-flipped twins. ``DEFAULT_POSE``
  shows it directly — right leg == -(left leg), joint for joint:
  hip_roll ∓0.0873, hip_pitch ∓0.4579, knee ∓0.0049, ankle ±0.4530. Verified
  against the simulator in ``tests/test_symmetry.py``, which mirrors a pose
  through this map and checks that MuJoCo's forward kinematics puts every body
  at its partner's reflected frame.
- ``head_yaw`` and ``head_roll`` negate (yaw/roll axes reverse under a
  left-right reflection); ``neck_pitch``/``head_pitch`` are sagittal-plane
  joints and do not.

Base and command blocks:

- ``base_ang_vel`` (gyro, body frame) is a pseudovector: negate roll (x) and
  yaw (z), keep pitch (y). Formally, with S = diag(1,-1,1) the mirrored body
  rotation is ``R' = S R S`` and ``ω'_body = -S ω_body``.
- ``projected_gravity`` (body frame) is a true vector: ``g'_body = S g_body``,
  i.e. negate ``gy`` only.
- twist command: negate ``lin_vel_y`` and ``ang_vel_z``; ``lin_vel_x`` stays.
- head command ``(neck_pitch, head_pitch, head_yaw, head_roll)``: negate the
  last two, matching the joints they command.
- body command ``(x, y, z, roll, pitch, yaw)``: negate ``y``, ``roll``, ``yaw``.

All command slots mirror IN PLACE — no index permutation, only signs.

Scope of the guarantee
----------------------
The map is EXACT on kinematics and on the observation, and both halves of that
are verified against MuJoCo in ``tests/test_symmetry.py``: mirroring a pose
through this map puts every body at its partner's reflected frame (to 1e-8,
while corrupting any single sign or index blows that up by 6+ orders of
magnitude), and the mirrored *state* produces exactly ``mirror_obs`` of the
original 61-D observation (to 1e-6, with MuJoCo computing the IMU and gravity
vectors itself).

The DYNAMICS are only approximately mirror-symmetric, and it is worth being
precise about how far. The leg chains are exact mirror twins (equal mass to
5e-7 kg, equal inertia to 1e-10), but the head/neck subassembly is not —
``yaw_roll_motion``'s CoM sits 5.0 mm off the midline and ``jaw_soft``'s
1.2 mm. A mirrored rollout therefore tracks the mirror of the original to
~3e-3 for the first two control steps, then head_roll's VELOCITY breaks away
(~5e-2 by step 3) and by ~5 steps the residual has fed the trunk and the
contacts and the state diverges chaotically. Joint POSITIONS stay within
~8e-3 rad throughout. The long-horizon divergence is not attributed to a
single cause here (an attempt to pin it on the head by zeroing that
subassembly's mass changed the numerics too much to be evidence either way);
contact-solver ordering and MuJoCo's friction-pyramid tangent frames are not
mirror-equivariant either.

None of that weakens the mirror loss, which only ever consumes single
observations — it never assumes a mirrored trajectory stays mirrored. It does
mean "mirrored rollouts stay identical" is false, and the tests assert bounded
short-horizon tracking rather than equality.
"""

from __future__ import annotations

import math
from typing import TypeVar

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from . import contract as C
from .ppo_hparams import DEFAULT_DESIRED_KL, DEFAULT_SYMMETRY_COEF, UPSTREAM_DESIRED_KL

__all__ = [
    "JOINT_PERM", "JOINT_SIGN", "OBS_PERM", "OBS_SIGN",
    "mirror_obs", "mirror_action", "mirror_normalized_obs",
    "DEFAULT_SYMMETRY_COEF", "DEFAULT_DESIRED_KL", "UPSTREAM_DESIRED_KL",
    "SymmetryPPO",
]

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

# Verbatim from microduck_rl/.../tasks/symmetry.py — the authority. Ours are
# derived from JOINT_NAMES below and asserted equal to these, so a reordering
# of either side fails loudly at import instead of silently training a policy
# against a wrong mirror.
_UPSTREAM_JOINT_PERM = (9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4)
_UPSTREAM_JOINT_SIGN = (-1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1)


def _derive_joint_perm() -> np.ndarray:
    """left_X <-> right_X by name; everything else maps to itself."""
    names = list(C.JOINT_NAMES)
    perm = []
    for n in names:
        if n.startswith("left_"):
            partner = "right_" + n[len("left_"):]
        elif n.startswith("right_"):
            partner = "left_" + n[len("right_"):]
        else:
            partner = n
        perm.append(names.index(partner))
    return np.array(perm, dtype=np.int64)


def _derive_joint_sign() -> np.ndarray:
    """+1 for the sagittal (pitch) midline joints, -1 for everything else.

    Leg joints negate because left/right home frames are sign-flipped twins
    (see the module docstring); head_yaw/head_roll negate because yaw and roll
    reverse under a left-right reflection. Only neck_pitch and head_pitch —
    rotations *about* the lateral axis, which the reflection preserves — keep
    their sign.
    """
    sign = np.full(C.NUM_JOINTS, -1.0, dtype=np.float32)
    for i, n in enumerate(C.JOINT_NAMES):
        if n in ("neck_pitch", "head_pitch"):
            sign[i] = 1.0
    return sign


JOINT_PERM = _derive_joint_perm()
JOINT_SIGN = _derive_joint_sign()

assert tuple(JOINT_PERM) == _UPSTREAM_JOINT_PERM, (
    f"joint mirror permutation drifted from microduck_rl: {tuple(JOINT_PERM)}"
)
assert tuple(int(s) for s in JOINT_SIGN) == _UPSTREAM_JOINT_SIGN, (
    f"joint mirror signs drifted from microduck_rl: {tuple(JOINT_SIGN)}"
)

# Full 61-D observation map, in contract order. Only the three 14-joint blocks
# permute; the base and command blocks mirror in place (signs only).
OBS_PERM = np.concatenate([
    np.arange(0, 3),            # base_ang_vel
    np.arange(3, 6),            # projected_gravity
    6 + JOINT_PERM,             # joint_pos_rel
    20 + JOINT_PERM,            # joint_vel
    34 + JOINT_PERM,            # last_action
    np.arange(48, 61),          # twist(3) + head(4) + body(6) commands
]).astype(np.int64)

OBS_SIGN = np.concatenate([
    [-1.0, 1.0, -1.0],                      # gyro: negate roll, yaw
    [1.0, -1.0, 1.0],                       # projected gravity: negate gy
    JOINT_SIGN, JOINT_SIGN, JOINT_SIGN,     # joint_pos, joint_vel, last_action
    [1.0, -1.0, -1.0],                      # twist: negate vy, wz
    [1.0, 1.0, -1.0, -1.0],                 # head: negate head_yaw, head_roll
    [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],      # body: negate y, roll, yaw
]).astype(np.float32)

assert OBS_PERM.shape == (C.OBS_DIM,) and OBS_SIGN.shape == (C.OBS_DIM,)

# ---------------------------------------------------------------------------
# Mirror operators
# ---------------------------------------------------------------------------

A = TypeVar("A", np.ndarray, torch.Tensor)


class _SignedPermutation:
    """``M(v)[i] = sign[i] * v[perm[i]]`` on the last axis, numpy or torch.

    Torch views are built once per device: the tables are constants, and
    reallocating them for every minibatch would show up in a CPU trainer's
    profile.
    """

    def __init__(self, perm: np.ndarray, sign: np.ndarray, what: str):
        self.perm, self.sign, self.what = perm, sign, what
        self._torch: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def __call__(self, x: A) -> A:
        if x.shape[-1] != len(self.perm):
            raise ValueError(
                f"expected (..., {len(self.perm)}) {self.what}, got {tuple(x.shape)}"
            )
        if isinstance(x, torch.Tensor):
            tables = self._torch.get(x.device)
            if tables is None:
                tables = (
                    torch.as_tensor(self.perm, dtype=torch.long, device=x.device),
                    torch.as_tensor(self.sign, dtype=torch.float32, device=x.device),
                )
                self._torch[x.device] = tables
            perm, sign = tables
            return x.index_select(-1, perm) * sign.to(x.dtype)
        x = np.asarray(x)
        return x[..., self.perm] * self.sign.astype(x.dtype, copy=False)


#: Reflect a 61-D observation (or a batch, shape ``(..., 61)``) about the
#: sagittal plane. Involutive: ``mirror_obs(mirror_obs(o)) == o``.
mirror_obs = _SignedPermutation(OBS_PERM, OBS_SIGN, "obs")

#: Reflect a 14-D action (or a batch, shape ``(..., 14)``). Involutive.
mirror_action = _SignedPermutation(JOINT_PERM, JOINT_SIGN, "action")


def mirror_normalized_obs(obs: A, mean: A, std: A) -> A:
    """Mirror an observation that has already been through VecNormalize.

    The mirror is defined on RAW observations, and normalization does not
    commute with it: ``(M o - mu)/sigma != M((o - mu)/sigma)`` unless the
    running statistics are themselves mirror-symmetric, which empirical ones
    never exactly are. So un-normalize, mirror, re-normalize — exact for the
    stats passed in.

    (VecNormalize also clips to ``clip_obs``; that is not re-applied here.
    ``clip_obs`` is 100.0 in this trainer and normalized observations do not
    come near it, and clipping is sign/permutation-equivariant anyway.)
    """
    return (mirror_obs(obs * std + mean) - mean) / std


# ---------------------------------------------------------------------------
# Rollout-lean ActorCriticPolicy
# ---------------------------------------------------------------------------


class FastActorCriticPolicy(ActorCriticPolicy):
    """``ActorCriticPolicy`` with a hand-rolled rollout ``forward()``.

    Stock SB3 builds a ``torch.distributions.Normal`` object per vec-step and
    routes sampling and log-prob through it. The math for a diagonal Gaussian
    is six tensor ops, written out below. Measured honestly (batch 32, this
    M5 Max, no profiler): 234 → 215 us per forward, an 8% saving — cProfile
    had made the distribution plumbing look far more expensive than it is.
    Kept because it is free and exact: identical distribution and log-prob
    formula; the only observable difference is the RNG call (``randn_like``
    vs ``torch.normal``), so a seeded run samples a different — equally
    distributed — noise stream than stock SB3.

    Only the plain shared-extractor diag-Gaussian setup (what "MlpPolicy"
    builds for a Box action space, i.e. every trainer in this repo) takes the
    fast path; anything else falls back to the stock forward. ``predict()``,
    ``evaluate_actions()`` and the ONNX export path are untouched.
    """

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        from stable_baselines3.common.distributions import DiagGaussianDistribution
        if (self.use_sde or not self.share_features_extractor
                or not isinstance(self.action_dist, DiagGaussianDistribution)):
            return super().forward(obs, deterministic)
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        mean = self.action_net(latent_pi)
        log_std = self.log_std
        if deterministic:
            actions = mean
            # z = 0 for every dim: only the normalization terms remain.
            log_prob = (-log_std.sum()
                        - 0.5 * math.log(2 * math.pi) * mean.shape[1]
                        ).expand(mean.shape[0])
        else:
            noise = torch.randn_like(mean)
            actions = torch.addcmul(mean, noise, torch.exp(log_std))
            # Diagonal-Gaussian log density with z = (x - mean)/std = noise,
            # summed over action dims — the same formula
            # DiagGaussianDistribution.log_prob evaluates.
            log_prob = (-0.5 * (noise * noise).sum(dim=1) - log_std.sum()
                        - 0.5 * math.log(2 * math.pi) * mean.shape[1])
        return actions, values, log_prob


# ---------------------------------------------------------------------------
# PPO with a symmetry (mirror) loss
# ---------------------------------------------------------------------------

# DEFAULT_SYMMETRY_COEF / DEFAULT_DESIRED_KL / UPSTREAM_DESIRED_KL live in
# ppo_hparams.py so the trainers can import them without importing torch.
#
# Why DEFAULT_DESIRED_KL is off: three 1.5M-step `run` arms, one seed
# (episode length at the end / peak ep_rew):
#   constant 1e-3 (no controller)  180 steps  354
#   desired_kl 0.05               111 steps  216
#   desired_kl 0.01 (upstream)     14 steps   30
# Upstream's 0.01 target pins the rate to its 1e-5 floor at our batch
# (16 x 256 = 4,096 vs their 98,304 samples/update). The collapse that
# motivated the controller was a symptom of the broken run reward: fixing
# `_run_speed` dropped median KL from ~0.30 to 0.044 at constant 1e-3.


class SymmetryPPO(PPO):
    """SB3 PPO plus rsl_rl's mirror loss.

    Adds ``symmetry_coef * MSE(pi_mean(M o), M pi_mean(o).detach())`` to the
    PPO loss, which is exactly what rsl_rl 5.0.1's
    ``PPO.update()`` does under ``use_mirror_loss`` (and what the official
    stack configures via ``SYMMETRY_CFG``): one forward pass over
    ``[o ; M o]``, the mirrored *target* detached so the gradient only flows
    through the ``pi(M o)`` branch, added to the same loss before the same
    single ``backward()``.

    Upstream sets ``use_data_augmentation=False``, so this is the mirror-loss
    variant, NOT batch augmentation — the rollout batch keeps its original
    size and the mirrored samples never enter the surrogate/value losses.

    ``symmetry_coef=0`` short-circuits the whole block, making this class
    behave bit-for-bit like stock ``PPO`` (locked by
    ``tests/test_symmetry.py::test_zero_coef_is_bit_identical_to_stock_ppo``).

    ``train()`` below is SB3 2.9.0's ``PPO.train()`` with the symmetry block
    spliced in; SB3 offers no hook between loss assembly and ``backward()``,
    so there is nothing to subclass more narrowly. The zero-coef equivalence
    test is the guard that the copy stays faithful across SB3 upgrades.
    """

    def __init__(self, *args, symmetry_coef: float = DEFAULT_SYMMETRY_COEF,
                 desired_kl: float | None = DEFAULT_DESIRED_KL,
                 ent_anneal: bool = False, overlap_update: bool = False,
                 update_device: str | None = None, **kwargs):
        # overlap_update runs the PPO update in a background thread while the
        # NEXT rollout is collected with a frozen pre-update snapshot of the
        # policy (see learn()). Off by default: it introduces a one-update
        # policy lag in the collected data, which stock PPO does not have —
        # and the A/B measured that lag costing ~2x reward-per-step early in
        # training (README), so it stays a benchmarking tool.
        self.overlap_update = bool(overlap_update)
        # update_device shuttles the policy + optimizer to another device
        # (in practice "mps") for the minibatch loop only; rollouts stay on
        # self.device, where batch-32 forwards beat the GPU's dispatch
        # latency. Measured on this M5 Max: fwd+bwd+adam at batch 2048 is
        # 8.6 ms on CPU, 2.2 ms on MPS. Same math, different kernels — NOT
        # bit-identical to the CPU update, which is why it is None by
        # default (the zero-coef stock-PPO parity test must keep passing).
        self.update_device = update_device
        # ent_anneal linearly decays ent_coef to ZERO over training. OFF here
        # (the zero-coef bit-parity test needs stock behavior); train_behavior
        # turns it on. Why it exists: a constant entropy bonus keeps the
        # action std high forever, so the MEAN policy is never forced to work
        # without its noise — three policies in one day scored well
        # stochastically and fell in under a second deterministically (the
        # exported ONNX is the mean). The best "peak" ever archived fell in
        # 0.26 s.
        self.ent_anneal = bool(ent_anneal)
        self._ent0: float | None = None
        self.symmetry_coef = float(symmetry_coef)
        # None / <=0 disables the controller and restores SB3's constant rate.
        self.desired_kl = (None if desired_kl is None or desired_kl <= 0
                           else float(desired_kl))
        # The rate the controller owns, carried across train() calls and saved
        # with the checkpoint so a warm restart resumes annealed, not hot.
        self._adaptive_lr: float | None = None
        super().__init__(*args, **kwargs)

    # -- symmetry pieces ----------------------------------------------------

    def _obs_normalizer(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Current (mean, std) of the wrapped VecNormalize, or None if unwrapped.

        Snapshotted per ``train()`` call: the running stats keep updating
        during rollout collection, so the un-normalize/re-normalize round trip
        is exact only for the stats in force right now. The residual is a
        second-order effect on already-converged statistics.
        """
        venv = self.get_vec_normalize_env()
        if venv is None or not getattr(venv, "norm_obs", False):
            return None
        rms = getattr(venv, "obs_rms", None)
        if rms is None:  # dict-obs VecNormalize keeps a per-key dict instead
            return None
        mean = torch.as_tensor(rms.mean, dtype=torch.float32, device=self.device)
        std = torch.sqrt(
            torch.as_tensor(rms.var, dtype=torch.float32, device=self.device)
            + venv.epsilon
        )
        return mean, std

    def _policy_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """The action distribution's MEAN (not a sample) for ``obs``."""
        return self.policy.get_distribution(obs).distribution.mean

    def _evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """``policy.evaluate_actions`` plus the action mean, one feature extract.

        The symmetry loss needs ``mean(obs)``. Calling ``evaluate_actions``
        and then ``_policy_mean(cat([obs, M obs]))`` would forward the actor
        on ``obs`` twice (N + 2N). This shares the extractor/MLP with the
        PPO surrogate and leaves only the mirrored half as extra work.
        """
        policy = self.policy
        features = policy.extract_features(obs)
        if policy.share_features_extractor:
            latent_pi, latent_vf = policy.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = policy.mlp_extractor.forward_actor(pi_features)
            latent_vf = policy.mlp_extractor.forward_critic(vf_features)
        distribution = policy._get_action_dist_from_latent(latent_pi)
        values = policy.value_net(latent_vf)
        return (values, distribution.log_prob(actions), distribution.entropy(),
                distribution.distribution.mean)

    def _symmetry_loss(
        self, obs: torch.Tensor, norm: tuple[torch.Tensor, torch.Tensor] | None,
        mean_orig: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mirrored = (
            mirror_obs(obs) if norm is None
            else mirror_normalized_obs(obs, norm[0], norm[1])
        )
        if mean_orig is None:
            # Tests and the standalone call: one forward over [o ; M o], as
            # rsl_rl does. The train() hot path passes mean_orig so it does
            # not re-forward obs.
            both = self._policy_mean(torch.cat([obs, mirrored], dim=0))
            n = obs.shape[0]
            mean_orig, mean_mirrored_obs = both[:n], both[n:]
        else:
            mean_mirrored_obs = self._policy_mean(mirrored)
        target = mirror_action(mean_orig).detach()
        return F.mse_loss(mean_mirrored_obs, target)

    # -- vendored SB3 2.9.0 PPO.train() + the symmetry term -----------------

    def _adapt_learning_rate(self, kl: float) -> None:
        """rsl_rl's adaptive step, applied to the live optimizer.

        Note the DEADBAND: between half and twice the target the rate is left
        alone, so the controller settles at whatever rate holds KL near target
        rather than running to a clamp. That is the intended behavior, and it
        is why an "impossible" target does not drive the rate to 1e-5 -- it
        drives it until KL falls back inside the band.
        """
        if self.desired_kl is None:
            return
        if kl > self.desired_kl * 2.0:
            self._adaptive_lr = max(1e-5, self._adaptive_lr / 1.5)
        elif 0.0 < kl < self.desired_kl / 2.0:
            self._adaptive_lr = min(1e-2, self._adaptive_lr * 1.5)
        for group in self.policy.optimizer.param_groups:
            group["lr"] = self._adaptive_lr

    def train(self) -> None:
        """Update policy using the currently gathered rollout buffer."""
        th = torch  # keep the vendored body diffable against SB3's source

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Hop the policy + optimizer state to the update device for the
        # minibatch loop (see __init__). Moved back at the end of this
        # method; an exception in between kills the process anyway, so no
        # try/finally — nothing observes the stranded state.
        update_dev = (torch.device(self.update_device)
                      if self.update_device else None)
        if update_dev is not None:
            self.policy.to(update_dev)
            self._optimizer_to(update_dev)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # ...then hand it straight back to the KL controller, which owns the
        # rate once training starts. _update_learning_rate just re-applied
        # lr_schedule (a constant, for a float learning_rate), which would
        # otherwise undo every adjustment made in the previous update.
        if self.desired_kl is not None:
            if self._adaptive_lr is None:
                self._adaptive_lr = float(self.policy.optimizer.param_groups[0]["lr"])
            for group in self.policy.optimizer.param_groups:
                group["lr"] = self._adaptive_lr
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        if self.ent_anneal:
            if self._ent0 is None:
                self._ent0 = float(self.ent_coef)
            self.ent_coef = self._ent0 * max(float(self._current_progress_remaining), 0.0)

        # --- symmetry: snapshot the obs normalizer once per update ---
        use_symmetry = self.symmetry_coef > 0.0
        norm = self._obs_normalizer() if use_symmetry else None
        if update_dev is not None and norm is not None:
            norm = (norm[0].to(update_dev), norm[1].to(update_dev))
        symmetry_losses: list[float] = []

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # KL is only consumed by the adaptive-LR controller, SB3's target_kl
        # early-stop, and the train/approx_kl log line. Computing it (and
        # .cpu().numpy()-syncing it) every minibatch was free on a GPU and
        # a real slice of the CPU update — skip the tensor work when nothing
        # reads it. Logging still wants a number, so we keep the last
        # minibatch's estimator when the controller is on, and skip the
        # record entirely when it is off (tensorboard already has entropy /
        # clip_fraction / loss).
        need_kl = self.desired_kl is not None or self.target_kl is not None
        approx_kl_divs: list[float] = []
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                if update_dev is not None:
                    # The buffer lives on self.device (cpu); unified memory
                    # makes this ~0.6 MB copy per minibatch cheap.
                    rollout_data = type(rollout_data)(
                        *(t.to(update_dev) for t in rollout_data))
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                if use_symmetry:
                    values, log_prob, entropy, mean_orig = self._evaluate_actions(
                        rollout_data.observations, actions
                    )
                else:
                    # Zero-coef path must call evaluate_actions, not our
                    # helper: test_zero_coef_is_bit_identical_to_stock_ppo
                    # compares weights against stock PPO.train().
                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations, actions
                    )
                    mean_orig = None
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # --- symmetry (mirror) loss, rsl_rl's formulation ---
                if use_symmetry:
                    symmetry_loss = self._symmetry_loss(
                        rollout_data.observations, norm, mean_orig=mean_orig
                    )
                    loss = loss + self.symmetry_coef * symmetry_loss

                if need_kl:
                    # Calculate approximate form of reverse KL Divergence for
                    # early stopping / the adaptive-LR controller.
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean(
                            (th.exp(log_ratio) - 1) - log_ratio
                        ).item()
                        approx_kl_divs.append(approx_kl_div)

                    # --- adaptive learning rate: rsl_rl's rule, verbatim ---
                    # (rsl_rl/algorithms/ppo.py: kl > 2x target -> /1.5, kl < half
                    # target -> *1.5, clamped to [1e-5, 1e-2].) Applied before this
                    # minibatch's optimizer.step(), same as upstream. One
                    # deliberate difference: rsl_rl computes the analytic Gaussian
                    # KL from the stored old distribution parameters, while SB3's
                    # buffer keeps only old_log_prob -- so this drives on the k3
                    # sample estimator above. It is a low-variance estimator of the
                    # same quantity, which is what the thresholds care about.
                    self._adapt_learning_rate(float(approx_kl_div))

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # Logging AFTER the step so .item() does not join the OpenMP
                # workers in the middle of the graph. On CPU these are just
                # scalar reads, but they still force a Python round-trip
                # that used to sit between backward and step.
                pg_losses.append(policy_loss.item())
                clip_fractions.append(
                    th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                )
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                if use_symmetry:
                    symmetry_losses.append(symmetry_loss.item())

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        if approx_kl_divs:
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        if self.desired_kl is not None:  # the rate actually in force
            self.logger.record("train/learning_rate", self._adaptive_lr)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if symmetry_losses:
            self.logger.record("train/symmetry_loss", np.mean(symmetry_losses))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        if update_dev is not None:
            self.policy.to(self.device)
            self._optimizer_to(torch.device(self.device))

    def _optimizer_to(self, device: torch.device) -> None:
        """Move the optimizer's state tensors (Adam moments) with the policy.

        ``nn.Module.to`` does not touch optimizer state; a step() with
        parameters and moments on different devices raises.
        """
        for state in self.policy.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # -- overlapped update ---------------------------------------------------

    def learn(self, total_timesteps, callback=None, log_interval: int = 1,
              tb_log_name: str = "SymmetryPPO", reset_num_timesteps: bool = True,
              progress_bar: bool = False):
        """Stock SB3 learn unless ``overlap_update`` — then rollout k+1 is
        collected while the update on rollout k runs in a background thread.

        Stock PPO strictly alternates a worker-bound rollout with a
        trainer-bound update; at 32 envs on this machine the two phases are
        nearly equal, so the alternation idles almost half the hardware at
        any moment. Overlapping hides the shorter phase entirely.

        Correctness: the collector uses a FROZEN copy of the policy taken
        before the update starts, so a rollout is internally consistent (its
        actions, values and old_log_probs all come from one set of weights).
        The cost is that the data trained on is one update stale — PPO's
        clipped importance ratio is built for exactly this mismatch, but it
        is a real change to training dynamics, hence opt-in and A/B'd rather
        than default (see tests/test_overlap.py and the A/B in the README).

        Thread-safety inventory: the update thread touches self.policy, the
        optimizer, self.rollout_buffer (the buffer it was handed) and the
        logger's record dict; the main thread touches the frozen copy, the
        OTHER buffer, the env and self.num_timesteps. The only rendezvous is
        the join, placed BEFORE callback.on_rollout_end so callbacks (e.g.
        train_behavior's ONNX snapshots) never see mid-update weights.
        """
        if not self.overlap_update:
            return super().learn(
                total_timesteps, callback=callback, log_interval=log_interval,
                tb_log_name=tb_log_name, reset_num_timesteps=reset_num_timesteps,
                progress_bar=progress_bar)
        import copy
        import threading

        total_timesteps, callback = self._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name,
            progress_bar)
        callback.on_training_start(locals(), globals())
        assert self.env is not None

        # Rebuild-from-constructor rather than deepcopy: after any forward,
        # ActorCriticPolicy caches its last torch Distribution (graph, non-
        # leaf tensors), which deepcopy refuses. _get_constructor_parameters
        # + load_state_dict is the same pattern SB3's save/load uses.
        rollout_policy = type(self.policy)(
            **self.policy._get_constructor_parameters()).to(self.device)
        rollout_policy.load_state_dict(self.policy.state_dict())
        buf_a, buf_b = self.rollout_buffer, copy.deepcopy(self.rollout_buffer)
        iteration = 0
        collect_buf = buf_a
        continue_training = self._collect_with(
            rollout_policy, self.env, callback, collect_buf, self.n_steps)
        while continue_training and self.num_timesteps < total_timesteps:
            iteration += 1
            self._update_current_progress_remaining(
                self.num_timesteps, total_timesteps)
            if log_interval is not None and iteration % log_interval == 0:
                self.dump_logs(iteration)
            self.rollout_buffer = collect_buf
            errors: list[BaseException] = []

            def _update() -> None:
                try:
                    self.train()
                except BaseException as e:  # surfaced after the join
                    errors.append(e)

            thread = threading.Thread(target=_update, name="ppo-update")
            thread.start()
            collect_buf = buf_b if collect_buf is buf_a else buf_a
            continue_training = self._collect_with(
                rollout_policy, self.env, callback, collect_buf, self.n_steps,
                pre_rollout_end=thread.join)
            thread.join()  # normal exit already joined; early return has not
            if errors:
                raise errors[0]
            # Next rollout's frozen weights = the update we just joined.
            rollout_policy.load_state_dict(self.policy.state_dict())
        if continue_training:
            # The loop's last collected buffer has not been trained on.
            iteration += 1
            self._update_current_progress_remaining(
                self.num_timesteps, total_timesteps)
            if log_interval is not None and iteration % log_interval == 0:
                self.dump_logs(iteration)
            self.rollout_buffer = collect_buf
            self.train()
        callback.on_training_end()
        return self

    def collect_rollouts(self, env, callback, rollout_buffer,
                         n_rollout_steps: int) -> bool:
        """Route ALL collection through the vendored loop so the deferred
        buffer add (below) applies to plain training, not just overlap.
        Value-identical to stock SB3 collection — pinned bitwise by
        tests/test_overlap.py::test_collect_matches_stock_sb3_bitwise."""
        return self._collect_with(self.policy, env, callback, rollout_buffer,
                                  n_rollout_steps)

    def _collect_with(self, policy, env, callback, rollout_buffer,
                      n_rollout_steps: int, pre_rollout_end=None) -> bool:
        """SB3 2.9.0's ``collect_rollouts`` with three changes: the acting
        policy is a PARAMETER instead of ``self.policy``; an optional
        ``pre_rollout_end`` hook runs before ``callback.on_rollout_end()``
        (the overlap join point); and each step's ``rollout_buffer.add`` is
        DEFERRED into the next step's env wait — after ``step_async`` wakes
        the workers, the parent copies the previous transition into the
        buffer while the physics runs, instead of idling on the completion
        semaphore (profiled: the parent spends ~1 ms/vec-step blocked there
        at 64 envs). Safe because every array involved is a fresh copy per
        step (ForkVecEnv and VecNormalize both return new arrays), so the
        stashed references stay valid; the data written is bit-identical,
        only WHEN it is written moves. Vendored for the same reason
        ``train()`` is: SB3 offers no seam for any of this.
        """
        th = torch
        from stable_baselines3.common.utils import obs_as_tensor

        assert self._last_obs is not None, "No previous observation was provided"
        policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        pending_add = None  # previous step's transition, written during the wait
        while n_steps < n_rollout_steps:
            if (self.use_sde and self.sde_sample_freq > 0
                    and n_steps % self.sde_sample_freq == 0):
                policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if policy.squash_output:
                    clipped_actions = policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high)

            env.step_async(clipped_actions)
            if pending_add is not None:
                rollout_buffer.add(*pending_add)
                pending_add = None
            new_obs, rewards, dones, infos = env.step_wait()

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                # pending_add is always None here (flushed right after this
                # step's step_async), and stock SB3 also drops the current
                # step on early termination — parity holds.
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)):
                    terminal_obs = policy.obs_to_tensor(
                        infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            pending_add = (self._last_obs, actions, rewards,
                           self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        if pending_add is not None:  # the final step has no next wait to hide in
            rollout_buffer.add(*pending_add)

        with th.no_grad():
            values = policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones)

        callback.update_locals(locals())
        if pre_rollout_end is not None:
            pre_rollout_end()
        callback.on_rollout_end()

        return True
