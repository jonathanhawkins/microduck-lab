"""Velocity-command walking env for the Unitree G1.

`MicroduckWalkEnv` with a different body (robots/spec.py) and a different
observation: the G1 speaks the 99-d LuckyRobots layout its shipped
`walker.onnx` was trained against, not the duck's 61-d deployment contract.
Keeping that layout verbatim buys two things the duck's stack has no
equivalent of:

  - a GOLDEN TEST — `walker.onnx` must walk inside this env. If it does, the
    scene, the obs assembly and the action scaling are right; if it does not,
    every reward number measured here would have been noise
    (tests/test_g1_env.py).
  - a TEACHER — the same policy is a behaviour clone source for `distill.py`,
    which is what makes a 29-DoF humanoid a CPU project at all.

What is NOT inherited from the duck:
  * the head-pose reward term (the G1 has no neck servos in the 29),
  * the 1-step joint-velocity lag (a Dynamixel present_velocity artifact),
  * the air-time window, measured off the shipped walker here rather than
    ported (see AIR_TIME_MIN/MAX).

Honesty, per AGENTS.md: the 99-d layout feeds base LINEAR velocity, which no
real humanoid observes without state estimation. This is a lab contract for
prototyping and for the /sim world's walking person — not a sim2real one.
"""

from __future__ import annotations

import numpy as np

from ..walk_env import MicroduckWalkEnv
from . import g1


class G1WalkEnv(MicroduckWalkEnv):
    """Velocity-command walking on the frozen-hands G1."""

    # Rewards: the duck's weights, minus the head term it has no joints for.
    W_HEAD_POSE = 0.0
    # Air time. MEASURED off the shipped walker in this scene (two seeds,
    # 30 s a command, ~135 swings each):
    #     cmd 0.5 m/s   median 0.160 s   p90 0.180   max 0.200
    #     cmd 0.8 m/s   median 0.200 s   p90 0.220   max 0.240
    # A 1.3 m humanoid swings FASTER than the duck's window assumes, not
    # slower: the guessed [0.25, 0.60] this started as paid on 0.0% of real
    # swings — a reward term that could never fire. The floor sits above the
    # one- to three-step blips the contact scan reports (p10 is 0.02 s, a
    # single control step) and the ceiling above the longest real swing;
    # 75% of the shipped walker's swings pay inside it at both commands.
    AIR_TIME_MIN = 0.08
    AIR_TIME_MAX = 0.30
    # Pose: hold every joint near default (arms included — the shipped
    # walker swings them, and an unpriced arm flails).
    W_POSE = 0.5
    POSE_STD2 = 4.0          # 29 joints in the sum, not 10

    def __init__(self, *args, joint_vel_lag: bool = False, **kwargs):
        kwargs.setdefault("robot", g1.G1_SPEC._resolve())
        # The G1's MJCF drives position servos; the BAM model is an XL330
        # identification and does not describe this robot's actuators.
        act = kwargs.get("actuator_force") or kwargs.get("actuator") or "xml"
        if str(act).lower() == "bam":
            raise ValueError(
                "actuator='bam' is the duck's XL330 servo model — the G1 runs "
                "its MJCF position actuators ('xml')")
        kwargs.setdefault("actuator_force", "xml")
        self.joint_vel_lag = joint_vel_lag
        super().__init__(*args, **kwargs)

    # ----------------------------------------------------------- observations

    def _get_obs(self) -> np.ndarray:
        """99-d, LuckyRobots order — see the module docstring."""
        lin = self.body_lin_vel().astype(np.float32)
        gyro = self._gyro.astype(np.float32)
        gravity = self._projected_gravity()
        joint_pos = self._joint_pos_rel()
        if self.joint_vel_lag:
            joint_vel = self.prev_joint_vel
            self.prev_joint_vel = self._joint_vel().copy()
        else:
            joint_vel = self._joint_vel()
            self.prev_joint_vel = joint_vel

        if self.obs_noise:
            r = self._rng
            spec = self.robot
            nj = self.nj
            lin = lin + r.uniform(-spec.noise_gyro, spec.noise_gyro, 3).astype(np.float32)
            gyro = gyro + r.uniform(-spec.noise_gyro, spec.noise_gyro, 3).astype(np.float32)
            gravity = gravity + r.uniform(
                -spec.noise_gravity, spec.noise_gravity, 3).astype(np.float32)
            joint_pos = joint_pos + r.uniform(
                -spec.noise_joint_pos, spec.noise_joint_pos, nj).astype(np.float32)
            joint_vel = joint_vel + r.uniform(
                -spec.noise_joint_vel, spec.noise_joint_vel, nj).astype(np.float32)

        nj = self.nj
        obs = np.empty(self.robot.obs_dim, np.float32)
        obs[0:3] = lin
        obs[3:6] = gyro
        obs[6:9] = gravity
        obs[9:9 + nj] = joint_pos
        obs[9 + nj:9 + 2 * nj] = joint_vel
        obs[9 + 2 * nj:9 + 3 * nj] = self.last_action
        obs[9 + 3 * nj:9 + 3 * nj + 3] = self.twist_cmd
        return obs

    # -------------------------------------------------------------- commands

    def _sample_commands(self) -> None:
        """Twist only. The duck's head/body command slots exist to keep the
        61-obs contract's unused neurons alive; this robot's obs has none."""
        r = self._rng
        u = r.uniform()
        stand = self.zero_command_prob
        turn = stand + self.turn_in_place_prob
        fwd = turn + self.forward_command_prob
        spec = self.robot
        if u < stand:
            self.twist_cmd[:] = 0.0
        elif u < turn:
            self.twist_cmd[:] = (0.0, 0.0, r.uniform(*spec.ang_vel_z_range))
        elif u < fwd:
            vx = abs(float(r.uniform(*spec.lin_vel_x_range)))
            self.twist_cmd[:] = (max(vx, 0.3), 0.0, 0.0)
        else:
            self.twist_cmd[:] = (
                r.uniform(*spec.lin_vel_x_range),
                r.uniform(*spec.lin_vel_y_range),
                r.uniform(*spec.ang_vel_z_range),
            )
        # head/body command vectors stay zero: nothing reads them here.

    # --------------------------------------------------------------- rewards

    def _compute_reward(self) -> tuple[float, dict[str, float]]:
        reward, terms = super()._compute_reward()
        # W_HEAD_POSE = 0 already zeroes the term's contribution; drop the key
        # so an episode's reward breakdown does not carry a row that is
        # structurally zero (the viewer draws one bar per term).
        head = terms.pop("head_pose", 0.0)
        return float(reward - head), terms


def make_g1_env(**kwargs) -> G1WalkEnv:
    """Factory with the training defaults (used by train-walk --robot g1)."""
    kwargs.setdefault("max_episode_s", 20.0)
    return G1WalkEnv(**kwargs)


class G1StandEnv(G1WalkEnv):
    """Stand still: the G1's idle.

    Why this exists as a TRAINED policy rather than "send the walker a zero
    command": the lab's drive script asks for 0.9 m/s 27 s out of every 30,
    so a velocity policy in a roster slot walks out of frame. An idle brain
    ignores the command and holds its ground — the same role `alpha_stand`
    plays for the duck, which the Lucky Robots drop never shipped.

    It is not a freebie. Holding the default pose with this robot's MJCF
    position servos does NOT hold it up: measured, ctrl pinned at the default
    pose collapses the pelvis from 0.76 m to 0.13 m. Standing here is active
    balance, the same control problem as walking minus the travel.

    The reward is the walk recipe's frame with the travel terms removed:
    upright and at height, NOT moving, both feet down, joints near default,
    and the usual self-negating penalties.
    """

    # Commands are pinned to zero (see `_sample_commands`), so the keep-alive
    # ranges the walk env samples do not apply.
    # The finished idle holds its ground (2.3 cm net drift over 60 s) but
    # SWAYS: ~3.7 cm of path travelled every second for the first ~20 s,
    # damping to 0.15 cm/s after that. At std2 0.02 a 0.05 m/s sway still
    # collects 88% of this term, so tightening it to 0.005 looked like the
    # obvious fix — and it COLLAPSED the task (stage 1 fell from 10.4 s
    # episodes to 0.4 s). Squeezing a still-robot's earners is the same
    # mistake as letting a penalty ramp over them (see ACTION_RATE_W): the
    # policy stops being paid for the corrections that keep it up. The sway
    # is bought with episode LENGTH instead — see `train.env_kwargs_from_args`,
    # which practises 40 s holds so the settled regime is most of the episode
    # rather than a tail the policy rarely reaches.
    STILL_STD2 = 0.02        # (m/s)^2 — 0.14 m/s costs half the term
    SPIN_STD2 = 0.10         # (rad/s)^2
    HEIGHT_STD2 = 0.004      # m^2 — 6 cm of sag costs half
    W_STILL = 3.0
    W_SPIN = 1.0
    W_UPRIGHT = 2.0
    W_HEIGHT = 2.0
    W_PLANTED = 1.0          # both feet on the floor
    W_POSE = 1.0
    POSE_STD2 = 4.0
    # Waist + arms held at the default. std2 0.25 over 17 joints: ~3 deg a
    # joint pays 0.83, ~5 deg pays 0.60, and the sloppy carriage measured
    # above (1.05 rad^2) pays 0.015 — a real gradient where `pose` had none.
    W_CARRIAGE = 0.0          # off here: the shipped idle predates the term
    CARRIAGE_STD2 = 0.25
    W_JOINT_VEL = 0.002      # penalty; a still robot has still joints
    # A swaying biped rocks on its ANKLES, and nothing above prices that: the
    # duck's own stand recipe pays for `flat_feet` for exactly this reason.
    # An earner, not a penalty — a bounded term cannot take over the reward
    # the way ACTION_RATE_W did.
    W_FLAT_FEET = 1.0
    FLAT_STD2 = 0.01         # on (1 - cos tilt) per sole; 8 deg costs ~half
    # …and nothing prices POSITION either, only velocity, so a sway that
    # averages out is free. The duck charges `stay_home` for this.
    W_ANCHOR = 1.0
    ANCHOR_STD2 = 0.0025     # m^2 — 5 cm from the spawn spot costs ~63%

    # The duck's action-rate penalty RAMPS 0.1 -> 1.0 over a run
    # (walk_env._ACTION_RATE_STAGES) because a walking policy has a large
    # positive reward to outrun it. A standing one does not, and this robot
    # sums the penalty over 29 joints instead of 14. MEASURED on the first
    # 1.5M-step run: at weight 0.1-0.2 the policy held 10 s episodes and
    # scored 4334; as the ramp passed 0.4 every positive term collapsed
    # toward zero, the penalty grew to -116, and the rendered policy hinged
    # forward at the waist and dived in 0.84 s — going limp is the cheapest
    # way to stop changing your actions. Fixed, not ramped, at the value that
    # was working.
    ACTION_RATE_W = 0.1

    def _action_rate_weight(self) -> float:
        return self.ACTION_RATE_W

    def __init__(self, *args, command_mix: float = 0.0, **kwargs):
        """`command_mix` is the fraction of episodes that SHOW a nonzero
        drive command in the observation while still paying for standing —
        an idle that has only ever seen zeros is out of distribution the
        moment something drives it (measured on the first clone: collapsed
        in 0.7 s at a 0.9 m/s command). 0.0 during cloning, where the point
        is to copy the teacher's zero-command behaviour; turned up for the
        PPO polish, where it buys command immunity."""
        self.command_mix = float(command_mix)
        # Pushes OFF, like every other behavior recipe (BehaviorEnv does the
        # same setdefault). AGENTS.md: "Velocity pushes are part of the walk
        # env's DR and OFF in every behavior recipe. Turning them on for a
        # trick is an experiment to name and measure, not a fix." These two
        # tasks reach the env through train.py rather than BehaviorEnv, so
        # they were inheriting the WALK env's DR and being shoved +/-0.4 m/s
        # every 3-6 s while their reward scored `still` and `stay_home`.
        kwargs.setdefault("push_robot", False)
        super().__init__(*args, **kwargs)
        # Every joint above the hips: the ones that decide whether the robot
        # LOOKS composed, and none of the ones it balances with.
        self._carriage_ids = np.array(
            [i for i, n in enumerate(self.robot.joint_names)
             if any(k in n for k in ("waist", "shoulder", "elbow", "wrist"))],
            dtype=int)
        # The bodies the foot pads hang off — one per side, for `flat_feet`.
        self._foot_body_ids = sorted({
            int(self.model.geom_bodyid[g])
            for ids in self.foot_geom_ids.values() for g in ids})
        self._spawn_xy = np.zeros(2)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # `stay_home` is measured from where THIS episode started, not from
        # the world origin: the spawn carries pose and height noise.
        self._spawn_xy = np.asarray(self._trunk_xpos[:2], float).copy()
        return obs, info

    def target_height(self) -> float:
        """The pelvis height the `height` term pays for. Standing, here; a
        crouch task retargets it (G1SquatEnv) instead of copying the reward."""
        return self.stand_z

    def _sample_commands(self) -> None:
        """An idle is asked for nothing — and ignores it when it is asked
        anyway. The lab also stops driving it (`pinned_command` in run.json,
        viz_server.is_trick_duck), so this is belt and braces."""
        if self.command_mix <= 0.0 or self._rng.uniform() >= self.command_mix:
            self.twist_cmd[:] = 0.0
            return
        spec = self.robot
        r = self._rng
        self.twist_cmd[:] = (
            r.uniform(*spec.lin_vel_x_range),
            r.uniform(*spec.lin_vel_y_range),
            r.uniform(*spec.ang_vel_z_range),
        )

    def _compute_reward(self) -> tuple[float, dict[str, float]]:
        self._lifetime_steps += 1
        v = self.body_lin_vel()
        gyro = self._gyro
        gravity = self._projected_gravity()

        still = self.W_STILL * float(np.exp(-float((v ** 2).sum()) / self.STILL_STD2))
        spin = self.W_SPIN * float(np.exp(-float((gyro ** 2).sum()) / self.SPIN_STD2))
        tilt2 = float((gravity[:2] ** 2).sum())
        upright = self.W_UPRIGHT * float(np.exp(-tilt2 / self.UPRIGHT_STD2))
        dz = float(self._trunk_xpos[2]) - self.target_height()
        height = self.W_HEIGHT * float(np.exp(-(dz * dz) / self.HEIGHT_STD2))

        contacts = self._foot_contacts()
        planted = self.W_PLANTED * float(contacts["left"] and contacts["right"])

        # Soles parallel to the floor: the sole's own +z against world +z.
        flat = 0.0
        for bid in self._foot_body_ids:
            up = self.data.xmat[bid].reshape(3, 3)[2, 2]      # R[2,2] = z . z
            flat += float(np.exp(-((1.0 - up) ** 2) / self.FLAT_STD2))
        flat_feet = self.W_FLAT_FEET * flat / max(len(self._foot_body_ids), 1)

        # Distance from where this episode started.
        dxy = self._trunk_xpos[:2] - self._spawn_xy
        anchor = self.W_ANCHOR * float(
            np.exp(-float((dxy ** 2).sum()) / self.ANCHOR_STD2))

        joint_pos_rel = self._joint_pos_rel()
        pose_err = joint_pos_rel[self._pose_ids] - self.pose_target_rel()
        pose = self.W_POSE * float(np.exp(
            -float((pose_err ** 2).sum()) / self.POSE_STD2))

        # Carriage: waist and arms held at the default, TIGHTLY.
        #
        # `pose` spreads one Gaussian over all 29 joints at std2 4.0, which is
        # so wide it holds nothing — measured on the first squat that met its
        # height and uprightness targets: waist_roll +29.8 deg, waist_pitch
        # +29.9, right_shoulder_roll -29.1 against left -9.5, and left/right
        # mismatches of 20-25 deg on wrists, ankles and shoulders. It squatted
        # correctly and looked wrong: back twisted, one arm folded onto the
        # thigh. Total upper-body error was ~1.05 rad^2, which `pose` charges
        # exp(-1.05/4.0) = 0.77 of a weight-1.0 term. Nothing was asking for a
        # tidy carriage.
        #
        # This is a SEPARATE term, not a tighter `pose`, because the legs must
        # stay free to balance: pinning the whole body hard makes the tidiness
        # reward fight the height and upright terms that keep it up. The
        # target is the default upper body, which is already the clean pose —
        # waist at zero, shoulders mirrored at +/-11.5 deg, both elbows 34.4.
        carriage = 0.0 if not self.W_CARRIAGE else self.W_CARRIAGE * float(
            np.exp(-float((self.carriage_weights()
                           * (joint_pos_rel[self._carriage_ids]
                              - self.carriage_target_rel()) ** 2).sum())
                   / self.CARRIAGE_STD2))

        # Penalties — each <= 0 by construction (AGENTS.md).
        action_rate = self._action_rate_weight() * -float(
            ((self.last_action - self.prev_action) ** 2).sum())
        joint_vel = self.W_JOINT_VEL * -float((self._joint_vel() ** 2).sum())

        terms = {
            "still": still, "no_spin": spin, "upright": upright,
            "height": height, "feet_planted": planted, "pose": pose,
            "carriage": carriage,
            "flat_feet": flat_feet, "stay_home": anchor,
            "action_rate_penalty": action_rate,
            "joint_vel_penalty": joint_vel,
        }
        return float(sum(terms.values())), terms

    def carriage_weights(self):
        """Per-joint grip on the carriage. 1.0 everywhere unless a task needs
        some limb free for part of its cycle (see G1StrikeEnv)."""
        return 1.0

    def carriage_target_rel(self):
        """What a tidy carriage means right now, relative to the default pose.

        Zero is the standing pose. A task whose carriage MOVES — a strike
        that is mid-throw — overrides this so the term asks for the pose that
        instant calls for, not the one it started from.
        """
        return 0.0

    def pose_target_rel(self):
        """Joint angles the pose term pays for, relative to the default pose.

        Zero is the standing pose. A task that is held in a DIFFERENT
        configuration overrides this instead of switching the term off —
        see G1SquatEnv.
        """
        return 0.0


class G1SquatEnv(G1StandEnv):
    """Hold a squat: the idle, but parked lower.

    The duck's own `crouch` / `deep_squat` recipes are the template — height
    becomes the dominant earner, and the POSE term goes away entirely,
    because a squat is precisely not the default pose and paying for both
    asks the policy to bend its knees while holding them straight.

    Depth follows the duck's ratio rather than a guessed absolute: its crouch
    sits ~26% below its standing height, which on a 0.758 m pelvis is 0.20 m
    down. Everything that keeps the idle still — flat soles, staying on the
    spot, no spin, both feet down — is inherited unchanged, because a squat
    that wanders or rocks is the same failure as an idle that does.
    """

    SQUAT_DROP_M = 0.20      # pelvis target = stand_z - this (~26%, the duck's ratio)
    # The height term has to REACH the policy where it starts, which is 20 cm
    # above the target. MEASURED on the first attempt, with the idle's narrow
    # Gaussian (std2 0.006, weight 3): standing pays 0.004/step of height, so
    # the reward was flat exactly where the policy was standing — it collected
    # 2069 of `upright` and 17.8 of `height` over a run and never once tried
    # to bend. A wide Gaussian pays 1.58/step at 20 cm and 4.30 at 10 cm: a
    # slope to walk down. The weight goes up too, because squatting has to be
    # worth more than the comfort of the standing basin it must leave.
    W_HEIGHT = 6.0           # the dominant earner, as in the duck's recipes
    HEIGHT_STD2 = 0.03       # m^2 — 17 cm off target still pays a third
    # NOT zero. Deleting the pose term was measured to be the whole failure:
    # with W_POSE = 0 nothing at all specified the CONFIGURATION, only "be
    # 20 cm lower" — and the cheapest way to drop the pelvis 20 cm is to
    # pitch forward over the ankles. At 700k steps the height term was paying
    # 5.42 of 6.0 (it really was getting low) while 2 of 3 deterministic
    # seeds ended face-down inside 2 s, trunk up-axis 0.19-0.27 where 1.0 is
    # vertical. The third seed found the true squat and held 0.586 m upright
    # for the full 20 s, so the skill was reachable — it just was not the
    # only thing being paid for.
    #
    # The term is RETARGETED rather than removed: the docstring's objection
    # ("asks it to bend its knees and keep them straight at the same time")
    # is an objection to the TARGET, not to the term, and the solved squat
    # pose is a target that wants exactly what the height term wants.
    W_POSE = 1.0
    # ON for the squat: see G1StandEnv._compute_reward. The first squat that
    # held its height and uprightness still looked wrong — back twisted 30 deg,
    # one arm folded onto the thigh — because nothing paid for the carriage.
    W_CARRIAGE = 2.0
    # Terminating at the idle's 0.50 m would end an episode for squatting
    # correctly, so the floor moves down with the target.
    FALL_HEIGHT = 0.30

    # Fraction of episodes that START in the squat. THIS is what makes the
    # task learnable, and no amount of reward shaping replaced it: warm
    # started from a solid idle, with the height term widened so that
    # standing still pays 1.58/step of it, the policy held 0.757 m — dead on
    # standing height — for 30 s on every seed and never bent a knee. It
    # collected `upright` 2432 and `still` 3221 per run; descending risked
    # all of that for a term it had never once seen paid in full, because it
    # had never once BEEN there. AGENTS.md: "If rollouts never contain the
    # skill you're paying for, fix the physics curriculum, not the reward —
    # an unsampled state's value is never learned." Spawning in the squat
    # samples the state; the standing spawns then teach the descent into it.
    SQUAT_SPAWN_PROB = 0.5

    def __init__(self, *args, squat_drop_m: float | None = None,
                 squat_spawn_prob: float | None = None, **kwargs):
        self.squat_drop_m = (self.SQUAT_DROP_M if squat_drop_m is None
                             else float(squat_drop_m))
        self.squat_spawn_prob = (self.SQUAT_SPAWN_PROB if squat_spawn_prob is None
                                 else float(squat_spawn_prob))
        super().__init__(*args, **kwargs)
        self._fall_height = self.FALL_HEIGHT
        self._squat_joints = self._solve_squat_pose(self.squat_drop_m)
        # the pose term works in default-relative coordinates
        self._squat_pose_rel = (self._squat_joints
                                - self.default_pose)[self._pose_ids]

    def target_height(self) -> float:
        return self.stand_z - self.squat_drop_m

    def pose_target_rel(self) -> np.ndarray:
        return self._squat_pose_rel

    def _solve_squat_pose(self, drop: float) -> np.ndarray:
        """Joint angles that shorten the legs by `drop` metres, MEASURED.

        A crouch is one coordinated bend: knee +t, hip_pitch -t/2,
        ankle_pitch -t/2 keeps thigh and shin symmetric about the hip so the
        torso stays vertical while the body sinks.

        The quantity to solve against is LEG CLEARANCE — the pelvis above the
        soles — not the pelvis height. The pelvis IS this robot's free-joint
        root, so its z is `qpos[2]` and no joint angle can move it: bisecting
        on it solved against a constant and saturated at the 2.0 rad search
        bound, handing back a fold the robot could not stand up from.
        Measured clearance: 0.756 m at rest, 0.583 at 1.0 rad, 0.305 at 2.0.
        """
        import mujoco

        names = list(self.robot.joint_names)
        idx = {n: i for i, n in enumerate(names)}
        legs = [(idx[f"{s}_hip_pitch_joint"], idx[f"{s}_knee_joint"],
                 idx[f"{s}_ankle_pitch_joint"]) for s in ("left", "right")]
        data = mujoco.MjData(self.model)
        pads = [g for ids in self.foot_geom_ids.values() for g in ids]

        def bent(t: float) -> np.ndarray:
            q = self.default_pose.astype(np.float64).copy()
            for hip, knee, ankle in legs:
                q[hip] -= t / 2.0
                q[knee] += t
                q[ankle] -= t / 2.0
            return q

        def clearance(t: float) -> float:
            """Pelvis height above the lowest sole, at this bend."""
            mujoco.mj_resetDataKeyframe(self.model, data, self.key_stand)
            data.qpos[self.joint_qpos_adr] = bent(t)
            mujoco.mj_forward(self.model, data)
            sole = min(float(data.geom_xpos[g][2] - self.model.geom_size[g][0])
                       for g in pads)
            return float(data.xpos[self.trunk_body_id][2]) - sole

        stand_clear = clearance(0.0)
        want = stand_clear - drop
        lo, hi = 0.0, 2.0                      # rad of knee bend
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if clearance(mid) > want:
                lo = mid
            else:
                hi = mid
        t = 0.5 * (lo + hi)
        if t >= 1.99:                          # the bound, not a solution
            raise ValueError(
                f"no knee bend below 2 rad shortens the legs by {drop} m "
                f"(clearance at rest {stand_clear:.3f} m)")
        self.squat_knee_rad = t
        self.squat_clearance = clearance(t)
        return bent(t)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self._rng.uniform() < self.squat_spawn_prob:
            import mujoco

            d = self.data
            d.qpos[self.joint_qpos_adr] = self._squat_joints
            # Put the root at the SOLVED clearance so the soles start on the
            # floor: subtracting the nominal drop assumed the legs shortened
            # by exactly that much, which is only true if the solve was exact.
            d.qpos[self._root_qpos + 2] = self.squat_clearance + 0.002
            d.qvel[:] = 0.0
            self._write_ctrl(self._squat_joints)
            mujoco.mj_forward(self.model, d)
            self._refresh_derived()
            self._spawn_xy = np.asarray(self._trunk_xpos[:2], float).copy()
            self.last_spawn = "squat"
            obs = self._get_obs()
        else:
            self.last_spawn = "standing"
        return obs, info
