from .locomotion import *  # noqa: F401,F403 — cascades the full upstream namespace,

# mirroring the flat file's definition order exactly (each module sees
# everything defined before it, helpers included).

class BehaviorEnv(MicroduckWalkEnv):
    """Walking env with the reward stack replaced by a behavior's recipe.

    Commands are pinned to zero (tiny keep-alive noise stays via reset
    sampling being overridden) so the 61-obs contract is preserved and the
    exported policy hot-swaps like any other.
    """

    def __init__(self, behavior_id: str,
                 weight_overrides: dict[str, float] | None = None,
                 spawn_overrides: dict[str, str] | None = None,
                 spotter: bool = False, clip_name: str | None = None,
                 standing_spawns: bool = False, **kwargs):
        self.behavior = BEHAVIORS[behavior_id]
        kwargs.setdefault("max_episode_s", self.behavior.episode_s)
        # Per-stage episode length (MICRODUCK_EPISODE_S, instance override or
        # trainer env var): a landing rehearsal is decided in ~4 s — clipping
        # those stages' episodes multiplies rehearsals per wall-clock and
        # spares the viewer 10 s of a duck lying still. Hard override: the
        # stage knob must win even when a caller passes max_episode_s.
        ep = ((spawn_overrides or {}).get("MICRODUCK_EPISODE_S")
              or os.environ.get("MICRODUCK_EPISODE_S"))
        if ep:
            try:
                kwargs["max_episode_s"] = float(ep)
            except ValueError:
                pass
        if self.behavior.scene == "all":
            kwargs.setdefault("scene_xml", str(C.SCENE_ALL_XML))
        kwargs.setdefault("terminate_on_fall", self.behavior.terminate_on_fall)
        # GPU locomotion has no height termination; a bouncing stride can dip
        # the trunk through 0.07 m without having fallen.
        if self.behavior.forward_cmd:
            kwargs.setdefault("height_termination", False)
        self.foot_contact_state = {"left": True, "right": True}
        # Reference motion, if this behavior imitates one. The clip is
        # selectable per run (a user authors several in the timeline editor):
        # explicit kwarg wins, then MICRODUCK_CLIP for the trainer subprocess,
        # then the recipe's default.
        name = resolve_clip_name(self.behavior, clip_name)
        self.clip = motion.load_clip(name) if name else None
        if self.clip is not None and self.clip.loop:
            # Locomotion rules, as every walking task uses (and as our own
            # one_leg/crouch/stand behaviors already do): a fall ends the
            # episode. Without this the duck face-plants at 0.7 s and spends
            # the next 3 s earning pose matches from the floor.
            kwargs["terminate_on_fall"] = True
            kwargs.pop("scene_xml", None)          # the walk scene, not "all"
            self.behavior = replace(self.behavior, scene="walk",
                                    terminate_on_fall=True)
        # Demo assist (see Behavior.spotter_fn) — showcase previews only.
        self.spotter = bool(spotter) and self.behavior.spotter_fn is not None
        self.spotter_active = False
        # Every episode starts from the STAND keyframe, spawn recipes off —
        # the lab's plain (non-showcase) assign, which shows a finished trick
        # off from its feet. Training and the trainee preview leave this False;
        # only a viewer-facing env that must not drop a duck mid-maneuver (or
        # lie it on the floor, for `stand`) sets it.
        self.standing_spawns = bool(standing_spawns)
        # Per-instance stage knobs (spawn windows/mix), consulted BEFORE
        # os.environ by _spawn_knob: the trainer subprocess can keep riding
        # its environment, but envs living inside the lab process (the
        # trainee preview) need per-instance values — os.environ there is
        # shared across every duck and never carries the active stage.
        self.spawn_overrides = dict(spawn_overrides or {})
        # Clamped to >= 0: a negative weight on a self-negating penalty would
        # double-negate into a reward for the violation (AGENTS.md's four-env
        # sign bug) — the UI's sliders must not be able to reintroduce it.
        self.weight_overrides = {
            k: max(0.0, float(v)) for k, v in (weight_overrides or {}).items()
        }
        # Overrides for keys OUTSIDE the recipe adopt that CATALOG term at the
        # given weight — the "＋ add a term" channel, riding the same weights
        # pipe as the sliders (train_behavior/scale restarts need no changes).
        recipe_keys = {t.key for t in self.behavior.terms}
        self._terms = tuple(self.behavior.terms) + tuple(
            CATALOG[k] for k in self.weight_overrides
            if k not in recipe_keys and k in CATALOG
        )
        # Hot-loop view of the recipe: (key, output name, default weight, fn)
        # per term, resolved once — _compute_reward runs it every step.
        self._term_rows = tuple(
            (t.key, t.key if not t.is_penalty else t.key + "_penalty",
             t.weight, t.fn)
            for t in self._terms)
        super().__init__(**kwargs)
        # Flat-foot reference: super().__init__ leaves the model posed at the
        # STAND keyframe (that's how stand_z is measured), so each foot's
        # foot-frame gravity right now IS what "flat on the floor" looks like.
        self.foot_flat_ref = {}
        for side, gid in self.foot_geoms.items():
            R = self.data.geom_xmat[gid].reshape(3, 3)
            self.foot_flat_ref[side] = R.T @ np.array([0.0, 0.0, -1.0])

    def step(self, action):
        if self.spotter:
            self.spotter_active = bool(self.behavior.spotter_fn(self))
        obs, reward, terminated, truncated, info = super().step(action)
        # NO overshoot terminal for the headstand (removed 2026-09-01). The
        # gx < -0.2 terminal (added so mid-flip catches that rolled past
        # wouldn't spend the clip getting up) priced every UNFOLD attempt at
        # catastrophe: from the balanced tuck-ball, the likeliest failure of
        # extending is a backward topple, and ending the episode forfeits all
        # remaining ball income — the biggest possible attempt tax, and the
        # covenant's own contract says falling never ends the episode ("it
        # can try, crumble, and try again"). Ball-converged brains (943186:
        # feet_up income 3x'd, extension frozen at gap 0.013) never risked
        # the unfold while this stood. The backward route is priced by
        # wrong_way per-step instead; wasted get-up time earns nothing and
        # is bounded by the 8 s clip.
        return obs, reward, terminated, truncated, info

    def _knob_prob(self, name: str, default: float) -> float:
        """Spawn-family probability with a per-instance/env override — lets a
        FINISHED showcase spawn standing (the recovery-rehearsal spawns are a
        training device; on the viewer's R the user wants the clean trick)."""
        v = _spawn_knob(self, name)
        if v:
            try:
                return float(v)
            except ValueError:
                pass
        return default

    def reset(self, **kwargs):
        out = super().reset(**kwargs)
        self.data.qfrc_applied[:] = 0.0   # never carry an assist across episodes
        self.spotter_active = False
        # Per-episode reward state (see Behavior.state_fn). Zeroed before the
        # spawn families run so a mid-maneuver spawn can PRESET it to match
        # the attitude it poses (a body spawned halfway through a flip has,
        # by definition, already rotated halfway).
        self._bf_rot = 0.0
        # Headstand progress-pay baselines (see _hs_update): cleared so the
        # first post-spawn step re-anchors them to THIS episode's start pose.
        self._hs_prev = None
        self._hs_streak = 0
        # Air-time bookkeeping is per-episode state too. _run_air_time banks
        # time while a foot is off the ground and pays it out at touchdown, so
        # time accrued during a terminal FALL would otherwise be paid on the
        # next episode's first contact -- free money at every spawn.
        self._run_air = {"left": 0.0, "right": 0.0}
        self._skid_air = {"left": 0.0, "right": 0.0}
        self._run_was = {"left": True, "right": True}
        self._run_contact_age = {"left": 0, "right": 0}
        # Audit item: these lazy-init memories leaked across episodes — one
        # spurious soft-landing / torque-rate / head-contact charge on the
        # first step after a violent episode end.
        self._prev_tau = None
        self._prev_vz = None
        self._gp_prev_head = False
        # What kind of start this episode got — surfaced on the viewer's duck
        # label so a watcher can tell a landing rehearsal from a plain
        # standing start (visually near-identical for some spawn families).
        self.last_spawn = "standing"
        u = self._rng.uniform()
        if self.standing_spawns:
            pass                      # keyframe start, whatever the recipe asks
        elif self.behavior.spawn_families:
            # Per-stage spawn MIX override (comma-separated probs, positional):
            # a curriculum stage labeled "learning to land" must actually be
            # mostly landing spawns — the declared mix (tuned for the final
            # integration stage) left 55% of a focused stage's episodes as
            # plain upright starts, rehearsing nothing the stage is for (a
            # user watched stage 1 and rightly asked what it was doing).
            fams = self.behavior.spawn_families
            probs_env = _spawn_knob(self, "MICRODUCK_SPAWN_FAMILY_PROBS")
            if probs_env:
                try:
                    probs = [float(x) for x in probs_env.split(",")]
                except ValueError:
                    probs = []
                if len(probs) == len(fams):
                    fams = tuple((p, fn) for p, (_, fn) in zip(probs, fams))
            acc = 0.0
            for prob, fn in fams:
                acc += prob
                if u < acc:
                    out = (fn(self), out[1])
                    kind = fn.__name__.split("_spawn_")[-1].replace("_", "-")
                    if 0.0 < self._bf_rot < 2.0 * np.pi:
                        kind += f" {np.degrees(self._bf_rot):.0f}°"
                    self.last_spawn = kind
                    break
        elif u < self._knob_prob("MICRODUCK_INVERTED_SPAWN_PROB",
                                 self.behavior.inverted_spawn_prob):
            out = (self._spawn_inverted(), out[1])
            self.last_spawn = "inverted"
        elif u < (self._knob_prob("MICRODUCK_INVERTED_SPAWN_PROB",
                                  self.behavior.inverted_spawn_prob)
                  + self._knob_prob("MICRODUCK_MID_FLIP_SPAWN_PROB",
                                    self.behavior.mid_flip_spawn_prob)):
            out = (self._spawn_mid_flip(), out[1])
            self.last_spawn = "mid-flip"
        # Anchors for stay_home / face_home: where this episode began.
        self.home_xy = (float(self.data.xpos[self.trunk_body_id][0]),
                        float(self.data.xpos[self.trunk_body_id][1]))
        self.home_yaw = _trunk_yaw(self)
        return out

    def _spawn_inverted(self):
        """Reverse-curriculum spawn: drop onto the CROWN, still nose-down
        (~152°), legs up. That is the rounded-top pivot of a forward roll,
        not a vertical 180° sit on the flat underside of the head."""
        import mujoco
        r = self._rng
        d, m = self.data, self.model
        # ~152°, not 170°. At 170–180° the tucked head puts the FLAT
        # underside (bottom_head_shell) on the floor and limp falls onto
        # the back (pitch +96°) — the "rolling backwards on the bottom of
        # the head" in the viewer. At ~150° the ROUND crown is the contact,
        # gx > 0 (nose-down / forward roll), and the stack is 25° short of
        # vertical so the catch is rolling forward on that dome, which is
        # how the champion balanced.
        pitch = np.deg2rad(152.0) + r.uniform(-0.08, 0.08)
        # ROLL noise + an initial angular shove, both signs: the pure-pitch
        # spawn always toppled the same way, so the student practiced ONE
        # fall direction and its legs learned one flop (visible on screen —
        # every duck's legs collapsed over identically). Counterbalance is
        # only learnable if the falls come from everywhere.
        roll = r.uniform(-0.12, 0.12)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        d.qpos[:] = 0.0
        # quat = pitch-about-y composed with roll-about-x
        d.qpos[3:7] = [cp * cr, cp * sr, sp * cr, sp * sr]
        d.qpos[2] = 0.165 + r.uniform(-0.01, 0.01)  # head shell near the floor
        q = d.qpos  # leg joints ~straight (the target pose), light noise
        for i, adr in enumerate(self.joint_qpos_adr):
            q[adr] = r.uniform(-0.08, 0.08)
        # Neck TUCKED (chin to chest) — opposite-sign joints, so the crown
        # (top_head_shell) meets the floor, not the face. Same-sign −0.6
        # planted the beak; see _HS_NECK_TUCK.
        q[self.joint_qpos_adr[5]] = _HS_NECK_TUCK + r.uniform(-0.15, 0.15)
        q[self.joint_qpos_adr[6]] = _HS_HEAD_TUCK + r.uniform(-0.15, 0.15)
        d.qvel[:] = 0.0
        # A random angular shove (both axes, both signs) so the drop demands
        # an ACTIVE catch in a random direction, not a static pose. Knobbed
        # per stage: at 0.4 a fresh brain could never catch ANY drop and
        # learned a shove-proof CROUCH instead of balance (0ff0f7-s1, seen
        # on the watch sheet) — perturbation size is a curriculum too.
        kick = float(_spawn_knob(self, "MICRODUCK_INV_SPAWN_KICK", "0.08"))
        d.qvel[3] = r.uniform(-kick, kick)
        d.qvel[4] = r.uniform(-kick, kick)
        d.ctrl[:] = d.qpos[self.joint_qpos_adr]
        mujoco.mj_forward(m, d)
        # Plant the crown. The unplanted pose hovered ~6 mm (hold = 0 on
        # frame 0 of the one spawn that is supposed to pay it); a from-
        # scratch brain then never sampled the hold and learned to crumple.
        for _ in range(40):
            if _head_on_floor(self):
                break
            d.qpos[2] -= 0.002
            mujoco.mj_forward(m, d)
        self.prev_joint_vel = self._joint_vel().copy()
        return self._get_obs()

    def _spawn_mid_flip(self):
        """Mid-maneuver spawn: the face-plant tripod — nose down past
        vertical, crown on the floor, neck tucked, legs folded with feet
        still planted, butt at a random height. The kick-over's launch pad."""
        import mujoco
        r = self._rng
        d, m = self.data, self.model
        # Window overridable per curriculum stage (the backflip's
        # MICRODUCK_BF_SPAWN_LO/HI pattern): 2.5 is nearly done, 1.7 is the
        # shallow arrival a dive from standing actually reaches.
        # 2.2–2.55 rad (~126–146°): hips already going OVER the head, not
        # the shallow 1.7 rad toe-press (feet behind, body mass on the
        # wrong side of the pivot — ba4c43 camped there pushing on tiptoes).
        lo = float(_spawn_knob(self, "MICRODUCK_MF_PITCH_LO", "2.2"))
        hi = float(_spawn_knob(self, "MICRODUCK_MF_PITCH_HI", "2.55"))
        pitch = r.uniform(lo, hi)
        z = 0.13 + r.uniform(-0.015, 0.015)
        hip = r.uniform(0.15, 0.55)
        knee = r.uniform(0.2, 0.7)
        # Slump variant (MICRODUCK_MF_SLUMP_PROB): the dive's ACTUAL arrival,
        # measured off deterministic renders of teach-headstand-840b8a-s4 and
        # its fine-tune — pitch ~146°, trunk low (~0.065), legs folded under
        # with the shins resting and the feet UNLOADED. The launch-pad tripod
        # above always plants the feet, so a from-standing policy that dove
        # into this state had never once rehearsed getting out of it, and
        # froze there. Spawning it makes the recovery on-policy.
        if r.uniform() < self._knob_prob("MICRODUCK_MF_SLUMP_PROB", 0.0):
            pitch = r.uniform(2.4, 2.7)
            z = 0.075 + r.uniform(-0.01, 0.01)
            hip = r.uniform(1.0, 1.5)
            knee = r.uniform(1.0, 1.6)
        d.qpos[:] = 0.0
        d.qpos[3:7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
        d.qpos[2] = z
        q = d.qpos
        for i, adr in enumerate(self.joint_qpos_adr):
            q[adr] = r.uniform(-0.1, 0.1)
        q[self.joint_qpos_adr[2]] = -hip   # left hip_pitch (folded under)
        q[self.joint_qpos_adr[11]] = hip   # right hip_pitch (mirrored sign)
        q[self.joint_qpos_adr[3]] = -knee
        q[self.joint_qpos_adr[12]] = knee
        q[self.joint_qpos_adr[5]] = _HS_NECK_TUCK + r.uniform(-0.15, 0.15)
        q[self.joint_qpos_adr[6]] = _HS_HEAD_TUCK + r.uniform(-0.15, 0.15)
        d.qvel[:] = 0.0
        # The roll MOMENTUM a real dive carries (caught in review: the
        # backflip's mid-roll spawns are "still rolling", but these were
        # frozen statues — so the drills taught finishing from a static
        # tripod, never CATCHING legs that are actually flipping over,
        # which is the state a real entry produces).
        # Was 1.5: enough leftover pitch rate to fly past the crown and
        # spend the clip getting up. 0.6 still arrives rolling, not as a
        # statue, without the overshoot.
        spin = float(_spawn_knob(self, "MICRODUCK_MF_SPIN_MAX", "0.6"))
        d.qvel[4] = r.uniform(0.2, max(0.25, spin))  # nose-down pitch rate
        d.ctrl[:] = d.qpos[self.joint_qpos_adr]
        mujoco.mj_forward(m, d)
        self.prev_joint_vel = self._joint_vel().copy()
        return self._get_obs()

    def _sample_commands(self) -> None:
        # Tricks: zero twist, keep-alive noise on head/body slots.
        # Locomotion: GPU run command mix — standing bucket, 55% straight
        # forward with vx clamped ≥ 0.3, remainder omnidirectional. Speed
        # ceiling and standing fraction follow the GPU curricula.
        r = self._rng
        self.twist_cmd[:] = 0.0
        # Spin: per-episode DIRECTION command in the observable wz slot (the
        # signed spin_fast pay needs the policy to know which way; it also
        # makes the trick steerable). Reasserted in _get_obs so the farm's
        # trick-duck command zeroing can't blank it.
        if self.behavior.id == "spin":
            self._spin_dir = float(r.choice((-1.0, 1.0)))
            self.twist_cmd[2] = self._spin_dir
        if self.behavior.forward_cmd:
            pinned = _spawn_knob(self, "MICRODUCK_RUN_CMD")
            if pinned:
                try:
                    self.twist_cmd[0] = float(pinned)
                except ValueError:
                    pinned = None
            if not pinned:
                n = getattr(self, "_lifetime_steps", 0)
                speed = _run_cmd_speed(n)
                stand_p = _run_standing_frac(n)
                u = r.uniform()
                if u < stand_p:
                    self.twist_cmd[:] = 0.0
                elif u < stand_p + _RUN_FORWARD_FRAC:
                    vx = abs(float(r.uniform(-speed, speed)))
                    vx = max(vx, min(_RUN_FORWARD_CLAMP, speed))
                    self.twist_cmd[:] = (vx, 0.0, 0.0)
                else:
                    self.twist_cmd[:] = (
                        r.uniform(-speed, speed),
                        r.uniform(*_RUN_LIN_VEL_Y),
                        r.uniform(*_RUN_ANG_VEL_Z),
                    )
        self.head_cmd[:] = [r.uniform(lo, hi) for lo, hi in C.HEAD_CMD_RANGES]
        self.body_cmd[:] = [r.uniform(lo, hi) for lo, hi in C.BODY_CMD_RANGES]
        # Re-anchor the straightness terms to HERE: after an obedient turn
        # segment, the old spawn heading/line is ancient history — measured
        # up to ~-1500/episode charged for perfectly tracking the NEW straight
        # command against the OLD line (audit finding #1).
        if getattr(self, "trunk_body_id", None) is not None:
            self.home_xy = (float(self.data.xpos[self.trunk_body_id][0]),
                            float(self.data.xpos[self.trunk_body_id][1]))
            self.home_yaw = _trunk_yaw(self)

    def _get_obs(self):
        # Imitation needs a sense of TIME: the policy is memoryless and the
        # same body pose means different things at different points in a clip.
        # The phase rides in two body-command slots (noise otherwise), so the
        # 61-dim contract is untouched — see motion.phase_signal.
        if self.clip is not None:
            s, c = self.clip.phase(self.step_count)
            self.body_cmd[4], self.body_cmd[5] = s, c
        if self.behavior.id == "spin":
            self.twist_cmd[2] = getattr(self, "_spin_dir", 1.0)
        return super()._get_obs()

    def _compute_reward(self):
        # Counter is owned and seeded by MicroduckWalkEnv.__init__ (which
        # reads MICRODUCK_RAMP_OFFSET so warm restarts resume ramps at
        # strength); here it only advances.
        self._lifetime_steps += 1
        self.foot_contact_state = self._foot_contacts()
        if self.behavior.state_fn is not None:
            self.behavior.state_fn(self)
        # _term_rows precomputes the "<key>_penalty" output names (an f-string
        # per penalty per step adds up at ~20 terms x 10k steps/s). The total
        # MUST stay builtin sum(): since 3.12 it runs Neumaier-compensated
        # summation over floats, so a plain `total += v` loop differs by an
        # ulp (found the hard way by the parity goldens).
        terms = {}
        wo = self.weight_overrides
        if wo:
            for key, out_key, w, fn in self._term_rows:
                terms[out_key] = wo.get(key, w) * fn(self)
        else:
            for key, out_key, w, fn in self._term_rows:
                terms[out_key] = w * fn(self)
        return float(sum(terms.values())), terms


# Star-export EVERYTHING (helpers included) so downstream modules and the
# package __init__ can reassemble the old flat-module surface exactly.
__all__ = [n for n in dir() if not n.startswith("__")]
