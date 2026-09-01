"""Bit-parity regression for the optimized obs/reward hot path.

walk_env.py and behaviors.py were restructured for per-step CPU cost (the
profiled ~40-55% of env.step that is our Python, not MuJoCo): per-step caches
for projected gravity / joint pos+vel / heading velocity / contact scans
(active ONLY inside step(), so out-of-band callers and tests always see fresh
state), persistent mjData views (gyro, trunk xpos/xquat/xmat, actuator_force,
joint qpos/qvel slices), a slice-assembled observation vector, cached
limit-parking ranges (was 14 model.joint(name) lookups per step ≈ 40-50 us),
loop contact scans, and `x.sum()` / `.clip()` methods in place of the
`np.sum`/`np.mean`/`np.clip` wrappers where the reduction is the same code
path. Every rewrite is supposed to be BIT-IDENTICAL, so this file pins:

1. golden rollout fingerprints: multi-episode random-action rollouts (fixed
   seeds, both actuators, several behaviors incl. "run" and "backflip", plus
   the plain walk env) hashed step by step over the exact obs bytes, the
   reward float64 bytes, every per-term running sum's float64 bytes, and the
   termination/truncation flags — any single-bit drift anywhere in the
   pipeline changes the digest. Per-term first-episode sums are pinned as
   float.hex too, so a digest mismatch names the term that moved.
2. unit parity of the algorithmically rewritten pieces against verbatim
   copies of the pre-optimization code (_foot_contacts, _limit_parking_pen,
   foot flatness, head/body floor-contact scans), over real rollout states.
3. cache coldness: helpers consulted from OUTSIDE step() (tests and tools
   re-pose the env with mj_forward and read helpers directly) must always
   recompute from the live mjData.

The goldens were captured from the PRE-optimization implementation. Provenance
note (2026-08-30): the backflip recipe was being actively tuned in a parallel
session during the optimization pass (still_head + stick_it terms, bounded
straight_flip, reset-time clearing of _prev_tau/_prev_vz/_gp_prev_head), so
the backflip/imitate goldens were recaptured after those RECIPE changes
landed; the optimization itself was verified bit-identical against
pre-optimization goldens on every config at each stage. If an intentional env
change ever invalidates a golden, recapture it with the OLD reward code —
never by pasting in whatever new optimization work produces.
"""

import hashlib
import struct

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.behaviors import BehaviorEnv, _limit_parking_pen
from microduck_local.walk_env import MicroduckWalkEnv


@pytest.fixture(autouse=True)
def _clean_env_overrides(monkeypatch):
    for var in ("MICRODUCK_ACTUATOR", "MICRODUCK_EPISODE_S", "MICRODUCK_CLIP",
                "MICRODUCK_SPAWN_FAMILY_PROBS", "MICRODUCK_BF_SPAWN_LO",
                "MICRODUCK_BF_SPAWN_HI", "MICRODUCK_RAMP_OFFSET",
                "MICRODUCK_RUN_CMD"):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------- fingerprints

# (name, actuator, weight_overrides). "walkenv" is MicroduckWalkEnv itself;
# the rest are BehaviorEnv recipes. "run" carries the exact make_env training
# config (BAM + DR + random yaw); one_leg carries slider overrides plus an
# adopted CATALOG term (the "+ add a term" channel); imitate carries the
# reference-clip path (phase in the command slots, _mi terms).
CONFIGS = {
    "walkenv-xml": ("walkenv", "xml", None),
    "walkenv-bam": ("walkenv", "bam", None),
    "run-bam": ("run", "bam", None),
    "run-xml": ("run", "xml", None),
    "backflip-xml": ("backflip", "xml", None),
    "backflip-bam": ("backflip", "bam", None),
    "one_leg-xml": ("one_leg", "xml",
                    {"one_leg_hold": 2.0, "calm_body": 0.5, "no_stall": 1.0}),
    "stand-xml": ("stand", "xml", None),
    "imitate-xml": ("imitate", "xml", None),
}

SEED = 7
STEPS = 240
RESET_EVERY = 80  # forced resets → several episodes → several spawn draws


def build_env(key):
    name, actuator, overrides = CONFIGS[key]
    if name == "walkenv":
        # 1 s command resample so mid-episode _sample_commands runs too.
        return MicroduckWalkEnv(seed=SEED, actuator=actuator,
                                command_resample_s=1.0)
    kw = dict(seed=SEED, actuator=actuator, command_resample_s=1.0)
    if name == "run":
        kw.update(domain_rand=True, random_yaw=True, height_termination=False)
    else:
        kw.update(domain_rand=False, random_yaw=False)
    return BehaviorEnv(name, weight_overrides=overrides, **kw)


def fingerprint(env, steps=STEPS, reset_every=RESET_EVERY, seed=SEED):
    """sha256 over every obs byte, reward, per-term running sum and done flag
    of a multi-episode random-action rollout — plus the first episode's
    per-term sums as float.hex, for a readable diff when the digest moves."""
    h = hashlib.sha256()
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    h.update(obs.tobytes())
    first_ep = None
    for i in range(steps):
        a = rng.uniform(-1.0, 1.0, C.NUM_JOINTS).astype(np.float32)
        obs, r, term, trunc, _ = env.step(a)
        h.update(obs.tobytes())
        h.update(struct.pack("<d", r))
        h.update(b"\x01" if term else b"\x00")
        h.update(b"\x01" if trunc else b"\x00")
        for k, v in env.reward_sums.items():
            h.update(k.encode())
            h.update(struct.pack("<d", v))
        if term or trunc or (i + 1) % reset_every == 0:
            if first_ep is None:
                first_ep = {k: float(v).hex()
                            for k, v in env.reward_sums.items()}
            obs, _ = env.reset()
            h.update(obs.tobytes())
    return h.hexdigest(), first_ep or {}


# Captured from the pre-optimization implementation — see module docstring.
# fmt: off
# 2026-08-31: no_stall was added to the headstand recipe and removed again
# the same day (it taxed hold practice, never charged the relaxed parking
# heap). ALL fingerprints verified bit-identical across both changes — no
# headstand config is in this battery, and one_leg's no_stall comes from
# the explicit weights above, not the headstand recipe.
# backflip-*/imitate-xml recaptured 2026-08-30 after an INTENTIONAL reward
# change (still_head term, straight_flip bound, per-episode reset of
# _prev_tau/_prev_vz/_gp_prev_head). run/walk/stand/one_leg digests were
# verified bit-identical across the change.
GOLDEN = {'backflip-bam': ('9ec130e649abdf289fdea677cc460cc11a9229af378564627107d18fcc254eb8',
                  {'arch_over': '0x0.0p+0',
                   'calm_landed_penalty': '0x0.0p+0',
                   'feet_under': '0x0.0p+0',
                   'flip_progress': '0x1.3a79dbd77f9a8p+0',
                   'gentle_head_penalty': '-0x1.791d1f170f726p+1',
                   'gentle_joints_penalty': '-0x1.0c76e68991535p+2',
                   'head_rise': '0x0.0p+0',
                   'land_tall': '0x0.0p+0',
                   'landed_hold': '0x0.0p+0',
                   'lean_back': '0x1.cbce29ff80000p+2',
                   'legs_over': '0x1.95d954694ca05p-3',
                   'neck_kip': '0x0.0p+0',
                   'neck_pushup': '0x0.0p+0',
                   'no_jaw_parking_penalty': '0x0.0p+0',
                   'no_limit_parking_penalty': '-0x1.6826786707ae1p+0',
                   'push_off': '0x1.908ce1f7ec925p+3',
                   'save_energy_penalty': '-0x1.d747cb4d7d082p+3',
                   'smooth_moves_penalty': '-0x1.e2b164bd70a3dp+3',
                   'soft_landings_penalty': '-0x1.0c63eb80707d8p+0',
                   'stand_pose': '0x0.0p+0',
                   'stay_home_penalty': '-0x1.36b53378dca21p-1',
                   'stick_it_penalty': '0x0.0p+0',
                   'still_head_penalty': '-0x1.f7eac7cdac219p+2',
                   'straight_flip_penalty': '-0x1.98f6b25946fc1p+4',
                   'tuck_ball': '0x1.3361b5f71013fp+2'}),
 'backflip-xml': ('bf392e15e84896add7b520102484da37567124c73060e157d9701dcc8a91c3be',
                  {'arch_over': '0x0.0p+0',
                   'calm_landed_penalty': '0x0.0p+0',
                   'feet_under': '0x0.0p+0',
                   'flip_progress': '0x1.fcb15fe575e77p-12',
                   'gentle_head_penalty': '-0x1.362cd6da4d046p+1',
                   'gentle_joints_penalty': '-0x1.f09bc4d513229p+2',
                   'head_rise': '0x0.0p+0',
                   'land_tall': '0x0.0p+0',
                   'landed_hold': '0x0.0p+0',
                   'lean_back': '0x0.0p+0',
                   'legs_over': '0x0.0p+0',
                   'neck_kip': '0x0.0p+0',
                   'neck_pushup': '0x0.0p+0',
                   'no_jaw_parking_penalty': '-0x1.30cc2674d8d40p+4',
                   'no_limit_parking_penalty': '-0x1.76c4e46919c9bp+1',
                   'push_off': '0x1.e9c9d5cbcda45p+2',
                   'save_energy_penalty': '-0x1.2190ca397ad96p+3',
                   'smooth_moves_penalty': '-0x1.e2b164bd70a3dp+3',
                   'soft_landings_penalty': '-0x1.93dfd1ed6a677p-1',
                   'stand_pose': '0x0.0p+0',
                   'stay_home_penalty': '-0x1.3b1f8b4778391p-3',
                   'stick_it_penalty': '0x0.0p+0',
                   'still_head_penalty': '-0x1.1c2479c61b8bfp+2',
                   'straight_flip_penalty': '-0x1.3c427833cf49ep+4',
                   'tuck_ball': '0x0.0p+0'}),
 'imitate-xml': ('a76aac5d6946b548a7773c433b84110472d6f84d09e261586b9a1ddc46e7241a',
                 {'gentle_head_penalty': '-0x1.9d911e7866b08p+1',
                  'no_limit_parking_penalty': '-0x1.76c4e46919c9bp+1',
                  'no_slip_penalty': '0x0.0p+0',
                  'no_spin_penalty': '0x0.0p+0',
                  'on_feet': '0x0.0p+0',
                  'pose_match': '0x1.e3e619ceb00f7p+7',
                  'rotation_match': '0x1.922751bf6f23bp+6',
                  'save_energy_penalty': '-0x1.2190ca397ad96p+2',
                  'soft_landings_penalty': '-0x1.93dfd1ed6a677p-1',
                  'stick_it': '0x0.0p+0',
                  'travel': '0x0.0p+0'}),
 'one_leg-xml': ('36d7e7ace2ff33f7f0f6589174329d3b04c614656aea96e0761dd121c1be8790',
                 {'calm_body_penalty': '-0x1.14e5dc404ee58p+3',
                  'face_home_penalty': '-0x1.a80ab28602d09p+3',
                  'flat_stance_foot': '0x1.9875d537f54abp+5',
                  'foot_in_air': '0x1.b439f9534edddp-1',
                  'gentle_joints_penalty': '-0x1.b7451ae8ba2ecp+3',
                  'head_up': '0x1.fecbd7a340562p+3',
                  'head_up_pull': '0x1.ffadd5d8e80ecp+4',
                  'no_stall_penalty': '0x0.0p+0',
                  'one_leg_hold': '0x1.0f9e02e7b0000p+4',
                  'planted_foot': '0x1.f000000000000p+3',
                  'save_energy_penalty': '-0x1.5b58b94507a49p+2',
                  'smooth_moves_penalty': '-0x1.642d36999999cp+4',
                  'stay_home_penalty': '-0x1.522ae80def5d0p-2',
                  'stay_put_penalty': '-0x1.c2edc35d5efe1p+0',
                  'stay_upright': '0x1.21eb0a0e28283p+5'}),
 'run-bam': ('85ca0a2ba22bca35522155d05473d3d7d2092c1959a4a36f5b0e3f0dbc164e5e',
             {'air_time': '0x0.0p+0',
              'calm_roll_penalty': '-0x1.d90753004357ep+2',
              'foot_clearance_penalty': '-0x1.ea015b7da1de7p-2',
              'head_up': '0x1.2dda930500000p+7',
              'keep_pace': '0x1.7f179f4d3af10p+1',
              'no_limit_parking_penalty': '-0x1.92befbdc163bcp+0',
              'plant_the_foot_penalty': '-0x1.e92f02c8df63fp-5',
              'pose': '0x1.9c39f3aa12fffp+4',
              'smooth_moves_penalty': '-0x1.5417e7999999cp+5',
              'stay_upright': '0x1.0989faca90000p+6',
              'track_turn': '0x1.a85643e8b4cf2p+1'}),
 'run-xml': ('642d38dcf376af521cc8ffced9f57331dd534e711df5ffc0701a74cf5f91da33',
             {'air_time': '0x0.0p+0',
              'calm_roll_penalty': '-0x1.63a26826ddfafp+3',
              'foot_clearance_penalty': '-0x1.c9651af382dd1p-1',
              'head_up': '0x1.a515983f00000p+7',
              'keep_pace': '0x1.0e318d39f2144p+5',
              'no_limit_parking_penalty': '-0x1.f48de1e0690b0p+0',
              'plant_the_foot_penalty': '-0x1.54956149d64d4p-3',
              'pose': '0x1.1c30a0edfed85p+5',
              'smooth_moves_penalty': '-0x1.f432d7f333336p+5',
              'stay_upright': '0x1.9c1e04474c000p+6',
              'track_turn': '0x1.7508918ed4ff8p+3'}),
 'stand-xml': ('034adbb243264dee0055faf3f5b47fc7714777bfc03caaab98f2c18545e5dc54',
               {'both_feet': '0x1.6800000000000p+4',
                'face_home_penalty': '-0x1.a80ab28602d09p+2',
                'flat_feet': '0x1.34e72cea4e8edp+5',
                'gentle_joints_penalty': '-0x1.24d8bc9b26c9ap+2',
                'head_up': '0x1.89d4e6d0bf655p+5',
                'hold_still_penalty': '-0x1.14e5dc404ee58p+4',
                'no_limit_parking_penalty': '-0x1.949ff21479d6dp+1',
                'normal_pose': '0x1.142ab8c28409cp+6',
                'save_energy_penalty': '-0x1.5b58b94507a49p+2',
                'smooth_moves_penalty': '-0x1.1cf0f87ae147bp+3',
                'stand_tall': '0x1.13cc4d11b4613p+7',
                'stay_home_penalty': '-0x1.522ae80def5d0p-2',
                'stay_upright': '0x1.828eb812e035ap+5'}),
 'walkenv-bam': ('e377aa752e35a8fce7ae06896dadea1122b7420080e2d3464e123e31f3d38936',
                 {'action_rate_penalty': '-0x1.5417e7999999cp+5',
                  'ang_vel_xy_penalty': '-0x1.d90753004357ep+3',
                  'feet_air_time': '0x0.0p+0',
                  'head_pose': '0x1.58f9cc9800000p+6',
                  'pose': '0x1.058027120a5ffp+5',
                  'track_ang_vel': '0x1.a85643e8b4cf2p+1',
                  'track_lin_vel': '0x1.c5a02920f346ep+2',
                  'upright': '0x1.c74dfa58a4220p+5'}),
 'walkenv-xml': ('4440ba91ed475c128022ceaad73313d46b8d3a862fad265a6e8a1ec07262ae6c',
                 {'action_rate_penalty': '-0x1.f432d7f333336p+5',
                  'ang_vel_xy_penalty': '-0x1.63a26826ddfafp+4',
                  'feet_air_time': '0x0.0p+0',
                  'head_pose': '0x1.d9aff89800000p+6',
                  'pose': '0x1.63534e9dbebfdp+5',
                  'track_ang_vel': '0x1.66faa98f43bb2p+3',
                  'track_lin_vel': '0x1.ca2751a3441a1p+4',
                  'upright': '0x1.57c377e2f50c0p+6'})}
# fmt: on


def _fingerprint_all():
    out = {}
    for key in CONFIGS:
        env = build_env(key)
        digest, first_ep = fingerprint(env)
        env.close()
        out[key] = (digest, first_ep)
    return out


@pytest.mark.parametrize("key", sorted(CONFIGS))
def test_rollout_fingerprint_matches_pre_optimization_golden(key):
    digest, first_ep = GOLDEN[key]
    env = build_env(key)
    try:
        got_digest, got_first = fingerprint(env)
    finally:
        env.close()
    # Per-term sums first: on a mismatch they say WHICH term moved.
    assert got_first == first_ep
    assert got_digest == digest


# ------------------------------------------------ unit parity vs verbatim refs
# Verbatim copies of the pre-optimization code paths (do not "modernize").


def ref_foot_contacts(env):
    n = int(env.data.ncon)
    if n == 0:
        return {"left": False, "right": False}
    g1 = env.data.contact.geom1[:n]
    g2 = env.data.contact.geom2[:n]
    floor = env.floor_geom
    left, right = env.foot_geoms["left"], env.foot_geoms["right"]
    with_floor = (g1 == floor) | (g2 == floor)
    return {
        "left": bool((((g1 == left) | (g2 == left)) & with_floor).any()),
        "right": bool((((g1 == right) | (g2 == right)) & with_floor).any()),
    }


def ref_limit_parking_pen(env):
    m, d = env.model, env.data
    q = d.qpos[env.joint_qpos_adr]
    lo = m.jnt_range[:, 0][[m.joint(n).id for n in C.JOINT_NAMES]]
    hi = m.jnt_range[:, 1][[m.joint(n).id for n in C.JOINT_NAMES]]
    mid, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-6)
    frac = np.abs(q - mid) / half
    over = np.maximum(0.0, frac - 0.88)
    return -25.0 * float(np.sum(over ** 2))


def ref_stance_flat(env, side):
    ref = env.foot_flat_ref[side]
    R = env.data.geom_xmat[env.foot_geoms[side]].reshape(3, 3)
    g_local = R.T @ np.array([0.0, 0.0, -1.0])
    d2 = float(np.sum((g_local - ref) ** 2))
    return float(np.exp(-d2 / 0.45 ** 2))


def ref_head_on_floor(env, heads):
    for i in range(env.data.ncon):
        g1, g2 = env.data.contact.geom1[i], env.data.contact.geom2[i]
        if env.floor_geom in (g1, g2):
            other = g2 if g1 == env.floor_geom else g1
            if int(env.model.geom_bodyid[other]) in heads:
                return True
    return False


def ref_body_floor_contacts(env, heads, feet):
    n = 0
    for i in range(env.data.ncon):
        g1, g2 = env.data.contact.geom1[i], env.data.contact.geom2[i]
        if env.floor_geom in (g1, g2):
            other = g2 if g1 == env.floor_geom else g1
            b = int(env.model.geom_bodyid[other])
            if b not in heads and b not in feet:
                n += 1
    return n


def test_rewritten_pieces_match_references_over_a_contact_rich_rollout():
    """backflip on the full-collision scene: trunk/head/floor contacts of
    every kind, plus limit-parked joints under random drive."""
    from microduck_local.behaviors import (
        _body_floor_contacts,
        _feet_body_ids,
        _flat_feet,
        _head_bodies,
        _head_on_floor,
        _stance_flat,
    )
    env = BehaviorEnv("backflip", seed=3, domain_rand=False, random_yaw=False)
    env.reset(seed=3)
    heads, feet = _head_bodies(env), _feet_body_ids(env)
    flat_l = _stance_flat("left")
    rng = np.random.default_rng(3)
    for i in range(150):
        env.step(rng.uniform(-1.5, 1.5, C.NUM_JOINTS).astype(np.float32))
        assert env._foot_contacts() == ref_foot_contacts(env)
        assert _limit_parking_pen(env) == ref_limit_parking_pen(env)
        assert _head_on_floor(env) == ref_head_on_floor(env, heads)
        assert _body_floor_contacts(env) == ref_body_floor_contacts(
            env, heads, feet)
        assert flat_l(env) == ref_stance_flat(env, "left")
        assert _flat_feet(env) == (ref_stance_flat(env, "left")
                                   + ref_stance_flat(env, "right")) / 2.0
        if (i + 1) % 60 == 0:
            env.reset()
    env.close()


# ----------------------------------------------------------- cache discipline


def test_helpers_track_out_of_band_state_changes():
    """Tests and tools re-pose the env (qpos write + mj_forward) and read the
    helpers directly, between steps. The step-scoped caches must never leak
    into that path: every helper reads the live mjData when consulted from
    outside step()."""
    env = BehaviorEnv("run", seed=0, domain_rand=False, random_yaw=False)
    env.reset(seed=0)
    env.step(np.zeros(C.NUM_JOINTS, np.float32))

    pitch = np.deg2rad(40)
    env.data.qpos[3:7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
    env.data.qvel[:] = 0.0
    env.data.qvel[0:3] = (0.3, 0.0, 0.0)
    mujoco.mj_forward(env.model, env.data)

    g = env._projected_gravity()
    assert abs(float(g[0]) - np.sin(pitch)) < 1e-6   # sees the NEW attitude
    fwd, _, _ = env.heading_lin_vel()
    assert abs(fwd - 0.3) < 1e-6                     # sees the NEW velocity
    body = env.body_lin_vel()
    assert abs(float(body[0]) - 0.3 * np.cos(pitch)) < 1e-6

    env.twist_cmd[:] = (0.25, 0.0, 0.0)              # out-of-band command poke
    from microduck_local.behaviors import _run_cmd_norm
    assert _run_cmd_norm(env) == pytest.approx(0.25)

    # ...and the next step() still recomputes everything freshly.
    obs, *_ = env.step(np.zeros(C.NUM_JOINTS, np.float32))
    assert np.isfinite(obs).all()
    env.close()
