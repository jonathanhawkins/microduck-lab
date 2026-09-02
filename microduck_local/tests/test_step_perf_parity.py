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

2026-09-02: the upstream pin moved to the CAD re-export (microduck_rl
badc4e7), which moved every per-term sum in the 5th digit — a model
change, not a code change (the same code on the previous model still
matched). The goldens were recaptured on that model, and moved out of this
file into tests/goldens/ PER PLATFORM (golden_store.py): the digests lock
one machine's bit patterns, so each platform records its own and one
without a recording skips instead of failing — CI's Linux runner runs
them, a Mac records its own with MICRODUCK_RECORD_GOLDENS=1.
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


# The goldens live in tests/goldens/step_perf_parity-<platform>.json (see
# golden_store.py): the bit patterns are a property of one machine, and a
# model re-export moves them — record them on the machine and model they
# are meant to pin, never paste in what new optimization work produces.
GOLDEN_NAME = "step_perf_parity"


def _fingerprint_all():
    out = {}
    for key in CONFIGS:
        env = build_env(key)
        digest, first_ep = fingerprint(env)
        env.close()
        out[key] = (digest, first_ep)
    return out


def _golden():
    """This platform's recorded fingerprints, or None (skip); records them
    when MICRODUCK_RECORD_GOLDENS is set."""
    import golden_store as gs
    if gs.RECORD:
        data = {k: list(v) for k, v in _fingerprint_all().items()}
        print(f"recorded {gs.save(GOLDEN_NAME, data)}")
        return {"provenance": gs.provenance(), "data": data}
    return gs.load(GOLDEN_NAME)


@pytest.mark.parametrize("key", sorted(CONFIGS))
def test_rollout_fingerprint_matches_pre_optimization_golden(key):
    import golden_store as gs
    golden = _golden()
    if golden is None:
        pytest.skip(gs.skip_reason(GOLDEN_NAME))
    digest, first_ep = golden["data"][key]
    env = build_env(key)
    try:
        got_digest, got_first = fingerprint(env)
    finally:
        env.close()
    # Per-term sums first: on a mismatch they say WHICH term moved — and a
    # golden made against another model or library is named before that.
    stale = gs.check_provenance(golden)
    assert got_first == first_ep, stale or "a per-term sum moved (same model, same libraries)"
    assert got_digest == digest, stale or "the rollout digest moved"


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
