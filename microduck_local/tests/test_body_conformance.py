"""What this harness means by "a supported robot body", as one test file.

The robot tests are per-body — `test_g1_*`, a `needs_g1` skip in each — and
only `test_robot_spec.py` and `test_pose.py` walk `robots.spec.registry()`,
so nothing said what a body must SATISFY to be listed there. The G1 went in
behind five files and each property was pinned wherever it was convenient; a
third body (the Innate MARS, `docs/mars-roadmap.md` §6.2) would otherwise be
a third pass over the same ground, with its own idea of which properties
matter.

So this file is the definition: every case is parameterised over the
registry, and **a body is supported when this file is green for it**. What it
pins, and the mistake each case prevents, is in the individual docstrings.

Two conventions worth knowing before adding a case:

* **Every positive case has a planted negative beside it** (the repo rule: a
  test proves nothing until it has been shown to fail — two toothless tests
  were caught here that way). The positives put their assertions in small
  helpers (`_missing_names`, `_dimension_report`, `_check_visual_scene`) so
  the negative can hand the SAME code a broken copy of the spec and watch it
  fail. `RobotSpec` is `frozen=True, eq=False`, so a broken copy is
  `dataclasses.replace`; the G1's spec is a lazy proxy, hence `_resolved`.
* **Two seams do not exist yet** and are faked here by a one-line table with
  the body's id in it: which shipped policy holds a body up, and where its
  visual scene comes from. Both are `Body.shipped_policies()` /
  `Body.visual_scene()` in the MARS plan (§1); `viz_server.get_scene` carries
  the same if-chain today. When the seam lands, the tables below collapse to
  a registry lookup and nothing else in this file changes.
"""

from __future__ import annotations

import dataclasses
import json
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import pytest

from microduck_local import contract as C
from microduck_local.robots import spec as S
from microduck_local.train import env_class
from microduck_local.world.compose import ROBOT_XML

# The lab's stage pitch per body is this ratio on the body's measured width:
# 0.65 m of floor for a duck measured 0.1845 m across (robots/spec.py's own
# comment on `lab_spacing_m`). Six 1.3 m G1s inside each other is what a
# single duck-sized constant looked like.
LAB_PITCH_RATIO = 3.52
# Contacts during a 2 s hold, MEASURED on this Mac: 5 at the peak for the
# duck (2 foot pads on a plane), 28 for the G1 (14 foot capsules). The
# ceiling is a blow-up detector, not a tight bound — a solver that has lost
# the plot reports contacts in the hundreds or thousands.
CONTACT_CEILING = 200
HOLD_SECONDS = 2.0


# --------------------------------------------------------------- the roster

def _resolved(spec):
    """The `RobotSpec` behind a registry entry.

    The G1's is a `_LazyG1Spec` proxy (`load_config()` reads the fetched
    cache, so importing the module on a machine that never ran `fetch-g1`
    must not explode). `dataclasses.replace` needs the real thing.
    """
    return spec._resolve() if hasattr(spec, "_resolve") else spec


def _assets_ready(robot_id: str) -> bool:
    """Are this body's assets on disk? (`Body.ready()` in the MARS plan.)

    The fallback is generic — a body is fetched when the scene it trains in
    exists — but the G1 answers for itself, because its `scene_fn` GENERATES
    that scene and would compile a 0.7 s model at collection time.
    """
    if robot_id == "g1":
        from microduck_local.robots.g1 import g1_ready
        return g1_ready()
    try:
        return Path(S.get(robot_id).scene_fn()).is_file()
    except Exception:                       # a body that cannot even say
        return False


# What to run when a body's assets are missing (`Body.setup_hint()`).
_SETUP_HINT = {
    "microduck": "clone microduck_rl next to microduck_local, or set MICRODUCK_RL_DIR",
    "g1": "uv run fetch-g1",
}


def _robot_params():
    """Every id in the registry, each skipped if its assets are missing.

    Built from `registry()` rather than listed, so a body that is added
    without appearing here — the whole failure mode this file exists for —
    is impossible.
    """
    out = []
    for rid in sorted(S.registry()):
        hint = _SETUP_HINT.get(rid, "fetch this body's assets")
        reason = f"{rid} assets missing — {hint}"
        out.append(pytest.param(
            rid, marks=pytest.mark.skipif(not _assets_ready(rid), reason=reason)))
    return out


ROBOTS = _robot_params()


@lru_cache(maxsize=4)
def _model(robot_id: str) -> mujoco.MjModel:
    """The body's own training scene, compiled. Cached: the G1's costs 0.7 s."""
    return mujoco.MjModel.from_xml_path(str(S.get(robot_id).scene_fn()))


def _env(robot_id: str, **kw):
    """The walking env for this body, every randomizer off.

    `actuator_force` is passed explicitly, never left to the process:
    MICRODUCK_ACTUATOR is read by the plain `actuator` kwarg, so a lab or
    trainer process that exported it would otherwise change what this file
    measures. ("bam" is the duck's XL330 identification and the G1 refuses
    it, so "xml" is the one setting every body can be held to.)
    """
    kw.setdefault("obs_noise", False)
    kw.setdefault("domain_rand", False)
    kw.setdefault("action_delay", False)
    kw.setdefault("random_yaw", False)
    kw.setdefault("actuator_force", "xml")
    kw.setdefault("seed", 0)
    return env_class(robot_id, "walk")(**kw)


# ------------------------------------------------- 1. the scene and its names

def _missing_names(model: mujoco.MjModel, spec) -> list[str]:
    """Every NAME the spec declares that the compiled model does not have.

    A spec is "either a NAME the compiled model is asked for or a dimension"
    (robots/spec.py). The point of a name is that it fails LOUDLY: a wrong
    foot pad used to resolve to id -1, no contact ever matched it, and the
    air-time reward silently paid nothing. This is that scan, over every
    field at once, so a new field cannot be added without a home here.
    """
    miss: list[str] = []

    def need(kind: str, objtype, name: str) -> None:
        if mujoco.mj_name2id(model, objtype, name) < 0:
            miss.append(f"{kind}={name!r}")

    need("base_body", mujoco.mjtObj.mjOBJ_BODY, spec.base_body)
    need("gyro_sensor", mujoco.mjtObj.mjOBJ_SENSOR, spec.gyro_sensor)
    need("floor_geom", mujoco.mjtObj.mjOBJ_GEOM, spec.floor_geom)
    need("stand_keyframe", mujoco.mjtObj.mjOBJ_KEY, spec.stand_keyframe)
    for side, geoms in spec.foot_geoms.items():
        for g in geoms:
            need(f"foot_geoms[{side}]", mujoco.mjtObj.mjOBJ_GEOM, g)
    for b in spec.com_bodies:
        need("com_bodies", mujoco.mjtObj.mjOBJ_BODY, b)
    for e in spec.effectors:
        need(f"effectors[{e.id}].body", mujoco.mjtObj.mjOBJ_BODY, e.body)
    for j in spec.joint_names:
        need("joint_names", mujoco.mjtObj.mjOBJ_JOINT, j)
    # A rig control is a direction in (joints + rootPitch) space keyed by
    # joint NAME, served verbatim to the viewer, which DROPS a joint the body
    # lacks. Dropping is the right behaviour for a control authored for
    # another body; a control in THIS body's own spec naming a joint it does
    # not have is a mis-wired slider that fails silently in the browser.
    for rc in spec.rig_controls:
        names = set(rc.get("parts") or {}) | set(rc.get("pick") or ())
        handle = (rc.get("handle") or {}).get("joint")
        if handle:
            names.add(handle)
        for j in sorted(names - {"root"}):   # "root" is the free base, not a joint
            need(f"rig_controls[{rc.get('id')}]", mujoco.mjtObj.mjOBJ_JOINT, j)
    return miss


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_bodys_scene_compiles_and_every_declared_name_resolves(robot):
    """The floor of "supported": `scene_fn()` compiles, it carries the STAND
    keyframe the env spawns from, and nothing the spec names is a typo."""
    spec = S.get(robot)
    model = _model(robot)
    assert model.nq > 0 and model.nu > 0
    assert _missing_names(model, spec) == []
    key = model.key(spec.stand_keyframe)         # raises if absent
    assert key.qpos.shape[0] == model.nq, "the STAND keyframe is a different nq"
    # walk_env reads the gyro as sensordata[adr:adr+3]; a 1-d sensor under
    # that name would hand the policy two neighbouring channels.
    assert int(model.sensor(spec.gyro_sensor).dim[0]) == 3


def _name_plants(robot: str) -> list[tuple[str, dict]]:
    """One planted wrong name per KIND of name a spec carries."""
    spec = _resolved(S.get(robot))
    side, geoms = next(iter(spec.foot_geoms.items()))
    rig = spec.rig_controls[0]
    return [
        ("base_body", dict(base_body="no_such_link")),
        ("gyro_sensor", dict(gyro_sensor="no_such_sensor")),
        ("floor_geom", dict(floor_geom="no_such_floor")),
        ("stand_keyframe", dict(stand_keyframe="NO_SUCH_KEY")),
        ("foot_geoms", dict(foot_geoms={**spec.foot_geoms,
                                        side: ("no_such_pad",) + geoms[1:]})),
        ("com_bodies", dict(com_bodies=spec.com_bodies + ("no_such_body",))),
        ("effectors", dict(effectors=tuple(
            dataclasses.replace(e, body="no_such_body") if i == 0 else e
            for i, e in enumerate(spec.effectors)))),
        ("joint_names", dict(joint_names=("no_such_joint",) + spec.joint_names[1:])),
        ("rig_controls", dict(rig_controls=(
            {**rig, "parts": {**rig["parts"], "no_such_joint": 1.0}},)
            + spec.rig_controls[1:])),
    ]


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_name_scan_catches_a_wrong_name_of_every_kind(robot):
    """The planted regression for the scan above.

    Each field gets a deliberately wrong name in a copy of the spec, and the
    same scan must report THAT field. Without this, a scan that silently
    stopped checking a field (a renamed attribute, an empty tuple) would keep
    passing — which is how a toothless test gets written.
    """
    model = _model(robot)
    spec = _resolved(S.get(robot))
    for kind, override in _name_plants(robot):
        miss = _missing_names(model, dataclasses.replace(spec, **override))
        assert any(m.startswith(kind) for m in miss), (
            f"{robot}: a wrong {kind} was not reported ({miss})")


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_env_refuses_a_spec_naming_a_geom_the_model_lacks(robot):
    """Not just the scan above: the ENV itself must refuse the bad name.

    `test_robot_spec.py` pins this for the duck. It is a property of every
    body — the lookups are spec-driven — and a body whose env swallowed a
    missing foot pad would train against a reward that never fires.
    """
    spec = _resolved(S.get(robot))
    side, geoms = next(iter(spec.foot_geoms.items()))
    bad = dataclasses.replace(
        spec, foot_geoms={**spec.foot_geoms, side: ("no_such_pad",) + geoms[1:]})
    with pytest.raises(KeyError, match="no_such_pad"):
        _env(robot, robot=bad)


# ----------------------------------------------------------- 2. the dimensions

def _dimension_report(spec) -> list[str]:
    """Every way a spec's numbers can disagree with each other."""
    bad: list[str] = []
    nj = spec.num_joints
    if len(spec.default_pose) != nj:
        bad.append(f"default_pose is {len(spec.default_pose)} for {nj} joints")
    if spec.num_actions != nj:
        bad.append(f"num_actions {spec.num_actions} != {nj} joints")
    if spec.action_scale is not None and len(spec.action_scale) != nj:
        bad.append(f"action_scale is {len(spec.action_scale)} for {nj} joints")
    if spec.pose_joint_ids is not None:
        ids = np.asarray(spec.pose_joint_ids)
        if ids.size and (ids.min() < 0 or ids.max() >= nj):
            bad.append(f"pose_joint_ids {ids.min()}..{ids.max()} outside 0..{nj - 1}")
    if spec.joint_groups is not None and len(spec.joint_groups) != nj:
        bad.append(f"joint_groups is {len(spec.joint_groups)} for {nj} joints")
    lo, hi = spec.twist_obs_slice
    if not (0 <= lo < hi <= spec.obs_dim):
        bad.append(f"twist_obs_slice {(lo, hi)} does not fit obs_dim {spec.obs_dim}")
    # A twist is at most (vx, vy, wz) — both legged bodies use all three; a
    # wheeled base would use two. Its consumer (`_seed_command_stats`) is
    # width-agnostic, so the ceiling is here to catch a slice typed one field
    # too wide, which would silently stop normalizing real observations.
    if not 1 <= hi - lo <= 3:
        bad.append(f"twist_obs_slice {(lo, hi)} is {hi - lo} wide, not 1-3")
    if spec.obs_dim <= 0:
        bad.append(f"obs_dim {spec.obs_dim}")
    return bad


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_specs_own_numbers_agree_with_each_other(robot):
    """A spec's dimensions are read by code that cannot sanity-check them.

    `scale_action` adds `default_pose` to the action, the pose reward indexes
    `pose_joint_ids`, the 🎬 panel labels joints by `joint_groups`, and
    `train._seed_command_stats` writes into `twist_obs_slice` — the last one
    because a policy cloned under a PINNED command has ~zero variance there
    and the first real command normalizes to thousands. Each of those is a
    silent wrong answer, not a crash, when the length is off by one.
    """
    assert _dimension_report(_resolved(S.get(robot))) == []


@pytest.mark.parametrize("robot", ROBOTS)
def test_a_planted_dimension_mismatch_is_caught(robot):
    """The planted regression: one broken number at a time, all reported."""
    spec = _resolved(S.get(robot))
    groups = spec.joint_groups or ("a",) * spec.num_joints
    lo, _hi = spec.twist_obs_slice
    plants = [
        ("a joint short of a default pose", "default_pose",
         dict(default_pose=spec.default_pose[:-1])),
        ("a joint short of a group label", "joint_groups",
         dict(joint_groups=groups[:-1])),
        ("a pose id one past the last joint", "pose_joint_ids",
         dict(pose_joint_ids=np.array([spec.num_joints]))),
        ("a twist slice hanging off the end", "twist_obs_slice",
         dict(twist_obs_slice=(spec.obs_dim - 1, spec.obs_dim + 2))),
        ("a twist slice eight floats wide", "twist_obs_slice",
         dict(twist_obs_slice=(lo, lo + 8))),
        ("one action scale too many", "action_scale",
         dict(action_scale=np.ones(spec.num_joints + 1, np.float32))),
    ]
    for label, field, override in plants:
        bad = _dimension_report(dataclasses.replace(spec, **override))
        assert any(field in b for b in bad), f"{label} not reported: {bad}"


# -------------------------------------------------------------- 3. the mirror

@pytest.mark.parametrize("robot", ROBOTS)
def test_the_mirror_is_a_permutation_and_its_own_inverse(robot):
    """`symmetry.py` augments a rollout by applying this map to the joints.

    If it is not a permutation the augmented sample is a different robot
    (two joints reading the same servo, one read by none); if it is not an
    involution, mirroring twice is not the identity and the augmentation
    drifts the dataset instead of doubling it.
    """
    spec = S.get(robot)
    perm = spec.mirror_joint_perm()
    assert sorted(perm.tolist()) == list(range(spec.num_joints)), (
        f"{robot}: mirror is not a permutation: {perm.tolist()}")
    assert np.array_equal(perm[perm], np.arange(spec.num_joints))


@pytest.mark.parametrize("robot", ROBOTS)
def test_a_duplicated_joint_name_breaks_the_mirror_and_is_caught(robot):
    """The planted regression: `mirror_joint_perm` pairs by NAME with
    `names.index`, so a copy-pasted duplicate makes two joints mirror to the
    same slot and leaves another unreachable — a permutation no more."""
    spec = _resolved(S.get(robot))
    names = list(spec.joint_names)
    names[-1] = names[0]                      # the copy-paste
    perm = dataclasses.replace(spec, joint_names=tuple(names)).mirror_joint_perm()
    assert sorted(perm.tolist()) != list(range(len(names)))


# ------------------------------------------------------- 4. the policy contract

@pytest.mark.parametrize("robot", ROBOTS)
def test_the_env_serves_exactly_the_width_the_spec_declares(robot):
    """The lab refuses a policy on the wrong body BY OBSERVATION WIDTH (61 vs
    99, guarded at four sites in `viz_server`), and the exporter shapes the
    ONNX graph from the same number. So the spec's width, the env's declared
    space and the array `_get_obs()` actually returns have to be one number.

    They are three places today: `walk_env` sizes `observation_space` from
    the spec but the duck's `_get_obs` writes a literal 61-float layout (the
    deployment contract, deliberately fixed). This is the cross-check that
    they cannot drift apart.
    """
    spec = S.get(robot)
    env = _env(robot)
    obs, _info = env.reset(seed=0)
    assert env.observation_space.shape == (spec.obs_dim,)
    assert obs.shape == (spec.obs_dim,)
    assert env._get_obs().shape == (spec.obs_dim,)
    assert env.action_space.shape == (spec.num_actions,)
    assert np.isfinite(obs).all()
    # A zero action is the default pose, whatever the body's action scale.
    assert np.allclose(spec.scale_action(np.zeros(spec.num_actions, np.float32)),
                       spec.default_pose, atol=1e-6)


@pytest.mark.parametrize("robot", ROBOTS)
def test_a_spec_that_understates_its_obs_width_cannot_serve_an_observation(robot):
    """The planted regression for the cross-check above: a spec one float
    short of what its `_get_obs` assembles must not quietly hand out an
    observation of the declared length (the duck's fills 61 slots by
    literal, so the mismatch shows as a space that disagrees with the array;
    the G1's is spec-sized, so it raises)."""
    spec = _resolved(S.get(robot))
    bad = dataclasses.replace(spec, obs_dim=spec.obs_dim - 1)
    try:
        env = _env(robot, robot=bad)
        obs, _ = env.reset(seed=0)
    except (ValueError, IndexError):
        return                                 # raised: caught
    assert obs.shape != env.observation_space.shape, (
        f"{robot}: a short obs_dim served a {obs.shape} observation as "
        f"{env.observation_space.shape} — the widths can drift silently")


# ----------------------------------------------------------------- 5. it holds

def _hold_policy(robot_id: str) -> Path | None:
    """The shipped policy that holds this body STILL, or None.

    `Body.shipped_policies()` in the MARS plan; today the duck's live in
    `../microduck/policies` (the lab's `POLICIES_DIR`) and the G1's in its
    fetched cache, and only this table knows which one is the idle. The G1's
    `walker.onnx` IS its idle — measured at zero command, 60 s, two seeds:
    1.2-1.7 cm of drift, both feet down, no falls (robots/g1.py).
    """
    if robot_id == "microduck":
        from microduck_local.viz_server import POLICIES_DIR
        p = POLICIES_DIR / "alpha_stand.onnx"
        return p if p.is_file() else None
    if robot_id == "g1":
        from microduck_local.robots.g1 import walker_onnx
        p = walker_onnx()
        return p if p.is_file() else None
    return None


def _hold(robot_id: str, seconds: float, policy: Path | None):
    """Spawn at STAND, hold the zero command for `seconds`, report the worst.

    `policy=None` is the open-loop hold (a zero action = the default pose
    through the position servos), which is the planted regression below.
    """
    spec = S.get(robot_id)
    env = _env(robot_id, max_episode_s=60.0)
    env.reset(seed=0)
    sess = inp = None
    if policy is not None:
        sess = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        shape = sess.get_inputs()[0].shape
        assert int(shape[1]) == spec.obs_dim, (
            f"{policy.name} takes obs{shape}, {robot_id} speaks {spec.obs_dim}")
    worst = dict(z=np.inf, gravity_z=-np.inf, ncon=0, nan=False, terminated_at=None)
    for i in range(int(round(seconds / C.CTRL_DT))):
        # Pin every command slot: the env resamples on its own clock, and a
        # body asked to walk is not being asked to hold still.
        env.twist_cmd[:] = 0.0
        for slot in ("head_cmd", "body_cmd"):
            if hasattr(env, slot):
                getattr(env, slot)[:] = 0.0
        obs = env._get_obs()
        if sess is None:
            action = np.zeros(spec.num_actions, np.float32)
        else:
            action = sess.run(None, {inp: obs.reshape(1, -1)})[0][0].astype(np.float32)
        _obs, _r, terminated, _trunc, _info = env.step(action)
        worst["z"] = min(worst["z"], float(env.data.xpos[env.trunk_body_id][2]))
        worst["gravity_z"] = max(worst["gravity_z"], float(env._projected_gravity()[2]))
        worst["ncon"] = max(worst["ncon"], int(env.data.ncon))
        worst["nan"] = worst["nan"] or not (
            np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all())
        if terminated and worst["terminated_at"] is None:
            worst["terminated_at"] = i
    return worst


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_shipped_hold_policy_keeps_the_body_up_for_two_seconds(robot):
    """The spawn state has to be a state the body can be held in.

    A scene that compiles can still spawn a body inside the floor, with a
    keyframe from a different nq, or with an actuator sign flipped — none of
    which any name check sees, and all of which look like "the policy is
    bad" once training starts. Two seconds of holding still, with the fall
    thresholds the spec itself declares, is the cheapest statement that the
    spawn, the keyframe, the obs assembly and the action scaling all agree.

    MEASURED (this Mac, xml actuators, seeded reset):
        microduck  alpha_stand   z 0.125 -> 0.116 (min 0.114), gravity_z
                                 -0.998, ncon <= 5
        g1         walker.onnx   z 0.763 -> 0.765 (min 0.760), gravity_z
                                 -1.000, ncon <= 28

    A body with no shipped policy (Innate's MARS ships none — its learned
    skills are ACT checkpoints, not ONNX) skips this, and the open-loop case
    below carries the sanity half on its own.
    """
    policy = _hold_policy(robot)
    if policy is None:
        pytest.skip(f"{robot} ships no hold policy — see _hold_policy")
    spec = S.get(robot)
    worst = _hold(robot, HOLD_SECONDS, policy)
    assert not worst["nan"], f"{robot}: NaN in qpos/qvel during the hold"
    assert worst["terminated_at"] is None, (
        f"{robot}: the env called it a fall at step {worst['terminated_at']} "
        f"(z {worst['z']:.3f}, gravity_z {worst['gravity_z']:+.3f})")
    assert worst["z"] > spec.fall_height, (
        f"{robot}: base down to {worst['z']:.3f} m, floor is {spec.fall_height}")
    assert worst["gravity_z"] < spec.fall_gravity_z, (
        f"{robot}: tilted to gravity_z {worst['gravity_z']:+.3f}, "
        f"kill is {spec.fall_gravity_z}")
    assert worst["ncon"] < CONTACT_CEILING, (
        f"{robot}: {worst['ncon']} contacts — a solver blow-up, not a stand")


@pytest.mark.parametrize("robot", ROBOTS)
def test_an_open_loop_hold_topples_inside_that_window(robot):
    """The planted regression for the settle test, and the measurement that
    decides how it is written.

    "Spawn at `default_pose` and settle 2 s" cannot be asked of these two
    bodies open-loop: commanding the default pose through the position
    servos and waiting, both FALL — the duck at step 49 (0.98 s, xml; 81 =
    1.62 s under BAM), the G1 at step 61 (1.22 s). Both are legged and
    balance actively, so that is the physics, not a defect: a hold is a
    trained behaviour here (AGENTS.md, "open-loop holds topple"). Hence the
    shipped policy above.

    Keeping it as a test does two jobs: it shows the settle assertions bite
    (the same window, no policy, fails them), and its first two assertions —
    no NaN, no contact explosion — are the half of "it settles" that needs no
    policy, so a body that ships none is still held to something here.

    Every id in the registry today is one the WALKING env can walk
    (`RobotSpec`'s own first line), so every one of them topples. A wheeled
    base cannot fall at all: when `Body.kind` lands (MARS plan §1) this case
    keeps the two assertions above and asks for the topple only when
    `kind == "legged"`.
    """
    spec = S.get(robot)
    worst = _hold(robot, HOLD_SECONDS, None)
    assert not worst["nan"], f"{robot}: open-loop hold produced NaN, not a topple"
    assert worst["ncon"] < CONTACT_CEILING, (
        f"{robot}: {worst['ncon']} contacts holding the spawn pose open-loop")
    fell = (worst["terminated_at"] is not None
            or worst["z"] <= spec.fall_height
            or worst["gravity_z"] >= spec.fall_gravity_z)
    assert fell, (
        f"{robot} held the default pose open-loop for {HOLD_SECONDS} s "
        f"(z {worst['z']:.3f}, gravity_z {worst['gravity_z']:+.3f}). For a "
        "legged body that is GOOD NEWS — and it means the settle test above "
        "is passing for a reason its docstring no longer describes: "
        "re-measure and rewrite both. For a wheeled body it is expected: "
        "guard this assertion on the body's kind.")


# --------------------------------------------------- 6. two bodies, one model

def _floor_world(robot_id: str) -> mujoco.MjSpec:
    """An empty floor world, with the option block `compose()` uses."""
    world = mujoco.MjSpec()
    world.modelname = f"conformance:{robot_id}"
    world.option.timestep = C.PHYSICS_DT
    if robot_id == "g1":
        # compose() does this before attaching: the G1's XML wants
        # implicitfast / 10 / 20 and `attach` keeps the PARENT's options.
        world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        world.option.iterations = 10
        world.option.ls_iterations = 20
    w = world.worldbody
    w.add_light(pos=[0, 0, 3.5], dir=[0, 0, -1])
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
               size=[10, 10, 0.05], group=0)
    return world


def _attachable(robot_id: str) -> mujoco.MjSpec:
    """One body as an `MjSpec` ready to attach — `compose()`'s own sources."""
    if robot_id == "g1":
        from microduck_local.robots.g1 import g1_spec
        return g1_spec()
    return mujoco.MjSpec.from_file(str(ROBOT_XML["walk"]))


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_body_attaches_under_a_prefix_beside_a_duck_in_one_model(robot):
    """The lab and /sim put N robots in ONE compiled model.

    Everything per-robot then resolves by `"<id>/" + name` — joints,
    actuators, sensors, geoms — so a body that cannot be attached under a
    prefix (or whose meshes, materials or keyframes collide with a
    neighbour's) cannot be in a room, a roster or a match, whatever its
    training env does. This is `world/compose.py`'s pattern on the smallest
    possible world: this body as "a/", a duck as "b/".
    """
    world = _floor_world(robot)
    w = world.worldbody
    world.attach(_attachable(robot), prefix="a/",
                 frame=w.add_frame(pos=[0, 1.0, 0.0]))
    world.attach(_attachable("microduck"), prefix="b/",
                 frame=w.add_frame(pos=[0, -1.0, 0.0]))
    model = world.compile()

    for rid, prefix in ((robot, "a/"), ("microduck", "b/")):
        for j in S.get(rid).joint_names:
            assert mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, prefix + j) >= 0, (
                f"{prefix}{j} did not survive the attach")
    # The complement: the bare names must NOT resolve, or "resolved by
    # prefix" is an illusion and two bodies are reading each other's joints.
    for j in S.get(robot).joint_names:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) < 0

    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()


@pytest.mark.parametrize("robot", ROBOTS)
def test_two_bodies_in_one_model_need_the_prefix(robot):
    """The planted regression: attach the same body twice under no prefix and
    MuJoCo refuses the repeated name. If this ever stops raising, the test
    above is not proving that the prefix did the work."""
    world = _floor_world(robot)
    w = world.worldbody
    world.attach(_attachable(robot), prefix="", frame=w.add_frame(pos=[0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="repeated name"):
        world.attach(_attachable(robot), prefix="",
                     frame=w.add_frame(pos=[0, -1.0, 0.0]))
        world.compile()


# ------------------------------------------------------------- 7. the lab slot

def _widest_horizontal_extent_m(robot_id: str) -> float:
    """Widest HORIZONTAL extent of the whole body at its STAND keyframe, m.

    The geom AABBs in world axes — what must not overlap on the stage.
    `tests/test_lab_robots.py` measures the same thing by hand for the duck
    and the G1; here it is re-derived for whatever is in the registry.
    """
    spec = S.get(robot_id)
    model = _model(robot_id)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(spec.stand_keyframe).id)
    mujoco.mj_forward(model, data)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:            # world geoms (the floor plane)
            continue
        rot = data.geom_xmat[g].reshape(3, 3)
        centre = data.geom_xpos[g] + rot @ model.geom_aabb[g, :3]
        ext = np.abs(rot) @ model.geom_aabb[g, 3:]
        lo = np.minimum(lo, centre - ext)
        hi = np.maximum(hi, centre + ext)
    return float(max(hi[0] - lo[0], hi[1] - lo[1]))


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_lab_pitch_is_the_ducks_ratio_on_this_bodys_measured_width(robot):
    """`lab_spacing_m` is a number typed into a spec; this re-measures it.

    One duck-sized constant put six 1.3 m G1 helpers inside each other on
    the stage, so the pitch is per body now — and a per-body number written
    by hand drifts from the model it describes the moment the MJCF changes
    shape. 5 % is the band: the duck is 3.522x its 0.1845 m and the G1
    3.522x its 0.5338 m, both within 0.1 % of the ratio.
    """
    spec = S.get(robot)
    width = _widest_horizontal_extent_m(robot)
    assert width > 0
    assert spec.lab_spacing_m == pytest.approx(LAB_PITCH_RATIO * width, rel=0.05), (
        f"{robot} measures {width:.4f} m across, so its stage pitch should be "
        f"~{LAB_PITCH_RATIO * width:.3f} m; the spec says {spec.lab_spacing_m}")


@pytest.mark.parametrize("robot", ROBOTS)
def test_a_pitch_that_drifted_from_the_measured_width_is_caught(robot):
    """The planted regression: the same check on a spec whose pitch is 1.5x
    what its body measures (the G1's real bug was the duck's 0.65 m on a
    0.53 m body — a 3.5x error; 1.5x is the smallest thing worth catching)."""
    spec = _resolved(S.get(robot))
    bad = dataclasses.replace(spec, lab_spacing_m=spec.lab_spacing_m * 1.5)
    width = _widest_horizontal_extent_m(robot)
    assert bad.lab_spacing_m != pytest.approx(LAB_PITCH_RATIO * width, rel=0.05)


# --------------------------------------------------------------- 8. the export

def _tiny_run(robot_id: str, out: Path, robot_in_json: str | None = None) -> Path:
    """The smallest run dir `export_onnx.export` will read: a PPO checkpoint
    on this body's real env, its VecNormalize stats, and a run.json naming
    the body. No training — the weights are the initialisation, which is all
    a shape round-trip needs.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    out.mkdir(parents=True, exist_ok=True)
    venv = VecNormalize(DummyVecEnv([lambda: _env(robot_id)]),
                        norm_obs=True, norm_reward=False)
    venv.reset()
    model = PPO("MlpPolicy", venv, n_steps=8, batch_size=8, n_epochs=1,
                policy_kwargs=dict(net_arch=[8]), device="cpu", seed=0)
    model.save(str(out / "model"))
    venv.save(str(out / "vecnormalize.pkl"))
    (out / "run.json").write_text(json.dumps(
        {"run_name": out.name, "robot": robot_in_json or robot_id}))
    venv.close()
    return out


@pytest.mark.parametrize("robot", ROBOTS)
def test_a_random_weight_policy_round_trips_through_the_exporter(robot, tmp_path):
    """Export is the only way a policy leaves this harness, for every body.

    `export-walk` bakes the observation normaliser into the graph (an
    un-baked checkpoint sees unnormalised observations at deployment and
    silently misbehaves) and shapes the graph from the run's own robot. So
    for a body to be supported, a checkpoint trained on its env has to come
    out the far side as `obs[1, obs_dim] -> actions[1, num_actions]` and
    agree with the torch policy it came from. Random weights are the point:
    the arithmetic is what is under test, not the behaviour.
    """
    import pickle

    import torch
    from stable_baselines3 import PPO

    from microduck_local.export_onnx import OnnxWalkPolicy, export

    spec = S.get(robot)
    run = _tiny_run(robot, tmp_path / f"{robot}-conformance")
    onnx_path = export(run, run / "policy.onnx")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp, outp = sess.get_inputs()[0], sess.get_outputs()[0]
    assert inp.name == "obs" and outp.name == "actions", (
        "the runtime loads these graphs by input/output NAME")
    assert [int(d) for d in inp.shape] == [1, spec.obs_dim]
    assert [int(d) for d in outp.shape] == [1, spec.num_actions]

    loaded = PPO.load(str(run / "model"), device="cpu")
    with open(run / "vecnormalize.pkl", "rb") as f:
        vn = pickle.load(f)
    wrapper = OnnxWalkPolicy(loaded.policy, vn.obs_rms.mean, vn.obs_rms.var,
                             vn.clip_obs).eval()
    rng = np.random.default_rng(0)
    for _ in range(3):
        obs = rng.normal(0, 1, (1, spec.obs_dim)).astype(np.float32)
        with torch.no_grad():
            want = wrapper(torch.tensor(obs)).numpy()
        got = sess.run(["actions"], {"obs": obs})[0]
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_exporter_refuses_a_run_that_names_another_body(robot, tmp_path):
    """The planted regression for the round trip: a run.json that names the
    wrong body must be refused, not exported at the wrong width. Handing
    someone a policy whose graph is 61-in when the weights are 99-in is a
    file that loads and does nothing sane — the lab's width guard is the
    only thing between that file and a duck."""
    from microduck_local.export_onnx import export

    others = [r for r in S.registry() if r != robot]
    if not others:
        pytest.skip("only one body in the registry")
    run = _tiny_run(robot, tmp_path / "mislabelled", robot_in_json=others[0])
    with pytest.raises(ValueError, match="but the checkpoint is"):
        export(run, run / "policy.onnx")


# ---------------------------------------------------------- 9. the viewer sees

def _visual_scene(robot_id: str) -> tuple[dict, int]:
    """The body's viewer scene, and the `nmesh` of the model it came from.

    `GET /scene?robot=<id>` is this same if-chain in `viz_server`, and
    `Body.visual_scene()` is where both are going (MARS plan §1).
    """
    if robot_id == "microduck":
        from microduck_local.viz_server import extract_scene
        from microduck_local.world.compose import scene_model
        return extract_scene(), int(scene_model().nmesh)
    if robot_id == "g1":
        from microduck_local.robots.g1 import g1_xml, visual_scene
        return visual_scene(), int(mujoco.MjModel.from_xml_path(str(g1_xml())).nmesh)
    raise AssertionError(f"no visual scene for {robot_id!r} — see _visual_scene")


def _check_visual_scene(scene: dict, nmesh: int) -> list[str]:
    """Everything the viewer needs from a `{bodies, meshes, geoms}` dump."""
    bad: list[str] = []
    bodies, meshes, geoms = scene["bodies"], scene["meshes"], scene["geoms"]
    if not bodies:
        bad.append("no bodies")
    if len(meshes) != nmesh:
        bad.append(f"{len(meshes)} meshes for a model with nmesh {nmesh}")
    if not geoms:
        bad.append("no geoms")
    for i, g in enumerate(geoms):
        if not (0 <= g["body"] < len(bodies)):
            bad.append(f"geom {i} names body {g['body']} of {len(bodies)}")
        if not (0 <= g["mesh"] < len(meshes)):
            bad.append(f"geom {i} names mesh {g['mesh']} of {len(meshes)}")
        # A material kind per geom: a named material, or an rgba to fall back
        # on. Neither means the viewer paints it black.
        if not g.get("mat") and len(g.get("rgba") or ()) != 4:
            bad.append(f"geom {i} has neither a material nor an rgba")
    for i, m in enumerate(meshes):
        if not m["v"] or not m["f"]:
            bad.append(f"mesh {i} is empty")
        if len(m["v"]) % 3 or len(m["f"]) % 3:
            bad.append(f"mesh {i} is not a flat triple array")
    return bad


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_visual_scene_serves_every_mesh_the_model_has(robot):
    """A body nobody can SEE is not on the stage.

    The viewer draws from this dump alone and indexes it positionally, so a
    geom pointing past the end of the body list, a mesh that came out empty
    or a count that disagrees with the model's `nmesh` is a robot that
    renders wrong or not at all — and "blank stage" has already cost a
    debugging session that went looking in the viewer.
    """
    scene, nmesh = _visual_scene(robot)
    assert _check_visual_scene(scene, nmesh) == []
    assert len(scene["bodies"]) > 0 and len(scene["meshes"]) == nmesh


@pytest.mark.parametrize("robot", ROBOTS)
def test_a_doctored_visual_scene_is_caught(robot):
    """The planted regression: drop a mesh, point a geom past the body list,
    empty a mesh, strip a geom's material AND its rgba — each must be
    reported by the same checker the test above trusts."""
    scene, nmesh = _visual_scene(robot)
    plants = {
        "meshes": {**scene, "meshes": scene["meshes"][:-1]},
        "body": {**scene, "geoms": [{**scene["geoms"][0], "body": len(scene["bodies"])}]
                 + scene["geoms"][1:]},
        "mesh": {**scene, "geoms": [{**scene["geoms"][0], "mesh": len(scene["meshes"])}]
                 + scene["geoms"][1:]},
        "empty": {**scene, "meshes": [{"v": [], "f": []}] + scene["meshes"][1:]},
        "unpainted": {**scene,
                      "geoms": [{**scene["geoms"][0], "mat": "", "rgba": []}]
                      + scene["geoms"][1:]},
    }
    for kind, doctored in plants.items():
        assert _check_visual_scene(doctored, nmesh), f"{kind} was not reported"
