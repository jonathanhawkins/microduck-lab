"""Self-describing policies: the contract id that replaces an obs width.

`docs/mars-roadmap.md` §6.3. Two claims are under test, and they need
different kinds of evidence:

  * **the record is arithmetic** — a contract's slot table partitions its
    body's `obs_dim` exactly, its dims are the body's own, and it survives
    JSON and an ONNX `metadata_props` round trip. Cheap, and checked for
    every registered body so a fourth one cannot be listed without it.
  * **the record is TRUE of the env** — the duck's `joint_pos_rel[6:20]` has
    to be where the joint positions actually are. That cannot be checked
    against a constant (the constant is what would be wrong); it is checked
    against a LIVE env, by taking an observation and reading each slot back
    against the state the env holds. A layout that moves without its id
    moving fails here, which is the whole reason the id exists.

Every positive has a planted negative in this file or in the plant table in
the phase report: a swapped-but-still-tiling slot table must fail the live
check, an unstamped ONNX must fall through the precedence, and an export
whose run recorded another body's contract must be refused.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from microduck_local import contract as C
from microduck_local import run_record as RR
from microduck_local.robots import registry as R
from microduck_local.robots.g1 import g1_ready
from microduck_local.robots.policy_contract import (
    KEY_ID,
    KEY_JSON,
    PolicyContract,
    Slot,
    from_run_json,
    recorded,
    resolve,
)

needs_g1 = pytest.mark.skipif(
    not g1_ready(), reason="G1 assets missing — uv run fetch-robot g1")

BODIES = sorted(R.registry())


# --------------------------------------------------------------- the arithmetic

@pytest.mark.parametrize("robot", BODIES)
def test_every_bodys_contract_tiles_its_own_observation(robot):
    """The table describes every float of `obs_dim`, once.

    A gap is an undeclared float and an overlap is two names for one float;
    either makes the table a decoration rather than a contract (see
    `PolicyContract.tiles`). MARS's four reserved zeros are a NAMED slot for
    exactly this reason.
    """
    body = R.get(robot)
    c = body.contract()
    assert c.tiles() == (), c.tiles()
    assert sum(s.width for s in c.slots) == c.obs_dim
    assert c.obs_dim == body.obs_dim
    assert c.act_dim == body.num_actions
    assert c.robot == body.id
    assert c.rate_hz > 0
    # The honesty sentence is not optional: it is the thing that gets lost
    # when a .onnx is passed along (robots/policy_contract.PolicyContract).
    assert c.deploy.strip()


def test_the_three_bodies_ids_are_distinct_and_name_their_width():
    """Two bodies sharing an id would cross exactly as two sharing a width
    do — the failure this whole record exists to remove."""
    ids = {b: R.get(b).contract().id for b in BODIES}
    assert len(set(ids.values())) == len(ids), ids
    for body_id, contract_id in ids.items():
        assert str(R.get(body_id).obs_dim) in contract_id, (body_id, contract_id)


@pytest.mark.parametrize("robot", BODIES)
def test_a_contract_survives_the_json_round_trip_that_run_json_is(robot):
    """`run.json` stores `as_dict()`; `resolve()` reads it back with
    `from_dict`. Slots arrive as lists of lists (JSON has no tuples), so the
    equality is also a test of the normalisation in `__post_init__`."""
    c = R.get(robot).contract()
    back = PolicyContract.from_dict(json.loads(json.dumps(c.as_dict())))
    assert back == c
    assert back.slots == c.slots and isinstance(back.slots[0], Slot)


def test_matches_is_the_id_and_takes_a_contract_or_a_bare_string():
    duck = R.get("microduck").contract()
    assert duck.matches(duck) and duck.matches(duck.id)
    assert duck.matches(dataclasses.replace(duck, deploy="anything else"))
    assert not duck.matches(None)
    assert not duck.matches("microduck-61-v2")
    for other in (b for b in BODIES if b != "microduck"):
        assert not duck.matches(R.get(other).contract())


def test_describe_says_the_id_the_dims_the_rate_and_the_caveat():
    """`export-walk` prints this line: it is the last moment the harness can
    attach the deployment caveat to the file it just wrote."""
    line = R.get("microduck").contract().describe()
    for part in ("microduck-61-v1", "obs[1,61]", "actions[1,14]", "50 Hz",
                 "bakes the normaliser"):
        assert part in line, line


# ------------------------------------------- the constructor's planted negatives

def _slots(*triples) -> tuple[Slot, ...]:
    return tuple(Slot(*t) for t in triples)


@pytest.mark.parametrize("slots,why", [
    (_slots(("a", 0, 3), ("b", 2, 6)), "overlaps"),
    (_slots(("a", 0, 3), ("b", 4, 6)), "nothing describes"),
    (_slots(("a", 0, 3)), "obs_dim is 6"),
    (_slots(("a", 0, 3), ("b", 3, 6), ("c", 6, 9)), "obs_dim is 6"),
    (_slots(("a", 0, 3), ("a", 3, 6)), "repeated slot name"),
    (_slots(("a", 0, 3), ("b", 5, 5), ("c", 3, 6)), "empty or inverted"),
    ((), "no slots declared"),
])
def test_a_table_that_does_not_tile_is_refused_at_construction(slots, why):
    """Construction, not a later check: a contract is built the first time
    anything asks a body for one, so a bad table is an error at the first
    `fetch-robot` listing rather than a mislabelled file found next month."""
    with pytest.raises(ValueError, match=why):
        PolicyContract(id="x-6-v1", robot="x", obs_dim=6, act_dim=2,
                       rate_hz=50.0, slots=slots)


@pytest.mark.parametrize("kw,why", [
    ({"obs_dim": 0}, "must be positive"),
    ({"act_dim": 0}, "must be positive"),
    ({"rate_hz": 0.0}, "must be positive"),
    ({"id": ""}, "needs an id"),
])
def test_the_other_construction_refusals(kw, why):
    args = dict(id="x-6-v1", robot="x", obs_dim=6, act_dim=2, rate_hz=50.0,
                slots=_slots(("a", 0, 6)))
    args.update(kw)
    with pytest.raises(ValueError, match=why):
        PolicyContract(**args)


def test_a_body_that_declares_no_contract_says_so_by_name():
    """`BodyBase.contract()` raises rather than synthesising one.

    An id and a width could be spelled from the body; the control rate and
    the deploy sentence cannot, and a guessed contract over a wrong rate is
    the silent cross this module exists to stop.
    """
    from microduck_local.robots.body import BodyBase

    @dataclasses.dataclass(frozen=True, eq=False, kw_only=True)
    class Mute(BodyBase):
        pass

    body = Mute(id="mute", joint_names=("a",), default_pose=np.zeros(1),
                obs_dim=4)
    with pytest.raises(NotImplementedError, match="does not declare its policy"):
        body.contract()


# ------------------------------------ the duck's table, against a LIVE env

DUCK_TABLE = (
    Slot("base_ang_vel", 0, 3),
    Slot("projected_gravity", 3, 6),
    Slot("joint_pos_rel", 6, 20),
    Slot("joint_vel", 20, 34),
    Slot("last_action", 34, 48),
    Slot("twist_cmd", 48, 51),
    Slot("head_pose_cmd", 51, 55),
    Slot("body_pose_cmd", 55, 61),
)


def test_the_ducks_contract_id_and_table_are_the_deployment_contract():
    """Pinned literally, because this one is not ours to change: the 61
    floats in this order are what the robot's own software hot-swaps behind
    and what every shipped alpha policy speaks. A diff here is a diff with
    Pollen's runtime, and the id has to move with it."""
    c = R.get("microduck").contract()
    assert c.id == "microduck-61-v1"
    assert c.slots == DUCK_TABLE
    assert (c.obs_dim, c.act_dim) == (C.OBS_DIM, C.NUM_JOINTS)
    assert c.rate_hz == pytest.approx(1.0 / C.CTRL_DT)


def _reads_back(obs: np.ndarray, c: PolicyContract, name: str,
                want: np.ndarray) -> bool:
    s = c.slot(name)
    return (s.width == len(np.atleast_1d(want))
            and np.allclose(obs[s.start:s.stop], want, atol=0, rtol=0))


@pytest.fixture(scope="module")
def duck_env():
    from microduck_local.walk_env import MicroduckWalkEnv
    env = MicroduckWalkEnv(seed=0, obs_noise=False, domain_rand=False,
                           action_delay=False, random_yaw=False)
    try:
        yield env
    finally:
        env.close()


def test_every_duck_slot_reads_back_the_state_the_env_holds(duck_env):
    """The table against the env that FILLS it, not against a constant.

    `walk_env._get_obs` writes eight slices into one 61-float array. This
    takes a real observation and reads each declared slot back against the
    state the env is holding at that instant — so a slice that moved, or a
    table that was copied from the docstring instead of the code, fails
    here. All eight, because the cost of one more is a line.

    `joint_vel` is checked across two steps on purpose: the duck's obs
    carries a ONE-STEP-LAGGED joint velocity (a Dynamixel present_velocity
    artefact upstream models too), so the slot in step N+1 holds the
    velocity measured at the end of step N. A test that compared it to the
    current velocity would fail against correct code, and a table that
    listed it as current would pass against a wrong one.
    """
    env = duck_env
    c = R.get("microduck").contract()
    obs, _ = env.reset(seed=3)

    # Fresh out of reset: the command slots and the (zeroed) last action.
    assert _reads_back(obs, c, "base_ang_vel", np.asarray(env._gyro, np.float32))
    assert _reads_back(obs, c, "projected_gravity", env._projected_gravity())
    assert _reads_back(obs, c, "joint_pos_rel", env._joint_pos_rel())
    assert _reads_back(obs, c, "twist_cmd", env.twist_cmd)
    assert _reads_back(obs, c, "head_pose_cmd", env.head_cmd)
    assert _reads_back(obs, c, "body_pose_cmd", env.body_cmd)
    assert _reads_back(obs, c, "last_action", np.zeros(C.NUM_JOINTS, np.float32))

    # last_action is the RAW policy output, so an action with a distinct
    # value per joint pins the ORDER of the slot as well as its bounds.
    act = np.linspace(-0.3, 0.3, C.NUM_JOINTS).astype(np.float32)
    obs1, *_ = env.step(act)
    assert _reads_back(obs1, c, "last_action", act)
    vel_after_first = np.asarray(env.prev_joint_vel, np.float32).copy()
    obs2, *_ = env.step(np.zeros(C.NUM_JOINTS, np.float32))
    assert _reads_back(obs2, c, "joint_vel", vel_after_first)
    assert _reads_back(obs2, c, "last_action", np.zeros(C.NUM_JOINTS, np.float32))


def test_a_swapped_but_still_tiling_table_fails_the_live_check(duck_env):
    """The planted negative for the test above, and the reason it exists.

    `joint_pos_rel` and `joint_vel` are both 14 wide, so exchanging their
    offsets leaves a table that still tiles 61 floats perfectly and still
    passes every arithmetic case in this file. Only the live env can tell
    the two apart — which is what makes reading the slots back the load-
    bearing test and the arithmetic a sanity check.
    """
    env = duck_env
    c = R.get("microduck").contract()
    swapped = dataclasses.replace(c, slots=tuple(
        Slot("joint_pos_rel", 20, 34) if s.name == "joint_pos_rel" else
        Slot("joint_vel", 6, 20) if s.name == "joint_vel" else s
        for s in c.slots))
    assert swapped.tiles() == ()                    # still a legal table
    obs, _ = env.reset(seed=3)
    assert not _reads_back(obs, swapped, "joint_pos_rel", env._joint_pos_rel())


def test_a_slot_shifted_by_one_float_cannot_even_be_constructed():
    """The other plant, and it never reaches the live check.

    Sliding `joint_pos_rel` to `[5:19]` overlaps `projected_gravity` and
    leaves `[19:20]` undescribed, so `tiles()` refuses it — which is the
    division of labour worth having: an off-by-one is arithmetic and dies at
    construction, and only a table that is internally perfect (the swap
    above) needs an env to catch.
    """
    c = R.get("microduck").contract()
    with pytest.raises(ValueError, match="overlaps|nothing describes"):
        dataclasses.replace(c, slots=tuple(
            Slot(s.name, s.start - 1, s.stop - 1)
            if s.name == "joint_pos_rel" else s for s in c.slots))


def test_the_ducks_slot_names_can_be_looked_up_and_a_typo_cannot():
    c = R.get("microduck").contract()
    assert c.slot("twist_cmd") == Slot("twist_cmd", 48, 51)
    with pytest.raises(KeyError, match="has no slot 'twist'"):
        c.slot("twist")


# ------------------------------------- the G1's table, against a LIVE env

@needs_g1
def test_two_g1_slots_read_back_the_state_its_env_holds():
    """The G1's 99-d layout is LuckyRobots', kept verbatim so their
    `walker.onnx` stays a golden test and a teacher. Its first three floats
    are base LINEAR velocity — the thing no humanoid observes without state
    estimation, which is why this body's `deploy` says "lab contract".
    """
    MOVED = 40           # control steps; see below

    from microduck_local.robots.g1_env import G1WalkEnv

    c = R.get("g1").contract()
    assert c.id == "g1-lucky-99-v1"
    env = G1WalkEnv(seed=0, obs_noise=False, domain_rand=False,
                    action_delay=False, random_yaw=False)
    try:
        env.reset(seed=1)
        # MOVE IT FIRST. At the STAND keyframe the base is at rest, so
        # `base_lin_vel` and `base_ang_vel` are BOTH three zeros and reading
        # one back where the other lives passes — the plant table caught this
        # test doing exactly that ("perturb what the dynamics can feel").
        # Forty steps of a small random action gives every float its own
        # value, and the assertions below then pin the ORDER.
        rng = np.random.default_rng(0)
        obs = None
        for _ in range(MOVED):
            obs, *_ = env.step(rng.normal(0, 0.2, env.action_space.shape)
                               .astype(np.float32))
        lin, gyro = env.body_lin_vel(), np.asarray(env._gyro, np.float32)
        assert np.abs(lin).max() > 1e-3 and np.abs(gyro).max() > 1e-3
        assert not np.allclose(lin, gyro), "the two 3-slots must differ here"
        assert _reads_back(obs, c, "base_lin_vel", lin)
        assert _reads_back(obs, c, "base_ang_vel", gyro)
        assert _reads_back(obs, c, "joint_pos_rel", env._joint_pos_rel())
        # …and one more, free: the command block at the far end of the vector.
        assert _reads_back(obs, c, "twist_cmd", env.twist_cmd)
        assert c.slot("twist_cmd").stop == c.obs_dim
    finally:
        env.close()


@needs_g1
def test_the_g1_table_is_not_accidentally_the_ducks():
    """Both bodies name six of their seven/eight slots the same; the offsets
    are what differ. A copied table would tile 99 and read back nothing."""
    duck, g1 = R.get("microduck").contract(), R.get("g1").contract()
    assert duck.slot("joint_pos_rel").start != g1.slot("joint_pos_rel").start
    assert "base_lin_vel" not in [s.name for s in duck.slots]


# ---------------------------------------------------- the ONNX metadata

def _stub_onnx(path: Path, obs_dim: int, act_dim: int) -> Path:
    """A minimal, RUNNABLE `obs[1,n] -> actions[1,m]` graph.

    Real enough for onnxruntime to load and score, and a hundred times
    cheaper than a PPO checkpoint — which matters because the precedence
    cases below need several policy files and none of them care what the
    weights are. The export cases use a real one.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array(
        np.full((obs_dim, act_dim), 0.01, np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "w"], ["actions"])], "stub",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, act_dim])],
        [w])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10          # the IR this repo's onnxruntime reads
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def _run(path: Path, obs: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: obs})[0]


def test_a_stamp_round_trips_and_does_not_touch_the_graph(tmp_path):
    """Metadata must be free. `export()` stamps BEFORE its own onnxruntime
    cross-check so every export proves this for itself; here it is proved
    directly, bit for bit, on the same inputs either side of the stamp."""
    c = R.get("microduck").contract()
    p = _stub_onnx(tmp_path / "stub.onnx", c.obs_dim, c.act_dim)
    obs = np.random.default_rng(0).normal(0, 1, (1, c.obs_dim)).astype(np.float32)
    before = _run(p, obs)
    assert PolicyContract.from_onnx(p) is None      # the planted absence
    c.write_onnx_metadata(p)
    after = _run(p, obs)
    np.testing.assert_array_equal(before, after)
    assert PolicyContract.from_onnx(p) == c


def test_re_stamping_replaces_rather_than_appends(tmp_path):
    """A re-export must not leave two contracts in one file for a reader to
    choose between (`write_onnx_metadata`)."""
    import onnx

    duck = R.get("microduck").contract()
    p = _stub_onnx(tmp_path / "stub.onnx", duck.obs_dim, duck.act_dim)
    duck.write_onnx_metadata(p)
    duck.write_onnx_metadata(p)
    props = [q.key for q in onnx.load(str(p)).metadata_props]
    assert props.count(KEY_JSON) == 1 and props.count(KEY_ID) == 1
    assert PolicyContract.from_onnx(p) == duck


def test_the_bare_id_is_in_the_file_for_a_reader_that_parses_nothing(tmp_path):
    import onnx

    c = R.get("mars").contract() if "mars" in BODIES else R.get("microduck").contract()
    p = _stub_onnx(tmp_path / "stub.onnx", c.obs_dim, c.act_dim)
    c.write_onnx_metadata(p)
    props = {q.key: q.value for q in onnx.load(str(p)).metadata_props}
    assert props[KEY_ID] == c.id
    assert json.loads(props[KEY_JSON])["id"] == c.id


def test_a_corrupt_metadata_block_reads_as_absent_not_as_an_exception(tmp_path):
    """A label must never raise where it is read — in the lab that is inside
    the 50 Hz loop, and a stopped loop blanks the viewer for every duck. The
    rung below answers instead (`from_onnx`)."""
    import onnx

    duck = R.get("microduck").contract()
    p = _stub_onnx(tmp_path / "stub.onnx", duck.obs_dim, duck.act_dim)
    duck.write_onnx_metadata(p)
    model = onnx.load(str(p))
    for q in model.metadata_props:
        if q.key == KEY_JSON:
            q.value = "{ not json"
    onnx.save(model, str(p))
    assert PolicyContract.from_onnx(p) is None
    assert resolve(p).matches(duck)                 # the duck fallback answers


def test_a_file_that_is_not_an_onnx_at_all_reads_as_absent(tmp_path):
    p = tmp_path / "policy.onnx"
    p.write_bytes(b"not a real onnx, but the palette lists it")
    assert PolicyContract.from_onnx(p) is None


# ------------------------------------------------ the precedence, rung by rung

def _run_json(d: Path, **meta) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps(meta))
    return d


def test_rung_1_the_onnx_metadata_beats_the_run_json_beside_it(tmp_path):
    """The only rung that survives the file being MOVED, so it wins: a
    `policy.onnx` dropped into another run's directory must not inherit that
    run's identity.

    The competing rung has to be a RECORDED contract, not a robot name: with
    only a name beside it there is nothing for rung 2 to answer with, and
    reversing the order would still give the same result — which is what the
    plant table caught this test doing.
    """
    others = [b for b in BODIES if b != "microduck"]
    if not others:
        pytest.skip("only one body in the registry")
    want = R.get(others[0]).contract()
    duck = R.get("microduck").contract()
    d = _run_json(tmp_path / "run", robot="microduck", contract=duck.as_dict())
    p = _stub_onnx(d / "policy.onnx", want.obs_dim, want.act_dim)
    want.write_onnx_metadata(p)
    assert resolve(p) == want, "the stamp must beat the run.json's contract"
    assert recorded(p) == want
    # …and it beats a bare robot name too (the rung-3 case).
    d2 = _run_json(tmp_path / "run2", robot="microduck")
    p2 = _stub_onnx(d2 / "policy.onnx", want.obs_dim, want.act_dim)
    want.write_onnx_metadata(p2)
    assert resolve(p2) == want


def test_rung_2_the_run_jsons_own_contract_beats_the_robot_name(tmp_path):
    """The rung that covers a checkpoint ONNX (`select-run`) and any file
    exported before the stamp existed, beside a run that knows itself."""
    other = [b for b in BODIES if b != "microduck"][0]
    want = R.get(other).contract()
    d = _run_json(tmp_path / "run", robot="microduck",
                  contract=want.as_dict())
    p = _stub_onnx(d / "policy.onnx", want.obs_dim, want.act_dim)
    assert resolve(p) == want
    assert from_run_json(d) == want


def test_rung_3_an_old_run_is_still_read_by_its_robot_name(tmp_path):
    """Every run trained before this module recorded only a name. Mapping it
    through the registry is exactly what `run_robot` + `spec.get` did."""
    other = [b for b in BODIES if b != "microduck"][0]
    d = _run_json(tmp_path / "run", robot=other)
    p = _stub_onnx(d / "policy.onnx", R.get(other).obs_dim,
                   R.get(other).num_actions)
    assert resolve(p) == R.get(other).contract()
    assert recorded(p) is None            # nothing SELF-describing here
    from microduck_local.export_onnx import run_robot
    assert run_robot(d) == other


def test_rung_4_a_policy_with_nothing_to_say_is_a_duck(tmp_path):
    """`run_robot` has answered "microduck" for a missing or unreadable
    run.json since it was written, because when those files were made the
    duck was the only body there was. Trick runs (behavior.json, no
    run.json) are the live population that depends on it."""
    duck = R.get("microduck").contract()
    d = tmp_path / "trick-run"
    p = _stub_onnx(d / "policy.onnx", duck.obs_dim, duck.act_dim)
    assert resolve(p) == duck
    assert resolve(d) == duck
    (d / "run.json").write_text("{ not json")
    assert resolve(p) == duck             # unreadable is the same as absent


def test_an_unknown_robot_name_is_an_error_and_not_a_duck(tmp_path):
    """The one place `resolve` raises. Quietly calling an unrecognised body a
    duck is the silent cross this file exists to prevent; `registry.get`'s
    message names the ids and the fetch command."""
    d = _run_json(tmp_path / "run", robot="nosuchbot")
    with pytest.raises(KeyError, match="unknown robot"):
        resolve(d)


def test_a_directory_is_asked_run_json_first(tmp_path):
    """Deliberate, and the one place the order differs. Naming a DIRECTORY
    asks what the run drives, and the run's own record answers that directly
    rather than by inference from one artefact inside it — and the lab
    resolves every run in the palette on a timer, where a few hundred bytes
    of JSON beats parsing a policy's protobuf per run per poll."""
    other = [b for b in BODIES if b != "microduck"][0]
    duck = R.get("microduck").contract()
    d = _run_json(tmp_path / "run", robot="microduck", contract=duck.as_dict())
    stray = R.get(other).contract()
    p = _stub_onnx(d / "policy.onnx", stray.obs_dim, stray.act_dim)
    stray.write_onnx_metadata(p)
    assert resolve(d) == duck             # the directory: its own record
    assert resolve(p) == stray            # the file: itself


# ------------------------------------------------------------- the exporter

@pytest.fixture(scope="module")
def tiny_duck_run(tmp_path_factory):
    """The smallest run `export_onnx.export` will read: a PPO checkpoint on
    the duck's real env plus its VecNormalize stats. No learning — the
    weights are the initialisation, which is all a stamp needs."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from microduck_local.walk_env import MicroduckWalkEnv

    out = tmp_path_factory.mktemp("contract-run")
    venv = VecNormalize(DummyVecEnv([lambda: MicroduckWalkEnv(seed=0)]),
                        norm_obs=True, norm_reward=False)
    venv.reset()
    model = PPO("MlpPolicy", venv, n_steps=8, batch_size=8, n_epochs=1,
                policy_kwargs=dict(net_arch=[8]), device="cpu", seed=0)
    model.save(str(out / "model"))
    venv.save(str(out / "vecnormalize.pkl"))
    venv.close()
    return out


def test_export_stamps_the_contract_and_it_reads_back_equal(tiny_duck_run,
                                                            tmp_path):
    """The producer side of §6.3, end to end: `export-walk` is the only way a
    policy leaves this harness, so it is where the file learns what it is."""
    from microduck_local.export_onnx import export

    (tiny_duck_run / "run.json").write_text(json.dumps({"robot": "microduck"}))
    out = export(tiny_duck_run, tmp_path / "policy.onnx")
    want = R.get("microduck").contract()
    assert PolicyContract.from_onnx(out) == want
    # …and it travels: no run.json anywhere near this copy.
    moved = tmp_path / "elsewhere" / "downloaded.onnx"
    moved.parent.mkdir()
    moved.write_bytes(Path(out).read_bytes())
    assert resolve(moved) == want
    assert recorded(moved) == want


def test_export_refuses_a_run_whose_recorded_contract_is_another_bodys(
        tiny_duck_run, tmp_path):
    """The planted negative for the stamp: `export-walk --robot <other>` on a
    run that RECORDED a contract.

    That override is the one way the two can disagree — with no `--robot`,
    the recorded contract is itself what `resolve()` answers with, so there
    is nothing to contradict. The other way in is a version bump: the day a
    body's id goes `v1` -> `v2`, every old run's recorded `v1` stops matching
    and its re-export is refused until someone decides what that means.

    Refused by ID, and the message carries both — a width would say
    "61 vs 99" and leave the reader to guess which robot each was.
    """
    from microduck_local.export_onnx import export

    other = [b for b in BODIES if b != "microduck"][0]
    duck = R.get("microduck").contract()
    (tiny_duck_run / "run.json").write_text(json.dumps(
        {"robot": "microduck", "contract": duck.as_dict()}))
    try:
        with pytest.raises(ValueError) as e:
            export(tiny_duck_run, tmp_path / "policy.onnx", robot=other)
        msg = str(e.value)
        assert R.get(other).contract().id in msg
        assert duck.id in msg
        # …and the same export without the override is fine: the run agrees
        # with itself, so nothing is refused (the positive control).
        assert export(tiny_duck_run, tmp_path / "ok.onnx").exists()
    finally:
        (tiny_duck_run / "run.json").write_text(json.dumps({"robot": "microduck"}))


def test_export_still_refuses_a_checkpoint_that_disagrees_by_shape(
        tiny_duck_run, tmp_path):
    """The check that existed before contracts did, unchanged: a run.json
    naming a body whose dims are not the checkpoint's. Kept because a
    contract cannot see the weights."""
    from microduck_local.export_onnx import export

    others = [b for b in BODIES if R.get(b).obs_dim != C.OBS_DIM]
    if not others:
        pytest.skip("no second body with a different width")
    (tiny_duck_run / "run.json").write_text(json.dumps({"robot": others[0]}))
    try:
        with pytest.raises(ValueError, match="but the checkpoint is"):
            export(tiny_duck_run, tmp_path / "policy.onnx")
    finally:
        (tiny_duck_run / "run.json").write_text(json.dumps({"robot": "microduck"}))


# ---------------------------------------------------------------- eval-walk

def test_eval_refuses_a_robot_flag_that_contradicts_the_file(tmp_path,
                                                             monkeypatch):
    """`eval-walk policy.onnx --robot g1` on a file that RECORDS the duck.

    The refusal names both contract ids, not two widths: "61 vs 99" does not
    say which robot either belongs to, and the whole point of an id is that
    the answer is legible in the message.
    """
    from microduck_local import eval_onnx

    other = [b for b in BODIES if b != "microduck"][0]
    duck = R.get("microduck").contract()
    p = _stub_onnx(tmp_path / "policy.onnx", duck.obs_dim, duck.act_dim)
    duck.write_onnx_metadata(p)
    monkeypatch.setattr(sys, "argv", ["eval-walk", str(p), "--robot", other])
    with pytest.raises(SystemExit) as e:
        eval_onnx.main()
    msg = str(e.value)
    assert duck.id in msg and R.get(other).contract().id in msg


def test_eval_still_honours_the_flag_for_a_file_that_declares_nothing(
        tmp_path, monkeypatch):
    """The planted negative for the refusal above — and a real case.

    `--robot` exists for an old .onnx moved away from its run directory: it
    records nothing, so the caller is the only one who knows. Refusing there
    would break the flag's only use. The test gets as far as the BEHAVIOR
    check, which is the next line in `main` — proof the contract check let
    it through.
    """
    from microduck_local import eval_onnx

    others = [b for b in BODIES if b != "microduck"]
    if not others:
        pytest.skip("only one body in the registry")
    body = R.get(others[0])
    p = _stub_onnx(tmp_path / "policy.onnx", body.obs_dim, body.num_actions)
    assert recorded(p) is None
    monkeypatch.setattr(sys, "argv",
                        ["eval-walk", str(p), "--robot", others[0],
                         "--behavior", "stand"])
    with pytest.raises(SystemExit, match="reward recipe"):
        eval_onnx.main()


# ------------------------------------------------------------- the records

@pytest.mark.parametrize("robot", BODIES)
def test_the_run_json_a_trainer_writes_is_what_resolve_reads(robot, tmp_path):
    """`train.py` and `distill.py` write `body.contract().as_dict()` under
    `"contract"`; this is that dict going in and the contract coming back."""
    want = R.get(robot).contract()
    d = _run_json(tmp_path / robot, robot=robot, contract=want.as_dict())
    assert resolve(d) == want
    assert from_run_json(d) == want


def test_the_record_a_person_reads_carries_the_id_from_the_run_json():
    """`record.json` gets the ID only: it is the part that is legible in a
    palette tooltip, where a slot table would be noise."""
    want = R.get("microduck").contract()
    rec = RR.default_record("r", {"robot": "microduck", "task": "walk",
                                  "steps": 1000, "contract": want.as_dict()})
    assert rec["contract_id"] == want.id


def test_the_record_falls_back_to_the_registry_for_an_old_run_json():
    """`describe-run --backfill` reads run.json files written before the
    contract key existed."""
    rec = RR.default_record("r", {"robot": "microduck", "task": "walk"})
    assert rec["contract_id"] == "microduck-61-v1"


def test_a_label_never_raises_over_a_body_it_cannot_load(monkeypatch):
    """The planted negative: a record is a LABEL, and a label must not be
    able to take down the run it labels (`run_record`'s first rule). A body
    whose assets are absent raises in `registry.get`; the record reads
    "not recorded", which is what every run before this key says anyway."""
    def boom(_body_id):
        raise KeyError("unknown robot 'ghost' — have []")

    monkeypatch.setattr("microduck_local.robots.registry.get", boom)
    rec = RR.default_record("r", {"robot": "ghost", "task": "walk"})
    assert rec["contract_id"] is None
    assert rec["robot"] == "ghost"
