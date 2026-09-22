"""train-walk / export-walk / render-rollout for a second robot.

These are the CLI seams: which env class a `--robot` builds, which knobs are
refused because they are the duck's, what the exported graph's shape is and
where that shape comes from, and the camera framing that made the first G1
contact sheet twelve tiles of shin.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from microduck_local import export_onnx as E
from microduck_local import render_rollout as R
from microduck_local import train as T
from microduck_local.robots.g1 import g1_ready
from microduck_local.walk_env import MicroduckWalkEnv

needs_g1 = pytest.mark.skipif(not g1_ready(), reason="G1 assets missing — uv run fetch-g1")


# ---------------------------------------------------------------- train-walk

def test_env_class_picks_the_body():
    assert T.env_class("microduck") is MicroduckWalkEnv
    assert T.env_class(None) is MicroduckWalkEnv
    with pytest.raises(SystemExit, match="unknown --robot"):
        T.env_class("wombat")


@needs_g1
def test_env_class_g1_is_the_g1_env():
    from microduck_local.robots.g1_env import G1WalkEnv
    assert T.env_class("g1") is G1WalkEnv


def test_the_duck_default_is_untouched():
    a = T.parse_args([])
    assert a.robot == "microduck"
    kw = T.env_kwargs_from_args(a)
    assert kw["actuator"] == T.DEFAULT_ACTUATOR == "bam"
    assert "actuator_force" not in kw


def test_g1_trains_on_position_servos_not_the_xl330_model():
    a = T.parse_args(["--robot", "g1"])
    kw = T.env_kwargs_from_args(a)
    assert kw["actuator_force"] == "xml"
    assert "actuator" not in kw


def test_bam_is_refused_for_the_g1_at_the_cli():
    a = T.parse_args(["--robot", "g1", "--actuator", "bam"])
    with pytest.raises(SystemExit, match="XL330"):
        T.env_kwargs_from_args(a)


def test_head_range_is_refused_for_a_robot_without_a_head_command():
    a = T.parse_args(["--robot", "g1", "--head-range=0,0,0,0,0,0,0,0"])
    with pytest.raises(SystemExit, match="head-pose command"):
        T.env_kwargs_from_args(a)
    # …and still works for the duck (the gaze poses from the README)
    b = T.parse_args(["--head-range=-0.75,0.05,-0.05,0.8,-1.4,1.4,-0.015,0.015"])
    assert len(T.env_kwargs_from_args(b)["head_cmd_ranges"]) == 4


# --------------------------------------------------------------- export-walk

def test_run_robot_reads_run_json(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    assert E.run_robot(d) == "microduck"          # no run.json at all
    (d / "run.json").write_text(json.dumps({"robot": "g1"}))
    assert E.run_robot(d) == "g1"
    (d / "run.json").write_text("{ not json")
    assert E.run_robot(d) == "microduck"          # unreadable -> the duck


class _FakeRms:
    mean = np.zeros(99, np.float32)
    var = np.ones(99, np.float32)


class _FakeVecNormalize:
    """Module level so pickle can round-trip it (export reads the real file)."""
    obs_rms = _FakeRms()
    clip_obs = 100.0


def test_export_refuses_a_checkpoint_that_disagrees_with_run_json(tmp_path, monkeypatch):
    """run.json says one robot, the weights are another's: exporting the wrong
    shape hands someone a policy that loads and does nothing sane."""
    pytest.importorskip("torch")
    import types

    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"robot": "microduck"}))

    class FakePolicy:
        observation_space = types.SimpleNamespace(shape=(99,))
        action_space = types.SimpleNamespace(shape=(29,))

    fake_model = types.SimpleNamespace(policy=FakePolicy())
    monkeypatch.setattr(E.PPO, "load", staticmethod(lambda *a, **k: fake_model))

    import pickle
    (run / "vecnormalize.pkl").write_bytes(pickle.dumps(_FakeVecNormalize()))
    with pytest.raises(ValueError, match="run.json says robot=microduck"):
        E.export(run, run / "policy.onnx")


# ------------------------------------------------------------ render-rollout

def test_the_camera_frames_the_body_it_is_given():
    duck = R.make_camera("side", R.CAM_DISTANCE, stand_z=R.DUCK_STAND_Z)
    assert duck.distance == pytest.approx(R.CAM_DISTANCE)   # unchanged for the duck
    assert float(duck.lookat[2]) == pytest.approx(R.DUCK_STAND_Z * 0.5)
    g1 = R.make_camera("side", R.CAM_DISTANCE, stand_z=0.758)
    assert g1.distance > duck.distance * 2
    assert float(g1.lookat[2]) > float(duck.lookat[2])
    # no stand height -> exactly the old behaviour
    plain = R.make_camera("side", R.CAM_DISTANCE)
    assert plain.distance == pytest.approx(R.CAM_DISTANCE)
    assert float(plain.lookat[2]) == 0.0


@needs_g1
def test_build_env_renders_the_g1_walking_env_with_the_seconds_flag(monkeypatch):
    # `build_env` publishes its --env overrides into os.environ (the trainer
    # subprocess reads them there), so the knob has to be sandboxed: without
    # monkeypatch this test leaked MICRODUCK_EPISODE_S into every LATER test
    # in the process and failed one of them 300 tests downstream.
    monkeypatch.setenv("MICRODUCK_EPISODE_S", "4")
    env = R.build_env("g1-walk", {"MICRODUCK_EPISODE_S": "4"}, seed=0, robot="g1")
    assert env.robot.id == "g1"
    assert env.max_steps == 200          # 4 s at 50 Hz
    assert env.observation_space.shape == (99,)


# ------------------------------------------------------------- eval-walk

def test_eval_refuses_a_duck_recipe_for_another_body(monkeypatch, tmp_path):
    """`--behavior` names a Microduck reward recipe. Running one against a G1
    would build a BehaviorEnv for duck geometry and fail somewhere deep.

    `eval-run` publishes MICRODUCK_RUN_CMD process-wide, so the refusal has to
    happen BEFORE that write — and the test sandboxes the variable either way
    (an earlier version of this test leaked 0.4 into every later test in the
    process and failed two of them 700 tests downstream)."""
    import os
    import sys

    from microduck_local import eval_onnx as EV

    monkeypatch.delenv("MICRODUCK_RUN_CMD", raising=False)
    onnx = tmp_path / "policy.onnx"
    onnx.write_bytes(b"x")
    monkeypatch.setattr(sys, "argv",
                        ["eval-walk", str(onnx), "--robot", "g1", "--behavior", "run"])
    with pytest.raises(SystemExit, match="Microduck reward recipe"):
        EV.main()
    assert "MICRODUCK_RUN_CMD" not in os.environ, "the refusal armed a global knob"


# --------------------------------------------- a locally trained G1 policy

@needs_g1
def test_the_world_can_run_a_locally_trained_g1_policy(monkeypatch, tmp_path):
    """`export-walk` writes the same 99 -> 29 graph the shipped walker has, so
    MICRODUCK_G1_WALKER swaps a policy trained here into the /sim person."""
    from microduck_local.robots import g1

    assert g1.walker_onnx().name == "walker.onnx"
    monkeypatch.setenv("MICRODUCK_G1_WALKER", str(g1.CACHE_DIR / "walker.onnx"))
    assert g1.walker_onnx() == g1.CACHE_DIR / "walker.onnx"
    monkeypatch.setenv("MICRODUCK_G1_WALKER", str(tmp_path / "nope.onnx"))
    with pytest.raises(FileNotFoundError):
        g1.walker_onnx()


@needs_g1
def test_a_policy_of_the_wrong_width_is_refused_by_the_world(tmp_path):
    """The /sim person's session checks its input width: a 61-obs duck policy
    dropped in here used to load and drive a humanoid with duck actions."""
    torch = pytest.importorskip("torch")
    from microduck_local.robots import g1

    net = torch.nn.Linear(61, 14)
    p = tmp_path / "duck.onnx"
    torch.onnx.export(net, (torch.zeros(1, 61),), str(p), input_names=["obs"],
                      output_names=["actions"], opset_version=17, dynamo=False)
    with pytest.raises(ValueError, match="99-d"):
        g1._walker_session(str(p))


# ------------------------------------------- the warm-start normalizer rules
#
# Two failures cost a full training run each, and both were the observation
# NORMALIZER rather than the reward or the budget. These pin the fixes.

class _Rms:
    def __init__(self, n):
        self.mean = np.zeros(n, np.float64)
        self.var = np.ones(n, np.float64)


class _Venv:
    def __init__(self, n):
        self.obs_rms = _Rms(n)
        self.training = True


def test_the_twist_command_is_never_normalized():
    """A policy cloned under a PINNED zero command has ~zero variance in the
    command slots. Normalized, its first real 0.9 arrives as ~6400 (clipped
    to 100) and the policy collapses; "fixing" that with the sampler's true
    mean/var moves ZERO off zero and collapses it differently. mean 0 / var 1
    has neither problem and leaves a zero command mapping to zero."""
    from microduck_local.robots import g1
    from microduck_local.train import _pass_through_command_dims

    spec = g1.G1_SPEC
    venv = _Venv(spec.obs_dim)
    start, stop = spec.twist_obs_slice
    venv.obs_rms.var[start:stop] = 1e-8          # what the clone leaves behind
    venv.obs_rms.mean[3] = 0.5                   # an unrelated dim
    venv.obs_rms.var[3] = 0.25

    bad = (0.9 - venv.obs_rms.mean[start]) / np.sqrt(venv.obs_rms.var[start] + 1e-8)
    assert bad > 1000, "the failure this guards against should be enormous"

    _pass_through_command_dims(venv, spec)
    assert np.allclose(venv.obs_rms.mean[start:stop], 0.0)
    assert np.allclose(venv.obs_rms.var[start:stop], 1.0)
    # a command now arrives as itself…
    good = (0.9 - venv.obs_rms.mean[start]) / np.sqrt(venv.obs_rms.var[start] + 1e-8)
    assert abs(good - 0.9) < 1e-3
    # …zero still maps to zero, so a policy trained before the switch is
    # unaffected by it…
    assert abs((0.0 - venv.obs_rms.mean[start]) / np.sqrt(venv.obs_rms.var[start])) < 1e-9
    # …and no other dimension was touched.
    assert venv.obs_rms.mean[3] == 0.5 and venv.obs_rms.var[3] == 0.25


def test_the_duck_command_slots_are_the_ones_the_contract_names():
    from microduck_local import contract as C
    start, stop = C.MICRODUCK.twist_obs_slice
    assert (start, stop) == (48, 51)             # twist_cmd in the 61-d layout
    assert C.OBS_DIM - C.CMD_DIM == start


@needs_g1
def test_the_stand_task_does_not_ramp_its_action_rate_penalty():
    """The duck ramps this 0.1 -> 1.0 over a run; a walking policy earns
    enough to outrun it. The G1 idle does not, and sums it over 29 joints:
    measured, the ramp drove ep_rew to -92 and the rendered policy hinged at
    the waist and dived. Pinned at the weight that was working."""
    from microduck_local.robots.g1_env import G1StandEnv, G1WalkEnv

    stand = G1StandEnv(seed=0, obs_noise=False, domain_rand=False)
    walk = G1WalkEnv(seed=0, obs_noise=False, domain_rand=False)
    for steps in (0, 50_000, 5_000_000):
        stand._lifetime_steps = walk._lifetime_steps = steps
        assert stand._action_rate_weight() == G1StandEnv.ACTION_RATE_W == 0.1
    assert walk._action_rate_weight() > 0.1, "the walk task keeps the ramp"


@needs_g1
def test_the_idle_ignores_the_command_only_when_asked_to():
    """`command_mix` is 0 while cloning (copy the teacher's zero-command
    behaviour) and turned up for the stage that buys command immunity."""
    from microduck_local.robots.g1_env import G1StandEnv

    pinned = G1StandEnv(seed=0, obs_noise=False, domain_rand=False)
    for _ in range(20):
        pinned.reset()
        assert not pinned.twist_cmd.any()

    mixed = G1StandEnv(seed=0, obs_noise=False, domain_rand=False, command_mix=1.0)
    seen = 0
    for i in range(20):
        mixed.reset(seed=i)
        seen += int(bool(mixed.twist_cmd.any()))
    assert seen >= 18, f"a full mix should nearly always command something ({seen}/20)"
