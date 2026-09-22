"""A walk/task run of another body previews in ITS OWN env on the lab page.

`train-walk --robot g1 --task imitate` writes a run.json and no
behavior.json. The lab's env lookup only read behavior.json, so it answered
{} and `Duck._make_env` fell through to `task="walk"`: the G1's
measured-best front kick was stepped in G1WalkEnv, where the three command
slots carry a held locomotion twist instead of the clip's clock. The policy
read each held twist as one frozen phase. Filmed on the lab page
(2026-09-17) it kicked once after a reset and then stood in a split stance
for the rest of the episode; headless, 2 lifts in 12 s against 4 in its own
env (one per 3 s loop of the clip).

These go through `env_kwargs_for_policy_path` — the seam spawn, assign and
restore all call — so against the old lookup they fail on the BEHAVIOUR (an
empty dict, a walk env, a clock that does not turn), not on a missing name.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from microduck_local import viz_server as V
from microduck_local.robots.g1 import g1_ready

needs_g1 = pytest.mark.skipif(not g1_ready(), reason="G1 assets missing — uv run fetch-g1")

CLIP = "g1-front-kick"          # committed in clips/, a 3.0 s loop


def _task_run(tmp_path: Path, name: str = "g1-kick", **meta) -> str:
    """A run dir as `train-walk` leaves it: run.json, a policy, no behavior.json."""
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "policy.onnx").write_bytes(b"not really onnx")
    (d / "run.json").write_text(json.dumps({"run_name": name, **meta}))
    return str(d / "policy.onnx")


def test_an_imitation_run_previews_in_its_task_env_on_its_clip(tmp_path):
    path = _task_run(tmp_path, robot="g1", task="imitate",
                     env_kwargs={"domain_rand": True, "obs_noise": True, "command_mix": 0.0,
                                 "max_episode_s": 40.0, "clip_name": CLIP})
    kw = V.env_kwargs_for_policy_path(path)
    assert kw.get("task") == "imitate", f"an imitation run fell through to the walk env: {kw!r}"
    assert kw.get("clip_name") == CLIP
    assert kw.get("max_episode_s") == 40.0
    # The trainer's noise and randomisation are the TRAINER's: the preview's
    # own `common` pins them off, and passing them twice is a TypeError.
    assert not {"domain_rand", "obs_noise", "command_mix"} & set(kw)


@pytest.mark.parametrize("task", ["stand", "squat", "front_kick", "punch"])
def test_every_task_env_that_owns_the_command_slots_is_named(tmp_path, task):
    kw = V.env_kwargs_for_policy_path(_task_run(tmp_path, robot="g1", task=task))
    assert kw == {"task": task}


@pytest.mark.parametrize("meta", [
    {"robot": "g1", "task": "walk"},          # the walk env IS the right env
    {"robot": "g1"},                          # …and the default
    {"robot": "microduck", "task": "walk"},
    {},                                       # a duck run from before run.json said `robot`
])
def test_a_walk_run_keeps_the_walk_env(tmp_path, meta):
    assert V.env_kwargs_for_policy_path(_task_run(tmp_path, **meta)) == {}


@pytest.mark.parametrize("meta", [
    {"robot": "g1", "task": "moonwalk"},                                       # env_class raises SystemExit
    {"robot": "g1", "task": "imitate", "env_kwargs": {"clip_name": "no-such-clip"}},
    {"robot": "g1", "task": "imitate", "env_kwargs": "not a dict"},               # …so no clip at all:
    {"robot": "g1", "task": "imitate"},                                        # G1ImitateEnv raises without one
])
def test_a_run_it_cannot_build_degrades_to_the_walk_env(tmp_path, meta):
    """Spawn, assign and restore all call this; a raise there costs the roster."""
    assert V.env_kwargs_for_policy_path(_task_run(tmp_path, **meta)) == {}


def test_a_behavior_json_still_wins(tmp_path):
    path = _task_run(tmp_path, robot="g1", task="imitate", env_kwargs={"clip_name": CLIP})
    (Path(path).parent / "behavior.json").write_text(json.dumps({"behavior": "one_leg"}))
    assert V.env_kwargs_for_policy_path(path).get("behavior_id") == "one_leg"


@needs_g1
def test_the_lab_slot_for_an_imitation_run_turns_the_clips_clock(tmp_path):
    """The thing that was filmed. In the OBSERVATION the policy is handed, the
    three command slots must advance one clip step per tick — (sin, cos, 0)
    going round the unit circle once per loop. In the walk env they hold one
    twist for seconds, which is the freeze.

    Stepped as `lab_loop` steps a trick slot: `set_cmd(zeros)`, then `tick()`.
    The lab writes its command into `twist_cmd` every tick and the env's
    `_get_obs` writes the clock back over it, so this also pins the ORDER —
    a lab command that reached the observation would read as a frozen phase.

    A zero-action G1 falls inside a second and each reset draws a fresh
    phase, so this asserts the rate between resets, not a full sweep."""
    path = _task_run(tmp_path, robot="g1", task="imitate",
                     env_kwargs={"max_episode_s": 40.0, "clip_name": CLIP})
    duck = V.Duck("d0", "g1-kick", lambda obs: np.zeros(29, np.float32), seed=37,
                  robot="g1", env_kwargs=V.env_kwargs_for_policy_path(path))
    assert type(duck.env).__name__ == "G1ImitateEnv", type(duck.env).__name__
    nj = duck.env.num_joints if hasattr(duck.env, "num_joints") else 29
    slots = slice(9 + 3 * nj, 9 + 3 * nj + 3)             # robots/g1_env._get_obs
    assert np.allclose(duck.obs[slots], duck.env.twist_cmd), "the command slots moved in the 99-d layout"
    per_tick = 2 * math.pi / duck.env.clip.steps          # 150 steps in the 3.0 s loop
    prev_angle, prev_count, advanced = None, -1, 0
    for _ in range(3 * 50):
        duck.set_cmd(np.zeros(3, np.float32))             # what lab_loop hands a trick slot…
        duck.tick()                                       # …then the lab's own step, resets included
        s, c, z = (float(x) for x in duck.obs[slots])
        assert z == 0.0 and math.isclose(s * s + c * c, 1.0, abs_tol=1e-4), (
            f"the policy was handed {(s, c, z)} — a lab command, not the clip's clock")
        angle = math.atan2(s, c)
        if duck.env.step_count <= prev_count:
            angle = None                                  # an episode ended: a fresh phase was drawn
        elif prev_angle is not None:
            step = (angle - prev_angle + math.pi) % (2 * math.pi) - math.pi
            assert math.isclose(step, per_tick, abs_tol=1e-3), (
                f"the clock moved {step:+.4f} rad in one tick, the clip asks for {per_tick:+.4f}")
            advanced += 1
        prev_angle, prev_count = angle, duck.env.step_count
    assert advanced >= 100, f"only {advanced} of 150 ticks were inside an episode"
