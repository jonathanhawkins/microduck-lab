"""`eval-brain --jobs N` must return exactly what `--jobs 1` returns.

That rests on one property: an episode is a pure function of (seed, ep).
BrainEnv.reset() re-seeds every generator that outlives world.reset() and
zeroes the commanded twist, so nothing rides from episode k-1 into episode
k, and a worker that starts at episode 7 gets episode 7's numbers.

The property is easy to lose — a new sensor with its own generator, or a
new piece of duck state _respawn forgets, and `--jobs N` silently stops
agreeing with `--jobs 1` while every test that only looks at one episode
still passes. These tests are the guard: the first two pin the property
directly, the third pins the thing it exists for.
"""

import numpy as np
import pytest

from microduck_local.brain.brain_env import FollowTask
from microduck_local.eval_brain import _build, _chunks, _episode, run

TASK = FollowTask(episode_s=4.0)          # short: these run in CI, not a battery
SEED = 100


def test_an_episode_does_not_depend_on_what_ran_before_it():
    """Episode k scored after episodes 0..k-1 == episode k scored alone.
    This is the carrier test: before BrainEnv.reset() re-seeded the sensors
    and zeroed the commanded twist, episode 0 matched and every episode
    after it drifted."""
    brain, env = _build("follow", "hostile", SEED, TASK, False)
    chained = [_episode(brain, env, TASK, SEED, ep) for ep in range(3)]
    for ep, want in enumerate(chained):
        b2, e2 = _build("follow", "hostile", SEED, TASK, False)   # a virgin env
        assert _episode(b2, e2, TASK, SEED, ep) == want, f"episode {ep} carried state"


def test_episode_order_does_not_matter():
    """The same env, episodes taken out of order: each still its own."""
    brain, env = _build("follow", "hostile", SEED, TASK, False)
    forward = [_episode(brain, env, TASK, SEED, ep) for ep in range(3)]
    b2, e2 = _build("follow", "hostile", SEED, TASK, False)
    backward = {ep: _episode(b2, e2, TASK, SEED, ep) for ep in (2, 1, 0)}
    assert [backward[ep] for ep in range(3)] == forward


@pytest.mark.parametrize("jobs", [2, 3])
def test_jobs_n_is_exactly_jobs_1(jobs):
    """What --jobs is for: N processes, one env each, identical rows."""
    serial = run("follow", "hostile", 4, SEED, TASK, jobs=1)
    par = run("follow", "hostile", 4, SEED, TASK, jobs=jobs)
    assert par["rows"] == serial["rows"]
    for k in ("in_band", "seen", "dist_err", "return", "falls", "bumps"):
        assert par[k] == serial[k], k
    assert par["episodes"] == serial["episodes"] == 4


def test_chunks_cover_every_episode_once():
    for episodes in (1, 4, 7, 24):
        for jobs in (1, 2, 3, 4, 8):
            cs = _chunks(episodes, jobs)
            assert sorted(e for c in cs for e in c) == list(range(episodes))
            assert len(cs) <= min(jobs, episodes)
            assert all(cs), "no empty chunk: a worker with nothing to do is a wasted env"


def test_jobs_is_clamped_to_the_episode_count():
    """--jobs 8 on 2 episodes must not start 8 envs to run 2 episodes."""
    res = run("follow", "hostile", 2, SEED, TASK, jobs=8)
    assert len(res["rows"]) == 2
    assert res["rows"] == run("follow", "hostile", 2, SEED, TASK, jobs=1)["rows"]


def test_sensor_streams_restart_each_episode():
    """The carrier itself, at the source: two resets from the same seed
    leave the sensor generators in the same state."""
    _, env = _build("follow", "hostile", SEED, TASK, False)
    env.reset(seed=SEED)
    first = [g.rng.bit_generator.state for g in (env.duck.tof, env.duck.detector, env.world)]
    env.reset(seed=SEED + 1)          # a different episode moves them
    env.reset(seed=SEED)              # ... and coming back restores them
    assert [g.rng.bit_generator.state for g in (env.duck.tof, env.duck.detector, env.world)] == first


def test_respawn_leaves_no_command_standing():
    """The other carrier: the warm-up steps in reset() must not be driven by
    what the last episode was asking for."""
    _, env = _build("follow", "hostile", SEED, TASK, False)
    env.reset(seed=SEED)
    env.step(np.array([0.15, 0.0, 0.5], np.float32))
    assert np.any(env.duck.twist_cmd), "test is inert if the command never moved"
    env.reset(seed=SEED + 1)
    assert not np.any(env.duck.twist_cmd), "the next episode warms up on the last one's command"
    assert not np.any(env.duck.head_cmd)
