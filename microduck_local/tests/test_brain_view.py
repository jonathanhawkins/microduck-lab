"""A learned brain reports what its network saw and said (brain.view).

The /sim inspector draws it: the 80 observation floats, the action before
and after the intent clip, and the clip bounds. The exported graph clamps
to the same bounds, so for a shipped brain the two actions agree and the
saturated-mean tell — the log_std trap — is an action pinned on a bound,
visible live instead of only after a deterministic probe.
"""

import numpy as np
import pytest

from microduck_local.brain.brain_env import BRAIN_OBS_DIM
from microduck_local.brain.learned import brains_dir
from microduck_local.brain.runtime import REGISTRY, Senses, brain_view, payload
from microduck_local.sensors.detector import Detection, DetectionFrame

pytestmark = pytest.mark.skipif(
    not (brains_dir() / "follow-v2" / "brain.onnx").exists(), reason="brains/follow-v2 not shipped here")


def _senses(t: float) -> Senses:
    det = DetectionFrame(t=t - 0.05, detections=[Detection("person", "p0", 0.2, -0.1, 0.3, 1.4, 0.9)])
    return Senses(t=t, tof=None, tof_age=None, det=det, det_age=0.05, speed=0.12, odom=(0.0, 0.0, 0.0))


def test_view_is_none_until_the_first_decision():
    brain = REGISTRY.make("learned:follow-v2")
    assert brain.view() is None
    assert "view" not in payload(brain, None, "auto")


def test_view_carries_the_decision_and_the_payload_forwards_it():
    brain = REGISTRY.make("learned:follow-v2")
    intent = brain.step(_senses(1.05))
    v = brain.view()
    assert v is not None
    assert len(v["obs"]) == BRAIN_OBS_DIM and v["obs_version"] == 2 and v["decide_every"] == brain.decide_every
    act = v["act"]
    assert len(act["raw"]) == len(act["clipped"]) == len(act["low"]) == len(act["high"]) == 3
    lo, hi = np.asarray(act["low"]), np.asarray(act["high"])
    np.testing.assert_allclose(act["clipped"], np.clip(act["raw"], lo, hi), atol=2e-3)
    assert np.all(np.asarray(act["clipped"]) >= lo - 1e-6) and np.all(np.asarray(act["clipped"]) <= hi + 1e-6)
    # The clipped action IS the intent the duck was given.
    np.testing.assert_allclose(act["clipped"], intent.twist, atol=2e-3)
    # The observation is the one the contract describes: a person was seen.
    assert v["obs"][65] == 1.0
    assert payload(brain, intent, "auto")["view"] == v


def test_view_is_held_between_decisions_and_cleared_by_reset():
    brain = REGISTRY.make("learned:follow-v2")
    brain.step(_senses(1.05))
    first = brain.view()
    brain.step(_senses(1.07))            # not a decision tick: same view
    assert brain.view() == first
    brain.reset()
    assert brain.view() is None


def test_brain_view_rounds_and_exposes_saturation():
    v = brain_view(np.zeros(4), [0.9, -2.5, 0.12345], [0.6, -1.0, 0.123], [-0.2, -0.3, -1.0], [0.6, 0.3, 1.0], 2, 5)
    assert v["obs"] == [0.0, 0.0, 0.0, 0.0]
    assert v["act"]["raw"] == [0.9, -2.5, 0.123]          # rounded to 3 dp, NOT clipped
    assert v["act"]["clipped"] == [0.6, -1.0, 0.123]
    assert brain_view(None, None, None, None, None, 2, 5) is None
