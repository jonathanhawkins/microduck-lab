"""The brain layer (docs/sim-roadmap.md, Track 2): controllers that read
simulated senses and emit only the robot's own intents, a twist and a head
pose, for the unchanged reflex policy underneath. Nothing here touches
physics or the 61-obs contract."""

from .controllers import Wander, WanderParams, wander_from_tof

__all__ = ["Wander", "WanderParams", "wander_from_tof"]
