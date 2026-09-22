"""The brain layer (docs/sim-roadmap.md, Track 2): controllers that read
simulated senses and emit only the robot's own intents, a twist and a head
pose, for the unchanged reflex policy underneath. Nothing here touches
physics or the 61-obs contract."""

from .controllers import Follow, FollowParams, Script, Wander, WanderParams, wander_from_tof
from .runtime import REGISTRY, Brain, Intent, Senses
from .tidy import Tidy, TidyParams
from .tidy_arm import TidyArm, TidyArmParams

__all__ = ["Brain", "Follow", "FollowParams", "Intent", "REGISTRY", "Script", "Senses",
           "Tidy", "TidyArm", "TidyArmParams", "TidyParams", "Wander", "WanderParams", "wander_from_tof"]
