"""Simulated exteroception for the /sim world.

Every sensor here is sampled at the REAL device's rate and shaped like the
robot's own output (see docs/sim-roadmap.md, Track 1). The physics never
sees these — a brain does, and it only ever talks back to the reflex policy
through the 13 command slots of the 61-obs contract.
"""

from .detector import (
    DETECT_CLASSES,
    Detection,
    DetectionFrame,
    Detector,
    DetectorNoise,
    DetectorSpec,
    Target,
)
from .ray import RayFan, RayHits, planar_fan, tof_fan
from .tof import TofFrame, TofNoise, TofSensor, TofSpec

__all__ = [
    "DETECT_CLASSES", "Detection", "DetectionFrame", "Detector", "DetectorNoise", "DetectorSpec", "Target",
    "RayFan", "RayHits", "planar_fan", "tof_fan",
    "TofFrame", "TofNoise", "TofSensor", "TofSpec",
]
