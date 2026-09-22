"""Worlds: many ducks and objects in ONE MuJoCo model (docs/sim-roadmap.md, Track 0).

`scenario.py` is the on-disk contract (what a /sim scenario JSON may say);
`compose.py` turns one into a compiled `MjModel` by attaching the upstream
robot MJCF once per duck with `MjSpec`, so the whole room costs one physics
step instead of one per duck.
"""

from .arena import World, WorldDuck, WorldPerson, WorldRobot, zero_infer
from .compose import DUCK_ROBOT, ROBOT_XML, DuckAddress, compose, duck_prefix, spawn_duck
from .scenario import (
    Ball,
    Basket,
    Box,
    Duck,
    Person,
    Pickable,
    Scenario,
    Wall,
    formation_roles,
    load_scenario,
    make_pitch,
    make_playroom,
    make_room,
    validate_scenario,
)

__all__ = [
    "DUCK_ROBOT", "ROBOT_XML", "DuckAddress", "World", "WorldDuck", "WorldPerson",
    "WorldRobot", "compose", "duck_prefix",
    "spawn_duck", "zero_infer",
    "Ball", "Basket", "Box", "Duck", "Person", "Pickable", "Scenario", "Wall",
    "load_scenario", "make_pitch", "make_playroom", "make_room", "formation_roles",
    "validate_scenario",
]
