"""The scenario contract: a room, its objects, and the ducks in it.

Format v1 (JSON, saved under microduck_local/scenarios/<name>.json):

    {"version": 1, "name": "living-room", "seed": 0,
     "floor": {"size": [4.0, 3.0]},               # half-extents NOT — full x, y metres
     "walls": [{"from": [x, y], "to": [x, y], "height": 0.3, "thickness": 0.02}],
     "boxes": [{"pos": [x, y, z], "size": [sx, sy, sz], "yaw": 0.0,
                "mass": 0.0, "rgba": [r, g, b, a]}],     # mass 0 = static scenery
     "balls": [{"pos": [x, y], "radius": 0.035, "mass": 0.015, "rolling": 0.002}],  # rolling: the floor
     "ducks": [{"id": "d0", "spawn": [x, y, yaw], "policy": "pollen:alpha_walking",
                "tof": "datasheet", "detector": "datasheet", "brain": "follow",
                "team": "cream", "role": "striker",        # soccer: a colorway, a job
                "robot": "microduck"}],                    # WHICH BODY (robots/registry.ids())
     "goal_width": 0.7,                                # > 0 makes it a pitch
     "attacks": {"cream": "right"},                    # …and which mouth a team attacks
     "persons": [{"id": "p0", "pos": [x, y], "yaw": 0.0, "path": [[x, y], ...],
                  "speed": 0.3, "radius": 0.2, "height": 1.0, "yield_m": 0.55,
                  "kind": "capsule"|"g1"}],  # capsule = mocap; g1 = Unitree G1 + walker.onnx
     "pickables": [{"id": "t0", "kind": "brick"|"block"|"sock", "pos": [x, y], "yaw": 0.0}],
     "basket": {"pos": [x, y], "size": [0.3, 0.3], "rim": 0.06} | null,   # the tidy target
     "physics_dt": 0.005,                               # MuJoCo timestep (see Scenario.physics_dt)
     "collision": "all"}                                # "all" | "walk" robot MJCF ("walk": only the soles collide)

Everything is metres, radians, world frame, z up. Validation is strict on
purpose: a scenario that compiles into a model nobody meant is worse than a
loud error in the editor. Names are the only free-form strings, and they are
constrained to what is safe as a file name and an MJCF prefix.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCENARIO_VERSION = 1
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
DUCK_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,15}$")
# A brain is a registry kind ("follow", "chase") or a shipped brain,
# "learned:<run>" — run names carry dashes ("follow-v4", "p-n256-s31"), so
# the duck-id pattern that used to gate this field rejected every learned
# brain, and a scene could not be SAVED on one even though the inspector
# could switch a duck to it live.
BRAIN_RE = re.compile(r"^[a-z][a-z0-9_]*(?::[A-Za-z0-9][A-Za-z0-9_.-]{0,63})?$")
MAX_DUCKS = 12
MAX_PERSONS = 4

# A team IS a colorway (roadmap Track 4.2). Two reasons, one practical and one
# honest. Practical: the editor and the viewer need to say which duck is on
# which side, and a shell colour is the only thing a person watching six ducks
# can read at a glance. Honest: on the robot two ducks of the same colorway
# cannot be told apart by any sensor either, which is exactly what a team is —
# so a team that is a colour is a team the hardware could actually play.
#
# The four Pollen ships (press kit): the shell colour and the trim-and-beak
# colour that goes with it. sRGB, as the MJCF materials are.
#
# TWO colours, and every printed part on the duck takes one of them — head,
# trunk, legs and hips are all the SHELL colour, beak, feet, ankles and soles
# are all the TRIM. A colorway is a set of printed parts, so a duck wearing one
# is that colour from the beak down, exactly as the press kit shows it.
#
# This replaced a version that gave the legs a deeper cast of the shell. It
# looked reasonable in the table and wrong on the duck: the MJCF materials are
# OnShape export appearances, and several PRINTED parts carry colours no
# colorway ever claimed — a teal thigh plate and shoe rim
# (`upper_leg_rigidity_plate`, `sole_*` at #89dad3), a pale-blue hip
# (`yaw_roll_motion`), a pink mouth (`jaw_soft`, `soft_mouth_top`) that is the
# same pink on all four colorways. A second body colour on top of those made
# five, and the duck read as a patchwork rather than a printed shell. The
# material groups in `world/compose.py` are the list of what a colorway owns.
TEAM_COLORWAYS: dict[str, dict[str, tuple[float, float, float]]] = {
    "cream":    {"shell": (0.969, 0.902, 0.796), "trim": (0.95, 0.55, 0.13)},   # #f7e6cb, orange trim
    "graphite": {"shell": (0.424, 0.416, 0.408), "trim": (0.98, 0.78, 0.10)},   # #6c6a68, yellow trim
    "lavender": {"shell": (0.749, 0.663, 0.812), "trim": (0.98, 0.78, 0.10)},   # #bfa9cf, yellow trim
    "sky":      {"shell": (0.663, 0.859, 0.910), "trim": (0.95, 0.55, 0.13)},   # #a9dbe8, orange trim
}
# The pair `make_pitch` puts on a pitch, and what a legacy "left"/"right"
# scene loads as.
#
# Cream v GRAPHITE, chosen by measuring rather than by eye. Every pair of the
# four ships, CIE76 dE between the shell colours and the gap in lightness:
#
#     cream v graphite    49.0   47.0   different trim   <- this one
#     graphite v sky      43.7   39.5   different trim
#     cream v lavender    39.7   19.8   different trim
#     graphite v lavender 35.7   27.2   SAME trim
#     cream v sky         31.6    7.5   SAME trim
#     lavender v sky      31.1   12.2   different trim
#
# Cream v graphite wins on both columns and by a wide margin. The lightness
# gap is the one that matters at the size a duck actually appears — 60 px in
# a wide 3v3 shot, less in a 640-px head-camera frame — because hue is the
# first thing to go when a shape is small, blurred by motion, or lit from one
# side, and light-against-dark survives all three. The trim column is the
# second cue: cream's beak and feet are orange and graphite's are yellow, so
# a duck that is only a few pixels of leg still says which side it is on.
# (Cream v lavender shipped first and reads fine on a still; graphite v
# lavender was considered and is worse on both counts, and shares a trim.)
PITCH_TEAMS: tuple[str, str] = ("cream", "graphite")
# What the first pitches called their teams. They were the two SIDES of the
# pitch, which collided head-on with the goal MOUTH keys the World writes
# (`goals["right"]` is the mouth at +x, which the "left" team attacks) — a
# collision the README needed a standing warning paragraph for. A saved scene
# still loads: the names map to the two colorways `make_pitch` now uses.
LEGACY_TEAMS = dict(zip(("left", "right"), PITCH_TEAMS))
# What a duck is for on a pitch (roadmap Track 4.3). None keeps today's
# behaviour: the team blackboard picks one attacker by predicted time to the
# ball and the rest support it.
ROLES = ("defender", "midfielder", "striker", "keeper")   # keeper: roadmap Track 4 s6 B.2 (brain/team.py ROLE_ZONES)


def formation_roles(n: int) -> list[str | None]:
    """Static jobs for one side of `n` ducks. 1v1 has none (the chase-vs-chase
    control); 2 is defender + striker; 3+ is defender, midfielder(s), striker.
    `make_pitch(..., formation=True)` stamps these; `eval-pitch` does not."""
    n = max(0, int(n))
    if n <= 1:
        return [None] * n
    if n == 2:
        return ["defender", "striker"]
    return ["defender"] + ["midfielder"] * (n - 2) + ["striker"]
MAX_OBJECTS = 200
MAX_FLOOR_M = 20.0
# 2.0 admitted every duck room (walls are 0.3 m) but not the 2.4 m ones the
# G1 follow scene builds: a 1.32 m person walks through a 30 cm ceiling and
# looks like a chimney, so `world_server` raises them — and the built-in then
# failed its OWN validator the moment the G1 assets were fetched
# (tests/test_world_server.py::test_builtin_scenarios_validate_and_list).
# The cap exists to stop a scenario asking for a skyscraper, not to pick the
# room height, so it clears a person-sized room with headroom.
MAX_WALL_HEIGHT_M = 3.0
TOF_PRESETS = ("ideal", "datasheet", "hostile")

# The physics timestep a room runs at, and the control tick every world in
# this repo streams and decides at. Re-declared here rather than imported
# because `contract.py` pulls in `robots/microduck.py` and this module is the
# ON-DISK contract — the editor and `record-world` import it to validate a
# file and must not drag a robot package in to read a room's walls (the same
# reason `_robot_ids` imports inside the function). `tests/test_tidy_arm.py`
# asserts both against `contract.PHYSICS_DT` / `contract.CTRL_DT`, so a change
# to the world's clock fails there instead of drifting.
DEFAULT_PHYSICS_DT = 0.005
CONTROL_DT = 0.02
#: A room's timestep must divide the control tick to this tolerance — the
#: arena derives its substeps per tick by rounding, and a 3 ms step (6.67
#: substeps) would silently run the world at 49.5 Hz.
PHYSICS_DT_RANGE = (0.0005, CONTROL_DT)


class ScenarioError(ValueError):
    pass


@dataclass
class Wall:
    start: tuple[float, float]
    end: tuple[float, float]
    height: float = 0.3
    thickness: float = 0.02


@dataclass
class Box:
    pos: tuple[float, float, float]
    size: tuple[float, float, float]   # FULL extents (x, y, z), metres
    yaw: float = 0.0
    mass: float = 0.0                  # 0 = static scenery, else a free body
    rgba: tuple[float, float, float, float] = (0.55, 0.45, 0.35, 1.0)


@dataclass
class Ball:
    pos: tuple[float, float]
    radius: float = 0.035              # upstream's 70 mm kick ball
    mass: float = 0.015
    # Rolling resistance against the floor, MuJoCo's rolling-friction
    # coefficient (units of length: the rolling torque the contact resists
    # is `rolling` x the normal force). It is the FLOOR that this number
    # describes; the ball is the same hollow plastic everywhere. Measured
    # roll-out on an open floor (tests/test_world.py guards the first row):
    #     rolling   nudge 0.2 m/s   walked into 0.45 m/s   kick 1.4 m/s
    #     0 (none)  12 m, never     27 m, never            84 m, never
    #     0.001     0.84 m / 16 s   2.4 m / 20 s           9.9 m / 29 s
    #     0.0015    0.41 m / 7 s    1.2 m / 10 s           5.2 m / 14 s
    #     0.002     0.25 m / 4 s    0.72 m / 6 s           3.5 m / 9 s
    # 0.002 is a short carpet / rubber mat, the floor a home robot lives
    # on: a bump stops within a stride, a kick crosses the pitch. Before
    # 2026-09-06 the ball had NO rolling resistance at all (the coefficient
    # was set but the geom's condim of 3 ignores it) - a bumped ball rolled
    # until a wall stopped it.
    rolling: float = 0.002


@dataclass
class Duck:
    """One ROBOT in a room. Named `Duck` because for two tracks it could only
    be one, and the name is now the wire format — every saved scenario and
    every `GET /scenarios` payload says `ducks`.

    `robot` is which BODY the entry attaches (`robots/registry.ids()`), and
    `"microduck"` is the default so that every scenario written before this
    field is byte-for-byte the scenario it was. A non-duck body reads the
    same fields with its own meanings, documented where they differ:

    * `policy` — the duck's reflex walker. MARS has no walker (there is no
      gait to learn on a wheeled base), so `world/arena.WorldRobot` drives it
      with `Body.driver()` and this field is unused.
    * `tof` — the entry's RANGE-SENSOR preset, one of `TOF_PRESETS`. On the
      duck that sensor is the 8x8 ToF on the head; on MARS it is the 360-deg
      lidar on the chassis lid (`sensors/lidar.py`), which is why a MARS with
      `tof: null` has no range sensor at all and falls back to `script` the
      same way a blind duck does. A per-sensor field per device would be the
      tidier contract and is not worth a format version for one preset name.
    * `team` / `role` / `odom` — unchanged; `team` paints nothing on a body
      that has no colorway materials (`compose.paint_team` returns 0).
    """

    id: str
    spawn: tuple[float, float, float]  # x, y, yaw
    policy: str | None = None          # palette id; None = zero-action stand
    tof: str | None = "datasheet"      # range-sensor noise preset, None = no sensor
    detector: str | None = "datasheet" # camera+NPU detector preset, None = none
    brain: str | None = None           # brain kind in auto mode; None = wander if ToF else script
    odom: str = "ideal"                # odometry drift preset the brain's (x, y, yaw) carries (roadmap 1.7)
    team: str | None = None            # soccer: a TEAM_COLORWAYS name; teammates share a blackboard (brain/team.py)
    role: str | None = None            # soccer: a ROLES name; None = the board's dynamic attacker/support
    robot: str = "microduck"           # which BODY (robots/registry.ids()); absent means the duck


@dataclass
class Person:
    """A kinematic walker (a mocap capsule): what a duck follows.

    A mocap body has INFINITE mass: it is wherever the world writes it each
    tick and nothing a duck does moves it, so a person-duck contact cannot
    yield momentum the way a real person's would. It has to be AVOIDED, not
    softened - which is what `yield_m` does. Measured 2026-09-06 (collision
    "all", the shipped walker standing, capsule walking through its spot):
    at 0.10 m/s the capsule shoves the duck 0.22 m (peak 0.30 m/s, no
    fall); from 0.15 m/s up it knocks it over broadside (peak 0.21 m/s,
    then a fall), and head-on it pushes it at 0.35 m/s (0.15) to 0.61 m/s
    (0.3). The 4-6 m/s "fling" the physics audit saw was the fallen duck
    RESPAWNING inside the capsule (`World._respawn` now spawns clear of a
    person). No walking speed is safe for a walk-through person, so the
    loader does not cap `speed`; a polite walker never touches at any
    speed the loader allows (surface gap 34 cm at 1.5 m/s). `yield_m` 0
    reproduces the pre-2026-09 world, where the capsule walked through.
    """
    id: str
    pos: tuple[float, float]
    yaw: float = 0.0
    path: list[tuple[float, float]] = field(default_factory=list)   # waypoints, looped
    speed: float = 0.3                 # m/s along the path (with yield_m 0 anything >= 0.15 topples a duck, above)
    radius: float = 0.2
    height: float = 1.0
    # A polite walker: with a duck inside `yield_m` (centre to centre) ahead
    # on its way it stops, and after a wait steps on to its next waypoint
    # instead of walking through the duck. 0.55 is the follow benchmark's
    # `polite` (brain/brain_env.py FollowTask, which should read this): the
    # capsule's surface stops 0.35 m from the trunk. 0: walks through - see
    # the docstring for what that does to a duck.
    yield_m: float = 0.55
    # "capsule" is the mocap walker above. "g1" attaches the Unitree G1 MJCF
    # and drives it with the shipped walk ONNX along the same path — see
    # robots/g1.py. Capsule stays the default so every existing scenario and
    # test is unchanged.
    kind: str = "capsule"


PERSON_KINDS = ("capsule", "g1")


# What a duck can pick up: full extents (m), mass (kg), colour. Sizes are
# the real things — a 2×4 brick, a wooden block, a rolled-up sock.
PICKABLE_KINDS: dict[str, dict] = {
    "brick": {"size": (0.032, 0.016, 0.0096), "mass": 0.0025, "rgba": (0.85, 0.15, 0.15, 1.0)},
    "block": {"size": (0.04, 0.04, 0.04), "mass": 0.02, "rgba": (0.95, 0.75, 0.2, 1.0)},
    "sock": {"size": (0.06, 0.035, 0.025), "mass": 0.02, "rgba": (0.6, 0.6, 0.9, 1.0)},
}


@dataclass
class Pickable:
    id: str
    kind: str
    pos: tuple[float, float]
    yaw: float = 0.0


@dataclass
class Basket:
    """A low tray the duck drops things into: four thin walls on a floor
    plate. `rim` must sit below the beak when standing (~0.2 m)."""
    pos: tuple[float, float]
    size: tuple[float, float] = (0.3, 0.3)
    rim: float = 0.06


@dataclass
class Scenario:
    name: str
    seed: int = 0
    floor: tuple[float, float] = (4.0, 4.0)   # full x, y extents
    walls: list[Wall] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)
    balls: list[Ball] = field(default_factory=list)
    ducks: list[Duck] = field(default_factory=list)
    persons: list[Person] = field(default_factory=list)
    pickables: list[Pickable] = field(default_factory=list)
    basket: Basket | None = None
    goal_width: float = 0.0            # > 0: a pitch — goals on both short walls this wide (World counts them)
    # Which goal MOUTH each team attacks ({"cream": "right"}), for a pitch
    # whose ducks are not all placed facing it. Absent, a team attacks the
    # mouth its spawn heading faces, which is what every pitch did before this
    # field and what `World.goal_for` still falls back to.
    attacks: dict[str, str] = field(default_factory=dict)
    # Which robot MJCF the ducks are attached from. "all" gives every body its
    # collision mesh; "walk" - upstream's flat-floor training variant, and the
    # default here until 2026-09-06 - meets the world through the two 13 mm
    # soles ONLY: a ball at trunk height passed through a standing duck, a
    # walker had its beak 9 cm inside a wall before a foot touched it, two
    # walkers head-on overlapped to 3-7 cm trunk to trunk. The shipped
    # walker's flat-floor trajectory is bit-identical under both
    # (tests/test_arena.py), so "all" costs a policy nothing.
    collision: str = "all"
    # A quarter-round COVE along the base of every wall, this radius (m):
    # a ball rolling into the boards climbs it and rolls back out by
    # gravity instead of dying flush against the wall. The physics audit
    # (2026-09-06) found MuJoCo's soft contact does not model restitution -
    # a 1.4 m/s kick rebounds at e = 0.06 where a real hollow ball is
    # 0.5-0.7 - so in the sim a kicked ball sits at the wall it hits, and
    # the ball-out rule (`World.ball_out_s`) is the referee that patches
    # it. A cove returns the ball by a mechanism MuJoCo does model (a ball
    # rolling on a slope; the climb at 1.4 m/s is 17 cm, at 1 m/s 8.5 cm,
    # I = 2/3 m r^2), and on a real table it is a strip of quarter-round
    # moulding. A dribbled ball settles at the cove's foot, one radius off
    # the wall. Cut at the goal mouths, whose ends are the posts. 0 = flat,
    # every number measured before it. compose.py builds it from tangent
    # boxes; `make_pitch(cove=)` and `eval-pitch --cove` set it.
    cove: float = 0.0
    # The MuJoCo timestep this room is COMPILED at (`compose` writes it into
    # `option.timestep`, and `world/arena.World` derives its substeps per
    # 50 Hz tick from it). A property of the ROOM and not of a body, because
    # `MjSpec.attach` keeps the parent's `<option>`: there is one timestep and
    # every robot in the room runs on it.
    #
    # **Why it exists: a MARS cannot GRASP at 5 ms.** MEASURED 2026-09-18
    # (`docs/mars-roadmap.md` Phase 4b, `scripts/probe_mars_pick.py
    # --scripted`, 16 spots of the reach shell, each an IK place / close /
    # lift 10 cm / hold 2 s):
    #
    #     timestep   MARS's own option block   the world's (compose defaults)
    #     5 ms                   4/16                      1/16
    #     2 ms                  14/16                     14/16
    #
    # What fails at 5 ms is not slip but EJECTION — the block leaves at
    # 0.08-6.6 m of travel in the two seconds after the lift, a contact
    # impulse the coarse step cannot integrate. The elliptic cone and
    # `impratio 10` are a NULL (2 ms with the world's pyramidal cone at
    # impratio 1 scores exactly what Innate's block does), so the lever is the
    # step and nothing else — and a timestep is a whole-world decision, which
    # is why it is a scenario field and not something the composer could set
    # for one body.
    #
    # Every DUCK scenario stays at `DEFAULT_PHYSICS_DT`: the shipped walker's
    # trajectory is a golden bit (`tests/test_arena.py` locks the composed
    # world step for step against the walk env) and 2 ms would be a different
    # robot. `mars-playroom.json` and `mars-follow.json` carry 0.002, and
    # `robots/mars.MarsBody.physics_dt` is where that number comes from, so a
    # builder asks the BODY rather than typing it (`make_playroom`).
    physics_dt: float = DEFAULT_PHYSICS_DT
    version: int = SCENARIO_VERSION

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["floor"] = {"size": list(self.floor)}
        d["basket"] = None if self.basket is None else asdict(self.basket)
        d["walls"] = [{"from": list(w.start), "to": list(w.end),
                       "height": w.height, "thickness": w.thickness}
                      for w in self.walls]
        return d

    @classmethod
    def from_dict(cls, raw: dict) -> "Scenario":
        return validate_scenario(raw)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


# -- validation --------------------------------------------------------------

def _num(x, what: str, lo: float = -math.inf, hi: float = math.inf) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ScenarioError(f"{what}: expected a finite number, got {x!r}")
    if not (lo <= x <= hi):
        raise ScenarioError(f"{what}: {x} outside [{lo}, {hi}]")
    return float(x)


def _vec(x, n: int, what: str, lo: float = -math.inf, hi: float = math.inf) -> tuple:
    if not isinstance(x, (list, tuple)) or len(x) != n:
        raise ScenarioError(f"{what}: expected {n} numbers, got {x!r}")
    return tuple(_num(v, f"{what}[{i}]", lo, hi) for i, v in enumerate(x))


def robot_physics_dt(robot: str) -> float:
    """The timestep a room holding this BODY has to run at.

    The body declares it (`robots/body.BodyBase.physics_dt`): MARS answers
    2 ms because its claw ejects a block at 5 ms, the duck and the G1 answer
    nothing and get `DEFAULT_PHYSICS_DT`. Asked by the procedural builders
    (`make_playroom`) and by anything else that assembles a scenario in code,
    so the 2 ms lives in ONE place and no builder carries an `if robot ==`
    (`docs/mars-roadmap.md` §1 counts what that pattern cost the G1).

    A hand-written JSON is NOT silently upgraded: `physics_dt` is whatever the
    file says, because a room's clock is the room's and a loader that rewrote
    it would change the physics of a saved scene under its author.
    """
    from ..robots.registry import get

    return float(getattr(get(robot), "physics_dt", None) or DEFAULT_PHYSICS_DT)


def _robot_ids() -> tuple[str, ...]:
    """Every body id a scenario may name — `robots.registry.ids()`.

    `ids()` and not `registry()`: a scenario that names a body whose assets
    are not downloaded must still LOAD and then fail at compose with the
    fetch command, exactly as `--robot g1` does on a fresh checkout
    (`robots/registry.ids`'s docstring). Refusing it here would report a
    missing download as a malformed file.

    Imported inside the function so that `world/scenario.py` — the on-disk
    contract, which the editor and `record-world` both import to validate a
    file — does not drag the robot package (and its entry-point scan) in
    just to read a room's walls. The scan is memoised, so the second call
    costs a dict lookup.
    """
    from ..robots.registry import ids
    return ids()


def validate_scenario(raw: dict) -> Scenario:
    if not isinstance(raw, dict):
        raise ScenarioError("scenario must be a JSON object")
    if raw.get("version", SCENARIO_VERSION) != SCENARIO_VERSION:
        raise ScenarioError(f"unsupported scenario version {raw.get('version')!r}")
    name = raw.get("name", "")
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ScenarioError(f"bad scenario name {name!r}")
    seed = raw.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ScenarioError("seed must be a non-negative integer")
    floor_raw = raw.get("floor", {})
    if not isinstance(floor_raw, dict):
        raise ScenarioError("floor must be an object")
    floor = _vec(floor_raw.get("size", [4.0, 4.0]), 2, "floor.size", 0.5, MAX_FLOOR_M)
    half = (floor[0] / 2, floor[1] / 2)
    bound = max(half) + 1.0

    walls = []
    for i, w in enumerate(raw.get("walls", []) or []):
        if not isinstance(w, dict):
            raise ScenarioError(f"walls[{i}] must be an object")
        s = _vec(w.get("from"), 2, f"walls[{i}].from", -bound, bound)
        e = _vec(w.get("to"), 2, f"walls[{i}].to", -bound, bound)
        if math.dist(s, e) < 1e-3:
            raise ScenarioError(f"walls[{i}] has zero length")
        walls.append(Wall(s, e,
                          _num(w.get("height", 0.3), f"walls[{i}].height", 0.01, MAX_WALL_HEIGHT_M),
                          _num(w.get("thickness", 0.02), f"walls[{i}].thickness", 0.005, 0.5)))
    boxes = []
    for i, b in enumerate(raw.get("boxes", []) or []):
        if not isinstance(b, dict):
            raise ScenarioError(f"boxes[{i}] must be an object")
        boxes.append(Box(
            _vec(b.get("pos"), 3, f"boxes[{i}].pos", -bound, bound),
            _vec(b.get("size"), 3, f"boxes[{i}].size", 0.005, 5.0),
            _num(b.get("yaw", 0.0), f"boxes[{i}].yaw", -2 * math.pi, 2 * math.pi),
            _num(b.get("mass", 0.0), f"boxes[{i}].mass", 0.0, 50.0),
            _vec(b.get("rgba", [0.55, 0.45, 0.35, 1.0]), 4, f"boxes[{i}].rgba", 0.0, 1.0)))
    balls = []
    for i, b in enumerate(raw.get("balls", []) or []):
        if not isinstance(b, dict):
            raise ScenarioError(f"balls[{i}] must be an object")
        balls.append(Ball(
            _vec(b.get("pos"), 2, f"balls[{i}].pos", -bound, bound),
            _num(b.get("radius", 0.035), f"balls[{i}].radius", 0.005, 0.5),
            _num(b.get("mass", 0.015), f"balls[{i}].mass", 0.001, 5.0),
            _num(b.get("rolling", Ball.rolling), f"balls[{i}].rolling", 0.0, 0.05)))
    if len(walls) + len(boxes) + len(balls) > MAX_OBJECTS:
        raise ScenarioError(f"more than {MAX_OBJECTS} objects")

    ducks = []
    seen: set[str] = set()
    for i, d in enumerate(raw.get("ducks", []) or []):
        if not isinstance(d, dict):
            raise ScenarioError(f"ducks[{i}] must be an object")
        did = d.get("id", f"d{i}")
        if not isinstance(did, str) or not DUCK_ID_RE.match(did):
            raise ScenarioError(f"ducks[{i}].id {did!r} must match {DUCK_ID_RE.pattern}")
        if did in seen:
            raise ScenarioError(f"duplicate duck id {did!r}")
        seen.add(did)
        sp = d.get("spawn", [0.0, 0.0, 0.0])
        if not isinstance(sp, (list, tuple)) or len(sp) != 3:
            raise ScenarioError(f"ducks[{i}].spawn must be [x, y, yaw]")
        xy = _vec(sp[:2], 2, f"ducks[{i}].spawn", -bound, bound)
        yaw = _vec([sp[2]], 1, f"ducks[{i}].spawn yaw", -2 * math.pi, 2 * math.pi)   # a heading, not a coordinate
        spawn = (xy[0], xy[1], yaw[0])
        policy = d.get("policy")
        if policy is not None and (not isinstance(policy, str) or len(policy) > 200):
            raise ScenarioError(f"ducks[{i}].policy must be a palette id string or null")
        tof = d.get("tof", "datasheet")
        if tof is not None and tof not in TOF_PRESETS:
            raise ScenarioError(f"ducks[{i}].tof must be one of {TOF_PRESETS} or null")
        det = d.get("detector", "datasheet")
        if det is not None and det not in TOF_PRESETS:
            raise ScenarioError(f"ducks[{i}].detector must be one of {TOF_PRESETS} or null")
        brain = d.get("brain")
        if brain is not None and (not isinstance(brain, str) or not BRAIN_RE.match(brain)):
            raise ScenarioError(f"ducks[{i}].brain must be a brain kind name (or learned:<run>) or null")
        odom = d.get("odom", "ideal") or "ideal"
        if odom not in TOF_PRESETS:
            raise ScenarioError(f"ducks[{i}].odom must be one of {TOF_PRESETS}")
        team = d.get("team")
        if isinstance(team, str):
            team = LEGACY_TEAMS.get(team, team)      # a scene saved when teams were sides
        if team is not None and team not in TEAM_COLORWAYS:
            raise ScenarioError(f"ducks[{i}].team must be one of {sorted(TEAM_COLORWAYS)} or null")
        role = d.get("role")
        if role is not None and role not in ROLES:
            raise ScenarioError(f"ducks[{i}].role must be one of {sorted(ROLES)} or null")
        if role is not None and team is None:
            raise ScenarioError(f"ducks[{i}] has a role but no team")
        robot = d.get("robot") or Duck.robot
        if robot not in _robot_ids():
            raise ScenarioError(
                f"ducks[{i}].robot must be one of {sorted(_robot_ids())}, got {robot!r}")
        ducks.append(Duck(did, spawn, policy, tof, det, brain, odom, team, role, robot))
    if len(ducks) > MAX_DUCKS:
        raise ScenarioError(f"more than {MAX_DUCKS} ducks")
    persons = []
    for i, q in enumerate(raw.get("persons", []) or []):
        if not isinstance(q, dict):
            raise ScenarioError(f"persons[{i}] must be an object")
        pid = q.get("id", f"p{i}")
        if not isinstance(pid, str) or not DUCK_ID_RE.match(pid) or pid in seen:
            raise ScenarioError(f"persons[{i}].id {pid!r} bad or duplicate")
        seen.add(pid)
        path = [_vec(w, 2, f"persons[{i}].path[{k}]", -bound, bound)
                for k, w in enumerate(q.get("path", []) or [])]
        kind = q.get("kind", "capsule") or "capsule"
        if kind not in PERSON_KINDS:
            raise ScenarioError(f"persons[{i}].kind must be one of {PERSON_KINDS}")
        g1 = kind == "g1"
        persons.append(Person(
            pid, _vec(q.get("pos", [0.0, 0.0]), 2, f"persons[{i}].pos", -bound, bound),
            _num(q.get("yaw", 0.0), f"persons[{i}].yaw", -2 * math.pi, 2 * math.pi),
            path,
            _num(q.get("speed", 0.3), f"persons[{i}].speed", 0.0, 1.5),
            _num(q.get("radius", 0.25 if g1 else 0.2), f"persons[{i}].radius", 0.05, 0.5),
            _num(q.get("height", 1.32 if g1 else 1.0), f"persons[{i}].height", 0.2, 2.0),
            _num(q.get("yield_m", 0.80 if g1 else Person.yield_m), f"persons[{i}].yield_m", 0.0, 2.0),
            kind))
    if len(persons) > MAX_PERSONS:
        raise ScenarioError(f"more than {MAX_PERSONS} persons")
    pickables = []
    for i, q in enumerate(raw.get("pickables", []) or []):
        if not isinstance(q, dict):
            raise ScenarioError(f"pickables[{i}] must be an object")
        tid = q.get("id", f"t{i}")
        if not isinstance(tid, str) or not DUCK_ID_RE.match(tid) or tid in seen:
            raise ScenarioError(f"pickables[{i}].id {tid!r} bad or duplicate")
        seen.add(tid)
        kind = q.get("kind", "brick")
        if kind not in PICKABLE_KINDS:
            raise ScenarioError(f"pickables[{i}].kind must be one of {sorted(PICKABLE_KINDS)}")
        pickables.append(Pickable(tid, kind, _vec(q.get("pos"), 2, f"pickables[{i}].pos", -bound, bound),
                                  _num(q.get("yaw", 0.0), f"pickables[{i}].yaw", -2 * math.pi, 2 * math.pi)))
    if len(pickables) > 40:
        raise ScenarioError("more than 40 pickables")
    basket = None
    braw = raw.get("basket")
    if braw is not None:
        if not isinstance(braw, dict):
            raise ScenarioError("basket must be an object or null")
        basket = Basket(_vec(braw.get("pos"), 2, "basket.pos", -bound, bound),
                        _vec(braw.get("size", [0.3, 0.3]), 2, "basket.size", 0.1, 1.0),
                        _num(braw.get("rim", 0.06), "basket.rim", 0.02, 0.18))
    collision = raw.get("collision", Scenario.collision)
    if collision not in ("walk", "all"):
        raise ScenarioError("collision must be 'walk' or 'all'")
    goal_width = raw.get("goal_width", 0.0) or 0.0
    if not isinstance(goal_width, (int, float)) or not 0.0 <= goal_width <= 5.0:
        raise ScenarioError("goal_width must be a number in [0, 5]")
    cove = _num(raw.get("cove", 0.0) or 0.0, "cove", 0.0, 0.5)
    physics_dt = _num(raw.get("physics_dt") or DEFAULT_PHYSICS_DT, "physics_dt", *PHYSICS_DT_RANGE)
    sub = CONTROL_DT / physics_dt
    if abs(sub - round(sub)) > 1e-9:
        raise ScenarioError(
            f"physics_dt {physics_dt} does not divide the {CONTROL_DT} s control tick "
            f"({sub:.4f} substeps) — the arena rounds, so the world would run off 50 Hz")
    attacks = _validate_attacks(raw.get("attacks") or {}, ducks, float(goal_width))
    return Scenario(name=name, seed=seed, floor=floor, walls=walls, boxes=boxes, goal_width=float(goal_width),
                    balls=balls, ducks=ducks, persons=persons, pickables=pickables,
                    basket=basket, collision=collision, attacks=attacks, cove=cove,
                    physics_dt=physics_dt)


def _validate_attacks(raw: dict, ducks: list[Duck], goal_width: float) -> dict[str, str]:
    """Which mouth each team attacks, and the check that every team has ONE.

    Without `attacks` a duck attacks the mouth its spawn heading faces
    (`World.goal_for`), so a roster placed by hand with one duck turned round
    is a team attacking both goals at once. That used to surface as a
    `ValueError` out of `PitchMetrics` — raised AFTER the world had been
    swapped in, so the page answered 500 and went on streaming the old world's
    score. It belongs here, where the editor's save is refused with the duck's
    name in the message and nothing has been swapped anywhere."""
    if not isinstance(raw, dict):
        raise ScenarioError("attacks must be an object")
    teams = {d.team for d in ducks if d.team}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if k not in teams:
            raise ScenarioError(f"attacks[{k!r}]: no duck is on that team")
        if v not in ("left", "right"):
            raise ScenarioError(f"attacks[{k!r}] must be 'left' or 'right' (the goal MOUTH), got {v!r}")
        out[k] = v
    if len(set(out.values())) < len(out):
        raise ScenarioError("attacks: two teams cannot attack the same goal")
    if goal_width <= 0:
        return out
    for tm in sorted(teams - set(out)):
        facing = {d.id: (math.cos(d.spawn[2]) >= 0) for d in ducks if d.team == tm}
        first = next(iter(facing.values()))
        if len(set(facing.values())) > 1:
            odd = sorted(k for k, v in facing.items() if v != first)
            raise ScenarioError(
                f"team {tm!r} faces both goals ({', '.join(odd)} the other way), so which one it "
                f"attacks is undefined - turn the duck round or set attacks[{tm!r}]")
    return out


def load_scenario(path: Path) -> Scenario:
    return validate_scenario(json.loads(Path(path).read_text()))


# -- procedural rooms --------------------------------------------------------

def make_room(seed: int = 0, size: tuple[float, float] = (3.0, 2.5),
              n_boxes: int = 4, n_ducks: int = 1, name: str | None = None,
              wall_height: float = 0.3) -> Scenario:
    """A walled rectangle with a few boxes and ducks spawned clear of them.
    Deterministic in `seed`, so a lesson can say "room 7" and mean it."""
    import numpy as np

    rng = np.random.default_rng(seed)
    hx, hy = size[0] / 2, size[1] / 2
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    walls = [Wall(corners[i], corners[(i + 1) % 4], wall_height, 0.02) for i in range(4)]
    boxes: list[Box] = []
    placed: list[tuple[float, float, float]] = []   # x, y, clearance radius

    def free(x: float, y: float, r: float) -> bool:
        return all(math.dist((x, y), (px, py)) > r + pr for px, py, pr in placed)

    for _ in range(n_boxes):
        for _try in range(50):
            s = (float(rng.uniform(0.1, 0.4)), float(rng.uniform(0.1, 0.4)),
                 float(rng.uniform(0.08, 0.3)))
            x = float(rng.uniform(-hx + 0.4, hx - 0.4))
            y = float(rng.uniform(-hy + 0.4, hy - 0.4))
            r = math.hypot(s[0], s[1]) / 2
            if free(x, y, r + 0.25):
                boxes.append(Box((x, y, s[2] / 2), s, float(rng.uniform(0, math.pi))))
                placed.append((x, y, r))
                break
    ducks: list[Duck] = []
    for i in range(n_ducks):
        for _try in range(100):
            x = float(rng.uniform(-hx + 0.3, hx - 0.3))
            y = float(rng.uniform(-hy + 0.3, hy - 0.3))
            if free(x, y, 0.3):
                ducks.append(Duck(f"d{i}", (x, y, float(rng.uniform(-math.pi, math.pi)))))
                placed.append((x, y, 0.2))
                break
    return Scenario(name=name or f"room-{seed}", seed=seed, floor=(size[0] + 0.5, size[1] + 0.5),
                    walls=walls, boxes=boxes, ducks=ducks)


def make_playroom(seed: int = 0, n: int = 6, size: tuple[float, float] = (3.0, 2.5),
                  name: str | None = None, robot: str = "microduck",
                  brain: str = "tidy", kinds: tuple[str, ...] | None = None) -> Scenario:
    """A walled room with `n` toys scattered on the floor, a low basket in a
    corner, and one robot facing the mess. Deterministic in `seed`.

    `robot` / `brain` are what `eval-tidy --robot mars` builds the benchmark
    from: the same room, the same layouts, the same seeds, with a MARS in the
    middle running `tidy_arm` instead of a duck running `tidy`. The timestep
    comes from the BODY (`robot_physics_dt`) — a MARS room is 2 ms because its
    claw cannot hold a block at 5 ms, and a duck room stays at 5 ms so every
    `eval-tidy` number ever measured is still the number it was.

    `kinds` restricts which `PICKABLE_KINDS` are scattered; None cycles all
    three, which is what the duck's benchmark has always done.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    hx, hy = size[0] / 2, size[1] / 2
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    walls = [Wall(corners[i], corners[(i + 1) % 4], 0.3, 0.02) for i in range(4)]
    basket = Basket((hx - 0.35, hy - 0.35), (0.3, 0.3), 0.06)
    kind_list = list(kinds if kinds else PICKABLE_KINDS)
    toys: list[Pickable] = []
    for i in range(n):
        for _try in range(50):
            x = float(rng.uniform(-hx + 0.3, hx - 0.3))
            y = float(rng.uniform(-hy + 0.3, hy - 0.3))
            if math.dist((x, y), basket.pos) < 0.45 or math.dist((x, y), (0.0, 0.0)) < 0.35:
                continue
            if all(math.dist((x, y), t.pos) > 0.2 for t in toys):
                toys.append(Pickable(f"t{i}", kind_list[i % len(kind_list)], (x, y), float(rng.uniform(0, math.pi))))
                break
    return Scenario(name=name or f"playroom-{seed}", seed=seed, floor=(size[0] + 0.5, size[1] + 0.5),
                    walls=walls,
                    ducks=[Duck("d0", (0.0, 0.0, 0.0), None, "datasheet", "datasheet", brain,
                                robot=robot)],
                    pickables=toys, basket=basket, physics_dt=robot_physics_dt(robot))


def make_pitch(size: tuple[float, float] | None = None, name: str | None = None,
               goal_width: float = 0.7, per_side: int = 1,
               teams: tuple[str, str] = PITCH_TEAMS, formation: bool = False,
               cove: float = 0.0, corner: float = 0.0) -> Scenario:
    """`per_side` ducks a side, one ball, walls all round (the soccer track).
    A goal is the ball crossing either short wall's line inside
    `goal_width`; the World counts them and re-centres the ball. The CREAM
    team (d0…) spawns at −x and attacks the +x mouth; the GRAPHITE team
    attacks −x; the pitch grows a little with the roster. Teammates share a
    blackboard (brain/team.py) — a message a second over Wi-Fi on the
    robot — that says who attacks and where the ball was seen.

    `formation=True` stamps static jobs from `formation_roles` (the lab
    builtins `pitch-2v2` / `pitch-3v3`). Off, which is the default and what
    `eval-pitch` uses, so the chase-vs-chase control stays role-free.

    `cove` is the quarter-round along the base of the boards (`Scenario.
    cove`, the radius) and `corner` chamfers each corner at 45 degrees,
    starting that far along each wall from the corner point, so a ball
    cannot wedge in a square corner. Both 0 by default: the benchmark's
    pitch is bit for bit what it was.

    The teams are colorways and not "left"/"right" on purpose: those were
    the two SIDES, and the World writes its goal counts under the two
    MOUTHS with the same two words, so `goals["right"]` was the left
    team's tally and every per-side reading of a row was one slip away
    from being inverted (`eval_pitch` carries the warning that slip
    earned). A cream duck and the +x mouth cannot be confused."""
    per_side = max(1, int(per_side))
    if size is None:
        size = (3.0 + 0.4 * (per_side - 1), 2.5 + 0.35 * (per_side - 1))
    hx, hy = size[0] / 2, size[1] / 2
    pts = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    if corner > 0:
        # Chamfered corners: each side stops `corner` short of the corner
        # point and a diagonal joins it to the next, counter-clockwise, so
        # every wall still shares its endpoints with its neighbours (the
        # cove's mitres in compose.py find the neighbours that way).
        c = min(float(corner), hx - 0.05, hy - 0.05)
        pts = [(-hx + c, -hy), (hx - c, -hy), (hx, -hy + c), (hx, hy - c),
               (hx - c, hy), (-hx + c, hy), (-hx, hy - c), (-hx, -hy + c)]
    walls = [Wall(pts[i], pts[(i + 1) % len(pts)], 0.3, 0.02) for i in range(len(pts))]
    ducks = []
    ys = [0.0] if per_side == 1 else [(-0.5 + i / (per_side - 1)) * (hy - 0.5) * 1.4 for i in range(per_side)]
    home, away = teams
    jobs = formation_roles(per_side) if formation else [None] * per_side
    for i, y in enumerate(ys):
        x = 0.9 + 0.3 * (i % 2)                       # a little staggered, so nobody starts nose to nose
        ducks.append(Duck(f"d{i}", (-x, y, 0.0), None, "datasheet", "datasheet", "chase",
                          team=home, role=jobs[i]))
    for i, y in enumerate(ys):
        x = 0.9 + 0.3 * (i % 2)
        ducks.append(Duck(f"d{per_side + i}", (x, -y, math.pi), None, "datasheet", "datasheet", "chase",
                          team=away, role=jobs[i]))
    return Scenario(name=name or ("pitch" if per_side == 1 else f"pitch-{per_side}v{per_side}"),
                    floor=(size[0] + 0.5, size[1] + 0.5), walls=walls,
                    balls=[Ball((0.0, 0.0))], ducks=ducks, goal_width=goal_width, cove=float(cove))
