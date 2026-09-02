"""The scenario contract: a room, its objects, and the ducks in it.

Format v1 (JSON, saved under microduck_local/scenarios/<name>.json):

    {"version": 1, "name": "living-room", "seed": 0,
     "floor": {"size": [4.0, 3.0]},               # half-extents NOT — full x, y metres
     "walls": [{"from": [x, y], "to": [x, y], "height": 0.3, "thickness": 0.02}],
     "boxes": [{"pos": [x, y, z], "size": [sx, sy, sz], "yaw": 0.0,
                "mass": 0.0, "rgba": [r, g, b, a]}],     # mass 0 = static scenery
     "balls": [{"pos": [x, y], "radius": 0.035, "mass": 0.015}],
     "ducks": [{"id": "d0", "spawn": [x, y, yaw], "policy": "pollen:alpha_walking",
                "tof": "datasheet", "detector": "datasheet", "brain": "follow"}],
     "persons": [{"id": "p0", "pos": [x, y], "yaw": 0.0, "path": [[x, y], ...],
                  "speed": 0.3, "radius": 0.2, "height": 1.0}],  # kinematic walkers
     "pickables": [{"id": "t0", "kind": "brick"|"block"|"sock", "pos": [x, y], "yaw": 0.0}],
     "basket": {"pos": [x, y], "size": [0.3, 0.3], "rim": 0.06} | null,   # the tidy target
     "collision": "walk"}                               # "walk" | "all" robot MJCF

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
MAX_DUCKS = 12
MAX_PERSONS = 4
MAX_OBJECTS = 200
MAX_FLOOR_M = 20.0
MAX_WALL_HEIGHT_M = 2.0
TOF_PRESETS = ("ideal", "datasheet", "hostile")


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


@dataclass
class Duck:
    id: str
    spawn: tuple[float, float, float]  # x, y, yaw
    policy: str | None = None          # palette id; None = zero-action stand
    tof: str | None = "datasheet"      # ToF noise preset, None = no sensor
    detector: str | None = "datasheet" # camera+NPU detector preset, None = none
    brain: str | None = None           # brain kind in auto mode; None = wander if ToF else script
    odom: str = "ideal"                # odometry drift preset the brain's (x, y, yaw) carries (roadmap 1.7)
    team: str | None = None            # soccer: teammates share a blackboard (brain/team.py); None = alone


@dataclass
class Person:
    """A kinematic walker (a mocap capsule): what a duck follows."""
    id: str
    pos: tuple[float, float]
    yaw: float = 0.0
    path: list[tuple[float, float]] = field(default_factory=list)   # waypoints, looped
    speed: float = 0.3
    radius: float = 0.2
    height: float = 1.0


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
    collision: str = "walk"
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
            _num(b.get("mass", 0.015), f"balls[{i}].mass", 0.001, 5.0)))
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
        if brain is not None and (not isinstance(brain, str) or not DUCK_ID_RE.match(brain)):
            raise ScenarioError(f"ducks[{i}].brain must be a brain kind name or null")
        odom = d.get("odom", "ideal") or "ideal"
        if odom not in TOF_PRESETS:
            raise ScenarioError(f"ducks[{i}].odom must be one of {TOF_PRESETS}")
        team = d.get("team")
        if team is not None and (not isinstance(team, str) or not DUCK_ID_RE.match(team)):
            raise ScenarioError(f"ducks[{i}].team must be a short name or null")
        ducks.append(Duck(did, spawn, policy, tof, det, brain, odom, team))
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
        persons.append(Person(
            pid, _vec(q.get("pos", [0.0, 0.0]), 2, f"persons[{i}].pos", -bound, bound),
            _num(q.get("yaw", 0.0), f"persons[{i}].yaw", -2 * math.pi, 2 * math.pi),
            path,
            _num(q.get("speed", 0.3), f"persons[{i}].speed", 0.0, 1.5),
            _num(q.get("radius", 0.2), f"persons[{i}].radius", 0.05, 0.5),
            _num(q.get("height", 1.0), f"persons[{i}].height", 0.2, 2.0)))
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
    collision = raw.get("collision", "walk")
    if collision not in ("walk", "all"):
        raise ScenarioError("collision must be 'walk' or 'all'")
    goal_width = raw.get("goal_width", 0.0) or 0.0
    if not isinstance(goal_width, (int, float)) or not 0.0 <= goal_width <= 5.0:
        raise ScenarioError("goal_width must be a number in [0, 5]")
    return Scenario(name=name, seed=seed, floor=floor, walls=walls, boxes=boxes, goal_width=float(goal_width),
                    balls=balls, ducks=ducks, persons=persons, pickables=pickables,
                    basket=basket, collision=collision)


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
                  name: str | None = None) -> Scenario:
    """A walled room with `n` toys scattered on the floor, a low basket in a
    corner, and one duck facing the mess. Deterministic in `seed`."""
    import numpy as np

    rng = np.random.default_rng(seed)
    hx, hy = size[0] / 2, size[1] / 2
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    walls = [Wall(corners[i], corners[(i + 1) % 4], 0.3, 0.02) for i in range(4)]
    basket = Basket((hx - 0.35, hy - 0.35), (0.3, 0.3), 0.06)
    kinds = list(PICKABLE_KINDS)
    toys: list[Pickable] = []
    for i in range(n):
        for _try in range(50):
            x = float(rng.uniform(-hx + 0.3, hx - 0.3))
            y = float(rng.uniform(-hy + 0.3, hy - 0.3))
            if math.dist((x, y), basket.pos) < 0.45 or math.dist((x, y), (0.0, 0.0)) < 0.35:
                continue
            if all(math.dist((x, y), t.pos) > 0.2 for t in toys):
                toys.append(Pickable(f"t{i}", kinds[i % len(kinds)], (x, y), float(rng.uniform(0, math.pi))))
                break
    return Scenario(name=name or f"playroom-{seed}", seed=seed, floor=(size[0] + 0.5, size[1] + 0.5),
                    walls=walls, ducks=[Duck("d0", (0.0, 0.0, 0.0), None, "datasheet", "datasheet", "tidy")],
                    pickables=toys, basket=basket)


def make_pitch(size: tuple[float, float] | None = None, name: str | None = None,
               goal_width: float = 0.7, per_side: int = 1) -> Scenario:
    """`per_side` ducks a side, one ball, walls all round (the soccer track).
    A goal is the ball crossing either short wall's line inside
    `goal_width`; the World counts them and re-centres the ball. The left
    team ("left", d0…) attacks +x, the right team ("right") attacks −x;
    the pitch grows a little with the roster. Teammates share a
    blackboard (brain/team.py) — a message a second over Wi-Fi on the
    robot — that says who attacks and where the ball was seen."""
    per_side = max(1, int(per_side))
    if size is None:
        size = (3.0 + 0.4 * (per_side - 1), 2.5 + 0.35 * (per_side - 1))
    hx, hy = size[0] / 2, size[1] / 2
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    walls = [Wall(corners[i], corners[(i + 1) % 4], 0.3, 0.02) for i in range(4)]
    ducks = []
    ys = [0.0] if per_side == 1 else [(-0.5 + i / (per_side - 1)) * (hy - 0.5) * 1.4 for i in range(per_side)]
    for i, y in enumerate(ys):
        x = 0.9 + 0.3 * (i % 2)                       # a little staggered, so nobody starts nose to nose
        ducks.append(Duck(f"d{i}", (-x, y, 0.0), None, "datasheet", "datasheet", "chase", team="left"))
    for i, y in enumerate(ys):
        x = 0.9 + 0.3 * (i % 2)
        ducks.append(Duck(f"d{per_side + i}", (x, -y, math.pi), None, "datasheet", "datasheet", "chase", team="right"))
    return Scenario(name=name or ("pitch" if per_side == 1 else f"pitch-{per_side}v{per_side}"),
                    floor=(size[0] + 0.5, size[1] + 0.5), walls=walls,
                    balls=[Ball((0.0, 0.0))], ducks=ducks, goal_width=goal_width)
