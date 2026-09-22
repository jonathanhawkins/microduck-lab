"""World mode for the lab: the /sim page's backend (docs/sim-roadmap.md 0.4).

Mounted onto the lab's FastAPI app by `make_app`, beside the roster loop.
The roster (`/ws`) keeps its private-env ducks, teach jobs and captures; this
module owns ONE `World` — a scenario's room, its objects, N ducks in one
mjData — and streams it on a second socket so the two pages never fight
over a frame format.

HTTP:
  GET  /scenarios              [{name, builtin, ducks, robots, objects, modified}]
  GET  /scenarios/{name}       the scenario JSON (built-ins are generated)
  PUT  /scenarios/{name}       save a user scenario (validated; built-ins are read-only)
  DELETE /scenarios/{name}     remove a user scenario
  GET  /world                  {scenario, loading, ducks: [...]} — what is live now
  POST /world/load {"scenario": name}      compose + swap (a second or so;
                               the loop keeps streaming the old world meanwhile)
  POST /world/noise {"duck": id, "preset": "ideal"|"datasheet"|"hostile",
                     "sensor": "tof"|"lidar"|"det"|"odom"}
  POST /world/speed {"x": 0.25..8}         run the world that many times wall
                               speed (SPEED_CHOICES); `rtf` says what it got
  GET  /replay/ring?last=N     the last N frames the loop broadcast (a ring of
                               RING_S seconds at 25 Hz, kept whether or not a
                               browser is attached) — the page's scrub bar
  POST /replay/save {"name"}   write the ring to recordings/<name>.jsonl.gz
  GET  /recordings             [{name, frames, span, scenario, saved}]
  GET  /recordings/{name}      the frames of one recording (JSON array)
  DELETE /recordings/{name}

WS /ws/sim — 25 Hz frames. Speed does not change the frame RATE, only the
sim time between frames: a world at 4x jumps further per frame, it does not
send more of them. (A scene that saturates the box does drop frames — a 3v3
at 8x measured ~17 a second — because the loop is late, not because it is
fast.)
  {t, tick, rtf, simSpeed, perf: {stepMs, sensorMs}, scenario, cmd, mode, events,
   ducks: [{id, robot, name, policy, falls, step, rew, speed, cmdSpeed, steerable,
            brain: {kind, state, cmd, head, note, inputs: {tof|lidar: {age, stale, max}, det: {age, stale, n}, target?}},
            bodies: [[x,y,z,qw,qx,qy,qz] × 17] (world first, as GET /scene lists bodies),
            sensors: {tof: {t, mm[64], age}, det: {t, age, items: [{cls, name, bearing, elevation, width, range, conf}]}} | null}],
   objects: [{id, kind: "ball"|"box"|"person", pose, possessed?}], possessed: person id | null}

`robot` is which BODY the entry is (`world/scenario.Duck.robot`, always
present), and it is what decides how the rest of the row reads:

  "microduck"  the duck, exactly as above: 17 bodies in `GET /scene`'s order,
               `sensors.tof` 64 zones off the head, `falls` counts topples.
  anything     a driver-stepped body (`world/arena.WorldRobot`) — MARS today.
  else         `bodies` is ITS scene's list, in `GET /scene?robot=<id>` order
               (18 for a MARS, world first); `sensors.lidar` carries one
               360-ray scan (`+ minRange`, `footprint`, `maxRange`, `mount` —
               what a polar plot of it cannot be drawn honestly without) and
               there is NO `sensors.tof` (the 8x8 its brains read is adapted,
               and drawing it would be a sensor the robot does not have —
               `tof_payload`); `brain.inputs` reports its range freshness
               under `lidar` and not `tof`, for the same reason
               (`brain/runtime.age_inputs`); a body with a HAND also ships
               `sensors.gripper {load, holding, limit, hold}` and
               `sensors.arm {q, cmd, limits}`, whose `q` vs `cmd` gap is the
               servo lag a brain pre-compensates (`gripper_payload` /
               `arm_payload`); `falls` is always 0, because a planar base
               cannot topple; `steerable` is true and WASD drives it through
               the same `cmd`.
accepts:
  {"cmd": [vx, vy, wz]}   drive every duck (held OVERRIDE_HOLD_S); otherwise a duck with
                          a ToF wanders on the brain layer's Wander controller and a
                          blind duck follows the demo script
  {"reset": true}         respawn everything
  {"assign": {"duck": id, "policy": palette id}}
  {"noise": {"duck": id, "preset": name, "sensor": "tof"|"lidar"|"det"|"odom"}}
                          "tof" is the RANGE channel (the scenario's own field
                          name): on a body whose range sensor is the scanner it
                          sets the LIDAR's preset, and "lidar" is an alias
  {"brain": {"duck": id, "kind": "wander"|"follow"|"script"}}   swap a duck's brain
  {"possess": person id | null}   your cmd drives that person (ducks stay on their brains)
  {"head": {"duck": id, "apply": bool}}   let a brain's gaze intent reach the walker's command block
  {"speed": 0.25..8}      wall-clock speed of the world (same as POST /world/speed)

Scenario files live in microduck_local/scenarios/ (MICRODUCK_SCENARIOS_DIR
relocates it). Built-ins are generated in code so a fresh checkout has
something to load, and they cannot be overwritten — save under another name.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import math
import os
import re
import time
import traceback
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel

from .brain import REGISTRY, Intent, Senses
from .brain import runtime as brain_runtime
from .brain import tidy as _tidy  # noqa: F401  (registers the tidy brain)
from .brain import tidy_arm as _tidy_arm  # noqa: F401  (…and the arm's, for a MARS)
from .brain.graph import payload as graph_payload
from .brain.learned import learned_index
from .brain.mapping import GridSpec, OccupancyGrid
from .brain.tether import Tether
from .sensors import DetectorNoise, LidarNoise, TofNoise
from .world import (
    DUCK_ROBOT,
    Ball,
    Duck,
    Person,
    Scenario,
    Wall,
    World,
    WorldRobot,
    make_pitch,
    make_playroom,
    make_room,
)
from .world.scenario import NAME_RE, TOF_PRESETS, ScenarioError, validate_scenario

# The lab's pitches have the COVE (roadmap Track 4 item 14): a 15 cm
# quarter-round along the base of the boards and a 45-degree chamfer 30 cm
# across each corner, so a ball rolling into a wall climbs it and rolls back
# out. The sim's wall is otherwise dead (MuJoCo models no restitution: e =
# 0.06 where a real hollow ball is 0.5-0.7), and the ball-out rule below was
# the referee that patched it. Measured 2026-09-09, 2v2 on the same seeds:
# the cove alone takes dead-ball time 89 -> 65% and kicks 1.2 -> 3.7 a run,
# which is what the referee did (66%, 3.5) - with no teleport. So the referee
# is OFF on the lab's pitches, which now play exactly the "cove alone" arm.
# `World.ball_out_s` stays a knob for eval-pitch (`--ball-out-s`), whose
# baseline pitch is still flat and square (`--cove`, `--corner` opt in), so
# no published number moves. The viewer draws the cove (SimStage.tsx).
PITCH_COVE = 0.15
PITCH_CORNER = 0.30
PITCH_BALL_OUT_S = 0.0

# …and a fallen duck on a pitch GETS UP instead of vanishing and reappearing
# (roadmap B.1). Without this the lab respawns it the instant it falls, which
# is the one thing on /sim that is plainly not what a robot does. The shipped
# `alpha_stand` recovers from lying on the back, front and side 100% of the
# time on the bench, and 3 of 3 real falls over 24 runs of 3v3, with the
# ledger flat (possession, advance, progress all unmoved) — so this is honesty
# on the page bought for nothing measurable. `PITCH_GETUP_S` is the TIMEOUT: a
# duck that cannot make it up in that long is respawned as before. Batteries
# are unaffected — `eval-pitch` still defaults to the teleport, and takes
# `--getup-policy` to opt in — so no published number moves.
PITCH_GETUP_S = 5.0
PITCH_GETUP_POLICY = "pollen:alpha_stand"

TICK_HZ = 50
SEND_EVERY = 2
MAP_EVERY = 12               # occupancy maps ride every 12th frame (~2 Hz): 3–4 kB each per duck
# How fast the world runs against the wall clock. The multiplier is a
# SIM-TIME BUDGET, not a faster wall clock: one 20 ms wall tick buys `speed`
# ticks of sim and the wire keeps its 25 Hz, so 4x is four steps between
# frames (a bigger jump each frame) and 0.5x is a step every other tick.
# Everything inside the sim already lives on `World.t` — the sensors, the
# tether, the throw-in and get-up clocks, every brain — so none of it can
# tell. Only the manual-drive hold (OVERRIDE_HOLD_S) stays on the wall,
# where the hand that set it is.
# The ceiling is what the loop body costs: measured here at 0.2–0.7 ms a
# tick for a room and 6.6 ms for a 3v3 pitch, against a 20 ms budget. So 8x
# is real in a room and a 3v3 tops out near 3x. Asking for more is not an
# error — the loop runs flat out and the frame's `rtf` says what it got.
SPEED_CHOICES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SPEED_MIN, SPEED_MAX = SPEED_CHOICES[0], SPEED_CHOICES[-1]
# A loop that fell behind must not BANK the lost wall time, or it would
# sprint to burn the debt the moment the scene got cheap again.
MAX_LAG_S = 0.25
RTF_WINDOW_S = 1.0           # wall seconds the measured speed is averaged over
# …and a batch of sim ticks must come back to the wire. Left unbounded, 8x on
# a 3v3 runs eight ~7 ms steps before it looks at the socket again, and the
# browser gets 10 frames a second: a fast-forward nobody can watch. Capping
# the batch at its own wall slot is not a trade — measured over repeats on
# pitch-3v3, it buys 17 fps against 10 at 8x and changes the achieved speed
# at 2x, 4x and 8x by less than the run-to-run spread.
STEP_BUDGET_S = 1.0 / TICK_HZ
OVERRIDE_HOLD_S = 6.0
RING_S = 120.0                       # the scrub bar reaches this far back at 1x
                                     # (frames in which the world MOVED, so a slow
                                     #  world does not pad it with repeats)
RING_FRAMES = int(RING_S * TICK_HZ / SEND_EVERY)
RECORDING_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
DEFAULT_POLICY = "pollen:alpha_walking"
# A gentle drive script for the auto mode: walk, turn, stand, repeat.
DEMO_SCRIPT: list[tuple[float, tuple[float, float, float]]] = [
    (4.0, (0.3, 0.0, 0.0)),
    (2.0, (0.0, 0.0, 0.8)),
    (3.0, (0.3, 0.0, 0.0)),
    (2.0, (0.0, 0.0, 0.0)),
]

Infer = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Composed:
    """A world and everything derived from it, built off to one side.

    `compose()` fills this from a worker thread and `install()` publishes it
    in one block. Keeping them apart is what stops the running loop seeing a
    new brain set against the old world for the tick that a load takes.
    """
    world: World
    scenario: Scenario
    brains: dict
    teams: dict
    maps: dict
    goal_seq: int
    out_seq: int


def clamp_speed(x: float) -> float:
    """Any number the page or a script sends, brought into SPEED_CHOICES'
    range. Values between the presets are allowed — the presets are what the
    UI offers, not what the loop can run.

    Junk is REFUSED rather than clamped, and always as ValueError so a caller
    has one thing to catch. NaN is the reason: every comparison against it is
    False, so `max(SPEED_MIN, min(nan, SPEED_MAX))` quietly returns SPEED_MIN
    — a bad value would read back as a deliberate request for quarter speed.
    An integer too large for a float (400 digits of JSON) raises OverflowError
    out of `float()`, which is the same class of mistake, so it arrives the
    same way instead of escaping and tearing down the socket."""
    try:
        x = float(x)
    except OverflowError as e:
        raise ValueError(f"speed out of range: {e}") from e
    if not math.isfinite(x):
        raise ValueError(f"speed must be a finite number, got {x!r}")
    return float(max(SPEED_MIN, min(x, SPEED_MAX)))


def recordings_dir() -> Path:
    return Path(os.environ.get(
        "MICRODUCK_RECORDINGS_DIR",
        Path(__file__).resolve().parents[2] / "recordings"))


def scenarios_dir() -> Path:
    return Path(os.environ.get(
        "MICRODUCK_SCENARIOS_DIR",
        Path(__file__).resolve().parents[2] / "scenarios"))


# -- built-in scenarios -------------------------------------------------------

def _g1_person_kw() -> dict:
    """Follow scenes use the Unitree G1 when its MJCF+ONNX have been fetched."""
    from .robots.g1 import g1_ready
    if g1_ready():
        return {"kind": "g1", "height": 1.32, "radius": 0.25, "yield_m": 0.45}
    return {}


def builtin_scenarios() -> dict[str, Scenario]:
    empty = Scenario(
        name="empty-floor", floor=(6.0, 6.0),
        ducks=[Duck(f"d{i}", (0.0, 0.6 * i - 0.6, 0.0), DEFAULT_POLICY, "datasheet")
               for i in range(3)])
    wall = Scenario(
        name="wall-test", floor=(6.0, 6.0),
        walls=[Wall((1.0, -1.5), (1.0, 1.5), 0.6, 0.02)],
        ducks=[Duck("d0", (0.0, 0.0, 0.0), None, "ideal")])
    room = make_room(seed=1, size=(3.0, 2.5), n_boxes=4, n_ducks=2, name="living-room")
    room.balls.append(Ball((0.0, 0.0)))
    for d in room.ducks:
        d.policy = DEFAULT_POLICY
    fx, fy = 3.0, 2.5
    g1_kw = _g1_person_kw()
    # Duck rooms are 30 cm walls. A 1.32 m G1 walks through that ceiling and
    # looks like a chimney; raise the walls when the G1 is the person.
    wall_h = 2.4 if g1_kw else 0.3
    follow = Scenario(
        name="follow-me", floor=(6.5, 5.5),
        walls=[Wall((-fx, -fy), (fx, -fy), wall_h), Wall((fx, -fy), (fx, fy), wall_h),
               Wall((fx, fy), (-fx, fy), wall_h), Wall((-fx, fy), (-fx, -fy), wall_h)],
        # The shipped follower, not the rule brain: the scene exists to watch
        # a brain follow a person, and this is the brain that goes on the
        # robot (README "the follow pick"; swap to "follow" in the inspector
        # to compare). If this clone lacks brains/follow-v4, load_world's
        # fallback below runs the duck on "script" and says so in events.
        ducks=[Duck("d0", (0.0, 0.0, 0.0), DEFAULT_POLICY, "datasheet", "datasheet", "learned:follow-v4")],
        persons=[Person("p0", (1.2, 0.0), 1.57,
                        path=[(1.2, 1.2), (-1.2, 1.2), (-1.2, -1.2), (1.2, -1.2)],
                        speed=0.5 if g1_kw else 0.25, **g1_kw)])
    playroom = make_playroom(seed=0, n=6, name="playroom")
    playroom.ducks[0].policy = DEFAULT_POLICY
    pitch = make_pitch(name="pitch", cove=PITCH_COVE, corner=PITCH_CORNER)
    pitch2 = make_pitch(name="pitch-2v2", per_side=2, formation=True, cove=PITCH_COVE, corner=PITCH_CORNER)
    pitch3 = make_pitch(name="pitch-3v3", per_side=3, formation=True, cove=PITCH_COVE, corner=PITCH_CORNER)
    for d in pitch2.ducks + pitch3.ducks:
        d.policy = DEFAULT_POLICY
    for d in pitch.ducks:
        d.policy = DEFAULT_POLICY
    return {s.name: s for s in (empty, wall, room, follow, playroom, pitch, pitch2, pitch3)}


BUILTIN_NAMES = frozenset(builtin_scenarios().keys())


def _robot_counts(sc: Scenario) -> list[dict]:
    """What BODIES a scenario holds: `[{id, n, noun}]`, commonest first.

    Beside `ducks`, which stays the total and stays named that for every
    caller that already reads it. The count is what a menu can honestly show:
    `mars-follow` holds one MARS and no duck at all, and the /sim scenario
    picker called it "1 ducks" until this existed.

    `robot_noun` is imported inside the function for `world/scenario.py`'s
    reason — listing rooms must not drag a robot package (and its MJCF) in
    on import.
    """
    from .lab.robots import robot_noun
    counts: dict[str, int] = {}
    for duck in sc.ducks:
        rid = getattr(duck, "robot", "") or "microduck"
        counts[rid] = counts.get(rid, 0) + 1
    return [{"id": rid, "n": n, "noun": robot_noun(rid)}
            for rid, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def _no_housing() -> int:
    """`camera_housing_body` for an entry that does not answer one.

    Only `WorldRobot` does today. A `WorldDuck` has a housing too — its
    detector is site-mounted, so the housing is `jaw_soft`, the bill, and
    MEASURED it IS inside the duck's own 116x60 frame. It is 0.94 cm from
    the lens, which is a third of the /sim inset's 3 cm near plane, so it is
    clipped with room to spare where MARS's shell at 1.5-2.8 cm is not. The
    duck's body list does not run through `scene_bodies` (it is the fixed
    17-body duck scene), so wiring an index for it is work with nothing to
    show; it is deferred, not overlooked.

    The viewer reads -1 as "draw everything", which is what it did before
    this field existed and what an older lab still sends.
    """
    return -1


def list_scenarios() -> list[dict]:
    out = []
    for name, sc in builtin_scenarios().items():
        out.append({"name": name, "builtin": True, "ducks": len(sc.ducks),
                    "robots": _robot_counts(sc),
                    "objects": len(sc.walls) + len(sc.boxes) + len(sc.balls),
                    "modified": None})
    d = scenarios_dir()
    if d.exists():
        for p in sorted(d.glob("*.json")):
            try:
                sc = validate_scenario(json.loads(p.read_text()))
            except (ScenarioError, ValueError, OSError):
                continue
            out.append({"name": p.stem, "builtin": p.stem in BUILTIN_NAMES,
                        "ducks": len(sc.ducks),
                        "robots": _robot_counts(sc),
                        "objects": len(sc.walls) + len(sc.boxes) + len(sc.balls),
                        "modified": p.stat().st_mtime})
    return out


def resolve_scenario(name: str) -> Scenario:
    if not NAME_RE.match(name or ""):
        raise HTTPException(400, f"bad scenario name {name!r}")
    b = builtin_scenarios().get(name)
    if b is not None:
        return b
    p = scenarios_dir() / f"{name}.json"
    if not p.exists():
        raise HTTPException(404, f"no scenario {name!r}")
    try:
        return validate_scenario(json.loads(p.read_text()))
    except (ScenarioError, ValueError) as e:
        raise HTTPException(422, str(e)) from e


# -- state -----------------------------------------------------------------------

class WorldState:
    def __init__(self, load_infer: Callable[[str], Infer] | None):
        self.load_infer = load_infer
        self.world: World | None = None
        self.scenario: Scenario | None = None
        self.metrics = None                  # PitchMetrics on a pitch, else None
        self.clients: set[WebSocket] = set()
        self.override: np.ndarray | None = None
        self.override_until = 0.0
        self.script_t = 0.0
        self.events: deque[str] = deque(maxlen=200)
        self.loading = False
        self.rtf = 0.0
        self.maps: dict[str, OccupancyGrid] = {}
        self.send_maps = False
        # Roadmap 12.10: the brain tier "over a tether" — every intent lands
        # this long after the senses it came from. 0 = onboard.
        self.tether_ms = 0.0
        self._tether_queue: dict[str, Tether] = {}
        # Wall-clock speed of the world, and the fractional tick carried
        # between wall ticks so 0.25x steps once every fourth of them.
        self.speed = 1.0
        self._step_credit = 0.0
        self._ring_tick = -1         # world tick of the last frame kept for the scrub bar
        self._rtf_dirty = False      # a speed change restarts the rtf window
        self.task: asyncio.Task | None = None
        # Auto mode: each duck runs a brain from the registry (brain/runtime.py);
        # a blind duck gets the script. Intents are remembered for the frame.
        self.brains: dict[str, object] = {}
        self.teams: dict[str, object] = {}
        self.goal_seq = 0                    # World.goal_seq last acted on (kickoff_brains)
        self.out_seq = 0                     # …and World.ball_out_seq (throw_in_brains)
        self.intents: dict[str, Intent] = {}
        # Brains may ask for a head pose; the shipped walker never trained
        # with one (roadmap 3.7), so gaze intents are REPORTED but only
        # applied to ducks that opt in.
        self.head_cmds: set[str] = set()
        # Every broadcast frame, serialised, newest last. Kept without a
        # browser attached so "what just happened?" has an answer after the
        # fact — the roadmap's record/replay primitive (0.6).
        self.ring: deque[str] = deque(maxlen=RING_FRAMES)

    def current_cmd(self, now: float) -> tuple[np.ndarray, str]:
        if self.override is not None and now < self.override_until:
            return self.override, "manual"
        total = sum(s for s, _ in DEMO_SCRIPT)
        t = self.script_t % total
        for dur, cmd in DEMO_SCRIPT:
            if t < dur:
                return np.array(cmd, np.float32), "auto"
            t -= dur
        return np.zeros(3, np.float32), "auto"

    def infer_for(self, policy_id: str | None) -> Infer | None:
        if not policy_id or self.load_infer is None:
            return None
        try:
            return self.load_infer(policy_id)
        except Exception as e:  # a missing checkout, a bad id: stand instead
            self.events.append(f"{policy_id}: {type(e).__name__} — duck will stand")
            return None

    def compose(self, scenario: Scenario, seed: int | None = None) -> Composed:
        """Blocking: the world, its policies, its brains and its boards, built
        WITHOUT touching `self`. Call from a thread; hand the result to
        `install()` on the loop. `seed` pins the world's RNG (record-world
        replays a scenario headlessly with it); the lab leaves it to the
        scenario."""
        infer = {}
        for d in scenario.ducks:
            f = self.infer_for(d.policy)
            if f is not None:
                infer[d.id] = f
        world = World(scenario, infer_for=infer, seed=seed)
        if world.goal_width > 0:
            world.ball_out_s = PITCH_BALL_OUT_S      # the referee's throw-in (11b) - 0 since the cove (item 14)
            # A real get-up, when the policy is there. `infer_for` already
            # degrades to None on a missing checkout, which is exactly the
            # teleport this replaces — so a lab without the shipped policies
            # behaves as it always did.
            getup = self.infer_for(PITCH_GETUP_POLICY)
            if getup is not None:
                world.getup_infer, world.getup_s = getup, PITCH_GETUP_S
        # `build` runs in a worker thread (POST /world/load) while world_loop
        # is still driving the OLD world, so nothing here may be published
        # half-finished: the loop iterates `self.brains` every tick, and
        # filling it in place raced `throw_in_brains`/`kickoff_brains` into
        # "dictionary changed size during iteration" — raised inside the loop,
        # which has no handler, so /sim stopped streaming until a restart.
        # Build into LOCALS and hand them back for the caller to install in
        # one go — `install()`. Nothing here touches `self`, so the running
        # loop cannot see any of the new world's state against the old world.
        brains: dict[str, object] = {}
        teams: dict = {}                 # …the NEW world's boards, not the live ones
        for sd in scenario.ducks:
            kind = sd.brain or ("wander" if sd.tof is not None else "script")
            try:
                brains[sd.id] = self.make_brain(kind, sd, world, teams)
            except ValueError as e:
                self.events.append(f"{sd.id}: {e}; using script")
                brains[sd.id] = REGISTRY.make("script")
        # Room mapping (roadmap 4.x first step): an occupancy grid per duck
        # in ITS odometry frame, from its ToF frames — never from the sim.
        fx, fy = scenario.floor
        maps = {sd.id: OccupancyGrid(GridSpec(size=(fx + 1.0, fy + 1.0)))
                for sd in scenario.ducks if sd.tof is not None}
        return Composed(world=world, scenario=scenario, brains=brains, teams=teams,
                        maps=maps, goal_seq=world.goal_seq, out_seq=world.ball_out_seq)

    def install(self, c: Composed) -> World:
        """Make a composed world the live one. Every field the loop reads is
        rebound here with no await in between, so a tick sees either the whole
        old world or the whole new one — never a new brain set against an old
        world, which fired phantom throw-ins and folded the new room's
        occupancy grids with the previous room's ToF frames."""
        self.world, self.scenario = c.world, c.scenario
        self.goal_seq, self.out_seq = c.goal_seq, c.out_seq
        self.teams, self.brains, self.maps = c.teams, c.brains, c.maps
        self.intents = {}
        self.restart()                   # the new world starts its clock at zero
        return c.world

    def build(self, scenario: Scenario, seed: int | None = None) -> World:
        """Compose AND install, for the synchronous callers (preload,
        record-world) that have no loop running to race with."""
        return self.install(self.compose(scenario, seed=seed))

    def make_brain(self, kind: str, sd, world, teams: dict | None = None):
        """A brain for one duck; on a pitch a `chase` gets its goal and team.

        `teams` is the blackboard registry the duck JOINS — `brain_kwargs`
        does a `setdefault` into it, so it is an out-parameter as much as an
        in-one. A live swap passes None and joins the running boards; a world
        under construction passes its own dict, so the old world's boards are
        neither read nor written from the worker thread."""
        from .brain.runtime import attach_world
        from .brain.team import brain_kwargs
        spec = replace(sd, brain=kind)
        brain = REGISTRY.make(kind, **brain_kwargs(spec, world, self.teams if teams is None else teams))
        # A brain that declared a `world` / `robot_id` slot gets them
        # (`brain/runtime.attach_world`): `tidy_arm` reads the BASKET's rim
        # height from the scenario rather than carrying a constant, because a
        # deeper tray is a different place-over-the-rim target.
        return attach_world(brain, world, sd.id)

    def new_metrics(self):
        """The pitch's continuous metrics for the world just built, or None.
        The page shows what the benchmark judges by (world/metrics.py):
        goals are ~2.5 a run and cannot resolve anything, so a viewer
        watching for a difference needs the same instruments eval-pitch
        uses. Same class, so a number on screen IS the battery's number."""
        w, sc = self.world, self.scenario
        if w is None or sc is None or w.goal_width <= 0 or not sc.balls:
            return None
        from .world.metrics import PitchMetrics
        return PitchMetrics(w, {d.id: (d.team or d.id) for d in sc.ducks})

    def preload(self, name: str, seed: int | None = None) -> None:
        """Build a world before serving (the CLI's --world). Blocking."""
        sc = resolve_scenario(name)
        self.build(sc, seed=seed)        # installs, including the metrics
        print(f"[sim] world {sc.name}: {len(sc.ducks)} ducks", flush=True)

    def payload(self) -> dict:
        w = self.world
        return {
            "scenario": self.scenario.to_dict() if self.scenario else None,
            "loading": self.loading,
            "ducks": [duck_info(w, d, self.brains) for d in w.ducks.values()] if w else [],
            "presets": list(TOF_PRESETS),
            "brains": REGISTRY.available(),
            # The same learned brains with their titles and groups, so the
            # inspector's menu can file 49 runs under six headings instead of
            # listing p-batch-s14 next to p-batch-s13.
            "learned": learned_index(),
            # The state graphs the inspector draws (brain/graph.py): nodes
            # and what each one means, for every brain kind. Static, so it
            # rides the world-info message and never the frame — the frame
            # carries only which graph a duck is on, and its state.
            "graphs": graph_payload(),
        }

    def senses_for(self, d) -> Senses:
        """What a brain is handed this tick, for either kind of body.

        `World.senses_tof` is the one place that decides where the 8x8 comes
        from: a duck's own sensor, or a planar scan adapted for a wheeled
        body. `lidar` rides beside it for a brain that wants the whole turn."""
        w = self.world
        tof, tof_age = w.senses_tof(d)
        det = d.detector.last if d.detector is not None else None
        lidar = getattr(d, "lidar", None)
        lf = None if lidar is None else lidar.last
        arm_fn = getattr(d, "arm_qpos", None)
        return Senses(t=w.t, tof=tof, tof_age=tof_age,
                      det=det, det_age=None if det is None else w.t - det.t,
                      lidar=lf, lidar_age=None if lf is None else w.t - lf.t,
                      speed=d.heading_speed(w.data),
                      odom=w.odom(d),
                      # The ACHIEVED joint positions of a body with an arm
                      # (`Senses.arm`): a commanded pose is not the pose, and
                      # the 28.5 mm that costs is on that field.
                      arm=None if arm_fn is None else arm_fn(w.data),
                      holding=d.holding is not None, skill=d.skill, bumped=w.bumped(d))

    def drive(self, cmd: np.ndarray, mode: str) -> None:
        """Set every duck's command for this tick. A possessed person takes
        the manual command instead of the ducks; otherwise manual overrides
        every duck, and in auto each duck runs its brain (script = the demo)."""
        w = self.world
        if w is None:
            return
        possessed = next((p for p in w.persons.values() if p.possessed), None)
        if possessed is not None:
            possessed.cmd = cmd if mode == "manual" else None
        manual_ducks = mode == "manual" and possessed is None
        for d in w.ducks.values():
            brain = self.brains.get(d.id)
            if manual_ducks or brain is None or brain.kind == "script":
                d.set_cmd(w.data, cmd if (manual_ducks or brain is None or possessed is None) else cmd)
                self.intents[d.id] = Intent(twist=tuple(float(v) for v in cmd))
                continue
            if self.tether_ms > 0:
                # Over the tether (brain/tether.py): the senses the brain gets
                # are half a round trip old, the intent lands half a round
                # trip later - what a link does, and what lets a brain read
                # its own latency off its sensor ages.
                th = self._tether_queue.get(d.id)
                if th is None or abs(th.delay - self.tether_ms / 1000.0) > 1e-9:
                    th = self._tether_queue[d.id] = Tether(self.tether_ms / 1000.0)
                intent = th.intent_out(brain.step(th.senses_in(self.senses_for(d))), w.t)
            else:
                intent = brain.step(self.senses_for(d))
            self.intents[d.id] = intent
            w.apply_intent(d, intent)
            if d.skill is None:              # a running skill owns the command block
                d.set_cmd(w.data, intent.twist,
                          intent.head if head_applied(d, brain, self.head_cmds) else None)

    def after_step(self) -> None:
        """A goal restarts play: the World moved everyone (World.kickoff);
        the brains and the team boards forget their plans here. A THROW-IN
        (`ball_out_seq`) is the lighter case — the referee moved only the
        ball, so the ball beliefs go and nothing else does."""
        w = self.world
        if w is None:
            return
        if w.ball_out_seq != self.out_seq:
            from .brain.team import throw_in_brains  # noqa: PLC0415
            self.out_seq = w.ball_out_seq
            throw_in_brains(self.brains, self.teams)
        if w.goal_seq == self.goal_seq:
            return
        from .brain.team import kickoff_brains
        self.goal_seq = w.goal_seq
        kickoff_brains(self.brains, self.teams, w)
        for th in self._tether_queue.values():
            th.clear()
        self.intents.clear()
        self.events.append(f"GOAL {w.last_goal} — kickoff")

    def brain_payload(self, d, mode: str) -> dict:
        possessed = any(p.possessed for p in self.world.persons.values()) if self.world else False
        eff = "auto" if (mode == "manual" and possessed) else mode
        return brain_runtime.payload(self.brains.get(d.id), self.intents.get(d.id), eff)

    def set_brain(self, duck_id: str, kind: str) -> None:
        w = self.world
        if w is None or duck_id not in w.ducks:
            raise KeyError(duck_id)
        sd = next((d for d in self.scenario.ducks if d.id == duck_id), None) if self.scenario else None
        self.brains[duck_id] = self.make_brain(kind, sd, w) if sd is not None else REGISTRY.make(kind)
        self.events.append(f"{duck_id} brain → {kind}")

    def override_hold_s(self) -> float:
        """How long a manual command holds, in WALL seconds — scaled so its
        cost in SIM seconds does not.

        `drive()` skips `brain.step()` outright while the override stands, so
        the hold is not just "your twist persists": it is "no brain runs".
        At a flat 6 wall seconds that was 6 sim seconds at 1x and 48 at 8x —
        16% of a 300 s pitch run with every Chase suspended, and a 48 s jump
        in `senses.t` handed to each brain on resume.

        The hold is therefore bounded by BOTH clocks: never more than
        OVERRIDE_HOLD_S of wall time (or slow motion would leave you unable
        to hand the ducks back for half a minute) and never more than
        OVERRIDE_HOLD_S of sim time (or fast-forward would suspend the
        brains for most of a match). Above 1x the wall bound is the loose
        one, below 1x the sim bound is — so the cost is
        `min(OVERRIDE_HOLD_S, OVERRIDE_HOLD_S * speed)` sim seconds.
        """
        return OVERRIDE_HOLD_S / max(self.speed, 1.0)

    def restart(self) -> None:
        """Everything keyed to the world's clock, dropped because that clock
        just went back to zero. Each of these outlived a reset:

        - `metrics` has no reset() and `PitchMetrics.row()` scales by
          `60 / w.t`, so one frame after R a carried-over 12 s of possession
          printed as ~7e11 per minute against a 0-0 scoreboard.
        - `_tether_queue` holds senses and intents stamped in the OLD clock.
          They are due hundreds of seconds in the future, so nothing ever
          pops: every brain keeps being handed the pre-reset frame (whose
          negative age reads as FRESH) and the pre-reset intent.
        - `ring` is the scrub bar's history. Kept across a reset it makes
          `/replay/save` write two runs under one header, with `t` jumping
          backwards in the middle and a span computed across the seam.
        """
        self.metrics = self.new_metrics()
        self._tether_queue.clear()
        self.ring.clear()
        self._ring_tick = -1
        # The team blackboards keep deadlines in the clock that just moved:
        # a board holding `_kick_until = 133` from a goal at t=120 still
        # answers `waits(t)` true after a reset puts t back to 0, so the
        # scoring side stands off for the next two sim minutes of the new run.
        for tm in self.teams.values():
            tm.reset()
        # …and the occupancy grids are drawn in a room that may not be there
        # any more. (The websocket reset also clears these; a second reset of
        # a freshly-built grid costs nothing and means /world/load cannot
        # forget.)
        for g in self.maps.values():
            g.reset()
        # Resync the sequence counters with the world we are now on. Stale
        # ones fire a phantom throw-in or kickoff on the first tick after the
        # clock moves: World.reset() zeroes ball_out_seq but not goal_seq.
        w = self.world
        if w is not None:
            self.goal_seq, self.out_seq = w.goal_seq, w.ball_out_seq

    def set_speed(self, x: float) -> float:
        """Run the world faster or slower than the wall clock. Sticky across
        a restart and a new scenario, exactly like the tether — it is a knob
        on the bench, not a property of the world."""
        was, self.speed = self.speed, clamp_speed(x)
        if self.speed == was:
            # A no-op — and it MUST stay one. A held `[` re-sends the same
            # speed every ~30 ms; 0.25x needs four wall ticks (80 ms) of
            # credit to buy a single step, so zeroing the credit here stopped
            # the world dead for as long as the key was down.
            return self.speed
        self._step_credit = 0.0
        # Re-price a hold that is already running. The deadline was set in
        # wall time at the OLD speed, so leaving it alone let a change after
        # the keypress restore exactly what override_hold_s() bounds: tap W
        # at 1x then go to 8x and the remaining ~6 wall seconds became 48 sim
        # seconds with every brain suspended.
        now = time.monotonic()
        if self.override_until > now:
            left = (self.override_until - now) * max(was, 1.0) / max(self.speed, 1.0)
            self.override_until = now + min(left, self.override_hold_s())
        # The rtf window spans the change, so it now measures neither speed.
        # Drop it: a world with no measurement reports 0, which is what the
        # page reads as "no number yet" rather than as a shortfall.
        self.rtf, self._rtf_dirty = 0.0, True
        self.events.append(f"speed {self.speed:g}x")
        return self.speed

    def frame(self, cmd: np.ndarray, mode: str) -> dict:
        w = self.world
        ducks = []
        if w is not None:
            for d in w.ducks.values():
                ducks.append({
                    **duck_info(w, d, self.brains),
                    "brain": self.brain_payload(d, mode),
                    "headApplied": head_applied(d, self.brains.get(d.id), self.head_cmds),
                    # Body 0 is the WORLD in the viewer's scene (GET /scene),
                    # so a duck's 16 bodies ride behind one identity pose and
                    # the same Duck renderer works on both pages. 16, not 15:
                    # `split_jaw` hangs the hinged `mouth` body off the head in
                    # BOTH models, and this list is mapped onto /scene's
                    # positionally (tests/test_arena.py locks the order).
                    #
                    # A non-duck body answers for itself, in ITS scene's body
                    # order (`world/arena.WorldRobot.bodies_payload` — mapped
                    # onto `GET /scene?robot=<id>` the same way, which
                    # `viz_server._robot_scene_to_env` resolves by name).
                    "bodies": (d.bodies_payload(w.data) if isinstance(d, WorldRobot)
                               else [[0, 0, 0, 1, 0, 0, 0]] + w.duck_pose(d.id)),
                    "sensors": tof_payload(w, d),
                })
        return {
            "t": round(w.t, 3) if w else 0.0,
            "tick": w.tick if w else 0,
            "rtf": round(self.rtf, 2),
            "perf": ({k: round(v, 3) for k, v in w.perf.items()} if w else None),
            "scenario": self.scenario.name if self.scenario else None,
            "loading": self.loading,
            "cmd": [round(float(v), 3) for v in cmd],
            "mode": mode,
            "events": list(self.events)[-5:],
            "ducks": ducks,
            "objects": (w.objects_payload() + w.persons_payload()) if w else [],
            "tidy": w.tidy_score() if (w and w.pickables) else None,
            "soccer": ({**w.soccer_score(), **(self.metrics.row() if self.metrics else {})}
                       if (w and w.soccer_score()) else None),
            "maps": ({k: g.payload() for k, g in self.maps.items()} if (w and self.send_maps) else None),
            "tetherMs": self.tether_ms,
            # What was ASKED for. What the box managed is `rtf` above — on a
            # 3v3 pitch the two part company above 3x, and the page says so.
            "simSpeed": self.speed,
            "possessed": next((p.id for p in w.persons.values() if p.possessed), None) if w else None,
        }


def head_applied(d, brain, head_cmds: set[str]) -> bool:
    """Does a brain's gaze intent reach this body's servos?

    The duck's answer is an opt-in, and for a measured reason (roadmap 3.7):
    its head pose rides in the 61-obs command block, and the shipped walker
    never trained with one, so a gaze reaching it changes the observation of a
    policy that has never seen it. A driver-stepped body has no such
    observation — its head is a position servo outside every policy — so it
    declares `head_always` and the gaze always lands
    (`world/arena.WorldRobot`). One function, because the frame reports this
    flag and `drive()` acts on it, and the two answering differently would
    show a gaze the robot is not following.
    """
    return (getattr(d, "head_always", False) or d.id in head_cmds
            or bool(getattr(brain, "wants_head", False)))


def duck_info(w: World, d, brains: dict | None = None) -> dict:
    # `team` is a colorway and `role` is the job it plays (world/scenario.py):
    # what the page paints the duck and what it labels it. Both come off the
    # scenario, so a duck that is not on a pitch carries None for both.
    spec = next((x for x in w.scenario.ducks if x.id == d.id), None)
    return {
        "id": d.id,
        # WHICH BODY this is (`world/scenario.Duck.robot`), so the viewer can
        # pick the mesh set to draw it with — `GET /scene?robot=<id>` — instead
        # of assuming a duck. Always present and always the duck's id on a
        # duck, so a reader needs no absent-field case.
        "robot": DUCK_ROBOT if spec is None else spec.robot,
        "team": None if spec is None else spec.team,
        "role": None if spec is None else spec.role,
        "name": d.id if d.policy_id is None else f"{d.id} · {d.policy_id.split(':', 1)[-1]}",
        "policy": d.policy_id,
        "falls": d.falls,
        # Separate episodes of contact with the room's boards and its static
        # furniture (`World.wall_bumps`; `wallTicks` is how long). Not the
        # `bumped` a brain senses, which is body-on-body — see
        # `World._stamp_bumps`. A wheeled body's Phase 3 bar is stated in
        # this number, and it is the one thing a room can tell you about a
        # driver that a fall count cannot.
        "wallBumps": w.wall_bumps.get(d.id, 0),
        "wallTicks": w.wall_ticks.get(d.id, 0),
        "step": d.step_count,
        "rew": 0.0,
        "speed": round(d.heading_speed(w.data), 3),
        "cmdSpeed": round(float(d.twist_cmd[0]), 3),
        # Every body in `World.ducks` takes a twist (`set_cmd`), duck or
        # driver-stepped, so the page's WASD drive works on all of them.
        "steerable": True,
        "tof": None if d.tof is None else preset_name(d.tof.noise),
        # The 360-degree scan a wheeled body carries instead of a ToF
        # (`sensors/lidar.py`). None on a duck.
        "lidar": (None if getattr(d, "lidar", None) is None
                  else lidar_preset_name(d.lidar.noise)),
        "detector": None if d.detector is None else det_preset_name(d.detector.noise),
        "brainKind": getattr(brains.get(d.id), "kind", "script") if brains is not None else None,
        "holding": d.holding,
        "odom": d.odom_preset,
        "odomEst": [round(float(v), 3) for v in d.odom_est],
        "skill": d.skill,
        "beak": "closed" if d.beak_closed else "open",
        # The 15th servo as the robot takes it: an opening fraction, 0 shut to
        # 1 wide. `beak` above is the GRASP state (is it holding), which is a
        # different question from how far the bill is actually open.
        "mouth": round(float(d.mouth), 3),
    }


def det_preset_name(noise: DetectorNoise) -> str:
    for name in TOF_PRESETS:
        if DetectorNoise.preset(name) == noise:
            return name
    return "custom"


def preset_name(noise: TofNoise) -> str:
    for name in TOF_PRESETS:
        if TofNoise.preset(name) == noise:
            return name
    return "custom"


def lidar_preset_name(noise: LidarNoise) -> str:
    for name in TOF_PRESETS:
        if LidarNoise.preset(name) == noise:
            return name
    return "custom"


ARM_PAYLOAD_DIGITS = 4


def gripper_payload(w: World, d) -> dict | None:
    """The claw as a READING, for a body that has one. None for the rest.

    Four fields and each is somebody else's answer, read rather than redone:

    * `load` — `WorldRobot.gripper_load`, which delegates to the driver that
      owns the measurement (`MarsDriver.gripper_load`: the CONSTRAINT torque
      at joint6, because the servo's own torque saturates on a close through
      air and cannot tell "closing" from "holding"). SIGNED: closing is one
      direction and opening the other, so a consumer takes the magnitude.
    * `holding` — `d.holding is not None`, which is exactly the bool the brain
      is handed as `Senses.holding`. **Not the threshold re-applied here**: it
      is `MarsDriver.held_body`'s two-part predicate (load past the hold line
      AND a blade contact with a body that is not part of the robot) already
      resolved to a pickable id once this tick by `World.sense_grip`. The bar
      alone cannot answer it — a close on AIR drives the blade into its own
      hard stop and reads 0.0 N*m, identical to an open jaw.

      What that does mean, precisely, is "holding a TOY": `sense_grip` maps
      the held body through the room's pickables, so a claw clamped on the
      basket rim shows a load past the mark and `holding` false. That is the
      right flag to show BECAUSE it is the one the brain acts on, and the
      frame carries the toy's id beside it (`duck_info`'s `holding`) for the
      panel's badge.
    * `limit` / `hold` — the servo's torque clamp and the hold threshold, off
      the BODY (`MarsBody.gripper_limit_nm` / `hold_load_nm` through
      `WorldRobot`). A bar needs a full scale and a mark, and shipping them
      per frame is what keeps a viewer from keeping its own copy of another
      robot's constants.

    A driver with no claw answers None for the load and the body declares
    neither constant, so this returns None and the channel is absent.
    """
    fn = getattr(d, "gripper_load", None)
    if fn is None:
        return None
    load = fn(w.data)
    if load is None:
        return None
    out = {"load": round(float(load), 4), "holding": d.holding is not None}
    limit, hold = getattr(d, "gripper_limit_nm", None), getattr(d, "hold_load_nm", None)
    if limit is not None:
        out["limit"] = float(limit)
    if hold is not None:
        out["hold"] = float(hold)
    return out


def arm_payload(w: World, d) -> dict | None:
    """The arm as a READING: achieved angles, commanded targets, travel.

    `q` is `WorldRobot.arm_qpos` — what Innate's `/mars/arm/state` publishes —
    and `cmd` is `arm_targets()`, the same numbers the driver clamped and
    stored. **Both, and that is the point**: the gap between them is the only
    thing in the frame that shows a servo lagging or loaded, and `brain/
    tidy_arm.py` pre-compensates its descent against exactly this pair (the
    measurement is on `Senses.arm`: at the pick pose joints 2/3 read 0.067 /
    0.048 rad off the command, which is 28.5 mm of claw height — and Phase 5's
    second bug was that most of it turned out to be the arm BLOCKED by the
    block, not compliance, which is a thing you can only see by watching the
    two together).

    `limits` is (lo, hi) per joint off this model (`WorldRobot.arm_limits`),
    so a bar is a fraction of real travel; a joint the model leaves unlimited
    is simply absent from it. Rounded to 4 decimals — 1e-4 rad is 0.006
    degrees, two orders below the backlash this is drawn to show, and it
    keeps a 7-joint block around 200 bytes at 25 Hz.

    None for a body with no arm.
    """
    qpos_fn = getattr(d, "arm_qpos", None)
    if qpos_fn is None:
        return None
    q = qpos_fn(w.data)
    if not q:
        return None
    out: dict = {"q": {k: round(float(v), ARM_PAYLOAD_DIGITS) for k, v in q.items()}}
    cmd_fn = getattr(d, "arm_targets", None)
    cmd = None if cmd_fn is None else cmd_fn()
    if cmd:
        out["cmd"] = {k: round(float(v), ARM_PAYLOAD_DIGITS) for k, v in cmd.items()}
    lim_fn = getattr(d, "arm_limits", None)
    lim = None if lim_fn is None else lim_fn()
    if lim:
        out["limits"] = {k: [round(float(lo), ARM_PAYLOAD_DIGITS),
                             round(float(hi), ARM_PAYLOAD_DIGITS)] for k, (lo, hi) in lim.items()}
    return out


def tof_payload(w: World, d) -> dict | None:
    """A robot's senses for the frame: the ToF matrix, the 360-degree scan, the
    detector's frame (the page draws the detection rays and the head-camera
    inset's boxes from it - bearing, elevation, apparent width, and the field
    of view they sit in), and — for a body with a hand — the claw and the arm.

    ONE KEY PER CHANNEL THE BODY HAS, and never a placeholder for one it does
    not: the /sim inspector renders a block per key present, so an absent key
    is how it knows not to draw an instrument. A body with no claw ships no
    `gripper` and the panel shows no load bar, rather than a 0.00 N*m reading
    of a gripper that is not there.

    A wheeled body ships `lidar` and NO `tof`, deliberately. Its brains do
    read an 8x8 (`World.senses_tof` adapts the scan for them), but that frame
    is a fiction of 64 bearing bins with no elevation and no mount pose, and
    the page draws a ToF as a cone of zone points hung off the HEAD CAMERA's
    pose. Shipping it would draw a sensor the robot does not have, pointing
    where it is not — the class of wrong picture this repo's rule about
    looking before claiming exists for. The scan is the honest thing to draw
    and the viewer's job (Phase 2b).
    """
    out: dict = {}
    lidar = getattr(d, "lidar", None)
    if lidar is not None and lidar.last is not None:
        f = lidar.last
        out["lidar"] = {**f.as_payload(), "age": round(w.t - f.t, 4),
                        "maxRange": lidar.max_range,
                        # The two numbers a polar plot of this scan cannot be
                        # drawn HONESTLY without, and which the payload used
                        # to leave the viewer to hard-code per robot id:
                        #
                        #   minRange  the device's own floor. A return nearer
                        #             than this is clipped and marked invalid
                        #             (`LidarSensor.scan`), so it arrives as a
                        #             0 and the plot must draw the disc it
                        #             cannot see inside of. A scanner with a
                        #             different floor drew the wrong disc.
                        #   footprint how far out a return is the robot
                        #             looking at its own arm, dropped by the
                        #             adapter every brain here reads
                        #             (`World.senses_tof` passes exactly this
                        #             to `tof_from_lidar`). Off the BODY
                        #             (`WorldRobot.footprint_m` ->
                        #             `MarsBody.footprint_m`), so the frame
                        #             names no robot; 0.0 means "declared
                        #             none, every return kept", which is the
                        #             adapter's own default and not a missing
                        #             measurement.
                        "minRange": round(float(lidar.min_range), 4),
                        "footprint": round(float(getattr(d, "footprint_m", 0.0)), 4),
                        # Where the scanner is in the base's HEADING frame, so
                        # the viewer can draw the scan from the aperture and
                        # not from the chassis origin — 76 mm apart on MARS.
                        "mount": (None if f.mount_pos is None
                                  else [round(float(v), 4) for v in f.mount_pos])}
    gripper = gripper_payload(w, d)
    if gripper is not None:
        out["gripper"] = gripper
    arm = arm_payload(w, d)
    if arm is not None:
        out["arm"] = arm
    if d.tof is not None and d.tof.last is not None:
        f = d.tof.last
        # No world points here: the page reconstructs each zone's point from the
        # head pose it already has plus the fixed zone directions (lib/sim.ts
        # `tofZonePoints`), which cut a 2-duck stream from 145 to ~45 kB/s.
        out["tof"] = {
            "t": round(f.t, 4),
            "mm": f.depth_mm.reshape(-1).tolist(),
            "age": round(w.t - f.t, 4),
        }
    if d.detector is not None and d.detector.last is not None:
        f = d.detector.last
        out["det"] = {"t": round(f.t, 4), "age": round(w.t - f.t, 4),
                      "fov": [d.detector.spec.fov_h_deg, d.detector.spec.fov_v_deg],
                      "cam": [round(v, 4) for v in f.cam_pose],
                      # Which of this robot's own bodies wraps the lens, so the
                      # /sim inset can leave it out of the picture the camera
                      # draws — the same body the detector refuses to detect.
                      "selfBody": getattr(d, "camera_housing_body", _no_housing)(),
                      "items": [x.as_payload() for x in f.detections]}
    return out or None


# -- requests --------------------------------------------------------------------

class LoadReq(BaseModel):
    scenario: str


class TetherReq(BaseModel):
    ms: float = 0.0


class SpeedReq(BaseModel):
    x: float = 1.0


class NoiseReq(BaseModel):
    duck: str
    preset: str
    # "tof" is the RANGE channel — the scenario's own field name — and on a
    # body whose range sensor is the 360-degree scanner it sets the LIDAR's
    # preset. "lidar" is an accepted alias for that; see `set_noise`.
    sensor: str = "tof"        # "tof" | "lidar" | "det" | "odom"


class BrainReq(BaseModel):
    duck: str
    kind: str


class SaveReq(BaseModel):
    name: str


# -- mounting --------------------------------------------------------------------

def mount_world(app: FastAPI, *, load_infer: Callable[[str], Infer] | None,
                origin_allowed: Callable[[str | None], bool]) -> WorldState:
    st = WorldState(load_infer)
    app.state.world = st

    @app.get("/scenarios")
    def get_scenarios() -> dict:
        return {"scenarios": list_scenarios()}

    @app.get("/scenarios/{name}")
    def get_scenario(name: str) -> dict:
        return resolve_scenario(name).to_dict()

    @app.put("/scenarios/{name}")
    def put_scenario(name: str, raw: dict) -> dict:
        if not NAME_RE.match(name or ""):
            raise HTTPException(400, f"bad scenario name {name!r}")
        if name in BUILTIN_NAMES:
            raise HTTPException(409, f"{name!r} is built in — save under another name")
        raw = dict(raw)
        raw["name"] = name
        try:
            sc = validate_scenario(raw)
        except ScenarioError as e:
            raise HTTPException(422, str(e)) from e
        d = scenarios_dir()
        d.mkdir(parents=True, exist_ok=True)
        sc.save(d / f"{name}.json")
        return sc.to_dict()

    @app.delete("/scenarios/{name}")
    def delete_scenario(name: str) -> dict:
        if not NAME_RE.match(name or ""):
            raise HTTPException(400, f"bad scenario name {name!r}")
        if name in BUILTIN_NAMES:
            raise HTTPException(409, f"{name!r} is built in")
        p = scenarios_dir() / f"{name}.json"
        if not p.exists():
            raise HTTPException(404, f"no scenario {name!r}")
        p.unlink()
        return {"deleted": name}

    @app.get("/world")
    def get_world() -> dict:
        return st.payload()

    @app.get("/scene/g1")
    def get_g1_scene() -> dict:
        """Visual meshes for a G1 person — same shape as GET /scene, different robot."""
        from .robots.g1 import g1_ready, visual_scene
        if not g1_ready():
            raise HTTPException(404, "G1 assets missing — run `uv run fetch-g1`")
        return visual_scene()

    @app.post("/world/load")
    async def load_world(req: LoadReq) -> dict:
        if st.loading:
            raise HTTPException(409, "a world is already loading")
        sc = resolve_scenario(req.scenario)
        st.loading = True
        try:
            composed = await asyncio.to_thread(st.compose, sc)
        except Exception as e:
            st.events.append(f"load failed: {type(e).__name__}: {e}")
            raise HTTPException(500, f"could not build {req.scenario!r}: {e}") from e
        finally:
            st.loading = False
        st.script_t = 0.0
        st.install(composed)     # one block, no await inside: the loop never straddles it
        st.events.append(f"loaded {sc.name}: {len(sc.ducks)} ducks")
        return st.payload()

    @app.post("/world/tether")
    async def set_tether(req: TetherReq) -> dict:
        """Roadmap 12.10: run every brain 'over a tether' with this much
        senses→intent round-trip latency (0 = onboard). Watch what a laptop
        brain over Wi-Fi does to a pick, live."""
        st.tether_ms = float(max(0.0, min(req.ms, 2000.0)))
        st._tether_queue.clear()
        st.events.append(f"tether {st.tether_ms:.0f} ms" if st.tether_ms else "brains onboard (no tether)")
        return {"tetherMs": st.tether_ms}

    @app.post("/world/speed")
    async def set_speed(req: SpeedReq) -> dict:
        """Run the world faster or slower than the wall clock. Clamped to
        SPEED_CHOICES' range; what the box ACTUALLY managed is the frame's
        `rtf`, which on a 3v3 pitch stops climbing around 3x."""
        try:
            return {"simSpeed": st.set_speed(req.x)}
        except (ValueError, OverflowError) as e:
            raise HTTPException(422, str(e)) from e

    @app.post("/world/noise")
    def set_noise(req: NoiseReq) -> dict:
        """Re-noise one sense channel of one body, live.

        **`"tof"` is the RANGE channel and not the device.** It is the name
        the scenario has (`world/scenario.Duck.tof` — "how noisy is this
        robot's range sense" is one question per body, and a body has one
        range sensor), so it is also the name the wire keeps: the /sim panel
        labels its select after the device the body actually carries and sends
        `tof` either way. On a body whose range sensor is the 360-degree
        scanner this therefore sets the LIDAR's preset — before that branch
        existed the request hit `d.tof is None` and came back 409 "has no
        ToF", which put "noise ignored" in the event log every time someone
        touched a MARS's select while the scenario's own field set it happily
        at load.

        `"lidar"` is accepted as an alias for the same thing, so a caller that
        names the device it can see in the frame is not punished for it; the
        EVENT names whichever device was actually re-noised, because "m0 tof
        noise" on a robot with no ToF is the same lie the frame refuses to
        tell.
        """
        w = st.world
        if w is None or req.duck not in w.ducks:
            raise HTTPException(404, f"no duck {req.duck!r}")
        if req.preset not in TOF_PRESETS:
            raise HTTPException(422, f"preset must be one of {TOF_PRESETS}")
        d = w.ducks[req.duck]
        channel = req.sensor
        if req.sensor == "det":
            if d.detector is None:
                raise HTTPException(409, f"{req.duck} has no detector in this scenario")
            d.detector.noise = DetectorNoise.preset(req.preset)
        elif req.sensor in ("tof", "lidar"):
            lidar = getattr(d, "lidar", None)
            if d.tof is not None and req.sensor == "tof":
                d.tof.noise = TofNoise.preset(req.preset)
                channel = "tof"
            elif lidar is not None:
                lidar.noise = LidarNoise.preset(req.preset)
                channel = "lidar"
            else:
                raise HTTPException(
                    409, f"{req.duck} has no {'ToF or lidar' if req.sensor == 'tof' else 'lidar'} "
                         "in this scenario")
        elif req.sensor == "odom":
            w.set_odom_preset(d, req.preset)
        else:
            raise HTTPException(422, "sensor must be 'tof' (the range channel), 'lidar', 'det' or 'odom'")
        st.events.append(f"{req.duck} {channel} noise → {req.preset}")
        return duck_info(w, d)

    @app.post("/world/brain")
    def set_brain(req: BrainReq) -> dict:
        try:
            st.set_brain(req.duck, req.kind)
        except KeyError:
            raise HTTPException(404, f"no duck {req.duck!r}") from None
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return {"duck": req.duck, "kind": req.kind, "kinds": REGISTRY.available()}

    @app.get("/replay/ring")
    def replay_ring(last: int = 1500) -> Response:
        n = max(0, min(int(last), len(st.ring)))
        frames = list(st.ring)[-n:] if n else []
        body = '{"frames":[' + ",".join(frames) + '],"count":' + str(n) + "}"
        return Response(content=body, media_type="application/json")

    @app.post("/replay/save")
    def replay_save(req: SaveReq) -> dict:
        if not RECORDING_RE.match(req.name or ""):
            raise HTTPException(400, f"bad recording name {req.name!r}")
        frames = list(st.ring)
        if not frames:
            raise HTTPException(409, "nothing recorded yet")
        d = recordings_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{req.name}.jsonl.gz"
        first, last_ = json.loads(frames[0]), json.loads(frames[-1])
        header = {"version": 1, "name": req.name,
                  "scenario": st.scenario.name if st.scenario else None,
                  "saved": time.time(), "frames": len(frames),
                  "span": round(float(last_.get("t", 0)) - float(first.get("t", 0)), 3)}
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")
            for fr in frames:
                f.write(fr + "\n")
        st.events.append(f"saved {len(frames)} frames as {req.name}")
        return header

    @app.get("/recordings")
    def get_recordings() -> dict:
        out = []
        d = recordings_dir()
        if d.exists():
            for p in sorted(d.glob("*.jsonl.gz")):
                try:
                    with gzip.open(p, "rt", encoding="utf-8") as f:
                        out.append(json.loads(f.readline()))
                except (OSError, ValueError):
                    continue
        return {"recordings": out}

    def recording_path(name: str) -> Path:
        if not RECORDING_RE.match(name or ""):
            raise HTTPException(400, f"bad recording name {name!r}")
        p = recordings_dir() / f"{name}.jsonl.gz"
        if not p.exists():
            raise HTTPException(404, f"no recording {name!r}")
        return p

    @app.get("/recordings/{name}")
    def get_recording(name: str) -> Response:
        p = recording_path(name)
        with gzip.open(p, "rt", encoding="utf-8") as f:
            header = f.readline().strip()
            frames = [ln.strip() for ln in f if ln.strip()]
        body = '{"header":' + header + ',"frames":[' + ",".join(frames) + "]}"
        return Response(content=body, media_type="application/json")

    @app.delete("/recordings/{name}")
    def delete_recording(name: str) -> dict:
        recording_path(name).unlink()
        return {"deleted": name}

    @app.websocket("/ws/sim")
    async def ws_sim(sock: WebSocket):
        if not origin_allowed(sock.headers.get("origin")):
            await sock.close(code=1008)
            return
        await sock.accept()
        st.clients.add(sock)
        try:
            while True:
                msg = json.loads(await sock.receive_text())
                if "cmd" in msg:
                    st.override = np.clip(np.array(msg["cmd"], np.float32),
                                          [-0.9, -0.3, -1.0], [0.9, 0.3, 1.0])
                    st.override_until = time.monotonic() + st.override_hold_s()
                if msg.get("reset") and st.world is not None:
                    st.world.reset()
                    for g in st.maps.values():
                        g.reset()
                    for d in st.world.ducks.values():
                        d.falls = 0
                    for b in st.brains.values():
                        b.reset()
                    st.intents.clear()
                    st.script_t = 0.0
                    st.restart()
                if "assign" in msg and st.world is not None:
                    a = msg["assign"]
                    asyncio.create_task(do_assign(str(a.get("duck")), str(a.get("policy"))))
                if "tether" in msg:
                    st.tether_ms = float(max(0.0, min(float(msg["tether"] or 0.0), 2000.0)))
                    st._tether_queue.clear()
                if "speed" in msg:
                    try:
                        st.set_speed(float(msg["speed"]))
                    except (TypeError, ValueError, OverflowError):
                        st.events.append(f"speed ignored: {msg['speed']!r}")
                if "noise" in msg and st.world is not None:
                    n = msg["noise"]
                    try:
                        set_noise(NoiseReq(duck=str(n.get("duck")), preset=str(n.get("preset")),
                                           sensor=str(n.get("sensor", "tof"))))
                    except HTTPException as e:
                        st.events.append(f"noise ignored: {e.detail}")
                if "brain" in msg and st.world is not None:
                    b = msg["brain"]
                    try:
                        set_brain(BrainReq(duck=str(b.get("duck")), kind=str(b.get("kind"))))
                    except HTTPException as e:
                        st.events.append(f"brain ignored: {e.detail}")
                if "possess" in msg and st.world is not None:
                    who = msg["possess"]
                    who = None if who in (None, "", False) else str(who)
                    if who is not None and who not in st.world.persons:
                        st.events.append(f"possess ignored: no person {who}")
                    else:
                        st.world.possess(who)
                        st.events.append(f"you are {who}" if who else "released")
                if "head" in msg and st.world is not None:
                    h = msg["head"]
                    did = str(h.get("duck"))
                    if did in st.world.ducks:
                        (st.head_cmds.add if h.get("apply") else st.head_cmds.discard)(did)
        except WebSocketDisconnect:
            pass
        finally:
            st.clients.discard(sock)

    async def do_assign(duck_id: str, policy_id: str) -> None:
        w = st.world
        if w is None or duck_id not in w.ducks:
            st.events.append(f"assign failed: no duck {duck_id}")
            return
        infer = await asyncio.to_thread(st.infer_for, policy_id)
        if infer is None:
            return
        w.set_policy(duck_id, infer, policy_id)
        st.events.append(f"{duck_id} now runs {policy_id.split(':', 1)[-1]}")

    async def world_loop() -> None:
        tick = 0
        next_t = time.monotonic()
        window_t0, window_sim = next_t, 0.0
        while True:
            now = time.monotonic()
            cmd, mode = st.current_cmd(now)     # …and again per step below, so
            w = st.world                        # a 0.25x tick that steps none still has one
            # Spend the wall tick's sim-time budget (SPEED_CHOICES): `speed`
            # sim ticks at 4x, one every fourth wall tick at 0.25x. Capped at
            # one tick of the top speed, so a loop that fell behind catches
            # up rather than sprinting.
            st._step_credit = min(st._step_credit + st.speed, SPEED_MAX)
            budget_end = now + STEP_BUDGET_S
            while st._step_credit >= 1.0:
                st._step_credit -= 1.0
                # Per STEP, not per wall tick: the demo script lives on
                # `script_t`, so sampling it once a tick would hand all four
                # of a 4x batch the command belonging to the first. That is
                # the only thing that could have made the world come out
                # differently at speed, and tests/test_world_server.py
                # pins that it does not.
                cmd, mode = st.current_cmd(now)
                st.script_t += 1.0 / TICK_HZ
                if w is None:
                    continue
                st.drive(cmd, mode)
                w.step()
                if st.metrics is not None:
                    st.metrics.tick()
                st.after_step()
                window_sim += 1.0 / TICK_HZ
                for did, grid in st.maps.items():
                    d = w.ducks.get(did)
                    if d is not None and d.tof is not None:
                        grid.update(d.tof.last, w.odom(d))
                # Out of wall slot with ticks still owed: keep the credit (it
                # is capped, so the debt cannot grow) and go serve the socket.
                if st._step_credit >= 1.0 and time.monotonic() >= budget_end:
                    break
            tick += 1
            # The rtf window is a wall SECOND, measured as one — not
            # `tick % TICK_HZ`, which is only a second while the loop keeps
            # its schedule, and at 8x on a heavy scene it does not. A speed
            # change throws the window away rather than reporting a figure
            # averaged across both speeds; until the next one closes `rtf`
            # stays 0, which is what the page reads as "no measurement yet"
            # instead of as a shortfall.
            if st._rtf_dirty:
                st._rtf_dirty = False
                window_t0, window_sim = now, 0.0
            elif now - window_t0 >= RTF_WINDOW_S:
                st.rtf = window_sim / (now - window_t0)
                window_t0, window_sim = now, 0.0
            if tick % SEND_EVERY == 0:
                t_enc = time.perf_counter()
                st.send_maps = (tick // SEND_EVERY) % MAP_EVERY == 0
                frame = json.dumps(st.frame(cmd, mode))
                if w is not None:                      # what the loop spends on the wire, not the sim
                    w.perf["encodeMs"] += 0.05 * ((time.perf_counter() - t_enc) * 1e3 - w.perf.get("encodeMs", 0.0))
                st.events.clear()
                # A ring of "no world" frames is nothing to save (and a race
                # in the tests) — and neither is a ring of the SAME world:
                # below 0.5x the sim steps less often than the loop sends, so
                # half the frames repeat a world that has not moved and would
                # otherwise dilute the scrub bar's two minutes with padding.
                if w is not None and w.tick != st._ring_tick:
                    st._ring_tick = w.tick
                    st.ring.append(frame)
                dead = []
                for c in list(st.clients):
                    try:
                        await c.send_text(frame)
                    except Exception:
                        dead.append(c)
                for c in dead:
                    st.clients.discard(c)
            # Pace against the wall, but never bank a debt: at 4x on a 3v3
            # the body costs more than the 20 ms it is given, and a loop that
            # kept the arrears would run flat out for as long again once the
            # scene got cheap.
            next_t = max(next_t + 1.0 / TICK_HZ, time.monotonic() - MAX_LAG_S)
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

    def start() -> None:
        st.task = asyncio.create_task(world_loop())

        def died(t: asyncio.Task) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                traceback.print_exception(type(exc), exc, exc.__traceback__)
                print("[sim] FATAL: the world loop stopped — /sim frames will not be "
                      "sent. Restart the lab.", flush=True)
        st.task.add_done_callback(died)

    def stop() -> None:
        if st.task is not None:
            st.task.cancel()

    st.start, st.stop = start, stop  # type: ignore[attr-defined]
    return st


__all__ = ["WorldState", "builtin_scenarios", "list_scenarios", "mount_world",
           "resolve_scenario", "scenarios_dir"]
