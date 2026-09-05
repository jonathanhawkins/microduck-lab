"""World mode for the lab: the /sim page's backend (docs/sim-roadmap.md 0.4).

Mounted onto the lab's FastAPI app by `make_app`, beside the roster loop.
The roster (`/ws`) keeps its private-env ducks, teach jobs and captures; this
module owns ONE `World` — a scenario's room, its objects, N ducks in one
mjData — and streams it on a second socket so the two pages never fight
over a frame format.

HTTP:
  GET  /scenarios              [{name, builtin, ducks, objects, modified}]
  GET  /scenarios/{name}       the scenario JSON (built-ins are generated)
  PUT  /scenarios/{name}       save a user scenario (validated; built-ins are read-only)
  DELETE /scenarios/{name}     remove a user scenario
  GET  /world                  {scenario, loading, ducks: [...]} — what is live now
  POST /world/load {"scenario": name}      compose + swap (a second or so;
                               the loop keeps streaming the old world meanwhile)
  POST /world/noise {"duck": id, "preset": "ideal"|"datasheet"|"hostile"}
  GET  /replay/ring?last=N     the last N frames the loop broadcast (a ring of
                               RING_S seconds at 25 Hz, kept whether or not a
                               browser is attached) — the page's scrub bar
  POST /replay/save {"name"}   write the ring to recordings/<name>.jsonl.gz
  GET  /recordings             [{name, frames, span, scenario, saved}]
  GET  /recordings/{name}      the frames of one recording (JSON array)
  DELETE /recordings/{name}

WS /ws/sim — 25 Hz frames:
  {t, tick, rtf, perf: {stepMs, sensorMs}, scenario, cmd, mode, events,
   ducks: [{id, name, policy, falls, step, rew, speed, cmdSpeed, steerable,
            brain: {kind, state, cmd, head, note, inputs: {tof: {age, stale}, det: {age, stale, n}, target?}},
            bodies: [[x,y,z,qw,qx,qy,qz] × 16] (world first, as GET /scene lists bodies),
            sensors: {tof: {t, mm[64], age}, det: {t, age, items: [{cls, name, bearing, elevation, width, range, conf}]}} | null}],
   objects: [{id, kind: "ball"|"box"|"person", pose, possessed?}], possessed: person id | null}
accepts:
  {"cmd": [vx, vy, wz]}   drive every duck (held OVERRIDE_HOLD_S); otherwise a duck with
                          a ToF wanders on the brain layer's Wander controller and a
                          blind duck follows the demo script
  {"reset": true}         respawn everything
  {"assign": {"duck": id, "policy": palette id}}
  {"noise": {"duck": id, "preset": name, "sensor": "tof"|"det"}}
  {"brain": {"duck": id, "kind": "wander"|"follow"|"script"}}   swap a duck's brain
  {"possess": person id | null}   your cmd drives that person (ducks stay on their brains)
  {"head": {"duck": id, "apply": bool}}   let a brain's gaze intent reach the walker's command block

Scenario files live in microduck_local/scenarios/ (MICRODUCK_SCENARIOS_DIR
relocates it). Built-ins are generated in code so a fresh checkout has
something to load, and they cannot be overwritten — save under another name.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import re
import time
import traceback
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel

from .brain import REGISTRY, Intent, Senses
from .brain.learned import learned_index
from .brain import runtime as brain_runtime
from .brain import tidy as _tidy  # noqa: F401  (registers the tidy brain)
from .brain.mapping import GridSpec, OccupancyGrid
from .brain.tether import Tether
from .sensors import DetectorNoise, TofNoise
from .world import Ball, Duck, Person, Scenario, Wall, World, make_pitch, make_playroom, make_room
from .world.scenario import NAME_RE, TOF_PRESETS, ScenarioError, validate_scenario

TICK_HZ = 50
SEND_EVERY = 2
MAP_EVERY = 12               # occupancy maps ride every 12th frame (~2 Hz): 3–4 kB each per duck
OVERRIDE_HOLD_S = 6.0
RING_S = 120.0                       # the scrub bar reaches this far back
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


def recordings_dir() -> Path:
    return Path(os.environ.get(
        "MICRODUCK_RECORDINGS_DIR",
        Path(__file__).resolve().parents[2] / "recordings"))


def scenarios_dir() -> Path:
    return Path(os.environ.get(
        "MICRODUCK_SCENARIOS_DIR",
        Path(__file__).resolve().parents[2] / "scenarios"))


# -- built-in scenarios -------------------------------------------------------

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
    follow = Scenario(
        name="follow-me", floor=(6.5, 5.5),
        walls=[Wall((-fx, -fy), (fx, -fy)), Wall((fx, -fy), (fx, fy)),
               Wall((fx, fy), (-fx, fy)), Wall((-fx, fy), (-fx, -fy))],
        # The shipped follower, not the rule brain: the scene exists to watch
        # a brain follow a person, and this is the brain that goes on the
        # robot (README "the follow pick"; swap to "follow" in the inspector
        # to compare). If this clone lacks brains/follow-v4, load_world's
        # fallback below runs the duck on "script" and says so in events.
        ducks=[Duck("d0", (0.0, 0.0, 0.0), DEFAULT_POLICY, "datasheet", "datasheet", "learned:follow-v4")],
        persons=[Person("p0", (1.2, 0.0), 1.57,
                        path=[(1.2, 1.2), (-1.2, 1.2), (-1.2, -1.2), (1.2, -1.2)], speed=0.25)])
    playroom = make_playroom(seed=0, n=6, name="playroom")
    playroom.ducks[0].policy = DEFAULT_POLICY
    pitch = make_pitch(name="pitch")
    pitch2 = make_pitch(name="pitch-2v2", per_side=2)
    pitch3 = make_pitch(name="pitch-3v3", per_side=3)
    for d in pitch2.ducks + pitch3.ducks:
        d.policy = DEFAULT_POLICY
    for d in pitch.ducks:
        d.policy = DEFAULT_POLICY
    return {s.name: s for s in (empty, wall, room, follow, playroom, pitch, pitch2, pitch3)}


BUILTIN_NAMES = frozenset(builtin_scenarios().keys())


def list_scenarios() -> list[dict]:
    out = []
    for name, sc in builtin_scenarios().items():
        out.append({"name": name, "builtin": True, "ducks": len(sc.ducks),
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
        self.task: asyncio.Task | None = None
        # Auto mode: each duck runs a brain from the registry (brain/runtime.py);
        # a blind duck gets the script. Intents are remembered for the frame.
        self.brains: dict[str, object] = {}
        self.teams: dict[str, object] = {}
        self.goal_seq = 0                    # World.goal_seq last acted on (kickoff_brains)
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

    def build(self, scenario: Scenario) -> World:
        """Blocking: compose + policies. Call from a thread."""
        infer = {}
        for d in scenario.ducks:
            f = self.infer_for(d.policy)
            if f is not None:
                infer[d.id] = f
        world = World(scenario, infer_for=infer)
        self.brains = {}
        self.teams = {}
        self.goal_seq = world.goal_seq
        for sd in scenario.ducks:
            kind = sd.brain or ("wander" if sd.tof is not None else "script")
            try:
                self.brains[sd.id] = self.make_brain(kind, sd, world)
            except ValueError as e:
                self.events.append(f"{sd.id}: {e}; using script")
                self.brains[sd.id] = REGISTRY.make("script")
        self.intents = {}
        # Room mapping (roadmap 4.x first step): an occupancy grid per duck
        # in ITS odometry frame, from its ToF frames — never from the sim.
        fx, fy = scenario.floor
        self.maps = {sd.id: OccupancyGrid(GridSpec(size=(fx + 1.0, fy + 1.0)))
                     for sd in scenario.ducks if sd.tof is not None}
        return world

    def make_brain(self, kind: str, sd, world):
        """A brain for one duck; on a pitch a `chase` gets its goal and team."""
        from .brain.team import brain_kwargs
        spec = replace(sd, brain=kind)
        return REGISTRY.make(kind, **brain_kwargs(spec, world, self.teams))

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

    def preload(self, name: str) -> None:
        """Build a world before serving (the CLI's --world). Blocking."""
        sc = resolve_scenario(name)
        self.world, self.scenario = self.build(sc), sc
        self.metrics = self.new_metrics()
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
        }

    def senses_for(self, d) -> Senses:
        w = self.world
        tof = d.tof.last if d.tof is not None else None
        det = d.detector.last if d.detector is not None else None
        return Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                      det=det, det_age=None if det is None else w.t - det.t,
                      speed=d.heading_speed(w.data),
                      odom=w.odom(d),
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
                use_head = d.id in self.head_cmds or getattr(brain, "wants_head", False)
                d.set_cmd(w.data, intent.twist, intent.head if use_head else None)

    def after_step(self) -> None:
        """A goal restarts play: the World moved everyone (World.kickoff);
        the brains and the team boards forget their plans here."""
        w = self.world
        if w is None or w.goal_seq == self.goal_seq:
            return
        from .brain.team import kickoff_brains
        self.goal_seq = w.goal_seq
        kickoff_brains(self.brains, self.teams)
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

    def frame(self, cmd: np.ndarray, mode: str) -> dict:
        w = self.world
        ducks = []
        if w is not None:
            for d in w.ducks.values():
                ducks.append({
                    **duck_info(w, d, self.brains),
                    "brain": self.brain_payload(d, mode),
                    "headApplied": d.id in self.head_cmds or bool(getattr(self.brains.get(d.id), "wants_head", False)),
                    # Body 0 is the WORLD in the viewer's scene (GET /scene),
                    # so a duck's 15 bodies ride behind one identity pose and
                    # the same Duck renderer works on both pages.
                    "bodies": [[0, 0, 0, 1, 0, 0, 0]] + w.duck_pose(d.id),
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
            "possessed": next((p.id for p in w.persons.values() if p.possessed), None) if w else None,
        }


def duck_info(w: World, d, brains: dict | None = None) -> dict:
    return {
        "id": d.id,
        "name": d.id if d.policy_id is None else f"{d.id} · {d.policy_id.split(':', 1)[-1]}",
        "policy": d.policy_id,
        "falls": d.falls,
        "step": d.step_count,
        "rew": 0.0,
        "speed": round(d.heading_speed(w.data), 3),
        "cmdSpeed": round(float(d.twist_cmd[0]), 3),
        "steerable": True,
        "tof": None if d.tof is None else preset_name(d.tof.noise),
        "detector": None if d.detector is None else det_preset_name(d.detector.noise),
        "brainKind": getattr(brains.get(d.id), "kind", "script") if brains is not None else None,
        "holding": d.holding,
        "odom": d.odom_preset,
        "odomEst": [round(float(v), 3) for v in d.odom_est],
        "skill": d.skill,
        "beak": "closed" if d.beak_closed else "open",
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


def tof_payload(w: World, d) -> dict | None:
    """A duck's senses for the frame: the ToF matrix and the detector's
    frame (the page draws the detection rays and the head-camera inset's
    boxes from it - bearing, elevation, apparent width, and the field of
    view they sit in)."""
    out: dict = {}
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
                      "items": [x.as_payload() for x in f.detections]}
    return out or None


# -- requests --------------------------------------------------------------------

class LoadReq(BaseModel):
    scenario: str


class TetherReq(BaseModel):
    ms: float = 0.0


class NoiseReq(BaseModel):
    duck: str
    preset: str
    sensor: str = "tof"        # "tof" | "det" | "odom"


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

    @app.post("/world/load")
    async def load_world(req: LoadReq) -> dict:
        if st.loading:
            raise HTTPException(409, "a world is already loading")
        sc = resolve_scenario(req.scenario)
        st.loading = True
        try:
            world = await asyncio.to_thread(st.build, sc)
        except Exception as e:
            st.events.append(f"load failed: {type(e).__name__}: {e}")
            raise HTTPException(500, f"could not build {req.scenario!r}: {e}") from e
        finally:
            st.loading = False
        st.world, st.scenario = world, sc
        st.metrics = st.new_metrics()
        st.script_t = 0.0
        st.events.append(f"loaded {sc.name}: {len(sc.ducks)} ducks")
        return st.payload()

    @app.post("/world/tether")
    def set_tether(req: TetherReq) -> dict:
        """Roadmap 12.10: run every brain 'over a tether' with this much
        senses→intent round-trip latency (0 = onboard). Watch what a laptop
        brain over Wi-Fi does to a pick, live."""
        st.tether_ms = float(max(0.0, min(req.ms, 2000.0)))
        st._tether_queue.clear()
        st.events.append(f"tether {st.tether_ms:.0f} ms" if st.tether_ms else "brains onboard (no tether)")
        return {"tetherMs": st.tether_ms}

    @app.post("/world/noise")
    def set_noise(req: NoiseReq) -> dict:
        w = st.world
        if w is None or req.duck not in w.ducks:
            raise HTTPException(404, f"no duck {req.duck!r}")
        if req.preset not in TOF_PRESETS:
            raise HTTPException(422, f"preset must be one of {TOF_PRESETS}")
        d = w.ducks[req.duck]
        if req.sensor == "det":
            if d.detector is None:
                raise HTTPException(409, f"{req.duck} has no detector in this scenario")
            d.detector.noise = DetectorNoise.preset(req.preset)
        elif req.sensor == "tof":
            if d.tof is None:
                raise HTTPException(409, f"{req.duck} has no ToF in this scenario")
            d.tof.noise = TofNoise.preset(req.preset)
        elif req.sensor == "odom":
            w.set_odom_preset(d, req.preset)
        else:
            raise HTTPException(422, "sensor must be 'tof' or 'det'")
        st.events.append(f"{req.duck} {req.sensor} noise → {req.preset}")
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
                    st.override_until = time.monotonic() + OVERRIDE_HOLD_S
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
                if "assign" in msg and st.world is not None:
                    a = msg["assign"]
                    asyncio.create_task(do_assign(str(a.get("duck")), str(a.get("policy"))))
                if "tether" in msg:
                    st.tether_ms = float(max(0.0, min(float(msg["tether"] or 0.0), 2000.0)))
                    st._tether_queue.clear()
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
            cmd, mode = st.current_cmd(now)
            st.script_t += 1.0 / TICK_HZ
            w = st.world
            if w is not None:
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
            tick += 1
            if tick % TICK_HZ == 0:
                wall = now - window_t0
                st.rtf = window_sim / wall if wall > 0 else 0.0
                window_t0, window_sim = now, 0.0
            if tick % SEND_EVERY == 0:
                t_enc = time.perf_counter()
                st.send_maps = (tick // SEND_EVERY) % MAP_EVERY == 0
                frame = json.dumps(st.frame(cmd, mode))
                if w is not None:                      # what the loop spends on the wire, not the sim
                    w.perf["encodeMs"] += 0.05 * ((time.perf_counter() - t_enc) * 1e3 - w.perf.get("encodeMs", 0.0))
                st.events.clear()
                if w is not None:            # a ring of "no world" frames is nothing to save (and a race in the tests)
                    st.ring.append(frame)
                dead = []
                for c in list(st.clients):
                    try:
                        await c.send_text(frame)
                    except Exception:
                        dead.append(c)
                for c in dead:
                    st.clients.discard(c)
            next_t += 1.0 / TICK_HZ
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
