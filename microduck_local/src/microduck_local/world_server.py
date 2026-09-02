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
            brain: {kind: "wander"|"script"|"manual", state, cmd},
            bodies: [[x,y,z,qw,qx,qy,qz] × 16] (world first, as GET /scene lists bodies),
            sensors: {tof: {t, mm[64], age}} | null}],   (zone points: page-side)
   objects: [{id, kind, pose}]}
accepts:
  {"cmd": [vx, vy, wz]}   drive every duck (held OVERRIDE_HOLD_S); otherwise a duck with
                          a ToF wanders on the brain layer's Wander controller and a
                          blind duck follows the demo script
  {"reset": true}         respawn everything
  {"assign": {"duck": id, "policy": palette id}}
  {"noise": {"duck": id, "preset": name}}

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
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel

from .brain import Wander
from .sensors import TofNoise
from .world import Ball, Duck, Scenario, Wall, World, make_room
from .world.scenario import NAME_RE, TOF_PRESETS, ScenarioError, validate_scenario

TICK_HZ = 50
SEND_EVERY = 2
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
    return {s.name: s for s in (empty, wall, room)}


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
        self.clients: set[WebSocket] = set()
        self.override: np.ndarray | None = None
        self.override_until = 0.0
        self.script_t = 0.0
        self.events: deque[str] = deque(maxlen=200)
        self.loading = False
        self.rtf = 0.0
        self.task: asyncio.Task | None = None
        # Auto mode: a duck with a ToF wanders on its own brain (the first
        # controller of roadmap Track 2); a blind duck follows the script.
        self.brains: dict[str, Wander] = {}
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
        self.brains = {d.id: Wander() for d in world.ducks.values() if d.tof is not None}
        return world

    def preload(self, name: str) -> None:
        """Build a world before serving (the CLI's --world). Blocking."""
        sc = resolve_scenario(name)
        self.world, self.scenario = self.build(sc), sc
        print(f"[sim] world {sc.name}: {len(sc.ducks)} ducks", flush=True)

    def payload(self) -> dict:
        w = self.world
        return {
            "scenario": self.scenario.to_dict() if self.scenario else None,
            "loading": self.loading,
            "ducks": [duck_info(w, d) for d in w.ducks.values()] if w else [],
            "presets": list(TOF_PRESETS),
        }

    def drive(self, cmd: np.ndarray, mode: str) -> None:
        """Set every duck's command for this tick: the manual override for
        all of them, else each duck's own brain (or the script, blind)."""
        w = self.world
        if w is None:
            return
        for d in w.ducks.values():
            brain = self.brains.get(d.id) if mode == "auto" else None
            if brain is None:
                d.set_cmd(w.data, cmd)
                continue
            tof = d.tof.last if d.tof is not None else None
            d.set_cmd(w.data, brain.step(
                None if tof is None else tof.depth_mm,
                None if tof is None else tof.valid,
                w.t, d.heading_speed(w.data)))

    def brain_payload(self, d, mode: str) -> dict:
        brain = self.brains.get(d.id)
        if mode == "manual" or brain is None:
            return {"kind": "manual" if mode == "manual" else "script",
                    "state": mode, "cmd": [round(float(v), 3) for v in d.twist_cmd]}
        return {"kind": "wander", "state": brain.state,
                "cmd": [round(float(v), 3) for v in brain.last]}

    def frame(self, cmd: np.ndarray, mode: str) -> dict:
        w = self.world
        ducks = []
        if w is not None:
            for d in w.ducks.values():
                ducks.append({
                    **duck_info(w, d),
                    "brain": self.brain_payload(d, mode),
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
            "objects": w.objects_payload() if w else [],
        }


def duck_info(w: World, d) -> dict:
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
    }


def preset_name(noise: TofNoise) -> str:
    for name in TOF_PRESETS:
        if TofNoise.preset(name) == noise:
            return name
    return "custom"


def tof_payload(w: World, d) -> dict | None:
    if d.tof is None or d.tof.last is None:
        return None
    f = d.tof.last
    # No world points here: the page reconstructs each zone's point from the
    # head pose it already has plus the fixed zone directions (lib/sim.ts
    # `tofZonePoints`), which cut a 2-duck stream from 145 to ~45 kB/s.
    return {"tof": {
        "t": round(f.t, 4),
        "mm": f.depth_mm.reshape(-1).tolist(),
        "age": round(w.t - f.t, 4),
    }}


# -- requests --------------------------------------------------------------------

class LoadReq(BaseModel):
    scenario: str


class NoiseReq(BaseModel):
    duck: str
    preset: str


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
        st.script_t = 0.0
        st.events.append(f"loaded {sc.name}: {len(sc.ducks)} ducks")
        return st.payload()

    @app.post("/world/noise")
    def set_noise(req: NoiseReq) -> dict:
        w = st.world
        if w is None or req.duck not in w.ducks:
            raise HTTPException(404, f"no duck {req.duck!r}")
        if req.preset not in TOF_PRESETS:
            raise HTTPException(422, f"preset must be one of {TOF_PRESETS}")
        tof = w.ducks[req.duck].tof
        if tof is None:
            raise HTTPException(409, f"{req.duck} has no ToF in this scenario")
        tof.noise = TofNoise.preset(req.preset)
        st.events.append(f"{req.duck} ToF noise → {req.preset}")
        return duck_info(w, w.ducks[req.duck])

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
                    for d in st.world.ducks.values():
                        d.falls = 0
                    for b in st.brains.values():
                        b.reset()
                    st.script_t = 0.0
                if "assign" in msg and st.world is not None:
                    a = msg["assign"]
                    asyncio.create_task(do_assign(str(a.get("duck")), str(a.get("policy"))))
                if "noise" in msg and st.world is not None:
                    n = msg["noise"]
                    try:
                        set_noise(NoiseReq(duck=str(n.get("duck")), preset=str(n.get("preset"))))
                    except HTTPException as e:
                        st.events.append(f"noise ignored: {e.detail}")
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
                window_sim += 1.0 / TICK_HZ
            tick += 1
            if tick % TICK_HZ == 0:
                wall = now - window_t0
                st.rtf = window_sim / wall if wall > 0 else 0.0
                window_t0, window_sim = now, 0.0
            if tick % SEND_EVERY == 0:
                frame = json.dumps(st.frame(cmd, mode))
                st.events.clear()
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
