"""Record a WORLD scenario (a room, the playroom, a pitch) to video, a contact
sheet and an events log — the debugging eye for the /sim page.

    cd microduck_local
    uv run record-world pitch-2v2 --seconds 30 --out /tmp/rw-pitch
    uv run record-world living-room --seconds 20 --camera follow:d0 --out /tmp/rw-room
    uv run record-world playroom --seconds 120 --skip 60 --brain d0=tidy --out /tmp/rw-tidy
    uv run record-world scenarios/my-room.json --seed 3 --out /tmp/rw-mine

`render-rollout` looks at ONE policy in its training env; `render_pitch.py`
looks at a roster from the battery's side. This one looks at what the lab
actually runs: `world_server.WorldState` builds the world (same policies,
same brains, same team boards, same tether, same kickoff rule), and the loop
here is `world_loop` without the socket — so a bug seen on the /sim page is
reproduced here under a seed, and what is recorded IS what the lab would do.

Outputs, under `--out`:

- `world.mp4`  — for the human. `--stride 5` renders every 5th control step
  (10 fps of sim time) and `--fps 10` plays it back in real time.
- `sheet.png`  — for the agent: `--sheet-frames` evenly spaced tiles with the
  numbers burned in (time, score, and one line per duck: brain / state,
  falls, speed, what it holds, the skill running, the brain's note). A tile
  is highlighted when a duck FELL inside its stride window.
- `events.txt` — every brain state transition, fall (with position, state and
  note), pick-up / release, goal and kickoff, with sim time. Read this first:
  it is the whole run in a few hundred lines, and it says WHEN to look.

Cameras: `top` (the whole floor, the pitch's view), `follow:<duck>` (a
three-quarter camera that tracks that duck — read a fall or a grasp),
`side` / `front` / `three-quarter` (fixed at the room centre).

Headless on the Mac's CGL backend with no MUJOCO_GL set (render_rollout.py
explains); set MUJOCO_GL=egl only on Linux.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .render_rollout import CAMERAS, LOOKAT_Z, build_sheet, sheet_indices
from .world.scenario import Scenario, load_scenario

TOP_ELEVATION = -70.0     # near-plan, a little depth left (scripts/render_pitch.py's angle)
FOLLOW_DISTANCE = 1.1     # m — a duck fills ~30% of the tile, the toy or ball it works stays in frame


def parse_brains(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--brain wants DUCK=KIND, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def resolve(name_or_path: str) -> Scenario:
    """A builtin name (`GET /scenarios`), a user scenario name, or a .json path."""
    p = Path(name_or_path)
    if p.suffix == ".json":
        return load_scenario(p)
    from .world_server import resolve_scenario
    return resolve_scenario(name_or_path)


def make_camera(spec: str, sc: Scenario):
    import mujoco
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.0, 0.0, 0.0)
    follow = None
    if spec == "top":
        cam.azimuth, cam.elevation = 0.0, TOP_ELEVATION
        cam.distance = max(sc.floor) * 1.05
    elif spec.startswith("follow:"):
        follow = spec.split(":", 1)[1]
        if follow not in {d.id for d in sc.ducks}:
            raise SystemExit(f"--camera follow:{follow}: no such duck in {sc.name} "
                             f"({', '.join(d.id for d in sc.ducks)})")
        cam.azimuth, cam.elevation = CAMERAS["three-quarter"]
        cam.distance = FOLLOW_DISTANCE
    elif spec in CAMERAS:
        cam.azimuth, cam.elevation = CAMERAS[spec]
        cam.distance = max(sc.floor) * 0.9
        cam.lookat[2] = LOOKAT_Z
    else:
        raise SystemExit(f"--camera must be top, follow:<duck>, or one of {sorted(CAMERAS)}")
    return cam, follow


def score_line(w, st, final: bool = False) -> str:
    """The tile's score: goals and the ball on a pitch, the basket count in the
    playroom. `final` adds the pitch's possession per team (the sheet header)."""
    if w.soccer_score():
        line = f"goals {w.goals['left']}-{w.goals['right']}"
        ball = w.ball_xy()
        if ball is not None:
            line += f" ball {ball[0]:+.2f},{ball[1]:+.2f}"
        if final and st.metrics is not None:
            poss = st.metrics.row().get("possession", {})
            if poss:
                line += "  possession s/min: " + " ".join(f"{k} {v}" for k, v in poss.items())
        return line
    if w.pickables:
        t = w.tidy_score()
        return f"tidy {t.get('inBasket', 0)}/{len(w.pickables)} in basket"
    return ""


def duck_lines(w, d, st) -> list[str]:
    """One tile line per duck, two when it holds / runs a skill / says something.
    build_sheet fits 35 monospace columns whatever the tile width, so the
    line is `id role-or-kind/state v<speed> f<falls> x,y` and nothing else."""
    brain = st.brains.get(d.id)
    intent = st.intents.get(d.id)
    kind = getattr(brain, "kind", "script")
    state = str(getattr(brain, "state", "-"))
    job = getattr(brain, "job", None)
    pos = d.trunk_pos(w.data)
    who = (job or kind)[:5]
    speed = f"{d.heading_speed(w.data):.2f}".replace("0.", ".")     # 35 columns, exactly
    lines = [f"{d.id} {who}/{state[:7]:7s} v{speed} f{d.falls} {pos[0]:+.2f},{pos[1]:+.2f}"]
    extra = []
    if d.holding:
        extra.append(f"hold={d.holding}")
    if d.skill:
        extra.append(f"skill={d.skill}")
    if intent is not None and intent.note and intent.note != state:
        extra.append(intent.note)
    if extra:
        lines.append("   " + " ".join(extra))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", help="builtin name (living-room, playroom, pitch-2v2, ...), a user scenario name, or a .json path")
    ap.add_argument("--seconds", type=float, default=30.0, help="sim seconds to record")
    ap.add_argument("--skip", type=float, default=0.0, help="sim seconds to run BEFORE recording starts (see late play)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/rw")
    ap.add_argument("--camera", default="top", help="top | follow:<duck> | side | front | three-quarter")
    ap.add_argument("--brain", action="append", default=[], metavar="DUCK=KIND",
                    help="override a duck's brain (repeatable), e.g. d0=tidy, d1=wander")
    ap.add_argument("--tether-ms", type=float, default=0.0, help="brain round-trip latency, as the lab and eval-tidy apply it")
    ap.add_argument("--stride", type=int, default=5, help="render every Nth control step (50 Hz); 5 = 10 fps of sim time")
    ap.add_argument("--fps", type=int, default=10, help="playback fps; stride 5 + fps 10 = real time")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--sheet-frames", type=int, default=12)
    args = ap.parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    import imageio.v2 as imageio
    import mujoco

    from .viz_server import load_policy_infer
    from .world_server import WorldState

    sc = resolve(args.scenario)
    st = WorldState(load_infer=load_policy_infer)
    st.tether_ms = args.tether_ms
    st.world, st.scenario = st.build(sc, seed=args.seed), sc
    st.metrics = st.new_metrics()
    w = st.world
    for did, kind in parse_brains(args.brain).items():
        if did not in w.ducks:
            raise SystemExit(f"--brain {did}=...: no such duck in {sc.name} ({', '.join(w.ducks)})")
        st.set_brain(did, kind)
    for line in st.events:                       # a missing policy or brain says so here, not silently
        print(f"[build] {line}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(w.model, height=args.height, width=args.width)
    # What to DRAW. MuJoCo's default geomgroup is [1,1,1,0,0,0], and the toys
    # are group 4 (`world.compose.PICKABLE_GROUP`, so the detector's
    # line-of-sight looks THROUGH them) - so every playroom clip ever recorded
    # showed a duck tidying an empty room, basket and all, with the six toys
    # invisible. Draw scenery (0), the hinged bill (1), the robot's visual
    # meshes (2) and the toys (4); leave the collision pads (3) off, they sit
    # on top of the visuals.
    opt = mujoco.MjvOption()
    for g, on in enumerate((1, 1, 1, 0, 1, 0)):
        opt.geomgroup[g] = on
    cam, follow = make_camera(args.camera, sc)
    cmd = np.zeros(3, np.float32)

    frames: list[np.ndarray] = []
    caps: list[list[str]] = []
    fell: list[bool] = []
    events: list[str] = [f"# {sc.name} seed {args.seed} — ducks: " + ", ".join(
        f"{d.id}={getattr(st.brains.get(d.id), 'kind', 'script')}" for d in w.ducks.values())]
    last_state = {did: None for did in w.ducks}
    last_hold = {did: None for did in w.ducks}
    falls = {did: 0 for did in w.ducks}
    goal_seq = w.goal_seq
    fell_since_tile = False
    t_end = args.skip + args.seconds

    def log(msg: str) -> None:
        events.append(f"t={w.t:7.2f}  {msg}")

    while w.t < t_end:
        st.drive(cmd, "auto")
        w.step()
        if st.metrics is not None:
            st.metrics.tick()
        st.after_step()
        # -- the event log: what changed this tick --------------------------
        for d in w.ducks.values():
            brain = st.brains.get(d.id)
            state = getattr(brain, "state", None)
            if state != last_state[d.id]:
                note = getattr(st.intents.get(d.id), "note", "") or ""
                log(f"{d.id} -> {state}" + (f"  ({note})" if note else ""))
                last_state[d.id] = state
            if d.holding != last_hold[d.id]:
                log(f"{d.id} {'holds ' + d.holding if d.holding else 'released ' + str(last_hold[d.id])}")
                last_hold[d.id] = d.holding
            if d.falls > falls[d.id]:
                falls[d.id] = d.falls
                fell_since_tile = True
                pos = d.trunk_pos(w.data)
                log(f"FALL {d.id} #{d.falls} state={state} skill={d.skill} holding={d.holding} "
                    f"after respawn @{pos[0]:+.2f},{pos[1]:+.2f}")
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            log(f"GOAL {w.last_goal} — kickoff, score {w.goals.get('left', 0)}-{w.goals.get('right', 0)}")
        if w.t < args.skip or w.tick % args.stride:
            continue
        # -- a frame -----------------------------------------------------------
        if follow is not None:
            p = w.ducks[follow].trunk_pos(w.data)
            cam.lookat[:] = (p[0], p[1], LOOKAT_Z)
        renderer.update_scene(w.data, cam, opt)
        frames.append(renderer.render().copy())
        s = score_line(w, st)
        caps.append([f"t {w.t:5.1f}s" + (" " + s if s else "")]     # <= 35 columns
                    + [ln for d in w.ducks.values() for ln in duck_lines(w, d, st)])
        fell.append(fell_since_tile)
        fell_since_tile = False

    if not frames:
        raise SystemExit("nothing recorded — --seconds too short for --stride?")
    (out / "events.txt").write_text("\n".join(events) + "\n")
    idx = sheet_indices(len(frames), args.sheet_frames)
    total_falls = sum(falls.values())
    header = [f"{sc.name} — seed {args.seed}, {args.seconds:g} s recorded"
              + (f" after {args.skip:g} s skipped" if args.skip else "") + f", camera {args.camera}"
              + (f", tether {args.tether_ms:g} ms" if args.tether_ms else ""),
              "ducks: " + ", ".join(
                  f"{d.id} {getattr(st.brains.get(d.id), 'kind', 'script')}"
                  + (f"+{getattr(st.brains.get(d.id), 'job', None)}" if getattr(st.brains.get(d.id), 'job', None) else "")
                  + f" ({d.policy_id or 'stand'})" for d in w.ducks.values()),
              f"falls {total_falls} ({', '.join(f'{k} {v}' for k, v in falls.items())})"
              + ("  " + score_line(w, st, final=True) if score_line(w, st) else "")]
    footer = ["per duck: role-or-brain/state v<speed m/s> f<falls> x,y; a second line for hold= skill= and the brain's note.",
              "a highlighted tile is one where a duck FELL since the previous tile (events.txt has the second).",
              "read events.txt for every transition, fall, pick, release and goal, with sim time."]
    build_sheet([frames[i] for i in idx], [caps[i] for i in idx], header, footer,
                [fell[i] for i in idx], out / "sheet.png")
    imageio.mimsave(out / "world.mp4", frames, fps=args.fps, macro_block_size=1)
    print("\n".join(events[-40:]))
    print(f"\nwrote {out / 'world.mp4'} ({len(frames)} frames), {out / 'sheet.png'}, {out / 'events.txt'}")
    print(f"falls {total_falls}" + (f"; {score_line(w, st, final=True)}" if score_line(w, st) else ""))
    print("READ sheet.png and events.txt before saying what the ducks did.")


if __name__ == "__main__":
    main()
