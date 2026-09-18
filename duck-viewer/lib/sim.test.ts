// The brain menu files learned runs under their use case; this is the
// arithmetic behind the headings (lib/sim.ts groupLearned).

import { describe, expect, it } from "vitest";

import { applyFloorClick, makePitch, makeRoom } from "@/components/SimEditor";
import { goalDefenders, groupLearned, LEARNED_GROUPS, lidarRingPoints, PITCH_TEAMS, RUG_LONG_MAX,
  RUG_SHORT_MAX, rugSize, SIM_SPEEDS, SIM_SPEED_DEFAULT, simRate, SimClient, speedLabel,
  speedShortfall, stepSpeed, type LearnedInfo, type Scenario } from "./sim";

const b = (name: string, group: string | null, title: string | null = null): LearnedInfo => ({
  name, group, title, description: null,
});

describe("groupLearned", () => {
  it("files by group in menu order and skips empty groups", () => {
    const out = groupLearned([b("z1-s81", "null-pair"), b("follow-v4", "shipped-followers", "Follower v4"), b("z2-s81", "null-pair")]);
    expect(out.map(([label]) => label)).toEqual(["Followers (shipped)", "Null pair (seeds 81–84)"]);
    expect(out[1][1].map((x) => x.name)).toEqual(["z1-s81", "z2-s81"]);
  });

  it("files an unknown or missing group under Other, never drops it", () => {
    const out = groupLearned([b("striker-v1", null), b("odd", "no-such-group")]);
    expect(out).toEqual([["Other", [b("striker-v1", null), b("odd", "no-such-group")]]]);
  });

  it("returns nothing for nothing", () => {
    expect(groupLearned([])).toEqual([]);
    expect(LEARNED_GROUPS.at(-1)![0]).toBe("other");
  });
});

import { menuBrains } from "./sim";

describe("menuBrains", () => {
  const runs: LearnedInfo[] = [
    b("follow-v4", "shipped-followers", "Follower v4"),
    b("follow-v1", "shipped-followers", "Follower v1"),
    b("p-n256-s31", "capacity"), b("p-n256-s32", "capacity"),
    b("z1-s81", "null-pair"),
  ];

  it("offers only the shipped brains by default, and says how many it hid", () => {
    const m = menuBrains(runs, "wander", false);
    expect(m.groups.map(([label, bs]) => [label, bs.length])).toEqual([["Followers (shipped)", 2]]);
    expect(m.hidden).toBe(3);
  });

  it("never hides the brain the duck is on", () => {
    const m = menuBrains(runs, "learned:z1-s81", false);
    expect(m.groups.map(([label]) => label)).toEqual(["Followers (shipped)", "Null pair (seeds 81–84)"]);
    expect(m.hidden).toBe(2);
  });

  it("shows everything when asked", () => {
    const m = menuBrains(runs, null, true);
    expect(m.groups.length).toBe(3);
    expect(m.hidden).toBe(0);
  });
});

describe("applyFloorClick: placing a duck on a pitch", () => {
  // The editor's job in Track 4.2: a duck placed on a pitch joins the team of
  // the half it stands in and faces the goal that team attacks. Any other
  // placement is refused by the server on save (a team facing both goals),
  // so the default has to be the legal one.
  const pitch = (): Scenario => ({
    version: 1,
    name: "p",
    seed: 0,
    floor: { size: [4, 3] },
    walls: [],
    boxes: [],
    balls: [{ pos: [0, 0], radius: 0.035, mass: 0.015 }],
    ducks: [],
    goal_width: 0.7,
    collision: "walk",
  });
  const place = (draft: Scenario, x: number) =>
    applyFloorClick({ draft, tool: "duck", wallStart: null }, x, 0).draft.ducks.at(-1)!;

  it("puts a duck in its own half's team, facing the other goal", () => {
    const [home, away] = PITCH_TEAMS;
    const a = place(pitch(), -1);
    expect([a.team, a.brain, a.spawn[2]]).toEqual([home, "chase", 0]);
    const b = place({ ...pitch(), ducks: [a] }, 1);
    expect([b.team, b.brain, b.spawn[2]]).toEqual([away, "chase", Math.PI]);
  });

  it("follows the teams already on the pitch rather than assuming the default pair", () => {
    const sky = { ...place(pitch(), -1), team: "sky" as const };
    const mate = place({ ...pitch(), ducks: [sky] }, -1.2);
    expect(mate.team).toBe("sky");
    expect(place({ ...pitch(), ducks: [sky] }, 1.2).team).toBe(PITCH_TEAMS[1]);
  });

  it("leaves a duck teamless off a pitch", () => {
    const room = { ...pitch(), goal_width: 0 };
    const d = place(room, -1);
    expect(d.team).toBeUndefined();
    expect(d.brain).toBeUndefined();
  });
});

describe("makePitch / makeRoom: a pitch drawn in the editor is the lab's pitch", () => {
  // The lab's pitch builtins carry a 15 cm cove and 30 cm chamfered corners
  // (world_server.PITCH_COVE / PITCH_CORNER); a pitch toggled on in the
  // editor must match them or it plays a different game than /sim's own.
  const wall = (a: [number, number], b: [number, number]) => ({ from: a, to: b, height: 0.3, thickness: 0.02 });
  const room = (walls: Scenario["walls"]): Scenario => ({
    version: 1, name: "r", seed: 0, floor: { size: [3.9, 3.35] }, walls, boxes: [], balls: [], ducks: [], collision: "all",
  });
  const rect = room([
    wall([-1.7, -1.425], [1.7, -1.425]), wall([1.7, -1.425], [1.7, 1.425]),
    wall([1.7, 1.425], [-1.7, 1.425]), wall([-1.7, 1.425], [-1.7, -1.425]),
  ]);
  it("gives a rectangular room the cove and chamfered corners, and takes them back", () => {
    const p = makePitch(rect);
    expect(p.goal_width).toBe(0.7);
    expect(p.cove).toBe(0.15);
    expect(p.walls).toHaveLength(8);
    for (let i = 0; i < 8; i++) expect(p.walls[i].to).toEqual(p.walls[(i + 1) % 8].from);
    expect(p.walls[0].from).toEqual([-1.4, -1.425]);          // 0.3 in from the corner, as make_pitch(corner=0.3)
    expect(p.walls[1].to).toEqual([1.7, -1.125]);
    expect(p.balls).toHaveLength(1);
    const r = makeRoom(p);
    expect(r.goal_width).toBe(0);
    expect(r.cove).toBe(0);
    expect(r.walls).toHaveLength(4);
    expect(r.walls.map((w) => w.from)).toEqual([[-1.7, -1.425], [1.7, -1.425], [1.7, 1.425], [-1.7, 1.425]]);
  });
  it("leaves a room that is not a plain rectangle as drawn, cove and all", () => {
    const ell = room([...rect.walls, wall([0, -1.425], [0, 0])]);   // an interior wall: the user's layout
    const p = makePitch(ell);
    expect(p.cove).toBe(0.15);
    expect(p.walls).toEqual(ell.walls);
    expect(makeRoom(p).walls).toEqual(ell.walls);
  });
  it("draws the sides in either direction and still finds the rectangle", () => {
    const flipped = room(rect.walls.map((w) => wall(w.to, w.from)).reverse());
    expect(makePitch(flipped).walls).toHaveLength(8);
  });
});

describe("goalDefenders: whose end is which", () => {
  // The stage paints a goal frame in the colours of the team that KEEPS it,
  // and the mouth keys are the World's (`right` is the mouth at +x), so the
  // team spawned at −x facing +x defends `left`. Getting this backwards paints
  // both ends the wrong colour, which is worse than painting neither.
  const pitch = (ducks: Scenario["ducks"], attacks?: Scenario["attacks"]): Scenario => ({
    version: 1, name: "p", seed: 0, floor: { size: [4, 3] }, walls: [], boxes: [],
    balls: [{ pos: [0, 0], radius: 0.035, mass: 0.015 }], ducks, goal_width: 0.7,
    attacks, collision: "walk",
  });
  const duck = (id: string, x: number, yaw: number, team: string): Scenario["ducks"][number] =>
    ({ id, spawn: [x, 0, yaw], policy: null, tof: null, team: team as never });
  const [home, away] = PITCH_TEAMS;

  it("reads the spawn headings when the scenario declares nothing", () => {
    const s = pitch([duck("d0", -1, 0, home), duck("d1", 1, Math.PI, away)]);
    expect(goalDefenders(s)).toEqual({ left: home, right: away });
  });

  it("takes the declaration over the heading — a defender faces its OWN goal", () => {
    const s = pitch([duck("d0", -1, Math.PI, home), duck("d1", 1, Math.PI, away)],
                    { [home]: "right", [away]: "left" } as Scenario["attacks"]);
    expect(goalDefenders(s)).toEqual({ left: home, right: away });
  });

  it("leaves the far end unpainted on a one-team pitch, and both off a pitch", () => {
    expect(goalDefenders(pitch([duck("d0", -1, 0, home)]))).toEqual({ left: home, right: null });
    expect(goalDefenders({ ...pitch([duck("d0", -1, 0, home)]), goal_width: 0 }))
      .toEqual({ left: null, right: null });
    expect(goalDefenders(null)).toEqual({ left: null, right: null });
  });

  it("leaves an end unpainted when two teams attack it", () => {
    const s = pitch([duck("d0", -1, 0, home), duck("d1", 1, Math.PI, away), duck("d2", 1, Math.PI, "sky")]);
    expect(goalDefenders(s)).toEqual({ left: home, right: null });
  });
});

// The speed control's arithmetic: what [ and ] land on, and when the HUD is
// entitled to say the lab is not keeping up.
describe("sim speed presets", () => {
  it("steps through the presets and stops at both ends", () => {
    expect(SIM_SPEEDS.map((x) => stepSpeed(x, 1))).toEqual([0.5, 1, 2, 4, 8, 8]);
    expect(SIM_SPEEDS.map((x) => stepSpeed(x, -1))).toEqual([0.25, 0.25, 0.5, 1, 2, 4]);
  });

  it("moves an off-preset speed towards the neighbour it is heading for", () => {
    expect(stepSpeed(3, 1)).toBe(4);
    expect(stepSpeed(3, -1)).toBe(2);
  });

  it("labels a speed the way a player would", () => {
    expect([0.25, 1, 8].map(speedLabel)).toEqual(["0.25\u00d7", "1\u00d7", "8\u00d7"]);
  });

  it("calls a shortfall only when the lab measurably missed the ask", () => {
    expect(speedShortfall(3.0, 4)).toBe(true);      // a 3v3 at 4x: the measured ceiling
    expect(speedShortfall(3.9, 4)).toBe(false);     // within the RTF window's own jitter
    expect(speedShortfall(1.0, 1)).toBe(false);
  });

  it("reads only an absent measurement as fine, not a slow one", () => {
    // The lab reports exactly 0 until a window closes, and zeroes it on every
    // speed change — that is the one value meaning "no number yet".
    expect(speedShortfall(0, 4)).toBe(false);
    expect(speedShortfall(0, 0.25)).toBe(false);
    // An absolute floor (the first cut used rtf > 0.05) reported a frozen
    // world as healthy at slow speeds, which is exactly where the loop's
    // own starvation bug lived: 0.04 against an asked 0.25 is a 6x miss.
    expect(speedShortfall(0.04, 0.25)).toBe(true);
    expect(speedShortfall(0.24, 0.25)).toBe(false);
  });
});

// The pose smoothers that read `simRate` are shared with the lab pages, which
// stream at 1x and have no speed knob — so a /sim speed must not outlive the
// socket that set it.
describe("the shared speed store", () => {
  it("hands the speed back when the sim socket closes", () => {
    const orig = globalThis.WebSocket;
    // A stub socket: SimClient only needs the constructor and close().
    class FakeWS {
      static OPEN = 1;
      readyState = 1;
      onopen: (() => void) | null = null;
      onmessage: ((e: { data: string }) => void) | null = null;
      onclose: (() => void) | null = null;
      onerror: (() => void) | null = null;
      close() {}
      send() {}
    }
    (globalThis as unknown as { WebSocket: unknown }).WebSocket = FakeWS;
    try {
      const c = new SimClient();
      simRate.speed = 8;                 // as a frame at 8x would leave it
      c.close();
      expect(simRate.speed).toBe(SIM_SPEED_DEFAULT);
    } finally {
      (globalThis as unknown as { WebSocket: unknown }).WebSocket = orig;
      simRate.speed = SIM_SPEED_DEFAULT;
    }
  });
});

describe("rugSize", () => {
  // The rooms the /sim page actually draws, and what a real rug is in them.
  it("fills a big room instead of leaving a doormat", () => {
    // follow-me / flock: 6 x 5 m of walls. The old rule capped this at
    // 1.6 x 1.2 — a 5 x 4 ft accent rug in a 20 x 16 ft room.
    const [w, h] = rugSize(6.0, 5.0);
    expect(w).toBeCloseTo(3.3, 2);
    expect(h).toBeCloseTo(2.74, 2);
    expect(w).toBeGreaterThan(1.6);
    expect(h).toBeGreaterThan(1.2);
  });

  it("stays proportional in a small room", () => {
    // living-room / playroom: 3 x 2.5 m — under the caps, so it scales.
    expect(rugSize(3.0, 2.5)).toEqual([3.0 * 0.55, 2.5 * 0.6]);
  });

  it("never outgrows a 9 x 12 ft rug, however big the room", () => {
    expect(rugSize(40, 40)).toEqual([RUG_LONG_MAX, RUG_SHORT_MAX]);
  });

  it("lies along the room's long axis, portrait rooms included", () => {
    const [w, h] = rugSize(5.0, 6.0);
    expect(h).toBeGreaterThan(w);
    expect(rugSize(5.0, 6.0)).toEqual(rugSize(6.0, 5.0).slice().reverse());
  });

  it("leaves floor showing on every side", () => {
    for (const [rw, rh] of [[3.0, 2.5], [6.0, 5.0], [6.5, 5.5], [1.3, 1.3]] as const) {
      const [w, h] = rugSize(rw, rh);
      expect(w).toBeLessThan(rw);
      expect(h).toBeLessThan(rh);
    }
  });
});

describe("lidarRingPoints", () => {
  const scan = (mm: number[], a0 = 0, da = Math.PI / 2, maxRange = 6) => ({ a0, da, mm, maxRange });

  it("puts the robot's FORWARD at the top of the drawing", () => {
    // MuJoCo bearings are CCW from +x and SVG's y grows downward. A return
    // dead ahead at half range must land ABOVE the centre; getting the sign
    // wrong draws a room mirrored front-to-back, which reads as plausible.
    const [p] = lidarRingPoints(scan([3000]), 100);
    expect(p.x).toBeCloseTo(50, 6);
    expect(p.y).toBeCloseTo(25, 6); // above centre (y = 50)
  });

  it("rebuilds every bearing from a0 + i·da", () => {
    // 4 rays a quarter turn apart: ahead, left, behind, right.
    const pts = lidarRingPoints(scan([6000, 6000, 6000, 6000], 0, Math.PI / 2, 6), 100);
    expect(pts).toHaveLength(4);
    expect(pts[0].x).toBeCloseTo(50, 6); // ahead → top
    expect(pts[0].y).toBeCloseTo(0, 6);
    expect(pts[1].x).toBeCloseTo(0, 6); // the robot's LEFT → left
    expect(pts[1].y).toBeCloseTo(50, 6);
    expect(pts[2].y).toBeCloseTo(100, 6); // behind → bottom
    expect(pts[3].x).toBeCloseTo(100, 6); // the robot's right → right
  });

  it("drops a 0 — that is NO READING, not a contact on the robot", () => {
    // The payload's own convention (sensors/lidar.LidarFrame.as_payload).
    // Drawing them puts a blob of false contacts on the origin.
    expect(lidarRingPoints(scan([0, 0, 0, 0]), 100)).toEqual([]);
    expect(lidarRingPoints(scan([0, 3000]), 100)).toHaveLength(1);
  });

  it("clamps past max range instead of dropping the direction", () => {
    const [p] = lidarRingPoints(scan([99000]), 100);
    expect(p.y).toBeCloseTo(0, 6); // on the inscribed circle, not beyond it
  });

  it("falls back to a 6 m scale for a lab that sends no maxRange", () => {
    const [p] = lidarRingPoints({ a0: 0, da: 0, mm: [6000] }, 100);
    expect(p.y).toBeCloseTo(0, 6);
  });
});
