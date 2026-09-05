// The inspector's brain strip and gauges rest on this layout knowledge;
// pinned against the contract in brain_env.py (version 2) and striker.py.

import { describe, expect, it } from "vitest";
import { type BrainView, atBound, gauge, normalizeObs, obsGroups, obsSlots, targetReadout } from "./brainview";

const view = (obs: number[], raw = [0.3, 0, -0.5]): BrainView => ({
  obs,
  obs_version: 2,
  decide_every: 5,
  act: { raw, clipped: raw.map((r, i) => Math.min(0.6, Math.max([-0.2, -0.3, -1][i], r))), low: [-0.2, -0.3, -1], high: [0.6, 0.3, 1] },
});

describe("obs layout", () => {
  it("knows the 80-slot follow contract and the 88-slot striker one", () => {
    expect(obsSlots(80)).toHaveLength(80);
    expect(obsSlots(80)[64].label).toBe("tof age");
    expect(obsSlots(80)[65].group).toBe("target");
    expect(obsSlots(80)[72].label).toBe("since seen");
    expect(obsSlots(80)[78].label).toBe("yaw rate");
    expect(obsSlots(88)[87].label).toBe("kick busy");
    expect(obsSlots(7).every((s) => s.group === "obs")).toBe(true);
  });

  it("groups cover every slot contiguously and in order", () => {
    const g = obsGroups(80);
    expect(g[0]).toEqual({ group: "tof", start: 0, end: 64 });
    expect(g.map((x) => x.group)).toEqual(["tof", "age", "target", "age", "since", "act", "speed", "track"]);
    expect(g[g.length - 1].end).toBe(80);
    for (let i = 1; i < g.length; i++) expect(g[i].start).toBe(g[i - 1].end);
  });
});

describe("normalizeObs", () => {
  it("maps each slot by its own range, with the brain's bounds for the last action", () => {
    const obs = new Array(80).fill(0);
    obs[0] = 2;          // 2 m of 4 → half
    obs[66] = 0;         // straight ahead → half
    obs[65] = 1;         // hit → full
    obs[69] = 8;         // range past the clip → full, not beyond
    obs[73] = 0.6;       // last vx at the brain's own top bound → full
    obs[75] = -1;        // last wz at the bottom bound → empty
    const n = normalizeObs(view(obs));
    expect(n[0]).toBeCloseTo(0.5);
    expect(n[66]).toBeCloseTo(0.5);
    expect(n[65]).toBe(1);
    expect(n[69]).toBe(1);
    expect(n[73]).toBeCloseTo(1);
    expect(n[75]).toBe(0);
    expect(n).toHaveLength(80);
  });
});

describe("actions", () => {
  it("flags an action pinned on its bound", () => {
    // The exported graph clamps, so raw == clipped; the tell is the edge.
    const v = view(new Array(80).fill(0), [0.9, 0, -0.5]);
    expect(atBound(v.act)).toEqual([true, false, false]);
    expect(atBound(view(new Array(80).fill(0), [0.6, 0.3, -1]).act)).toEqual([true, true, true]);
    expect(atBound(view(new Array(80).fill(0), [0.3, 0.1, 0.2]).act)).toEqual([false, false, false]);
  });

  it("places zero, the value and the raw ask on the gauge track", () => {
    const v = view(new Array(80).fill(0), [0.9, 0, -0.5]);
    const g = gauge(v.act, 0);   // vx: -0.2..0.6
    expect(g.zero).toBeCloseTo(0.25);
    expect(g.value).toBeCloseTo(1);     // clipped to 0.6
    expect(g.raw).toBeCloseTo(1);       // raw pinned at the edge
    expect(gauge(v.act, 2).value).toBeCloseTo(0.25);   // wz -0.5 of -1..1
  });
});

describe("targetReadout", () => {
  it("says no target with the time since it was seen", () => {
    const obs = new Array(80).fill(0);
    obs[72] = 5;
    expect(targetReadout(view(obs))).toBe("no target · last seen 5+ s ago");
  });
  it("reads a hit track in degrees, metres and confidence", () => {
    const obs = new Array(80).fill(0);
    obs[65] = 1; obs[66] = 0.2; obs[67] = -0.1; obs[68] = 0.3; obs[69] = 1.4; obs[70] = 0.9; obs[79] = 1;
    expect(targetReadout(view(obs))).toBe("● hit · 11° · -6° up · 1.40 m · w 0.30 · conf 0.90 · confirmed");
  });
  it("marks a coasting track and how long ago it was hit", () => {
    const obs = new Array(80).fill(0);
    obs[66] = 0.5; obs[69] = 2; obs[72] = 0.8; obs[77] = 1;
    expect(targetReadout(view(obs))).toContain("○ coasting");
    expect(targetReadout(view(obs))).toContain("seen 0.8 s ago");
  });
});
