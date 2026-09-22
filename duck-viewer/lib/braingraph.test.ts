import { describe, expect, it } from "vitest";
import {
  type BrainGraphInfo,
  LAYOUT,
  SCALE_MAX,
  SCALE_MIN,
  clampScale,
  labelWidth,
  StateTrace,
  edgeAlpha,
  edgeKey,
  edgePath,
  layoutGraph,
  TRAIL,
} from "./braingraph";

const G: BrainGraphInfo = {
  key: "chase",
  title: "Chase",
  note: "",
  groups: [
    ["find", "find the ball"],
    ["go", "go to it"],
  ],
  nodes: [
    { name: "search", group: "find", note: "turn and look" },
    { name: "hunt", group: "find", note: "keep driving" },
    { name: "seek", group: "find", note: "walk to memory" },
    { name: "look", group: "find", note: "sweep the head" },
    { name: "turn", group: "find", note: "turn first" },
    { name: "chase", group: "go", note: "walk at it" },
  ],
};

describe("layoutGraph", () => {
  it("wraps a group's chips at the row width and starts each group below the last", () => {
    const l = layoutGraph(G);
    const perRow = Math.floor((LAYOUT.width + LAYOUT.gapX) / (LAYOUT.chipW + LAYOUT.gapX));
    expect(perRow).toBe(4);
    const find = l.nodes.filter((n) => n.group === "find");
    // five chips, four per row: the fifth wraps under the first
    expect(find[4].x).toBe(find[0].x);
    expect(find[4].y).toBeGreaterThan(find[0].y);
    // and the next group's heading clears the wrapped row
    const go = l.nodes.find((n) => n.group === "go")!;
    expect(go.y).toBeGreaterThan(find[4].y);
  });

  it("places every declared node exactly once, inside the box it reports", () => {
    const l = layoutGraph(G);
    expect(l.nodes.map((n) => n.name).sort()).toEqual(G.nodes.map((n) => n.name).sort());
    for (const n of l.nodes) {
      expect(n.x).toBeGreaterThanOrEqual(0);
      expect(n.x + n.w).toBeLessThanOrEqual(l.width);
      expect(n.y + n.h).toBeLessThanOrEqual(l.height);
    }
  });

  it("is deterministic — a node does not move when nothing about the graph changed", () => {
    expect(layoutGraph(G)).toEqual(layoutGraph(G));
  });

  it("never divides by zero when the panel is narrower than one chip", () => {
    const l = layoutGraph(G, { ...LAYOUT, width: 10 });
    expect(l.nodes).toHaveLength(G.nodes.length);
    // one per row, so every chip shares the left edge
    expect(new Set(l.nodes.map((n) => n.x))).toEqual(new Set([0]));
  });
});

describe("StateTrace", () => {
  it("counts entries and edges, and does not book an edge for a state that did not change", () => {
    const tr = new StateTrace();
    tr.step("search", 0);
    tr.step("search", 0.1);
    tr.step("chase", 0.2);
    expect(tr.visits.get("search")!.count).toBe(1);
    expect(tr.edges.size).toBe(1);
    expect(tr.edges.get(edgeKey("search", "chase"))!.count).toBe(1);
    expect(tr.current).toBe("chase");
  });

  it("accumulates dwell across separate spells in the same state", () => {
    const tr = new StateTrace();
    tr.step("search", 0);
    tr.step("search", 1);      // 1 s banked
    tr.step("chase", 1);
    tr.step("chase", 2);
    tr.step("search", 2);
    tr.step("search", 3);      // 1 s more
    expect(tr.visits.get("search")!.total).toBeCloseTo(2, 6);
    expect(tr.visits.get("search")!.count).toBe(2);
  });

  it("reports dwell in the current state, not the total", () => {
    const tr = new StateTrace();
    tr.step("search", 10);
    expect(tr.dwell(12.5)).toBeCloseTo(2.5, 6);
    tr.step("chase", 13);
    expect(tr.dwell(13.25)).toBeCloseTo(0.25, 6);
  });

  it("starts over when sim time goes backwards — a restart is not a transition", () => {
    const tr = new StateTrace();
    tr.step("search", 5);
    tr.step("chase", 6);
    expect(tr.edges.size).toBe(1);
    tr.step("search", 0);      // R restarted the world
    expect(tr.edges.size).toBe(0);
    expect(tr.current).toBe("search");
    expect(tr.visits.get("search")!.count).toBe(1);
    expect(tr.dwell(0.5)).toBeCloseTo(0.5, 6);
  });

  it("never books negative dwell across a scrub", () => {
    const tr = new StateTrace();
    tr.step("search", 5);
    tr.step("search", 1);
    for (const v of tr.visits.values()) expect(v.total).toBeGreaterThanOrEqual(0);
  });

  it("says when the drawn set changed, so the panel re-renders only then", () => {
    const tr = new StateTrace();
    expect(tr.step("search", 0)).toBe(true);      // new node
    expect(tr.step("search", 1)).toBe(false);     // same state
    expect(tr.step("chase", 2)).toBe(true);       // new node and edge
    expect(tr.step("search", 3)).toBe(true);      // new edge back
    expect(tr.step("chase", 4)).toBe(false);      // both already drawn
  });

  it("keeps the trail bounded and newest-last", () => {
    const tr = new StateTrace();
    for (let i = 0; i < TRAIL + 6; i++) tr.step(i % 2 ? "search" : "chase", i);
    expect(tr.trail).toHaveLength(TRAIL);
    expect(tr.trail[tr.trail.length - 1]).toBe(tr.current);
  });

  it("ignores a null state — a duck with no graph must not clear the trace", () => {
    const tr = new StateTrace();
    tr.step("search", 0);
    expect(tr.step(null, 1)).toBe(false);
    expect(tr.current).toBe("search");
  });
});

describe("edge drawing", () => {
  const a = { name: "a", group: "g", note: "", x: 0, y: 0, w: 74, h: 20 };
  const b = { name: "b", group: "g", note: "", x: 0, y: 60, w: 74, h: 20 };

  it("bows the two directions of a pair apart, so both are visible", () => {
    expect(edgePath(a, b)).not.toBe(edgePath(b, a));
  });

  it("emits a path that starts and ends at the chip centres", () => {
    expect(edgePath(a, b)).toMatch(/^M 37 10 Q [-\d.]+ [-\d.]+ 37 70$/);
  });

  it("fades an edge with age but never below a floor that grows with use", () => {
    const fresh = { from: "a", to: "b", count: 1, t: 10 };
    expect(edgeAlpha(fresh, 10)).toBe(1);
    expect(edgeAlpha(fresh, 14)).toBeCloseTo(0.5, 6);
    expect(edgeAlpha(fresh, 1000)).toBeCloseTo(0.37, 6);
    const worn = { from: "a", to: "b", count: 20, t: 10 };
    expect(edgeAlpha(worn, 1000)).toBeCloseTo(0.8, 6);    // capped, so nothing is opaque
  });
});

describe("panel scale", () => {
  it("clamps to a range where the chip text is still readable", () => {
    expect(clampScale(1)).toBe(1);
    expect(clampScale(0.1)).toBe(SCALE_MIN);
    expect(clampScale(99)).toBe(SCALE_MAX);
  });

  it("survives a corrupt persisted value rather than collapsing the panel", () => {
    // Anything not finite is corrupt localStorage, not a very large request:
    // fall back to the default rather than to the ceiling.
    expect(clampScale(NaN)).toBe(1);
    expect(clampScale(Infinity)).toBe(1);
    expect(clampScale(-Infinity)).toBe(1);
  });

  it("sizes a label patch from its text, not from a measured box", () => {
    expect(labelWidth("find the ball", 9)).toBeCloseTo(13 * 9 * 0.6, 6);
    expect(labelWidth("", 9)).toBe(0);
  });
});
