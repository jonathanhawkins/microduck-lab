// "Our runs" grouped by trick, and the run a 🎓 chip's ▶ plays.
//
// The rule these pin: only a MEASUREMENT outranks recency. A trick's lead is
// its measured pick plus its newest run — never "the last one" dressed up as
// the best one, which is how the G1 kick chain handed out its worst stage.

import { describe, expect, it } from "vitest";
import type { Policy } from "./lab";
import { bestRunFor, rowKey, runRows, splitTail, trickGroups } from "./tricks";

function run(label: string, extra: Partial<Policy> = {}): Policy {
  return { id: `run:${label}`, label, group: "runs", path: `runs/${label}/policy.onnx`, ...extra };
}

// Newest first, as the lab sends them.
const LIST: Policy[] = [
  run("kick-new", { trick: "g1_front_kick", robot: "g1", mtime: 900 }),
  run("teach-g1_punch-aa-s1", { trick: "g1_punch", robot: "g1", mtime: 800, chain: "teach-g1_punch-aa", stage: 1 }),
  run("teach-kick-bb-s3", { trick: "g1_front_kick", robot: "g1", mtime: 700, chain: "teach-kick-bb", stage: 3 }),
  run("teach-kick-bb-s2", { trick: "g1_front_kick", robot: "g1", mtime: 600, chain: "teach-kick-bb", stage: 2, pick: true }),
  run("kick-old", { trick: "g1_front_kick", robot: "g1", mtime: 500 }),
  run("ab-kl-05", { mtime: 400 }),
];

describe("trickGroups", () => {
  const groups = trickGroups(runRows(LIST))!;

  it("orders tricks by their newest run and keeps unplaced runs apart", () => {
    expect(groups.map((g) => g.trick)).toEqual(["g1_front_kick", "g1_punch", ""]);
    expect(groups[0].newest).toBe(900);
  });

  it("leads with the measured pick, then the newest run", () => {
    const kick = groups[0];
    expect(kick.lead.map(rowKey)).toEqual(["teach-kick-bb", "run:kick-new"]);
    expect(kick.rest.map(rowKey)).toEqual(["run:kick-old"]);
  });

  it("leads with the newest alone when nothing was measured", () => {
    const unpicked = trickGroups(runRows(LIST.map((p) => ({ ...p, pick: false }))))!;
    expect(unpicked[0].lead.map(rowKey)).toEqual(["run:kick-new"]);
    expect(unpicked[0].rest.map(rowKey)).toEqual(["teach-kick-bb", "run:kick-old"]);
  });

  it("shows one row when the pick IS the newest", () => {
    const g = trickGroups(runRows([run("a", { trick: "spin", pick: true, mtime: 2 }), run("b", { trick: "spin", mtime: 1 })]))!;
    expect(g[0].lead.map(rowKey)).toEqual(["run:a"]);
  });

  it("never loses a row", () => {
    const seen = groups.flatMap((g) => [...g.lead, ...g.rest]).map(rowKey).sort();
    expect(seen).toEqual(runRows(LIST).map(rowKey).sort());
  });

  it("is null for a lab that names no tricks, so the list stays flat", () => {
    expect(trickGroups(runRows([run("a"), run("b")]))).toBeNull();
  });
});

describe("bestRunFor", () => {
  it("finds the measured pick of that trick on that robot", () => {
    expect(bestRunFor(LIST, "g1_front_kick", "g1")?.label).toBe("teach-kick-bb-s2");
  });

  it("offers nothing when no run of the trick was ever measured best", () => {
    expect(bestRunFor(LIST, "g1_punch", "g1")).toBeUndefined();
  });

  it("does not cross robots, and an unnamed trick matches nothing", () => {
    expect(bestRunFor(LIST, "g1_front_kick", "microduck")).toBeUndefined();
    // The duck-only fallback chips carry behavior "" — must not match `trick: undefined`.
    expect(bestRunFor([run("x", { pick: true })], "", "microduck")).toBeUndefined();
  });
});

describe("splitTail", () => {
  it("leaves a short label whole", () => {
    expect(splitTail("Front kick (G1)")).toEqual(["Front kick (G1)", ""]);
  });

  it("keeps the END of a long one apart — where a battery's titles differ", () => {
    const [headA, tailA] = splitTail("Last metre, landscape camera, deep gaze only (left, seed 2)");
    const [headB, tailB] = splitTail("Last metre, landscape camera, deep gaze only (left, seed 3)");
    expect(headA).toBe(headB); // the shared opening is what may be ellipsized
    expect(tailA).not.toBe(tailB);
    expect(tailA).toBe(", seed 2)"); // 9 characters
  });

  it("loses nothing, spaces included", () => {
    const t = "Kick the ball it can see out to 0.60 m (left foot)";
    expect(splitTail(t, 11).join("")).toBe(t);
    expect(splitTail(t, 11)[1]).toBe("(left foot)");
  });

  it("never halves an emoji", () => {
    expect(splitTail("a long enough title ending 🦆🦆", 2)[1]).toBe("🦆🦆");
  });
});
