// Robot selection on the lab stage: which body each roster row draws.
//
// The lab was one robot deep — every row was a Microduck and the stage had
// one mesh set. A G1 row streams poses in the G1 scene's OWN body order, so
// drawing it with the duck's geometry is silently wrong; these pin the two
// pure helpers the stage uses to decide.

import { describe, expect, it } from "vitest";
import { robotTag, robotsInFrame, type RobotId } from "./lab";

describe("robotsInFrame", () => {
  it("treats a row with no robot field as a duck (older servers)", () => {
    expect(robotsInFrame([{}, {}])).toEqual(["microduck"]);
  });

  it("dedupes and lists every body on the stage", () => {
    const rows = [{ robot: "g1" }, { robot: "microduck" }, { robot: "g1" }];
    expect(new Set(robotsInFrame(rows))).toEqual(new Set(["g1", "microduck"]));
    expect(robotsInFrame(rows)).toHaveLength(2);
  });

  it("is empty for an empty roster", () => {
    expect(robotsInFrame([])).toEqual([]);
  });

  it("keeps a mixed roster's duck rows drawing as ducks", () => {
    const rows: { robot?: string }[] = [{ robot: "g1" }, {}];
    const ids = robotsInFrame(rows);
    expect(ids).toContain("microduck");
    expect(ids).toContain("g1" as RobotId);
  });
});

describe("robotTag", () => {
  it("is empty for the duck, so a one-robot palette reads unchanged", () => {
    expect(robotTag(undefined)).toBe("");
    expect(robotTag("microduck")).toBe("");
  });

  it("names any other body on the chip", () => {
    expect(robotTag("g1")).toBe("g1 · ");
  });

  it("stays quiet inside a section that is already one robot", () => {
    // "Unitree G1 (shipped)" — every chip there is a G1, so the prefix would
    // repeat the heading on every row.
    expect(robotTag("g1", "g1")).toBe("");
    // …but a G1 run listed among the duck runs still says so.
    expect(robotTag("g1", "runs")).toBe("g1 · ");
    expect(robotTag("g1", "checkpoints")).toBe("g1 · ");
  });
});
