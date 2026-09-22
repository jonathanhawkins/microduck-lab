// The one robot every panel shares.
//
// Three panels used to persist three answers to "which robot?", so the 🧠
// palette could sit on the duck while 🎓 teach was about to train a G1. These
// pin the pure parts: what a returning user lands on, the single label every
// switch prints, and the fallback when the lab no longer lists the choice.

import { describe, expect, it } from "vitest";
import { initialActiveRobot, resolveRobot, robotChipLabel } from "./activeRobot";

const store = (m: Record<string, unknown>) => (k: string) => m[k] ?? null;

describe("initialActiveRobot", () => {
  it("is the duck for a first visit", () => {
    expect(initialActiveRobot(store({}))).toBe("microduck");
  });

  it("prefers the saved shared choice", () => {
    expect(initialActiveRobot(store({ activeRobot: "g1", teachRobot: "microduck" }))).toBe("g1");
  });

  it("adopts the old per-panel keys — teach's choice before the palette's", () => {
    expect(initialActiveRobot(store({ teachRobot: "g1", policyRobot: "microduck" }))).toBe("g1");
    expect(initialActiveRobot(store({ policyRobot: "g1" }))).toBe("g1");
  });

  it("never takes the palette's 'all' for a robot", () => {
    expect(initialActiveRobot(store({ policyRobot: "all" }))).toBe("microduck");
  });

  it("ignores junk", () => {
    expect(initialActiveRobot(store({ activeRobot: 7, teachRobot: "" }))).toBe("microduck");
  });
});

describe("robotChipLabel", () => {
  it("prints one label whichever endpoint described the robot", () => {
    const fromRobots = { id: "g1", title: "Unitree G1", noun: "G1" }; // GET /robots
    const fromPolicies = { id: "g1", label: "Unitree G1", noun: "G1" }; // GET /policies
    expect(robotChipLabel(fromRobots)).toBe("🤖 G1");
    expect(robotChipLabel(fromPolicies)).toBe(robotChipLabel(fromRobots));
  });

  it("gives the duck its emoji", () => {
    expect(robotChipLabel({ id: "microduck", title: "Microduck", noun: "duck" })).toBe("🦆 duck");
  });

  it("falls back to the full name for a lab that sends no noun", () => {
    expect(robotChipLabel({ id: "g1", label: "Unitree G1" })).toBe("🤖 Unitree G1");
  });
});

describe("resolveRobot", () => {
  const robots = [{ id: "microduck" }, { id: "g1" }];

  it("is the active robot when the lab lists it", () => {
    expect(resolveRobot(robots, "g1")).toEqual({ id: "g1" });
  });

  it("falls back to the lab's first when it doesn't", () => {
    expect(resolveRobot(robots, "spot")).toEqual({ id: "microduck" });
    expect(resolveRobot([], "g1")).toBeUndefined();
  });
});
