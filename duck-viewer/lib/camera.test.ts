// The keyboard → camera map, pinned. The vertical pair is the reason this
// file exists: E rises and Q sinks (they were the other way round until
// 2026-09-21), and both pages' control panels spell the pair out in words —
// a silent flip back would leave the tutorial text lying rather than break
// anything loudly. The rest of the map rides along, since this module is the
// single place the lab page and /sim read the keyboard from.

import { beforeEach, describe, expect, it } from "vitest";
import {
  cameraKeyDown,
  cameraKeyUp,
  cameraKeysClear,
  heldMotions,
  takeReset,
} from "@/lib/camera";

// Module-level state, shared across cases: start each one empty.
beforeEach(() => {
  cameraKeysClear();
  takeReset();
});

describe("camera keys", () => {
  it("flies up on E and down on Q", () => {
    expect(cameraKeyDown("e")).toBe(true);
    expect([...heldMotions()]).toEqual(["up"]);
    cameraKeyUp("e");
    expect(cameraKeyDown("q")).toBe(true);
    expect([...heldMotions()]).toEqual(["down"]);
  });

  it("reads the same with Caps Lock on", () => {
    cameraKeyDown("E");
    cameraKeyDown("Q");
    expect(heldMotions()).toEqual(new Set(["up", "down"]));
    cameraKeyUp("E");
    expect(heldMotions()).toEqual(new Set(["down"]));
  });

  it("keeps the rest of the map", () => {
    for (const k of ["a", "d", "w", "s", "arrowleft", "arrowright"]) cameraKeyDown(k);
    expect(heldMotions()).toEqual(
      new Set(["truckLeft", "truckRight", "dollyIn", "dollyOut", "orbitLeft", "orbitRight"]),
    );
  });

  it("leaves plain R to the caller and takes Shift+R as the view reset", () => {
    expect(cameraKeyDown("r")).toBe(false);
    expect(takeReset()).toBe(false);
    expect(cameraKeyDown("r", true)).toBe(true);
    expect(takeReset()).toBe(true);
    expect(takeReset()).toBe(false); // drained
  });

  it("ignores a key it does not own", () => {
    expect(cameraKeyDown("p")).toBe(false);
    expect(heldMotions().size).toBe(0);
  });
});
