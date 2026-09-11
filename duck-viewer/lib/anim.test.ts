// The balance readout's wording and colour, and the toggle that shows it.
//
// The fixtures are shapes, not measurements: a margin is a signed distance
// in millimetres and the readout only formats it, so what these lock is the
// wording (which foot, inside or outside, what an airborne foot does to the
// answer) and the tests name the numbers they use. The measurement itself is
// the server's, and tests/test_clips.py holds it against the model.
//
// What a test cannot do is judge the marker. Whether a ball and a crosshair
// land where the eye expects is pixels, and pixels are looked at.

import { describe, expect, it } from "vitest";
import {
  animStore,
  animVersion,
  BALANCE_IN,
  BALANCE_OUT,
  balanceColor,
  balanceLabel,
  nearestFoot,
  sameBalance,
  setShowBalance,
  subscribeAnim,
  type Balance,
  type FootBalance,
} from "./anim";

const foot = (marginMm: number, grounded = true): FootBalance => ({ marginMm, grounded });
const balance = (left: FootBalance, right: FootBalance, over: Balance["over"] = null): Balance => ({
  com: [0, 0, 0.14],
  feet: { left, right },
  over,
});

/** Standing square: the CoM on the midline, the same distance outside both. */
const SQUARE = balance(foot(-25.0), foot(-25.0));
/** Inside the left sole, the right foot lifted. */
const ON_LEFT = balance(foot(3.2), foot(-52.3, false), "left");
/** Leaning left, still outside both feet, both down. */
const LEANING = balance(foot(-4.2), foot(-33.1));

describe("nearestFoot", () => {
  it("names the foot with more room, standing or not", () => {
    expect(nearestFoot(ON_LEFT)).toEqual({ side: "left", marginMm: 3.2 });
    expect(nearestFoot(LEANING)).toEqual({ side: "left", marginMm: -4.2 });
    expect(nearestFoot(balance(foot(-33.1), foot(-4.2)))).toEqual({ side: "right", marginMm: -4.2 });
  });
  it("will not pick a foot that is in the air", () => {
    // The CoM is nearer the lifted right foot, but it cannot be stood on.
    expect(nearestFoot(balance(foot(-30.0), foot(-2.0, false)))).toEqual({ side: "left", marginMm: -30.0 });
  });
});

describe("balanceLabel", () => {
  it("says which sole the CoM is inside, and by how much", () => {
    expect(balanceLabel(ON_LEFT)).toBe("3.2 mm inside the left sole");
  });
  it("blames no foot when the pose is square", () => {
    expect(balanceLabel(SQUARE)).toBe("25.0 mm outside both soles");
  });
  it("names the near foot once there is a real lean", () => {
    expect(balanceLabel(LEANING)).toBe("4.2 mm outside the left sole");
  });
  it("says when the other foot is off the floor", () => {
    expect(balanceLabel(balance(foot(-0.5), foot(-52.0, false)))).toBe(
      "0.5 mm outside the left sole (right foot in the air)"
    );
  });
});

describe("balanceColor", () => {
  it("goes green only once the CoM is inside a sole", () => {
    expect(balanceColor(ON_LEFT)).toBe(BALANCE_IN);
    expect(balanceColor(SQUARE)).toBe(BALANCE_OUT);
    expect(balanceColor(null)).toBe(BALANCE_OUT);
  });
});

describe("sameBalance", () => {
  it("is equal when the readout would be, whatever the CoM did", () => {
    expect(sameBalance(SQUARE, { ...SQUARE, com: [0.01, 0, 0.13] })).toBe(true);
    expect(sameBalance(SQUARE, balance(foot(-25.0), foot(-24.9)))).toBe(false);
    expect(sameBalance(SQUARE, balance(foot(-25.0), foot(-25.0, false)))).toBe(false);
    expect(sameBalance(null, SQUARE)).toBe(false);
    expect(sameBalance(null, null)).toBe(true);
  });
});

describe("setShowBalance", () => {
  it("notifies React, unlike the per-frame balance data itself", () => {
    let seen = 0;
    const stop = subscribeAnim(() => seen++);
    const v0 = animVersion();
    try {
      setShowBalance(true);
      expect(animStore.showBalance).toBe(true);
      expect(seen).toBe(1);
      setShowBalance(true); // no change, no render
      expect(seen).toBe(1);
      setShowBalance(false);
      expect(seen).toBe(2);
      expect(animVersion()).toBe(v0 + 2);
    } finally {
      stop();
      animStore.showBalance = false;
    }
  });
});
