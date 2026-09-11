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
  BALANCE_STANCE,
  balanceColor,
  balanceLabel,
  balanceState,
  nearestFoot,
  sameBalance,
  setShowBalance,
  subscribeAnim,
  type Balance,
  type FootBalance,
} from "./anim";

const SOLE = [
  [-0.02, -0.02],
  [0.02, -0.02],
  [0.02, 0.02],
  [-0.02, 0.02],
];
const foot = (marginMm: number, grounded = true): FootBalance => ({ marginMm, grounded, outline: SOLE });
/** `support` defaults to what the server would send: the grounded feet, and
 *  the stance margin the pose's own story implies — well inside for the
 *  square stance, the stance sole's own margin once the other foot lifts. */
const balance = (
  left: FootBalance,
  right: FootBalance,
  over: Balance["over"] = null,
  supportMm?: number
): Balance => {
  const feet = { left, right };
  const down = (["left", "right"] as const).filter((s) => feet[s].grounded);
  const marginMm =
    supportMm ?? (down.length === 1 ? feet[down[0]].marginMm : Math.max(left.marginMm, right.marginMm) + 40);
  return { com: [0, 0, 0.14], feet, over, support: { feet: down, marginMm, outline: SOLE } };
};

/** Standing square: the CoM on the midline, the same distance outside both
 *  soles, and 16 mm inside the stance (what the model measures). */
const SQUARE = balance(foot(-25.0), foot(-25.0), null, 16.3);
/** Inside the left sole, the right foot lifted. */
const ON_LEFT = balance(foot(3.2), foot(-52.3, false), "left");
/** Leaning left, still outside both feet, both down, inside the stance. */
const LEANING = balance(foot(-4.2), foot(-33.1), null, 9.8);
/** Pitched forward past the toes: both feet down, outside the stance. */
const TOPPLING = balance(foot(-52.2), foot(-52.2), null, -40.5);

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
  it("says the square stance stands, and how far a sole is", () => {
    expect(balanceLabel(SQUARE)).toBe("16.3 mm inside the stance, 25.0 mm short of a sole");
  });
  it("names the near foot once there is a real lean", () => {
    expect(balanceLabel(LEANING)).toBe("9.8 mm inside the stance, 4.2 mm short of the left sole");
  });
  it("says outside once the CoM has left the stance", () => {
    expect(balanceLabel(TOPPLING)).toBe("52.2 mm outside both soles");
  });
  it("says when the other foot is off the floor", () => {
    expect(balanceLabel(balance(foot(-0.5), foot(-52.0, false)))).toBe(
      "0.5 mm outside the left sole (right foot in the air)"
    );
  });
});

describe("balanceState / balanceColor", () => {
  it("is green over a sole, blue inside the stance, amber outside", () => {
    expect(balanceState(ON_LEFT)).toBe("sole");
    expect(balanceColor(ON_LEFT)).toBe(BALANCE_IN);
    expect(balanceState(SQUARE)).toBe("stance");
    expect(balanceColor(SQUARE)).toBe(BALANCE_STANCE);
    expect(balanceState(TOPPLING)).toBe("out");
    expect(balanceColor(TOPPLING)).toBe(BALANCE_OUT);
    expect(balanceColor(null)).toBe(BALANCE_OUT);
  });
  it("cannot be inside the stance while over a lifted foot's sole", () => {
    // The server's `over` is grounded-only; the stance margin follows the
    // stance foot, so a lifted foot the CoM lines up with reads as out.
    const b = balance(foot(-30.0), foot(3.0, false));
    expect(b.support.marginMm).toBe(-30.0);
    expect(balanceState(b)).toBe("out");
  });
});

describe("sameBalance", () => {
  it("is equal when the readout would be, whatever the CoM did", () => {
    expect(sameBalance(SQUARE, { ...SQUARE, com: [0.01, 0, 0.13] })).toBe(true);
    expect(sameBalance(SQUARE, balance(foot(-25.0), foot(-24.9)))).toBe(false);
    expect(sameBalance(SQUARE, balance(foot(-25.0), foot(-25.0, false)))).toBe(false);
    expect(sameBalance(SQUARE, balance(foot(-25.0), foot(-25.0), null, 16.2))).toBe(false);
    // The outlines move with every slider tick; the readout does not print them.
    const moved = balance(foot(-25.0), foot(-25.0), null, 16.3);
    moved.feet.left.outline = SOLE.map(([x, y]) => [x + 0.01, y]);
    expect(sameBalance(SQUARE, moved)).toBe(true);
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
