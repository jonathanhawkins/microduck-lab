// The IK handle's drag plane: where a pointer ray meets it, and what target
// a move asks for at full and at fine gain.

import { describe, expect, it } from "vitest";
import { dragPointOnViewPlane, ikDragStep, projectOntoPlane, type V3 } from "./ikdrag";

const close = (a: V3 | null, b: V3) => {
  expect(a).not.toBeNull();
  (a as V3).forEach((v, i) => expect(v).toBeCloseTo(b[i], 9));
};

describe("dragPointOnViewPlane", () => {
  const down: V3 = [0, 0, -1]; // a camera at +z looking down −z
  it("meets the plane through the anchor at the pointer's depth", () => {
    close(dragPointOnViewPlane([0, 0, 5], down, [1, 2, 0], down), [0, 0, 0]);
    // An oblique ray lands where it crosses z = 0; `dir` need not be unit.
    close(dragPointOnViewPlane([0, 0, 5], [1, 0, -1], [0, 0, 0], down), [5, 0, 0]);
    // The anchor only fixes the plane's depth, not where the hit lands.
    close(dragPointOnViewPlane([0, 0, 5], [0.5, 0, -1], [7, 7, 1], down), [2, 0, 1]);
  });
  it("is null for a ray along the plane or a plane behind the camera", () => {
    expect(dragPointOnViewPlane([0, 0, 5], [1, 0, 0], [0, 0, 0], down)).toBeNull();
    expect(dragPointOnViewPlane([0, 0, -5], down, [0, 0, 0], down)).toBeNull();
  });
  it("takes the view direction's sign either way round", () => {
    close(dragPointOnViewPlane([0, 0, 5], down, [0, 0, 1], [0, 0, 1]), [0, 0, 1]);
  });
});

describe("projectOntoPlane", () => {
  it("drops the component along the normal, whatever its length", () => {
    close(projectOntoPlane([1, 2, 3], [0, 0, 1]), [1, 2, 0]);
    close(projectOntoPlane([1, 2, 3], [0, 0, -4]), [1, 2, 0]);
    close(projectOntoPlane([1, 2, 3], [0, 0, 0]), [1, 2, 3]); // no plane, no change
  });
});

describe("ikDragStep", () => {
  const n: V3 = [0, 0, 1];
  it("at full gain keeps the pointer's offset from the grab, however far the limb lags", () => {
    const offset: V3 = [0.1, 0, 0]; // grabbed 10 cm left of the effector
    const s = ikDragStep([9, 9, 9], [1, 2, 0], null, offset, n, 1);
    close(s.target, [1.1, 2, 0]);
    expect(s.grabOffset).toBe(offset);
  });
  it("at fine gain inches from where the effector IS by a fraction of the in-plane travel", () => {
    const s = ikDragStep([0, 0, 0], [4, 0, 1], [0, 0, 0], [0, 0, 0], n, 0.25);
    close(s.target, [1, 0, 0]); // a quarter of 4 along x; the 1 along the normal is not travel
    // Re-anchored so that a following full-gain move continues from here…
    close(s.grabOffset, [-3, 0, -1]);
    const next = ikDragStep([1, 0, 0], [5, 0, 1], [4, 0, 1], s.grabOffset, n, 1);
    close(next.target, [2, 0, 0]); // …with no jump: one more unit for one more unit
  });
  it("with no previous hit a fine step asks for where the effector already is", () => {
    close(ikDragStep([0.3, 0.2, 0.1], [4, 0, 1], null, [0, 0, 0], n, 0.25).target, [0.3, 0.2, 0.1]);
  });
});
