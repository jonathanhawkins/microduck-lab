import { describe, expect, it } from "vitest";
import { clampPos, moved } from "./drag";

describe("clampPos", () => {
  const size = { w: 280, h: 200 };
  const view = { w: 1000, h: 800 };
  it("leaves a box that fits where it is", () => {
    expect(clampPos({ x: 300, y: 250 }, size, view, 80, 10)).toEqual({ x: 300, y: 250 });
  });
  it("keeps the box off every edge and out from under the top bar", () => {
    expect(clampPos({ x: -50, y: 20 }, size, view, 80, 10)).toEqual({ x: 10, y: 80 });
    expect(clampPos({ x: 5000, y: 5000 }, size, view, 80, 10)).toEqual({ x: 710, y: 590 });
  });
  it("hugs the top-left when the viewport is smaller than the box", () => {
    expect(clampPos({ x: 400, y: 400 }, size, { w: 200, h: 100 }, 80, 10)).toEqual({ x: 10, y: 80 });
  });
  it("rounds to whole pixels", () => {
    expect(clampPos({ x: 10.6, y: 99.4 }, size, view)).toEqual({ x: 11, y: 99 });
  });
});

describe("moved", () => {
  it("treats a few pixels of wobble as a click", () => {
    expect(moved({ x: 0, y: 0 }, { x: 3, y: -3 })).toBe(false);
    expect(moved({ x: 0, y: 0 }, { x: 5, y: 0 })).toBe(true);
  });
});
