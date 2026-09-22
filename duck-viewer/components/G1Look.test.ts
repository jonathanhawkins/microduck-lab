// The weld tolerance, which is the one number in G1Look.tsx that a second
// robot can get wrong. Everything else there is a material.
//
// `weldAndSmooth` itself needs a WebGL-free BufferGeometry round trip and is
// covered by looking at the page (.claude/skills/sim-smoke); what is worth a
// unit test is the bound the tolerance has to respect, because getting it
// wrong is invisible until a shell arrives pinched.

import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { CREASE_ANGLE_DEG, weldAndSmooth, weldTolerance } from "./G1Look";

describe("weldTolerance", () => {
  it("never reaches the next point on a quantised dump's lattice", () => {
    // The whole contract: duplicates on a lattice are EXACTLY equal, so any
    // tolerance under one step catches them, and any tolerance at or over a
    // step merges two points that the dump meant to keep apart.
    for (const vertScale of [1e-3, 1e-4, 5e-4, 2e-5]) {
      expect(weldTolerance(vertScale), `${vertScale}`).toBeGreaterThan(0);
      expect(weldTolerance(vertScale), `${vertScale}`).toBeLessThan(vertScale);
    }
  });

  it("is finer on MARS's dump than on the G1's, by the ratio of the dumps", () => {
    // MARS is quantised ten times finer (robots/mars.VERT_SCALE_M) because a
    // millimetre lattice moved its head's vertices 0.50 mm on average. The
    // tolerance has to follow, or the finer dump is welded straight back
    // into the coarse one's shape — which was the trap: 0.4 mm of weld on a
    // 0.1 mm lattice pulls four points into one.
    expect(weldTolerance(1e-4)).toBeCloseTo(weldTolerance(1e-3) / 10, 12);
    expect(weldTolerance(1e-3)).toBeLessThan(4e-4 * 2);
  });

  it("keeps the measured 0.4 mm for a dump that has no lattice", () => {
    // The duck streams float metres (vertScale 1). There is no lattice to
    // bound, so the number that was measured on CAD shells stands, and a
    // missing vertScale must not be read as "lattice of one metre".
    expect(weldTolerance(1)).toBeCloseTo(4e-4, 12);
    expect(weldTolerance()).toBeCloseTo(4e-4, 12);
    expect(weldTolerance(0)).toBeCloseTo(4e-4, 12);
  });
});

/** Two unit quads sharing one edge, bent by `deg` — a CAD crease in
 *  miniature, with every vertex duplicated per face the way an STL dump
 *  sends them. */
function bentStrip(deg: number): THREE.BufferGeometry {
  const r = (deg * Math.PI) / 180;
  const dy = Math.cos(r), dz = Math.sin(r);
  // flat quad on z = 0, then a second quad rotated about the x axis
  const a = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]];
  const b = [[0, 1, 0], [1, 1, 0], [1, 1 + dy, dz], [0, 1 + dy, dz]];
  const tri = (q: number[][]) => [q[0], q[1], q[2], q[0], q[2], q[3]];
  const pos = [...tri(a), ...tri(b)].flat();
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  return g;
}

describe("weldAndSmooth", () => {
  it("welds the duplicates an STL dump sends", () => {
    // The founding finding of this file: without the weld every triangle
    // keeps its own flat normal and a CAD shell renders faceted.
    const welded = weldAndSmooth(bentStrip(90), 1e-4);
    expect(welded.getAttribute("position").count).toBeLessThan(12);
  });

  it("smooths across a hard edge when no crease angle is given", () => {
    // This is the behaviour MARS could not use. The shared edge's normals
    // are averaged between the two faces, so neither reads flat — which is
    // what fluted its head bar.
    const g = weldAndSmooth(bentStrip(90), 1e-4);
    const n = g.getAttribute("normal");
    const seen = new Set<string>();
    for (let i = 0; i < n.count; i++) {
      seen.add([n.getX(i), n.getY(i), n.getZ(i)].map((v) => v.toFixed(2)).join(","));
    }
    // A 90° bend of two faces, smoothed, gives a third direction on the seam.
    expect(seen.size).toBeGreaterThan(2);
  });

  it("keeps a bevel hard and a shallow bend smooth", () => {
    // The whole point of the crease angle: a 90° panel edge stays an edge,
    // while the many shallow steps of a tessellated barrel stay smooth, so
    // a wheel does not turn into a polygon.
    const hard = weldAndSmooth(bentStrip(90), 1e-4, CREASE_ANGLE_DEG);
    const hn = hard.getAttribute("normal");
    for (let i = 0; i < hn.count; i++) {
      const v = [hn.getX(i), hn.getY(i), hn.getZ(i)].map(Math.abs);
      // every normal is axis-aligned: no averaging happened
      expect(Math.max(...v), `vertex ${i}`).toBeCloseTo(1, 3);
    }
    const soft = weldAndSmooth(bentStrip(10), 1e-4, CREASE_ANGLE_DEG);
    const sn = soft.getAttribute("normal");
    let averaged = 0;
    for (let i = 0; i < sn.count; i++) {
      const v = [sn.getX(i), sn.getY(i), sn.getZ(i)].map(Math.abs);
      if (Math.max(...v) < 0.999) averaged++;
    }
    expect(averaged, "a 10° bend must still be smoothed").toBeGreaterThan(0);
  });

  it("is below the bevels it has to keep and above the steps it must not", () => {
    expect(CREASE_ANGLE_DEG).toBeLessThan(45);   // MARS's shell bevels
    expect(CREASE_ANGLE_DEG).toBeGreaterThan(20); // a tessellated barrel's steps
  });
});
