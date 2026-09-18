// MARS's look, pinned where it can silently rot: the part-kind classifier
// (a body that falls through it is drawn in the wrong material or not at
// all) and the colorway table.
//
// Node environment, like every other test here: these are string and colour
// assertions, and nothing in them can substitute for LOOKING at the stage
// (.claude/skills/sim-smoke, and the before/after shots that motivated the
// file in the first place).

import { describe, expect, it } from "vitest";
import * as THREE from "three";
import {
  MARS_COLORWAYS,
  MARS_DEFAULT_COLORWAY,
  MARS_KINDS,
  MIN_BODY_LUMA,
  makeMarsMaterial,
  marsColorway,
  marsHeadColor,
  marsPartKind,
  srgbLuma,
  type MarsPartKind,
} from "./MarsLook";

// Every body of the MARS dump, in the order the lab streams it:
// `microduck_local/src/microduck_local/robots/mars.visual_scene()["bodies"]`,
// which is the URDF's link list. Duplicated on purpose — a vitest cannot see
// across the two repos — and this comment is the link, the same way
// lib/sim.ts TEAM_COLORWAYS mirrors the Python table.
const MARS_BODIES = [
  "world",
  "base_link",
  "base_footprint",
  "base_laser",
  "link1",
  "link2",
  "link3",
  "link4",
  "link5",
  "link61",
  "link62",
  "ee_link",
  "arm_camera_link",
  "head",
  "head_camera_left",
  "camera_optical_frame",
  "head_camera_right",
  "right_camera_optical_frame",
] as const;

// The nine MESH geoms that actually arrive (the collision boxes and the
// marker spheres are filtered out server-side): body name → geom name, as
// `GET /scene?robot=mars` sends them. The geom names drop the `_link`
// suffix, which is exactly the mismatch the classifier has to absorb.
const MARS_GEOMS: readonly (readonly [string, string])[] = [
  ["base_link", "base"],
  ["link1", "link1"],
  ["link2", "link2"],
  ["link3", "link3"],
  ["link4", "link4"],
  ["link5", "link5"],
  ["link61", "link61"],
  ["link62", "link62"],
  ["head", "head"],
];

describe("marsPartKind", () => {
  it("maps every body of the MARS dump to a kind", () => {
    for (const body of MARS_BODIES) {
      expect(MARS_KINDS, body).toContain(marsPartKind(body));
    }
  });

  it("puts each streamed geom on the part it belongs to", () => {
    const got = MARS_GEOMS.map(([body, mesh]) => marsPartKind(body, mesh));
    expect(got).toEqual([
      "chassis", // base.STL — the box, the turret and BOTH wheels in one mesh
      "arm", "arm", "arm", "arm", "arm", // link1..link5
      "gripper", "gripper", // link61 / link62, the two fingers
      "head",
    ]);
  });

  it("draws nothing for a frame marker", () => {
    // Every one of these is a pure frame in the URDF. They are filtered out
    // of the dump today, so this is the guard for the day one is not.
    for (const f of ["ee_link", "arm_camera_link", "head_camera_left",
                     "head_camera_right", "camera_optical_frame",
                     "right_camera_optical_frame", "base_footprint",
                     "base_laser", "world"]) {
      expect(marsPartKind(f), f).toBe("marker");
    }
    expect(makeMarsMaterial("marker", marsColorway()).visible).toBe(false);
  });

  it("reads a camera frame as a marker, not as the head", () => {
    // `head_camera_left` contains "head": the rule order is the whole test.
    expect(marsPartKind("head_camera_left")).toBe("marker");
    expect(marsPartKind("head")).toBe("head");
  });

  it("reads link61/link62 as fingers, not as link 6 of the arm", () => {
    expect(marsPartKind("link61")).toBe("gripper");
    expect(marsPartKind("link62")).toBe("gripper");
  });

  it("shows an unknown part in the shell colour rather than hiding it", () => {
    // A body a later URDF adds must be visible and wrong, not invisible: a
    // part that vanishes is a bug nobody sees.
    expect(marsPartKind("gantry_plate_7")).toBe("chassis");
    expect(marsPartKind(null, null)).toBe("chassis");
  });
});

describe("MARS_COLORWAYS", () => {
  const entries = Object.entries(MARS_COLORWAYS);

  it("carries Innate's four store shells, with distinct ids", () => {
    expect(entries).toHaveLength(4);
    const ids = entries.map(([, cw]) => cw.id);
    expect(new Set(ids).size).toBe(4);
    expect(ids.sort()).toEqual(["black", "blue-white", "orange-black", "white"]);
    // The key IS the id — a table keyed one way and labelled another is how
    // a colorway becomes unreachable.
    for (const [key, cw] of entries) expect(cw.id).toBe(key);
  });

  it("states every colour as an sRGB hex triple", () => {
    for (const [, cw] of entries) {
      for (const c of [cw.accent, cw.body]) {
        expect(c, `${cw.id} ${c}`).toMatch(/^#[0-9a-f]{6}$/);
      }
      expect(cw.label.length).toBeGreaterThan(0);
    }
  });

  it("defaults to Innate's orange/black hero", () => {
    expect(MARS_DEFAULT_COLORWAY).toBe("orange-black");
    expect(marsColorway().id).toBe("orange-black");
    expect(marsColorway(null).id).toBe("orange-black");
    expect(marsColorway("no-such-shell").id).toBe("orange-black");
    expect(marsColorway("white").id).toBe("white");
  });

  it("keeps every chassis above the dark-stage luminance floor", () => {
    // The whole reason this file exists: the server's charcoal 0.16 is BELOW
    // this, and on the lab's #101216 stage it read as the backdrop.
    expect(srgbLuma("#101216")).toBeLessThan(MIN_BODY_LUMA);
    expect(srgbLuma("#292929")).toBeLessThan(MIN_BODY_LUMA); // charcoal 0.16
    for (const [, cw] of entries) {
      expect(srgbLuma(cw.body), `${cw.id} body`).toBeGreaterThanOrEqual(MIN_BODY_LUMA);
      expect(srgbLuma(marsHeadColor(cw)), `${cw.id} head`).toBeGreaterThanOrEqual(MIN_BODY_LUMA);
    }
  });

  it("puts the head between the accent and the chassis", () => {
    // Not the full accent (a whole orange head reads as a traffic cone) and
    // not the chassis (the two-tone shells carry the accent onto the head).
    for (const [, cw] of entries) {
      const head = srgbLuma(marsHeadColor(cw));
      const lo = Math.min(srgbLuma(cw.accent), srgbLuma(cw.body));
      const hi = Math.max(srgbLuma(cw.accent), srgbLuma(cw.body));
      expect(head, `${cw.id}`).toBeGreaterThan(lo - 1e-6);
      expect(head, `${cw.id}`).toBeLessThan(hi + 1e-6);
      expect(marsHeadColor(cw), `${cw.id}`).not.toBe(cw.accent);
    }
  });

  it("keeps the blue shell off Unitree's blue", () => {
    // The G1 stands on the same stage; a MARS in the same blue would read as
    // one of theirs.
    const blue = new THREE.Color(MARS_COLORWAYS["blue-white"].accent);
    expect(blue.b).toBeGreaterThan(blue.r);
    expect(blue.b).toBeGreaterThan(blue.g);
    expect(blue.g).toBeLessThan(0.2); // no cyan lean
  });
});

describe("makeMarsMaterial", () => {
  it("answers for every kind, so no group can index past the array", () => {
    // Duck.tsx groups the merged geometry by MARS_KINDS.indexOf(kind) and
    // hands three.js one material per index — a hole here is a black body.
    const cw = marsColorway();
    for (const k of MARS_KINDS) {
      const m = makeMarsMaterial(k as MarsPartKind, cw);
      expect(m, k).toBeInstanceOf(THREE.Material);
    }
    expect(MARS_KINDS.every((k) => MARS_KINDS.indexOf(k) >= 0)).toBe(true);
  });

  it("gives the chassis and head a specular top, which is what shows shape", () => {
    const cw = marsColorway();
    for (const k of ["chassis", "head"] as const) {
      const m = makeMarsMaterial(k, cw) as THREE.MeshPhysicalMaterial;
      expect(m.clearcoat, k).toBeGreaterThan(0);
      expect(m.metalness, k).toBeGreaterThan(0);
      expect(m.envMapIntensity, k).toBeGreaterThan(0);
    }
  });

  it("paints the arm and gripper in the colorway's accent", () => {
    const want = new THREE.Color(MARS_COLORWAYS.white.accent);
    for (const k of ["arm", "gripper"] as const) {
      const m = makeMarsMaterial(k, MARS_COLORWAYS.white) as THREE.MeshPhysicalMaterial;
      expect(m.color.getHexString(), k).toBe(want.getHexString());
    }
  });

  it("scales the reflection by envScale, so the lab and /sim can differ", () => {
    // The lab already has three lights; /sim has none. One table, two
    // intensities — the split useG1Materials documents.
    const lab = makeMarsMaterial("chassis", marsColorway(), { envScale: 0.35 }) as THREE.MeshPhysicalMaterial;
    const sim = makeMarsMaterial("chassis", marsColorway(), { envScale: 1 }) as THREE.MeshPhysicalMaterial;
    expect(lab.envMapIntensity).toBeLessThan(sim.envMapIntensity);
  });
});
