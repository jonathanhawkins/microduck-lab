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
  EYE_GREY,
  FACE_BLACK,
  INNATE_BLACK,
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

// The fourteen MESH geoms that actually arrive (the collision boxes and the
// marker spheres are filtered out server-side): body name → geom name, as
// `GET /scene?robot=mars` sends them. The geom names drop the `_link`
// suffix, which is exactly the mismatch the classifier has to absorb.
//
// `base_wheels`, `link2_servo`, `link3_servo`, `head_face` and `head_eyes`
// are the odd ones: they are the meshes in a MARS that Innate did not ship. The description has the tyres as triangles
// inside `base.STL`, and the server cuts them out (microduck_local
// robots/mars.split_base_mesh) so that "black wheels" is a colour some geom
// can be given. Both of its names say `base_link`/`base_wheels`, so the rule
// that catches it has to beat the `base` one.
const MARS_GEOMS: readonly (readonly [string, string])[] = [
  ["base_link", "base"],
  ["base_link", "base_wheels"],
  ["link1", "link1"],
  ["link2", "link2"],
  ["link2", "link2_servo"],
  ["link3", "link3"],
  ["link3", "link3_servo"],
  ["link4", "link4"],
  ["link5", "link5"],
  ["link61", "link61"],
  ["link62", "link62"],
  ["head", "head"],
  ["head", "head_face"],
  ["head", "head_eyes"],
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
      "chassis",   // base — base.STL minus the tyres — the box, the turret, the neck
      "wheel",     // base_wheels — the two tyres, cut out of base.STL
      "servo",     // link1 — the arm's first joint, a bare black servo case
      "arm",       // link2 — the lower arm itself
      "servo",     // link2_servo — the shoulder's housing, cut off link2
      "arm",       // link3 — the forearm itself
      "servo",     // link3_servo — the elbow's housing, cut off link3
      "servo",     // link4 — the wrist, a whole mesh of its own
      "arm",       // link5 — the gripper body
      "gripper",   // link61 — a finger
      "gripper",   // link62 — the other finger
      "head",      // head — the blue head bar
      "face",      // head_face — the flat black panel, cut off the head shell
      "lens",      // head_eyes — the two camera domes, their own shell
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

  it("reads the tyres as wheels although both names say `base`", () => {
    // The planted regression for the rule ORDER: `base_wheels` matches the
    // chassis rule too, and before the wheel rule was put in front of it the
    // tyres were painted with the body. A lab too old to send a `base_wheels`
    // mesh has no geom here at all, which is the fallback that matters: it
    // draws a white chassis with white wheels, not a robot with a hole in it.
    expect(marsPartKind("base_link", "base_wheels")).toBe("wheel");
    expect(marsPartKind("base_link", "base")).toBe("chassis");
    expect(marsPartKind("base_link")).toBe("chassis");
  });

  it("wears rubber on the wheels, not the shell's paint", () => {
    const cw = marsColorway();
    const tyre = makeMarsMaterial("wheel", cw, { envScale: 1 }) as THREE.MeshPhysicalMaterial;
    const shell = makeMarsMaterial("chassis", cw, { envScale: 1 }) as THREE.MeshPhysicalMaterial;
    expect(tyre.color.getHexString(THREE.SRGBColorSpace)).toBe(INNATE_BLACK.slice(1));
    expect(tyre.clearcoat).toBe(0);              // paint has a clearcoat
    expect(tyre.metalness).toBe(0);
    expect(tyre.roughness).toBeGreaterThan(shell.roughness);
    // And it is EXEMPT from the light-shell reflection cut: `reflect` is a
    // correction for a near-white body, and scaling a black tyre by it again
    // flattened the tread into one blob.
    const dark = makeMarsMaterial("wheel", MARS_COLORWAYS["orange-black"], { envScale: 1 }) as THREE.MeshPhysicalMaterial;
    expect(cw.reflect).toBeLessThan(1);          // the shell IS cutting it
    expect(tyre.envMapIntensity).toBeCloseTo(dark.envMapIntensity, 6);
  });

  it("reads the arm's first joint as a bare servo, not as arm shell", () => {
    // The planted regression for the rule ORDER again: `link1` matches the
    // arm rule too. On the real robot that joint is the moulded black case
    // in its bracket by the right wheel — the one part of the arm the shell
    // colour does not reach — and it was drawn in the arm's paint until the
    // servo rule went in front.
    expect(marsPartKind("link1")).toBe("servo");
    expect(marsPartKind("link4")).toBe("servo");
    // …and the two CUT housings, which arrive on the arm's own body with a
    // `_servo` mesh name: the arm rule matches "link2"/"link3" too, so this
    // is the same order test as the tyres'.
    expect(marsPartKind("link2", "link2_servo")).toBe("servo");
    expect(marsPartKind("link3", "link3_servo")).toBe("servo");
    expect(marsPartKind("link2", "link2")).toBe("arm");
    expect(marsPartKind("link3", "link3")).toBe("arm");
    for (const l of ["link2", "link3", "link5"]) {
      expect(marsPartKind(l), l).toBe("arm");
    }
    const cw = marsColorway();
    const servo = makeMarsMaterial("servo", cw, { envScale: 1 }) as THREE.MeshPhysicalMaterial;
    expect(servo.color.getHexString(THREE.SRGBColorSpace)).toBe(INNATE_BLACK.slice(1));
    // It does NOT follow the colorway: the case is moulded, not painted, so
    // every shell shows the same black there.
    for (const id of Object.keys(MARS_COLORWAYS)) {
      const m = makeMarsMaterial("servo", MARS_COLORWAYS[id]) as THREE.MeshPhysicalMaterial;
      expect(m.color.getHexString(THREE.SRGBColorSpace), id).toBe(INNATE_BLACK.slice(1));
    }
  });

  it("reads the head's panel and lenses before the head itself", () => {
    // Both names start with "head", so the rule ORDER is the test again:
    // before this went in front, Innate's black face and grey eyes were
    // drawn in the head bar's blue.
    expect(marsPartKind("head", "head_face")).toBe("face");
    expect(marsPartKind("head", "head_eyes")).toBe("lens");
    expect(marsPartKind("head", "head")).toBe("head");
    // …and a camera FRAME is still a marker, not a lens: the rule that
    // catches `head_camera_left` runs earlier and stays there.
    expect(marsPartKind("head_camera_left")).toBe("marker");
    const face = makeMarsMaterial("face", marsColorway(), { envScale: 1 }) as THREE.MeshPhysicalMaterial;
    const lens = makeMarsMaterial("lens", marsColorway(), { envScale: 1 }) as THREE.MeshPhysicalMaterial;
    expect(face.color.getHexString(THREE.SRGBColorSpace)).toBe(FACE_BLACK.slice(1));
    expect(lens.color.getHexString(THREE.SRGBColorSpace)).toBe(EYE_GREY.slice(1));
    // The lens has to read LIGHTER than the panel it sits in — that is the
    // whole reason it is a kind of its own — and glossier, so it catches a
    // highlight instead of going flat.
    expect(srgbLuma(EYE_GREY)).toBeGreaterThan(srgbLuma(FACE_BLACK) * 1.5);
    expect(lens.roughness).toBeLessThan(face.roughness);
    expect(lens.envMapIntensity).toBeGreaterThan(face.envMapIntensity);
    // Neither follows the colorway: they are moulded parts, not paint.
    for (const id of Object.keys(MARS_COLORWAYS)) {
      const f = makeMarsMaterial("face", MARS_COLORWAYS[id]) as THREE.MeshPhysicalMaterial;
      const l = makeMarsMaterial("lens", MARS_COLORWAYS[id]) as THREE.MeshPhysicalMaterial;
      expect(f.color.getHexString(THREE.SRGBColorSpace), id).toBe(FACE_BLACK.slice(1));
      expect(l.color.getHexString(THREE.SRGBColorSpace), id).toBe(EYE_GREY.slice(1));
    }
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

  it("defaults to the Blue / White shell Innate's own photos show", () => {
    // Two things at once, and both are the point. A light body is the one
    // that READS on the lab's #101216 floor at a glance from across the grid
    // (measured on the live page — the far slot's chassis crop went 0.241 ->
    // 0.697, 3.1x the background to 8.6x). And blue-white rather than white
    // is the machine itself: a blue head bar and blue gripper fingers on
    // black tyres. Both other shells are still in the table.
    expect(MARS_DEFAULT_COLORWAY).toBe("blue-white");
    expect(marsColorway().id).toBe("blue-white");
    expect(marsColorway(null).id).toBe("blue-white");
    expect(marsColorway("no-such-shell").id).toBe("blue-white");
    expect(marsColorway("orange-black").id).toBe("orange-black");
    // The default's head is BLUE, not a wash of it: this is the assertion
    // that fails if the head ever goes back through MARS_HEAD_MIX, which
    // would pull it 45 % of the way to a white chassis.
    const cw = marsColorway();
    expect(marsHeadColor(cw)).toBe(cw.accent);
  });

  it("keeps the default body an OFF-white, cool, and never #fff", () => {
    // A pure white clear-coated shell has nowhere for a highlight to GO: the
    // seam glints merge into the panel they are meant to cut, and on /sim's
    // pale room the whole chassis went to one blob (sd 0.038 across the crop).
    // The band is the off-whites; White sits at its floor.
    const body = MARS_COLORWAYS.white.body;
    expect(body).not.toBe("#ffffff");
    expect(srgbLuma(body)).toBeGreaterThanOrEqual(srgbLuma("#e8eaee") - 1e-6);
    expect(srgbLuma(body)).toBeLessThanOrEqual(srgbLuma("#f2f2f0") + 1e-6);
    // COOL, not warm: /sim's floor is #d0ac86 wood under a #cfcabb wall, so a
    // cool shell keeps a hue difference there even where the luminances cross.
    const c = new THREE.Color(body);
    expect(c.b).toBeGreaterThan(c.r);
  });

  it("steps the white arm 10-15 % below the white body", () => {
    // Innate's store White is a white arm on a white body, and drawn that way
    // the arm's joints and the two gripper fingers vanish into the chassis
    // they fold against. Below 10 % there is no part line; above 15 % the arm
    // stops reading as the same shell and starts reading as a grey spare.
    const w = MARS_COLORWAYS.white;
    const step = 1 - srgbLuma(w.accent) / srgbLuma(w.body);
    expect(step).toBeGreaterThan(0.1);
    expect(step).toBeLessThan(0.15);
  });

  it("cuts the reflection map on the light shell, and only on it", () => {
    // The inversion of this file's founding finding: a DARK shell reads by its
    // specular top, a near-white one is flattened by it (the wash lands on the
    // shadowed faces, where there is no diffuse signal to compete with it).
    for (const id of ["white", "blue-white"]) {
      expect(MARS_COLORWAYS[id].reflect, id).toBeLessThan(0.5);
    }
    for (const id of ["orange-black", "black"]) {
      expect(MARS_COLORWAYS[id].reflect, id).toBeUndefined();
    }
    // …and it has to REACH the material, on every PAINTED part kind: one part
    // of a shell lit differently from the next is worse than either setting.
    // `wheel` is not in this list on purpose — see the test below.
    const refl = MARS_COLORWAYS.white.reflect as number;
    for (const k of ["chassis", "head", "arm", "gripper"] as const) {
      const dark = makeMarsMaterial(k, MARS_COLORWAYS["orange-black"], { envScale: 1 }) as THREE.MeshPhysicalMaterial;
      const light = makeMarsMaterial(k, MARS_COLORWAYS.white, { envScale: 1 }) as THREE.MeshPhysicalMaterial;
      expect(light.envMapIntensity, k).toBeCloseTo(dark.envMapIntensity * refl, 6);
      expect(light.envMapIntensity, k).toBeGreaterThan(0); // not "off"
    }
  });

  it("leaves the two dark shells exactly as they shipped", () => {
    // `toEqual` also fails if one of them grows a `reflect`, a `wheel`, a
    // `head` or an `arm`: a dark shell wants the full specular top and the
    // accent on its arm, and every field added for the light default is a
    // field that must NOT have leaked onto these two.
    expect(MARS_COLORWAYS["orange-black"]).toEqual(
      { id: "orange-black", label: "Orange / Black", accent: "#ff8a1f", body: "#3b3f47" });
    expect(MARS_COLORWAYS["black"]).toEqual(
      { id: "black", label: "Black", accent: "#5b616b", body: "#3b3f47" });
    // And the arm of a shell that names no `arm` is still its accent.
    for (const id of ["orange-black", "black"]) {
      const m = makeMarsMaterial("arm", MARS_COLORWAYS[id]) as THREE.MeshPhysicalMaterial;
      expect(m.color.getHexString(THREE.SRGBColorSpace), id)
        .toBe(MARS_COLORWAYS[id].accent.slice(1));
    }
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

  it("mixes the head toward the chassis unless the shell names one", () => {
    // The MIX is for a shell whose accent is its ARM: a whole orange head
    // reads as a traffic cone, and a head in the chassis colour loses the
    // two-tone the real shells carry, so it lands between the two. A shell
    // that names `head` is saying its head is not a function of its arm —
    // Innate's Blue / White is a white arm with a blue head — and then the
    // only thing to check is that it is not just the chassis again.
    for (const [, cw] of entries) {
      const head = marsHeadColor(cw);
      if (cw.head) {
        expect(head, `${cw.id}`).toBe(cw.head);
        expect(head, `${cw.id}`).not.toBe(cw.body);
        continue;
      }
      const lo = Math.min(srgbLuma(cw.accent), srgbLuma(cw.body));
      const hi = Math.max(srgbLuma(cw.accent), srgbLuma(cw.body));
      expect(srgbLuma(head), `${cw.id}`).toBeGreaterThan(lo - 1e-6);
      expect(srgbLuma(head), `${cw.id}`).toBeLessThan(hi + 1e-6);
      expect(head, `${cw.id}`).not.toBe(cw.accent);
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
