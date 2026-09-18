"use client";

// How an Innate MARS is drawn — shared by the lab stage (Duck.tsx, via
// buildBodyGeometries' `look: "mars"`) and the /sim page (SimStage.tsx's
// RobotBody), so the two cannot drift apart the way the G1's two draws once
// did.
//
// WHY this file exists at all. MARS used to take the "generic" look: every
// geom painted from the scene dump's own `rgba`, which is the SERVER's paint
// (microduck_local robots/mars.style_visual_geoms — links 1/3/5 bright
// orange, "matt black" lifted to charcoal 0.16, frame markers hidden). 0.16
// is a lighter BLACK, not a visible grey: against the lab stage's #101216
// background the chassis and the head landed within a few percent of the
// backdrop, so the only part of the robot that read was the orange arm and
// MARS looked like an arm floating in the dark.
//
// The fix belongs HERE, not in the server's dump, for two reasons:
//   1. It is a STAGE decision. The same 0.16 chassis is a fine silhouette on
//      /sim's pale wooden floor and invisible on the lab's near-black one;
//      what a colour has to survive is a property of the renderer, not of the
//      robot. The server's rgba stays honest about Innate's paint.
//   2. It needs a material, not a colour. A dark shell reads on a dark stage
//      because of its SPECULAR top — a clearcoat and a reflection give the
//      chassis a highlight along every edge, which is what shows its shape.
//      Diffuse colour alone would have to go so light the robot stopped
//      looking like Innate's matte-black machine.
//
// WHAT THE WHITE SHELL ADDED. The default is now Innate's White
// (MARS_DEFAULT_COLORWAY), and a near-white shell inverts the rule above
// exactly. A dark shell is legible BECAUSE of the reflection; a light one is
// flattened by it, because the wash lands on the shadowed faces where there is
// no diffuse signal to compete with it. Measured with the white body on the
// UNSCALED reflection (`reflect` absent, /sim's envScale 1): the chassis crop
// went to mean 0.896 with a spread of sd 0.038 and a darkest pixel of 0.835 —
// the panel seams, the turret and the wheels in one blob at the top of the
// range — against sd 0.096 and a darkest 0.505 on the lab stage, which passes
// the same map at 0.35. Taking 0.3 of the map put /sim back at sd 0.089. So
// `reflect` scales the shared
// reflection map per COLORWAY, and White takes 0.3 of it: one number, applied
// to all four part kinds, left at 1 (bit-identical materials) for the three
// shells whose bodies still need the specular top.
//
// Everything else follows G1Look.tsx: weld before smoothing (the STL dump
// duplicates every vertex per face, so computeVertexNormals alone gives the
// faceted CAD look), a material per PART KIND rather than a component per
// robot (docs/mars-roadmap.md §6.4), and ONE procedural RoomEnvironment per
// renderer — shared with the G1's materials, and applied to these materials
// only rather than to scene.environment, so the ducks keep exactly the
// lighting they had.

import { useThree } from "@react-three/fiber";
import { useMemo } from "react";
import * as THREE from "three";
import { roomEnv } from "./G1Look";

export const MARS_KINDS = ["chassis", "head", "arm", "gripper", "marker"] as const;
export type MarsPartKind = (typeof MARS_KINDS)[number];

/** Innate's four store colorways for the real MARS.
 *
 *  `accent` is the arm (links 1-5) and the gripper fingers; `body` is the
 *  chassis. The head takes a MIX of the two (see `marsHeadColor`): the real
 *  two-tone shells split the body diagonally and carry the accent up onto the
 *  head, which one `head.STL` geom cannot show as a band — a whole head in
 *  full accent read as a traffic cone on a robot whose head is mostly a
 *  camera housing, so it gets the accent pulled back toward the body.
 *
 *  Why no near-black `body` even on the "Black" shell: the chassis has to
 *  READ on the lab's #101216 stage (the whole point of this file). Every
 *  `body` here is at or above MIN_BODY_LUMA, so "Black" is the graphite a
 *  matte-black shell actually photographs as under a light, not #000.
 *
 *  Blue / White's accent is Innate's own deep blue and deliberately NOT
 *  Unitree's cyan-lean blue — the G1 stands on the same stage. */
export interface MarsColorway {
  id: string;
  label: string;
  /** Arm + gripper (and, mixed toward `body`, the head). */
  accent: string;
  /** The chassis. */
  body: string;
  /** How much of the shared RoomEnvironment reflection this shell takes,
   *  0..1, default 1 (see the header). A dark shell reads BY its specular top
   *  and takes all of it; a near-white shell is flattened by it and takes a
   *  fraction. Scales `envMapIntensity` on every part kind, so it cannot make
   *  one part of a shell disagree with another. */
  reflect?: number;
}

export const MARS_COLORWAYS: Record<string, MarsColorway> = {
  "orange-black": { id: "orange-black", label: "Orange / Black", accent: "#ff8a1f", body: "#3b3f47" },
  "blue-white": { id: "blue-white", label: "Blue / White", accent: "#2f34e6", body: "#e7e9ee" },
  black: { id: "black", label: "Black", accent: "#5b616b", body: "#3b3f47" },
  // The DEFAULT shell (see MARS_DEFAULT_COLORWAY), so its three numbers are
  // the measured ones:
  //   body    — a COOL off-white at the bottom of the off-white band. Cool
  //             because both stages are warm where it matters: /sim's floor is
  //             #d0ac86 wood under a #cfcabb wall, and a cool shell keeps a
  //             hue difference there even where the luminances cross; the lab
  //             stage's own #101216 is blue-black, but the G1 standing on it
  //             is a neutral #d8d8d8 (G1Look.G1_SHELL) and a WARM MARS white
  //             would have been the odd robot out rather than a different
  //             one. Bottom of the band, not the top, because a clearcoat
  //             highlight needs somewhere ABOVE the shell to go: at #f2f2f0
  //             the seam glints merge into the panel they are supposed to cut.
  //   accent  — 12 % darker than the body in Rec.709 luma (0.807 against
  //             0.918). Innate's store White is a white arm on a white body,
  //             and drawing it that way made the arm's joints and the two
  //             gripper fingers vanish into the chassis they fold against.
  //             12 % is the smallest step that still reads as one shell.
  //   reflect — 0.3: the light-shell rule from the header.
  white: { id: "white", label: "White", accent: "#cbced5", body: "#e8eaee", reflect: 0.3 },
};

/** Innate's White shell. A slot with no colorway of its own is drawn in it
 *  because the first thing a stage has to do is let you FIND the robot. On the
 *  live lab page, at one camera, the same chassis crop measured 0.241 under the
 *  orange/black shell's graphite body and 0.697 under this one — 3.1x the
 *  #101216 background against 8.6x — and the arm stopped being the only part
 *  of MARS you could see from across the grid. The orange/black hero (Innate's
 *  press photo, and what the server paints) was the default while "does the
 *  chassis read AT ALL" was the open question; nothing about it changed, and
 *  the day a roster row carries a colorway it is one field away. Note the two
 *  dark shells share that graphite body, so this is a change of default and
 *  not a claim that a dark MARS is unreadable — it reads, by its specular top,
 *  which is what the rest of this file is about. */
export const MARS_DEFAULT_COLORWAY = "white";

/** How far the head is pulled from `accent` back toward `body` (0 = full
 *  accent, 1 = the chassis colour). 0.45 leaves the head unmistakably in the
 *  accent family while keeping it darker than the arm, which is how the real
 *  robot's accented head reads next to its orange links. */
export const MARS_HEAD_MIX = 0.45;

/** The luminance floor the chassis and head must clear. Rec.709 weights on
 *  the sRGB components — the same cheap measure `g1PartKind` uses to decide a
 *  geom is "dark", and the thing that was wrong before: the server's charcoal
 *  0.16 scores 0.16 here against a 0.07 stage. Anything under 0.2 is a
 *  silhouette on the lab's floor, not a shape. */
export const MIN_BODY_LUMA = 0.2;

/** Rec.709 luminance of an sRGB hex colour, 0..1 (no linearisation — this is
 *  the "does it read against the backdrop" measure, not a photometric one). */
export function srgbLuma(hex: string): number {
  const c = new THREE.Color();
  c.setStyle(hex, THREE.LinearSRGBColorSpace); // no transform: keep sRGB components
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

/** The head's colour for a colorway: the accent mixed back toward the body in
 *  LINEAR space (mixing sRGB components darkens a saturated accent unevenly). */
export function marsHeadColor(cw: MarsColorway): string {
  const a = new THREE.Color().setStyle(cw.accent, THREE.SRGBColorSpace);
  const b = new THREE.Color().setStyle(cw.body, THREE.SRGBColorSpace);
  return "#" + a.lerp(b, MARS_HEAD_MIX).getHexString(THREE.SRGBColorSpace);
}

/** Resolve a colorway id (a slot's choice, or nothing) to a colorway. */
export function marsColorway(id?: string | null): MarsColorway {
  return (id && MARS_COLORWAYS[id]) || MARS_COLORWAYS[MARS_DEFAULT_COLORWAY];
}

// Body/mesh name → part kind, first match wins. ORDER MATTERS: every camera
// frame is called `*_camera_*`, so the marker rule has to be tried before the
// `head` rule or `head_camera_left` would be drawn as the head.
//
// The names are MARS's own link names (microduck_local
// robots/mars.visual_scene()'s `bodies`, which is the URDF's link list):
// base_link, link1..link5, link61/link62 (the two gripper fingers), head, and
// the pure frames — base_footprint, base_laser, ee_link, arm_camera_link,
// head_camera_left/right and the two optical frames. The geom names in the
// dump are the same minus the `_link` suffix (`base`, `link1`, …), so the
// classifier is given both and matches on either.
const MARS_RULES: readonly (readonly [RegExp, MarsPartKind])[] = [
  // Pure frames: the URDF draws several as 5 mm spheres and the MuJoCo
  // importer keeps them out of the visual group, so nothing usually arrives
  // for these — but a body list is a contract, and a frame that DID arrive
  // must be invisible rather than a mystery blob. `base_laser` is one of
  // them: the scanner's housing is a geom of `base_link`, not of the frame
  // (docs/mars-roadmap.md §2b). `world` is the MJCF root and holds nothing.
  [/^world$/, "marker"],
  [/camera|optical|laser|footprint|^ee_link$|marker/, "marker"],
  // The two fingers, before the arm rule — `link61`/`link62` are not links
  // 6 and 1/2, and a "6" on a 5-joint arm is always a finger.
  [/^link6\d$|finger|gripper/, "gripper"],
  [/^link[1-5]$/, "arm"],
  [/head/, "head"],
  // The base, INCLUDING its wheels: both wheels are triangles inside
  // base.STL — the dump carries one mesh geom for the whole chassis — so a
  // `wheel` kind would have nothing to select and is deliberately absent.
  [/base/, "chassis"],
];

/** Which material a MARS geom takes, from its body name and (optionally) its
 *  mesh/geom name. An unrecognised part falls back to `chassis`, never to
 *  `marker`: a body this build has not heard of should be VISIBLE in the
 *  shell colour, not silently deleted from the robot. */
export function marsPartKind(bodyName?: string | null, meshName?: string | null): MarsPartKind {
  const names = [bodyName, meshName].filter(Boolean).map((s) => String(s).toLowerCase());
  for (const [re, kind] of MARS_RULES) {
    if (names.some((n) => re.test(n))) return kind;
  }
  return "chassis";
}

/** The material for one part kind, in one colorway.
 *
 *  The numbers, and why: the chassis and head carry METALNESS and a CLEARCOAT
 *  because that is what makes a dark shell legible against a dark stage —
 *  the diffuse term is nearly the background, and the specular top draws a
 *  highlight along every edge and panel line. The arm is semi-gloss injection
 *  moulding (bright enough already; a mirror finish just blew the orange
 *  out), and the gripper is a touch matter so the fingers separate from the
 *  wrist they sit against. `marker` is not transparent but INVISIBLE: a
 *  transparent material still sorts and still costs a draw. */
export function makeMarsMaterial(
  kind: MarsPartKind,
  cw: MarsColorway,
  opts: { envMap?: THREE.Texture | null; envScale?: number } = {}
): THREE.Material {
  const envMap = opts.envMap ?? null;
  // A shell that asks for less of the reflection map gets it on EVERY part:
  // the wash flattens an arm exactly the way it flattens a chassis.
  const env = (opts.envScale ?? 1) * (cw.reflect ?? 1);
  switch (kind) {
    case "head":
      // The camera housing: the glossiest PAINTED part on the robot — hence
      // a strong clearcoat but low metalness. At metalness 0.25 the mixed
      // orange head rendered as polished copper on the lab stage (looked at
      // it), which is a different object from a gloss-painted shell.
      return new THREE.MeshPhysicalMaterial({
        envMap, color: marsHeadColor(cw), roughness: 0.3, metalness: 0.08,
        clearcoat: 0.75, clearcoatRoughness: 0.14, envMapIntensity: 1.1 * env,
      });
    case "arm":
      return new THREE.MeshPhysicalMaterial({
        envMap, color: cw.accent, roughness: 0.3, metalness: 0.1,
        clearcoat: 0.55, clearcoatRoughness: 0.22, envMapIntensity: 1.0 * env,
      });
    case "gripper":
      return new THREE.MeshPhysicalMaterial({
        envMap, color: cw.accent, roughness: 0.38, metalness: 0.12,
        clearcoat: 0.35, clearcoatRoughness: 0.3, envMapIntensity: 0.9 * env,
      });
    case "marker":
      return new THREE.MeshBasicMaterial({ visible: false });
    default:
      return new THREE.MeshPhysicalMaterial({
        envMap, color: cw.body, roughness: 0.38, metalness: 0.3,
        clearcoat: 0.45, clearcoatRoughness: 0.3, envMapIntensity: 1.1 * env,
      });
  }
}

/** The MARS material array, indexed like MARS_KINDS (the merged geometry's
 *  groups carry the index). Null when `enabled` is false, so a stage with no
 *  MARS on it pays nothing — same contract as `useG1Materials`.
 *
 *  `envScale` defaults to the LAB's 0.35 for the same reason the G1's hook
 *  does: the lab stage already has a hemisphere and two directional lights
 *  that /sim does not, and the full reflection on top of them washes the
 *  shell out. /sim passes 1. */
export function useMarsMaterials(
  enabled: boolean,
  opts: { colorway?: string | null; envScale?: number } = {}
): THREE.Material[] | null {
  const gl = useThree((s) => s.gl);
  const { colorway = null, envScale = 0.35 } = opts;
  return useMemo(() => {
    if (!enabled) return null;
    const cw = marsColorway(colorway);
    const envMap = roomEnv(gl);
    return MARS_KINDS.map((k) => makeMarsMaterial(k, cw, { envMap, envScale }));
  }, [enabled, gl, colorway, envScale]);
}
