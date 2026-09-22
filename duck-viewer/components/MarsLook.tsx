"use client";

// How an Innate MARS is drawn — shared by the lab stage (Duck.tsx, via
// buildBodyGeometries' `look: "mars"`) and the /sim page (SimStage.tsx's
// RobotBody), so the two cannot drift apart the way the G1's two draws once
// did.
//
// WHY this file exists at all. MARS used to take the "generic" look: every
// geom painted from the scene dump's own `rgba`, which was then the SERVER's
// port of Innate's sim styling — links 1/3/5 bright orange, "matt black"
// lifted to charcoal 0.16, frame markers hidden. 0.16 is a lighter BLACK, not
// a visible grey: against the lab stage's #101216 background the chassis and
// the head landed within a few percent of the backdrop, so the only part of
// the robot that read was the orange arm and MARS looked like an arm floating
// in the dark.
//
// The fix belongs HERE, not in the server's dump, because a colour is not the
// whole answer: it needs a MATERIAL. A dark shell reads on a dark stage
// because of its SPECULAR top — a clearcoat and a reflection give the chassis
// a highlight along every edge, which is what shows its shape. Diffuse colour
// alone would have to go so light the robot stopped looking like the machine
// it is. What the server sends is a per-geom rgba and nothing else, so the
// gloss, the metalness and the rubber can only be decided here.
//
// WHAT IS NO LONGER TRUE is that the two are independent. This file used to
// say the shell was a STAGE decision — one colorway per floor, the server's
// rgba left alone — and that produced a robot that was orange on /sim, white
// on the lab stage and grey in every MuJoCo contact sheet. The server paints
// Innate's Blue / White onto the spec now (robots/mars.paint_shell), both
// pages default to the same shell, and the three COLOURS here are the three
// there; what stays a stage decision is the reflection each page takes
// (`envScale`) and the material each part kind wears.
//
// WHAT THE WHITE SHELL ADDED. The default shell is a light one
// (MARS_DEFAULT_COLORWAY), and a near-white shell inverts the rule above
// exactly. A dark shell is legible BECAUSE of the reflection; a light one is
// flattened by it, because the wash lands on the shadowed faces where there is
// no diffuse signal to compete with it. Measured with the white body on the
// UNSCALED reflection (`reflect` absent, /sim's envScale 1): the chassis crop
// went to mean 0.896 with a spread of sd 0.038 and a darkest pixel of 0.835 —
// the panel seams, the turret and the wheels in one blob at the top of the
// range — against sd 0.096 and a darkest 0.505 on the lab stage, which passes
// the same map at 0.35. Taking 0.3 of the map put /sim back at sd 0.089. So
// `reflect` scales the shared reflection map per COLORWAY, and the two light
// shells take 0.3 of it: one number, applied to every painted part kind, left
// at 1 (bit-identical materials) for the two dark shells whose bodies still
// need the specular top. The tyres are exempt — see `makeMarsMaterial`.
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

export const MARS_KINDS = ["chassis", "wheel", "servo", "head", "face", "lens", "arm", "gripper", "marker"] as const;
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
  /** The two tyres. Defaults to INNATE_BLACK — every shell Innate sells rolls
   *  on the same black rubber, so this exists for a hypothetical one that
   *  does not rather than because any current row sets it. They are a part
   *  kind of their own because the SERVER now cuts them out of base.STL
   *  (microduck_local robots/mars.split_base_mesh); before that cut there was
   *  no geometry to give a second colour to, and the tyres were chassis. */
  wheel?: string;
  /** The bare servo cases (`servo` parts — one per arm joint). Defaults to
   *  INNATE_BLACK, which is what every Innate shell shows: the case is
   *  moulded, not painted, so it does NOT follow the colorway. */
  servo?: string;
  /** The head bar, when the `accent`-toward-`body` mix is not what the shell
   *  does. Innate's Blue / White carries FULL blue on the head — it is the
   *  most visible blue on the robot in their own photographs — and the mix
   *  would wash it halfway to the white chassis. Absent, `marsHeadColor`
   *  mixes as before, so the three other shells are untouched. */
  head?: string;
  /** The five arm links, when they are not the `accent`. A shell whose accent
   *  is a strong colour puts it on the GRIPPER and leaves the arm in the body
   *  family (Innate's Blue / White is a white arm with a blue claw); a shell
   *  whose accent IS the arm leaves this out. */
  arm?: string;
  /** How much of the shared RoomEnvironment reflection this shell takes,
   *  0..1, default 1 (see the header). A dark shell reads BY its specular top
   *  and takes all of it; a near-white shell is flattened by it and takes a
   *  fraction. Scales `envMapIntensity` on every part kind, so it cannot make
   *  one part of a shell disagree with another. */
  reflect?: number;
}

/** The tyres AND the bare servo cases, on every shell that does not say
 *  otherwise — one colour, because on the real robot the tread and the servo
 *  case are the same moulded black. A touch above #000 so the tread and the
 *  case edges read as shapes instead of holes: the same argument
 *  MIN_BODY_LUMA makes for the chassis, which these two are exempt from
 *  because neither is ever the silhouette. Matches the server's
 *  INNATE_BLACK (microduck_local robots/mars.py) so a MuJoCo contact sheet
 *  and the browser show one robot. */
export const INNATE_BLACK = "#25262a";

/** The head's face panel and its two camera lenses. Both are moulded parts
 *  on every shell Innate sells — no colorway touches them — so they are
 *  constants rather than `MarsColorway` fields, the way INNATE_BLACK is.
 *  SAMPLED off Innate's own front shot: the panel's dark quartile is
 *  rgb(31, 35, 42), a BLUE-black rather than a neutral one, and the lens
 *  domes read up to rgb(72, 72, 77), neutral and clearly lighter than the
 *  panel they sit in. Matching the server's FACE_BLACK / EYE_GREY
 *  (microduck_local robots/mars.py). */
export const FACE_BLACK = "#1f232a";
export const EYE_GREY = "#525459";

export const MARS_COLORWAYS: Record<string, MarsColorway> = {
  "orange-black": { id: "orange-black", label: "Orange / Black", accent: "#ff8a1f", body: "#3b3f47" },
  // The DEFAULT shell (see MARS_DEFAULT_COLORWAY), and the one Innate's own
  // product shots show: a white chassis and a white ARM, black tyres, and
  // their deep blue on the head bar and the gripper's two fingers only.
  //   accent  — SAMPLED off those shots rather than guessed: rgb(51, 97, 177)
  //             in the gripper's mid-tone, which is also what the server
  //             paints (INNATE_BLUE in robots/mars.py). Deliberately not
  //             Unitree's cyan-lean blue — the G1 stands on the same stage.
  //   arm     — 12 % darker than the body in Rec.709 luma, which is the White
  //             shell's number below and is here for its reason: a white arm
  //             drawn in the body colour lost its joints and its wrist into
  //             the chassis they fold against.
  //   reflect — 0.3: the light-shell rule from the header.
  "blue-white": {
    id: "blue-white", label: "Blue / White", accent: "#3361b1",
    arm: "#cbced5", head: "#3361b1", body: "#e8eaee", reflect: 0.3,
  },
  black: { id: "black", label: "Black", accent: "#5b616b", body: "#3b3f47" },
  // Innate's all-white shell: a white arm on a white body, no blue anywhere.
  // It was the default while "does the chassis read AT ALL" was the open
  // question, and its three numbers are the measured ones that answered it:
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

/** Innate's Blue / White shell — the machine in their own product shots, and
 *  what a slot with no colorway of its own is drawn in.
 *
 *  Two reasons, and the second is the one that changed:
 *
 *  1. A stage's first job is to let you FIND the robot. On the live lab page,
 *     at one camera, the same chassis crop measured 0.241 under the
 *     orange/black shell's graphite body and 0.697 under a white one — 3.1x
 *     the #101216 background against 8.6x — and the arm stopped being the
 *     only part of MARS you could see from across the grid. Note the two dark
 *     shells share that graphite body, so this is a change of default and not
 *     a claim that a dark MARS is unreadable: it reads, by its specular top,
 *     which is what the rest of this file is about.
 *  2. It is what the ROBOT looks like. The all-white shell above was the
 *     default first, and it is a real Innate colorway — but their hero
 *     machine is white with a BLUE head bar and blue gripper fingers on black
 *     tyres, and a viewer that draws a robot the owner would not recognise is
 *     wrong about the one thing it is for. The server paints the same three
 *     colours onto the spec now (robots/mars.paint_shell), so a MuJoCo
 *     contact sheet and this stage no longer disagree about which robot it
 *     is. The day a roster row carries a colorway, it is one field away. */
export const MARS_DEFAULT_COLORWAY = "blue-white";

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

/** The head's colour for a colorway: the shell's own `head` when it names one
 *  (Innate's Blue / White does — full blue, not a mix), otherwise the accent
 *  mixed back toward the body in LINEAR space (mixing sRGB components darkens
 *  a saturated accent unevenly). */
export function marsHeadColor(cw: MarsColorway): string {
  if (cw.head) return cw.head;
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
  // BARE SERVO CASES, before the arm rule, which would otherwise take all of
  // them. Every joint on the arm has one and none of them is painted; two
  // arrive as whole meshes and two as cuts:
  //   link1        — the first joint, the black box in its bracket by the
  //                  right wheel; link4 — the wrist.
  //   link2_servo  — the shoulder's housing, and link3_servo the elbow's.
  //                  Innate prints each case as one part with the arm it
  //                  drives, so the server cuts them off on a plane
  //                  (robots/mars.split_housing_mesh) and sends each as its
  //                  own `_servo` mesh on the arm's body.
  // A lab too old to send the cuts simply has no such geoms, and those two
  // arms stay white end to end.
  [/^link[14]$|_servo$/, "servo"],
  // The head's two cut parts, BEFORE the head rule (their names start with
  // `head`) and before the marker rule can be reached. The server cuts each
  // out of head.STL by a different method, because the mesh offers two: the
  // lenses are a shell of their own, the panel is the flat front of the blue
  // shell picked out by its NORMALS (robots/mars.split_head_mesh).
  [/head_face/, "face"],
  [/head_eyes/, "lens"],
  [/^link[1-5]$/, "arm"],
  [/head/, "head"],
  // The tyres, BEFORE the chassis rule — `base_wheels` matches both. They
  // used to be part of the chassis and could not be anything else: both
  // wheels are triangles inside base.STL, so a `wheel` kind had nothing to
  // select. The server cuts them out now (microduck_local
  // robots/mars.split_base_mesh writes a `base_wheels` mesh), and the dump
  // names a geom after its MESH — so this rule fires on the name that cut
  // brought with it, and a lab too old to send one paints white wheels
  // rather than dropping them.
  [/wheel|tyre|tire/, "wheel"],
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
        envMap, color: cw.arm ?? cw.accent, roughness: 0.3, metalness: 0.1,
        clearcoat: 0.55, clearcoatRoughness: 0.22, envMapIntensity: 1.0 * env,
      });
    case "face":
      // The panel behind which both cameras sit: matte, so it reads as a
      // window surround rather than another painted shell, and it takes
      // almost none of the reflection — a wash on a near-black panel is the
      // one thing that would stop the grey lenses separating from it.
      return new THREE.MeshPhysicalMaterial({
        envMap, color: FACE_BLACK, roughness: 0.55, metalness: 0.0,
        clearcoat: 0.25, clearcoatRoughness: 0.35,
        envMapIntensity: 0.3 * (opts.envScale ?? 1),
      });
    case "lens":
      // GLASS, and the only part of MARS that is: smooth, a full clearcoat
      // and the most reflection of any kind here, so each dome catches the
      // highlight Innate's own render shows and reads as an eye rather than
      // a grey disc.
      return new THREE.MeshPhysicalMaterial({
        envMap, color: EYE_GREY, roughness: 0.12, metalness: 0.15,
        clearcoat: 1.0, clearcoatRoughness: 0.05,
        envMapIntensity: 1.4 * (opts.envScale ?? 1),
      });
    case "servo":
      // Moulded ABS, not a painted panel: matte, no clearcoat, and exempt
      // from `cw.reflect` for the tyre's reason below — it is already dark.
      return new THREE.MeshPhysicalMaterial({
        envMap, color: cw.servo ?? INNATE_BLACK, roughness: 0.6, metalness: 0.05,
        clearcoat: 0.12, clearcoatRoughness: 0.4,
        envMapIntensity: 0.45 * (opts.envScale ?? 1),
      });
    case "wheel":
      // RUBBER, and the one part kind that is not painted: no clearcoat, no
      // metalness, and a quarter of the reflection the shell takes. It is
      // also the only kind that ignores `cw.reflect` — that number is the
      // light-shell correction from the header, and a black tyre is not a
      // light shell; scaling it again put the tread back into one flat blob.
      return new THREE.MeshPhysicalMaterial({
        envMap, color: cw.wheel ?? INNATE_BLACK, roughness: 0.85, metalness: 0.0,
        clearcoat: 0.0, envMapIntensity: 0.25 * (opts.envScale ?? 1),
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
