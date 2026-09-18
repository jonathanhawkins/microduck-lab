"use client";

// How a Unitree G1 is drawn — shared by the lab stage (Duck.tsx, via
// buildBodyGeometries' `look: "g1"`) and the /sim page's G1 person
// (SimStage.tsx), so the two can't drift apart again.
//
// Two things make it read as the real robot instead of a low-poly CAD dump:
//   1. WELD before smoothing. The STL/OBJ meshes duplicate every vertex per
//      face, so computeVertexNormals alone gives each triangle its own flat
//      normal — the faceted look. mergeVertices first, then normals.
//   2. A material per PART KIND: a glossy visor, dark metal on the joints and
//      feet, a flat logo, and a clear-coated pale shell — each lit by a
//      procedural RoomEnvironment reflection (clearcoat and metalness look
//      like grey plastic without something to reflect).

import { useThree } from "@react-three/fiber";
import { useMemo } from "react";
import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { mergeVertices } from "three/addons/utils/BufferGeometryUtils.js";

export const G1_KINDS = ["body", "dark", "visor", "logo"] as const;
export type G1PartKind = (typeof G1_KINDS)[number];

/** The shell colour: the G1's MJCF paints it a flat mid grey. */
export const G1_SHELL = "#d8d8d8";

/** Which material a G1 geom takes, from its mesh name, MJCF material and colour. */
export function g1PartKind(meshName: string | undefined, mat: string | undefined, col: THREE.Color): G1PartKind {
  const name = (meshName ?? "").toLowerCase();
  const lum = 0.2126 * col.r + 0.7152 * col.g + 0.0722 * col.b;
  if (/head/.test(name)) return "visor";
  if (/logo/.test(name)) return "logo";
  if (mat === "black" || lum < 0.25) return "dark";
  return "body";
}

/** Weld a CAD mesh's per-face duplicate vertices, then smooth its normals. */
export function weldAndSmooth(g: THREE.BufferGeometry): THREE.BufferGeometry {
  const welded = mergeVertices(g, 4e-4);
  welded.computeVertexNormals();
  return welded;
}

/** The material for one part kind. `vertexColors` takes the geom's own colour
 *  (visor, dark, logo) from the geometry; the shell ignores it for G1_SHELL. */
export function makeG1Material(
  kind: G1PartKind,
  opts: { color?: string; vertexColors?: boolean; envMap?: THREE.Texture | null; envScale?: number } = {}
): THREE.Material {
  const base = { vertexColors: !!opts.vertexColors && kind !== "body", envMap: opts.envMap ?? null };
  const env = opts.envScale ?? 1;
  const color = kind === "body" ? G1_SHELL : opts.vertexColors ? "#ffffff" : (opts.color ?? "#888888");
  switch (kind) {
    case "visor":
      return new THREE.MeshPhysicalMaterial({ ...base, color, roughness: 0.08, metalness: 0.85, clearcoat: 1, clearcoatRoughness: 0.06, envMapIntensity: 1.6 * env });
    case "logo":
      return new THREE.MeshStandardMaterial({ ...base, color, roughness: 0.55, metalness: 0, envMapIntensity: 0.4 * env });
    case "dark":
      return new THREE.MeshStandardMaterial({ ...base, color, roughness: 0.35, metalness: 0.45, envMapIntensity: 0.9 * env });
    default:
      return new THREE.MeshPhysicalMaterial({ ...base, color, roughness: 0.34, metalness: 0.06, clearcoat: 0.35, clearcoatRoughness: 0.28, envMapIntensity: 0.75 * env });
  }
}

// One reflection map per renderer. It goes on the G1's materials only, not on
// scene.environment, so the lab's ducks keep exactly the lighting they had.
const envByRenderer = new WeakMap<THREE.WebGLRenderer, THREE.Texture>();

/** The shared procedural reflection, built once per renderer. Exported so a
 *  second CAD look (MarsLook.tsx) reflects the SAME room rather than paying
 *  for a second PMREM render of an identical scene. */
export function roomEnv(gl: THREE.WebGLRenderer): THREE.Texture {
  let env = envByRenderer.get(gl);
  if (!env) {
    const pmrem = new THREE.PMREMGenerator(gl);
    env = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();
    envByRenderer.set(gl, env);
  }
  return env;
}

/** The G1 material array, indexed like G1_KINDS (geometry groups carry the
 *  index). Null when `enabled` is false, so a duck pays nothing.
 *
 *  The lab stage already has a hemisphere and two directional lights, which
 *  /sim does not; the full /sim reflection on top of them blew the shell out
 *  to flat white. The lab takes the same materials at 0.35 of the reflection. */
export function useG1Materials(enabled: boolean): THREE.Material[] | null {
  const gl = useThree((s) => s.gl);
  return useMemo(() => {
    if (!enabled) return null;
    const envMap = roomEnv(gl);
    return G1_KINDS.map((k) => makeG1Material(k, { vertexColors: true, envMap, envScale: 0.35 }));
  }, [enabled, gl]);
}
