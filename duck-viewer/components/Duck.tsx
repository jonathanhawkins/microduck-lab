"use client";

// One duck: a Three.js group per MuJoCo body, poses smoothed toward the
// latest streamed frame. All geoms of a body are merged into ONE geometry at
// load time, so a duck is ~16 draw calls — the naive per-geom version
// (70 meshes/duck, all casting shadows) lost the WebGL context with 8 ducks
// on screen. Per-geom MJCF material colors survive the merge as a
// vertex-color channel, so the eye ring / mouth / shells keep their own
// colors without costing extra draw calls.

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import type { DuckFrame, Scene, SceneGeom } from "@/lib/lab";
import { assignDrag } from "@/lib/assign";
import { captureWantsCleanFrame } from "@/lib/record";
import { getSelectedDuck } from "@/lib/select";
import { getDuckLabels } from "@/lib/ui";
import { G1_KINDS, g1PartKind, useG1Materials, weldAndSmooth } from "./G1Look";
import { MARS_KINDS, marsPartKind, useMarsMaterials } from "./MarsLook";
import { duckMouths, MOUTH_TRAVEL_RAD } from "@/lib/mouth";
import { SHELL_MATERIALS, TEAM_COLORWAYS, TRIM_MATERIALS, teamColor, type TeamName, POSE_SMOOTH_HZ, simRate } from "@/lib/sim";

// FALLBACK body-name → color, used only against servers that predate rgba
// streaming (whole body painted one guessed color).
function bodyColor(name: string): string {
  if (/foot|ankle/.test(name)) return "#e8862e"; // webbed orange
  if (/head|beak/.test(name)) return "#f5efe0";
  if (/neck/.test(name)) return "#e8862e";
  if (/trunk/.test(name)) return "#f5efe0"; // cream body
  if (/hip|knee|leg/.test(name)) return "#4a4e57"; // dark joints
  return "#c8c2b4";
}

// The MJCF materials are OnShape-export appearances, and a few don't match
// the printed robot: the yellow eye ring exports dark grey, the soft TPU
// mouth pink, and the beak/shoes a flat gold where the real parts are
// orange with yellow soles. Override those by material name; everything
// else renders straight from the streamed rgba.
// Per-material colour overrides by MJCF material name. The 2026-09 upstream
// CAD re-export (microduck_rl #29) carries the right colours itself — orange
// beak and shoes, yellow eye ring and soles, light-grey face — so the table is
// empty; it stays as the place to put the next export's mistakes.
const MATERIAL_FIX: Record<string, string> = {};

/** A team's repaint of the printed parts, by MJCF material name: the four
 *  shell parts (head, trunk, legs, hips) take the colorway and the trim parts
 *  (beak, feet, ankles, soles) take its trim — every printed part gets one of
 *  the two, so a duck is one colour from the beak down. The server paints the
 *  same names in the composed world (world/compose.py); the viewer has to do
 *  its own because it draws every duck from ONE single-robot scene. */
function teamPaint(team: string | null | undefined): Record<string, string> {
  const look = teamColor(team) && team && team in TEAM_COLORWAYS ? TEAM_COLORWAYS[team as TeamName] : null;
  if (!look) return {};
  const out: Record<string, string> = {};
  for (const m of SHELL_MATERIALS) out[m] = look.shell;
  for (const m of TRIM_MATERIALS) out[m] = look.trim;
  return out;
}

/** Resolved sRGB→linear color for one geom (team → override → MJCF rgba → fallback). */
function geomColor(g: SceneGeom, bodyName: string, out: THREE.Color, paint: Record<string, string> = {}): THREE.Color {
  const team = g.mat ? paint[g.mat] : undefined;
  if (team) return out.set(team);
  const fix = g.mat ? MATERIAL_FIX[g.mat] : undefined;
  if (fix) return out.set(fix);
  if (g.rgba) return out.setRGB(g.rgba[0], g.rgba[1], g.rgba[2], THREE.SRGBColorSpace);
  return out.set(bodyColor(bodyName));
}

export interface BodyGeometry {
  name: string;
  geometry: THREE.BufferGeometry | null;
  /** Which material set draws this body (lib/robots.robotLook):
   *
   *  - undefined / "duck" — the merged vertex-colour mesh the duck has
   *    always been;
   *  - "g1" — welded + smoothed, groups indexing G1_KINDS materials
   *    (components/G1Look.tsx), the same look as the /sim page's G1;
   *  - "mars" — the same shape of thing for MARS_KINDS
   *    (components/MarsLook.tsx): a graphite chassis and head that survive
   *    the dark stage, the colorway's accent on the arm and gripper, and the
   *    frame markers drawn as nothing;
   *  - "generic" — welded + smoothed like the G1 but painted per geom from
   *    the scene dump's own `rgba`. That is how a Menagerie model arrives in
   *    its MJCF's colours, with no component and no colour table per robot.
   */
  look?: "g1" | "generic" | "mars";
}

/** Merge every geom of every body into one geometry per body (body-local
 *  frame), painting each geom's material color into a vertex-color channel.
 *
 *  `team` repaints the printed parts in that colorway. The colour is baked
 *  into the geometry, so a caller wanting two teams on screen builds one set
 *  PER COLORWAY and shares it across that team's ducks — not one per duck.
 *  Eight ducks with a set each is what lost the WebGL context before the
 *  bodies were merged at all (duck-viewer/README.md). */
export function buildBodyGeometries(
  scene: Scene,
  team?: string | null,
  opts: { look?: "g1" | "generic" | "mars" } = {}
): BodyGeometry[] {
  const paint = teamPaint(team);
  // A non-duck body: welded + smoothed, one draw per material group. The G1
  // and MARS group by PART KIND (a visor or a chassis, each with its own
  // material); "generic" groups by nothing and is drawn from its own vertex
  // colours, so a robot the viewer has never seen still arrives in its own
  // paint. `kinds` is both the switch and the index table — a look with a
  // material array is exactly a look with a kind list.
  const kinds: readonly string[] | null =
    opts.look === "g1" ? G1_KINDS : opts.look === "mars" ? MARS_KINDS : null;
  const cad = kinds !== null || opts.look === "generic";
  // Vertices arrive in metres (the duck) or in millimetre ints with a
  // vertScale (the G1 — a 21 MB dump instead of 78 MB of floats). Scaling
  // here rather than at the call site is what keeps a second robot from
  // arriving 1000x too big, off camera, with nothing in the console.
  const vs = scene.vertScale ?? 1;
  const meshGeos = scene.meshes.map((m) => {
    const g = new THREE.BufferGeometry();
    const v = vs === 1 ? m.v : m.v.map((x) => x * vs);
    g.setAttribute("position", new THREE.Float32BufferAttribute(v, 3));
    g.setIndex(m.f);
    return g;
  });
  const mat = new THREE.Matrix4();
  const quat = new THREE.Quaternion();
  const col = new THREE.Color();
  const out = scene.bodies.map((name, b) => {
    const parts = scene.geoms
      .filter((g) => g.body === b)
      .map((g) => {
        const geo = cad ? weldAndSmooth(meshGeos[g.mesh]) : meshGeos[g.mesh].clone();
        quat.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]); // wxyz → xyzw
        mat.compose(new THREE.Vector3(...g.pos), quat, new THREE.Vector3(1, 1, 1));
        geo.applyMatrix4(mat);
        geomColor(g, name, col, paint);
        const n = geo.getAttribute("position").count;
        const colors = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) col.toArray(colors, i * 3);
        geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
        const kind =
          opts.look === "g1"
            ? G1_KINDS.indexOf(g1PartKind(g.name, g.mat, col))
            : opts.look === "mars"
              ? MARS_KINDS.indexOf(marsPartKind(name, g.name))
              : 0;
        return { geo, kind };
      });
    if (!parts.length) return { name, geometry: null };
    if (!cad) {
      const merged = mergeGeometries(parts.map((p) => p.geo), false);
      parts.forEach((p) => p.geo.dispose());
      merged.computeVertexNormals();
      return { name, geometry: merged };
    }
    if (!kinds) {
      // Generic: keep the welded normals (re-smoothing the merge would blend
      // across part seams, which is what makes a CAD robot look melted) and
      // merge into ONE group — the per-geom colour already rode in on the
      // vertex-colour channel, so one draw call paints the whole body.
      const merged = mergeGeometries(parts.map((p) => p.geo), false);
      parts.forEach((p) => p.geo.dispose());
      return { name, geometry: merged, look: "generic" as const };
    }
    // G1 / MARS: keep each part's welded normals (re-smoothing the merge would
    // blend across part seams) and group the parts by kind, one draw per
    // material.
    parts.sort((a, b) => a.kind - b.kind);
    const merged = mergeGeometries(parts.map((p) => p.geo), true);
    parts.forEach((p) => p.geo.dispose());
    const groups: { start: number; count: number; materialIndex: number }[] = [];
    merged.groups.forEach((grp, i) => {
      const last = groups[groups.length - 1];
      if (last && last.materialIndex === parts[i].kind) last.count += grp.count;
      else groups.push({ start: grp.start, count: grp.count, materialIndex: parts[i].kind });
    });
    merged.clearGroups();
    groups.forEach((grp) => merged.addGroup(grp.start, grp.count, grp.materialIndex));
    return { name, geometry: merged, look: opts.look as "g1" | "mars" };
  });
  meshGeos.forEach((g) => g.dispose());
  return out;
}

export function Duck({
  bodies,
  frameRef,
  offset,
  label,
  duckId,
}: {
  bodies: BodyGeometry[];
  frameRef: React.MutableRefObject<DuckFrame | null>;
  offset: [number, number]; // grid offset in MuJoCo XY
  label: string;
  duckId: string; // stable stream id ("d0"…, "trainee") — assignment target
}) {
  const bodyRefs = useRef<(THREE.Group | null)[]>([]);
  // The hinged lower bill (world/compose.py `split_jaw`): its group takes the
  // streamed pose like every body, and its MESH takes the voice on top — a
  // rotation about the body origin, which is the pivot, on the hinge axis.
  const mouthIdx = useMemo(() => bodies.findIndex((b) => b.name === "mouth"), [bodies]);
  const billRef = useRef<THREE.Mesh>(null);
  const g1Materials = useG1Materials(bodies.some((b) => b.look === "g1"));
  // MARS's material table. The colorway is a VIEWER default today
  // (MARS_DEFAULT_COLORWAY, Innate's orange/black hero): a duck's colorway
  // rides in on the frame's `team`, and that channel carries the four Pollen
  // DUCK colorways — the lab validates it against them and rejects anything
  // else (world/scenario.py), so a MARS cannot borrow it. A per-slot choice
  // would pass a `colorway` here from a new field on the roster row.
  const marsMaterials = useMarsMaterials(bodies.some((b) => b.look === "mars"));
  // A generic body is lit like the G1's shell but takes its colour from the
  // geometry, so one material serves every one of them.
  const genericMaterial = useMemo(
    () => new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.42, metalness: 0.2 }),
    []
  );
  const labelRef = useRef<THREE.Group>(null);
  const labelDivRef = useRef<HTMLDivElement>(null);
  const spawnDivRef = useRef<HTMLDivElement>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const ballRef = useRef<THREE.Mesh>(null);
  const tmpP = useMemo(() => new THREE.Vector3(), []);
  const tmpQ = useMemo(() => new THREE.Quaternion(), []);

  useFrame((_, dt) => {
    const duck = frameRef.current;
    if (!duck) return;
    // Scaled by the world's speed: the filter's lag is fixed in WALL
    // time, the sim time a frame carries is not (lib/sim.ts POSE_SMOOTH_HZ).
    const alpha = 1 - Math.exp(-POSE_SMOOTH_HZ * simRate.speed * Math.min(dt, 0.1));
    duck.bodies.forEach((pose, b) => {
      const grp = bodyRefs.current[b];
      if (!grp) return;
      tmpP.set(pose[0], pose[1], pose[2]);
      tmpQ.set(pose[4], pose[5], pose[6], pose[3]); // wxyz → xyzw
      grp.position.lerp(tmpP, alpha);
      grp.quaternion.slerp(tmpQ, alpha);
    });
    // Lip-sync: open the bill to the wider of what the servo is doing and
    // what the voice asks, so a duck carrying a toy keeps its grip while it
    // chirps, and a silent duck shows exactly the physics. The servo's
    // opening is already in the streamed pose, so only the EXCESS is added.
    if (billRef.current) {
      const voice = duckMouths.open(duckId, performance.now() / 1000);
      billRef.current.rotation.y = Math.max(0, voice - (duck.mouth ?? 0)) * MOUTH_TRAVEL_RAD;
    }
    // Float the label above the trunk (body 1 = trunk_base in this model).
    const trunk = duck.bodies[1];
    if (labelRef.current && trunk) {
      tmpP.set(trunk[0], trunk[1], trunk[2] + 0.22);
      labelRef.current.position.lerp(tmpP, alpha);
    }
    // Drop-target highlight while a policy chip is dragged/armed: floor ring
    // under the duck + emphasized label. Driven straight off the shared store
    // (no React state — this flips at pointer speed). The same ring doubles
    // as the click-selection marker (Delete removes the selected duck); an
    // active assign hover wins the color so the drop target stays legible.
    const hovered = assignDrag.mode !== null && assignDrag.hoverDuck === duckId;
    const selected = getSelectedDuck() === duckId;
    // The ring lives IN the 3D scene, so unlike the DOM labels it would land
    // in captured footage — hide it while a 🎥 take is framing/rolling.
    // (📷 snapshots hide it themselves via the hideInCapture tag below.)
    const filming = captureWantsCleanFrame();
    if (ringRef.current) {
      ringRef.current.visible = (hovered || selected) && !filming;
      (ringRef.current.material as THREE.MeshBasicMaterial).color.set(
        hovered ? "#7db8d8" : "#e8b24a"
      );
      if (trunk) {
        tmpP.set(trunk[0], trunk[1], 0.004);
        ringRef.current.position.lerp(tmpP, alpha);
      }
    }
    // The find_ball ball: streamed as [x, y, z, r] in the duck's own frame
    // (it lives in the env, not the physics, so it is not a body). Hidden
    // for every other brain.
    if (ballRef.current) {
      const ball = duck.ball;
      ballRef.current.visible = !!ball;
      if (ball) {
        tmpP.set(ball[0], ball[1], ball[2]);
        ballRef.current.position.lerp(tmpP, alpha);
        const r = ball[3] || 0.035;
        ballRef.current.scale.set(r, r, r);
      }
    }
    if (labelDivRef.current) {
      const s = labelDivRef.current.style;
      // HUD 🏷 toggle, applied per-frame like the rest of the label styling
      // (a React subscription inside the Canvas tree flushed a beat late).
      s.display = getDuckLabels() ? "" : "none";
      s.transform = hovered ? "scale(1.3)" : selected ? "scale(1.15)" : "none";
      s.color = hovered ? "#9fd4f0" : selected ? "#e8b24a" : "#fff";
      s.fontWeight = hovered || selected ? "700" : "400";
    }
    // Spawn note updates straight off the stream (textContent, no React
    // churn) — it changes at every episode reset.
    if (spawnDivRef.current) {
      const spawn = duck.spawn && duck.spawn !== "standing" ? `↻ ${duck.spawn}` : "";
      const parts = [spawn];
      if (duck.assist) parts.push("🤝 spotting");
      if (duck.handed && duck.handoff) parts.push(`→ ${duck.handoff}`);
      const txt = parts.filter(Boolean).join(" · ");
      if (spawnDivRef.current.textContent !== txt)
        spawnDivRef.current.textContent = txt;
    }
  });

  return (
    <group position={[offset[0], offset[1], 0]}>
      {bodies.map((body, b) =>
        body.geometry ? (
          <group key={b} ref={(el) => void (bodyRefs.current[b] = el)}>
            {body.look === "g1" && g1Materials ? (
              <mesh geometry={body.geometry} material={g1Materials} />
            ) : body.look === "mars" && marsMaterials ? (
              <mesh geometry={body.geometry} material={marsMaterials} />
            ) : body.look === "generic" ? (
              <mesh geometry={body.geometry} material={genericMaterial} />
            ) : (
              <mesh geometry={body.geometry} ref={b === mouthIdx ? billRef : undefined}>
                <meshStandardMaterial vertexColors roughness={0.55} metalness={0.08} />
              </mesh>
            )}
          </group>
        ) : (
          <group key={b} ref={(el) => void (bodyRefs.current[b] = el)} />
        )
      )}
      {/* the ball a 🔎 find_ball duck is looking for (unit sphere, scaled to
          the streamed radius) — orange like the real 70 mm kick ball */}
      <mesh ref={ballRef} visible={false} castShadow>
        <sphereGeometry args={[1, 24, 16]} />
        <meshStandardMaterial color="#ff8c00" roughness={0.5} />
      </mesh>
      {/* drop-target ring, flat on the floor (XY plane in this Z-up group).
          hideInCapture: 📷 snapshots hide it for their capture render. */}
      <mesh
        ref={ringRef}
        visible={false}
        position={[0, 0, 0.004]}
        userData={{ hideInCapture: true }}
      >
        <ringGeometry args={[0.16, 0.19, 48]} />
        <meshBasicMaterial
          color="#7db8d8"
          transparent
          opacity={0.85}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      <group ref={labelRef}>
        {/* DOM label (drei Text's GPU glyph atlas was losing the WebGL
            context in the embedded browser — keep labels off the GPU).
            zIndexRange tops out below the overlay panels (zIndex 20) so
            labels can never scribble over the HUD/policies/teach UI.
            Stays mounted when labels are toggled off — the useFrame above
            flips the inner div's display instead. */}
        <Html center zIndexRange={[10, 0]} style={{ pointerEvents: "none" }}>
          <div
            ref={labelDivRef}
            style={{
              color: "#fff",
              fontFamily: "ui-monospace, Menlo, monospace",
              fontSize: 11,
              whiteSpace: "nowrap",
              textShadow: "0 1px 3px rgba(0,0,0,0.9)",
              opacity: 0.9,
              transition: "transform 120ms ease, color 120ms ease",
            }}
          >
            {label}
            <div
              ref={spawnDivRef}
              style={{
                fontSize: 9,
                color: "#e8b24a",
                textAlign: "center",
                minHeight: 11,
              }}
            />
          </div>
        </Html>
      </group>
    </group>
  );
}
