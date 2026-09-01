"use client";

// The 🎬 preview duck: a translucent ghost that shows the pose the animation
// editor is currently authoring. It is NOT a lab duck — no env, no policy, no
// WS stream. Its body transforms come from POST /pose (forward kinematics on
// the server's scratch model), so previewing can never perturb a live episode.
//
// It is also the direct-manipulation surface: click a body part to select the
// joint that moves it, drag to rotate that joint. The drag maps the pointer's
// SCREEN ANGLE around the joint's projected pivot back onto the hinge axis, so
// circling the cursor around a knee bends the knee the way it looks like it
// should — and it degrades gracefully when the axis points across the screen.

import { useEffect, useMemo, useRef, useSyncExternalStore } from "react";
import { useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import type { Scene } from "@/lib/lab";
import { RIG_CONTROLS } from "@/lib/rig";
import { buildBodyGeometries, type BodyGeometry } from "./Duck";
import {
  animStore,
  animVersion,
  PREVIEW_OFFSET,
  ROOT_SEL,
  setSelected,
  setSelectedRig,
  subscribeAnim,
} from "@/lib/anim";

// One extra merged-geometry set for the whole app, cached across mounts: the
// panel toggles this component on and off, and rebuilding ~16 merged meshes
// from the 20 MB scene payload each time would hitch the frame.
let geomCache: { scene: Scene; bodies: BodyGeometry[] } | null = null;
function previewGeometries(scene: Scene): BodyGeometry[] {
  if (!geomCache || geomCache.scene !== scene)
    geomCache = { scene, bodies: buildBodyGeometries(scene) };
  return geomCache.bodies;
}

const GHOST = new THREE.Color("#5fd0bd");
const GHOST_HOVER = new THREE.Color("#a9f0e4");
const SELECTED = new THREE.Color("#ffd166");
const EMIT_IDLE = new THREE.Color("#0e3a35");
const EMIT_SELECTED = new THREE.Color("#5a4108");

/** Below this the hinge axis lies almost in the screen plane and the screen
 *  angle stops tracking it — clamp so the drag stays usable instead of
 *  exploding. */
const MIN_AXIS_Z = 0.22;
/** Angle noise near the pivot is huge; ignore until the cursor is out here. */
const MIN_RADIUS_PX = 18;
/** Eye offset from the preview duck's trunk for the ◎ focus button (three.js
 *  world). ~1 m back reads the whole 25 cm robot without clipping. */
const EYE: [number, number, number] = [0.8, 0.39, 0.94];

interface DragState {
  joint: number; // joint index, or ROOT_SEL — in rig mode, the GEAR joint
  body: number; // body whose frame carries the hinge
  lastAngle: number | null; // screen angle (rad) at the previous move
  /** Set in rig mode: the drag's angle delta drives this control instead of
   *  the joint directly, geared so the part under the cursor tracks 1:1. */
  rig?: { id: string; coeff: number };
}

// --- the ⇕ rig handle: a game-rig-style gizmo for the ACTIVE rig control ---
// It drives whatever rig control is selected (squat when nothing is) and
// parks at that control's own anchor — head for look, thigh for a swing, the
// feet for toes — so where the diamond sits tells you what a drag will move.
/** Radians of control per pixel of vertical drag (down = +). */
const HANDLE_GAIN = 0.005;
const HANDLE_IDLE = new THREE.Color("#5fd0bd");
const HANDLE_HOVER = new THREE.Color("#a9f0e4");
const HANDLE_DRAG = new THREE.Color("#ffd166");

export function PoseDuck({ scene }: { scene: Scene }) {
  const visible = useSyncExternalStore(
    subscribeAnim,
    () => animStore.visible,
    () => false
  );
  if (!visible) return null;
  return <PoseDuckBody scene={scene} />;
}

function PoseDuckBody({ scene }: { scene: Scene }) {
  // Re-render on selection changes so materials/labels stay in step even when
  // no frame is being drawn (the per-frame path below does the fast work).
  useSyncExternalStore(subscribeAnim, animVersion, () => 0);

  const bodies = useMemo(() => previewGeometries(scene), [scene]);
  const { camera, gl } = useThree();
  const controls = useThree((s) => s.controls) as { enabled: boolean } | null;
  const rootRef = useRef<THREE.Group>(null);
  const groupRefs = useRef<(THREE.Group | null)[]>([]);
  const matRefs = useRef<(THREE.MeshStandardMaterial | null)[]>([]);
  const labelRef = useRef<THREE.Group>(null);
  const labelDivRef = useRef<HTMLDivElement>(null);
  const drag = useRef<DragState | null>(null);
  const rigDrag = useRef<{ lastY: number; id: string } | null>(null);
  const handleHover = useRef(false);
  const handleRef = useRef<THREE.Group>(null);
  const handleMatRefs = useRef<(THREE.MeshBasicMaterial | null)[]>([]);
  // The ⇕ handle's gesture functions live in the pointer-listener effect below
  // (the one place allowed to poke `controls`/`gl`); JSX reaches them here.
  const handleGestures = useRef<{
    down: (e: ThreeEvent<PointerEvent>) => void;
    over: (e: ThreeEvent<PointerEvent>) => void;
    out: () => void;
  } | null>(null);
  const tmpP = useMemo(() => new THREE.Vector3(), []);
  const tmpQ = useMemo(() => new THREE.Quaternion(), []);
  const tmpV = useMemo(() => new THREE.Vector3(), []);
  const axisV = useMemo(() => new THREE.Vector3(), []);
  const viewerDir = useMemo(() => new THREE.Vector3(), []);

  // --- per-frame: apply the previewed pose + selection tinting -------------
  useFrame((state, dt) => {
    const pose = animStore.bodies;
    // Until the first POST /pose lands, every body group still sits at the
    // group origin — a heap of parts on the floor. Stay hidden instead.
    if (rootRef.current) rootRef.current.visible = pose !== null;
    if (pose) {
      // Fast lerp: smooths the HTTP round trip without feeling laggy.
      const alpha = 1 - Math.exp(-45 * Math.min(dt, 0.1));
      pose.forEach((p, b) => {
        const grp = groupRefs.current[b];
        if (!grp) return;
        tmpP.set(p[0], p[1], p[2]);
        tmpQ.set(p[4], p[5], p[6], p[3]); // wxyz → xyzw
        grp.position.lerp(tmpP, alpha);
        grp.quaternion.slerp(tmpQ, alpha);
      });
      const trunk = pose[1];
      if (labelRef.current && trunk)
        labelRef.current.position.set(trunk[0], trunk[1], trunk[2] + 0.24);
      // The ⇕ handle parks at the ACTIVE control's anchor body (world-frame
      // offset, world-vertical rail) — it follows the part it moves and jumps
      // when the selection changes, which is how you see what it is armed with.
      const meta = animStore.meta;
      const active =
        RIG_CONTROLS.find((c) => c.id === (animStore.selectedRig?.id ?? "squat")) ??
        RIG_CONTROLS[0];
      const anchorBody =
        active.handle.joint === "root"
          ? meta?.trunkBody
          : meta?.joints.find((j) => j.name === active.handle.joint)?.body;
      const anchorGrp = anchorBody != null ? groupRefs.current[anchorBody] : null;
      if (handleRef.current && anchorGrp) {
        const [ox, oy, oz] = active.handle.offset;
        handleRef.current.position.set(
          anchorGrp.position.x + ox,
          anchorGrp.position.y + oy,
          anchorGrp.position.z + oz
        );
        const c = rigDrag.current ? HANDLE_DRAG : handleHover.current ? HANDLE_HOVER : HANDLE_IDLE;
        handleMatRefs.current.forEach((m) => m?.color.copy(c));
      }
    }
    const sel = animStore.selected;
    const selBody =
      sel == null ? -1 : animStore.meta?.joints.find((j) => j.index === sel)?.body ?? -1;
    const rootBody = sel === ROOT_SEL ? animStore.meta?.trunkBody ?? -1 : -1;
    // A selected rig control lights up EVERY body it drives — the coupling is
    // the thing being edited, and the highlight is how you read its extent.
    const rigBodies = animStore.selectedRig?.bodies;
    matRefs.current.forEach((m, b) => {
      if (!m) return;
      const isSel = rigBodies ? rigBodies.includes(b) : b === selBody || b === rootBody;
      const isHover = !isSel && b === animStore.hoveredBody;
      m.color.copy(isSel ? SELECTED : isHover ? GHOST_HOVER : GHOST);
      m.emissive.copy(isSel ? EMIT_SELECTED : EMIT_IDLE);
      m.opacity = isSel ? 0.95 : 0.72;
    });
    // Focus request from the panel's ◎ button — one-shot, consumed here so
    // the camera move lives with the thing it frames.
    if (animStore.focusRequest > 0 && pose) {
      animStore.focusRequest = 0;
      const c = state.controls as unknown as
        | { target: THREE.Vector3; update?: () => void }
        | null;
      const trunk = pose[1];
      if (c && trunk) {
        // MuJoCo (x, y, z) → three.js world (x, z, -y) through the scene's
        // -90°-about-X group, plus this duck's grid offset.
        const wx = trunk[0] + PREVIEW_OFFSET[0];
        const wy = trunk[2];
        const wz = -(trunk[1] + PREVIEW_OFFSET[1]);
        camera.position.set(wx + EYE[0], wy + EYE[1], wz + EYE[2]);
        const anchor = new THREE.Vector3(wx, wy, wz);
        c.target.copy(anchor);
        camera.lookAt(c.target);
        camera.updateMatrixWorld();
        // Frame the duck in the stage the editor panel is NOT covering. The
        // target is aimed BELOW the duck (which lifts it on screen) by an
        // amount measured through the actual projection — one finite-
        // difference probe beats any hand-tuned metre offset, and it stays
        // right at other FOVs, aspect ratios and panel heights.
        const toPx = (v: THREE.Vector3) =>
          ((1 - v.clone().project(camera).y) / 2) * state.size.height;
        const y0 = toPx(anchor);
        const pxPerMetre =
          (y0 - toPx(anchor.clone().add(new THREE.Vector3(0, 0.1, 0)))) / 0.1;
        const panelTop = animStore.panelEl?.getBoundingClientRect().top ?? state.size.height;
        const wantY = Math.max(90, Math.min(state.size.height * 0.55, panelTop * 0.68));
        if (Number.isFinite(pxPerMetre) && Math.abs(pxPerMetre) > 1) {
          c.target.set(wx, wy - (y0 - wantY) / pxPerMetre, wz);
          camera.lookAt(c.target);
        }
        c.update?.();
      }
    }
  });

  // --- drag: screen angle around the projected pivot → joint delta ---------

  /** World-space hinge pivot and axis, read off the CURRENT body transform
   *  (recomputed every move so the ring follows the part as it swings). */
  const hinge = (jointIdx: number, bodyIdx: number) => {
    const grp = groupRefs.current[bodyIdx];
    const meta = animStore.meta;
    if (!grp || !meta) return null;
    const j = jointIdx === ROOT_SEL ? null : meta.joints.find((x) => x.index === jointIdx);
    grp.updateWorldMatrix(true, false);
    // Hinge axis and anchor are given in the BODY frame; the trunk's pitch
    // axis is its own +Y (the rootPitch convention).
    const localAxis = j ? j.axis : [0, 1, 0];
    const localPos = j ? j.pos : [0, 0, 0];
    const pivot = tmpV.set(localPos[0], localPos[1], localPos[2]).applyMatrix4(grp.matrixWorld);
    const axis = axisV
      .set(localAxis[0], localAxis[1], localAxis[2])
      .transformDirection(grp.matrixWorld)
      .normalize();
    return { pivot, axis };
  };

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      // The ⇕ handle: vertical pixels → radians of whichever control it was
      // grabbed as (down = +, matching every control's "+ hint").
      const rd = rigDrag.current;
      if (rd) {
        const dy = e.clientY - rd.lastY;
        rd.lastY = e.clientY;
        animStore.applyRigDelta?.(rd.id, dy * HANDLE_GAIN * (e.shiftKey ? 0.25 : 1));
        return;
      }
      const d = drag.current;
      if (!d) return;
      const h = hinge(d.joint, d.body);
      if (!h) return;
      const rect = gl.domElement.getBoundingClientRect();
      const p = h.pivot.clone().project(camera);
      const px = rect.left + ((p.x + 1) / 2) * rect.width;
      const py = rect.top + ((1 - p.y) / 2) * rect.height;
      const rx = e.clientX - px;
      const ry = e.clientY - py;
      if (Math.hypot(rx, ry) < MIN_RADIUS_PX) return; // too close: angle is noise
      const angle = Math.atan2(ry, rx);
      if (d.lastAngle === null) {
        d.lastAngle = angle;
        return;
      }
      let delta = angle - d.lastAngle;
      while (delta > Math.PI) delta -= 2 * Math.PI;
      while (delta < -Math.PI) delta += 2 * Math.PI;
      d.lastAngle = angle;
      // How much of the hinge axis points at the viewer decides both the sign
      // and the gearing: face-on, the part tracks the cursor 1:1.
      viewerDir.set(0, 0, 1).applyQuaternion(camera.quaternion);
      let k = h.axis.dot(viewerDir);
      if (Math.abs(k) < MIN_AXIS_Z) k = MIN_AXIS_Z * (k < 0 ? -1 : 1);
      // Screen y is down, so a positive rotation about a viewer-facing axis
      // DEcreases the pixel-space angle — hence the negation.
      const step = (-delta / k) * (e.shiftKey ? 0.25 : 1);
      // In rig mode the gear joint's motion is coeff × the control's, so
      // dividing the step by coeff keeps the grabbed part under the cursor
      // while the rest of the coupling follows.
      if (d.rig) animStore.applyRigDelta?.(d.rig.id, step / d.rig.coeff);
      else animStore.applyJointDelta?.(d.joint, step);
    };
    const onUp = () => {
      if (!drag.current && !rigDrag.current) return;
      drag.current = null;
      rigDrag.current = null;
      animStore.dragging = false;
      if (controls) controls.enabled = true;
      gl.domElement.style.cursor = "";
    };
    // Grabbing the ⇕ handle starts a squat drag and parks OrbitControls for
    // the gesture — same lifecycle as a body-part drag, so it shares this
    // effect (and the compiler's blessing to mutate `controls`/`gl` here).
    handleGestures.current = {
      down: (e) => {
        e.stopPropagation();
        rigDrag.current = {
          lastY: e.nativeEvent.clientY,
          id: animStore.selectedRig?.id ?? "squat",
        };
        animStore.dragging = true;
        if (controls) controls.enabled = false;
        gl.domElement.style.cursor = "ns-resize";
      },
      over: (e) => {
        e.stopPropagation();
        handleHover.current = true;
        if (!animStore.dragging) gl.domElement.style.cursor = "ns-resize";
      },
      out: () => {
        handleHover.current = false;
        if (!animStore.dragging) gl.domElement.style.cursor = "";
      },
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      handleGestures.current = null;
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      if (controls) controls.enabled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera, gl, controls]);

  const onBodyDown = (b: number) => (e: ThreeEvent<PointerEvent>) => {
    const meta = animStore.meta;
    // Rig mode: the clicked part selects its mapped control, and the drag
    // circles the gear joint's hinge but drives the whole coupling.
    const pick = animStore.mode === "rig" ? animStore.rigForBody[b] : null;
    if (pick && meta) {
      e.stopPropagation();
      setSelectedRig({ id: pick.rigId, label: pick.label, bodies: pick.bodies });
      const gearBody =
        pick.gearJoint === ROOT_SEL ? meta.trunkBody : meta.joints[pick.gearJoint]?.body ?? b;
      drag.current = {
        joint: pick.gearJoint,
        body: gearBody,
        lastAngle: null,
        rig: { id: pick.rigId, coeff: pick.gearCoeff },
      };
    } else {
      const joint = animStore.jointForBody[b];
      if (joint == null) return; // world/static body — let the click fall through
      e.stopPropagation();
      setSelected(joint);
      drag.current = { joint, body: b, lastAngle: null };
    }
    animStore.dragging = true;
    // OrbitControls listens on the same canvas; disabling it for the gesture
    // is the only way to stop a pose drag from also orbiting the camera.
    if (controls) controls.enabled = false;
    gl.domElement.style.cursor = "grabbing";
  };

  const selectedName = animStore.selectedRig
    ? `🎮 ${animStore.selectedRig.label}`
    : animStore.selected == null
      ? null
      : animStore.selected === ROOT_SEL
        ? "root pitch"
        : animStore.meta?.joints[animStore.selected]?.name ?? null;

  return (
    <group ref={rootRef} visible={false} position={[PREVIEW_OFFSET[0], PREVIEW_OFFSET[1], 0]}>
      {bodies.map((body, b) =>
        body.geometry ? (
          <group key={b} ref={(el) => void (groupRefs.current[b] = el)}>
            <mesh
              geometry={body.geometry}
              onPointerDown={onBodyDown(b)}
              onPointerOver={(e) => {
                e.stopPropagation();
                if (animStore.jointForBody[b] == null) return;
                animStore.hoveredBody = b;
                if (!animStore.dragging) gl.domElement.style.cursor = "grab";
              }}
              onPointerOut={() => {
                if (animStore.hoveredBody === b) animStore.hoveredBody = null;
                if (!animStore.dragging) gl.domElement.style.cursor = "";
              }}
            >
              <meshStandardMaterial
                ref={(el) => void (matRefs.current[b] = el)}
                color={GHOST}
                emissive={EMIT_IDLE}
                transparent
                opacity={0.72}
                roughness={0.4}
                metalness={0.05}
              />
            </mesh>
          </group>
        ) : (
          <group key={b} ref={(el) => void (groupRefs.current[b] = el)} />
        )
      )}

      {/* The ⇕ rig handle — the way a game rig gives the animator one grabbable
          control per track: it drives the SELECTED rig control (squat when
          nothing is selected), parks at that control's anchor on the duck, and
          wears its name. Grab it, drag down = +. */}
      <group
        ref={handleRef}
        onPointerDown={(e) => handleGestures.current?.down(e)}
        onPointerOver={(e) => handleGestures.current?.over(e)}
        onPointerOut={() => handleGestures.current?.out()}
      >
        {/* diamond, with a vertical travel rail through it (trunk +z) */}
        <mesh>
          <octahedronGeometry args={[0.016, 0]} />
          <meshBasicMaterial ref={(el) => void (handleMatRefs.current[0] = el)} color={HANDLE_IDLE} />
        </mesh>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.0016, 0.0016, 0.085, 6]} />
          <meshBasicMaterial
            ref={(el) => void (handleMatRefs.current[1] = el)}
            color={HANDLE_IDLE}
            transparent
            opacity={0.6}
          />
        </mesh>
        {/* the handle wears the name of the control it is armed with */}
        <Html center zIndexRange={[10, 0]} position={[0, 0, -0.062]} style={{ pointerEvents: "none" }}>
          <div
            style={{
              color: "#8ee6d6",
              fontFamily: "ui-monospace, Menlo, monospace",
              fontSize: 9,
              whiteSpace: "nowrap",
              textShadow: "0 1px 3px rgba(0,0,0,0.9)",
            }}
          >
            ⇕ {animStore.selectedRig?.label ?? "squat"}
          </div>
        </Html>
      </group>

      {/* Locator ring — the ghost is translucent and easy to lose on a busy floor. */}
      <mesh position={[0, 0, 0.005]}>
        <ringGeometry args={[0.15, 0.175, 48]} />
        <meshBasicMaterial
          color="#5fd0bd"
          transparent
          opacity={0.5}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>

      <group ref={labelRef}>
        <Html center zIndexRange={[10, 0]} style={{ pointerEvents: "none" }}>
          <div
            ref={labelDivRef}
            style={{
              color: "#8ee6d6",
              fontFamily: "ui-monospace, Menlo, monospace",
              fontSize: 11,
              whiteSpace: "nowrap",
              textShadow: "0 1px 3px rgba(0,0,0,0.9)",
              textAlign: "center",
            }}
          >
            🎬 pose
            <div style={{ fontSize: 9, color: "#ffd166", minHeight: 11 }}>
              {selectedName ?? ""}
            </div>
          </div>
        </Html>
      </group>
    </group>
  );
}
