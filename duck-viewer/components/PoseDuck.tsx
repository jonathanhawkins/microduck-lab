"use client";

// The 🎬 preview robot: a translucent ghost that shows the pose the animation
// editor is currently authoring. It is NOT a lab duck — no env, no policy, no
// WS stream. Its body transforms come from POST /pose (forward kinematics on
// the server's scratch model), so previewing can never perturb a live episode.
//
// It is also the direct-manipulation surface: click a body part to select the
// joint that moves it, drag to rotate that joint. The drag maps the pointer's
// SCREEN ANGLE around the joint's projected pivot back onto the hinge axis, so
// circling the cursor around a knee bends the knee the way it looks like it
// should — and it degrades gracefully when the axis points across the screen.
//
// In 🎯 ik mode the surface is the effectors instead: one grabbable sphere per
// foot / hand / head, dragged in the view plane, and the server's solver
// (POST /ik) finds the joints that put it there.
//
// It draws whichever body the editor is posing (animStore.robot): the duck
// from the stage's own scene, any other robot from a /scene fetched here once.
// Every metre-sized constant below is multiplied by the body's `sizeScale`,
// so the G1's gizmos are G1-sized instead of duck-sized dots on a 1.3 m body.

import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { fetchScene, type RobotId, type Scene } from "@/lib/lab";
import { rigControlsFor } from "@/lib/rig";
import { dragPointOnViewPlane, ikDragStep, type V3 } from "@/lib/ikdrag";
import { buildBodyGeometries, type BodyGeometry } from "./Duck";
import {
  animStore,
  animVersion,
  BALANCE_IN,
  BALANCE_OUT,
  BALANCE_STANCE,
  balanceState,
  effectorForBody,
  previewOffset,
  ROOT_SEL,
  setSelected,
  setSelectedEffector,
  setSelectedRig,
  subscribeAnim,
} from "@/lib/anim";

// One merged-geometry set per scene for the whole app, cached across mounts:
// the panel toggles this component on and off (and switches robots), and
// rebuilding ~16 merged meshes from the 20 MB scene payload each time would
// hitch the frame. Keyed weakly so a scene the viewer drops is collectable.
const geomCache = new WeakMap<Scene, BodyGeometry[]>();
function previewGeometries(scene: Scene): BodyGeometry[] {
  let bodies = geomCache.get(scene);
  if (!bodies) {
    bodies = buildBodyGeometries(scene);
    geomCache.set(scene, bodies);
  }
  return bodies;
}

// The mesh set of a robot other than the stage's duck, fetched once per
// robot for the life of the page (the G1's is ~21 MB; the panel is not the
// place to pay that twice).
const sceneCache = new Map<RobotId, Promise<Scene>>();
function ghostScene(robot: RobotId): Promise<Scene> {
  let p = sceneCache.get(robot);
  if (!p) {
    p = fetchScene(robot);
    sceneCache.set(robot, p);
    p.catch(() => sceneCache.delete(robot)); // a 404 (assets not fetched) may be fixed later
  }
  return p;
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
/** Eye offset from the preview robot's trunk for the ◎ focus button (three.js
 *  world), for a duck-sized body. ~1 m back reads the whole 25 cm robot
 *  without clipping; a bigger body scales it out. */
const EYE: [number, number, number] = [0.8, 0.39, 0.94];
/** How far above the trunk the "🎬 pose" label floats (duck-sized). */
const LABEL_RISE = 0.24;

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
/** The diamond's radius, its rail's half-length and thickness, and where its
 *  name hangs below it — duck-sized metres. */
const HANDLE_R = 0.016;
const HANDLE_RAIL = 0.085;
const HANDLE_RAIL_R = 0.0016;
const HANDLE_LABEL_DROP = 0.062;

// --- the 🎯 IK handles: one grabbable sphere per effector -----------------
// Dragged in the VIEW PLANE (lib/ikdrag.ts): the pointer moves the point at
// the depth it already sits at. The colour is the answer at a glance —
// selected gold, hovered light, idle the ghost's teal, and amber once the
// solver could not get the limb there (a residual past IK_AMBER_M).
const IK_GIZMO_R = 0.013;
/** Residual beyond which a handle is drawn amber, duck-sized metres: a
 *  centimetre is where "not quite" becomes "that pose does not exist". */
const IK_AMBER_M = 0.01;
const IK_AMBER = new THREE.Color(BALANCE_OUT);

interface IkDragState {
  id: string;
  /** Where the pointer ray last crossed the drag plane, three.js WORLD. */
  lastHit: V3 | null;
  /** Effector minus hit at the grab (world), so the handle keeps its
   *  relationship to the pointer through the gesture. */
  grabOffset: V3;
  /** The camera's view direction at the grab — the plane normal. Orbit is
   *  parked for the gesture, so it does not change. */
  viewDir: V3;
}

// --- the ⊕ CoM marker: ball, plumb line, crosshair on the ground ----------
// Inside `rootRef` the frame is MuJoCo world, Z UP (hence the locator ring at
// z = 0.005), so the marker is built in that frame directly. It answers the
// question the ghost cannot: not what the pose LOOKS like, but whether the
// mass is over a foot.
const COM_BALL_R = 0.009;
/** Crosshair arm, half-length: about a sole's, so the cross reads against
 *  the foot it is being judged against. */
const CROSS_ARM = 0.024;
/** Just off the floor, under the locator ring: co-planar would z-fight. */
const FLOOR_Z = 0.0022;
const PLUMB_R = 0.0012;
const CROSS_RING: [number, number] = [0.0105, 0.0125];
/** The locator ring under the whole ghost. */
const LOCATOR_RING: [number, number] = [0.15, 0.175];
const COM_IN = new THREE.Color(BALANCE_IN);
const COM_STANCE = new THREE.Color(BALANCE_STANCE);
const COM_OUT = new THREE.Color(BALANCE_OUT);
const COM_COLOR = { sole: COM_IN, stance: COM_STANCE, out: COM_OUT } as const;
/** The sole outlines on the floor are furniture, not the answer: neutral,
 *  and dimmer for a foot that is in the air. */
const SOLE_LINE = "#c9d1e0";
const SOLE_DOWN_ALPHA = 0.55;
const SOLE_UP_ALPHA = 0.18;
/** Room for more corners than a sole's flat has (~65); the stance hull is
 *  never bigger than both soles' together. */
const OUTLINE_MAX = 160;

/** A closed line on the floor, updated in place from a wire outline. Line
 *  width is 1 px on WebGL whatever is asked, so the stance loop reads by its
 *  colour, not its weight. */
function makeLoop(color: string, opacity: number): THREE.LineLoop {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(OUTLINE_MAX * 3), 3));
  geom.setDrawRange(0, 0);
  const mat = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity,
    depthTest: false,
    depthWrite: false,
  });
  const loop = new THREE.LineLoop(geom, mat);
  loop.renderOrder = 3;
  loop.frustumCulled = false; // the draw range moves; the bounding sphere would not follow
  return loop;
}

function fillLoop(loop: THREE.LineLoop, outline: number[][]) {
  const pos = loop.geometry.getAttribute("position") as THREE.BufferAttribute;
  const n = Math.min(outline.length, OUTLINE_MAX);
  for (let i = 0; i < n; i++) pos.setXYZ(i, outline[i][0], outline[i][1], FLOOR_Z);
  pos.needsUpdate = true;
  loop.geometry.setDrawRange(0, n);
}

const v3 = (v: THREE.Vector3): V3 => [v.x, v.y, v.z];

/** The ghost of whichever body the editor is posing. `scene` is the stage's
 *  duck; another robot's meshes are fetched here (once) and nothing is drawn
 *  until they arrive. Keyed by robot so a switch remounts the body cleanly —
 *  the per-body ref arrays are the wrong length for the other one. */
export function PoseDuck({ scene }: { scene: Scene }) {
  useSyncExternalStore(subscribeAnim, animVersion, () => 0);
  const visible = animStore.visible;
  const robot = animStore.robot;
  const [fetched, setFetched] = useState<{ robot: RobotId; scene: Scene } | null>(null);
  useEffect(() => {
    if (!visible || robot === "microduck") return;
    let stale = false;
    ghostScene(robot)
      .then((s) => {
        if (!stale) setFetched({ robot, scene: s });
      })
      .catch(() => {
        // the panel already shows the /joints error for a body this lab lacks
      });
    return () => {
      stale = true;
    };
  }, [visible, robot]);
  if (!visible) return null;
  const ghost = robot === "microduck" ? scene : fetched?.robot === robot ? fetched.scene : null;
  if (!ghost) return null;
  return <PoseDuckBody key={robot} scene={ghost} />;
}

function PoseDuckBody({ scene }: { scene: Scene }) {
  // Re-render on selection changes so materials/labels stay in step even when
  // no frame is being drawn (the per-frame path below does the fast work).
  useSyncExternalStore(subscribeAnim, animVersion, () => 0);
  // Everything sized in metres is for a duck; this body's size against it
  // scales the lot. Constant for the life of this mount (keyed by robot).
  const S = animStore.sizeScale;
  const offset = previewOffset(S);

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
  const ikDrag = useRef<IkDragState | null>(null);
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
  // Same arrangement for the 🎯 handles: `begin` starts a drag of one
  // effector from a pointer position (a gizmo grab or a body click in ik
  // mode) and says whether it could.
  const ikGestures = useRef<{
    begin: (id: string, clientX: number, clientY: number) => boolean;
    over: (id: string) => void;
    out: (id: string) => void;
  } | null>(null);
  const gizmoRefs = useRef<Record<string, THREE.Mesh | null>>({});
  const gizmoMatRefs = useRef<Record<string, THREE.MeshBasicMaterial | null>>({});
  const hoverEffector = useRef<string | null>(null);
  const comRef = useRef<THREE.Group>(null);
  const comBallRef = useRef<THREE.Mesh>(null);
  const comPlumbRef = useRef<THREE.Mesh>(null);
  const comCrossRef = useRef<THREE.Group>(null);
  const comMatRefs = useRef<(THREE.MeshBasicMaterial | null)[]>([]);
  // The footprints under the crosshair: each sole's outline and the stance
  // (the hull of the grounded soles), straight from POST /pose. Built once;
  // their vertices are rewritten per-frame like everything else here.
  const loops = useMemo(
    () => ({
      left: makeLoop(SOLE_LINE, SOLE_DOWN_ALPHA),
      right: makeLoop(SOLE_LINE, SOLE_DOWN_ALPHA),
      stance: makeLoop(BALANCE_OUT, 0.9),
    }),
    []
  );
  useEffect(() => {
    return () => {
      for (const l of Object.values(loops)) {
        l.geometry.dispose();
        (l.material as THREE.Material).dispose();
      }
    };
  }, [loops]);
  const tmpP = useMemo(() => new THREE.Vector3(), []);
  const tmpQ = useMemo(() => new THREE.Quaternion(), []);
  const tmpV = useMemo(() => new THREE.Vector3(), []);
  const axisV = useMemo(() => new THREE.Vector3(), []);
  const comV = useMemo(() => new THREE.Vector3(), []);
  const viewerDir = useMemo(() => new THREE.Vector3(), []);
  const raycaster = useMemo(() => new THREE.Raycaster(), []);
  const ndc = useMemo(() => new THREE.Vector2(), []);

  // --- per-frame: apply the previewed pose + selection tinting -------------
  useFrame((state, dt) => {
    const pose = animStore.bodies;
    const meta = animStore.meta;
    // Until the first POST /pose lands, every body group still sits at the
    // group origin — a heap of parts on the floor. Stay hidden instead.
    if (rootRef.current) rootRef.current.visible = pose !== null;
    // Fast lerp: smooths the HTTP round trip without feeling laggy. The CoM
    // marker below rides the same alpha so it never leads the ghost it
    // belongs to.
    const alpha = 1 - Math.exp(-45 * Math.min(dt, 0.1));
    if (pose) {
      pose.forEach((p, b) => {
        const grp = groupRefs.current[b];
        if (!grp) return;
        tmpP.set(p[0], p[1], p[2]);
        tmpQ.set(p[4], p[5], p[6], p[3]); // wxyz → xyzw
        grp.position.lerp(tmpP, alpha);
        grp.quaternion.slerp(tmpQ, alpha);
      });
      const trunk = pose[meta?.trunkBody ?? 1];
      if (labelRef.current && trunk)
        labelRef.current.position.set(trunk[0], trunk[1], trunk[2] + LABEL_RISE * S);
      // The ⇕ handle parks at the ACTIVE control's anchor body (world-frame
      // offset, world-vertical rail) — it follows the part it moves and jumps
      // when the selection changes, which is how you see what it is armed with.
      const controlsHere = rigControlsFor(meta);
      const active =
        controlsHere.find((c) => c.id === (animStore.selectedRig?.id ?? "squat")) ?? controlsHere[0];
      const anchorBody =
        !active
          ? undefined
          : active.handle.joint === "root"
            ? meta?.trunkBody
            : meta?.joints.find((j) => j.name === active.handle.joint)?.body;
      const anchorGrp = anchorBody != null ? groupRefs.current[anchorBody] : null;
      if (handleRef.current) handleRef.current.visible = !!anchorGrp;
      if (handleRef.current && anchorGrp && active) {
        // The offsets are the server's, in THIS body's metres — not scaled.
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
    // --- the 🎯 IK handles -----------------------------------------------
    // Positions come with every /pose answer, so like `bodies` they are read
    // here per-frame and never cost a render.
    const effs = animStore.mode === "ik" && pose ? animStore.effectors : null;
    for (const [id, mesh] of Object.entries(gizmoRefs.current)) {
      if (!mesh) continue;
      const at = effs?.[id];
      mesh.visible = !!at;
      if (!at) continue;
      tmpP.set(at[0], at[1], at[2]);
      // Straight to the mark the first time: lerping out of the group
      // origin would streak the ball across the floor.
      if (mesh.position.lengthSq() === 0) mesh.position.copy(tmpP);
      else mesh.position.lerp(tmpP, alpha);
      const m = gizmoMatRefs.current[id];
      if (m) {
        const short = (animStore.ikResidual[id] ?? 0) > IK_AMBER_M * S;
        const c = short
          ? IK_AMBER
          : animStore.selectedEffector === id
            ? HANDLE_DRAG
            : hoverEffector.current === id
              ? HANDLE_HOVER
              : HANDLE_IDLE;
        m.color.copy(c);
      }
    }
    // --- the ⊕ CoM marker ---------------------------------------------
    // POST /pose delivers `balance` alongside `bodies`, so it is read here
    // per-frame for the same reason: a slider drag must not cost a render.
    const bal = animStore.balance;
    const comGrp = comRef.current;
    if (comGrp) {
      comGrp.visible = animStore.showBalance && pose !== null && bal !== null;
      if (comGrp.visible && bal) {
        const [cx, cy, cz] = bal.com;
        comV.set(cx, cy, cz);
        const ball = comBallRef.current;
        // Straight to the mark the first time it is shown: lerping out of the
        // group origin would streak the ball up off the floor.
        if (ball) {
          if (ball.position.lengthSq() === 0) ball.position.copy(comV);
          else ball.position.lerp(comV, alpha);
        }
        const at = ball?.position ?? comV;
        if (comPlumbRef.current) {
          comPlumbRef.current.position.set(at.x, at.y, at.z / 2);
          // The cylinder is a unit height along its own +Y, turned to point
          // along world Z by the rotation in the JSX; scaling it is what
          // makes the plumb line reach exactly the floor.
          comPlumbRef.current.scale.y = Math.max(at.z, 1e-4);
        }
        comCrossRef.current?.position.set(at.x, at.y, FLOOR_Z);
        const c = COM_COLOR[balanceState(bal)];
        comMatRefs.current.forEach((m) => m?.color.copy(c));
        // The footprints. A sole's outline is what its margin was measured
        // against; the stance's is the answer's own colour, so the eye
        // reads "the cross is inside THAT" without the number.
        for (const side of ["left", "right"] as const) {
          const foot = bal.feet[side];
          fillLoop(loops[side], foot.outline);
          (loops[side].material as THREE.LineBasicMaterial).opacity = foot.grounded
            ? SOLE_DOWN_ALPHA
            : SOLE_UP_ALPHA;
        }
        fillLoop(loops.stance, bal.support.outline);
        (loops.stance.material as THREE.LineBasicMaterial).color.copy(c);
      }
    }
    const sel = animStore.selected;
    const selBody =
      sel == null ? -1 : meta?.joints.find((j) => j.index === sel)?.body ?? -1;
    const rootBody = sel === ROOT_SEL ? meta?.trunkBody ?? -1 : -1;
    // A selected rig control lights up EVERY body it drives — the coupling is
    // the thing being edited, and the highlight is how you read its extent.
    // A selected effector lights up its body: the part the solver moves.
    const rigBodies = animStore.selectedRig?.bodies;
    const effBody = animStore.selectedEffector
      ? meta?.effectors.find((e) => e.id === animStore.selectedEffector)?.body ?? -1
      : -1;
    matRefs.current.forEach((m, b) => {
      if (!m) return;
      const isSel = rigBodies ? rigBodies.includes(b) : b === selBody || b === rootBody || b === effBody;
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
      const trunk = pose[meta?.trunkBody ?? 1];
      if (c && trunk) {
        // MuJoCo (x, y, z) → three.js world (x, z, -y) through the scene's
        // -90°-about-X group, plus this robot's grid offset.
        const wx = trunk[0] + offset[0];
        const wy = trunk[2];
        const wz = -(trunk[1] + offset[1]);
        camera.position.set(wx + EYE[0] * S, wy + EYE[1] * S, wz + EYE[2] * S);
        const anchor = new THREE.Vector3(wx, wy, wz);
        c.target.copy(anchor);
        camera.lookAt(c.target);
        camera.updateMatrixWorld();
        // Frame the robot in the stage the editor panel is NOT covering. The
        // target is aimed BELOW the robot (which lifts it on screen) by an
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

  /** An effector's CURRENT position (the server's last answer) in three.js
   *  world — the anchor of its drag plane. */
  const effectorWorld = (id: string): THREE.Vector3 | null => {
    const p = animStore.effectors?.[id];
    const root = rootRef.current;
    if (!p || !root) return null;
    root.updateWorldMatrix(true, false);
    return root.localToWorld(new THREE.Vector3(p[0], p[1], p[2]));
  };

  /** The pointer's ray into the scene, from client pixels. */
  const pointerRay = (clientX: number, clientY: number): THREE.Ray => {
    const rect = gl.domElement.getBoundingClientRect();
    ndc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    return raycaster.ray;
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
      // A 🎯 handle: the pointer ray meets the plane through the effector's
      // current position, facing the camera; the solver is asked for that
      // point (in the ghost group's own frame, the one /pose reports in).
      const ik = ikDrag.current;
      if (ik) {
        const at = effectorWorld(ik.id);
        const root = rootRef.current;
        if (!at || !root) return;
        const ray = pointerRay(e.clientX, e.clientY);
        const hit = dragPointOnViewPlane(v3(ray.origin), v3(ray.direction), v3(at), ik.viewDir);
        if (!hit) return;
        const step = ikDragStep(v3(at), hit, ik.lastHit, ik.grabOffset, ik.viewDir, e.shiftKey ? 0.25 : 1);
        ik.lastHit = hit;
        ik.grabOffset = step.grabOffset;
        const local = root.worldToLocal(new THREE.Vector3(step.target[0], step.target[1], step.target[2]));
        animStore.applyIkDrag?.(ik.id, [local.x, local.y, local.z]);
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
      if (!drag.current && !rigDrag.current && !ikDrag.current) return;
      drag.current = null;
      rigDrag.current = null;
      ikDrag.current = null;
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
    // The 🎯 handles, same lifecycle. `begin` is shared by a gizmo grab and
    // an ik-mode body click: both anchor the drag plane at the effector, so
    // the pointer's offset from it at the grab is kept through the gesture.
    ikGestures.current = {
      begin: (id, clientX, clientY) => {
        const at = effectorWorld(id);
        if (!at) return false;
        const viewDir = v3(camera.getWorldDirection(tmpV));
        const ray = pointerRay(clientX, clientY);
        const hit = dragPointOnViewPlane(v3(ray.origin), v3(ray.direction), v3(at), viewDir);
        ikDrag.current = {
          id,
          lastHit: hit,
          grabOffset: hit ? [at.x - hit[0], at.y - hit[1], at.z - hit[2]] : [0, 0, 0],
          viewDir,
        };
        setSelectedEffector(id);
        animStore.dragging = true;
        if (controls) controls.enabled = false;
        gl.domElement.style.cursor = "grabbing";
        return true;
      },
      over: (id) => {
        hoverEffector.current = id;
        if (!animStore.dragging) gl.domElement.style.cursor = "grab";
      },
      out: (id) => {
        if (hoverEffector.current === id) hoverEffector.current = null;
        if (!animStore.dragging) gl.domElement.style.cursor = "";
      },
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      handleGestures.current = null;
      ikGestures.current = null;
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      if (controls) controls.enabled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera, gl, controls]);

  const onBodyDown = (b: number) => (e: ThreeEvent<PointerEvent>) => {
    const meta = animStore.meta;
    // IK mode: the clicked part selects the effector whose limb it is on
    // (a shin → that foot) and drags it. A part no effector moves falls
    // through to plain joint editing.
    if (animStore.mode === "ik" && meta) {
      const eff = effectorForBody(meta, b);
      if (eff && ikGestures.current?.begin(eff.id, e.nativeEvent.clientX, e.nativeEvent.clientY)) {
        e.stopPropagation();
        return;
      }
    }
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

  const effectors = animStore.meta?.effectors ?? [];
  const selectedName = animStore.selectedRig
    ? `🎮 ${animStore.selectedRig.label}`
    : animStore.selectedEffector
      ? `🎯 ${effectors.find((x) => x.id === animStore.selectedEffector)?.label ?? animStore.selectedEffector}`
      : animStore.selected == null
        ? null
        : animStore.selected === ROOT_SEL
          ? "root pitch"
          : animStore.meta?.joints[animStore.selected]?.name ?? null;

  return (
    <group ref={rootRef} visible={false} position={[offset[0], offset[1], 0]}>
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
          nothing is selected), parks at that control's anchor on the robot,
          and wears its name. Grab it, drag down = +. */}
      <group
        ref={handleRef}
        onPointerDown={(e) => handleGestures.current?.down(e)}
        onPointerOver={(e) => handleGestures.current?.over(e)}
        onPointerOut={() => handleGestures.current?.out()}
      >
        {/* diamond, with a vertical travel rail through it (trunk +z) */}
        <mesh>
          <octahedronGeometry args={[HANDLE_R * S, 0]} />
          <meshBasicMaterial ref={(el) => void (handleMatRefs.current[0] = el)} color={HANDLE_IDLE} />
        </mesh>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[HANDLE_RAIL_R * S, HANDLE_RAIL_R * S, HANDLE_RAIL * S, 6]} />
          <meshBasicMaterial
            ref={(el) => void (handleMatRefs.current[1] = el)}
            color={HANDLE_IDLE}
            transparent
            opacity={0.6}
          />
        </mesh>
        {/* the handle wears the name of the control it is armed with */}
        <Html center zIndexRange={[10, 0]} position={[0, 0, -HANDLE_LABEL_DROP * S]} style={{ pointerEvents: "none" }}>
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

      {/* The 🎯 IK handles: one sphere per effector, at the point the solver
          moves, shown in ik mode only. Drawn through the shell (depthTest
          off) — a sole's centre is inside the foot. Raycasting does not
          honour `visible`, so outside ik mode the handlers let the event
          through to the body part behind. */}
      {effectors.map((e) => (
        <mesh
          key={e.id}
          ref={(el) => void (gizmoRefs.current[e.id] = el)}
          visible={false}
          renderOrder={4}
          onPointerDown={(ev) => {
            if (animStore.mode !== "ik") return;
            if (ikGestures.current?.begin(e.id, ev.nativeEvent.clientX, ev.nativeEvent.clientY))
              ev.stopPropagation();
          }}
          onPointerOver={(ev) => {
            if (animStore.mode !== "ik") return;
            ev.stopPropagation();
            ikGestures.current?.over(e.id);
          }}
          onPointerOut={() => ikGestures.current?.out(e.id)}
        >
          <sphereGeometry args={[IK_GIZMO_R * S, 16, 12]} />
          <meshBasicMaterial
            ref={(el) => void (gizmoMatRefs.current[e.id] = el)}
            color={HANDLE_IDLE}
            transparent
            opacity={0.85}
            depthTest={false}
          />
        </mesh>
      ))}

      {/* The ⊕ CoM marker: a ball at the centre of mass, a plumb line to the
          floor, and a crosshair where it lands, green once that point is
          inside a sole. renderOrder + depthTest off, because the CoM is
          INSIDE the shell and a marker you cannot see through the robot
          answers nothing. three.js takes groupOrder from EVERY Group it
          descends through, so the inner group carries it again. */}
      <group ref={comRef} visible={false} renderOrder={3}>
        <mesh ref={comBallRef}>
          <sphereGeometry args={[COM_BALL_R * S, 16, 12]} />
          <meshBasicMaterial
            ref={(el) => void (comMatRefs.current[0] = el)}
            color={BALANCE_OUT}
            transparent
            opacity={0.9}
            depthTest={false}
          />
        </mesh>
        <mesh ref={comPlumbRef} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[PLUMB_R * S, PLUMB_R * S, 1, 6]} />
          <meshBasicMaterial
            ref={(el) => void (comMatRefs.current[1] = el)}
            color={BALANCE_OUT}
            transparent
            opacity={0.5}
            depthTest={false}
            depthWrite={false}
          />
        </mesh>
        <group ref={comCrossRef} renderOrder={3}>
          <mesh>
            <planeGeometry args={[CROSS_ARM * 2 * S, 0.0018 * S]} />
            <meshBasicMaterial
              ref={(el) => void (comMatRefs.current[2] = el)}
              color={BALANCE_OUT}
              transparent
              opacity={0.9}
              side={THREE.DoubleSide}
              depthTest={false}
              depthWrite={false}
            />
          </mesh>
          <mesh>
            <planeGeometry args={[0.0018 * S, CROSS_ARM * 2 * S]} />
            <meshBasicMaterial
              ref={(el) => void (comMatRefs.current[3] = el)}
              color={BALANCE_OUT}
              transparent
              opacity={0.9}
              side={THREE.DoubleSide}
              depthTest={false}
              depthWrite={false}
            />
          </mesh>
          <mesh>
            <ringGeometry args={[CROSS_RING[0] * S, CROSS_RING[1] * S, 28]} />
            <meshBasicMaterial
              ref={(el) => void (comMatRefs.current[4] = el)}
              color={BALANCE_OUT}
              transparent
              opacity={0.8}
              side={THREE.DoubleSide}
              depthTest={false}
              depthWrite={false}
            />
          </mesh>
        </group>
        <primitive object={loops.left} />
        <primitive object={loops.right} />
        <primitive object={loops.stance} />
      </group>

      {/* Locator ring — the ghost is translucent and easy to lose on a busy floor. */}
      <mesh position={[0, 0, 0.005]}>
        <ringGeometry args={[LOCATOR_RING[0] * S, LOCATOR_RING[1] * S, 48]} />
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
