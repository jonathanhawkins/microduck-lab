"use client";

// The 3D stage: fetches /scene once, streams frames over WS, renders every
// duck on a shared floor. MuJoCo is Z-up; everything sim-space lives inside
// one group rotated -90° about X so three.js Y-up lighting/controls stay sane.
// Rendering is kept deliberately light (no shadow maps, merged geometry,
// capped DPR) — the first version lost the WebGL context.

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { duckRowKeys, FarmClient, fetchScene, type DuckFrame, type Scene } from "@/lib/farm";
import { assignDrag, nearestDuck, type AssignTarget } from "@/lib/assign";
import {
  cameraKeyDown,
  cameraKeyUp,
  cameraKeysClear,
  heldMotions,
  takeReset,
  takeTruckImpulse,
  truckImpulse,
} from "@/lib/camera";
import { loadJSON, saveJSON } from "@/lib/persist";
import { buildBodyGeometries, Duck } from "./Duck";
import { Hud } from "./Hud";
import { PolicyPanel } from "./PolicyPanel";
import { TeachPanel } from "./TeachPanel";
import { pushToast, Toasts } from "./Toasts";
import { AnimPanel } from "./AnimPanel";
import { PoseDuck } from "./PoseDuck";

function gridOffsets(n: number, spacing = 0.65): [number, number][] {
  const cols = Math.ceil(Math.sqrt(n));
  return Array.from({ length: n }, (_, i) => [
    (i % cols) * spacing - ((Math.min(n, cols) - 1) * spacing) / 2,
    Math.floor(i / cols) * spacing,
  ]);
}

function Ducks({ scene, client }: { scene: Scene; client: FarmClient }) {
  const bodies = useMemo(() => buildBodyGeometries(scene), [scene]);
  // Roster keyed by the STABLE stream id — a policy assign renames a duck,
  // which must update its label without remounting (and re-lerping) it.
  // (`key` is the id dedup-qualified by duckRowKeys: a roster with duplicate
  // ids — seen with legacy farm-state restores — must not collide React keys.)
  const [roster, setRoster] = useState<{ id: string; name: string; key: string }[]>([]);
  const rosterSig = useRef("");
  const duckRefs = useRef(new Map<string, React.MutableRefObject<DuckFrame | null>>());

  // Fan the single frame out into per-duck refs (no React re-render per frame).
  useFrame(() => {
    const f = client.frame;
    if (!f) return;
    const sig = f.ducks.map((d) => `${d.id}\t${d.name}`).join("\n");
    if (sig !== rosterSig.current) {
      rosterSig.current = sig;
      const keys = duckRowKeys(f.ducks);
      setRoster(f.ducks.map((d, i) => ({ id: d.id, name: d.name, key: keys[i] })));
    }
    f.ducks.forEach((d) => {
      const r = duckRefs.current.get(d.id);
      if (r) r.current = d;
    });
  });

  const offsets = gridOffsets(roster.length);
  return (
    <>
      {roster.map((d, i) => {
        let ref = duckRefs.current.get(d.id);
        if (!ref) {
          ref = { current: null };
          duckRefs.current.set(d.id, ref);
        }
        return (
          <Duck
            key={d.key}
            duckId={d.id}
            bodies={bodies}
            frameRef={ref}
            offset={offsets[i]}
            label={d.name}
          />
        );
      })}
    </>
  );
}

// Inside-the-Canvas helper: every frame, project each duck's trunk (frame body
// index 1) to client px and publish into the shared assign store. The scene
// group is rotated -90° about X, so MuJoCo (x, y, z) sits at three.js world
// (x, z, -y); each duck additionally carries its grid offset in MuJoCo XY.
// --- camera persistence ------------------------------------------------------

type Vec3Tuple = [number, number, number];
interface SavedCamera {
  p: Vec3Tuple; // camera position (three.js Y-up world)
  t: Vec3Tuple; // OrbitControls target
}

function isVec3(v: unknown): v is Vec3Tuple {
  return (
    Array.isArray(v) &&
    v.length === 3 &&
    v.every((n) => typeof n === "number" && Number.isFinite(n))
  );
}

/** Last saved camera pose, or null (first visit / bad data). */
function loadSavedCamera(): SavedCamera | null {
  const raw = loadJSON<Partial<SavedCamera> | null>("camera", null);
  return raw && isVec3(raw.p) && isVec3(raw.t) ? { p: raw.p, t: raw.t } : null;
}

// Structural slice of drei/three-stdlib OrbitControls — enough to subscribe to
// the "end" gesture event without importing its concrete class type.
interface ControlsLike {
  target: THREE.Vector3;
  addEventListener: (type: "end", cb: () => void) => void;
  removeEventListener: (type: "end", cb: () => void) => void;
  update?: () => void;
}

const HOME_CAM = { p: [1.2, 0.7, 1.4] as const, t: [0, 0.12, 0] as const };

/** Inside-the-Canvas helper: integrates held camera motions × dt every frame
 *  — smooth game-editor flow (truck/dolly/orbit rates scale with distance so
 *  the feel is constant whether you're nose-close or across the room). */
function CameraKeys() {
  const controls = useThree((s) => s.controls) as unknown as ControlsLike | null;
  const camera = useThree((s) => s.camera);
  const sph = useMemo(() => new THREE.Spherical(), []);
  const offset = useMemo(() => new THREE.Vector3(), []);
  const right = useMemo(() => new THREE.Vector3(), []);
  useFrame((_, dtRaw) => {
    // Drain the swipe impulse EVERY frame (even when unused/discarded) so a
    // burst can never pool up and teleport the view later — idle stays idle.
    const swipePx = takeTruckImpulse();
    if (!controls) return;
    if (takeReset()) {
      camera.position.set(...HOME_CAM.p);
      controls.target.set(...HOME_CAM.t);
      camera.lookAt(controls.target);
      controls.update?.();
      return;
    }
    const held = heldMotions();
    if (!held.size && swipePx === 0) return;
    const dt = Math.min(dtRaw, 0.05); // tab-stall guard: no teleport frames

    offset.copy(camera.position).sub(controls.target);
    sph.setFromVector3(offset);

    // Lateral/vertical truck moves camera AND target (the view slides).
    const truckSpeed = 0.9 * sph.radius * dt;
    if (held.has("truckLeft") || held.has("truckRight")) {
      right.set(1, 0, 0).applyQuaternion(camera.quaternion);
      right.y = 0; // keep trucking parallel to the floor
      right.normalize().multiplyScalar(
        held.has("truckRight") ? truckSpeed : -truckSpeed);
      camera.position.add(right);
      controls.target.add(right);
    }
    // Two-finger trackpad swipe → the same lateral truck, impulse-scaled.
    // Natural scrolling reports fingers-moving-LEFT as +deltaX, and that
    // gesture should slide the VIEW left (scene drifts right on screen) —
    // hence the negation. Distance scaling keeps the feel constant near and
    // far; the cap stops a momentum fling from delivering a teleport frame.
    if (swipePx !== 0) {
      const cap = 0.5 * sph.radius;
      const step = Math.max(-cap, Math.min(cap, -swipePx * 0.0015 * sph.radius));
      right.set(1, 0, 0).applyQuaternion(camera.quaternion);
      right.y = 0; // floor-parallel, like A/D
      right.normalize().multiplyScalar(step);
      camera.position.add(right);
      controls.target.add(right);
    }
    if (held.has("up") || held.has("down")) {
      const dy = (held.has("up") ? 1 : -1) * 0.6 * sph.radius * dt;
      camera.position.y += dy;
      controls.target.y = Math.max(0.02, controls.target.y + dy);
    }

    // Orbit/dolly work on the target-relative spherical frame.
    offset.copy(camera.position).sub(controls.target);
    sph.setFromVector3(offset);
    if (held.has("orbitLeft")) sph.theta += 1.7 * dt;
    if (held.has("orbitRight")) sph.theta -= 1.7 * dt;
    if (held.has("dollyIn")) sph.radius *= Math.exp(-1.5 * dt);
    if (held.has("dollyOut")) sph.radius *= Math.exp(1.5 * dt);
    sph.radius = Math.min(8, Math.max(0.25, sph.radius));
    sph.phi = Math.min(1.53, Math.max(0.1, sph.phi));
    offset.setFromSpherical(sph);
    camera.position.copy(controls.target).add(offset);

    camera.lookAt(controls.target);
    controls.update?.();
  });
  return null;
}

/** Inside-the-Canvas helper: persist the camera pose after every orbit/pan/zoom
 *  gesture (OrbitControls "end"). Restore happens via the Canvas/OrbitControls
 *  initial props, so there is no visible jump on load. */
function CameraPersistence() {
  const controls = useThree((s) => s.controls) as unknown as ControlsLike | null;
  const camera = useThree((s) => s.camera);
  useEffect(() => {
    if (!controls) return;
    const save = () =>
      saveJSON("camera", {
        p: camera.position.toArray(),
        t: controls.target.toArray(),
      });
    controls.addEventListener("end", save);
    return () => controls.removeEventListener("end", save);
  }, [controls, camera]);
  return null;
}

function AssignTargets({ client }: { client: FarmClient }) {
  const { camera, gl } = useThree();
  const v = useMemo(() => new THREE.Vector3(), []);

  useFrame(() => {
    const f = client.frame;
    if (!f) {
      assignDrag.targets = [];
      assignDrag.hoverDuck = null;
      return;
    }
    const offsets = gridOffsets(f.ducks.length);
    const rect = gl.domElement.getBoundingClientRect();
    const targets: AssignTarget[] = f.ducks.map((d, i) => {
      const t = d.bodies[1] ?? [0, 0, 0];
      v.set(t[0] + offsets[i][0], t[2], -(t[1] + offsets[i][1]));
      v.project(camera);
      return {
        id: d.id,
        x: rect.left + ((v.x + 1) / 2) * rect.width,
        y: rect.top + ((1 - v.y) / 2) * rect.height,
        visible: v.z < 1, // in front of the camera
      };
    });
    assignDrag.targets = targets;
    // Live highlight while a chip is dragged or armed (drop handlers redo the
    // lookup from the actual event position, so this is presentation-only).
    assignDrag.hoverDuck = assignDrag.mode
      ? nearestDuck(assignDrag.px, assignDrag.py)
      : null;
  });
  return null;
}

export default function Viewer() {
  const [scene, setScene] = useState<Scene | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Read once on mount (this component is ssr:false, so storage is available).
  const [savedCam] = useState(loadSavedCamera);
  const clientRef = useRef<FarmClient | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const client = new FarmClient(setConnected);
    clientRef.current = client;
    const load = () =>
      fetchScene()
        .then((s) => {
          setScene(s);
          setError(null);
        })
        .catch(() => {
          setError("duck-farm server not reachable on :8788 — start it with `uv run duck-farm …`");
          setTimeout(load, 2000);
        });
    load();

    // Drive keys. Registered on window in the *capture* phase so they work no
    // matter what has focus (canvas, OrbitControls, dev overlay) and nothing
    // downstream can swallow them first — but never while the user is TYPING,
    // and never with a browser shortcut modifier held (⌘R must reload, not
    // reset ducks).
    //
    // Only TEXT-ENTRY targets count as typing. A focused slider (the teach
    // panel's reward-weight <input type="range">) or a just-clicked pad
    // <button> must NOT eat W/A/S/D/Q/E/X/R — that stranded the keyboard
    // until the user clicked the bare canvas.
    const NON_TEXT_INPUTS = new Set([
      "range", "checkbox", "radio", "button", "submit", "reset",
      "color", "file", "image",
    ]);
    const isTyping = (t: EventTarget | null) => {
      if (!(t instanceof HTMLElement)) return false;
      if (t.isContentEditable) return true;
      if (t.tagName === "TEXTAREA" || t.tagName === "SELECT") return true;
      if (t instanceof HTMLInputElement) return !NON_TEXT_INPUTS.has(t.type);
      return false;
    };
    // Arrow keys are drive aliases — but many form controls consume arrows
    // natively (range sliders, selects, radio groups, text carets), so arrows
    // only drive from "neutral" focus (wrapper, canvas, body, plain buttons).
    const arrowsBelongToTarget = (t: EventTarget | null) =>
      t instanceof HTMLElement &&
      (t.tagName === "INPUT" ||
        t.tagName === "TEXTAREA" ||
        t.tagName === "SELECT" ||
        t.isContentEditable);
    // The keyboard flies the CAMERA (game-editor style) — the user teaches
    // ducks through RL, never by teleop, so no drive commands here. keydown
    // begins a motion, keyup ends it; CameraKeys (inside the Canvas, where
    // OrbitControls lives) integrates held motions × dt for smooth flow.
    // The exception is R: restarting the sim is the one action worth a bare
    // key, because a side-by-side comparison is only legible when every duck
    // starts its episode at the same moment. The view reset yields to Shift+R.
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (isTyping(e.target)) return;
      const arrow = e.key.startsWith("Arrow");
      if (arrow && arrowsBelongToTarget(e.target)) return; // slider keeps its arrows
      if (!e.shiftKey && e.key.toLowerCase() === "r") {
        e.preventDefault();
        if (e.repeat) return; // a leaned-on key must not machine-gun the farm
        clientRef.current?.sendReset();
        // The server resets silently (no `events` line back), so the only
        // confirmation the user gets is this local toast.
        pushToast("↺ sim restarted — every duck from zero");
        return;
      }
      if (cameraKeyDown(e.key, e.shiftKey)) e.preventDefault(); // no scroll/find-as-you-type
    };
    const onKeyUp = (e: KeyboardEvent) => cameraKeyUp(e.key);
    const onBlur = () => cameraKeysClear(); // no motion stuck across focus loss
    window.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("keyup", onKeyUp, true);
    window.addEventListener("blur", onBlur);

    // Two-finger trackpad swipes arrive as wheel events: horizontal-dominant
    // ones truck the camera (same motion as A/D). Vertical-dominant and
    // ctrlKey (pinch) events pass through untouched so OrbitControls keeps
    // zooming. Capture-phase + passive:false, because preventDefault must
    // beat the browser's two-finger back/forward navigation and
    // stopPropagation must keep OrbitControls from also treating the event
    // as zoom. Only STAGE events count: the side panels (policies, teach,
    // HUD) scroll with two fingers and must keep doing so, so anything whose
    // target isn't the wrapper itself or the <canvas> is ignored.
    // GESTURE-AXIS LOCK: a two-finger swipe is one gesture that should do ONE
    // thing — per-event dominance flipped slide/zoom mid-swipe on any slightly
    // diagonal motion (both at once, felt awful). The axis is decided once per
    // gesture, from the first ~6px of accumulated motion, and held until a
    // pause in the wheel stream (a momentum fling keeps events flowing well
    // under the gap, so the lock survives the coast).
    const GESTURE_GAP_MS = 180;
    const LOCK_AFTER_PX = 6;
    let gestureAxis: "x" | "y" | null = null;
    let undecidedX = 0, undecidedY = 0;
    let lastWheelAt = 0;
    const onWheel = (e: WheelEvent) => {
      if (e.ctrlKey) return; // pinch-zoom stays OrbitControls'
      const root = rootRef.current;
      const t = e.target;
      if (!root || !(t instanceof HTMLElement) || !root.contains(t)) return;
      if (t !== root && t.tagName !== "CANVAS") return; // panel UI scrolls natively

      const now = performance.now();
      if (now - lastWheelAt > GESTURE_GAP_MS) {
        gestureAxis = null; // stream paused → next events are a new gesture
        undecidedX = undecidedY = 0;
      }
      lastWheelAt = now;

      // Rare line-mode mice report lines, not pixels — normalize roughly.
      const dx = e.deltaMode ? e.deltaX * 16 : e.deltaX;
      const dy = e.deltaMode ? e.deltaY * 16 : e.deltaY;
      if (gestureAxis === null) {
        undecidedX += Math.abs(dx);
        undecidedY += Math.abs(dy);
        if (undecidedX + undecidedY < LOCK_AFTER_PX) {
          e.preventDefault(); // hold the ambiguous first pixels back from BOTH
          e.stopPropagation();
          return;
        }
        gestureAxis = undecidedX > undecidedY ? "x" : "y";
      }
      if (gestureAxis === "y") return; // whole gesture = zoom (OrbitControls)
      truckImpulse(dx); // whole gesture = slide; zoom never sees it
      e.preventDefault();
      e.stopPropagation();
    };
    window.addEventListener("wheel", onWheel, { capture: true, passive: false });

    // Pull keyboard focus into the page up front: embedded panes and some
    // browsers won't route key events to the document until something in it
    // has been focused.
    rootRef.current?.focus({ preventScroll: true });

    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("keyup", onKeyUp, true);
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("wheel", onWheel, true);
      client.close();
    };
  }, []);

  // Clicking the stage re-grabs focus for the wrapper; clicks on real
  // interactive elements (pad buttons, future chat input) keep their focus.
  const refocus = (e: React.PointerEvent<HTMLDivElement>) => {
    const t = e.target as HTMLElement | null;
    if (t?.closest("button, input, textarea, select, a, [contenteditable]")) return;
    rootRef.current?.focus({ preventScroll: true });
  };

  return (
    <div
      ref={rootRef}
      tabIndex={0}
      onPointerDown={refocus}
      style={{ position: "fixed", inset: 0, background: "#101216", outline: "none" }}
    >
      <Canvas
        dpr={[1, 1.5]}
        gl={{ antialias: true, powerPreference: "high-performance" }}
        camera={{
          position: savedCam?.p ?? [1.2, 0.7, 1.4],
          fov: 40,
          near: 0.01,
          far: 50,
        }}
      >
        <color attach="background" args={["#101216"]} />
        <fog attach="fog" args={["#101216", 4, 10]} />
        <hemisphereLight intensity={0.65} groundColor="#2a2c33" color="#dfe6f0" />
        <directionalLight position={[2.5, 4, 2]} intensity={1.9} />
        <directionalLight position={[-2, 2.5, -1.5]} intensity={0.4} color="#8fa3c7" />

        {/* floor + grid live in three's Y-up world */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.002, 0]}>
          <circleGeometry args={[9, 64]} />
          <meshStandardMaterial color="#181b21" roughness={0.95} />
        </mesh>
        <gridHelper args={[18, 72, "#3a4150", "#262a33"]} position={[0, -0.001, 0]} />

        {/* MuJoCo Z-up world */}
        <group rotation={[-Math.PI / 2, 0, 0]}>
          {scene && clientRef.current && (
            <Ducks scene={scene} client={clientRef.current} />
          )}
          {/* 🎬 editor's ghost duck — server-side FK only, no env, no stream */}
          {scene && <PoseDuck scene={scene} />}
        </group>
        {clientRef.current && <AssignTargets client={clientRef.current} />}

        <OrbitControls
          makeDefault
          target={savedCam?.t ?? [0, 0.12, 0]}
          maxPolarAngle={Math.PI / 2 - 0.02}
          minDistance={0.3}
          maxDistance={7}
          // Trackpad two-finger vertical (and wheel) zoom at the default 1.0
          // crossed half the zoom range in one small swipe — tame it.
          zoomSpeed={0.4}
        />
        <CameraPersistence />
        <CameraKeys />
      </Canvas>
      <Hud clientRef={clientRef} connected={connected} error={error} />
      <PolicyPanel clientRef={clientRef} />
      <TeachPanel clientRef={clientRef} />
      <AnimPanel />
      <Toasts clientRef={clientRef} />
    </div>
  );
}
