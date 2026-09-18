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
import {
  duckRowKeys,
  LabClient,
  fetchScene,
  robotsInFrame,
  type DuckFrame,
  type RobotId,
  type Scene,
} from "@/lib/lab";
import { assignDrag, nearestDuck, type AssignTarget } from "@/lib/assign";
import {
  cameraKeyDown,
  cameraKeyUp,
  cameraKeysClear,
} from "@/lib/camera";
import { loadJSON, saveJSON } from "@/lib/persist";
import { getCapture } from "@/lib/record";
import { getSelectedDuck, setSelectedDuck } from "@/lib/select";
import { modalIsOpen } from "@/lib/ui";
import { buildBodyGeometries, Duck } from "./Duck";
import { robotLook } from "@/lib/robots";
import CameraKeys, { type ControlsLike } from "./CameraKeys";
import { useTruckSwipe } from "./useTruckSwipe";
import { Hud } from "./Hud";
import { PolicyPanel } from "./PolicyPanel";
import { TeachPanel } from "./TeachPanel";
import { pushToast, Toasts } from "./Toasts";
import { AnimPanel } from "./AnimPanel";
import { RecordPanel } from "./RecordPanel";
import { CaptureCanvas, Snapshotter } from "./Capture";
import { PoseDuck } from "./PoseDuck";

/** FALLBACK layout only — one duck-sized pitch for the whole roster. The
 *  server sends each slot its own offset now (DuckFrame.offset), because the
 *  pitch belongs to the robot: this 0.65 m is the duck's, and six 1.3 m G1
 *  helpers laid out on it stood inside each other. Kept for a server that
 *  predates the field. */
function gridOffsets(n: number, spacing = 0.65): [number, number][] {
  const cols = Math.ceil(Math.sqrt(n));
  return Array.from({ length: n }, (_, i) => [
    (i % cols) * spacing - ((Math.min(n, cols) - 1) * spacing) / 2,
    Math.floor(i / cols) * spacing,
  ]);
}

/** Slot positions for one frame: the server's per-robot layout where it sends
 *  one, the duck-pitched grid where it does not. */
function frameOffsets(ducks: Pick<DuckFrame, "offset">[]): [number, number][] {
  const grid = gridOffsets(ducks.length);
  return ducks.map((d, i) => d.offset ?? grid[i]);
}

function Ducks({
  scene,
  scenes,
  client,
}: {
  scene: Scene;
  /** Mesh sets by robot id. The duck's is always present; another body's
   *  arrives once a roster row says it is that robot (GET /scene?robot=). */
  scenes: Partial<Record<RobotId, Scene>>;
  client: LabClient;
}) {
  const bodies = useMemo(() => buildBodyGeometries(scene), [scene]);
  // One geometry set per robot on the stage, built once — not per duck. The
  // duck's is built here as well as above: `bodies` still backs any row whose
  // mesh set has not arrived yet (and every server that predates robot
  // selection, where no row carries one).
  const bodiesByRobot = useMemo(() => {
    const out: Partial<Record<RobotId, ReturnType<typeof buildBodyGeometries>>> = {};
    for (const [id, s] of Object.entries(scenes)) {
      if (!s) continue;
      try {
        // The body's own look: "duck" merges and vertex-colours as it always
        // has, "g1" takes its material table, and every OTHER body is
        // "generic" — welded, smoothed, painted from the scene dump's own
        // rgba. That is the line that puts a MARS or a Menagerie model on
        // the stage without a component of its own (lib/robots.robotLook).
        const look = robotLook(id);
        out[id as RobotId] =
          look === "duck" ? buildBodyGeometries(s) : buildBodyGeometries(s, null, { look });
      } catch (e) {
        // A mesh set that fails to build must not take the stage with it:
        // this runs during render, so an uncaught throw here would unmount
        // the whole Canvas — every duck AND the floor — over one robot.
        console.error(`[viewer] ${id} meshes failed to build`, e);
      }
    }
    return out;
  }, [scenes]);
  // Roster keyed by the STABLE stream id — a policy assign renames a duck,
  // which must update its label without remounting (and re-lerping) it.
  // (`key` is the id dedup-qualified by duckRowKeys: a roster with duplicate
  // ids — seen with legacy lab-state restores — must not collide React keys.)
  const [roster, setRoster] = useState<
    {
      id: string;
      name: string;
      key: string;
      robot: RobotId;
      offset: [number, number];
    }[]
  >([]);
  const rosterSig = useRef("");
  const duckRefs = useRef(new Map<string, React.MutableRefObject<DuckFrame | null>>());

  // Fan the single frame out into per-duck refs (no React re-render per frame).
  useFrame(() => {
    const f = client.frame;
    if (!f) return;
    // The robot is part of the signature: a slot that changes body must
    // re-render with the other mesh set, and the name alone need not change.
    const sig = f.ducks.map((d) => `${d.id}\t${d.name}\t${d.robot ?? ""}`).join("\n");
    if (sig !== rosterSig.current) {
      rosterSig.current = sig;
      const keys = duckRowKeys(f.ducks);
      // Offsets ride in the roster state, not a per-frame recompute: the
      // layout can only change when the roster does, and every input to it
      // (the row count and every row's robot) is in the signature above.
      const offs = frameOffsets(f.ducks);
      setRoster(
        f.ducks.map((d, i) => ({
          id: d.id,
          name: d.name,
          key: keys[i],
          robot: (d.robot as RobotId) || "microduck",
          offset: offs[i],
        })),
      );
      // A removed duck must not stay "selected" — the Delete key would then
      // fire remove_duck at a ghost id forever.
      const sel = getSelectedDuck();
      if (sel && !f.ducks.some((d) => d.id === sel)) setSelectedDuck(null);
    }
    f.ducks.forEach((d) => {
      const r = duckRefs.current.get(d.id);
      if (r) r.current = d;
    });
  });

  return (
    <>
      {roster.map((d) => {
        let ref = duckRefs.current.get(d.id);
        if (!ref) {
          ref = { current: null };
          duckRefs.current.set(d.id, ref);
        }
        // A row whose mesh set has NOT arrived yet draws nothing at all.
        // Falling back to the duck's meshes drew a G1 as a scatter of duck
        // parts at humanoid joint positions — the poses are streamed in that
        // robot's own body order, so the wrong set is not a rough
        // approximation, it is debris. The duck itself always has its scene
        // (it is fetched at mount), so only another body can wait.
        const geo = d.robot === "microduck" ? bodies : bodiesByRobot[d.robot];
        if (!geo) return null;
        return (
          <Duck
            key={`${d.key}:${d.robot}`}
            duckId={d.id}
            bodies={geo}
            frameRef={ref}
            offset={d.offset}
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

const HOME_CAM = { p: [1.2, 0.7, 1.4] as const, t: [0, 0.12, 0] as const };

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

// 🎥 follow-cam shot parameters: ¾ front view (±az off the duck's heading),
// distance/height sized so a 25 cm duck fills about half the frame at the
// stage's 40° fov, plus a very slow azimuth drift so long takes stay alive.
const SHOT = { az: 0.61, dist: 0.78, height: 0.34, drift: 0.05 };

const wrapAngle = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));

/** Inside-the-Canvas helper for the 🎥 record flow (the canvas itself is
 *  registered and pumped by CaptureCanvas, Capture.tsx): while a capture is
 *  framing/recording flies the camera to a ¾ front view of the target duck
 *  and keeps it centered.
 *  OrbitControls is paused for the duration (CameraKeys takes a `paused`
 *  callback and drops held motions while it is true), and
 *  the camera simply stays where the take ended. The azimuth is chosen ONCE
 *  per take — the duck's heading rotated ±SHOT.az toward whichever side the
 *  camera already sits — not tracked per frame: a backflipping duck's heading
 *  whips 180° mid-roll and would slingshot the camera around it. */
function RecordCamera({ client }: { client: LabClient }) {
  const controls = useThree((s) => s.controls) as unknown as ControlsLike | null;
  const camera = useThree((s) => s.camera);
  // Never leave the user without controls (unmount mid-take).
  useEffect(() => {
    return () => {
      if (controls) controls.enabled = true;
    };
  }, [controls]);

  const shot = useRef<{ epoch: number; az: number; t0: number } | null>(null);
  const paused = useRef(false);
  const aim = useMemo(() => new THREE.Vector3(), []);
  const desired = useMemo(() => new THREE.Vector3(), []);
  const fwd = useMemo(() => new THREE.Vector3(), []);
  const q = useMemo(() => new THREE.Quaternion(), []);

  useFrame((st, dtRaw) => {
    const cap = getCapture();
    const active = cap.phase === "framing" || cap.phase === "recording";
    if (!active) {
      if (paused.current && controls) controls.enabled = true;
      paused.current = false;
      shot.current = null;
      return;
    }
    if (controls && !paused.current) {
      controls.enabled = false;
      paused.current = true;
    }
    const f = client.frame;
    const idx = f ? f.ducks.findIndex((d) => d.id === cap.duckId) : -1;
    const trunk = idx >= 0 ? f!.ducks[idx].bodies[1] : undefined;
    if (!f || !trunk) return; // duck vanished mid-take — hold the last shot
    const off = frameOffsets(f.ducks)[idx];
    // MuJoCo (x, y, z) → three world (x, z, -y), plus the duck's grid offset.
    aim.set(trunk[0] + off[0], trunk[2] + 0.02, -(trunk[1] + off[1]));

    const now = st.clock.elapsedTime;
    if (!shot.current || shot.current.epoch !== cap.epoch) {
      // Duck heading on the floor, as a three-world azimuth (atan2(x, z)).
      q.set(trunk[4], trunk[5], trunk[6], trunk[3]); // wxyz → xyzw
      fwd.set(1, 0, 0).applyQuaternion(q); // trunk +x = forward, MuJoCo frame
      const headingAz = Math.atan2(fwd.x, -fwd.y);
      const camAz = Math.atan2(
        camera.position.x - aim.x,
        camera.position.z - aim.z
      );
      // ¾ view on whichever side needs the shorter glide.
      const side =
        Math.abs(wrapAngle(headingAz + SHOT.az - camAz)) <=
        Math.abs(wrapAngle(headingAz - SHOT.az - camAz))
          ? 1
          : -1;
      shot.current = {
        epoch: cap.epoch,
        az: wrapAngle(headingAz + side * SHOT.az),
        t0: now,
      };
    }
    const az = shot.current.az + SHOT.drift * (now - shot.current.t0);
    desired.set(
      aim.x + Math.sin(az) * SHOT.dist,
      aim.y + SHOT.height,
      aim.z + Math.cos(az) * SHOT.dist
    );

    const dt = Math.min(dtRaw, 0.1); // tab-stall guard, like CameraKeys
    camera.position.lerp(desired, 1 - Math.exp(-3.5 * dt));
    if (controls) {
      controls.target.lerp(aim, 1 - Math.exp(-6 * dt));
      camera.lookAt(controls.target);
      controls.update?.();
    } else {
      camera.lookAt(aim);
    }
  });
  return null;
}

function AssignTargets({ client }: { client: LabClient }) {
  const { camera, gl } = useThree();
  const v = useMemo(() => new THREE.Vector3(), []);

  useFrame(() => {
    const f = client.frame;
    if (!f) {
      assignDrag.targets = [];
      assignDrag.hoverDuck = null;
      return;
    }
    const offsets = frameOffsets(f.ducks);
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
  // Mesh sets for every robot the roster is currently showing.
  const [scenes, setScenes] = useState<Partial<Record<RobotId, Scene>>>({});
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Read once on mount (this component is ssr:false, so storage is available).
  const [savedCam] = useState(loadSavedCamera);
  const clientRef = useRef<LabClient | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);
  // Two-finger horizontal swipe → the same lateral truck as A/D.
  useTruckSwipe(rootRef);

  // Load a robot's mesh set the first time a roster row says it is that
  // robot. Polled off the client's own frame (1 Hz) rather than subscribed:
  // the frame is a mutable ref read per-frame inside the Canvas, and a
  // React subscription to it would re-render the whole stage at 50 Hz.
  //
  // Mounted ONCE, with its bookkeeping in refs. Keyed on `scenes` instead,
  // every arriving scene tore the effect down and re-ran it — and a G1 fetch
  // in flight at that moment (21 MB, ~2.5 s) resolved into a dead closure and
  // was thrown away. The stage then drew a G1 with the duck's meshes.
  const loadedScenes = useRef(new Set<string>());
  const pendingScenes = useRef(new Set<string>());
  const sceneFailures = useRef(new Map<string, number>());
  useEffect(() => {
    let stopped = false;
    const tick = () => {
      const f = clientRef.current?.frame;
      if (!f) return;
      for (const robot of robotsInFrame(f.ducks)) {
        // A server without that robot's assets answers 404 every time; retry
        // a few times (a lab restart mid-session is the case worth covering)
        // and then stop, rather than polling a 404 once a second forever.
        if (
          loadedScenes.current.has(robot) ||
          pendingScenes.current.has(robot) ||
          (sceneFailures.current.get(robot) ?? 0) >= 5
        ) {
          continue;
        }
        pendingScenes.current.add(robot);
        fetchScene(robot)
          .then((s) => {
            loadedScenes.current.add(robot);
            if (!stopped) setScenes((prev) => ({ ...prev, [robot]: s }));
          })
          .catch(() => {
            sceneFailures.current.set(robot, (sceneFailures.current.get(robot) ?? 0) + 1);
            pendingScenes.current.delete(robot);   // retry on the next tick
          });
      }
    };
    const id = setInterval(tick, 1000);
    tick();
    return () => {
      stopped = true;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    const client = new LabClient(setConnected);
    clientRef.current = client;
    const load = () =>
      fetchScene()
        .then((s) => {
          setScene(s);
          setScenes((prev) => ({ ...prev, microduck: s }));
          setError(null);
        })
        .catch(() => {
          setError("duck-lab server not reachable on :8788 — start it with `uv run duck-lab …`");
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
      // A dialog owns the keyboard while it is open. isTyping() is not enough:
      // a dialog that focuses a button (both of ours do) would still let
      // Backspace remove the selected duck and `r` reset every episode behind
      // the overlay. The dialog cannot stop us from its own listener — two
      // window/capture listeners cannot suppress each other — so the scene
      // stands down voluntarily.
      if (modalIsOpen()) return;
      if (isTyping(e.target)) return;
      const arrow = e.key.startsWith("Arrow");
      if (arrow && arrowsBelongToTarget(e.target)) return; // slider keeps its arrows
      // Delete/Backspace removes the SELECTED duck (click a duck to select).
      // Backspace must be claimed too — un-prevented it can navigate back.
      if (e.key === "Delete" || e.key === "Backspace") {
        const sel = getSelectedDuck();
        if (!sel) return;
        e.preventDefault();
        if (e.repeat) return;
        const training = clientRef.current?.frame?.training;
        // Mirror the HUD-row rules: the server refuses these anyway, but a
        // keypress that silently does nothing reads as broken.
        if (sel === "trainee" && (training?.status === "training" || training?.restarting)) {
          pushToast("🎓 the trainee can't be removed while training");
          return;
        }
        if (sel.startsWith("helper") && training?.restarting) {
          pushToast("⏳ trainer restarting — try removing the helper again in a moment");
          return;
        }
        clientRef.current?.sendRemoveDuck(sel);
        setSelectedDuck(null);
        return;
      }
      if (e.key === "Escape") {
        setSelectedDuck(null);
        return;
      }
      if (!e.shiftKey && e.key.toLowerCase() === "r") {
        e.preventDefault();
        if (e.repeat) return; // a leaned-on key must not machine-gun the lab
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

    // Pull keyboard focus into the page up front: embedded panes and some
    // browsers won't route key events to the document until something in it
    // has been focused.
    rootRef.current?.focus({ preventScroll: true });

    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("keyup", onKeyUp, true);
      window.removeEventListener("blur", onBlur);
      client.close();
    };
  }, []);

  // Clicking the stage re-grabs focus for the wrapper; clicks on real
  // interactive elements (pad buttons, future chat input) keep their focus.
  // The pointer-down is also remembered for the click-to-select handler below:
  // position (to tell a click from an orbit drag) and whether an assign
  // gesture was live (armed chips assign on pointerdown, so by click time
  // assignDrag.mode is already cleared — snapshot it here instead).
  const downAt = useRef<{ x: number; y: number; assigning: boolean }>({
    x: 0,
    y: 0,
    assigning: false,
  });
  const refocus = (e: React.PointerEvent<HTMLDivElement>) => {
    downAt.current = { x: e.clientX, y: e.clientY, assigning: assignDrag.mode !== null };
    const t = e.target as HTMLElement | null;
    if (t?.closest("button, input, textarea, select, a, [contenteditable]")) return;
    rootRef.current?.focus({ preventScroll: true });
  };

  // Click a duck to select it (nearest projected duck within the same screen
  // radius the assign flow uses); click empty floor to deselect. Orbit drags
  // and armed-assign clicks don't count.
  const selectAt = (e: React.MouseEvent<HTMLDivElement>) => {
    const t = e.target as HTMLElement | null;
    if (t?.tagName !== "CANVAS") return; // panel/button clicks aren't stage clicks
    if (downAt.current.assigning) return;
    if (Math.hypot(e.clientX - downAt.current.x, e.clientY - downAt.current.y) > 5) return;
    setSelectedDuck(nearestDuck(e.clientX, e.clientY));
  };

  return (
    <div
      ref={rootRef}
      tabIndex={0}
      onPointerDown={refocus}
      onClick={selectAt}
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
            <Ducks scene={scene} scenes={scenes} client={clientRef.current} />
          )}
          {/* 🎬 editor's ghost duck — server-side FK only, no env, no stream */}
          {scene && <PoseDuck scene={scene} />}
        </group>
        {clientRef.current && <AssignTargets client={clientRef.current} />}
        {clientRef.current && <RecordCamera client={clientRef.current} />}

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
        <CameraKeys
          home={HOME_CAM}
          minDist={0.25}
          maxDist={8}
          paused={() => {
            const ph = getCapture().phase;
            return ph === "framing" || ph === "recording";
          }}
        />
        <CaptureCanvas />
        <Snapshotter />
      </Canvas>
      <Hud clientRef={clientRef} connected={connected} error={error} />
      <RecordPanel clientRef={clientRef} />
      <PolicyPanel clientRef={clientRef} />
      <TeachPanel clientRef={clientRef} />
      <AnimPanel />
      <Toasts clientRef={clientRef} />
    </div>
  );
}
