"use client";

// /sim — the WORLD page: one room, many ducks, and what each duck SENSES.
// Reuses the lab page's stage (scene meshes, Duck renderer, selection store)
// against the lab's world mode (/ws/sim, world_server.py). Keys: R restarts
// the world, P toggles drive mode (WASD / arrows steer every duck), T toggles
// the sensor overlay (a duck's ToF cone, a wheeled body's 360° scan — the
// button names whichever the selected body has), L the ducks’ name labels,
// M the occupancy map, Shift+M
// mutes their voices, 1–9 select a duck, Esc deselects.
//
// WASD/QE are shared between two consumers, split by drive mode: with drive
// OFF they fly the camera (the lab page's model, same CameraKeys component),
// with drive ON they steer the ducks. That is why edit is Shift+E — plain E
// is the camera's vertical truck.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { fetchScene, type DuckFrame, type Scene } from "@/lib/lab";
import { assignDrag, nearestDuck, type AssignTarget } from "@/lib/assign";
import { getSelectedDuck, setSelectedDuck } from "@/lib/select";
import { loadJSON, saveJSON } from "@/lib/persist";
import { Voices } from "@/lib/quack";
import { duckAudio } from "@/lib/quackaudio";
import { duckMouths } from "@/lib/mouth";
import { setDuckLabels } from "@/lib/ui";
import {
  cameraKeyDown,
  cameraKeyUp,
  cameraKeysClear,
} from "@/lib/camera";
import {
  depthColor,
  fetchRing,
  fetchScenarios,
  menuBrains,
  deleteScenario,
  fetchWorld,
  frameEvents,
  loadWorld,
  saveRecording,
  SimClient,
  capturePose,
  fetchG1Scene,
  detectionBox,
  detectionRay,
  headCameraPose,
  tofZonePoints,
  CAM_FOV_DEG,
  OVERLAY_LAYER,
  TOF_PRESETS,
  SIM_SPEEDS,
  SIM_SPEED_DEFAULT,
  speedLabel,
  speedShortfall,
  stepSpeed,
  teamColor,
  teamSwatch,
  type FrameEvent,
  type SimFrame,
  type ScenarioListing,
  type SimDuck,
  type TofPreset,
  type WorldInfo,
} from "@/lib/sim";
import { rangeChannel, sensorOverlayLabel } from "@/lib/lidar";
import { camAspect, renderInset } from "@/lib/inset";
import { buildBodyGeometries, Duck, type BodyGeometry } from "./Duck";
import { RobotBody } from "./SimStage";
import { ArmBlock, GripperBlock, LidarOverlay, LidarPlot } from "./SimLidar";
import { robotLook } from "@/lib/robots";
import CameraKeys from "./CameraKeys";
import { useTruckSwipe } from "./useTruckSwipe";
import { CaptureCanvas, Snapshotter } from "./Capture";
import { SimRecord } from "./SimRecord";
import { BrainPanel } from "./SimBrain";
import { PANEL, PanelToggle } from "./Panel";
import { StateGraphPanel } from "./SimGraph";
import { consumeDragged, HANDLE, useDrag } from "./useDrag";
import { applyFloorClick, emptyDraft, SimEditor, type EditorState } from "./SimEditor";
import { Dynamics, StageEnvironment, Statics } from "./SimStage";

const BG = "#101216";
// Panel geometry: everything overlaid on the room is inset PAD from the edge,
// and the inspector hangs GAP below the measured bottom of the top bar.
const PAD = 10;
const GAP = 8;
const INSPECTOR_W = 242;   // border box: the old 220 content width + PANEL padding and border
const TOP_BAR_MIN_BOTTOM = 46;
// The head-camera inset is as wide as the inspector (border-box), at the
// camera's aspect; the inspector reserves that much room so the pair fits.
// The head-camera inset lives top-left, under the top bar, where nothing
// else sits over it — bottom-right it was under the event log and the
// scrub bar. Wider than the inspector: it is the thing you look AT.
const CAM_W = 280;
// ...and when that inset is minimized, only its bar hangs off the edge.

const BTN_BORDER = "#2b313b";
// The border is spelled out rather than using the `border` shorthand: half a
// dozen buttons below override borderColor to show a toggle is on, and React
// warns (and can mis-render) when a longhand is dropped while the shorthand
// that also sets it stays put. The pair in the Timeline hit that every time
// you paused — the two branches are the same <button>, so going live→scrub
// REMOVED borderColor from an element whose `border` still set one.
const BTN: React.CSSProperties = {
  background: "#1f242c",
  color: "#e9edf1",
  borderWidth: 1,
  borderStyle: "solid",
  borderColor: BTN_BORDER,
  borderRadius: 4,
  padding: "3px 8px",
  fontFamily: "inherit",
  fontSize: 12,
  cursor: "pointer",
};

type Held = Set<string>;

/** A number in a fixed-width slot (ch units; the panel font is monospace),
 *  right-aligned so the digits line up as the value changes width. */
function Slot({ w, left, children }: { w: number; left?: boolean; children: React.ReactNode }) {
  return <span style={{ display: "inline-block", minWidth: `${w}ch`, textAlign: left ? "left" : "right" }}>{children}</span>;
}
/** Milliseconds to two places in a 5-char field ("0.65" → " 0.65"), so the
 *  perf triple keeps its shape when one term crosses 10 ms. The line is
 *  rendered with whiteSpace: nowrap, which collapses runs of spaces but keeps
 *  a single leading one. */
const ms = (v: number) => v.toFixed(2).padStart(5, "\u00a0");

/**
 * A sensor's age as a reading, not a number. The raw age is quantised to the
 * 40 ms frame tick and the ToF runs at 15 Hz, so "age 0 / 20 / 40 ms" cycled
 * every frame — flicker carrying no information. Inside the fresh window the
 * word is enough; beyond it the age matters, shown in 50 ms steps so it
 * changes when the situation does. The exact figure stays in the title.
 */
function freshness(ageS: number, freshMs: number): { text: string; color: string; title: string } {
  const ms = Math.round(ageS * 1000);
  if (ms <= freshMs) return { text: "fresh", color: "#9aa5b1", title: `last frame ${ms} ms ago` };
  return { text: `stale · ${Math.round(ms / 50) * 50} ms`, color: "#f2b632", title: `last frame ${ms} ms ago` };
}
const TOF_FRESH_MS = 150;    // two ToF periods at 15 Hz, with a frame of slack
const DET_FRESH_MS = 250;    // two detector periods at 10 Hz, likewise

/** Held drive keys → one twist command [vx, vy, wz]. */
function twistFromKeys(h: Held): [number, number, number] {
  const vx = (h.has("w") || h.has("arrowup") ? 0.3 : 0) + (h.has("s") || h.has("arrowdown") ? -0.2 : 0);
  const vy = (h.has("q") ? 0.15 : 0) + (h.has("e") ? -0.15 : 0);
  const wz = (h.has("a") || h.has("arrowleft") ? 0.8 : 0) + (h.has("d") || h.has("arrowright") ? -0.8 : 0);
  return [vx, vy, wz];
}
const DRIVE_KEYS = new Set(["w", "a", "s", "d", "q", "e", "arrowup", "arrowdown", "arrowleft", "arrowright"]);

// Shift+R flies back here — the Canvas's own opening pose.
const HOME_CAM = { p: [2.2, 1.6, 2.4] as const, t: [0, 0.12, 0] as const };
// Must be a stable reference: this component re-renders every 250 ms on the
// status tick, and a fresh [0, 0.12, 0] literal each time would let r3f
// re-apply the prop and snap the target back while you are trucking.
const ORBIT_TARGET: [number, number, number] = [0, 0.12, 0];
// A room is bigger than the lab floor, so the dolly gets more room than the
// lab page's 0.25..8 — these match the OrbitControls clamp below.
const CAM_MIN_DIST = 0.3;
const CAM_MAX_DIST = 12;

/** Every duck of the frame, rendered by the lab page's Duck component.
 *
 *  Geometry is built once PER COLORWAY, not per duck: a team's colours are
 *  baked into the vertex-color channel, and there are at most four teams, so
 *  a 3v3 pitch costs two geometry sets and a room costs one. (Per duck is
 *  what lost the WebGL context at eight ducks before the bodies were merged.) */
function SimDucks({
  scene,
  client,
  robotScenes,
}: {
  scene: Scene;
  client: SimClient;
  /** Mesh sets for the NON-duck bodies in the room, by robot id. A room can
   *  hold a MARS beside its ducks (world/scenario.Duck.robot), and each is
   *  drawn from its own `GET /scene?robot=<id>` — one scene per robot, not
   *  one per entry. */
  robotScenes: Record<string, Scene>;
}) {
  const plain = useMemo(() => buildBodyGeometries(scene), [scene]);
  const byTeam = useRef(new Map<string, BodyGeometry[]>());
  const [roster, setRoster] = useState<
    { id: string; name: string; team: string | null; robot: string }[]
  >([]);
  const sig = useRef("");
  const refs = useRef(new Map<string, React.MutableRefObject<DuckFrame | null>>());
  useEffect(() => {
    byTeam.current.clear();     // a new scene: the cached geometry is stale
  }, [scene]);
  const bodiesFor = (team: string | null) => {
    if (!team) return plain;
    let g = byTeam.current.get(team);
    if (!g) {
      g = buildBodyGeometries(scene, team);
      byTeam.current.set(team, g);
    }
    return g;
  };
  useFrame(() => {
    const f = client.frame;
    if (!f) return;
    const s = f.ducks
      .map((d) => `${d.id}\t${d.name}\t${d.team ?? ""}\t${d.robot ?? ""}`)
      .join("\n");
    if (s !== sig.current) {
      sig.current = s;
      setRoster(
        f.ducks.map((d) => ({
          id: d.id,
          name: d.name,
          team: d.team ?? null,
          robot: d.robot || "microduck",
        }))
      );
      const sel = getSelectedDuck();
      if (sel && !f.ducks.some((d) => d.id === sel)) setSelectedDuck(null);
    }
    f.ducks.forEach((d) => {
      const r = refs.current.get(d.id);
      if (r) r.current = d as unknown as DuckFrame;
    });
  });
  return (
    <>
      {roster.map((d) => {
        if (d.robot !== "microduck") {
          // Another body in the room. Its poses are already in the frame's
          // `ducks` list in ITS scene's order, so the only thing missing is
          // the mesh set — and until that arrives (a fetch away) it draws
          // nothing rather than borrowing the duck's meshes, which would put
          // a duck's parts on a MARS's joints.
          const rs = robotScenes[d.robot];
          // The body's own look, forwarded whole. "duck" cannot happen in
          // this branch; it collapses to "generic" only to satisfy the type.
          const look = robotLook(d.robot);
          return rs ? (
            <RobotBody
              key={d.id}
              id={d.id}
              scene={rs}
              client={client}
              from="ducks"
              // A look this page forgets to forward is how MARS stayed
              // "generic" and near-invisible after its own table existed.
              look={look === "duck" ? "generic" : look}
            />
          ) : null;
        }
        let ref = refs.current.get(d.id);
        if (!ref) {
          ref = { current: null };
          refs.current.set(d.id, ref);
        }
        return <Duck key={d.id} duckId={d.id} bodies={bodiesFor(d.team)} frameRef={ref} offset={[0, 0]} label={d.name} />;
      })}
    </>
  );
}

const MAX_DOTS = 64 * 12;
const CORNER_ZONES = [0, 7, 56, 63];
// Overlays (ToF dots, the LiDAR scan, detection rays, the map) live on
// OVERLAY_LAYER, which lib/sim owns: the orbit camera sees it, the
// head-camera inset does not — a robot does not see its own sensor drawings.
// The DOM box the head-camera inset renders into (CamInset owns the element,
// InsetRender reads its rectangle every frame). Module state on purpose: no
// React state per frame.
const camInset: { el: HTMLDivElement | null; duckId: string | null } = { el: null, duckId: null };

/** The duck whose senses the inspector shows: the selected one, else the first with a detector. */
function sensedDuck(f: SimFrame | null, duckId: string | null): SimDuck | undefined {
  return f?.ducks.find((x) => x.id === duckId) ?? f?.ducks.find((x) => x.sensors?.det) ?? f?.ducks.find((x) => x.sensors);
}

/** Renders the frame twice: the orbit view, then - inside the CamInset
 *  box's rectangle - the selected duck's head camera, at the detector's
 *  field of view. Taking the render loop over (priority 1) is what makes
 *  a scissored second pass possible without a render target and a
 *  readback: the inset is pixels in the same canvas, the DOM box over it
 *  is just a border and the detection boxes. */
function InsetRender({ scene, client, enabled }: { scene: Scene; client: SimClient; enabled: boolean }) {
  const { gl, scene: three, camera, size } = useThree();
  const jawIdx = useMemo(() => scene.bodies.indexOf("jaw_soft"), [scene]);
  const cam = useMemo(() => new THREE.PerspectiveCamera(CAM_FOV_DEG[1], 1.35, 0.03, 20), []);
  useEffect(() => {
    camera.layers.enable(OVERLAY_LAYER);
  }, [camera]);
  useFrame(() => {
    gl.setScissorTest(false);
    gl.render(three, camera);
    const el = camInset.el;
    const f = client.frame;
    if (!enabled || !el || !f || jawIdx < 0) return;
    const d = sensedDuck(f, camInset.duckId);
    const jaw = d?.bodies[jawIdx];
    if (!d || !jaw) return;
    const det = d.sensors?.det;
    const fov = det?.fov ?? CAM_FOV_DEG;
    // The pose the detector's frame was captured from, when there is one:
    // the boxes then sit on what it saw. By the time a frame is available
    // (10 Hz, plus latency) the walking head has moved the picture by up to
    // a fifth of its width - measured, and exactly the lag a brain acts on.
    const { origin, forward, up } = det?.cam && det.cam.length === 7 ? capturePose(det.cam) : headCameraPose(jaw);
    // MuJoCo z-up -> three y-up (the stage group is rotated -90 deg about x).
    cam.position.set(origin[0], origin[2], -origin[1]);
    cam.up.set(up[0], up[2], -up[1]);
    cam.lookAt(origin[0] + forward[0], origin[2] + forward[2], -(origin[1] + forward[1]));
    cam.fov = fov[1];
    // renderInset measures the box and shapes the camera to it: it takes the
    // two DOMRects, not a rectangle, because the measuring is what went wrong
    // once (a pixel ratio applied twice - lib/inset.ts, lib/inset.test.ts).
    renderInset(gl, three, cam, el.getBoundingClientRect(), gl.domElement.getBoundingClientRect(), size);
  }, 1);
  return null;
}

const MAX_BOXES = 12;

/** The head-camera inset: a bordered box top-left, under the top bar, that the
 *  canvas renders the camera view into (InsetRender), with the detector's
 *  boxes drawn over it from bearing, elevation and apparent width - the
 *  three numbers a brain gets per detection, and nothing more. */
function CamInset({ client, duckId, top, belowRef, hidden, enabled, open, onToggle }: {
  client: SimClient;
  duckId: string | null;
  /** Where the box goes when nothing else holds the corner: under the top bar — until the user drags it. */
  top: number;
  /** A panel that already holds the top-left corner (the pitch scoreboard); the inset docks under it. */
  belowRef: React.RefObject<HTMLDivElement | null>;
  /** The editor owns the corner while it is open. */
  hidden: boolean;
  enabled: boolean;
  open: boolean;
  onToggle: () => void;
}) {
  const box = useRef<HTMLDivElement>(null);
  const boxes = useRef<(HTMLDivElement | null)[]>([]);
  const meta = useRef<HTMLDivElement>(null);
  const drag = useDrag("simCamPos", box, top, PAD);
  useEffect(() => {
    // Minimized: the box is a bar, not a viewport — take it off the render
    // pass so InsetRender does not scissor a camera view into a 26 px strip.
    camInset.el = enabled && open && !hidden ? box.current : null;
    camInset.duckId = duckId;
    return () => {
      camInset.el = null;
    };
  }, [enabled, open, hidden, duckId]);
  useEffect(() => {
    if (!enabled) return;
    let raf = 0;
    const paint = () => {
      raf = requestAnimationFrame(paint);
      const el = box.current;
      if (!el) return;
      // Where the user put it, else the dock: under the scoreboard if there
      // is one, else under the top bar.
      const custom = drag.posRef.current;
      const under = custom ? null : belowRef.current?.getBoundingClientRect();
      el.style.top = `${Math.round(custom ? custom.y : under ? under.bottom + GAP : top)}px`;
      el.style.left = `${custom ? custom.x : PAD}px`;
      el.style.width = `${CAM_W}px`;
      const f = client.frame;
      const d = sensedDuck(f, duckId);
      const det = d?.sensors?.det;
      // Aspect must match the detector FOV the boxes are drawn in. A 116°×60°
      // duck camera in a 62°×48° rectangle is why the person box sat beside
      // the G1 (inset.test.ts: different aspect → different horizontal fov).
      const fov = det?.fov ?? CAM_FOV_DEG;
      el.style.height = open ? `${Math.round(CAM_W / camAspect(fov))}px` : "";
      // A minimized bar stays put even with nothing selected — it is the only
      // way back to the view. The full inset still hides when there is no duck,
      // and both give the corner to the editor while it is open.
      el.style.display = hidden ? "none" : d || !open ? "block" : "none";
      if (!open) return;
      let n = 0;
      if (det) {
        for (const it of det.items) {
          const b = boxes.current[n];
          if (!b || n >= MAX_BOXES) break;
          const { u, v, w, h } = detectionBox(it, fov);
          b.style.display = "block";
          b.style.left = `${((u - w / 2) * 100).toFixed(2)}%`;
          b.style.top = `${((v - h / 2) * 100).toFixed(2)}%`;
          b.style.width = `${(w * 100).toFixed(2)}%`;
          b.style.height = `${(h * 100).toFixed(2)}%`;
          const color = !it.name ? "#8a8f98" : it.cls === "person" ? "#5a8dd6" : it.cls === "ball" ? "#ff8c00" : it.cls === "duck" ? "#f2b632" : "#43c2b8";
          b.style.borderColor = color;
          b.style.borderStyle = it.name ? "solid" : "dashed";
          const label = b.firstChild as HTMLElement | null;
          if (label) {
            label.textContent = `${it.name ? it.cls : "ghost"} ${it.range.toFixed(2)} m`;
            label.style.background = color;
          }
          n++;
        }
      }
      for (let i = n; i < MAX_BOXES; i++) {
        const b = boxes.current[i];
        if (b) b.style.display = "none";
      }
      if (meta.current) {
        if (!d) meta.current.textContent = "";
        else if (!det) meta.current.textContent = `${d.id} · head camera · no detector`;
        else {
          const fresh = freshness(det.age, DET_FRESH_MS);
          const text = `${d.id} · head camera ${fov[0]}°×${fov[1]}° · ${det.items.length} det · ${fresh.text}`;
          if (meta.current.textContent !== text) meta.current.textContent = text;
          meta.current.title = fresh.title;
          meta.current.style.color = fresh.color;
        }
      }
    };
    raf = requestAnimationFrame(paint);
    return () => cancelAnimationFrame(raf);
  }, [client, duckId, top, belowRef, hidden, enabled, open, drag.posRef]);
  if (!enabled) return null;
  // Minimized: the same docked box, collapsed to a clickable bar — the head
  // camera's answer to the inspector's title bar.
  if (!open)
    return (
      <div
        ref={box}
        style={{ position: "absolute", top: 0, left: PAD, width: CAM_W, border: "1px solid #2b313b", borderRadius: 6, boxSizing: "border-box", zIndex: 20, overflow: "hidden", display: "none", background: "rgba(16,18,22,0.86)" }}
      >
        <button
          onPointerDown={drag.onPointerDown}
          onDoubleClick={drag.reset}
          onClick={(e) => {
            if (!consumeDragged(e.currentTarget)) onToggle();
          }}
          title="expand the head camera · drag to move · double-click to re-dock"
          aria-label="expand the head camera"
          style={{ ...HANDLE, display: "flex", alignItems: "center", width: "100%", background: "none", border: "none", color: "#9aa5b1", fontFamily: "ui-monospace, Menlo, monospace", fontSize: 10, padding: "4px 6px", pointerEvents: "auto" }}
        >
          <span style={{ flex: 1, textAlign: "left" }}>head camera</span>
          <span style={{ fontSize: 12, lineHeight: 1 }}>+</span>
        </button>
      </div>
    );
  return (
    <div
      ref={box}
      title="what the head camera sees, at the detector's field of view; boxes are the detections (bearing, elevation, apparent width) - all a brain gets"
      style={{ position: "absolute", top: 0, left: PAD, width: CAM_W, height: Math.round(CAM_W / camAspect(CAM_FOV_DEG)), border: "1px solid #2b313b", borderRadius: 6, boxSizing: "border-box", zIndex: 20, overflow: "hidden", pointerEvents: "none", display: "none" }}
    >
      {/* The grab strip: the whole top edge drags (the view underneath is
          pointer-transparent so the room stays clickable through the box). */}
      <div
        onPointerDown={drag.onPointerDown}
        onDoubleClick={drag.reset}
        title="drag to move · double-click to re-dock"
        style={{ ...HANDLE, position: "absolute", top: 0, left: 0, right: 0, height: 18, zIndex: 1, pointerEvents: "auto", background: "linear-gradient(rgba(16,18,22,0.75), rgba(16,18,22,0))" }}
      >
        <button
          onPointerDown={(e) => e.stopPropagation()}
          onClick={onToggle}
          title="minimize the head camera"
          aria-label="minimize the head camera"
          style={{ position: "absolute", top: 2, right: 2, background: "rgba(16,18,22,0.7)", border: "none", borderRadius: 3, color: "#9aa5b1", cursor: "pointer", fontFamily: "ui-monospace, Menlo, monospace", fontSize: 12, lineHeight: 1, padding: "2px 5px" }}
        >
          —
        </button>
      </div>
      <div style={{ position: "absolute", left: "50%", top: "50%", width: 10, height: 10, marginLeft: -5, marginTop: -5, border: "1px solid rgba(233,237,241,0.5)", borderRadius: "50%" }} />
      {Array.from({ length: MAX_BOXES }, (_, i) => (
        <div key={i} ref={(el) => { boxes.current[i] = el; }} style={{ position: "absolute", display: "none", border: "1.5px solid #fff", borderRadius: 2, boxSizing: "border-box" }}>
          <span style={{ position: "absolute", left: -1.5, top: -14, fontSize: 9, lineHeight: "12px", padding: "0 3px", color: "#101216", fontFamily: "ui-monospace, Menlo, monospace", whiteSpace: "nowrap", borderRadius: 2 }} />
        </div>
      ))}
      <div ref={meta} style={{ position: "absolute", left: 0, right: 0, bottom: 0, padding: "2px 6px", fontSize: 10, fontFamily: "ui-monospace, Menlo, monospace", color: "#9aa5b1", background: "rgba(16,18,22,0.7)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} />
    </div>
  );
}

/** ToF overlay: one dot per zone at the reported depth, colored by range,
 *  plus the frustum's four corner rays from the aperture. All ducks with a
 *  sensor, or only the selected one when something is selected. */
function TofOverlay({ scene, client, enabled }: { scene: Scene; client: SimClient; enabled: boolean }) {
  const inst = useRef<THREE.InstancedMesh>(null);
  const lines = useRef<THREE.LineSegments>(null);
  useEffect(() => {
    inst.current?.layers.set(OVERLAY_LAYER);
    lines.current?.layers.set(OVERLAY_LAYER);
  }, []);
  const jawIdx = useMemo(() => scene.bodies.indexOf("jaw_soft"), [scene]);
  const m = useMemo(() => new THREE.Matrix4(), []);
  const col = useMemo(() => new THREE.Color(), []);
  const linePos = useMemo(() => new Float32Array(MAX_DOTS * 2 * 3), []);
  useFrame(() => {
    const im = inst.current;
    const ls = lines.current;
    if (!im || !ls) return;
    const f = client.frame;
    let n = 0;
    let ln = 0;
    if (f && enabled && jawIdx >= 0) {
      const sel = getSelectedDuck();
      for (const d of f.ducks) {
        const tof = d.sensors?.tof;
        const jaw = d.bodies[jawIdx];
        if (!tof || !jaw || (sel && d.id !== sel)) continue;
        const { origin, pts } = tofZonePoints(jaw, tof.mm);
        pts.forEach((p, k) => {
          if (!p || n >= MAX_DOTS) return;
          m.makeTranslation(p[0], p[1], p[2]);
          im.setMatrixAt(n, m);
          col.set(depthColor(tof.mm[k]));
          im.setColorAt(n, col);
          n++;
        });
        // Frustum: aperture → the four corner zones that hit something.
        for (const k of CORNER_ZONES) {
          const p = pts[k];
          if (!p || ln * 6 + 6 > linePos.length) continue;
          const o = ln * 6;
          linePos[o] = origin[0];
          linePos[o + 1] = origin[1];
          linePos[o + 2] = origin[2];
          linePos[o + 3] = p[0];
          linePos[o + 4] = p[1];
          linePos[o + 5] = p[2];
          ln++;
        }
      }
    }
    im.count = n;
    im.instanceMatrix.needsUpdate = true;
    if (im.instanceColor) im.instanceColor.needsUpdate = true;
    const attr = ls.geometry.getAttribute("position") as THREE.BufferAttribute;
    attr.needsUpdate = true;
    ls.geometry.setDrawRange(0, ln * 2);
  });
  return (
    <>
      <instancedMesh ref={inst} args={[undefined, undefined, MAX_DOTS]} frustumCulled={false}>
        <sphereGeometry args={[0.008, 8, 6]} />
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
      <lineSegments ref={lines} frustumCulled={false}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[linePos, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#43c2b8" transparent opacity={0.35} />
      </lineSegments>
    </>
  );
}

/** Editing: an invisible floor that turns pointer clicks into world (x, y),
 *  plus markers for the draft's duck spawns and the armed wall start. */
function EditorFloor({ state, onClick }: { state: EditorState; onClick: (x: number, y: number) => void }) {
  const [fx, fy] = state.draft.floor.size;
  return (
    <group>
      <mesh
        position={[0, 0, 0.0005]}
        onPointerDown={(e) => {
          if (e.button !== 0) return;
          e.stopPropagation();
          onClick(e.point.x, -e.point.z);   // three world → MuJoCo (group is rotated -90° about X)
        }}
      >
        <planeGeometry args={[fx, fy]} />
        <meshBasicMaterial color="#43c2b8" transparent opacity={0.06} depthWrite={false} />
      </mesh>
      {state.draft.ducks.map((d) => {
        // A spawn marker wears its team's colour, so a roster is legible on
        // the floor before anything is loaded — and the arrow shows the
        // heading, which on a pitch is what decides the goal it attacks.
        const c = teamColor(d.team) ?? "#f2b632";
        return (
          <group key={d.id} position={[d.spawn[0], d.spawn[1], 0.01]} rotation={[0, 0, d.spawn[2]]}>
            <mesh>
              <ringGeometry args={[0.1, 0.13, 32]} />
              <meshBasicMaterial color={c} side={THREE.DoubleSide} />
            </mesh>
            <mesh position={[0.16, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
              <coneGeometry args={[0.03, 0.08, 12]} />
              <meshBasicMaterial color={c} />
            </mesh>
          </group>
        );
      })}
      {state.wallStart && (
        <mesh position={[state.wallStart[0], state.wallStart[1], 0.01]}>
          <ringGeometry args={[0.03, 0.05, 24]} />
          <meshBasicMaterial color="#cfcac2" side={THREE.DoubleSide} />
        </mesh>
      )}
    </group>
  );
}

const MAX_DET = 12 * 16;

/** Detector overlay: one ray per detection from the head camera, as long as
 *  the width-derived range, colored by class (person blue, duck amber, ball
 *  orange, ghost grey); a small ring at its end. */
function DetOverlay({ scene, client, enabled }: { scene: Scene; client: SimClient; enabled: boolean }) {
  const lines = useRef<THREE.LineSegments>(null);
  useEffect(() => {
    lines.current?.layers.set(OVERLAY_LAYER);
  }, []);
  const jawIdx = useMemo(() => scene.bodies.indexOf("jaw_soft"), [scene]);
  const pos = useMemo(() => new Float32Array(MAX_DET * 2 * 3), []);
  const colors = useMemo(() => new Float32Array(MAX_DET * 2 * 3), []);
  const col = useMemo(() => new THREE.Color(), []);
  useFrame(() => {
    const ls = lines.current;
    if (!ls) return;
    const f = client.frame;
    let n = 0;
    if (f && enabled && jawIdx >= 0) {
      const sel = getSelectedDuck();
      for (const d of f.ducks) {
        const det = d.sensors?.det;
        const jaw = d.bodies[jawIdx];
        if (!det || !jaw || (sel && d.id !== sel)) continue;
        for (const it of det.items) {
          if (n >= MAX_DET) break;
          const { origin, dir } = detectionRay(jaw, it);
          const L = Math.min(it.range, 4);
          const o = n * 6;
          pos[o] = origin[0]; pos[o + 1] = origin[1]; pos[o + 2] = origin[2];
          pos[o + 3] = origin[0] + dir[0] * L; pos[o + 4] = origin[1] + dir[1] * L; pos[o + 5] = origin[2] + dir[2] * L;
          col.set(!it.name ? "#8a8f98" : it.cls === "person" ? "#5a8dd6" : it.cls === "ball" ? "#ff8c00" : it.cls === "duck" ? "#f2b632" : "#43c2b8");
          for (let k = 0; k < 2; k++) col.toArray(colors, o + k * 3);
          n++;
        }
      }
    }
    (ls.geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
    (ls.geometry.getAttribute("color") as THREE.BufferAttribute).needsUpdate = true;
    ls.geometry.setDrawRange(0, n * 2);
  });
  return (
    <lineSegments ref={lines} frustumCulled={false}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[pos, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <lineBasicMaterial vertexColors transparent opacity={0.9} />
    </lineSegments>
  );
}

/** What a chase brain thinks about the ball, drawn on the pitch in its own
 *  odometry frame (the world frame under ideal odometry, like the map):
 *  the ball track's motion as a line from where the ball is to where the
 *  brain predicts it will stop (orange — the head yaws that way and the
 *  hunt aims there), the ball memory its search would walk to (grey ring),
 *  its line-up / push spot (teal), and the ball-sized blob its ToF sees on
 *  the floor at its feet (violet — `tofBall` arrives in the duck's heading
 *  frame, so it needs the duck's odometry pose to land here, and is skipped
 *  without one). Every chase duck, the selected one bright; under the
 *  sensors toggle (T). */
const MAX_CHASE = 16 * 40;
/** A bump is live for half a second — the window the brain stands through
 *  instead of turning — and the ToF's floor ball gets a violet of its own,
 *  distinct from the predicted-stop orange, memory grey and line-up teal. */
const BUMP_LIVE_S = 0.5;
const TOF_BALL_COLOR = "#b06cd9";
const TOF_BALL_DIM = "#4f3162";
/** A ball belief is drawn in the colour of the DUCK that holds it, not in a
 *  colour for the kind — asked on /sim 2026-09-09 ("is that for each team or
 *  each duck, it's hard to tell"), where four ducks drew identical orange and
 *  grey rings and nothing said whose was whose. The team's shell colour gives
 *  the hue and the duck's index within the team steps the lightness, so a ring
 *  reads as "cream's second duck" at a glance while the two teams stay apart.
 *  The KIND is the radius instead, ascending: line-up spot, predicted stop,
 *  memory. (The ToF blob keeps its own violet — it is a SENSOR reading rather
 *  than a belief, and it is off by default.) */
const BELIEF_R = { spot: 0.04, predicted: 0.055, memory: 0.07, tof: 0.03, shared: 0.10 };
function beliefTint(c: THREE.Color, team: string | null | undefined, idx: number, dim: boolean): string {
  c.set(teamSwatch(team, "#9aa5b1"));
  c.offsetHSL(0, dim ? -0.35 : 0.1, (idx % 3) * 0.13 - 0.09 + (dim ? -0.26 : 0));
  return `#${c.getHexString()}`;
}
function ChaseOverlay({ client, enabled }: { client: SimClient; enabled: boolean }) {
  const lines = useRef<THREE.LineSegments>(null);
  useEffect(() => {
    lines.current?.layers.set(OVERLAY_LAYER);
  }, []);
  const pos = useMemo(() => new Float32Array(MAX_CHASE * 2 * 3), []);
  const colors = useMemo(() => new Float32Array(MAX_CHASE * 2 * 3), []);
  const col = useMemo(() => new THREE.Color(), []);
  const tintCol = useMemo(() => new THREE.Color(), []);
  useFrame(() => {
    const ls = lines.current;
    if (!ls) return;
    const f = client.frame;
    let n = 0;
    const seg = (a: [number, number], b: [number, number], c: string) => {
      if (n >= MAX_CHASE) return;
      const o = n * 6;
      pos[o] = a[0]; pos[o + 1] = a[1]; pos[o + 2] = 0.012;
      pos[o + 3] = b[0]; pos[o + 4] = b[1]; pos[o + 5] = 0.012;
      col.set(c);
      col.toArray(colors, o);
      col.toArray(colors, o + 3);
      n++;
    };
    const ring = (c: [number, number], r: number, color: string, k = 10) => {
      for (let i = 0; i < k; i++) {
        const a0 = (i / k) * Math.PI * 2, a1 = ((i + 1) / k) * Math.PI * 2;
        seg([c[0] + r * Math.cos(a0), c[1] + r * Math.sin(a0)], [c[0] + r * Math.cos(a1), c[1] + r * Math.sin(a1)], color);
      }
    };
    if (f && enabled) {
      const sel = getSelectedDuck();
      const drawnTeams = new Set<string>();
      let idx = 0;
      for (const d of f.ducks) {
        const ch = d.brain?.inputs?.chase;
        if (!ch) continue;
        const dim = sel !== null && sel !== d.id;
        const tint = beliefTint(tintCol, d.team, idx++, dim);
        const ball = d.brain.inputs.tracks?.find((t) => t.cls === "ball" && t.xy);
        if (ch.predicted) {
          if (ball?.xy) seg(ball.xy, ch.predicted, tint);
          ring(ch.predicted, BELIEF_R.predicted, tint, 10);
        }
        if (ch.memory) ring(ch.memory, BELIEF_R.memory, tint, 6);
        if (ch.spot) ring([ch.spot[0], ch.spot[1]], BELIEF_R.spot, tint, 4);
        // …and the TEAM's shared belief, once per team: the board's own ball,
        // which is what a supporter steers by. Wider than any one duck's ring,
        // so "what we collectively think" reads apart from "what I think" —
        // the answer to whether they are communicating at all.
        const tb = d.brain.inputs.team;
        if (tb?.ball && !drawnTeams.has(tb.name)) {
          drawnTeams.add(tb.name);
          ring(tb.ball, BELIEF_R.shared, teamSwatch(d.team, "#9aa5b1"), 16);
        }
        // The ToF blob is a bearing/range off the duck's nose; the overlay
        // is odometry-frame, so it only lands with a pose to hang it on.
        if (ch.tofBall && d.odomEst) {
          const a = d.odomEst[2] + ch.tofBall[0], r = ch.tofBall[1];
          ring([d.odomEst[0] + r * Math.cos(a), d.odomEst[1] + r * Math.sin(a)], BELIEF_R.tof, dim ? TOF_BALL_DIM : TOF_BALL_COLOR, 8);
        }
      }
    }
    (ls.geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
    (ls.geometry.getAttribute("color") as THREE.BufferAttribute).needsUpdate = true;
    ls.geometry.setDrawRange(0, n * 2);
  });
  return (
    <lineSegments ref={lines} frustumCulled={false}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[pos, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <lineBasicMaterial vertexColors transparent opacity={0.9} />
    </lineSegments>
  );
}

/** Publishes projected duck trunks so clicks can select the nearest duck. */
function SimTargets({ client }: { client: SimClient }) {
  const { camera, gl } = useThree();
  const v = useMemo(() => new THREE.Vector3(), []);
  useFrame(() => {
    const f = client.frame;
    if (!f) {
      assignDrag.targets = [];
      return;
    }
    const rect = gl.domElement.getBoundingClientRect();
    const targets: AssignTarget[] = f.ducks.map((d) => {
      const t = d.bodies[1] ?? [0, 0, 0];
      v.set(t[0], t[2], -t[1]).project(camera);
      return {
        id: d.id,
        x: rect.left + ((v.x + 1) / 2) * rect.width,
        y: rect.top + ((1 - v.y) / 2) * rect.height,
        visible: v.z < 1,
      };
    });
    assignDrag.targets = targets;
  });
  return null;
}

/** What a duck believes the room looks like: its occupancy grid (brain
 *  layer, in its own odometry frame) painted onto a floor-level plane.
 *  Free = faint teal, occupied = amber, unknown = clear. Under odometry
 *  drift the grid closes the loop on its own walls (`map.corrections`
 *  counts the frames it nudged); what smear is left is what a 45° depth
 *  matrix cannot fix. */
function MapOverlay({ client, duckId, enabled }: { client: SimClient; duckId: string | null; enabled: boolean }) {
  const canvas = useMemo(() => document.createElement("canvas"), []);
  const texture = useMemo(() => {
    const t = new THREE.CanvasTexture(canvas);
    t.magFilter = THREE.NearestFilter;
    t.minFilter = THREE.NearestFilter;
    return t;
  }, [canvas]);
  const meshRef = useRef<THREE.Mesh>(null);
  useEffect(() => {
    meshRef.current?.layers.set(OVERLAY_LAYER);
  }, []);
  const last = useRef<{ frames: number; id: string | null }>({ frames: -1, id: null });
  useFrame(() => {
    const m = meshRef.current;
    if (!m) return;
    const f = client.frame;
    const id = duckId ?? f?.ducks[0]?.id ?? null;
    const map = enabled && f?.maps && id ? f.maps[id] : null;
    if (!map) {
      if (!f?.maps) return;          // keep the last painted map between map frames
      m.visible = false;
      return;
    }
    m.visible = true;
    if (map.frames === last.current.frames && id === last.current.id) return;
    last.current = { frames: map.frames, id };
    canvas.width = map.nx;
    canvas.height = map.ny;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(map.nx, map.ny);
    for (let j = 0; j < map.ny; j++) {
      for (let i = 0; i < map.nx; i++) {
        const c = map.cells.charCodeAt(j * map.nx + i) - 48;
        const k = ((map.ny - 1 - j) * map.nx + i) * 4;   // canvas rows run top-down; the grid runs from -y
        if (c === 2) { img.data[k] = 242; img.data[k + 1] = 182; img.data[k + 2] = 50; img.data[k + 3] = 220; }
        else if (c === 1) { img.data[k] = 67; img.data[k + 1] = 194; img.data[k + 2] = 184; img.data[k + 3] = 60; }
        else { img.data[k + 3] = 0; }
      }
    }
    ctx.putImageData(img, 0, 0);
    texture.needsUpdate = true;
    const w = map.nx * map.res, h = map.ny * map.res;
    m.scale.set(w, h, 1);
    m.position.set(map.origin[0] + w / 2, map.origin[1] + h / 2, 0.003);   // inside the stage's Z-up group
  });
  return (
    <mesh ref={meshRef} visible={false}>
      <planeGeometry args={[1, 1]} />
      <meshBasicMaterial map={texture} transparent depthWrite={false} />
    </mesh>
  );
}

/** The 8×8 depth matrix of the selected duck, painted straight off the
 *  stream (no React state per frame). */
function Heatmap({ client, duckId }: { client: SimClient; duckId: string | null }) {
  const cells = useRef<(HTMLDivElement | null)[]>([]);
  const meta = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  useEffect(() => {
    let raf = 0;
    const paint = () => {
      raf = requestAnimationFrame(paint);
      const f = client.frame;
      const d = f?.ducks.find((x) => x.id === duckId) ?? f?.ducks.find((x) => x.sensors);
      const tof = d?.sensors?.tof;
      for (let i = 0; i < 64; i++) {
        const c = cells.current[i];
        if (!c) continue;
        c.style.background = tof ? depthColor(tof.mm[i]) : "#1c2026";
      }
      if (meta.current) {
        if (!tof || !d) {
          meta.current.textContent = d ? "no ToF frame yet" : "select a duck with a ToF";
        } else {
          const fresh = freshness(tof.age, TOF_FRESH_MS);
          const z = hover !== null ? tof.mm[hover] : null;
          const zoneTxt =
            hover !== null && z !== null
              ? ` · zone ${Math.floor(hover / 8)},${hover % 8}: ${z ? (z / 1000).toFixed(2) + " m" : "no target"}`
              : "";
          const text = `${d.id} · ${d.tof ?? "?"} · ${fresh.text}${zoneTxt}`;
          if (meta.current.textContent !== text) meta.current.textContent = text;   // no DOM churn on a steady reading
          meta.current.title = fresh.title;
          meta.current.style.color = fresh.color;
        }
      }
    };
    raf = requestAnimationFrame(paint);
    return () => cancelAnimationFrame(raf);
  }, [client, duckId, hover]);
  return (
    <div>
      <div
        style={{ display: "grid", gridTemplateColumns: "repeat(8, 1fr)", gap: 2, width: 176, aspectRatio: "1" }}
        onMouseLeave={() => setHover(null)}
      >
        {Array.from({ length: 64 }, (_, i) => (
          <div
            key={i}
            ref={(el) => {
              cells.current[i] = el;
            }}
            onMouseEnter={() => setHover(i)}
            style={{ borderRadius: 2, background: "#1c2026", outline: hover === i ? "1px solid #fff" : "none" }}
          />
        ))}
      </div>
      <div ref={meta} style={{ marginTop: 6, color: "#9aa5b1", minHeight: 14 }} />
    </div>
  );
}

/** Scrub bar over the lab's ring of the last two minutes: pause pulls the
 *  ring, the slider picks a frame (every panel then reads that frame), tick
 *  marks are falls / brain transitions / drive-mode changes, save writes the
 *  ring to recordings/. Space toggles, ←/→ step while paused. */
function Timeline({ client }: { client: SimClient }) {
  const [frames, setFrames] = useState<SimFrame[] | null>(null);
  const [idx, setIdx] = useState(0);
  const [events, setEvents] = useState<FrameEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const framesRef = useRef<SimFrame[] | null>(null);
  framesRef.current = frames;
  const idxRef = useRef(0);
  idxRef.current = idx;

  const pause = async () => {
    setBusy(true);
    try {
      const ring = await fetchRing();
      if (!ring.length) {
        setMsg("nothing recorded yet");
        return;
      }
      setFrames(ring);
      setEvents(frameEvents(ring));
      const last = ring.length - 1;
      setIdx(last);
      client.scrub = ring[last];
    } catch (e) {
      setMsg(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  };
  const goLive = () => {
    client.scrub = null;
    setFrames(null);
    setEvents([]);
  };
  const seek = (i: number) => {
    const f = framesRef.current;
    if (!f) return;
    const j = Math.max(0, Math.min(f.length - 1, i));
    setIdx(j);
    client.scrub = f[j];
  };
  const save = async () => {
    const name = window.prompt("save the last two minutes as", `take-${new Date().toISOString().slice(11, 19).replace(/:/g, "")}`);
    if (!name) return;
    try {
      const h = await saveRecording(name);
      setMsg(`saved ${h.frames} frames (${h.span.toFixed(1)} s) as ${h.name}`);
    } catch (e) {
      setMsg(String((e as Error).message ?? e));
    }
  };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
      if (e.key === " ") {
        e.preventDefault();
        if (framesRef.current) goLive();
        else pause();
      } else if (framesRef.current && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
        e.preventDefault();
        seek(idxRef.current + (e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 25 : 1));
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    if (!msg) return;
    const id = setTimeout(() => setMsg(null), 4000);
    return () => clearTimeout(id);
  }, [msg]);

  const n = frames?.length ?? 0;
  const cur = frames?.[idx];
  const t0 = frames?.[0]?.t ?? 0;
  const t1 = frames?.[n - 1]?.t ?? 0;
  return (
    <div style={{ ...PANEL, bottom: 10, left: 320, right: 10, display: "flex", gap: 10, alignItems: "center" }}>
      {frames ? (
        <button style={{ ...BTN, borderColor: "#f2b632" }} onClick={goLive} title="space">
          ▶ live
        </button>
      ) : (
        <button style={{ ...BTN, borderColor: BTN_BORDER }} onClick={pause} disabled={busy} title="space">
          {busy ? "…" : "⏸ scrub"}
        </button>
      )}
      <div style={{ position: "relative", flex: 1, height: 22 }}>
        <input
          type="range"
          min={0}
          max={Math.max(0, n - 1)}
          value={frames ? idx : 0}
          disabled={!frames}
          onChange={(e) => seek(Number(e.target.value))}
          style={{ width: "100%", position: "absolute", top: 2, left: 0, margin: 0 }}
        />
        {frames &&
          events.map((ev, k) => (
            <div
              key={k}
              title={`${ev.t.toFixed(2)} s · ${ev.text}`}
              onClick={() => seek(ev.index)}
              style={{
                position: "absolute",
                left: `${(ev.index / Math.max(1, n - 1)) * 100}%`,
                top: 16,
                width: 2,
                height: 6,
                background: ev.kind === "fall" ? "#e5484d" : ev.kind === "brain" ? "#f2b632" : "#43c2b8",
                cursor: "pointer",
              }}
            />
          ))}
      </div>
      <span style={{ color: "#9aa5b1", minWidth: 150, textAlign: "right" }}>
        {frames && cur
          ? `${cur.t.toFixed(2)} s · frame ${idx + 1}/${n} · ${(t1 - t0).toFixed(0)} s ring`
          : msg ?? "space: pause + scrub"}
      </span>
      <button style={BTN} onClick={save} title="write the ring to recordings/">
        💾 save
      </button>
    </div>
  );
}

/**
 * Scene picker. A native <select> cannot carry a per-row control, and a
 * delete parked in the top bar costs a slot there for something you touch
 * once — so this is a small popup: click a row to pick it, click the ✕ on
 * one of your own saved scenes to delete it. Built-ins have no ✕.
 */
function ScenePicker({
  scenarios,
  pick,
  onPick,
  onDelete,
  busy,
}: {
  scenarios: ScenarioListing[];
  pick: string;
  onPick: (name: string) => void;
  onDelete: (name: string) => void | Promise<void>;
  busy: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [hover, setHover] = useState<string | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);

  // Close on an outside click or Escape, like the native menu it replaces.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!boxRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        setOpen(false);
      }
    };
    window.addEventListener("mousedown", onDown, true);
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("mousedown", onDown, true);
      window.removeEventListener("keydown", onKey, true);
    };
  }, [open]);

  const current = scenarios.find((s) => s.name === pick);
  const label = (s: ScenarioListing) => `${s.name} (${s.ducks} ducks, ${s.objects} obj)`;
  const groups: [string, ScenarioListing[]][] = [
    ["built in", scenarios.filter((s) => s.builtin)],
    ["saved by you", scenarios.filter((s) => !s.builtin)],
  ];

  const row = (s: ScenarioListing) => (
    <div
      key={s.name}
      onMouseEnter={() => setHover(s.name)}
      onMouseLeave={() => setHover((h) => (h === s.name ? null : h))}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        borderRadius: 4,
        background: s.name === pick ? "#2a3340" : hover === s.name ? "#1f242c" : "transparent",
      }}
    >
      <button
        onClick={() => {
          onPick(s.name);
          setOpen(false);
        }}
        style={{
          flex: 1,
          textAlign: "left",
          background: "transparent",
          border: "none",
          color: "#e9edf1",
          font: "inherit",
          padding: "3px 6px",
          cursor: "pointer",
          whiteSpace: "nowrap",
        }}
      >
        <span style={{ color: s.name === pick ? "#f2b632" : "transparent" }}>✓</span> {label(s)}
      </button>
      {!s.builtin && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete(s.name);
          }}
          disabled={busy}
          title={`delete "${s.name}" — removes its saved file`}
          style={{
            background: "transparent",
            border: "none",
            color: hover === s.name ? "#e06c6c" : "#5f6b78",
            font: "inherit",
            padding: "3px 7px",
            cursor: busy ? "not-allowed" : "pointer",
          }}
        >
          ✕
        </button>
      )}
    </div>
  );

  return (
    <div ref={boxRef} style={{ position: "relative" }}>
      <button style={{ ...BTN, padding: "3px 6px" }} onClick={() => setOpen((v) => !v)}>
        {current ? label(current) : pick || "…"} ▾
      </button>
      {open && (
        <div
          style={{
            ...PANEL,
            top: "calc(100% + 4px)",
            left: 0,
            padding: 4,
            minWidth: "100%",
            maxHeight: "60vh",
            overflowY: "auto",
            zIndex: 40,
            background: "#171a20",
            backdropFilter: "none",
            boxShadow: "0 8px 24px rgba(0,0,0,0.55)",
          }}
        >
          {groups.map(([title, list]) =>
            list.length === 0 ? null : (
              <div key={title}>
                <div style={{ color: "#5f6b78", padding: "4px 6px 2px", letterSpacing: ".06em" }}>{title}</div>
                {list.map(row)}
              </div>
            ),
          )}
        </div>
      )}
    </div>
  );
}

/**
 * The ducks' VOICES. Reads the live frame once a browser frame and, when a
 * follower's brain state asks for one, plays a quack (lib/quack.ts decides,
 * lib/quackaudio.ts makes the noise) and flaps the bill (lib/mouth.ts).
 * Renders nothing — it is a component only so it lives with the page.
 *
 * Always mounted: MUTE is only the sound. The scheduler keeps running and
 * the mouths keep moving with it, so a muted room still shows the ducks
 * talking — the first cut unmounted the whole loop on mute and the bills
 * went still with the audio, which reads as "the ducks stopped", not "I
 * turned the sound off".
 *
 * The audio box is shared and long-lived (`duckAudio()`), not per-mount: a
 * browser caps a page at a few dozen AudioContexts, and the 🎥 recorder's tap
 * hangs off the same one.
 */
function DuckVoices({ client, sound }: { client: SimClient; sound: boolean }) {
  const soundRef = useRef(sound);
  soundRef.current = sound;
  // The sound half: wake the audio box while unmuted, hand it back on mute.
  // Nothing here touches the loop below, so muting never restarts it.
  useEffect(() => {
    if (!sound) return;
    const audio = duckAudio();
    audio.setActive(true);
    audio.resume();
    // A context made before the page has been touched starts suspended
    // (autoplay policy) and no `resume()` of ours can lift that — the next
    // real gesture can. Cheap, once, and removed the moment it lands.
    const wake = () => audio.resume();
    window.addEventListener("pointerdown", wake);
    window.addEventListener("keydown", wake);
    return () => {
      audio.setActive(false);
      window.removeEventListener("pointerdown", wake);
      window.removeEventListener("keydown", wake);
    };
  }, [sound]);
  // The loop: what each duck says and when, muted or not.
  useEffect(() => {
    const voices = new Voices();
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const f = client.frame;
      if (!f) return;
      const ducks = f.ducks.map((d) => ({ id: d.id, graph: d.brain?.graph, state: d.brain?.state }));
      // Wall time schedules the voices; the world's own clock only says
      // whether it is running (a scrubbed or paused world is silent).
      const now = performance.now() / 1000;
      for (const ev of voices.update(ducks, now, f.t)) {
        duckMouths.say(ev.id, ev.kind, now);        // the bill moves with it (Duck.tsx)
        if (soundRef.current) duckAudio().play(ev.kind, ev.pitch);
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [client]);
  return null;
}

export default function SimViewer() {
  const [scene, setScene] = useState<Scene | null>(null);
  const [g1Scene, setG1Scene] = useState<Scene | null>(null);
  // Mesh sets for the non-duck BODIES a room holds (a MARS in the playroom,
  // a Menagerie model on a floor), by robot id. Filled on demand from the
  // frame stream — the same pattern the lab page uses — and never keyed on
  // the scenario, because a `/world/load` while a 21 MB fetch is in flight
  // would resolve it into a dead closure and throw the meshes away.
  const [robotScenes, setRobotScenes] = useState<Record<string, Scene>>({});
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [scenarios, setScenarios] = useState<ScenarioListing[]>([]);
  const [world, setWorld] = useState<WorldInfo | null>(null);
  const [pick, setPick] = useState<string>("living-room");
  const [loading, setLoading] = useState(false);
  const [driving, setDriving] = useState(false);
  const [showTof, setShowTof] = useState(true);
  const [showMap, setShowMap] = useState(true);
  const [showCam, setShowCam] = useState(() => loadJSON("simCam", true));
  // Floating duck name labels. The flag itself lives in the shared ui store,
  // read PER-FRAME by every Duck inside the Canvas (lib/ui.ts says why a
  // subscription there does not work); this owns the persistence, under the
  // same key as the lab page's 🏷 button, so the preference is one preference.
  const [showLabels, setShowLabels] = useState(() => loadJSON("duckLabels", true));
  // The ducks' voices, on by default: this page is as much for showing the
  // robots off as for debugging them, and only the FOLLOWERS have anything to
  // say (lib/quack.ts) — a soccer battery left running is silent either way.
  // Muting sticks, so a session that turns it off stays off across reloads.
  const [sound, setSound] = useState(() => loadJSON("simSound", true));
  const inspectorRef = useRef<HTMLDivElement>(null);
  const topBarRef = useRef<HTMLDivElement>(null);
  // The pitch scoreboard shares the top-left corner with the head-camera
  // inset; the inset docks under it when it is there.
  const pitchRef = useRef<HTMLDivElement>(null);
  // The top bar wraps to two or three rows on a narrow window; the
  // inspector is parked under whatever height it actually ends up with,
  // never at a guessed offset that lets it slide behind the header.
  const [frameBox, setFrameBox] = useState({ barBottom: TOP_BAR_MIN_BOTTOM, height: 0 });
  const [selected, setSelected] = useState<string | null>(null);
  const [editor, setEditor] = useState<EditorState | null>(null);
  const [possessed, setPossessed] = useState<string | null>(null);
  // The world's wall-clock speed. The LAB owns it (a second tab, or a script
  // POSTing /world/speed, must move this menu too), so this only ever
  // mirrors the frame — every control here sends and waits.
  const [speed, setSpeed] = useState<number>(SIM_SPEED_DEFAULT);
  // …but a keypress must answer at once, and it must COMPOUND: three taps of
  // ] are three separate events inside one 250 ms mirror window, so stepping
  // from the mirrored value would send 2x three times. `asked` is what we
  // last sent, held until the frame agrees (or gives up and tells us the lab
  // clamped it, or that another tab moved it).
  const asked = useRef<{ x: number; at: number } | null>(null);
  const askSpeed = useCallback((x: number) => {
    asked.current = { x, at: Date.now() };
    setSpeed(x);
    clientRef.current?.sendSpeed(x);
  }, []);
  // What the menu shows right now, readable from the key handler's own tick.
  const speedNowRef = useRef(speed);
  speedNowRef.current = speed;
  // The bottom-left controls panel folds into a pill, like the lab HUD's
  // 🎥 controls bar — it is reference text, and it sits over the room, so it
  // starts folded. Fresh storage key: the old one persisted the previous
  // default on first render, which is not a choice anyone made.
  const [lessonOpen, setLessonOpen] = useState(() => loadJSON("simControlsOpen", false));
  const [inspectorOpen, setInspectorOpen] = useState(() => loadJSON("simInspectorOpen", true));
  const [camOpen, setCamOpen] = useState(() => loadJSON("simCamOpen", true));
  // The score panel (pitch or tidy — a world has at most one) collapses to its
  // headline: minimizing a scoreboard should not cost you the score, only the
  // per-minute table under it, which is what covers the near half of the room.
  const [scoreOpen, setScoreOpen] = useState(() => loadJSON("simScoreOpen", true));
  const [graphOpen, setGraphOpen] = useState(() => loadJSON("simGraphOpen", true));
  // The brain menu starts on the five brains that ship. The 44 experiment
  // runs are one click away, not in the face of someone opening /sim for
  // the first time — that list was the most daunting thing on the page.
  const [allBrains, setAllBrains] = useState(() => loadJSON("simBrainMenuAll", false));
  const worldRef = useRef<WorldInfo | null>(null);
  worldRef.current = world;
  const [status, setStatus] = useState<{ rtf: number; mode: string; t: number; events: string[]; kbps: number; tidy: { total: number; inBasket: number; held: string[] } | null; perf: string; soccer: SimFrame["soccer"] }>({
    rtf: 0,
    perf: "",
    soccer: null,
    mode: "auto",
    t: 0,
    events: [],
    kbps: 0,
    tidy: null,
  });
  const lastBytes = useRef({ bytes: 0, at: 0 });
  const clientRef = useRef<SimClient | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);
  // Two-finger horizontal swipe → the same lateral truck as A/D, the
  // lab page's gesture (components/useTruckSwipe.ts). It stays live in
  // drive mode: nothing on the trackpad steers a duck, so there is no
  // owner to hand it to.
  useTruckSwipe(rootRef);
  const held = useRef<Held>(new Set());
  const drivingRef = useRef(driving);
  drivingRef.current = driving;
  // WASD/QE change owner when drive mode flips. Whatever was held belongs to
  // the previous owner, and it never sees the keyup — so hand both sides a
  // clean slate, or the camera flies on forever / the duck keeps its twist.
  useEffect(() => {
    cameraKeysClear();
    // Leaving drive with a key down strands that twist: clearing the held-set
    // means the coming keyup sends no stop. So stop the ducks — but ONLY then.
    // A bare zero is still a manual command, and the lab holds one for
    // OVERRIDE_HOLD_S (6 s), which would suspend every brain just because
    // someone tapped P twice.
    const stranded = held.current.size > 0;
    held.current.clear();
    if (!driving && stranded) clientRef.current?.sendCmd([0, 0, 0]);
  }, [driving]);

  useEffect(() => saveJSON("simControlsOpen", lessonOpen), [lessonOpen]);
  useEffect(() => saveJSON("simInspectorOpen", inspectorOpen), [inspectorOpen]);
  useEffect(() => saveJSON("simScoreOpen", scoreOpen), [scoreOpen]);
  useEffect(() => saveJSON("simGraphOpen", graphOpen), [graphOpen]);
  useEffect(() => saveJSON("simCamOpen", camOpen), [camOpen]);
  useEffect(() => saveJSON("simBrainMenuAll", allBrains), [allBrains]);

  // Measure the top bar (it grows a row at a time as the window narrows) and
  // the stage, so the inspector can be placed below the header and capped to
  // what is left of the viewport. Read once a frame, next to the cam inset's
  // own loop: the bar's height changes with the status text and the scenario
  // controls, not only with the window, and a stale offset here is what used
  // to let the inspector slide up behind the header.
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const bar = topBarRef.current;
      const root = rootRef.current;
      if (!bar || !root) return;
      const barBottom = Math.round(bar.offsetTop + bar.offsetHeight);
      const height = root.clientHeight;
      setFrameBox((prev) => (prev.barBottom === barBottom && prev.height === height ? prev : { barBottom, height }));
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);
  useEffect(() => saveJSON("simCam", showCam), [showCam]);
  useEffect(() => saveJSON("simSound", sound), [sound]);
  useEffect(() => {
    saveJSON("duckLabels", showLabels);
    setDuckLabels(showLabels);
  }, [showLabels]);

  // One mesh set per NON-DUCK body on the stage, loaded from the frame the
  // way the lab page does it: a poll rather than an effect keyed on the
  // scenario, so a scenario swap mid-fetch cannot strand the result, and a
  // body whose assets this lab does not have (a 404 every time) is retried a
  // few times and then left alone instead of polled forever.
  const robotScenesRef = useRef(new Set<string>());
  const robotSceneFails = useRef(new Map<string, number>());
  useEffect(() => {
    let stopped = false;
    const tick = () => {
      const f = clientRef.current?.frame;
      if (!f) return;
      for (const d of f.ducks) {
        const id = d.robot || "microduck";
        if (id === "microduck" || robotScenesRef.current.has(id)) continue;
        if ((robotSceneFails.current.get(id) ?? 0) >= 5) continue;
        robotScenesRef.current.add(id);
        fetchScene(id)
          .then((sc) => !stopped && setRobotScenes((prev) => ({ ...prev, [id]: sc })))
          .catch(() => {
            robotSceneFails.current.set(id, (robotSceneFails.current.get(id) ?? 0) + 1);
            robotScenesRef.current.delete(id);     // retry on the next tick
          });
      }
    };
    const iv = setInterval(tick, 1000);
    tick();
    return () => {
      stopped = true;
      clearInterval(iv);
    };
  }, []);

  useEffect(() => {
    const client = new SimClient(setConnected);
    clientRef.current = client;
    const load = () =>
      Promise.all([fetchScene(), fetchScenarios(), fetchWorld()])
        .then(([s, list, w]) => {
          setScene(s);
          setScenarios(list);
          setWorld(w);
          if (w.scenario) setPick(w.scenario.name);
          setError(null);
          if ((w.scenario?.persons ?? []).some((p) => p.kind === "g1")) {
            fetchG1Scene().then(setG1Scene).catch(() => setG1Scene(null));
          }
        })
        .catch(() => {
          setError("duck-lab not reachable on :8788 — start it with `uv run duck-lab …`");
          setTimeout(load, 2000);
        });
    load();
    // Low-rate status mirror (HUD numbers, toasts): 4 Hz is plenty.
    const statusTimer = setInterval(() => {
      const f = client.live;
      const ev = client.drainEvents();
      const now = Date.now();
      const lb = lastBytes.current;
      const kbps = lb.at ? ((client.bytes - lb.bytes) / 1024) / ((now - lb.at) / 1000) : 0;
      lastBytes.current = { bytes: client.bytes, at: now };
      setStatus((s) => ({
        rtf: f?.rtf ?? 0,
        mode: f?.mode ?? "auto",
        t: f?.t ?? 0,
        events: ev.length ? [...s.events, ...ev].slice(-4) : s.events,
        kbps: 0.7 * s.kbps + 0.3 * kbps,
        tidy: f?.tidy ?? null,
        soccer: f?.soccer ?? null,
        // Where the lab's 20 ms tick goes: physics+policies / sensors / frame encode.
        perf: f?.perf ? `${ms(f.perf.stepMs)}+${ms(f.perf.sensorMs)}+${ms(f.perf.encodeMs ?? 0)} ms` : "",
      }));
      setSelected(getSelectedDuck());
      setPossessed(f?.possessed ?? null);
      const served = f?.simSpeed ?? SIM_SPEED_DEFAULT;
      const a = asked.current;
      if (!a || a.x === served || Date.now() - a.at > 1500) {
        asked.current = null;
        setSpeed(served);
      }
    }, 250);
    // Drive: while P-mode is on, held keys become one twist, re-sent every
    // 100 ms (the lab holds a manual command for 6 s after the last one).
    //
    // This timer, not the keydown handler, is what actually feeds the world,
    // so it is where scrub has to be honoured: a key already DOWN when the
    // scrub began never sees another keydown, and would otherwise keep
    // driving ducks around a live world the user is no longer watching.
    // Entering the scrub retires those keys exactly like releasing them —
    // one stop, then silence (the coming keyup finds the set already empty,
    // so it sends nothing of its own).
    const driveTimer = setInterval(() => {
      if (clientRef.current?.scrub) {
        if (held.current.size) {
          held.current.clear();
          client.sendCmd([0, 0, 0]);
        }
        return;
      }
      if (drivingRef.current && held.current.size) client.sendCmd(twistFromKeys(held.current));
    }, 100);
    return () => {
      clearInterval(statusTimer);
      clearInterval(driveTimer);
      client.close();
    };
  }, []);

  useEffect(() => {
    const isTyping = (t: EventTarget | null) =>
      t instanceof HTMLElement &&
      (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || isTyping(e.target)) return;
      const k = e.key.toLowerCase();
      if (k === "r" && !e.shiftKey) {
        e.preventDefault();
        if (!e.repeat) clientRef.current?.sendReset();
        return;
      }
      if (k === "p") {
        if (!e.repeat) setDriving((v) => !v);
        return;
      }
      if (k === "t") {
        if (!e.repeat) setShowTof((v) => !v);
        return;
      }
      if (k === "v") {
        if (!e.repeat) setShowCam((v) => !v);
        return;
      }
      if (k === "l") {
        if (!e.repeat) setShowLabels((v) => !v);
        return;
      }
      // M is the map, as this page's button and the README have always said
      // — it simply never had a handler until the voices needed a key next
      // to it. Shift+M mutes the ducks: M alone is the mute key in every
      // video player, but it was claimed here first.
      if (k === "m") {
        if (!e.repeat) (e.shiftKey ? setSound : setShowMap)((v) => !v);
        return;
      }
      if (k === "i") {
        if (!e.repeat) setInspectorOpen((v) => !v);
        return;
      }
      if (k === "b") {
        if (!e.repeat) setScoreOpen((v) => !v);
        return;
      }
      if (k === "g") {
        if (!e.repeat) setGraphOpen((v) => !v);
        return;
      }
      // Shift+E, not E: plain E is the camera's vertical truck (lib/camera).
      if (k === "e" && e.shiftKey) {
        e.preventDefault();
        if (!e.repeat) setEditor((st) => (st ? null : { draft: emptyDraft(worldRef.current?.scenario ?? null), tool: null, wallStart: null }));
        return;
      }
      // [ slower, ] faster. Free keys (1-9 select ducks, WASD/QE fly or
      // drive), and the pair every timeline in the world already uses.
      if (k === "[" || k === "]") {
        e.preventDefault();
        // One step per PRESS, like every other key here. Auto-repeat walked
        // the whole ladder in ~150 ms, so holding the key to slow down
        // overshot to the far end and then re-sent that speed 30 times a
        // second for as long as it was down.
        if (!e.repeat) askSpeed(stepSpeed(asked.current?.x ?? speedNowRef.current, k === "]" ? 1 : -1));
        return;
      }
      if (k === "escape") {
        setSelectedDuck(null);
        return;
      }
      if (/^[1-9]$/.test(k)) {
        const d = clientRef.current?.frame?.ducks[Number(k) - 1];
        setSelectedDuck(d ? d.id : null);
        return;
      }
      // Scrubbing is a view of the PAST: the arrows step frames (the scrub bar
      // owns them), and nothing may command the live world you cannot see —
      // a twist sent here moves ducks off-screen and only shows up when you
      // go live again. Camera flight is exempt: pausing and then looking
      // around the frozen room is the whole point of the scrub.
      const scrubbing = !!clientRef.current?.scrub;
      if (scrubbing && k.startsWith("arrow")) return;
      if (!scrubbing && drivingRef.current && DRIVE_KEYS.has(k)) {
        e.preventDefault();
        held.current.add(k);
        // Send once immediately so a tap registers before the 100 ms tick.
        if (!e.repeat) clientRef.current?.sendCmd(twistFromKeys(held.current));
        return;
      }
      // Whenever drive is not consuming the keys — mode off, or scrubbing,
      // where it is inert — they fly the camera, exactly as on the lab page.
      // Shift+R is the view reset there, so it must reach the store even
      // though plain R restarts the world above.
      if ((scrubbing || !drivingRef.current) && cameraKeyDown(k, e.shiftKey)) e.preventDefault();
    };
    const onKeyUp = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      cameraKeyUp(k);
      if (held.current.delete(k) && held.current.size === 0) clientRef.current?.sendCmd([0, 0, 0]);
    };
    const onBlur = () => {
      // Stuck-key guard: never fly on after focus leaves — and never DRIVE on
      // either. A key down at cmd-tab never sees its keyup, and the lab holds
      // the last twist for OVERRIDE_HOLD_S, so the ducks would walk out the
      // hold with nothing in the viewer to stop them. Same shape as the scrub
      // branch of driveTimer: stop only if something was actually held, so a
      // bare zero never suspends the brains for nothing.
      if (held.current.size) {
        held.current.clear();
        clientRef.current?.sendCmd([0, 0, 0]);
      }
      cameraKeysClear();
    };
    window.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("keyup", onKeyUp, true);
    window.addEventListener("blur", onBlur);
    rootRef.current?.focus({ preventScroll: true });
    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("keyup", onKeyUp, true);
      window.removeEventListener("blur", onBlur);
    };
  }, [askSpeed]);

  const doLoad = async (name: string) => {
    setLoading(true);
    try {
      const w = await loadWorld(name);
      setWorld(w);
      setPick(name);
      setSelectedDuck(null);
      if ((w.scenario?.persons ?? []).some((p) => p.kind === "g1")) {
        fetchG1Scene().then(setG1Scene).catch(() => setG1Scene(null));
      } else {
        setG1Scene(null);
      }
    } catch (e) {
      setError(String((e as Error).message ?? e));
      setTimeout(() => setError(null), 4000);
    } finally {
      setLoading(false);
    }
  };

  // Only user scenarios can go — built-ins are generated, and the server
  // refuses them with a 409 anyway.
  const doDelete = async (name: string) => {
    if (!window.confirm(`delete the scene "${name}"? this cannot be undone.`)) return;
    setLoading(true);
    try {
      await deleteScenario(name);
      const list = await fetchScenarios();
      setScenarios(list);
      // The live world keeps running whatever it loaded; only the picker has
      // to move off a name that no longer exists.
      if (name === pick) {
        const live = worldRef.current?.scenario?.name;
        setPick(list.some((s) => s.name === live) ? live! : (list[0]?.name ?? ""));
      }
    } catch (e) {
      setError(String((e as Error).message ?? e));
      setTimeout(() => setError(null), 4000);
    } finally {
      setLoading(false);
    }
  };

  const downAt = useRef({ x: 0, y: 0 });
  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    downAt.current = { x: e.clientX, y: e.clientY };
    const t = e.target as HTMLElement | null;
    if (!t?.closest("button, input, select, a")) rootRef.current?.focus({ preventScroll: true });
  };
  const selectAt = (e: React.MouseEvent<HTMLDivElement>) => {
    const t = e.target as HTMLElement | null;
    if (t?.tagName !== "CANVAS" || editor) return;
    if (Math.hypot(e.clientX - downAt.current.x, e.clientY - downAt.current.y) > 5) return;
    setSelectedDuck(nearestDuck(e.clientX, e.clientY));
  };

  const selDuck: SimDuck | undefined = clientRef.current?.frame?.ducks.find((d) => d.id === selected);
  // WHICH RANGE SENSOR the panel is talking about. MARS has no ToF — its
  // range sense is a 360° planar scanner — so the inspector's block, the
  // brain-input row's label, the preset's label and the overlay toggle's
  // wording all come off the body's own channel instead of being the duck's
  // by default (lib/lidar.rangeChannel).
  const selChan = rangeChannel(selDuck);
  // The block's own body: the selected one, else the first with senses —
  // exactly the ToF grid's long-standing fallback, so the two cannot end up
  // describing different robots.
  const blockDuck = selDuck ?? clientRef.current?.frame?.ducks.find((d) => d.sensors);
  const blockChan = rangeChannel(blockDuck);
  // The toggle draws every body's range sensor when nothing is selected, so
  // its label follows the selection first and the room second.
  const overlayChan = selChan ?? rangeChannel(clientRef.current?.frame?.ducks.find((d) => rangeChannel(d)));
  const scenario = world?.scenario ?? null;
  const client = clientRef.current;
  // Asked for more than this box can step. Not an error — the loop runs
  // flat out and the RTF says what it got — but the page must not let the
  // menu imply a speed the world is not running at.
  // …and never while an ask is still in flight: `speed` is optimistic (the
  // key answers at once) but `status.rtf` is the lab's trailing window, so
  // comparing the two across the change accused the lab of a shortfall every
  // single time the speed went up.
  const behind = asked.current === null && speedShortfall(status.rtf, speed);
  // While editing, the statics on stage are the DRAFT's; the ducks and
  // objects keep streaming from the loaded world underneath.
  const shown = editor ? editor.draft : scenario;
  const inspectorTop = frameBox.barBottom + GAP;
  const inspectorDrag = useDrag("simInspectorPos", inspectorRef, inspectorTop, PAD);
  const inspectorY = inspectorDrag.pos?.y ?? inspectorTop;
  const inspectorMax = frameBox.height
    ? Math.max(120, frameBox.height - inspectorY - PAD)
    : undefined;

  return (
    <div
      ref={rootRef}
      tabIndex={0}
      onPointerDown={onPointerDown}
      onClick={selectAt}
      style={{ position: "fixed", inset: 0, background: BG, outline: "none" }}
    >
      <Canvas
        dpr={[1, 1.5]}
        gl={{ antialias: true, powerPreference: "high-performance" }}
        camera={{ position: [2.2, 1.6, 2.4], fov: 40, near: 0.01, far: 50 }}
      >
        <color attach="background" args={[BG]} />
        <fog attach="fog" args={[BG, 6, 14]} />
        {/* The stage (SimStage.tsx) paints its own floors, walls and contact
            blobs; the lights stay simple and shadow-free — a warm key, a cool
            fill, and a procedural environment map for highlights. */}
        <StageEnvironment />
        <hemisphereLight intensity={0.55} groundColor="#2a2c33" color="#dfe6f0" />
        <directionalLight position={[2.5, 4, 2]} intensity={1.7} color="#fff3e2" />
        <directionalLight position={[-2, 2.5, -1.5]} intensity={0.45} color="#8fa3c7" />
        <gridHelper args={[18, 72, "#3a4150", "#262a33"]} position={[0, -0.003, 0]} />
        <group rotation={[-Math.PI / 2, 0, 0]}>
          <Statics scenario={shown} />
          {editor && <EditorFloor state={editor} onClick={(x, y) => setEditor((st) => (st ? applyFloorClick(st, x, y) : st))} />}
          {client && <Dynamics scenario={scenario} client={client} g1Scene={g1Scene} />}
          {scene && client && (
            <SimDucks scene={scene} client={client} robotScenes={robotScenes} />
          )}
          {scene && client && <TofOverlay scene={scene} client={client} enabled={showTof} />}
          {/* …and the same toggle draws the planar scan for a body that has
              one instead of a ToF (components/SimLidar.tsx). */}
          {client && <LidarOverlay client={client} robotScenes={robotScenes} enabled={showTof} />}
          {scene && client && <DetOverlay scene={scene} client={client} enabled={showTof} />}
          {client && <ChaseOverlay client={client} enabled={showTof} />}
          {client && <MapOverlay client={client} duckId={selected} enabled={showMap} />}
          {scene && client && <InsetRender scene={scene} client={client} enabled={showCam} />}
        </group>
        {client && <SimTargets client={client} />}
        <OrbitControls
          makeDefault
          target={ORBIT_TARGET}
          maxPolarAngle={Math.PI / 2 - 0.02}
          minDistance={CAM_MIN_DIST}
          maxDistance={CAM_MAX_DIST}
          zoomSpeed={0.4}
        />
        <CameraKeys home={HOME_CAM} minDist={CAM_MIN_DIST} maxDist={CAM_MAX_DIST} />
        {/* 🎥 record / 📷 shot (SimRecord in the top bar) read the canvas through these */}
        <CaptureCanvas />
        <Snapshotter />
      </Canvas>

      {/* top bar */}
      <div
        ref={topBarRef}
        style={{ ...PANEL, top: PAD, left: PAD, right: PAD, display: "flex", flexWrap: "wrap", gap: 10, rowGap: 6, alignItems: "center", zIndex: 30 }}
      >
        <Link href="/" style={{ color: "#9aa5b1", textDecoration: "none" }}>
          ← lab
        </Link>
        <b>🌍 /sim</b>
        <Link
          href="/train"
          style={{ ...BTN, textDecoration: "none", lineHeight: 1.4 }}
          title="brain training runs — live curves from train-brain"
        >
          🎓 train
        </Link>
        <ScenePicker
          scenarios={scenarios}
          pick={pick}
          onPick={setPick}
          onDelete={doDelete}
          busy={loading}
        />
        <button style={BTN} disabled={loading} onClick={() => doLoad(pick)}>
          {loading ? "loading…" : "load"}
        </button>
        <button style={BTN} onClick={() => client?.sendReset()} title="R">
          ↺ restart
        </button>
        {/* Wall-clock speed. Amber when the lab cannot keep the promise —
            a 3v3 pitch costs ~6.6 ms of the 20 ms tick, so it runs out of
            box around 3x and asking for 4 gets you 3. The RTF beside it is
            the honest number; this menu is only the ask. */}
        <select
          value={speed}
          onChange={(e) => askSpeed(Number(e.target.value))}
          style={{
            ...BTN,
            padding: "3px 6px",
            borderColor: behind ? "#f2b632" : speed === 1 ? BTN_BORDER : "#43c2b8",
          }}
          title={
            behind
              ? `asked for ${speedLabel(speed)}, the lab is managing ${status.rtf.toFixed(2)}\u00d7 — this scene costs more than its 20 ms tick`
              : "wall-clock speed of the world ([ and ]). Fast forward to reach the interesting part; slow down to watch a fall or a kick land. The same 25 frames a second either way — a fast world jumps further between them."
          }
        >
          {/* A script may POST any speed in range, not just a preset, and a
              controlled <select> with no matching <option> renders EMPTY —
              the one thing this control must never do. So carry the current
              value whenever it is not on the ladder. */}
          {!SIM_SPEEDS.includes(speed as (typeof SIM_SPEEDS)[number]) && (
            <option value={speed}>{`⏩ ${speedLabel(speed)}`}</option>
          )}
          {SIM_SPEEDS.map((x) => (
            <option key={x} value={x}>
              {x === 1 ? `⏵ ${speedLabel(x)}` : x < 1 ? `🐌 ${speedLabel(x)}` : `⏩ ${speedLabel(x)}`}
            </option>
          ))}
        </select>
        <button
          style={{ ...BTN, background: driving ? "#3a2f10" : BTN.background, borderColor: driving ? "#f2b632" : BTN_BORDER }}
          onClick={() => setDriving((v) => !v)}
          title="P"
        >
          {driving ? (possessed ? `🎮 driving ${possessed}` : "🎮 driving ducks") : "🎮 drive"}
        </button>
        {(scenario?.persons?.length ?? 0) > 0 && (
          <select
            value={possessed ?? ""}
            onChange={(e) => {
              client?.sendPossess(e.target.value || null);
              if (e.target.value) setDriving(true);
            }}
            style={{ ...BTN, padding: "3px 6px" }}
            title="possess a person: your keys move them, the ducks keep their brains"
          >
            <option value="">drive: ducks</option>
            {scenario!.persons!.map((q) => (
              <option key={q.id} value={q.id}>
                be {q.id}
              </option>
            ))}
          </select>
        )}
        <button style={{ ...BTN, borderColor: showMap ? "#43c2b8" : BTN_BORDER }} onClick={() => setShowMap((v) => !v)} title="M: the selected duck's occupancy map, in its own odometry frame">
          map
        </button>
        {/* One toggle, whatever the selected body's range sensor is: the
            duck's ToF cone, or a MARS's 360° scan (lib/lidar.sensorOverlayLabel). */}
        <button style={{ ...BTN, borderColor: showTof ? "#43c2b8" : BTN_BORDER }} onClick={() => setShowTof((v) => !v)} title="T">
          {sensorOverlayLabel(overlayChan)}
        </button>
        <button style={{ ...BTN, borderColor: showCam ? "#43c2b8" : BTN_BORDER }} onClick={() => setShowCam((v) => !v)} title="V: the selected duck's head camera, with the detector's boxes">
          cam
        </button>
        <button style={{ ...BTN, borderColor: showLabels ? "#43c2b8" : BTN_BORDER }} onClick={() => setShowLabels((v) => !v)} title="L: the floating d0 · policy labels over the ducks">
          🏷 labels
        </button>
        <button
          style={{ ...BTN, borderColor: sound ? "#43c2b8" : BTN_BORDER, color: sound ? undefined : "#7c8796" }}
          onClick={() => setSound((v) => !v)}
          title="Shift+M: the ducks' voices — chirps while a follower has its person in sight, a questioning quack when it loses them. Only followers talk; a pitch or a tidy room is silent. Mute is the sound only: the bills keep moving."
        >
          {sound ? "🔊 quacks" : "🔇 quacks"}
        </button>
        <button
          style={{ ...BTN, borderColor: editor ? "#f2b632" : BTN_BORDER }}
          onClick={() => setEditor((st) => (st ? null : { draft: emptyDraft(scenario), tool: null, wallStart: null }))}
          title="Shift+E (plain E flies the camera down)"
        >
          ✎ edit
        </button>
        <SimRecord scenario={scenario?.name ?? null} btn={BTN} />
        <span style={{ flex: "1 1 0", minWidth: 0 }} />
        {/* Every number sits in a slot as wide as its widest value, so the
            line does not shuffle as t grows a digit, kB/s crosses 100, or
            "auto" becomes "manual" — a status line that moves is one you
            cannot read at a glance. The panel font is monospace, so ch is
            exact. */}
        <span style={{ color: "#9aa5b1", minWidth: 180, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {scenario ? scenario.name : "no world loaded"} · t <Slot w={6}>{status.t.toFixed(1)}</Slot> s · RTF{" "}
          <span style={{ color: behind ? "#f2b632" : undefined }} title={behind ? `${speedLabel(speed)} asked, ${status.rtf.toFixed(2)} managed` : undefined}>
            <Slot w={4}>{status.rtf.toFixed(2)}</Slot>
            {behind ? `/${speedLabel(speed)}` : ""}
          </span>{" "}
          · <Slot w={6} left>{status.mode}</Slot> · <Slot w={3}>{status.kbps.toFixed(0)}</Slot> kB/s
          {status.perf && (
            <span title="lab cost per 20 ms tick: physics+policies + sensors + frame encode"> · {status.perf}</span>
          )}
        </span>
        <span style={{ color: connected ? "#43c2b8" : "#f2b632", flexShrink: 0 }}>{connected ? "● live" : "○ offline"}</span>
      </div>

      {/* inspector */}
      <div
        ref={inspectorRef}
        style={{
          ...PANEL,
          // Docked top-right until the user drags it somewhere (persisted).
          top: inspectorY,
          ...(inspectorDrag.pos ? { left: inspectorDrag.pos.x } : { right: PAD }),
          width: INSPECTOR_W,
          boxSizing: "border-box",
          // never taller than the room left under wherever it sits
          maxHeight: inspectorMax,
          overflowY: "auto",
          overscrollBehavior: "contain",
        }}
      >
        <div
          onPointerDown={inspectorDrag.onPointerDown}
          onDoubleClick={inspectorDrag.reset}
          title="drag to move · double-click to re-dock"
          style={{ ...HANDLE, display: "flex", alignItems: "flex-start", marginBottom: inspectorOpen ? 6 : 0 }}
        >
          <div style={{ flex: 1, color: "#9aa5b1", letterSpacing: ".08em", textTransform: "uppercase", fontSize: 10 }}>
            Inspector · sensors
          </div>
          <PanelToggle open={inspectorOpen} onToggle={() => setInspectorOpen((v) => !v)} what="the inspector" hint="I" />
        </div>
        {/* Minimized (I, or the — above): the title bar stays, so the head-camera
            inset below still docks to a full-width panel edge. */}
        {inspectorOpen && (
          <>
            {selDuck ? (
              <div style={{ marginBottom: 8 }}>
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  {selDuck.team && (
                    <span
                      style={{ width: 10, height: 10, borderRadius: 2, background: teamSwatch(selDuck.team), border: "1px solid rgba(0,0,0,.4)" }}
                      title={`team ${selDuck.team}${selDuck.role ? ` · ${selDuck.role}` : ""}`}
                    />
                  )}
                  <b>{selDuck.id}</b> · {selDuck.policy ? selDuck.policy.split(":").pop() : "stand"}
                  {selDuck.team && (
                    <span style={{ color: "#9aa5b1" }}>· {selDuck.team}{selDuck.role ? ` ${selDuck.role}` : ""}</span>
                  )}
                </div>
                <div style={{ color: "#9aa5b1" }}>
                  falls {selDuck.falls} · speed {selDuck.speed.toFixed(2)} / {selDuck.cmdSpeed.toFixed(2)} m/s
                </div>
                <div style={{ color: "#9aa5b1" }}>
                  beak {selDuck.beak}
                  {selDuck.mouth === undefined ? "" : ` · mouth ${Math.round(selDuck.mouth * 100)}%`}
                  {selDuck.holding ? ` · holding ${selDuck.holding}` : ""}
                  {selDuck.skill ? ` · skill ${selDuck.skill}` : ""}
                </div>
                <div style={{ color: selDuck.brain.kind === "wander" ? "#43c2b8" : "#9aa5b1" }}>
                  {selDuck.brain.kind} · {selDuck.brain.state} · vx {selDuck.brain.cmd[0].toFixed(2)} wz {selDuck.brain.cmd[2].toFixed(2)}
                </div>
                <div style={{ marginTop: 4, display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                  brain{" "}
                  <select
                    value={selDuck.brain.kind === "manual" ? "" : selDuck.brain.kind}
                    onChange={(e) => client?.sendBrain(selDuck.id, e.target.value)}
                    style={{ ...BTN, padding: "1px 4px" }}
                  >
                    {/* Rule brains flat, then the learned ones filed under their
                        use case and shown by title — 49 runs called p-batch-s1x
                        in one flat list was the menu nobody could use. */}
                    {(world?.brains ?? ["wander", "follow", "script"]).filter((k) => !k.startsWith("learned:")).map((k) => (
                      <option key={k} value={k}>
                        {k}
                      </option>
                    ))}
                    {world?.learned?.length
                      ? menuBrains(world.learned, selDuck.brain.kind, allBrains).groups.map(([label, brains]) => (
                          <optgroup key={label} label={label}>
                            {brains.map((b) => (
                              <option key={b.name} value={`learned:${b.name}`} title={b.description ?? undefined}>
                                {b.title ? `${b.title} · ${b.name}` : b.name}
                              </option>
                            ))}
                          </optgroup>
                        ))
                      : (world?.brains ?? []).filter((k) => k.startsWith("learned:")).map((k) => (
                          <option key={k} value={k}>
                            {k}
                          </option>
                        ))}
                    {selDuck.brain.kind === "manual" && <option value="">manual</option>}
                  </select>
                  {world?.learned?.length ? (() => {
                    const hidden = menuBrains(world.learned, selDuck.brain.kind, allBrains).hidden;
                    return allBrains || hidden > 0 ? (
                      <button
                        onClick={() => setAllBrains((v) => !v)}
                        style={{ ...BTN, padding: "1px 6px", color: "#9aa5b1" }}
                        title={allBrains ? "back to the brains that ship" : "the sweep and A/B runs — experiments about the trainer, filed by what they tested"}
                      >
                        {allBrains ? "hide experiments" : `+${hidden} experiments`}
                      </button>
                    ) : null;
                  })() : null}
                  <label title="apply the brain's gaze intent to the walker's head command (the shipped walker never trained with one)">
                    <input type="checkbox" checked={selDuck.headApplied} onChange={(e) => client?.sendHead(selDuck.id, e.target.checked)} /> head
                  </label>
                </div>
                {selDuck.brain.inputs && (selDuck.brain.inputs.tof || selDuck.brain.inputs.lidar || selDuck.brain.inputs.det) && (
                  <div style={{ marginTop: 4, display: "grid", gridTemplateColumns: "auto 1fr", gap: "2px 8px", color: "#9aa5b1" }}>
                    {(["tof", "lidar", "det"] as const).map((k) => {
                      const inp = selDuck.brain.inputs[k];
                      if (!inp) return null;
                      // A body with a scanner and no ToF still reports its
                      // range input under `tof`, because the lab hands its
                      // brains an ADAPTED 8×8 and keeps the SCAN's timestamp
                      // (world/arena.World.senses_tof: a 6 Hz frame is up to
                      // 167 ms old and a brain gating on freshness must see
                      // that). So the row is labelled by the channel the body
                      // HAS, and the duplicate is dropped if a lab ever sends
                      // both.
                      if (k === "tof" && selChan === "lidar" && selDuck.brain.inputs.lidar) return null;
                      const label = k === "tof" && selChan === "lidar" ? "lidar" : k;
                      const age = inp.age === null ? null : Math.round(inp.age * 1000);
                      return [
                        <span key={`${k}l`} title={label === "lidar" ? "the age of the 360° SCAN the adapted 8×8 was binned from" : undefined}>{label}</span>,
                        <span key={`${k}v`} style={{ color: inp.stale ? "#f2b632" : "#43c2b8" }}>
                          {age === null ? "never" : `${age} ms`}{inp.stale ? " · stale" : ""}{k === "det" && "n" in inp ? ` · ${inp.n} seen` : ""}
                        </span>,
                      ];
                    })}
                    {selDuck.brain.inputs.target && (
                      <>
                        <span>target</span>
                        <span>
                          {(selDuck.brain.inputs.target.bearing * 57.3).toFixed(0)}° · {selDuck.brain.inputs.target.range?.toFixed(2) ?? "?"} m · {selDuck.brain.inputs.target.since.toFixed(1)} s ago
                        </span>
                      </>
                    )}
                    {selDuck.brain.inputs.chase && (
                      <>
                        <span>chase</span>
                        <span title="where the brain predicts the ball will stop (it looks and hunts there) · the ball memory its search walks to">
                          {selDuck.brain.inputs.chase.job
                            ? `${selDuck.brain.inputs.chase.job} · ${selDuck.brain.inputs.chase.role}`
                            : selDuck.brain.inputs.chase.role} · {selDuck.brain.inputs.chase.kicks} kicks
                          {selDuck.brain.inputs.chase.predicted ? ` · ball → ${selDuck.brain.inputs.chase.predicted[0].toFixed(2)}, ${selDuck.brain.inputs.chase.predicted[1].toFixed(2)}` : ""}
                          {selDuck.brain.inputs.chase.memory ? ` · memory ${selDuck.brain.inputs.chase.memory[0].toFixed(2)}, ${selDuck.brain.inputs.chase.memory[1].toFixed(2)}` : ""}
                        </span>
                        {selDuck.brain.inputs.chase.bumped !== undefined && (
                          <>
                            <span>bump</span>
                            <span
                              style={{ color: selDuck.brain.inputs.chase.bumped !== null && selDuck.brain.inputs.chase.bumped < BUMP_LIVE_S ? "#ffd166" : "#9aa5b1" }}
                              title="how long since this duck's feet last touched another duck or a person — the contact list here, the IMU and the servo loads on the robot. Inside half a second it stands instead of turning in place."
                            >
                              {selDuck.brain.inputs.chase.bumped === null
                                ? "never"
                                : selDuck.brain.inputs.chase.bumped < BUMP_LIVE_S
                                  ? `● being bumped · ${selDuck.brain.inputs.chase.bumped.toFixed(2)} s`
                                  : `${selDuck.brain.inputs.chase.bumped.toFixed(1)} s ago`}
                            </span>
                          </>
                        )}
                        {selDuck.brain.inputs.chase.tofBall && (
                          <>
                            <span>tof ball</span>
                            <span
                              style={{ color: TOF_BALL_COLOR }}
                              title="a ball-sized blob the 8×8 ToF sees on the floor at the duck's feet, in its heading frame — the last 30 cm the head camera loses a floor ball in. Drawn as a violet ring on the floor; not fed to the brain as a ball (a blob at the feet is as often the other duck's foot)."
                            >
                              {(selDuck.brain.inputs.chase.tofBall[0] * 57.3).toFixed(0)}° · {selDuck.brain.inputs.chase.tofBall[1].toFixed(2)} m
                            </span>
                          </>
                        )}
                      </>
                    )}
                  </div>
                )}
                {selDuck.sensors?.det && (
                  <div style={{ marginTop: 4, color: "#9aa5b1" }}>
                    sees:{" "}
                    {selDuck.sensors.det.items.length
                      ? selDuck.sensors.det.items.map((it, i) => (
                          <span key={i} style={{ color: it.name ? "#c9d0d8" : "#8a8f98" }}>
                            {it.cls} {(it.bearing * 57.3).toFixed(0)}° {it.range.toFixed(1)} m{i < selDuck.sensors!.det!.items.length - 1 ? ", " : ""}
                          </span>
                        ))
                      : "nothing"}
                    <span title="person / ball / marker are simulated-only classes today; the robot's NPU detects ducks"> ⓘ</span>
                  </div>
                )}
                {/* The ARM channels, for a body that has a hand. Both render
                    nothing until a lab sends them (lib/sim.GripperPayload /
                    ArmPayload name what is still owed); `holding` beside the
                    badge is the pickable's id, which the frame does carry. */}
                {selDuck.sensors?.gripper && <GripperBlock g={selDuck.sensors.gripper} holding={selDuck.holding} />}
                {selDuck.sensors?.arm && <ArmBlock arm={selDuck.sensors.arm} />}
                <div style={{ marginTop: 4, display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                  {selDuck.tof && (
                    <>
                      tof{" "}
                      <select value={selDuck.tof} onChange={(e) => client?.sendNoise(selDuck.id, e.target.value as TofPreset, "tof")} style={{ ...BTN, padding: "1px 4px" }}>
                        {TOF_PRESETS.map((p) => (
                          <option key={p} value={p}>
                            {p}
                          </option>
                        ))}
                        {selDuck.tof === "custom" && <option value="custom">custom</option>}
                      </select>
                    </>
                  )}
                  {/* The same control for a body whose range sensor is the
                      scanner. The LABEL is the device, because that is what
                      the noise belongs to; the WIRE stays `tof`, because the
                      scenario has one range-sensor field and
                      `world/scenario.Duck`'s docstring says why ("how noisy
                      is this robot's range sense" is one question). The lab
                      REFUSES it today — `world_server.set_noise` has only a
                      `tof` and a `det` branch and a MARS's `d.tof` is None,
                      so it answers 409 and the event log says "noise
                      ignored"; the scenario's own field is what sets it at
                      load. A `lidar` branch there is the fix. */}
                  {selDuck.lidar && (
                    <>
                      lidar{" "}
                      <select
                        value={selDuck.lidar}
                        onChange={(e) => client?.sendNoise(selDuck.id, e.target.value as TofPreset, "tof")}
                        style={{ ...BTN, padding: "1px 4px" }}
                        title="the 360° scanner's noise preset (the scenario field is `tof` — one range-sensor field per body). NOTE: the lab has no live lidar branch yet and refuses the change; set it in the scenario."
                      >
                        {TOF_PRESETS.map((p) => (
                          <option key={p} value={p}>
                            {p}
                          </option>
                        ))}
                        {selDuck.lidar === "custom" && <option value="custom">custom</option>}
                      </select>
                    </>
                  )}
                  {selDuck.odom && (
                    <div style={{ marginTop: 4 }}>
                      odom{" "}
                      <select value={selDuck.odom} onChange={(e) => client?.sendNoise(selDuck.id, e.target.value as TofPreset, "odom")} style={{ ...BTN, padding: "1px 4px" }} title="odometry drift the brain lives with (roadmap 1.7)">
                        <option value="ideal">ideal</option>
                        <option value="datasheet">datasheet</option>
                        <option value="hostile">hostile</option>
                      </select>
                    </div>
                  )}
                  {selDuck.detector && (
                    <>
                      det{" "}
                      <select value={selDuck.detector} onChange={(e) => client?.sendNoise(selDuck.id, e.target.value as TofPreset, "det")} style={{ ...BTN, padding: "1px 4px" }}>
                        {TOF_PRESETS.map((p) => (
                          <option key={p} value={p}>
                            {p}
                          </option>
                        ))}
                        {selDuck.detector === "custom" && <option value="custom">custom</option>}
                      </select>
                    </>
                  )}
                </div>
              </div>
            ) : (
              <div style={{ color: "#9aa5b1", marginBottom: 8 }}>click a duck (or press 1–9)</div>
            )}
            {/* A learned brain's own view of the world — nothing for rule brains. */}
            {client && <BrainPanel client={client} duckId={selected} />}
            {/* THE RANGE-SENSOR BLOCK, chosen by the body's CHANNEL and never
                by its id: the duck's 8×8 grid, a wheeled body's polar plot
                (components/SimLidar.tsx), or a line of text for a body with
                neither. An empty ToF grid saying "no ToF frame yet" on a
                robot that has no ToF is a lie about the hardware, which is
                what a MARS in the playroom used to show. */}
            {client && blockChan === "tof" && <Heatmap client={client} duckId={selected} />}
            {client && blockChan === "lidar" && <LidarPlot client={client} duckId={selected} />}
            {client && blockChan === null && (
              <div style={{ color: "#9aa5b1" }}>
                {blockDuck ? `${blockDuck.id} has no range sensor` : "select a robot with a range sensor"}
              </div>
            )}
          </>
        )}
      </div>
      {client && <DuckVoices client={client} sound={sound} />}
      {client && <CamInset client={client} duckId={selected} top={inspectorTop} belowRef={pitchRef} hidden={!!editor} enabled={showCam} open={camOpen} onToggle={() => setCamOpen((v) => !v)} />}
      {/* The states the selected duck's brain moves through, with the moves
          it has actually made drawn between them (components/SimGraph). */}
      {client && !editor && (
        <StateGraphPanel
          client={client}
          duckId={selected}
          graphs={world?.graphs}
          open={graphOpen}
          onToggle={() => setGraphOpen((v) => !v)}
          minY={inspectorTop}
        />
      )}

      {/* keys — collapsible: reference text sitting over the room */}
      {lessonOpen ? (
        <div style={{ ...PANEL, bottom: 10, left: 10, width: 300, color: "#c9d0d8" }}>
          <div style={{ display: "flex", alignItems: "flex-start" }}>
            <div style={{ flex: 1, color: "#9aa5b1", letterSpacing: ".08em", textTransform: "uppercase", fontSize: 10, marginBottom: 4 }}>
              Controls
            </div>
            <PanelToggle open onToggle={() => setLessonOpen(false)} what="the controls" />
          </div>
          WASD/QE fly the camera (A/D slide, W/S zoom, Q/E rise) · arrows orbit · Shift+R view home
          <div style={{ marginTop: 6 }}>
            R restart · P drive (the same WASD/arrows, Q/E steer the ducks instead) · T sensors · V cam ·
            L labels · M map · Shift+M mute the ducks · I inspector · B scoreboard · G states · Shift+E edit ·
            1–9 select · Esc · space scrub
          </div>
          <div style={{ marginTop: 6 }}>
            [ and ] set the world&apos;s speed (0.25× … 8×): fast-forward to the interesting part,
            slow down to watch a fall land. Frames arrive at the same rate either way — a fast
            world jumps further between them — and RTF turns amber when the lab cannot keep up
            (a 3v3 pitch runs out of box near 3×).
          </div>
        </div>
      ) : (
        <button
          onClick={() => setLessonOpen(true)}
          title="controls"
          aria-label="controls"
          style={{ ...PANEL, bottom: 10, left: 10, color: "#c9d0d8", cursor: "pointer", lineHeight: 1 }}
        >
          ℹ️
        </button>
      )}

      {editor && (
        <SimEditor
          state={editor}
          setState={setEditor}
          top={inspectorTop}
          brains={world?.brains}
          onClose={() => setEditor(null)}
          onLoaded={(w) => {
            setWorld(w);
            if (w.scenario) setPick(w.scenario.name);
            fetchScenarios().then(setScenarios).catch(() => {});
            setSelectedDuck(null);
            if ((w.scenario?.persons ?? []).some((p) => p.kind === "g1")) {
              fetchG1Scene().then(setG1Scene).catch(() => setG1Scene(null));
            } else {
              setG1Scene(null);
            }
          }}
        />
      )}
      {status.soccer && (() => {
        // The scoreboard is per TEAM, not per goal mouth. The mouth counts
        // (`left`/`right`) are what the World keeps, and reading them as team
        // scores is the inversion eval_pitch carries a warning paragraph
        // about — so the panel shows `goalsFor`, which is that translation
        // done once on the server, and keeps the mouths in the tooltip.
        const soc = status.soccer;
        const teams = Object.keys(soc.possession ?? soc.goalsFor ?? {}).sort();
        const num = (rec: Record<string, number | null> | undefined, t: string) => Number(rec?.[t] ?? 0);
        const row = (label: string, title: string, cell: (t: string) => React.ReactNode) => (
          <>
            <span style={{ color: "#5f6b78" }} title={title}>{label}</span>
            {teams.map((t) => <span key={`${label}${t}`}>{cell(t)}</span>)}
          </>
        );
        return (
        <div ref={pitchRef} style={{ position: "absolute", top: inspectorTop, left: PAD, maxWidth: `calc(100vw - ${INSPECTOR_W + PAD * 3}px)`, boxSizing: "border-box", background: "rgba(16,18,22,0.9)", border: "1px solid #2b313b", borderRadius: 6, color: "#e9edf1", fontFamily: "ui-monospace, Menlo, monospace", fontSize: 12, padding: "8px 10px", zIndex: 20 }}>
          <div style={{ display: "flex", alignItems: "flex-start" }}>
            <div style={{ flex: 1, color: "#9aa5b1", letterSpacing: ".08em", textTransform: "uppercase", fontSize: 10 }}>Pitch</div>
            <PanelToggle open={scoreOpen} onToggle={() => setScoreOpen((v) => !v)} what="the scoreboard" hint="B" />
          </div>
          <div style={{ fontSize: 22, fontWeight: 600, display: "flex", gap: 10, alignItems: "center" }}
               title={`goal mouths: left ${soc.left} · right ${soc.right}`}>
            {teams.map((t, i) => (
              <span key={t} style={{ display: "flex", gap: 6, alignItems: "center" }}>
                {i > 0 && <span style={{ color: "#5f6b78", marginRight: 4 }}>·</span>}
                <span style={{ width: 12, height: 12, borderRadius: 3, background: teamSwatch(t, "#9aa5b1"), border: "1px solid rgba(0,0,0,.35)" }} />
                {soc.goalsFor ? num(soc.goalsFor, t) : "—"}
                <span style={{ fontSize: 12, color: "#9aa5b1" }}>{t}</span>
              </span>
            ))}
          </div>
          {/* Minimized (B, or the — above): the score line stays and everything
              under it goes. The per-minute table is the tall part, and it is
              what lies over the near half of the pitch. */}
          {scoreOpen && (
          <div style={{ color: "#9aa5b1", fontSize: 11 }} title="a goal within 4 s of a kick is the kick's; the rest were walked into">
            {soc.kicked ?? 0} kicked · {soc.bumped ?? 0} walked in
            {soc.ownGoals ? ` · own ${teams.reduce((a, t) => a + num(soc.ownGoals, t), 0)}` : ""}
            {soc.goalsUnattributed ? ` · ${soc.goalsUnattributed} unplaced` : ""}
          </div>
          )}
          {scoreOpen && soc.possession && (
            <div style={{ marginTop: 6, paddingTop: 6, borderTop: "1px solid #2b313b", display: "grid", gridTemplateColumns: `auto repeat(${teams.length}, 1fr)`, gap: "2px 8px", fontSize: 11 }}>
              <span style={{ color: "#5f6b78" }} title="goals are ~2.5 a run and resolve almost nothing; these are what the benchmark judges by">
                per min
              </span>
              {teams.map((t) => (
                <span key={`h${t}`} style={{ color: teamColor(t) ?? "#9aa5b1" }}>{t}</span>
              ))}
              {row("possession", "seconds a minute one of ours is nearest the ball inside 0.25 m — the cheap screen (9 seeds to see a 25% change)",
                   (t) => <span style={{ color: "#43c2b8" }}>{num(soc.possession, t).toFixed(1)}s</span>)}
              {row("advance", "metres a minute the ball is carried toward the goal this team attacks — the discriminator (43 seeds), but inflated by churn: read it with signed progress",
                   (t) => <span style={{ color: "#ff8c00" }}>{num(soc.ballAdvance, t).toFixed(2)}</span>)}
              {row("signed", "the same, signed — pushing the ball back toward your own goal is charged for. Churn cannot inflate this one.",
                   (t) => <span style={{ color: num(soc.ballProgress, t) < 0 ? "#d9534f" : "#e9edf1" }}>{num(soc.ballProgress, t).toFixed(2)}</span>)}
              {soc.ownGoals && row("own goals", "goals this team put into the mouth it defends — credited to the kicker inside 4 s, else to the last team on the ball",
                   (t) => <span style={{ color: num(soc.ownGoals, t) > 0 ? "#d9534f" : "#e9edf1" }}>{num(soc.ownGoals, t)}</span>)}
              {soc.kickCount && row("kicks back", "kicks whose ball ended up nearer this team's OWN goal 2 s later, out of its kicks — measured off the ball, not off what the brain meant",
                   (t) => <span style={{ color: num(soc.kicksBack, t) > num(soc.kickCount, t) / 2 ? "#d9534f" : "#e9edf1" }}>
                     {num(soc.kicksBack, t)}/{num(soc.kickCount, t)}</span>)}
              {soc.crowd && soc.crowd[teams[0]] !== null && row("crowd", "the fraction of the time two of this team's ducks are within 0.5 m of the ball at once — a pile-up",
                   (t) => <span style={{ color: num(soc.crowd, t) > 0.25 ? "#d9534f" : "#e9edf1" }}>{(num(soc.crowd, t) * 100).toFixed(0)}%</span>)}
            </div>
          )}
          {/* The kickoff banner survives a minimize: it is live state you need
              at the moment it appears, and it is one line. */}
          {soc.kickoff > 0 ? (
            <div style={{ color: "#ffd166" }}>GOAL in the {soc.lastGoal} mouth · kickoff in {soc.kickoff.toFixed(1)} s{soc.kickoffTeam ? ` · ${soc.kickoffTeam} kicks off` : ""}</div>
          ) : soc.state === "kickoff" ? (
            <div style={{ color: "#ffd166" }}>{soc.kickoffTeam ?? "the side that conceded"} kicks off · the ball is on the spot</div>
          ) : scoreOpen ? (
            <div style={{ color: "#9aa5b1" }}>one ball · a goal restarts from the spawns</div>
          ) : null}
        </div>
        );
      })()}
      {status.tidy && (
        <div style={{ ...PANEL, top: inspectorTop, left: editor ? 270 : PAD, maxWidth: `calc(100vw - ${INSPECTOR_W + PAD * 3}px)`, boxSizing: "border-box", color: "#c9d0d8" }}>
          <div style={{ display: "flex", alignItems: "flex-start" }}>
            <div style={{ flex: 1, color: "#9aa5b1", letterSpacing: ".08em", textTransform: "uppercase", fontSize: 10 }}>Tidy score</div>
            <PanelToggle open={scoreOpen} onToggle={() => setScoreOpen((v) => !v)} what="the tidy score" hint="B" />
          </div>
          <div style={{ fontSize: 20, fontVariantNumeric: "tabular-nums" }}>
            {status.tidy.inBasket} / {status.tidy.total} <span style={{ fontSize: 12, color: "#9aa5b1" }}>in the basket</span>
          </div>
          {scoreOpen && status.tidy.held.length > 0 && <div style={{ color: "#9aa5b1" }}>carrying {status.tidy.held.join(", ")}</div>}
          {scoreOpen && selDuck?.brain.inputs.tidy && (
            <div style={{ color: "#9aa5b1" }}>
              picked {selDuck.brain.inputs.tidy.picked} · delivered {selDuck.brain.inputs.tidy.delivered}
              {selDuck.brain.inputs.tidy.givenUp.length ? ` · gave up on ${selDuck.brain.inputs.tidy.givenUp.join(", ")}` : ""}
            </div>
          )}
        </div>
      )}
      {client && <Timeline client={client} />}

      {/* events */}
      {(status.events.length > 0 || error) && (
        <div style={{ ...PANEL, bottom: 54, right: 10, maxWidth: 360, color: error ? "#f2b632" : "#c9d0d8" }}>
          {error ?? status.events.map((e, i) => <div key={i}>{e}</div>)}
        </div>
      )}
    </div>
  );
}
