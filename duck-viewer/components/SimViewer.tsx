"use client";

// /sim — the WORLD page: one room, many ducks, and what each duck SENSES.
// Reuses the lab page's stage (scene meshes, Duck renderer, selection store)
// against the lab's world mode (/ws/sim, world_server.py). Keys: R restarts
// the world, P toggles drive mode (WASD / arrows steer every duck), T toggles
// the ToF overlay, 1–9 select a duck, Esc deselects.

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { fetchScene, type DuckFrame, type Scene } from "@/lib/lab";
import { assignDrag, nearestDuck, type AssignTarget } from "@/lib/assign";
import { getSelectedDuck, setSelectedDuck } from "@/lib/select";
import { loadJSON, saveJSON } from "@/lib/persist";
import {
  depthColor,
  fetchRing,
  fetchScenarios,
  fetchWorld,
  frameEvents,
  loadWorld,
  saveRecording,
  SimClient,
  detectionRay,
  tofZonePoints,
  TOF_PRESETS,
  type FrameEvent,
  type SimFrame,
  type ScenarioListing,
  type SimDuck,
  type TofPreset,
  type WorldInfo,
} from "@/lib/sim";
import { buildBodyGeometries, Duck } from "./Duck";
import { applyFloorClick, emptyDraft, SimEditor, type EditorState } from "./SimEditor";
import { Dynamics, StageEnvironment, Statics } from "./SimStage";

const BG = "#101216";
const PANEL: React.CSSProperties = {
  position: "absolute",
  background: "rgba(16,18,22,0.86)",
  border: "1px solid #2b313b",
  borderRadius: 6,
  color: "#e9edf1",
  fontFamily: "ui-monospace, Menlo, monospace",
  fontSize: 12,
  padding: "8px 10px",
  zIndex: 20,
  backdropFilter: "blur(6px)",
};
const BTN: React.CSSProperties = {
  background: "#1f242c",
  color: "#e9edf1",
  border: "1px solid #2b313b",
  borderRadius: 4,
  padding: "3px 8px",
  fontFamily: "inherit",
  fontSize: 12,
  cursor: "pointer",
};

type Held = Set<string>;

/** Held drive keys → one twist command [vx, vy, wz]. */
function twistFromKeys(h: Held): [number, number, number] {
  const vx = (h.has("w") || h.has("arrowup") ? 0.3 : 0) + (h.has("s") || h.has("arrowdown") ? -0.2 : 0);
  const vy = (h.has("q") ? 0.15 : 0) + (h.has("e") ? -0.15 : 0);
  const wz = (h.has("a") || h.has("arrowleft") ? 0.8 : 0) + (h.has("d") || h.has("arrowright") ? -0.8 : 0);
  return [vx, vy, wz];
}
const DRIVE_KEYS = new Set(["w", "a", "s", "d", "q", "e", "arrowup", "arrowdown", "arrowleft", "arrowright"]);

/** Every duck of the frame, rendered by the lab page's Duck component. */
function SimDucks({ scene, client }: { scene: Scene; client: SimClient }) {
  const bodies = useMemo(() => buildBodyGeometries(scene), [scene]);
  const [roster, setRoster] = useState<{ id: string; name: string }[]>([]);
  const sig = useRef("");
  const refs = useRef(new Map<string, React.MutableRefObject<DuckFrame | null>>());
  useFrame(() => {
    const f = client.frame;
    if (!f) return;
    const s = f.ducks.map((d) => `${d.id}\t${d.name}`).join("\n");
    if (s !== sig.current) {
      sig.current = s;
      setRoster(f.ducks.map((d) => ({ id: d.id, name: d.name })));
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
        let ref = refs.current.get(d.id);
        if (!ref) {
          ref = { current: null };
          refs.current.set(d.id, ref);
        }
        return <Duck key={d.id} duckId={d.id} bodies={bodies} frameRef={ref} offset={[0, 0]} label={d.name} />;
      })}
    </>
  );
}

const MAX_DOTS = 64 * 12;
const CORNER_ZONES = [0, 7, 56, 63];

/** ToF overlay: one dot per zone at the reported depth, colored by range,
 *  plus the frustum's four corner rays from the aperture. All ducks with a
 *  sensor, or only the selected one when something is selected. */
function TofOverlay({ scene, client, enabled }: { scene: Scene; client: SimClient; enabled: boolean }) {
  const inst = useRef<THREE.InstancedMesh>(null);
  const lines = useRef<THREE.LineSegments>(null);
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
      {state.draft.ducks.map((d) => (
        <group key={d.id} position={[d.spawn[0], d.spawn[1], 0.01]} rotation={[0, 0, d.spawn[2]]}>
          <mesh>
            <ringGeometry args={[0.1, 0.13, 32]} />
            <meshBasicMaterial color="#f2b632" side={THREE.DoubleSide} />
          </mesh>
          <mesh position={[0.16, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
            <coneGeometry args={[0.03, 0.08, 12]} />
            <meshBasicMaterial color="#f2b632" />
          </mesh>
        </group>
      ))}
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
          const ageMs = Math.round(tof.age * 1000);
          const z = hover !== null ? tof.mm[hover] : null;
          const zoneTxt =
            hover !== null && z !== null
              ? ` · zone ${Math.floor(hover / 8)},${hover % 8}: ${z ? (z / 1000).toFixed(2) + " m" : "no target"}`
              : "";
          meta.current.textContent = `${d.id} · ${d.tof ?? "?"} · age ${ageMs} ms${zoneTxt}`;
          meta.current.style.color = ageMs > 150 ? "#f2b632" : "#9aa5b1";
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
        <button style={BTN} onClick={pause} disabled={busy} title="space">
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

export default function SimViewer() {
  const [scene, setScene] = useState<Scene | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [scenarios, setScenarios] = useState<ScenarioListing[]>([]);
  const [world, setWorld] = useState<WorldInfo | null>(null);
  const [pick, setPick] = useState<string>("living-room");
  const [loading, setLoading] = useState(false);
  const [driving, setDriving] = useState(false);
  const [showTof, setShowTof] = useState(true);
  const [showMap, setShowMap] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [editor, setEditor] = useState<EditorState | null>(null);
  const [possessed, setPossessed] = useState<string | null>(null);
  // The bottom-left lesson panel folds into a pill, like the lab HUD's
  // 🎥 controls bar — it is reference text, and it sits over the room.
  const [lessonOpen, setLessonOpen] = useState(() => loadJSON("simLessonOpen", true));
  const worldRef = useRef<WorldInfo | null>(null);
  worldRef.current = world;
  const [status, setStatus] = useState<{ rtf: number; mode: string; t: number; events: string[]; kbps: number; tidy: { total: number; inBasket: number; held: string[] } | null; perf: string; soccer: { left: number; right: number; lastGoal: "left" | "right" | null; kickoff: number } | null }>({
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
  const held = useRef<Held>(new Set());
  const drivingRef = useRef(driving);
  drivingRef.current = driving;

  useEffect(() => saveJSON("simLessonOpen", lessonOpen), [lessonOpen]);

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
        perf: f?.perf ? `${f.perf.stepMs.toFixed(2)}+${f.perf.sensorMs.toFixed(2)}+${(f.perf.encodeMs ?? 0).toFixed(2)} ms` : "",
      }));
      setSelected(getSelectedDuck());
      setPossessed(f?.possessed ?? null);
    }, 250);
    // Drive: while P-mode is on, held keys become one twist, re-sent every
    // 100 ms (the lab holds a manual command for 6 s after the last one).
    const driveTimer = setInterval(() => {
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
      if (k === "e") {
        if (!e.repeat) setEditor((st) => (st ? null : { draft: emptyDraft(worldRef.current?.scenario ?? null), tool: null, wallStart: null }));
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
      if (clientRef.current?.scrub) return;   // scrubbing: arrows step frames
      if (drivingRef.current && DRIVE_KEYS.has(k)) {
        e.preventDefault();
        held.current.add(k);
        // Send once immediately so a tap registers before the 100 ms tick.
        if (!e.repeat) clientRef.current?.sendCmd(twistFromKeys(held.current));
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (held.current.delete(k) && held.current.size === 0) clientRef.current?.sendCmd([0, 0, 0]);
    };
    const onBlur = () => held.current.clear();
    window.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("keyup", onKeyUp, true);
    window.addEventListener("blur", onBlur);
    rootRef.current?.focus({ preventScroll: true });
    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("keyup", onKeyUp, true);
      window.removeEventListener("blur", onBlur);
    };
  }, []);

  const doLoad = async (name: string) => {
    setLoading(true);
    try {
      const w = await loadWorld(name);
      setWorld(w);
      setPick(name);
      setSelectedDuck(null);
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
  const scenario = world?.scenario ?? null;
  const client = clientRef.current;
  // While editing, the statics on stage are the DRAFT's; the ducks and
  // objects keep streaming from the loaded world underneath.
  const shown = editor ? editor.draft : scenario;

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
          {client && <Dynamics scenario={scenario} client={client} />}
          {scene && client && <SimDucks scene={scene} client={client} />}
          {scene && client && <TofOverlay scene={scene} client={client} enabled={showTof} />}
          {scene && client && <DetOverlay scene={scene} client={client} enabled={showTof} />}
          {client && <MapOverlay client={client} duckId={selected} enabled={showMap} />}
        </group>
        {client && <SimTargets client={client} />}
        <OrbitControls
          makeDefault
          target={[0, 0.12, 0]}
          maxPolarAngle={Math.PI / 2 - 0.02}
          minDistance={0.3}
          maxDistance={12}
          zoomSpeed={0.4}
        />
      </Canvas>

      {/* top bar */}
      <div style={{ ...PANEL, top: 10, left: 10, right: 10, display: "flex", gap: 10, alignItems: "center" }}>
        <Link href="/" style={{ color: "#9aa5b1", textDecoration: "none" }}>
          ← lab
        </Link>
        <b>🌍 /sim</b>
        <select value={pick} onChange={(e) => setPick(e.target.value)} style={{ ...BTN, padding: "3px 6px" }}>
          {scenarios.map((s) => (
            <option key={s.name} value={s.name}>
              {s.name}
              {s.builtin ? "" : " ·"} ({s.ducks} ducks, {s.objects} obj)
            </option>
          ))}
        </select>
        <button style={BTN} disabled={loading} onClick={() => doLoad(pick)}>
          {loading ? "loading…" : "load"}
        </button>
        <button style={BTN} onClick={() => client?.sendReset()} title="R">
          ↺ restart
        </button>
        <button
          style={{ ...BTN, background: driving ? "#3a2f10" : BTN.background, borderColor: driving ? "#f2b632" : "#2b313b" }}
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
        <button style={{ ...BTN, borderColor: showMap ? "#43c2b8" : "#2b313b" }} onClick={() => setShowMap((v) => !v)} title="M: the selected duck's occupancy map, in its own odometry frame">
          map
        </button>
        <button style={{ ...BTN, borderColor: showTof ? "#43c2b8" : "#2b313b" }} onClick={() => setShowTof((v) => !v)} title="T">
          ToF overlay
        </button>
        <button
          style={{ ...BTN, borderColor: editor ? "#f2b632" : "#2b313b" }}
          onClick={() => setEditor((st) => (st ? null : { draft: emptyDraft(scenario), tool: null, wallStart: null }))}
          title="E"
        >
          ✎ edit
        </button>
        <span style={{ flex: 1 }} />
        <span style={{ color: "#9aa5b1" }}>
          {scenario ? scenario.name : "no world loaded"} · t {status.t.toFixed(1)} s · RTF {status.rtf.toFixed(2)} · {status.mode} ·{" "}
          {status.kbps.toFixed(0)} kB/s
          {status.perf && (
            <span title="lab cost per 20 ms tick: physics+policies + sensors + frame encode"> · {status.perf}</span>
          )}
        </span>
        <span style={{ color: connected ? "#43c2b8" : "#f2b632" }}>{connected ? "● live" : "○ offline"}</span>
      </div>

      {/* inspector */}
      <div style={{ ...PANEL, top: 56, right: 10, width: 220 }}>
        <div style={{ color: "#9aa5b1", letterSpacing: ".08em", textTransform: "uppercase", fontSize: 10, marginBottom: 6 }}>
          Inspector · sensors
        </div>
        {selDuck ? (
          <div style={{ marginBottom: 8 }}>
            <div>
              <b>{selDuck.id}</b> · {selDuck.policy ? selDuck.policy.split(":").pop() : "stand"}
            </div>
            <div style={{ color: "#9aa5b1" }}>
              falls {selDuck.falls} · speed {selDuck.speed.toFixed(2)} / {selDuck.cmdSpeed.toFixed(2)} m/s
            </div>
            <div style={{ color: "#9aa5b1" }}>
              beak {selDuck.beak}
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
                {(world?.brains ?? ["wander", "follow", "script"]).map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
                {selDuck.brain.kind === "manual" && <option value="">manual</option>}
              </select>
              <label title="apply the brain's gaze intent to the walker's head command (the shipped walker never trained with one)">
                <input type="checkbox" checked={selDuck.headApplied} onChange={(e) => client?.sendHead(selDuck.id, e.target.checked)} /> head
              </label>
            </div>
            {selDuck.brain.inputs && (selDuck.brain.inputs.tof || selDuck.brain.inputs.det) && (
              <div style={{ marginTop: 4, display: "grid", gridTemplateColumns: "auto 1fr", gap: "2px 8px", color: "#9aa5b1" }}>
                {(["tof", "det"] as const).map((k) => {
                  const inp = selDuck.brain.inputs[k];
                  if (!inp) return null;
                  const age = inp.age === null ? null : Math.round(inp.age * 1000);
                  return [
                    <span key={`${k}l`}>{k}</span>,
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
        {client && <Heatmap client={client} duckId={selected} />}
      </div>

      {/* lesson / keys — collapsible: it is reference text sitting over the room */}
      {lessonOpen ? (
        <div style={{ ...PANEL, bottom: 10, left: 10, width: 300, color: "#c9d0d8" }}>
          <div style={{ display: "flex", alignItems: "flex-start" }}>
            <div style={{ flex: 1, color: "#9aa5b1", letterSpacing: ".08em", textTransform: "uppercase", fontSize: 10, marginBottom: 4 }}>
              What the duck sees
            </div>
            <button
              onClick={() => setLessonOpen(false)}
              title="collapse"
              style={{ background: "none", border: "none", color: "#9aa5b1", cursor: "pointer", fontFamily: "inherit", fontSize: 12, padding: "0 4px", marginLeft: 10, lineHeight: 1 }}
            >
              —
            </button>
          </div>
          The walking policy is blind: 61 numbers about its own body, none about the room. The 8×8 time-of-flight
          matrix on its head is what a brain gets instead: 64 distances, 15 times a second, ~45° wide. Dots are
          where the sensor <i>claims</i> a surface is. In auto mode a tiny wander brain reads the middle columns
          and steers toward the open side; it emits only a twist, the same command the real robot takes.
          <div style={{ marginTop: 6, color: "#9aa5b1" }}>
            R restart · P drive (WASD/arrows, Q/E strafe) · T ToF · E edit · 1–9 select · Esc · space scrub
          </div>
        </div>
      ) : (
        <button
          onClick={() => setLessonOpen(true)}
          title="what the duck sees · keys"
          style={{ ...PANEL, bottom: 10, left: 10, color: "#c9d0d8", cursor: "pointer" }}
        >
          👁 what the duck sees
        </button>
      )}

      {editor && (
        <SimEditor
          state={editor}
          setState={setEditor}
          onClose={() => setEditor(null)}
          onLoaded={(w) => {
            setWorld(w);
            if (w.scenario) setPick(w.scenario.name);
            fetchScenarios().then(setScenarios).catch(() => {});
            setSelectedDuck(null);
          }}
        />
      )}
      {status.soccer && (
        <div style={{ position: "absolute", top: 56, left: 10, background: "rgba(16,18,22,0.9)", border: "1px solid #2b313b", borderRadius: 6, color: "#e9edf1", fontFamily: "ui-monospace, Menlo, monospace", fontSize: 12, padding: "8px 10px", zIndex: 20 }}>
          <div style={{ color: "#9aa5b1", letterSpacing: ".08em", textTransform: "uppercase", fontSize: 10 }}>Pitch</div>
          <div style={{ fontSize: 22, fontWeight: 600 }}>
            {status.soccer.left} <span style={{ fontSize: 12, color: "#9aa5b1" }}>left</span> · {status.soccer.right} <span style={{ fontSize: 12, color: "#9aa5b1" }}>right</span>
          </div>
          {status.soccer.kickoff > 0 ? (
            <div style={{ color: "#ffd166" }}>GOAL {status.soccer.lastGoal} · kickoff in {status.soccer.kickoff.toFixed(1)} s</div>
          ) : (
            <div style={{ color: "#9aa5b1" }}>goals · chase brains, one ball · a goal restarts from the spawns</div>
          )}
        </div>
      )}
      {status.tidy && (
        <div style={{ ...PANEL, top: 56, left: editor ? 270 : 10, color: "#c9d0d8" }}>
          <div style={{ color: "#9aa5b1", letterSpacing: ".08em", textTransform: "uppercase", fontSize: 10 }}>Tidy score</div>
          <div style={{ fontSize: 20, fontVariantNumeric: "tabular-nums" }}>
            {status.tidy.inBasket} / {status.tidy.total} <span style={{ fontSize: 12, color: "#9aa5b1" }}>in the basket</span>
          </div>
          {status.tidy.held.length > 0 && <div style={{ color: "#9aa5b1" }}>carrying {status.tidy.held.join(", ")}</div>}
          {selDuck?.brain.inputs.tidy && (
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
