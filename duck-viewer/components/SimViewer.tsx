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
import {
  depthColor,
  fetchScenarios,
  fetchWorld,
  loadWorld,
  SimClient,
  TOF_PRESETS,
  type Scenario,
  type ScenarioListing,
  type SimDuck,
  type TofPreset,
  type WorldInfo,
} from "@/lib/sim";
import { buildBodyGeometries, Duck } from "./Duck";

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

// Where the ToF aperture sits on the head: the MJCF `tof` site, in the
// jaw_soft body frame (robot_walk.xml). Used only to draw the frustum lines.
const TOF_SITE_POS: [number, number, number] = [0.0135, 0.0224086, -0.0733];

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

/** Walls, static boxes and the floor from the loaded scenario (Z-up group). */
function Statics({ scenario }: { scenario: Scenario | null }) {
  if (!scenario) return null;
  const [fx, fy] = scenario.floor.size;
  return (
    <group>
      <mesh position={[0, 0, -0.001]}>
        <planeGeometry args={[fx, fy]} />
        <meshStandardMaterial color="#1c2026" roughness={0.95} />
      </mesh>
      {scenario.walls.map((w, i) => {
        const dx = w.to[0] - w.from[0];
        const dy = w.to[1] - w.from[1];
        const len = Math.hypot(dx, dy);
        return (
          <mesh
            key={`w${i}`}
            position={[(w.from[0] + w.to[0]) / 2, (w.from[1] + w.to[1]) / 2, w.height / 2]}
            rotation={[0, 0, Math.atan2(dy, dx)]}
          >
            <boxGeometry args={[len, w.thickness, w.height]} />
            <meshStandardMaterial color="#cfcac2" roughness={0.9} />
          </mesh>
        );
      })}
      {scenario.boxes.map((b, i) =>
        b.mass > 0 ? null : (
          <mesh key={`b${i}`} position={b.pos} rotation={[0, 0, b.yaw]}>
            <boxGeometry args={b.size} />
            <meshStandardMaterial color={new THREE.Color(b.rgba[0], b.rgba[1], b.rgba[2])} roughness={0.85} />
          </mesh>
        )
      )}
    </group>
  );
}

/** Free objects (balls, boxes with mass) posed from the frame stream. */
function Dynamics({ scenario, client }: { scenario: Scenario | null; client: SimClient }) {
  const refs = useRef(new Map<string, THREE.Group>());
  const tmpP = useMemo(() => new THREE.Vector3(), []);
  const tmpQ = useMemo(() => new THREE.Quaternion(), []);
  useFrame((_, dt) => {
    const f = client.frame;
    if (!f) return;
    const a = 1 - Math.exp(-16 * Math.min(dt, 0.1));
    for (const o of f.objects) {
      const g = refs.current.get(o.id);
      if (!g) continue;
      tmpP.set(o.pose[0], o.pose[1], o.pose[2]);
      tmpQ.set(o.pose[4], o.pose[5], o.pose[6], o.pose[3]);
      g.position.lerp(tmpP, a);
      g.quaternion.slerp(tmpQ, a);
    }
  });
  if (!scenario) return null;
  const freeBoxes = scenario.boxes.map((b, i) => ({ b, i })).filter(({ b }) => b.mass > 0);
  return (
    <group>
      {scenario.balls.map((ball, i) => (
        <group
          key={`ball${i}`}
          ref={(el) => {
            if (el) refs.current.set(`ball${i}`, el);
          }}
        >
          <mesh>
            <sphereGeometry args={[ball.radius, 24, 16]} />
            <meshStandardMaterial color="#ff8c00" roughness={0.6} />
          </mesh>
        </group>
      ))}
      {freeBoxes.map(({ b, i }) => (
        <group
          key={`box${i}`}
          ref={(el) => {
            if (el) refs.current.set(`box${i}`, el);
          }}
        >
          <mesh>
            <boxGeometry args={b.size} />
            <meshStandardMaterial color={new THREE.Color(b.rgba[0], b.rgba[1], b.rgba[2])} roughness={0.85} />
          </mesh>
        </group>
      ))}
    </group>
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
  const q = useMemo(() => new THREE.Quaternion(), []);
  const origin = useMemo(() => new THREE.Vector3(), []);
  const jawPos = useMemo(() => new THREE.Vector3(), []);
  const linePos = useMemo(() => new Float32Array(MAX_DOTS * 2 * 3), []);
  useFrame(() => {
    const im = inst.current;
    const ls = lines.current;
    if (!im || !ls) return;
    const f = client.frame;
    let n = 0;
    let ln = 0;
    if (f && enabled) {
      const sel = getSelectedDuck();
      for (const d of f.ducks) {
        const tof = d.sensors?.tof;
        if (!tof || (sel && d.id !== sel)) continue;
        tof.pts.forEach((p, k) => {
          if (!p || n >= MAX_DOTS) return;
          m.makeTranslation(p[0], p[1], p[2]);
          im.setMatrixAt(n, m);
          col.set(depthColor(tof.mm[k]));
          im.setColorAt(n, col);
          n++;
        });
        // Frustum: aperture → the four corner zones that hit something.
        const jaw = jawIdx >= 0 ? d.bodies[jawIdx] : undefined;
        if (jaw) {
          q.set(jaw[4], jaw[5], jaw[6], jaw[3]);
          jawPos.set(jaw[0], jaw[1], jaw[2]);
          origin.set(...TOF_SITE_POS).applyQuaternion(q).add(jawPos);
          for (const k of CORNER_ZONES) {
            const p = tof.pts[k];
            if (!p || ln * 6 + 6 > linePos.length) continue;
            const o = ln * 6;
            linePos[o] = origin.x;
            linePos[o + 1] = origin.y;
            linePos[o + 2] = origin.z;
            linePos[o + 3] = p[0];
            linePos[o + 4] = p[1];
            linePos[o + 5] = p[2];
            ln++;
          }
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
  const [selected, setSelected] = useState<string | null>(null);
  const [status, setStatus] = useState<{ rtf: number; mode: string; t: number; events: string[] }>({
    rtf: 0,
    mode: "auto",
    t: 0,
    events: [],
  });
  const clientRef = useRef<SimClient | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const held = useRef<Held>(new Set());
  const drivingRef = useRef(driving);
  drivingRef.current = driving;

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
      const f = client.frame;
      const ev = client.drainEvents();
      setStatus((s) => ({
        rtf: f?.rtf ?? 0,
        mode: f?.mode ?? "auto",
        t: f?.t ?? 0,
        events: ev.length ? [...s.events, ...ev].slice(-4) : s.events,
      }));
      setSelected(getSelectedDuck());
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
      if (k === "escape") {
        setSelectedDuck(null);
        return;
      }
      if (/^[1-9]$/.test(k)) {
        const d = clientRef.current?.frame?.ducks[Number(k) - 1];
        setSelectedDuck(d ? d.id : null);
        return;
      }
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
    if (t?.tagName !== "CANVAS") return;
    if (Math.hypot(e.clientX - downAt.current.x, e.clientY - downAt.current.y) > 5) return;
    setSelectedDuck(nearestDuck(e.clientX, e.clientY));
  };

  const selDuck: SimDuck | undefined = clientRef.current?.frame?.ducks.find((d) => d.id === selected);
  const scenario = world?.scenario ?? null;
  const client = clientRef.current;

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
        <hemisphereLight intensity={0.65} groundColor="#2a2c33" color="#dfe6f0" />
        <directionalLight position={[2.5, 4, 2]} intensity={1.9} />
        <directionalLight position={[-2, 2.5, -1.5]} intensity={0.4} color="#8fa3c7" />
        <gridHelper args={[18, 72, "#3a4150", "#262a33"]} position={[0, -0.003, 0]} />
        <group rotation={[-Math.PI / 2, 0, 0]}>
          <Statics scenario={scenario} />
          {client && <Dynamics scenario={scenario} client={client} />}
          {scene && client && <SimDucks scene={scene} client={client} />}
          {scene && client && <TofOverlay scene={scene} client={client} enabled={showTof} />}
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
          {driving ? "🎮 driving (WASD)" : "🎮 drive"}
        </button>
        <button style={{ ...BTN, borderColor: showTof ? "#43c2b8" : "#2b313b" }} onClick={() => setShowTof((v) => !v)} title="T">
          ToF overlay
        </button>
        <span style={{ flex: 1 }} />
        <span style={{ color: "#9aa5b1" }}>
          {scenario ? scenario.name : "no world loaded"} · t {status.t.toFixed(1)} s · RTF {status.rtf.toFixed(2)} · {status.mode}
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
            <div style={{ color: selDuck.brain.kind === "wander" ? "#43c2b8" : "#9aa5b1" }}>
              brain {selDuck.brain.kind} · {selDuck.brain.state} · wz {selDuck.brain.cmd[2].toFixed(2)}
            </div>
            {selDuck.tof && (
              <div style={{ marginTop: 4 }}>
                noise{" "}
                <select
                  value={selDuck.tof}
                  onChange={(e) => client?.sendNoise(selDuck.id, e.target.value as TofPreset)}
                  style={{ ...BTN, padding: "1px 4px" }}
                >
                  {TOF_PRESETS.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                  {selDuck.tof === "custom" && <option value="custom">custom</option>}
                </select>
              </div>
            )}
          </div>
        ) : (
          <div style={{ color: "#9aa5b1", marginBottom: 8 }}>click a duck (or press 1–9)</div>
        )}
        {client && <Heatmap client={client} duckId={selected} />}
      </div>

      {/* lesson / keys */}
      <div style={{ ...PANEL, bottom: 10, left: 10, width: 300, color: "#c9d0d8" }}>
        <div style={{ color: "#9aa5b1", letterSpacing: ".08em", textTransform: "uppercase", fontSize: 10, marginBottom: 4 }}>
          What the duck sees
        </div>
        The walking policy is blind: 61 numbers about its own body, none about the room. The 8×8 time-of-flight
        matrix on its head is what a brain gets instead: 64 distances, 15 times a second, ~45° wide. Dots are
        where the sensor <i>claims</i> a surface is. In auto mode a tiny wander brain reads the middle columns
        and steers toward the open side; it emits only a twist, the same command the real robot takes.
        <div style={{ marginTop: 6, color: "#9aa5b1" }}>
          R restart · P drive (WASD/arrows, Q/E strafe) · T ToF · 1–9 select · Esc
        </div>
      </div>

      {/* events */}
      {(status.events.length > 0 || error) && (
        <div style={{ ...PANEL, bottom: 10, right: 10, maxWidth: 360, color: error ? "#f2b632" : "#c9d0d8" }}>
          {error ?? status.events.map((e, i) => <div key={i}>{e}</div>)}
        </div>
      )}
    </div>
  );
}
