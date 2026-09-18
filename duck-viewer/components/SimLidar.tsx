"use client";

// The /sim inspector's LIDAR instruments, and the scan's overlay in the room.
//
// A wheeled body (MARS) has NO ToF: its range sensor is a 360-degree planar
// scanner, its camera is the stereo head camera, and its hand is a gripper.
// So the inspector renders one block per channel the selected body actually
// has (`lib/lidar.rangeChannel`) instead of the duck's three:
//
//   lidar    -> LidarPlot: the polar plot, plus the eight bearing columns the
//               body's brains really steer on, adapted here the way the lab
//               adapts them (lib/lidar.tofFromLidar ports
//               `sensors/lidar.tof_from_lidar`)
//   gripper  -> GripperBlock: the constraint torque at joint6 against the
//               1.0 N·m hold threshold
//   arm      -> ArmBlock: the achieved joint angles, with the commanded ones
//               marked — the pair is the servo sag a brain pre-compensates
//
// The duck's 8x8 grid, its detections list and its presets are untouched.
//
// Both of the per-frame painters here follow the page's established rule: no
// React state per frame. The plot is a CANVAS repainted in a rAF loop (360
// points at 60 Hz is what a canvas is for; 360 SVG nodes is not), the strip
// is eight divs written by ref, and the overlay is ONE BufferGeometry whose
// positions and colors are rewritten in place — no per-ray meshes.

import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

import type { Scene } from "@/lib/lab";
import { getSelectedDuck } from "@/lib/select";
import {
  depthColor,
  OVERLAY_LAYER,
  TOF_COLS,
  TOF_FOV_DEG,
  TOF_MAX_RANGE_M,
  quatRotate,
  type ArmPayload,
  type GripperPayload,
  type LidarPayload,
  type SimClient,
  type SimDuck,
} from "@/lib/sim";
import {
  armBar,
  gripperBar,
  lidarMaxRange,
  lidarNearest,
  lidarPlotPoints,
  lidarPolar,
  LIDAR_FRESH_MS,
  LIDAR_MIN_RANGE_M,
  robotFootprintM,
  tofFromLidar,
} from "@/lib/lidar";

const DIM = "#9aa5b1";
const TEAL = "#43c2b8";
const AMBER = "#f2b632";

/** The plot's box, in CSS px. As wide as the inspector's content column (the
 *  panel is 242 px border-box), so the range rings have room for their labels
 *  — the first cut was the old 112 px ring, on which "1 m" and "2 m" landed
 *  on top of each other and nothing could be read off it. */
const PLOT = 210;

/** The scan's mount body, as the robot's scene lists it
 *  (`sensors/lidar.DEFAULT_MOUNT`, and `mars.LIDAR_SITE`). The overlay draws
 *  from the APERTURE's own pose, which is what makes the mount offset
 *  automatic: the payload's `mount` is only needed where a drawing has to be
 *  re-read in the BASE frame, which is the adapter's job and not the ray's. */
const LIDAR_MOUNT_BODY = "base_laser";

/** How long a hit's tick is drawn in the room, m. Short on purpose: a tick at
 *  the surface reads as a contact, a full-length ray per return reads as fog. */
const HIT_TICK_M = 0.05;

/** Per body: 360 hit ticks + the ~65 rays of the front sector, and four
 *  bodies' worth of room. A cap, not an expectation. */
const MAX_LIDAR_SEG = 4 * 430;

/** A sensor's age as a reading — the ToF block's own wording (SimViewer's
 *  `freshness`), against the scanner's 6 Hz window. Duplicated rather than
 *  exported across the two files because the two windows differ (150 ms at
 *  15 Hz, 374 ms at 6 Hz) and the text is the same three words. */
function scanFreshness(ageS: number): { text: string; color: string; title: string } {
  const ms = Math.round(ageS * 1000);
  if (ms <= LIDAR_FRESH_MS) return { text: "fresh", color: DIM, title: `last scan ${ms} ms ago` };
  return { text: `stale · ${Math.round(ms / 50) * 50} ms`, color: AMBER, title: `last scan ${ms} ms ago` };
}

const deg = (rad: number) => `${(rad * 57.2958).toFixed(0)}°`;

/** The body the inspector is showing: the selected one, else the first with a
 *  scan — the ToF block's own fallback rule, so the two panels never disagree
 *  about whose senses are on screen. */
function scannedDuck(ducks: SimDuck[] | undefined, duckId: string | null): SimDuck | undefined {
  if (!ducks) return undefined;
  return ducks.find((x) => x.id === duckId) ?? ducks.find((x) => x.sensors?.lidar);
}

/**
 * The 360-degree scan as an instrument: a polar plot with the robot's nose
 * UP, range rings, the blind disc it cannot see inside of, the front sector
 * its brains read, and under it the eight bearing columns that sector is
 * binned into.
 *
 * Why the plot is centred on the LASER and not the chassis: the ranges and
 * bearings ARE the laser's (`LidarFrame.angles` are in the mount frame), and
 * on MARS the scanner sits 76.4 mm behind the base origin — a third of the
 * robot's length. Drawing the rays from the chassis origin would put every
 * obstacle 76 mm nearer than the measurement says. So the sensor is the
 * origin, and the BASE is drawn where it actually is relative to it (the nose
 * marker and the footprint circle), which is also the offset the strip's
 * adapter applies.
 */
export function LidarPlot({ client, duckId }: { client: SimClient; duckId: string | null }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const cells = useRef<(HTMLDivElement | null)[]>([]);
  const meta = useRef<HTMLDivElement>(null);
  const note = useRef<HTMLDivElement>(null);
  const strip = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  // What the canvas was last drawn FROM. The scanner runs at 6 Hz and this
  // loop at 60, so nine of every ten repaints would redraw an identical
  // picture (360 points, the rings, the sector, the labels). The DOM readouts
  // below are written every frame either way — they are three strings — and
  // the key includes the body, so selecting another robot repaints at once.
  const drawn = useRef("");
  //: The adapted columns and the nearest contact of the LAST painted scan,
  //: for the readouts that are written every frame.
  const last = useRef<{ tof: ReturnType<typeof tofFromLidar> | null; near: ReturnType<typeof lidarNearest>; rays: number }>({ tof: null, near: null, rays: 0 });
  useEffect(() => {
    let raf = 0;
    const paint = () => {
      raf = requestAnimationFrame(paint);
      const el = canvas.current;
      if (!el) return;
      const f = client.frame;
      const d = scannedDuck(f?.ducks, duckId);
      const scan = d?.sensors?.lidar;
      const ctx = el.getContext("2d");
      if (!ctx) return;
      // Backing store at the device's own pixels: a 1 px point on a 2x
      // display is otherwise a grey smudge, and this plot IS single pixels.
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (el.width !== PLOT * dpr) {
        el.width = PLOT * dpr;
        el.height = PLOT * dpr;
      }
      // Redraw only when the picture can have changed: a new SCAN, another
      // body, or a resized backing store. Everything the canvas shows comes
      // off those three.
      const key = `${d?.id ?? "-"}|${scan?.t ?? "-"}|${el.width}`;
      if (key !== drawn.current) {
        drawn.current = key;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, PLOT, PLOT);
        const max = scan ? lidarMaxRange(scan) : 6;
        const c = PLOT / 2;
        const pxPerM = c / max;
        // -- the graticule ------------------------------------------------
        ctx.fillStyle = "#0d1218";
        ctx.fillRect(0, 0, PLOT, PLOT);
        // The front sector: ±fov/2 out to the ToF's OWN max range, because
        // that band is exactly what the adapter keeps (a 4.5 m wall is inside
        // the scan and invisible to these brains).
        const half = (TOF_FOV_DEG * Math.PI) / 360;
        const sectorR = Math.min(TOF_MAX_RANGE_M, max) * pxPerM;
        ctx.beginPath();
        ctx.moveTo(c, c);
        // Canvas angles grow clockwise from +x; the plot is forward-UP, so the
        // sector spans -90°±half in canvas terms.
        ctx.arc(c, c, sectorR, -Math.PI / 2 - half, -Math.PI / 2 + half);
        ctx.closePath();
        ctx.fillStyle = "rgba(67,194,184,0.10)";
        ctx.fill();
        ctx.strokeStyle = "rgba(67,194,184,0.35)";
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
        // Range rings every metre, labelled up the forward axis.
        ctx.font = "9px ui-monospace, Menlo, monospace";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        for (let m = 1; m <= Math.ceil(max); m++) {
          const r = Math.min(m, max) * pxPerM;
          ctx.beginPath();
          ctx.arc(c, c, r, 0, Math.PI * 2);
          ctx.strokeStyle = m === Math.ceil(max) ? "rgba(255,255,255,0.16)" : "rgba(255,255,255,0.07)";
          ctx.stroke();
          ctx.fillStyle = "rgba(154,165,177,0.75)";
          ctx.fillText(m === Math.ceil(max) ? `${m} m` : `${m}`, c + 2, c - r + 5);
        }
        // The blind disc: nearer than this the device cannot resolve, and a
        // return inside it arrives as a 0 (clipped and marked invalid).
        ctx.beginPath();
        ctx.arc(c, c, LIDAR_MIN_RANGE_M * pxPerM, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(242,182,50,0.45)";
        ctx.setLineDash([2, 2]);
        ctx.stroke();
        ctx.setLineDash([]);
        // -- the robot, where it is relative to the aperture ---------------
        // `mount` is the laser IN THE BASE frame, so in the laser's frame the
        // base origin is at -mount: 76 mm AHEAD of this plot's centre on MARS.
        const mx = scan?.mount ? scan.mount[0] : 0;
        const my = scan?.mount ? scan.mount[1] : 0;
        const base = lidarPolar(Math.atan2(-my, -mx), Math.hypot(mx, my), max, PLOT);
        const foot = robotFootprintM(d?.robot);
        if (foot > 0) {
          // What the adapter throws away as the robot looking at itself.
          ctx.beginPath();
          ctx.arc(base.x, base.y, foot * pxPerM, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(154,165,177,0.30)";
          ctx.setLineDash([2, 3]);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        ctx.beginPath();                            // the nose, up
        ctx.moveTo(base.x, base.y - 7);
        ctx.lineTo(base.x - 4, base.y + 4);
        ctx.lineTo(base.x + 4, base.y + 4);
        ctx.closePath();
        ctx.fillStyle = "rgba(67,194,184,0.85)";
        ctx.fill();
        ctx.strokeStyle = "rgba(233,237,241,0.55)";  // the aperture itself
        ctx.beginPath();
        ctx.moveTo(c - 3, c);
        ctx.lineTo(c + 3, c);
        ctx.moveTo(c, c - 3);
        ctx.lineTo(c, c + 3);
        ctx.stroke();
        // -- the returns ---------------------------------------------------
        const tof = scan ? tofFromLidar(scan, { footprintM: foot }) : null;
        const picks = new Set((tof?.cols ?? []).map((x) => x.ray).filter((r) => r >= 0));
        let hits = 0;
        if (scan) {
          for (const p of lidarPlotPoints(scan, PLOT)) {
            const mm = scan.mm[p.i];
            const miss = p.r >= max - 1e-9;
            ctx.fillStyle = miss ? "rgba(154,165,177,0.20)" : depthColor(mm, max * 1000);
            const s = picks.has(p.i) ? 2.6 : 1.7;
            ctx.fillRect(p.x - s / 2, p.y - s / 2, s, s);
            if (picks.has(p.i)) {
              // The eight returns the brain is steering on, ringed.
              ctx.beginPath();
              ctx.arc(p.x, p.y, 3.4, 0, Math.PI * 2);
              ctx.strokeStyle = "rgba(233,237,241,0.75)";
              ctx.stroke();
            }
            if (!miss) hits++;
          }
        }
        // The nearest real contact, as a spoke — the one number a driver cares
        // about, and the plot should show WHERE it is, not just say it.
        const near = scan ? lidarNearest(scan) : null;
        if (near) {
          const q = lidarPolar(near.bearing, near.rangeM, max, PLOT);
          ctx.beginPath();
          ctx.moveTo(c, c);
          ctx.lineTo(q.x, q.y);
          ctx.strokeStyle = "rgba(242,182,50,0.55)";
          ctx.stroke();
        }
        // -- the strip: the eight columns the brain reads ------------------
        for (let k = 0; k < TOF_COLS; k++) {
          const cell = cells.current[k];
          if (!cell) continue;
          const col = tof?.cols[k];
          // The ToF's own 4 m colour scale, so the strip reads against the
          // duck's 8x8 grid rather than against this plot's 6 m one.
          cell.style.background = col && col.mm ? depthColor(col.mm) : "#1c2026";
        }
        if (strip.current) strip.current.style.opacity = scan ? "1" : "0.35";
        last.current = { tof, near, rays: hits };
      }
      // -- the readouts, every frame (the AGE moves between scans) --------
      const { tof: shown, near: nearest, rays } = last.current;
      if (meta.current) {
        let text: string;
        let color = DIM;
        let title = "";
        if (!d) text = "select a robot with a scanner";
        else if (!scan) text = `${d.id} · no lidar scan yet`;
        else {
          const fresh = scanFreshness(scan.age ?? 0);
          color = fresh.color;
          title = fresh.title;
          const hoverTxt =
            hover !== null && shown
              ? ` · col ${hover}: ${shown.cols[hover].mm ? `${(shown.cols[hover].mm / 1000).toFixed(2)} m at ${deg(shown.cols[hover].bearing ?? 0)}` : "no target"}`
              : nearest
                ? ` · nearest ${deg(nearest.bearing)} ${nearest.rangeM.toFixed(2)} m`
                : " · nothing in range";
          text = `${d.id} · ${d.lidar ?? "?"} · ${rays}/${scan.mm.length} rays · ${fresh.text}${hoverTxt}`;
        }
        if (meta.current.textContent !== text) meta.current.textContent = text;
        meta.current.title = title;
        meta.current.style.color = color;
      }
      if (note.current) {
        // The contrast that explains half of this robot's misbehaviour: the
        // scan is one horizontal slice, so everything below its plane — the
        // basket's 6 cm rim, the toys, a step — is invisible to it, and the
        // head camera is the only sense that has them. MEASURED as the cause
        // of a lidar-only wander parking on the basket (mars-roadmap 3b).
        const plane = scan?.mount ? scan.mount[2] : null;
        const text = plane === null
          ? "one horizontal slice: nothing above or below the scan plane exists"
          : `one slice at ${plane.toFixed(2)} m: the basket's 6 cm rim and the toys are BELOW it — only the head camera has them (see “sees:”)`;
        if (note.current.textContent !== text) note.current.textContent = text;
      }
    };
    raf = requestAnimationFrame(paint);
    return () => cancelAnimationFrame(raf);
  }, [client, duckId, hover]);
  return (
    <div>
      <canvas
        ref={canvas}
        width={PLOT}
        height={PLOT}
        style={{ width: PLOT, height: PLOT, borderRadius: 6, border: "1px solid rgba(255,255,255,.08)", display: "block" }}
        title="the 360° scan in the ROBOT's frame — nose up, its left to the left. Rings are metres; the amber disc is the 0.15 m the device cannot resolve inside; the teal wedge is the 45° front sector its brains read, out to the ToF's own 4 m; the dashed circle is the footprint whose returns are dropped as the robot's own arm. Centred on the LASER, with the chassis drawn 76 mm ahead of it, because the ranges are the laser's."
      />
      <div
        ref={strip}
        style={{ display: "grid", gridTemplateColumns: `repeat(${TOF_COLS}, 1fr)`, gap: 2, width: PLOT, height: 16, marginTop: 4 }}
        onMouseLeave={() => setHover(null)}
        title="the 8 bearing columns the front sector is binned into — the 8×8 frame `wander` and `follow` actually steer on (sensors/lidar.tof_from_lidar). Column 0 is the robot's LEFT. Every row of that frame is this row: a planar scan has no elevation."
      >
        {Array.from({ length: TOF_COLS }, (_, i) => (
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
      <div ref={meta} style={{ marginTop: 6, color: DIM, minHeight: 14 }} />
      <div ref={note} style={{ marginTop: 2, color: "#7c8796", fontSize: 10, lineHeight: 1.35 }} />
    </div>
  );
}

/** The claw as a reading: the constraint torque at joint6, the 1.0 N·m hold
 *  threshold marked on the bar, and a badge when the lab says it is holding.
 *  Nothing at all when the frame carries no gripper — see
 *  `lib/sim.GripperPayload`, which no lab sends yet. */
export function GripperBlock({ g, holding }: { g: GripperPayload; holding?: string | null }) {
  const b = gripperBar(g);
  return (
    <div style={{ marginTop: 6 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, color: DIM }}>
        <span>gripper</span>
        <span style={{ color: "#c9d0d8" }}>{g.load.toFixed(2)} N·m</span>
        {b.holding && (
          <span
            style={{ background: TEAL, color: "#101216", borderRadius: 3, padding: "0 4px", fontSize: 10 }}
            title={b.inferred
              ? "torque past the hold threshold. The lab sends no `holding` flag, so this is the TORQUE alone — closing on air reads 0.0 N·m, but a blade stopped short by nothing at all could still load up."
              : `holding${holding ? ` ${holding}` : ""} — the driver's own predicate: torque past ${b.hold} N·m AND a blade contact with something that is not the robot`}
          >
            HOLDING{holding ? ` ${holding}` : ""}
          </span>
        )}
      </div>
      <div style={{ position: "relative", height: 8, background: "#1c2026", borderRadius: 2, marginTop: 3 }}>
        <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${(b.frac * 100).toFixed(1)}%`, background: b.holding ? TEAL : AMBER, borderRadius: 2 }} />
        {/* the hold threshold, where "closed" becomes "holding" */}
        <div
          style={{ position: "absolute", left: `${(b.markFrac * 100).toFixed(1)}%`, top: -2, bottom: -2, width: 1, background: "#e9edf1" }}
          title={`${b.hold} N·m — the hold threshold (mars.HOLD_LOAD_NM); full scale is the servo's own ${b.limit} N·m clamp`}
        />
      </div>
    </div>
  );
}

/** The arm's achieved joint angles as small bars, with the commanded target
 *  marked where the frame carries one — the gap between the two IS the
 *  servo's structural sag. Nothing when the frame carries no arm block. */
export function ArmBlock({ arm }: { arm: ArmPayload }) {
  const names = Object.keys(arm.q);
  if (!names.length) return null;
  return (
    <div style={{ marginTop: 6 }}>
      <div style={{ color: DIM }}>arm</div>
      {names.map((n) => {
        const lim = arm.limits?.[n];
        const q = arm.q[n];
        const cmd = arm.cmd?.[n];
        return (
          <div key={n} style={{ display: "flex", alignItems: "center", gap: 4, marginTop: 2 }}>
            <span style={{ color: "#7c8796", width: "6ch", fontSize: 10 }}>{n}</span>
            <div style={{ position: "relative", flex: 1, height: 6, background: "#1c2026", borderRadius: 2 }}>
              <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${(armBar(q, lim) * 100).toFixed(1)}%`, background: TEAL, borderRadius: 2 }} />
              {cmd !== undefined && (
                <div
                  style={{ position: "absolute", left: `${(armBar(cmd, lim) * 100).toFixed(1)}%`, top: -2, bottom: -2, width: 1, background: AMBER }}
                  title={`commanded ${cmd.toFixed(3)} rad — the gap to ${q.toFixed(3)} is the servo's sag`}
                />
              )}
            </div>
            <span style={{ color: "#c9d0d8", width: "7ch", textAlign: "right", fontSize: 10 }}>{q.toFixed(2)}</span>
          </div>
        );
      })}
    </div>
  );
}

/**
 * The scan in the ROOM: every ray from the aperture in the base's heading
 * frame, hits as short ticks coloured by range, misses faint out to max
 * range, and the front sector its brains read drawn as a tinted fan with the
 * eight winning rays brightest.
 *
 * The duck's answer to this is `TofOverlay`'s cone of zone dots; this is the
 * same idea for a sensor that measures the whole turn. One BufferGeometry
 * rewritten per frame, no per-ray meshes (the first /sim lost its WebGL
 * context to 560 meshes — duck-viewer/AGENTS.md).
 *
 * The rays leave the `base_laser` body's OWN pose, which is why no mount
 * offset appears here: the 76 mm is baked into where that body is. The
 * payload's `mount` is only needed to re-read a bearing in the BASE frame,
 * which is the adapter's job (`lib/lidar.tofFromLidar`) and shows up here
 * only through which rays the sector claims.
 */
export function LidarOverlay({
  client,
  robotScenes,
  enabled,
}: {
  client: SimClient;
  /** Mesh sets by robot id — the same map SimViewer hands the stage. Only the
   *  body-name LIST is used here, to find the scanner's pose in the frame. */
  robotScenes: Record<string, Scene>;
  enabled: boolean;
}) {
  const lines = useRef<THREE.LineSegments>(null);
  useEffect(() => {
    lines.current?.layers.set(OVERLAY_LAYER);
  }, []);
  const laserIdx = useMemo(() => {
    const out: Record<string, number> = {};
    for (const [id, sc] of Object.entries(robotScenes)) out[id] = sc.bodies.indexOf(LIDAR_MOUNT_BODY);
    return out;
  }, [robotScenes]);
  const pos = useMemo(() => new Float32Array(MAX_LIDAR_SEG * 2 * 3), []);
  const colors = useMemo(() => new Float32Array(MAX_LIDAR_SEG * 2 * 3), []);
  const col = useMemo(() => new THREE.Color(), []);
  useFrame(() => {
    const ls = lines.current;
    if (!ls) return;
    const f = client.frame;
    let n = 0;
    const seg = (
      o: [number, number, number],
      dir: [number, number, number],
      from: number,
      to: number,
      color: string,
    ) => {
      if (n >= MAX_LIDAR_SEG) return;
      const k = n * 6;
      pos[k] = o[0] + dir[0] * from;
      pos[k + 1] = o[1] + dir[1] * from;
      pos[k + 2] = o[2] + dir[2] * from;
      pos[k + 3] = o[0] + dir[0] * to;
      pos[k + 4] = o[1] + dir[1] * to;
      pos[k + 5] = o[2] + dir[2] * to;
      col.set(color);
      col.toArray(colors, k);
      col.toArray(colors, k + 3);
      n++;
    };
    if (f && enabled) {
      const sel = getSelectedDuck();
      for (const d of f.ducks) {
        const scan = d.sensors?.lidar;
        if (!scan || (sel && d.id !== sel)) continue;
        const idx = laserIdx[d.robot || "microduck"];
        const pose = idx === undefined || idx < 0 ? undefined : d.bodies[idx];
        if (!pose) continue;
        const origin: [number, number, number] = [pose[0], pose[1], pose[2]];
        const q = [pose[3], pose[4], pose[5], pose[6]];
        const max = lidarMaxRange(scan);
        // The sector, computed exactly as the brain's frame is, so the fan in
        // the room is the fan in the panel's strip — including the footprint
        // returns both of them drop.
        const tof = tofFromLidar(scan, { footprintM: robotFootprintM(d.robot) });
        const picks = new Set(tof.cols.map((x) => x.ray).filter((r) => r >= 0));
        for (let i = 0; i < scan.mm.length; i++) {
          const mm = scan.mm[i];
          if (!mm) continue; // 0 = no reading: nothing to draw, not a contact
          const a = scan.a0 + i * scan.da;
          const dir = quatRotate(q, [Math.cos(a), Math.sin(a), 0]);
          const r = mm / 1000;
          const miss = r >= max - 1e-9;
          if (miss) {
            // "Nothing within range" is information, drawn as the faintest
            // thing on the stage rather than left out.
            seg(origin, dir, 0, max, "#1b2026");
            continue;
          }
          if (picks.has(i)) seg(origin, dir, 0, r, "#43c2b8");
          else if (tof.rayCol[i] >= 0) seg(origin, dir, 0, r, "#1f4d4a");
          seg(origin, dir, Math.max(0, r - HIT_TICK_M), r, depthColor(mm, max * 1000));
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

export type { LidarPayload };
