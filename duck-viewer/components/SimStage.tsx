"use client";

// The /sim stage: everything in the room that is NOT a duck — floor, walls,
// goals, basket, furniture, toys, balls, persons — and the cheap tricks that
// make a flat WebGL room read as a place. All of it stays inside the stage
// rules the README spells out (the first viewer lost its WebGL context):
//
//   - NO shadow maps. Grounding comes from ONE instanced mesh of soft
//     multiply-blended contact blobs under everything that moves, soft
//     rectangles under furniture, and a darkening strip along every wall base.
//   - No image assets, no fetches: every texture is painted once on a canvas
//     (a mown pitch with its markings, oak planks, wicker, a rug, a 32-panel
//     ball skin) and shared; scenario-sized ones are rebuilt only when the
//     floor changes, not per edit click.
//   - Specular life comes from a procedural RoomEnvironment PMREM on
//     scene.environment — one texture, computed once, no lights added.
//   - No per-frame React state: the frame loop writes matrices and lerps
//     positions, exactly like Duck.tsx.
//
// MuJoCo is Z-up; these components render inside the page's -90° X group,
// so positions here are world (x, y, z) straight from the lab.

import { useEffect, useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { PICKABLE_COLORS, PICKABLE_SIZES, type Scenario, type SimClient } from "@/lib/sim";

// -- palette -----------------------------------------------------------------
// Kept close to the page's UI accents (amber / teal) so the stage and the
// overlays read as one thing.
const PLASTER = "#d8d1c4";
const PLASTER_CAP = "#c3bcaf";
const BASEBOARD = "#efe9de";
const BOARD = "#ebe9e2";
const BOARD_STRIPE = "#e8b24a";
const PERSON = "#4f7fc4";
const PERSON_NOSE = "#a9c8ff";

// -- small helpers -------------------------------------------------------------

/** Deterministic PRNG so a texture is the same every build (and in tests). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function paintedTexture(
  w: number,
  h: number,
  paint: (ctx: CanvasRenderingContext2D, w: number, h: number) => void,
  opts: { repeat?: [number, number]; linear?: boolean } = {}
): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  if (ctx) paint(ctx, w, h);
  const t = new THREE.CanvasTexture(c);
  if (!opts.linear) t.colorSpace = THREE.SRGBColorSpace;
  if (opts.repeat) {
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(opts.repeat[0], opts.repeat[1]);
  }
  t.anisotropy = 8; // three clamps to the device maximum
  return t;
}

function roundedRectPath(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.arcTo(x + w, y, x + w, y + rr, rr);
  ctx.lineTo(x + w, y + h - rr);
  ctx.arcTo(x + w, y + h, x + w - rr, y + h, rr);
  ctx.lineTo(x + rr, y + h);
  ctx.arcTo(x, y + h, x, y + h - rr, rr);
  ctx.lineTo(x, y + rr);
  ctx.arcTo(x, y, x + rr, y, rr);
  ctx.closePath();
}

/** Axis-aligned extent of the walls: the room proper, inside the floor's apron. */
interface Bounds { minX: number; maxX: number; minY: number; maxY: number }
function roomBounds(s: Scenario | null): Bounds | null {
  if (!s || !s.walls.length) return null;
  const b: Bounds = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity };
  for (const w of s.walls) {
    for (const p of [w.from, w.to]) {
      b.minX = Math.min(b.minX, p[0]);
      b.maxX = Math.max(b.maxX, p[0]);
      b.minY = Math.min(b.minY, p[1]);
      b.maxY = Math.max(b.maxY, p[1]);
    }
  }
  return b;
}
function isPitch(s: Scenario | null): boolean {
  return !!s && (s.goal_width ?? 0) > 0;
}

// -- shared textures (scenario-independent, painted once per page) -------------

let _planks: THREE.CanvasTexture | null = null;
let _wicker: THREE.CanvasTexture | null = null;
let _ball: THREE.CanvasTexture | null = null;
let _blob: THREE.CanvasTexture | null = null;
let _softRect: THREE.CanvasTexture | null = null;
let _wallBase: THREE.CanvasTexture | null = null;
let _net: THREE.CanvasTexture | null = null;
let _rug: THREE.CanvasTexture | null = null;

/** Oak planks: one 0.8 m tile, 8 boards across, staggered joints, wavy
 *  grain that wraps seamlessly (integer wave counts per tile). */
function plankTexture(): THREE.CanvasTexture {
  if (_planks) return _planks;
  _planks = paintedTexture(512, 512, (ctx, S) => {
    const rng = mulberry32(3);
    const rows = 8;
    const ph = S / rows;
    const tones = ["#b98a5e", "#ad7f55", "#c4956a", "#a67750", "#bd8c60", "#b2845a", "#c99d70"];
    ctx.fillStyle = "#a87a52";
    ctx.fillRect(0, 0, S, S);
    for (let r = 0; r < rows; r++) {
      const y0 = r * ph;
      const off = ((r * 0.37 + 0.11) % 1) * S;
      const joints = [off, (off + S / 2) % S].sort((a, b) => a - b);
      const segs: [number, number][] = [[0, joints[0]], [joints[0], joints[1]], [joints[1], S]];
      const toneA = tones[Math.floor(rng() * tones.length)];
      const toneB = tones[Math.floor(rng() * tones.length)];
      segs.forEach(([x0, x1], i) => {
        ctx.fillStyle = i === 1 ? toneB : toneA;   // the outer two pieces are one board across the seam
        ctx.fillRect(x0, y0, x1 - x0, ph);
      });
      for (let g = 0; g < 28; g++) {
        const y = y0 + rng() * ph;
        ctx.strokeStyle = rng() < 0.5 ? `rgba(70,40,20,${(0.07 + rng() * 0.1).toFixed(3)})` : `rgba(255,225,190,${(0.05 + rng() * 0.08).toFixed(3)})`;
        ctx.lineWidth = 0.6 + rng() * 1.2;
        const amp = 1 + rng() * 2, f = 1 + Math.floor(rng() * 3), phase = rng() * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(0, y + Math.sin(phase) * amp);
        for (let x = 16; x <= S; x += 16) ctx.lineTo(x, y + Math.sin((x / S) * Math.PI * 2 * f + phase) * amp);
        ctx.stroke();
      }
      ctx.fillStyle = "rgba(40,20,10,0.35)";
      ctx.fillRect(0, y0 + ph - 1.5, S, 1.5);
      ctx.fillStyle = "rgba(255,235,210,0.2)";
      ctx.fillRect(0, y0, S, 1);
      ctx.fillStyle = "rgba(40,20,10,0.45)";
      for (const j of joints) ctx.fillRect(j - 1, y0, 2, ph);
    }
  }, { repeat: [1, 1] });
  return _planks;
}

/** Basket weave: alternating over/under strands, shaded where they dip. */
function wickerTexture(): THREE.CanvasTexture {
  if (_wicker) return _wicker;
  _wicker = paintedTexture(256, 256, (ctx, S) => {
    const rng = mulberry32(5);
    const n = 8, cell = S / n;
    ctx.fillStyle = "#5e4428";
    ctx.fillRect(0, 0, S, S);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const horiz = (i + j) % 2 === 0;
        const x = i * cell, y = j * cell;
        const base = 150 + rng() * 30;
        ctx.fillStyle = `rgb(${Math.round(base + 30)},${Math.round(base - 10)},${Math.round(base - 62)})`;
        ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
        ctx.strokeStyle = "rgba(60,35,15,0.35)";
        ctx.lineWidth = 1;
        for (let k = 1; k < 4; k++) {
          ctx.beginPath();
          if (horiz) { ctx.moveTo(x + 1, y + (k * cell) / 4); ctx.lineTo(x + cell - 1, y + (k * cell) / 4); }
          else { ctx.moveTo(x + (k * cell) / 4, y + 1); ctx.lineTo(x + (k * cell) / 4, y + cell - 1); }
          ctx.stroke();
        }
        const grad = horiz ? ctx.createLinearGradient(x, 0, x + cell, 0) : ctx.createLinearGradient(0, y, 0, y + cell);
        grad.addColorStop(0, "rgba(0,0,0,0.4)");
        grad.addColorStop(0.25, "rgba(0,0,0,0)");
        grad.addColorStop(0.75, "rgba(0,0,0,0)");
        grad.addColorStop(1, "rgba(0,0,0,0.4)");
        ctx.fillStyle = grad;
        ctx.fillRect(x, y, cell, cell);
      }
    }
  }, { repeat: [3, 3] });
  return _wicker;
}

/** A 32-panel ball skin: black patches around the 12 icosahedron vertices,
 *  computed per pixel on the sphere so the poles don't smear. */
function ballTexture(): THREE.CanvasTexture {
  if (_ball) return _ball;
  _ball = paintedTexture(512, 256, (ctx, W, H) => {
    const img = ctx.createImageData(W, H);
    const phi = (1 + Math.sqrt(5)) / 2;
    const verts: number[][] = [];
    for (const s1 of [-1, 1]) for (const s2 of [-1, 1]) verts.push([0, s1, s2 * phi], [s1, s2 * phi, 0], [s2 * phi, 0, s1]);
    for (const v of verts) {
      const n = Math.hypot(v[0], v[1], v[2]);
      v[0] /= n; v[1] /= n; v[2] /= n;
    }
    const cosIn = Math.cos(0.29), cosOut = Math.cos(0.325);
    for (let j = 0; j < H; j++) {
      const lat = (0.5 - (j + 0.5) / H) * Math.PI;
      const cl = Math.cos(lat), sl = Math.sin(lat);
      for (let i = 0; i < W; i++) {
        const lon = ((i + 0.5) / W) * Math.PI * 2;
        const dx = cl * Math.cos(lon), dy = cl * Math.sin(lon), dz = sl;
        let best = -1;
        for (const v of verts) {
          const d = dx * v[0] + dy * v[1] + dz * v[2];
          if (d > best) best = d;
        }
        const t = best >= cosIn ? 1 : best <= cosOut ? 0 : (best - cosOut) / (cosIn - cosOut);
        const c = Math.round(238 + (26 - 238) * t);
        const k = (j * W + i) * 4;
        img.data[k] = c; img.data[k + 1] = c; img.data[k + 2] = c; img.data[k + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  });
  return _ball;
}

/** Radial contact blob: dark centre → white rim, multiply-blended on the floor. */
function blobTexture(): THREE.CanvasTexture {
  if (_blob) return _blob;
  _blob = paintedTexture(128, 128, (ctx, S) => {
    const g = ctx.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2);
    g.addColorStop(0, "rgb(70,70,70)");
    g.addColorStop(0.45, "rgb(150,150,150)");
    g.addColorStop(1, "rgb(255,255,255)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, S, S);
  }, { linear: true });
  return _blob;
}

/** Soft-edged rectangle for furniture footprints (same multiply idea). */
function softRectTexture(): THREE.CanvasTexture {
  if (_softRect) return _softRect;
  _softRect = paintedTexture(128, 128, (ctx, S) => {
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, S, S);
    for (let k = 0; k < 14; k++) {
      const inset = 4 + k * 3;
      roundedRectPath(ctx, inset, inset, S - 2 * inset, S - 2 * inset, 10);
      ctx.fillStyle = "rgba(0,0,0,0.055)";
      ctx.fill();
    }
  }, { linear: true });
  return _softRect;
}

/** Darkening strip across a wall's base (gradient along v, symmetric). */
function wallBaseTexture(): THREE.CanvasTexture {
  if (_wallBase) return _wallBase;
  _wallBase = paintedTexture(4, 64, (ctx, W, H) => {
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, "rgb(255,255,255)");
    g.addColorStop(0.5, "rgb(120,120,120)");
    g.addColorStop(1, "rgb(255,255,255)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
  }, { linear: true });
  return _wallBase;
}

/** Goal net: white cords on transparent, one tile = one mesh cell. */
function netTexture(): THREE.CanvasTexture {
  if (_net) return _net;
  _net = paintedTexture(32, 32, (ctx, S) => {
    ctx.clearRect(0, 0, S, S);
    ctx.strokeStyle = "rgba(255,255,255,0.8)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, 1); ctx.lineTo(S, 1);
    ctx.moveTo(1, 0); ctx.lineTo(1, S);
    ctx.stroke();
  }, { repeat: [1, 1] });
  return _net;
}

/** A flat-woven rug with a border, muted so the map overlay stays legible. */
function rugTexture(): THREE.CanvasTexture {
  if (_rug) return _rug;
  _rug = paintedTexture(512, 512, (ctx, S) => {
    const rng = mulberry32(11);
    ctx.fillStyle = "#8b5e52";
    ctx.fillRect(0, 0, S, S);
    // weave grain
    for (let i = 0; i < 14000; i++) {
      ctx.fillStyle = rng() < 0.5 ? "rgba(40,20,20,0.14)" : "rgba(255,220,200,0.08)";
      ctx.fillRect(rng() * S, rng() * S, 2, 1);
    }
    // border bands
    const band = (inset: number, w: number, color: string) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = w;
      ctx.strokeRect(inset + w / 2, inset + w / 2, S - 2 * inset - w, S - 2 * inset - w);
    };
    band(14, 10, "#d9c7a8");
    band(34, 4, "#d9c7a8");
    band(46, 22, "#6f4a44");
    band(72, 3, "#d9c7a8");
    // a lozenge field in the middle
    ctx.strokeStyle = "rgba(217,199,168,0.55)";
    ctx.lineWidth = 2;
    for (let k = 0; k < 6; k++) {
      const r = 40 + k * 32;
      ctx.beginPath();
      ctx.moveTo(S / 2, S / 2 - r);
      ctx.lineTo(S / 2 + r * 1.35, S / 2);
      ctx.lineTo(S / 2, S / 2 + r);
      ctx.lineTo(S / 2 - r * 1.35, S / 2);
      ctx.closePath();
      ctx.stroke();
    }
    // fringe hint on the short ends
    ctx.fillStyle = "rgba(217,199,168,0.7)";
    for (let x = 6; x < S; x += 8) {
      ctx.fillRect(x, 0, 3, 10);
      ctx.fillRect(x, S - 10, 3, 10);
    }
  });
  return _rug;
}

/** The pitch: mown stripes, a dark apron outside the boards, and the
 *  markings — touchlines, halfway line, centre circle, goal areas, penalty
 *  spots, corner arcs — sized to the room. Rebuilt only when the floor,
 *  the walls' extent or the goal width changes. */
function grassTexture(fx: number, fy: number, b: Bounds | null, gw: number): THREE.CanvasTexture {
  const ppm = 200;
  const W = Math.round(fx * ppm), H = Math.round(fy * ppm);
  return paintedTexture(W, H, (ctx) => {
    const X = (x: number) => (x + fx / 2) * ppm;
    const Y = (y: number) => (fy / 2 - y) * ppm;
    const rng = mulberry32(7);
    ctx.fillStyle = "#3f8b3e";
    ctx.fillRect(0, 0, W, H);
    const stripe = 0.25 * ppm;
    for (let x = 0, i = 0; x < W; x += stripe, i++) {
      ctx.fillStyle = i % 2 ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.07)";
      ctx.fillRect(x, 0, stripe, H);
    }
    for (let i = 0; i < (W * H) / 14; i++) {
      ctx.fillStyle = rng() < 0.5 ? "rgba(15,55,15,0.2)" : "rgba(190,235,140,0.1)";
      ctx.fillRect(rng() * W, rng() * H, 1 + rng() * 1.5, 1 + rng() * 2.5);
    }
    if (!b) return;
    // apron: dark rubber outside the boards
    ctx.fillStyle = "#2c2f34";
    ctx.beginPath();
    ctx.rect(0, 0, W, H);
    ctx.rect(X(b.minX), Y(b.maxY), (b.maxX - b.minX) * ppm, (b.maxY - b.minY) * ppm);
    ctx.fill("evenodd");
    ctx.fillStyle = "rgba(255,255,255,0.06)";
    for (let i = 0; i < (W * H) / 60; i++) {
      const x = rng() * W, y = rng() * H;
      const wx = x / ppm - fx / 2, wy = fy / 2 - y / ppm;
      if (wx > b.minX && wx < b.maxX && wy > b.minY && wy < b.maxY) continue;
      ctx.fillRect(x, y, 1.5, 1.5);
    }
    // markings
    const inset = 0.06;
    const L = b.minX + inset, R = b.maxX - inset, T = b.maxY - inset, Bo = b.minY + inset;
    const cx = (b.minX + b.maxX) / 2, cy = (b.minY + b.maxY) / 2;
    ctx.strokeStyle = "rgba(255,255,255,0.92)";
    ctx.fillStyle = "rgba(255,255,255,0.92)";
    ctx.lineWidth = 0.02 * ppm;
    ctx.strokeRect(X(L), Y(T), (R - L) * ppm, (T - Bo) * ppm);
    ctx.beginPath();
    ctx.moveTo(X(cx), Y(T));
    ctx.lineTo(X(cx), Y(Bo));
    ctx.stroke();
    const cr = Math.min(0.32, (T - Bo) * 0.18);
    ctx.beginPath();
    ctx.arc(X(cx), Y(cy), cr * ppm, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(X(cx), Y(cy), 0.02 * ppm, 0, Math.PI * 2);
    ctx.fill();
    if (gw > 0) {
      const gaW = Math.min(gw + 0.5, T - Bo - 0.1);
      const gaD = Math.min(0.32, (R - L) * 0.2);
      ctx.strokeRect(X(L), Y(cy + gaW / 2), gaD * ppm, gaW * ppm);
      ctx.strokeRect(X(R - gaD), Y(cy + gaW / 2), gaD * ppm, gaW * ppm);
      for (const px of [L + gaD + 0.16, R - gaD - 0.16]) {
        ctx.beginPath();
        ctx.arc(X(px), Y(cy), 0.015 * ppm, 0, Math.PI * 2);
        ctx.fill();
      }
      // the goal mouths: a worn, lighter patch between the posts
      ctx.fillStyle = "rgba(255,255,255,0.08)";
      ctx.fillRect(X(b.minX), Y(cy + gw / 2), 0.12 * ppm, gw * ppm);
      ctx.fillRect(X(b.maxX - 0.12), Y(cy + gw / 2), 0.12 * ppm, gw * ppm);
    }
    const ar = 0.08 * ppm;
    for (const [x, y, a0] of [[L, T, 0], [R, T, Math.PI / 2], [R, Bo, Math.PI], [L, Bo, -Math.PI / 2]] as const) {
      ctx.beginPath();
      ctx.arc(X(x), Y(y), ar, a0, a0 + Math.PI / 2);
      ctx.stroke();
    }
  });
}

// -- environment -----------------------------------------------------------------

/** A procedural studio environment on scene.environment: specular highlights
 *  on the ducks' shells and the ball, a little sheen on the planks. One
 *  PMREM, computed once; no HDR fetch, no extra lights. */
export function StageEnvironment({ intensity = 0.55 }: { intensity?: number }) {
  const { gl, scene } = useThree();
  useEffect(() => {
    const pmrem = new THREE.PMREMGenerator(gl);
    const env = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();
    scene.environment = env;
    scene.environmentIntensity = intensity;
    return () => {
      if (scene.environment === env) scene.environment = null;
      env.dispose();
    };
  }, [gl, scene, intensity]);
  return null;
}

// -- statics -----------------------------------------------------------------------

const GOAL_H = 0.22;   // posts + crossbar height: under the 0.3 m boards, over a duck's head

function boxGeometryKey(size: [number, number, number]): string {
  return size.map((v) => v.toFixed(4)).join("x");
}
function roundedBox(size: [number, number, number], bevel = 0.08): RoundedBoxGeometry {
  const r = Math.max(0.004, Math.min(0.02, Math.min(size[0], size[1], size[2]) * bevel));
  return new RoundedBoxGeometry(size[0], size[1], size[2], 2, r);
}

/** Walls, floor, goals, basket and furniture from the loaded scenario. */
export function Statics({ scenario }: { scenario: Scenario | null }) {
  const pitch = isPitch(scenario);
  const gw = pitch ? scenario!.goal_width! : 0;
  const fx = scenario?.floor.size[0] ?? 0;
  const fy = scenario?.floor.size[1] ?? 0;
  // Bounds as a string key so the floor texture survives edit clicks that
  // don't move a wall.
  const bKey = useMemo(() => {
    const b = roomBounds(scenario);
    return b ? [b.minX, b.maxX, b.minY, b.maxY].map((v) => v.toFixed(3)).join(",") : "";
  }, [scenario]);
  const bounds = useMemo<Bounds | null>(() => {
    if (!bKey) return null;
    const [minX, maxX, minY, maxY] = bKey.split(",").map(Number);
    return { minX, maxX, minY, maxY };
  }, [bKey]);

  const floorTex = useMemo(() => {
    if (!fx || !fy) return null;
    if (pitch) return grassTexture(fx, fy, bounds, gw);
    const t = plankTexture().clone();   // shares the image; only the repeat differs
    t.repeat.set(fx / 0.8, fy / 0.8);
    return t;
  }, [fx, fy, pitch, bounds, gw]);
  useEffect(() => () => void floorTex?.dispose(), [floorTex]);

  // One rounded geometry per distinct furniture size.
  const furniture = useMemo(() => {
    const geos = new Map<string, RoundedBoxGeometry>();
    for (const b of scenario?.boxes ?? []) {
      if (b.mass > 0) continue;
      const k = boxGeometryKey(b.size);
      if (!geos.has(k)) geos.set(k, roundedBox(b.size));
    }
    return geos;
  }, [scenario]);
  useEffect(() => () => furniture.forEach((g) => g.dispose()), [furniture]);

  // Goal nets: 5 cm cells, so the repeat depends on the goal width.
  const netTex = useMemo(() => {
    if (!pitch) return null;
    const t = netTexture().clone();
    t.repeat.set(gw / 0.05, GOAL_H / 0.05);
    return t;
  }, [pitch, gw]);
  useEffect(() => () => void netTex?.dispose(), [netTex]);

  if (!scenario || !floorTex) return null;
  const roomW = bounds ? bounds.maxX - bounds.minX : 0;
  const roomH = bounds ? bounds.maxY - bounds.minY : 0;
  const rug = !pitch && bounds && roomW > 1.2 && roomH > 1.2;
  const hx = fx / 2 - 0.25;   // where the World puts the goal lines (arena.py)
  return (
    <group>
      <mesh position={[0, 0, -0.001]}>
        <planeGeometry args={[fx, fy]} />
        <meshStandardMaterial map={floorTex} roughness={pitch ? 0.92 : 0.62} metalness={0} envMapIntensity={pitch ? 0.3 : 0.8} />
      </mesh>
      {rug && (
        <mesh position={[(bounds.minX + bounds.maxX) / 2, (bounds.minY + bounds.maxY) / 2, 0.0003]}>
          <planeGeometry args={[Math.min(roomW * 0.44, 1.6), Math.min(roomH * 0.5, 1.2)]} />
          <meshStandardMaterial map={rugTexture()} roughness={1} metalness={0} envMapIntensity={0.15} />
        </mesh>
      )}
      {scenario.walls.map((w, i) => {
        const dx = w.to[0] - w.from[0];
        const dy = w.to[1] - w.from[1];
        const len = Math.hypot(dx, dy);
        const yaw = Math.atan2(dy, dx);
        const cx = (w.from[0] + w.to[0]) / 2;
        const cy = (w.from[1] + w.to[1]) / 2;
        return (
          <group key={`w${i}`} position={[cx, cy, 0]} rotation={[0, 0, yaw]}>
            {/* contact darkening across the base, both sides */}
            <mesh position={[0, 0, 0.0012]}>
              <planeGeometry args={[len + 0.16, w.thickness + 0.18]} />
              <meshBasicMaterial map={wallBaseTexture()} blending={THREE.MultiplyBlending} premultipliedAlpha transparent depthWrite={false} />
            </mesh>
            <mesh position={[0, 0, w.height / 2]}>
              <boxGeometry args={[len, w.thickness, w.height]} />
              <meshStandardMaterial color={pitch ? BOARD : PLASTER} roughness={pitch ? 0.55 : 0.95} metalness={0} envMapIntensity={pitch ? 0.6 : 0.25} />
            </mesh>
            {pitch ? (
              <mesh position={[0, 0, Math.min(w.height * 0.62, w.height - 0.03)]}>
                <boxGeometry args={[len, w.thickness + 0.004, 0.03]} />
                <meshStandardMaterial color={BOARD_STRIPE} roughness={0.5} />
              </mesh>
            ) : (
              <mesh position={[0, 0, 0.018]}>
                <boxGeometry args={[len + 0.012, w.thickness + 0.014, 0.036]} />
                <meshStandardMaterial color={BASEBOARD} roughness={0.6} envMapIntensity={0.5} />
              </mesh>
            )}
            <mesh position={[0, 0, w.height - 0.005]}>
              <boxGeometry args={[len + 0.012, w.thickness + 0.016, 0.01]} />
              <meshStandardMaterial color={pitch ? "#f6f5f0" : PLASTER_CAP} roughness={0.7} />
            </mesh>
          </group>
        );
      })}
      {pitch &&
        netTex &&
        [-1, 1].map((s) => {
          const thick = scenario.walls[0]?.thickness ?? 0.02;
          const h = GOAL_H;
          return (
            <group key={`goal${s}`} position={[s * hx, 0, 0]}>
              {[-1, 1].map((q) => (
                <mesh key={q} position={[-s * (thick / 2 + 0.012), (q * gw) / 2, h / 2]} rotation={[Math.PI / 2, 0, 0]}>
                  <cylinderGeometry args={[0.012, 0.012, h, 12]} />
                  <meshStandardMaterial color="#f4f4f0" roughness={0.35} metalness={0.1} />
                </mesh>
              ))}
              <mesh position={[-s * (thick / 2 + 0.012), 0, h]}>
                <cylinderGeometry args={[0.012, 0.012, gw + 0.024, 12]} />
                <meshStandardMaterial color="#f4f4f0" roughness={0.35} metalness={0.1} />
              </mesh>
              {/* XYZ Euler: Ry then Rx takes the plane's width to world Y and its height to Z */}
              <mesh position={[-s * (thick / 2 + 0.003), 0, h / 2]} rotation={[Math.PI / 2, Math.PI / 2, 0]}>
                <planeGeometry args={[gw, h]} />
                <meshBasicMaterial map={netTex} transparent side={THREE.DoubleSide} depthWrite={false} toneMapped={false} />
              </mesh>
            </group>
          );
        })}
      {scenario.basket && (
        <group position={[scenario.basket.pos[0], scenario.basket.pos[1], 0]}>
          <mesh position={[0, 0, 0.0014]}>
            <planeGeometry args={[scenario.basket.size[0] * 1.35, scenario.basket.size[1] * 1.35]} />
            <meshBasicMaterial map={softRectTexture()} blending={THREE.MultiplyBlending} premultipliedAlpha transparent depthWrite={false} />
          </mesh>
          <mesh position={[0, 0, 0.006]}>
            <boxGeometry args={[scenario.basket.size[0], scenario.basket.size[1], 0.012]} />
            <meshStandardMaterial map={wickerTexture()} roughness={0.9} />
          </mesh>
          {[
            [0, -scenario.basket.size[1] / 2, scenario.basket.size[0], 0.012],
            [0, scenario.basket.size[1] / 2, scenario.basket.size[0], 0.012],
            [-scenario.basket.size[0] / 2, 0, 0.012, scenario.basket.size[1]],
            [scenario.basket.size[0] / 2, 0, 0.012, scenario.basket.size[1]],
          ].map(([x, y, sx, sy], i) => (
            <group key={i} position={[x, y, 0]}>
              <mesh position={[0, 0, scenario.basket!.rim / 2]}>
                <boxGeometry args={[sx, sy, scenario.basket!.rim]} />
                <meshStandardMaterial map={wickerTexture()} roughness={0.9} />
              </mesh>
              <mesh position={[0, 0, scenario.basket!.rim]}>
                <boxGeometry args={[sx + 0.008, sy + 0.008, 0.008]} />
                <meshStandardMaterial color="#5b4027" roughness={0.8} />
              </mesh>
            </group>
          ))}
        </group>
      )}
      {scenario.boxes.map((b, i) =>
        b.mass > 0 ? null : (
          <group key={`b${i}`} position={[b.pos[0], b.pos[1], 0]} rotation={[0, 0, b.yaw]}>
            <mesh position={[0, 0, 0.0014]}>
              <planeGeometry args={[b.size[0] * 1.3 + 0.04, b.size[1] * 1.3 + 0.04]} />
              <meshBasicMaterial map={softRectTexture()} blending={THREE.MultiplyBlending} premultipliedAlpha transparent depthWrite={false} />
            </mesh>
            <mesh position={[0, 0, b.pos[2]]} geometry={furniture.get(boxGeometryKey(b.size))}>
              <meshStandardMaterial color={new THREE.Color().setRGB(b.rgba[0], b.rgba[1], b.rgba[2], THREE.SRGBColorSpace)} roughness={0.72} metalness={0} envMapIntensity={0.6} />
            </mesh>
          </group>
        )
      )}
    </group>
  );
}

// -- dynamics -----------------------------------------------------------------------

/** One geometry per toy kind: a studded 2×4 brick, a bevelled block, a
 *  rolled sock (a box with big radii). Built once, shared by every toy. */
const toyGeometries = new Map<string, THREE.BufferGeometry>();
function toyGeometry(kind: string): THREE.BufferGeometry {
  const cached = toyGeometries.get(kind);
  if (cached) return cached;
  const size = PICKABLE_SIZES[kind] ?? [0.03, 0.03, 0.03];
  let g: THREE.BufferGeometry;
  if (kind === "brick") {
    const body = new RoundedBoxGeometry(size[0], size[1], size[2], 1, 0.0008);
    const parts: THREE.BufferGeometry[] = [body];
    const stud = new THREE.CylinderGeometry(0.0024, 0.0024, 0.0017, 10);
    stud.rotateX(Math.PI / 2);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 2; j++) {
        const s = stud.clone();
        s.translate((i - 1.5) * 0.008, (j - 0.5) * 0.008, size[2] / 2 + 0.00085);
        parts.push(s);
      }
    }
    stud.dispose();
    // mergeGeometries wants every part indexed or none: flatten them all.
    const flat = parts.map((p) => (p.index ? p.toNonIndexed() : p));
    g = mergeGeometries(flat, false) ?? new THREE.BoxGeometry(size[0], size[1], size[2]);
    parts.forEach((p) => p.dispose());
    flat.forEach((p) => p.dispose());
  } else if (kind === "sock") {
    g = new RoundedBoxGeometry(size[0], size[1], size[2], 3, Math.min(size[1], size[2]) * 0.45);
  } else {
    g = new RoundedBoxGeometry(size[0], size[1], size[2], 2, 0.004);
  }
  toyGeometries.set(kind, g);
  return g;
}

const MAX_BLOBS = 192;
const Z_BLOB = 0.0016;

/** Free objects (balls, boxes with mass, toys, persons) posed from the frame
 *  stream, plus the instanced contact blobs under them and under every duck. */
export function Dynamics({ scenario, client }: { scenario: Scenario | null; client: SimClient }) {
  const refs = useRef(new Map<string, THREE.Group>());
  const blobs = useRef<THREE.InstancedMesh>(null);
  const tmpP = useMemo(() => new THREE.Vector3(), []);
  const tmpQ = useMemo(() => new THREE.Quaternion(), []);
  const m = useMemo(() => new THREE.Matrix4(), []);
  const col = useMemo(() => new THREE.Color(), []);
  // Multiply-blended blob whose per-instance colour is a STRENGTH, not a
  // tint: the fragment lerps the texel toward white by it, so a faint blob
  // still fades to nothing at its rim instead of drawing a grey square.
  const blobMat = useMemo(() => {
    const mat = new THREE.MeshBasicMaterial({ map: blobTexture(), blending: THREE.MultiplyBlending, premultipliedAlpha: true, transparent: true, depthWrite: false });
    mat.onBeforeCompile = (shader) => {
      shader.fragmentShader = shader.fragmentShader.replace(
        "#include <color_fragment>",
        "#ifdef USE_INSTANCING_COLOR\n\tdiffuseColor.rgb = mix( vec3( 1.0 ), diffuseColor.rgb, vColor.r );\n#endif"
      );
    };
    mat.customProgramCacheKey = () => "contact-blob";
    return mat;
  }, []);
  useEffect(() => () => blobMat.dispose(), [blobMat]);

  // Footprint radius and rest height per streamed object, from the scenario.
  const footprints = useMemo(() => {
    const out = new Map<string, { r: number; z0: number }>();
    if (!scenario) return out;
    scenario.balls.forEach((b, i) => out.set(`ball${i}`, { r: b.radius * 1.15, z0: b.radius }));
    scenario.boxes.forEach((b, i) => {
      if (b.mass > 0) out.set(`box${i}`, { r: Math.hypot(b.size[0], b.size[1]) * 0.5, z0: b.size[2] / 2 });
    });
    for (const t of scenario.pickables ?? []) {
      const s = PICKABLE_SIZES[t.kind] ?? [0.03, 0.03, 0.03];
      out.set(t.id, { r: Math.max(0.028, Math.hypot(s[0], s[1]) * 0.6), z0: s[2] / 2 });
    }
    for (const p of scenario.persons ?? []) out.set(p.id, { r: p.radius * 1.25, z0: p.height / 2 });
    return out;
  }, [scenario]);

  const freeBoxGeos = useMemo(() => {
    const geos = new Map<string, RoundedBoxGeometry>();
    for (const b of scenario?.boxes ?? []) {
      if (b.mass <= 0) continue;
      const k = boxGeometryKey(b.size);
      if (!geos.has(k)) geos.set(k, roundedBox(b.size));
    }
    return geos;
  }, [scenario]);
  useEffect(() => () => freeBoxGeos.forEach((g) => g.dispose()), [freeBoxGeos]);

  useFrame((_, dt) => {
    const f = client.frame;
    const im = blobs.current;
    if (!f) {
      if (im) im.count = 0;
      return;
    }
    const a = 1 - Math.exp(-16 * Math.min(dt, 0.1));
    let n = 0;
    const put = (x: number, y: number, r: number, k: number) => {
      if (!im || n >= MAX_BLOBS || k <= 0.01) return;
      m.makeScale(r * 2, r * 2, 1);
      m.setPosition(x, y, Z_BLOB);
      im.setMatrixAt(n, m);
      col.setScalar(k);
      im.setColorAt(n, col);
      n++;
    };
    for (const o of f.objects) {
      const g = refs.current.get(o.id);
      if (g) {
        tmpP.set(o.pose[0], o.pose[1], o.pose[2]);
        tmpQ.set(o.pose[4], o.pose[5], o.pose[6], o.pose[3]);
        g.position.lerp(tmpP, a);
        g.quaternion.slerp(tmpQ, a);
      }
      const fp = footprints.get(o.id);
      if (!fp || o.held) continue;
      const lift = Math.max(0, o.pose[2] - fp.z0);
      put(o.pose[0], o.pose[1], fp.r * (1 + lift * 2), 0.75 * clamp01(1 - lift / 0.3));
    }
    for (const d of f.ducks) {
      const trunk = d.bodies[1];
      if (!trunk) continue;
      const lift = Math.max(0, trunk[2] - 0.17);   // trunk sits ~17 cm up when standing
      put(trunk[0], trunk[1], 0.125 * (1 + lift * 1.5), 0.7 * clamp01(1 - lift / 0.4));
    }
    if (im) {
      im.count = n;
      im.instanceMatrix.needsUpdate = true;
      if (im.instanceColor) im.instanceColor.needsUpdate = true;
    }
  });

  if (!scenario) return null;
  const freeBoxes = scenario.boxes.map((b, i) => ({ b, i })).filter(({ b }) => b.mass > 0);
  const persons = scenario.persons ?? [];
  const toys = scenario.pickables ?? [];
  const setRef = (id: string) => (el: THREE.Group | null) => {
    if (el) refs.current.set(id, el);
    else refs.current.delete(id);
  };
  return (
    <group>
      <instancedMesh ref={blobs} args={[undefined, undefined, MAX_BLOBS]} material={blobMat} frustumCulled={false}>
        <planeGeometry args={[1, 1]} />
      </instancedMesh>
      {toys.map((t) => (
        <group key={t.id} ref={setRef(t.id)}>
          <mesh geometry={toyGeometry(t.kind)}>
            <meshStandardMaterial color={PICKABLE_COLORS[t.kind] ?? "#cccccc"} roughness={t.kind === "sock" ? 0.95 : 0.45} metalness={0} envMapIntensity={t.kind === "sock" ? 0.2 : 0.9} />
          </mesh>
        </group>
      ))}
      {persons.map((q) => (
        <group key={q.id} ref={setRef(q.id)}>
          {/* a capsule standing on the floor; the nose cone shows its heading */}
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <capsuleGeometry args={[q.radius, Math.max(q.height - 2 * q.radius, 0.02), 8, 24]} />
            <meshStandardMaterial color={PERSON} roughness={0.6} metalness={0} envMapIntensity={0.5} transparent opacity={0.9} />
          </mesh>
          <mesh position={[q.radius + 0.03, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
            <coneGeometry args={[0.04, 0.08, 12]} />
            <meshStandardMaterial color={PERSON_NOSE} roughness={0.4} />
          </mesh>
        </group>
      ))}
      {scenario.balls.map((ball, i) => (
        <group key={`ball${i}`} ref={setRef(`ball${i}`)}>
          <mesh>
            <sphereGeometry args={[ball.radius, 32, 24]} />
            <meshStandardMaterial map={ballTexture()} roughness={0.42} metalness={0} envMapIntensity={1.0} />
          </mesh>
        </group>
      ))}
      {freeBoxes.map(({ b, i }) => (
        <group key={`box${i}`} ref={setRef(`box${i}`)}>
          <mesh geometry={freeBoxGeos.get(boxGeometryKey(b.size))}>
            <meshStandardMaterial color={new THREE.Color().setRGB(b.rgba[0], b.rgba[1], b.rgba[2], THREE.SRGBColorSpace)} roughness={0.7} metalness={0} envMapIntensity={0.6} />
          </mesh>
        </group>
      ))}
    </group>
  );
}
