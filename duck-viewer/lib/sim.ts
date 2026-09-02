// Types + client for the lab's WORLD mode (microduck_local/world_server.py):
// the /sim page's backend. Same lab host as lib/lab.ts, second socket.

import { LAB_HTTP } from "@/lib/lab";

export const SIM_WS = LAB_HTTP.replace(/^http/, "ws") + "/ws/sim";

export type TofPreset = "ideal" | "datasheet" | "hostile";
export const TOF_PRESETS: TofPreset[] = ["ideal", "datasheet", "hostile"];

export interface ScenarioWall { from: [number, number]; to: [number, number]; height: number; thickness: number }
export interface ScenarioBox { pos: [number, number, number]; size: [number, number, number]; yaw: number; mass: number; rgba: [number, number, number, number] }
export interface ScenarioBall { pos: [number, number]; radius: number; mass: number }
export interface ScenarioDuck { id: string; spawn: [number, number, number]; policy: string | null; tof: TofPreset | null; detector?: TofPreset | null; brain?: string | null }
export interface ScenarioPerson { id: string; pos: [number, number]; yaw: number; path: [number, number][]; speed: number; radius: number; height: number }
export interface ScenarioPickable { id: string; kind: "brick" | "block" | "sock"; pos: [number, number]; yaw: number }
export interface ScenarioBasket { pos: [number, number]; size: [number, number]; rim: number }
export const PICKABLE_SIZES: Record<string, [number, number, number]> = {
  brick: [0.032, 0.016, 0.0096],
  block: [0.04, 0.04, 0.04],
  sock: [0.06, 0.035, 0.025],
};
export const PICKABLE_COLORS: Record<string, string> = { brick: "#d92626", block: "#f2bf33", sock: "#9999e6" };
export interface Scenario {
  version: number;
  name: string;
  seed: number;
  floor: { size: [number, number] };
  walls: ScenarioWall[];
  boxes: ScenarioBox[];
  balls: ScenarioBall[];
  ducks: ScenarioDuck[];
  persons?: ScenarioPerson[];
  pickables?: ScenarioPickable[];
  basket?: ScenarioBasket | null;
  collision: "walk" | "all";
}
export interface ScenarioListing { name: string; builtin: boolean; ducks: number; objects: number; modified: number | null }

export interface TofPayload {
  t: number;
  mm: number[];                       // 64, row-major, 0 = no target
  age: number;
}

// The ToF's mount on the head: the MJCF `tof` site in the jaw_soft body
// frame (robot_walk.xml), x-forward / y-left / z-up. Mirrors sensors/tof.py.
export const TOF_ROWS = 8;
export const TOF_COLS = 8;
export const TOF_FOV_DEG = 45;
export const TOF_SITE_POS: [number, number, number] = [0.0135, 0.0224086, -0.0733];
export const TOF_SITE_QUAT_WXYZ: [number, number, number, number] = [0.707107, 0, 0.707107, 0];

/** Zone centre directions in the site frame, row-major (row 0 = up, col 0 = left). */
export const TOF_ZONE_DIRS: [number, number, number][] = (() => {
  const half = Math.tan((TOF_FOV_DEG * Math.PI) / 360);
  const out: [number, number, number][] = [];
  for (let r = 0; r < TOF_ROWS; r++)
    for (let c = 0; c < TOF_COLS; c++) {
      const y = half * (1 - (2 * c + 1) / TOF_COLS);
      const z = half * (1 - (2 * r + 1) / TOF_ROWS);
      const n = Math.hypot(1, y, z);
      out.push([1 / n, y / n, z / n]);
    }
  return out;
})();

function quatMul(a: number[], b: number[]): [number, number, number, number] {
  // wxyz
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}
function quatRotate(q: number[], v: [number, number, number]): [number, number, number] {
  // wxyz quaternion applied to v
  const [w, x, y, z] = q;
  const t0 = 2 * (y * v[2] - z * v[1]);
  const t1 = 2 * (z * v[0] - x * v[2]);
  const t2 = 2 * (x * v[1] - y * v[0]);
  return [
    v[0] + w * t0 + (y * t2 - z * t1),
    v[1] + w * t1 + (z * t0 - x * t2),
    v[2] + w * t2 + (x * t1 - y * t0),
  ];
}

/** World-frame aperture and zone points of a ToF frame, from the jaw body
 *  pose [x,y,z,qw,qx,qy,qz] the duck frame already carries. null = no target. */
export function tofZonePoints(jaw: number[], mm: number[]): { origin: [number, number, number]; pts: ([number, number, number] | null)[] } {
  const jq = [jaw[3], jaw[4], jaw[5], jaw[6]];
  const off = quatRotate(jq, TOF_SITE_POS);
  const origin: [number, number, number] = [jaw[0] + off[0], jaw[1] + off[1], jaw[2] + off[2]];
  const sq = quatMul(jq, TOF_SITE_QUAT_WXYZ);
  const pts = TOF_ZONE_DIRS.map((d, k) => {
    const depth = mm[k] / 1000;
    if (!depth) return null;
    const w = quatRotate(sq, d);
    return [origin[0] + w[0] * depth, origin[1] + w[1] * depth, origin[2] + w[2] * depth] as [number, number, number];
  });
  return { origin, pts };
}
export interface DetectionItem { cls: string; name: string; bearing: number; elevation: number; width: number; range: number; conf: number }
export interface DetPayload { t: number; age: number; items: DetectionItem[] }
export interface BrainInputs {
  tof?: { age: number | null; stale: boolean; max: number };
  det?: { age: number | null; stale: boolean; max: number; n: number };
  target?: { bearing: number; range: number | null; since: number } | null;
}
export interface SimDuck {
  id: string;
  name: string;
  policy: string | null;
  falls: number;
  step: number;
  rew: number;
  speed: number;
  cmdSpeed: number;
  steerable: boolean;
  tof: TofPreset | "custom" | null;
  detector: TofPreset | "custom" | null;
  holding: string | null;
  /** Odometry drift preset the brain's pose carries (roadmap 1.7), and the drifted estimate itself. */
  odom?: string;
  odomEst?: [number, number, number];
  skill: string | null;
  beak: "open" | "closed";
  /** Who is steering this duck this tick: a brain from the lab's registry
   *  (auto mode), the demo script (blind ducks), or you (manual). */
  brain: { kind: string; state: string; cmd: [number, number, number]; head?: number[]; note?: string; beak?: string | null; skill?: string | null; inputs: BrainInputs & { tidy?: { picked: number; delivered: number; givenUp: string[] } } };
  headApplied: boolean;
  bodies: number[][];
  sensors: { tof?: TofPayload; det?: DetPayload } | null;
}
export interface SimObject { id: string; kind: "ball" | "box" | "person" | "toy"; pose: number[]; possessed?: boolean; toy?: string; held?: string | null; inBasket?: boolean }
export interface TidyScore { total: number; inBasket: number; held: string[] }
export interface SimFrame {
  t: number;
  tick: number;
  rtf: number;
  /** Lab-side cost per control step (ms, running means): physics + policies, sensors, and the JSON frame encode. */
  perf: { stepMs: number; sensorMs: number; encodeMs?: number } | null;
  scenario: string | null;
  loading: boolean;
  cmd: [number, number, number];
  mode: "auto" | "manual";
  events: string[];
  ducks: SimDuck[];
  objects: SimObject[];
  possessed: string | null;
  tidy: TidyScore | null;
  /** Soccer score on a pitch scenario (goals per short wall, ball position), else null. */
  soccer: { left: number; right: number; ball: [number, number] } | null;
  /** Brain round-trip latency applied to every intent (roadmap 12.10), ms; 0 = onboard. */
  tetherMs?: number;
  /** Occupancy maps per duck, in each duck's ODOMETRY frame (brain-layer output, ~2 Hz; null on the other frames). */
  maps: Record<string, OccupancyMap> | null;
}
export interface OccupancyMap {
  nx: number;
  ny: number;
  res: number;
  origin: [number, number];
  frames: number;
  /** ny*nx chars, row-major from -y: '0' unknown, '1' free, '2' occupied. */
  cells: string;
  /** Loop closure (brain/mapping.py): the odometry→map correction (x, y, yaw) the
   *  wall-line matcher has accumulated, how many frames it corrected, and the
   *  corrected pose the last frame was folded in at. */
  offset?: [number, number, number];
  corrections?: number;
  pose?: [number, number, number] | null;
}
export interface WorldInfo {
  scenario: Scenario | null;
  loading: boolean;
  ducks: Omit<SimDuck, "bodies" | "sensors" | "brain" | "headApplied">[];
  presets: TofPreset[];
  brains: string[];
}

// The head camera: the MJCF `head_camera` site, x-forward, on jaw_soft.
export const CAM_SITE_POS: [number, number, number] = [0.0155, -0.0000913778, -0.0733];
export const CAM_SITE_QUAT_WXYZ: [number, number, number, number] = [0.707107, 0, 0.707107, 0];

/** World-frame ray for one detection (origin + unit direction) from the jaw pose. */
export function detectionRay(jaw: number[], d: DetectionItem): { origin: [number, number, number]; dir: [number, number, number] } {
  const jq = [jaw[3], jaw[4], jaw[5], jaw[6]];
  const off = quatRotate(jq, CAM_SITE_POS);
  const origin: [number, number, number] = [jaw[0] + off[0], jaw[1] + off[1], jaw[2] + off[2]];
  const sq = quatMul(jq, CAM_SITE_QUAT_WXYZ);
  const cb = Math.cos(d.bearing), sb = Math.sin(d.bearing), ce = Math.cos(d.elevation), se = Math.sin(d.elevation);
  const local: [number, number, number] = [cb * ce, sb * ce, se];
  return { origin, dir: quatRotate(sq, local) };
}

export async function fetchScenarios(): Promise<ScenarioListing[]> {
  const r = await fetch(`${LAB_HTTP}/scenarios`);
  if (!r.ok) throw new Error(`GET /scenarios ${r.status}`);
  return (await r.json()).scenarios;
}
export async function fetchWorld(): Promise<WorldInfo> {
  const r = await fetch(`${LAB_HTTP}/world`);
  if (!r.ok) throw new Error(`GET /world ${r.status}`);
  return r.json();
}
export async function loadWorld(name: string): Promise<WorldInfo> {
  const r = await fetch(`${LAB_HTTP}/world/load`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ scenario: name }),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `load ${r.status}`);
  return r.json();
}

export async function fetchRing(last = 1500): Promise<SimFrame[]> {
  const r = await fetch(`${LAB_HTTP}/replay/ring?last=${last}`);
  if (!r.ok) throw new Error(`GET /replay/ring ${r.status}`);
  return (await r.json()).frames;
}
export interface RecordingHeader { name: string; scenario: string | null; saved: number; frames: number; span: number }
export async function saveRecording(name: string): Promise<RecordingHeader> {
  const r = await fetch(`${LAB_HTTP}/replay/save`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `save ${r.status}`);
  return r.json();
}
export async function fetchRecordings(): Promise<RecordingHeader[]> {
  const r = await fetch(`${LAB_HTTP}/recordings`);
  if (!r.ok) throw new Error(`GET /recordings ${r.status}`);
  return (await r.json()).recordings;
}
export async function fetchRecording(name: string): Promise<{ header: RecordingHeader; frames: SimFrame[] }> {
  const r = await fetch(`${LAB_HTTP}/recordings/${encodeURIComponent(name)}`);
  if (!r.ok) throw new Error(`GET /recordings/${name} ${r.status}`);
  return r.json();
}

/** Something worth a tick mark on the scrub bar, found by diffing frames. */
export interface FrameEvent { index: number; t: number; kind: "fall" | "brain" | "mode"; text: string }
export function frameEvents(frames: SimFrame[]): FrameEvent[] {
  const out: FrameEvent[] = [];
  for (let i = 1; i < frames.length; i++) {
    const a = frames[i - 1], b = frames[i];
    if (a.mode !== b.mode) out.push({ index: i, t: b.t, kind: "mode", text: `${b.mode} drive` });
    for (const d of b.ducks) {
      const prev = a.ducks.find((x) => x.id === d.id);
      if (!prev) continue;
      if (d.falls > prev.falls) out.push({ index: i, t: b.t, kind: "fall", text: `${d.id} fell` });
      // Only the interesting brain transitions: a cruise↔steer flip every
      // few frames is noise on a bar this wide.
      if (d.brain && prev.brain && d.brain.state !== prev.brain.state && ["spin", "unstick", "blind", "lost"].includes(d.brain.state))
        out.push({ index: i, t: b.t, kind: "brain", text: `${d.id}: ${prev.brain.state} → ${d.brain.state}` });
    }
  }
  return out;
}

/** WebSocket client for /ws/sim: keeps the latest frame, reconnects. While
 *  scrubbing, `frame` is the scrubbed frame and live frames keep arriving
 *  underneath (`live`), so going back to live is instant. */
export class SimClient {
  live: SimFrame | null = null;
  scrub: SimFrame | null = null;
  get frame(): SimFrame | null {
    return this.scrub ?? this.live;
  }
  connected = false;
  lastFrameAt = 0;
  /** Bytes received so far (the perf HUD differentiates it). */
  bytes = 0;
  private ws: WebSocket | null = null;
  private closed = false;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private pendingEvents: string[] = [];

  constructor(private onStatus?: (connected: boolean) => void) {
    this.connect();
  }

  private connect() {
    if (this.closed) return;
    const ws = new WebSocket(SIM_WS);
    this.ws = ws;
    ws.onopen = () => {
      if (this.ws !== ws) { ws.close(); return; }
      this.connected = true;
      this.lastFrameAt = Date.now();
      this.onStatus?.(true);
    };
    ws.onmessage = (ev) => {
      if (this.ws !== ws) return;
      const frame: SimFrame = JSON.parse(ev.data);
      this.live = frame;
      this.bytes += ev.data.length;
      this.lastFrameAt = Date.now();
      if (frame.events?.length) {
        this.pendingEvents.push(...frame.events);
        if (this.pendingEvents.length > 50) this.pendingEvents = this.pendingEvents.slice(-50);
      }
    };
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.connected = false;
      this.lastFrameAt = 0;
      this.onStatus?.(false);
      if (!this.closed && this.reconnectTimer === null) {
        this.reconnectTimer = setTimeout(() => { this.reconnectTimer = null; this.connect(); }, 1500);
      }
    };
    ws.onerror = () => ws.close();
  }

  drainEvents(): string[] {
    const out = this.pendingEvents;
    this.pendingEvents = [];
    return out;
  }
  private send(obj: unknown) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(obj));
  }
  sendCmd(cmd: [number, number, number]) { this.send({ cmd }); }
  sendReset() { this.send({ reset: true }); }
  sendAssign(duck: string, policy: string) { this.send({ assign: { duck, policy } }); }
  sendNoise(duck: string, preset: TofPreset, sensor: "tof" | "det" | "odom" = "tof") { this.send({ noise: { duck, preset, sensor } }); }
  sendBrain(duck: string, kind: string) { this.send({ brain: { duck, kind } }); }
  sendPossess(person: string | null) { this.send({ possess: person }); }
  sendHead(duck: string, apply: boolean) { this.send({ head: { duck, apply } }); }
  close() {
    this.closed = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
  }
}

/** Depth → color for the 8×8 heatmap and the in-scene dots: near is warm
 *  amber, far is cool teal, no target is dark. Matches the lesson page. */
export function depthColor(mm: number, maxMm = 4000): string {
  if (mm <= 0) return "#262a33";
  const t = Math.max(0, Math.min(1, 1 - mm / maxMm)); // 1 = near
  // teal (#43c2b8) → amber (#f2b632)
  const r = Math.round(0x43 + (0xf2 - 0x43) * t);
  const g = Math.round(0xc2 + (0xb6 - 0xc2) * t);
  const b = Math.round(0xb8 + (0x32 - 0xb8) * t);
  return `rgb(${r},${g},${b})`;
}
