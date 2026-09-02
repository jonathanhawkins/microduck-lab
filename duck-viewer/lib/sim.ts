// Types + client for the lab's WORLD mode (microduck_local/world_server.py):
// the /sim page's backend. Same lab host as lib/lab.ts, second socket.

import { LAB_HTTP } from "@/lib/lab";

export const SIM_WS = LAB_HTTP.replace(/^http/, "ws") + "/ws/sim";

export type TofPreset = "ideal" | "datasheet" | "hostile";
export const TOF_PRESETS: TofPreset[] = ["ideal", "datasheet", "hostile"];

export interface ScenarioWall { from: [number, number]; to: [number, number]; height: number; thickness: number }
export interface ScenarioBox { pos: [number, number, number]; size: [number, number, number]; yaw: number; mass: number; rgba: [number, number, number, number] }
export interface ScenarioBall { pos: [number, number]; radius: number; mass: number }
export interface ScenarioDuck { id: string; spawn: [number, number, number]; policy: string | null; tof: TofPreset | null }
export interface Scenario {
  version: number;
  name: string;
  seed: number;
  floor: { size: [number, number] };
  walls: ScenarioWall[];
  boxes: ScenarioBox[];
  balls: ScenarioBall[];
  ducks: ScenarioDuck[];
  collision: "walk" | "all";
}
export interface ScenarioListing { name: string; builtin: boolean; ducks: number; objects: number; modified: number | null }

export interface TofPayload {
  t: number;
  mm: number[];                       // 64, row-major, 0 = no target
  age: number;
  pts: ([number, number, number] | null)[]; // world points (MuJoCo frame) per zone
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
  /** Who is steering this duck this tick: its own wander brain (auto mode,
   *  ducks with a ToF), the demo script (blind ducks), or you (manual). */
  brain: { kind: "wander" | "script" | "manual"; state: string; cmd: [number, number, number] };
  bodies: number[][];
  sensors: { tof: TofPayload } | null;
}
export interface SimObject { id: string; kind: "ball" | "box"; pose: number[] }
export interface SimFrame {
  t: number;
  tick: number;
  rtf: number;
  scenario: string | null;
  loading: boolean;
  cmd: [number, number, number];
  mode: "auto" | "manual";
  events: string[];
  ducks: SimDuck[];
  objects: SimObject[];
}
export interface WorldInfo {
  scenario: Scenario | null;
  loading: boolean;
  ducks: Omit<SimDuck, "bodies" | "sensors">[];
  presets: TofPreset[];
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

/** WebSocket client for /ws/sim: keeps the latest frame, reconnects. */
export class SimClient {
  frame: SimFrame | null = null;
  connected = false;
  lastFrameAt = 0;
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
      this.frame = frame;
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
  sendNoise(duck: string, preset: TofPreset) { this.send({ noise: { duck, preset } }); }
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
