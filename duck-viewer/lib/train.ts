// Types + client for the lab's /brains endpoint (viz_server.py).
//
// train-brain is a plain CLI process — it never talks to the lab — so this
// reads what the trainer writes to disk (brain.json, progress.jsonl) through
// the lab. Polling, not a socket: a rollout row lands every ~0.5 s at most,
// so a 2 s poll is already finer than the data changes.

import { LAB_HTTP } from "./lab";

export interface CurvePoint {
  steps: number;
  ep_rew: number;
  ep_len: number;
  elapsed_s: number;
}

export interface BrainRun {
  name: string;
  /** From brain.json — absent on a run that has not written one yet. */
  task?: string;
  obs_version?: number;
  envs?: number;
  /** The step BUDGET the run was launched with, not the steps done. */
  steps?: number;
  seed?: number;
  variety?: boolean;
  fixed_preset?: string | null;
  /** brain.onnx exists — the run finished and exported. */
  shipped: boolean;
  /** progress.jsonl grew within the last 30 s. */
  active: boolean;
  rollouts: number;
  last?: CurvePoint;
  /** done/budget, or null when the budget is unknown. */
  progress?: number | null;
  steps_per_s?: number | null;
  eta_s?: number | null;
  curve: CurvePoint[];
}

export async function fetchBrains(signal?: AbortSignal): Promise<BrainRun[]> {
  const r = await fetch(`${LAB_HTTP}/brains`, { signal, cache: "no-store" });
  if (!r.ok) throw new Error(`lab /brains → ${r.status}`);
  const j = (await r.json()) as { brains?: BrainRun[] };
  return j.brains ?? [];
}

/** "1h 04m", "6m 12s", "48s" — compact enough for a stat line. */
export function humanDuration(s: number | null | undefined): string {
  if (s == null || !isFinite(s) || s < 0) return "—";
  const t = Math.round(s);
  if (t < 60) return `${t}s`;
  const m = Math.floor(t / 60);
  if (m < 60) return `${m}m ${String(t % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

/** 1_127_424 → "1.13M", 24_576 → "24.6k". */
export function humanSteps(n: number | null | undefined): string {
  if (n == null || !isFinite(n)) return "—";
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

/** Stable per-run colour, so a run keeps its line colour across polls. */
const PALETTE = ["#6ee7b7", "#93c5fd", "#fcd34d", "#f9a8d4", "#c4b5fd", "#fdba74"];
export function runColor(name: string, i: number): string {
  return PALETTE[i % PALETTE.length];
}

/**
 * Trailing-mean smoothing over `w` rollouts. PPO's per-rollout episode
 * reward is noisy enough that the raw line hides the trend it exists to
 * show; the raw series is still drawn faintly underneath.
 */
export function smooth(pts: CurvePoint[], w = 9): CurvePoint[] {
  if (pts.length <= w) return pts;
  const out: CurvePoint[] = [];
  let sum = 0;
  for (let i = 0; i < pts.length; i++) {
    sum += pts[i].ep_rew;
    if (i >= w) sum -= pts[i - w].ep_rew;
    const n = Math.min(i + 1, w);
    out.push({ ...pts[i], ep_rew: sum / n });
  }
  return out;
}
