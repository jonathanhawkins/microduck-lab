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
  // --- the recipe, straight from brain.json (train_brain's meta dict) ---
  target_cls?: string;
  n_steps?: number;
  batch_size?: number;
  n_epochs?: number;
  lr?: number;
  lr_end?: number;
  net_arch?: string;
  log_std_max?: number | null;
  legacy_hparams?: boolean;
  polite?: number;
  polite_mix?: number[];
  decide_every_start?: number | null;
  init_from?: string | null;
  /** The commit the trainer ran from (short sha), and whether the tree was dirty. */
  git_sha?: string;
  git_dirty?: boolean;
  /** Written by select-brain: which checkpoint was shipped as brain.onnx, and why. */
  selected?: {
    /** "000751104" (a checkpoint's step, zero-padded) or "final". */
    tag: string;
    metric: string;
    score: number;
    final_score: number;
    seeds?: number[];
    episodes?: number;
  };
}

/**
 * The knobs a card diffs, in display order, with the label each shows as.
 * Anything else scalar in brain.json that differs is diffed too under its
 * raw key (see recipeDiff) — a new trainer flag shows up without a viewer
 * change, just less prettily.
 */
export const KNOBS: [keyof BrainRun, string][] = [
  ["task", "task"],
  ["obs_version", "obs"],
  ["envs", "envs"],
  ["steps", "budget"],
  ["seed", "seed"],
  ["n_steps", "n_steps"],
  ["batch_size", "batch"],
  ["n_epochs", "epochs"],
  ["lr", "lr"],
  ["lr_end", "lr_end"],
  ["net_arch", "arch"],
  ["log_std_max", "log_std_max"],
  ["legacy_hparams", "legacy"],
  ["variety", "variety"],
  ["polite", "polite"],
  ["polite_mix", "polite_mix"],
  ["fixed_preset", "preset"],
  ["decide_every_start", "decide_start"],
  ["init_from", "init_from"],
  ["git_sha", "git"],
];

/** Fields that describe the run's STATE or contract, not its recipe. */
const NOT_KNOBS = new Set<string>([
  "name", "curve", "last", "progress", "rollouts", "active", "shipped",
  "steps_per_s", "eta_s", "selected", "act_low", "act_high", "obs_dim",
  "decide_every", "probe_presets", "target_cls", "git_dirty",
]);

/** A knob value as a card shows it: 0.0003 → "3e-4", true → "on", [] → "—". */
export function fmtKnob(v: unknown): string {
  if (v === undefined || v === null) return "—";
  if (typeof v === "boolean") return v ? "on" : "off";
  if (typeof v === "number") {
    if (v !== 0 && Math.abs(v) < 0.01) return v.toExponential(0).replace("e-0", "e-").replace("e+0", "e");
    if (Number.isInteger(v) && v >= 1e6) return humanSteps(v);
    return String(v);
  }
  if (Array.isArray(v)) return v.length ? v.map(fmtKnob).join("/") : "—";
  return String(v);
}

export interface KnobDiff {
  key: string;
  label: string;
  /** This run's value, formatted. */
  value: string;
  /** The baseline's value, formatted. */
  from: string;
}

const same = (a: unknown, b: unknown) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

/**
 * Every knob on which `run` differs from `base`, in KNOBS order, then any
 * unlisted scalar keys alphabetically. Two runs with identical recipes diff
 * to []; a run is never diffed against itself.
 */
export function recipeDiff(run: BrainRun, base: BrainRun): KnobDiff[] {
  if (run.name === base.name) return [];
  const out: KnobDiff[] = [];
  const seen = new Set<string>();
  for (const [key, label] of KNOBS) {
    seen.add(key);
    const a = run[key];
    const b = base[key];
    if (same(a, b)) continue;
    out.push({ key, label, value: fmtKnob(a), from: fmtKnob(b) });
  }
  // brain.json is open-ended (the striker task adds a dozen keys), so the
  // typed interface is a subset — read the rest as a plain record.
  const R = run as unknown as Record<string, unknown>;
  const B = base as unknown as Record<string, unknown>;
  const extra = new Set<string>();
  for (const k of [...Object.keys(R), ...Object.keys(B)]) {
    if (seen.has(k) || NOT_KNOBS.has(k)) continue;
    const v = R[k] ?? B[k];
    if (v !== null && typeof v === "object" && !Array.isArray(v)) continue;
    extra.add(k);
  }
  for (const k of [...extra].sort()) {
    if (same(R[k], B[k])) continue;
    out.push({ key: k, label: k, value: fmtKnob(R[k]), from: fmtKnob(B[k]) });
  }
  return out;
}

/**
 * The step the shipped brain.onnx was taken from, or null when nothing was
 * selected yet. select-brain tags a checkpoint by its zero-padded step and
 * the end-of-run model as "final" — which is wherever the curve stops.
 */
export function shippedStep(run: BrainRun): number | null {
  const tag = run.selected?.tag;
  if (!tag) return null;
  if (/^\d+$/.test(tag)) return parseInt(tag, 10);
  if (tag === "final") return run.curve.length ? run.curve[run.curve.length - 1].steps : null;
  return null;
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
