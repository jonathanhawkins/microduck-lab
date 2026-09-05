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
 * brain.json is open-ended (the striker task adds a dozen keys), so this is
 * the DISPLAY order, not the universe: recipeKeys() appends whatever else a
 * run recorded, under its raw key — a new trainer flag shows up on the
 * cards, the matrix and the footer without a viewer change, just less
 * prettily.
 */
export type Knob = [key: string, label: string];
export const KNOBS: Knob[] = [
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

/** brain.json read as the open record it is; the typed interface is a subset. */
export const knob = (r: BrainRun, key: string): unknown => (r as unknown as Record<string, unknown>)[key];

/**
 * Every recipe key any of these runs records — KNOBS in display order, then
 * unlisted scalar keys alphabetically. The ONE list the cards' diff, the
 * matrix's columns and the footer all draw from, so a knob can never show
 * on one and be missing from another.
 */
export function recipeKeys(runs: BrainRun[]): Knob[] {
  const listed = new Set(KNOBS.map(([k]) => k));
  const extra = new Set<string>();
  for (const r of runs) {
    for (const k of Object.keys(r)) {
      if (listed.has(k) || NOT_KNOBS.has(k)) continue;
      const v = knob(r, k);
      if (v !== null && typeof v === "object" && !Array.isArray(v)) continue;
      extra.add(k);
    }
  }
  return [...KNOBS, ...[...extra].sort().map((k): Knob => [k, k])];
}

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

/**
 * A knob as a chip reads: null when there is nothing to say (unset, off,
 * empty), the bare label for a flag that is on, else "label value". The one
 * rule for the baseline card and the matrix footer.
 */
export function knobChip(label: string, v: unknown): string | null {
  if (v === undefined || v === null || v === false || (Array.isArray(v) && !v.length)) return null;
  return v === true ? label : `${label} ${fmtKnob(v)}`;
}

/** The whole recipe of one run as chips — what the baseline card shows. */
export function recipeChips(run: BrainRun): { key: string; text: string }[] {
  return recipeKeys([run]).flatMap(([key, label]) => {
    const text = knobChip(label, knob(run, key));
    return text === null ? [] : [{ key, text }];
  });
}

export interface KnobDiff {
  key: string;
  label: string;
  /** This run's value, formatted. */
  value: string;
  /** The baseline's value, formatted. */
  from: string;
}

/** Equality the way brain.json means it: null and absent are the same "unset". */
const canon = (v: unknown) => JSON.stringify(v ?? null);
const same = (a: unknown, b: unknown) => canon(a) === canon(b);

/**
 * Every knob on which `run` differs from `base`, in recipeKeys order. Two
 * runs with identical recipes diff to []; a run is never diffed against
 * itself.
 */
export function recipeDiff(run: BrainRun, base: BrainRun): KnobDiff[] {
  if (run.name === base.name) return [];
  const out: KnobDiff[] = [];
  for (const [key, label] of recipeKeys([run, base])) {
    const a = knob(run, key);
    const b = knob(base, key);
    if (same(a, b)) continue;
    out.push({ key, label, value: fmtKnob(a), from: fmtKnob(b) });
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

// ---------------------------------------------------------------- sweep matrix

/** "p-n256-s34" → "p-n256": the sweep family, i.e. the name minus its seed. */
export function familyOf(name: string): string {
  return name.replace(/-s\d+$/, "");
}

/** Values of a knob across the runs that RECORD it — silence is not a value. */
function recordedValues(runs: BrainRun[], key: string): Set<string> {
  return new Set(runs.filter((r) => knob(r, key) !== undefined).map((r) => JSON.stringify(knob(r, key))));
}

/**
 * The knobs on which the runs do NOT all agree — the only columns a sweep
 * matrix needs; a knob every run shares says nothing about the sweep.
 * Only runs that RECORD a knob are compared: an older brain.json that
 * predates a flag is silent about it, not different, and counting silence
 * as a value turned every column on as soon as one old run was on disk.
 */
export function varyingKnobs(runs: BrainRun[]): Knob[] {
  if (runs.length < 2) return [];
  return recipeKeys(runs).filter(([key]) => recordedValues(runs, key).size > 1);
}

/**
 * What every run shares, as chips — the matrix footer, said once instead
 * of 49 times. Each value comes from the first run that RECORDS it, not
 * from runs[0]: the runs agree (that is what "not varying" means), but the
 * first one may predate the knob.
 */
export function sharedKnobs(runs: BrainRun[]): string[] {
  const varying = new Set(varyingKnobs(runs).map(([k]) => k));
  return recipeKeys(runs).flatMap(([key, label]) => {
    if (varying.has(key)) return [];
    const src = runs.find((r) => knob(r, key) !== undefined && knob(r, key) !== null);
    const text = src ? knobChip(label, knob(src, key)) : null;
    return text === null ? [] : [text];
  });
}

export interface MatrixRow {
  /** Run name, or family name when grouped — "p-de (2)" for a second recipe under one name. */
  key: string;
  runs: BrainRun[];
  /** Formatted knob per column. */
  knobs: Record<string, string>;
  /** The same knob unformatted — what a column SORTS by. null when no member recorded it. */
  raw: Record<string, unknown>;
  /** Benchmark score of the shipped checkpoint — mean across members. */
  score: number | null;
  /** Spread across members (sample sd); null for a single run. */
  scoreSd: number | null;
  /** Benchmark score of the end-of-run model, mean. */
  final: number | null;
  /** Last training reward, mean. */
  reward: number | null;
  /** Step the shipped checkpoint came from, mean. */
  shippedAt: number | null;
}

function meanSd(xs: number[]): [number | null, number | null] {
  if (!xs.length) return [null, null];
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  if (xs.length < 2) return [m, null];
  const v = xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1);
  return [m, Math.sqrt(v)];
}

/**
 * Two runs are one recipe when they agree on every knob BOTH record, seed
 * aside. A run silent about a knob (older brain.json) fits either way.
 */
function sameRecipe(a: BrainRun, b: BrainRun, keys: Knob[]): boolean {
  for (const [key] of keys) {
    if (key === "seed") continue;
    const x = knob(a, key);
    const y = knob(b, key);
    if (x === undefined || y === undefined) continue;
    if (!same(x, y)) return false;
  }
  return true;
}

/**
 * One row per run, or per family when `byFamily` — the seeds of a sweep
 * collapsed to a mean ± sd, which is the number that decides whether a knob
 * moved anything: a single seed's in_band moves ±0.02 run to run on its own.
 *
 * A family is the name minus its seed AND one recipe: runs that share a name
 * but were run with different knobs split into "p-de" and "p-de (2)" rather
 * than being averaged together, so a family cell can never disagree with
 * itself — there is no "≠" to show.
 */
export function matrixRows(runs: BrainRun[], knobs: Knob[], byFamily: boolean): MatrixRow[] {
  const keys = recipeKeys(runs);
  const groups: [string, BrainRun[]][] = [];
  if (byFamily) {
    const byName = new Map<string, BrainRun[][]>();
    for (const r of runs) {
      const fam = familyOf(r.name);
      const clusters = byName.get(fam) ?? [];
      const home = clusters.find((c) => c.every((m) => sameRecipe(m, r, keys)));
      if (home) home.push(r);
      else clusters.push([r]);
      byName.set(fam, clusters);
    }
    for (const [fam, clusters] of byName) {
      clusters.forEach((members, i) => groups.push([i ? `${fam} (${i + 1})` : fam, members]));
    }
  } else {
    for (const r of runs) groups.push([r.name, [r]]);
  }
  return groups.map(([key, members]) => {
    const cells: Record<string, string> = {};
    const raw: Record<string, unknown> = {};
    for (const [k] of knobs) {
      // A family's seed column is its member count.
      if (byFamily && k === "seed" && members.length > 1) {
        raw[k] = members.length;
        cells[k] = `×${members.length}`;
        continue;
      }
      const src = members.find((r) => knob(r, k) !== undefined);
      raw[k] = src ? knob(src, k) ?? null : null;
      cells[k] = fmtKnob(raw[k]);
    }
    const pick = (f: (r: BrainRun) => number | null | undefined) =>
      members.map(f).filter((v): v is number => v != null && isFinite(v));
    const [score, scoreSd] = meanSd(pick((r) => r.selected?.score));
    const [final] = meanSd(pick((r) => r.selected?.final_score));
    const [reward] = meanSd(pick((r) => r.last?.ep_rew));
    const [shippedAt] = meanSd(pick(shippedStep));
    return { key, runs: members, knobs: cells, raw, score, scoreSd, final, reward, shippedAt };
  });
}
