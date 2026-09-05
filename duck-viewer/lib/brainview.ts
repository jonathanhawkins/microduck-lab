// What a learned brain sees and says, as the /sim inspector draws it.
//
// The wire shape is runtime.brain_view: the observation vector the network
// read at its last decision, the action it emitted before the intent clip,
// the clipped action, and the bounds. Everything here is layout knowledge
// — which slot means what and what its range is — so the panel can draw 80
// floats as a strip of bars that a person can read, and three (or five)
// actions as gauges. The layout is brain_env.py's contract, version 2, plus
// the striker's eight extra slots (striker.py).

export interface BrainAct {
  /** The network's output before the intent clip. */
  raw: number[];
  /** What the duck was actually given. */
  clipped: number[];
  low: number[];
  high: number[];
}

export interface BrainView {
  obs: number[];
  obs_version: number;
  decide_every: number;
  act: BrainAct;
}

export interface ObsSlot {
  /** Group the slot belongs to — the legend under the strip. */
  group: string;
  label: string;
  lo: number;
  hi: number;
}

const slot = (group: string, label: string, lo: number, hi: number): ObsSlot => ({ group, label, lo, hi });

/** The 80-slot follow contract (brain_env.py, version 2). */
const BASE: ObsSlot[] = [
  ...Array.from({ length: 64 }, (_, i) => slot("tof", `tof ${Math.floor(i / 8)},${i % 8}`, 0, 4)),
  slot("age", "tof age", 0, 1),
  slot("target", "hit this frame", 0, 1),
  slot("target", "bearing", -Math.PI, Math.PI),
  slot("target", "elevation", -Math.PI / 2, Math.PI / 2),
  slot("target", "width", 0, 1),
  slot("target", "range", 0, 4),
  slot("target", "conf", 0, 1),
  slot("age", "det age", 0, 1),
  slot("since", "since seen", 0, 5),
  slot("act", "last vx", -1, 1),
  slot("act", "last vy", -1, 1),
  slot("act", "last wz", -1, 1),
  slot("speed", "speed", 0, 0.6),
  slot("track", "coasting", 0, 1),
  slot("track", "yaw rate", -3, 3),
  slot("track", "confirmed", 0, 1),
];

/** The striker's eight extra slots (striker.py). */
const STRIKER: ObsSlot[] = [
  slot("goal", "sin goal", -1, 1),
  slot("goal", "cos goal", -1, 1),
  slot("goal", "goal range", 0, 1),
  slot("line", "sin ball→goal", -1, 1),
  slot("line", "cos ball→goal", -1, 1),
  slot("ball", "ball x", -2, 2),
  slot("ball", "ball y", -2, 2),
  slot("busy", "kick busy", 0, 1),
];

/** Slot layout for an observation of `n` floats; unknown sizes get generic slots. */
export function obsSlots(n: number): ObsSlot[] {
  if (n === BASE.length) return BASE;
  if (n === BASE.length + STRIKER.length) return [...BASE, ...STRIKER];
  return Array.from({ length: n }, (_, i) => slot("obs", `obs[${i}]`, -1, 1));
}

/** The legend: contiguous runs of one group, in slot order. */
export function obsGroups(n: number): { group: string; start: number; end: number }[] {
  const out: { group: string; start: number; end: number }[] = [];
  for (const [i, s] of obsSlots(n).entries()) {
    const last = out[out.length - 1];
    if (last && last.group === s.group) last.end = i + 1;
    else out.push({ group: s.group, start: i, end: i + 1 });
  }
  return out;
}

/**
 * Every slot mapped to 0..1 by its own range, so the strip reads at a
 * glance: a 2 m ToF cell is half a bar, a straight-ahead bearing is half a
 * bar, a hit flag is a full bar or nothing. The "last action" slots use the
 * brain's OWN bounds, which is what they were clipped to.
 */
export function normalizeObs(view: BrainView): number[] {
  const slots = obsSlots(view.obs.length);
  return view.obs.map((v, i) => {
    let { lo, hi } = slots[i];
    const a = i - 73;
    if (slots[i].group === "act" && a >= 0 && a < view.act.low.length) {
      lo = view.act.low[a];
      hi = view.act.high[a];
    }
    if (!(hi > lo)) return 0;
    return Math.min(1, Math.max(0, (v - lo) / (hi - lo)));
  });
}

export const ACTION_NAMES = ["vx", "vy", "wz", "kick L", "kick R"];

/**
 * True where the action sits ON its bound. The exported graph clamps the
 * network's output itself (train_brain.export_brain: maximum(minimum(...))),
 * so a raw ask past the bound is never visible — what IS visible is the
 * result pinned to the edge. One decision at the edge is a brain going
 * flat out; every decision at the edge, target or no target, is the
 * saturated-mean trap (see AGENTS.md on log_std) showing itself live.
 */
export function atBound(act: BrainAct, eps = 1e-3): boolean[] {
  return act.clipped.map((a, i) => a <= act.low[i] + eps || a >= act.high[i] - eps);
}

/** A gauge's geometry in 0..1 of its track: where zero sits, where the value ends, where the raw output lands (the same place for an exported brain — the graph clamps). */
export function gauge(act: BrainAct, i: number): { zero: number; value: number; raw: number } {
  const lo = act.low[i];
  const hi = act.high[i];
  const span = hi - lo || 1;
  const at = (v: number) => Math.min(1, Math.max(0, (v - lo) / span));
  return { zero: at(0), value: at(act.clipped[i]), raw: at(act.raw[i]) };
}

/** The target block in words — the part of the vector a person asks about first. */
export function targetReadout(view: BrainView): string {
  const o = view.obs;
  if (o.length < 80) return "";
  const hit = o[65] > 0.5;
  const coasting = view.obs_version >= 2 && o[77] > 0.5;
  const confirmed = view.obs_version >= 2 && o[79] > 0.5;
  if (!hit && !coasting) return `no target · last seen ${o[72] >= 5 ? "5+" : o[72].toFixed(1)} s ago`;
  const deg = (r: number) => `${(r * 57.2958).toFixed(0)}°`;
  const parts = [
    hit ? "● hit" : "○ coasting",
    `${deg(o[66])} · ${deg(o[67])} up`,
    `${o[69].toFixed(2)} m`,
    `w ${o[68].toFixed(2)}`,
    `conf ${o[70].toFixed(2)}`,
  ];
  if (view.obs_version >= 2) parts.push(confirmed ? "confirmed" : "unconfirmed");
  if (!hit) parts.push(`seen ${o[72].toFixed(1)} s ago`);
  return parts.join(" · ");
}
