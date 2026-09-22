// The state graph panel's arithmetic: where the chips go, and what has
// actually fired.
//
// The nodes come from the lab (brain/graph.py, in the world-info message):
// what states exist, what each means, which belong together. The EDGES do
// not come from anywhere — no scripted brain here declares its transitions,
// so drawing an allowed-transition diagram would mean inventing one. They
// are accumulated HERE from the states the frames actually go through, which
// is both honest and more useful: what you watch is this run's trajectory,
// on this pitch, for this duck.
//
// Everything in this file is pure so it can be tested without a canvas; the
// panel paints it per-frame off client.frame, like the ToF heatmap.

export interface GraphNode {
  name: string;
  group: string;
  note: string;
}

/** One brain's declared nodes (brain/graph.py `BrainGraph.to_dict`). */
export interface BrainGraphInfo {
  key: string;
  title: string;
  note: string;
  groups: [key: string, label: string][];
  nodes: GraphNode[];
}

export interface Placed extends GraphNode {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface GroupRow {
  key: string;
  label: string;
  y: number;
}

export interface Layout {
  nodes: Placed[];
  groups: GroupRow[];
  width: number;
  height: number;
}

export interface LayoutOpts {
  width: number;
  chipW: number;
  chipH: number;
  gapX: number;
  gapY: number;
  headH: number;
}

/** Sized for the panel's own content width, and for the longest state name
 *  that exists: `carry_explore` is 13 characters, which is what 74 px of
 *  9 px monospace holds. A narrower chip would ellipsise exactly one node in
 *  one brain.
 *
 *  The width matters more than it looks. The first version of this panel
 *  lived in the 242 px inspector and laid out at 316, so the SVG was scaled
 *  to 70% by `max-width: 100%` — the chips came out at 6 px and the arcs
 *  were invisible on the screenshot. The panel now sizes itself from this
 *  constant instead of the other way round. */
export const LAYOUT: LayoutOpts = { width: 316, chipW: 74, chipH: 20, gapX: 6, gapY: 12, headH: 15 };

/**
 * Group per band, chips wrapping left-to-right inside it. Deterministic:
 * the same graph always lays out the same way, so a node does not move
 * when an edge appears and the eye can learn where "kick" lives.
 */
export function layoutGraph(g: BrainGraphInfo, o: LayoutOpts = LAYOUT): Layout {
  const perRow = Math.max(1, Math.floor((o.width + o.gapX) / (o.chipW + o.gapX)));
  const nodes: Placed[] = [];
  const groups: GroupRow[] = [];
  let y = 0;
  for (const [key, label] of g.groups) {
    groups.push({ key, label, y });
    y += o.headH;
    const inGroup = g.nodes.filter((n) => n.group === key);
    inGroup.forEach((n, i) => {
      const col = i % perRow;
      const row = Math.floor(i / perRow);
      nodes.push({ ...n, x: col * (o.chipW + o.gapX), y: y + row * (o.chipH + o.gapY), w: o.chipW, h: o.chipH });
    });
    const rows = Math.max(1, Math.ceil(inGroup.length / perRow));
    y += rows * (o.chipH + o.gapY);
  }
  return { nodes, groups, width: o.width, height: Math.max(0, y - o.gapY) };
}

export interface Edge {
  from: string;
  to: string;
  count: number;
  /** Sim time this edge last fired — the panel fades an edge by its age. */
  t: number;
}

export interface Visit {
  /** How many times the duck has entered this state. */
  count: number;
  /** Seconds spent in it, summed over every spell (the current one included). */
  total: number;
}

export const edgeKey = (from: string, to: string): string => `${from}>${to}`;

/** How many states the breadcrumb keeps. Twelve is about one approach and
 *  one kick on the pitch — enough to read the loop the duck is stuck in. */
export const TRAIL = 12;
/** Longest gap between two frames that can be a real observation, in WALL
 *  seconds — the frames arrive on the wall clock (every 40 ms), so the budget
 *  is a wall budget and the caller scales it by the world's speed.
 *
 *  Sizing it in SIM seconds was wrong in both directions: a fixed 1 s of sim
 *  is only 0.125 s of wall at 8x (so ordinary jitter reads as a hole) and 4 s
 *  of wall at 0.25x (so a 3 s stall reads as real, booking dwell nobody
 *  watched and an edge that never fired). @see StateTrace.step */
export const MAX_GAP_WALL_S = 1.0;

/**
 * What this duck's brain has actually done, accumulated from the frames.
 *
 * Sim time is not monotonic on this page — R restarts the world and space
 * scrubs into the past — so `step` watches for time going backwards and
 * starts over rather than booking a negative dwell or an edge between two
 * states the duck never walked between.
 */
export class StateTrace {
  current: string | null = null;
  readonly edges = new Map<string, Edge>();
  readonly visits = new Map<string, Visit>();
  /** The last states in order, newest last — the breadcrumb under the graph. */
  trail: string[] = [];
  /** When the current spell began, in sim time. */
  since: number | null = null;
  private last = -Infinity;

  reset(): void {
    this.current = null;
    this.edges.clear();
    this.visits.clear();
    this.trail = [];
    this.since = null;
    this.last = -Infinity;
  }

  /** Seconds in the current state, 0 before the first frame. */
  dwell(t: number): number {
    return this.since === null ? 0 : Math.max(0, t - this.since);
  }

  /**
   * Fold one frame in. Returns true when the drawn SET changed — a new node
   * or a new edge — which is the only time the panel has to re-render
   * instead of just repainting.
   */
  step(state: string | null, t: number, speed = 1): boolean {
    if (state === null) return false;
    // Time went backwards: a restart, or a scrub into the past. Neither is a
    // transition, and a scrub is a view of a trajectory this trace already
    // holds, so start clean rather than stitch the two together.
    if (t < this.last) {
      this.reset();
    }
    // Time jumped FORWARD further than a frame can carry: a reconnect, a
    // stalled tab, a tab that was in the background. We did not watch that
    // interval, so it is not dwell — and whatever state we come back to is
    // not necessarily one transition away from the last one we saw. Keep the
    // history already collected, but re-enter as if this were the first
    // frame: no banked time, and no edge invented across the hole.
    const gap = t - this.last;
    if (Number.isFinite(gap) && gap > MAX_GAP_WALL_S * Math.max(speed, 0.01)) {
      this.current = null;
      this.since = null;
    }
    // Bank the time spent before the state changes, so `total` is dwell and
    // not entry count in disguise.
    if (this.current !== null && this.since !== null && t >= this.last) {
      const v = this.visits.get(this.current);
      if (v) v.total += t - this.last;
    }
    this.last = t;
    if (state === this.current) return false;

    let fresh = false;
    if (!this.visits.has(state)) {
      this.visits.set(state, { count: 0, total: 0 });
      fresh = true;
    }
    this.visits.get(state)!.count += 1;
    if (this.current !== null) {
      const k = edgeKey(this.current, state);
      const e = this.edges.get(k);
      if (e) {
        e.count += 1;
        e.t = t;
      } else {
        this.edges.set(k, { from: this.current, to: state, count: 1, t });
        fresh = true;
      }
    }
    this.current = state;
    this.since = t;
    this.trail = [...this.trail, state].slice(-TRAIL);
    return fresh;
  }
}

/**
 * A curved connector between two chips, as an SVG path.
 *
 * Straight lines between eighteen chips in five bands are unreadable — every
 * edge between adjacent bands lands on top of every other. Bowing each one
 * sideways by an amount that grows with its length spreads them, and bowing
 * upward-going edges the other way from downward-going ones means `chase →
 * lineup` and `lineup → chase` are two visible arcs rather than one line
 * drawn twice.
 */
export function edgePath(a: Placed, b: Placed): string {
  const [x0, y0] = [a.x + a.w / 2, a.y + a.h / 2];
  const [x1, y1] = [b.x + b.w / 2, b.y + b.h / 2];
  const [mx, my] = [(x0 + x1) / 2, (y0 + y1) / 2];
  const [dx, dy] = [x1 - x0, y1 - y0];
  const len = Math.hypot(dx, dy) || 1;
  // Normal to the run, signed so the two directions of a pair bow apart.
  const bow = Math.min(26, len * 0.22) * (dy > 0 || (dy === 0 && dx > 0) ? 1 : -1);
  return `M ${x0} ${y0} Q ${mx - (dy / len) * bow} ${my + (dx / len) * bow} ${x1} ${y1}`;
}

/** A self-transition cannot be a curve between two points: the panel draws a
 *  loop above the chip. Brains do re-enter a state they are already in (the
 *  chase brain re-plans `lineup` without leaving it), and hiding that would
 *  make a busy node look idle. */
export function selfPath(a: Placed): string {
  const [x, y] = [a.x + a.w / 2, a.y];
  return `M ${x - 7} ${y} C ${x - 10} ${y - 13}, ${x + 10} ${y - 13}, ${x + 7} ${y}`;
}

/** Edge opacity: the freshest transitions are the ones worth watching, but a
 *  well-worn edge should not vanish, so age fades toward a floor set by use.
 *
 *  The floor was 0.12 and the arcs were invisible in a screenshot of the real
 *  panel — the chips are opaque and sit close together, so an arc only shows
 *  in the gap between two rows and has to be bright to read there. The row
 *  gap in LAYOUT was opened up for the same reason. */
export function edgeAlpha(e: Edge, t: number, fadeS = 8): number {
  const age = Math.max(0, t - e.t);
  const floor = Math.min(0.8, 0.3 + 0.07 * e.count);
  return Math.max(floor, 1 - age / fadeS);
}

/** How far the panel may be scaled by its resize grip. The floor is set by
 *  the 9 px chip text — below ~0.7 the state names stop being readable, which
 *  defeats the panel — and the ceiling by the room it would otherwise eat. */
export const SCALE_MIN = 0.7;
export const SCALE_MAX = 2.6;

export const clampScale = (s: number): number =>
  !Number.isFinite(s) ? 1 : Math.min(SCALE_MAX, Math.max(SCALE_MIN, s));

/** Width of a run of monospace text, for the opaque patch drawn behind a
 *  group heading. The panel's font is `ui-monospace`, whose advance is very
 *  close to 0.6 em — exact enough for a backing rect, and it avoids a
 *  `getBBox` per label per commit. */
export const labelWidth = (text: string, fontSize: number): number => text.length * fontSize * 0.6;
