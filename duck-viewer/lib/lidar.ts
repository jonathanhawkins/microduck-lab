// The LiDAR instrument's arithmetic: the polar plot's mapping, and the
// lidar -> 8x8 ToF adapter the /sim inspector draws as a strip.
//
// WHY a client-side port of the adapter. The lab hands a wheeled body's
// brains an 8x8 frame ADAPTED from the planar scan
// (`sensors/lidar.tof_from_lidar`, called by `world/arena.World.senses_tof`)
// and deliberately does NOT stream it: that frame is a fiction of 64 bearing
// bins with no elevation and no mount pose, and the page draws a ToF as a
// cone of zone points hung off the head camera's pose, so shipping it would
// draw a sensor the robot does not have, pointing where it is not
// (`world_server.tof_payload` has the whole argument). But the bins are what
// `wander` STEERS ON, so the inspector has to show them — and the only
// honest way to show them without a second wire format is to compute them
// here from the scan the frame already carries, the same way.
//
// So this is a PORT, line for line, of `tof_from_lidar` and
// `tof_column_bearings`, and the vitest cases assert the numbers the Python
// itself prints for the same synthetic scans (lib/lidar.test.ts names them).
// If the two ever part company, the strip is a picture of a controller that
// does not exist.

import {
  TOF_COLS,
  TOF_FOV_DEG,
  TOF_MAX_RANGE_M,
  TOF_MIN_RANGE_M,
  type LidarPayload,
  type SimDuck,
} from "./sim";

/** FALLBACK floor (`sensors/lidar.DEFAULT_MIN_RANGE_M`), for a lab that
 *  predates `LidarPayload.minRange`. Prefer `lidarMinRange(scan)`: the frame
 *  now carries the device's own, and a scanner with a different floor drew
 *  the wrong blind disc for as long as this constant was the only answer. */
export const LIDAR_MIN_RANGE_M = 0.15;
/** Fallback scale when a lab sends no `maxRange` (`DEFAULT_MAX_RANGE_M`). */
export const LIDAR_MAX_RANGE_M = 6.0;
/** The scanner's rate (`DEFAULT_RATE_HZ`), and the freshness window built on
 *  it: two periods (167 ms each) plus a 40 ms frame tick of slack, so a scan
 *  is "fresh" for as long as the next one can legitimately be in flight. The
 *  ToF's own window is 150 ms for the same reason at 15 Hz. */
export const LIDAR_RATE_HZ = 6;
export const LIDAR_FRESH_MS = Math.round((2 * 1000) / LIDAR_RATE_HZ) + 40;

/** The claw's numbers (`robots/mars.HOLD_LOAD_NM` / `GRIPPER_EFFORT_LIMIT`):
 *  the hold threshold the bar marks, and the servo's own torque clamp as the
 *  bar's full scale. A payload may override both. */
export const GRIPPER_HOLD_NM = 1.0;
export const GRIPPER_LIMIT_NM = 2.0;

/** Returns closer than this to the BASE origin are the robot looking at
 *  itself, dropped by the adapter (`robots/mars.FOOTPRINT_M`, and the lab
 *  passes it: `World.senses_tof` reads `WorldRobot.footprint_m`).
 *
 *  MEASURED there: at `ARM_HOME` the folded arm blocks 4 of 360 rays at 7-10
 *  degrees from 0.156 m off `link5`, which in the base frame is ~0.08 m — so
 *  a brain reading those as an obstacle steers away from its own elbow for
 *  the whole run.
 *
 *  **A FALLBACK, keyed by robot id, and the reason `lidarFootprintM` exists.**
 *  A table here can only answer for a body this build has heard of: add a
 *  second wheeled robot and it keeps every return off ITS folded arm and
 *  draws them as obstacles, which is exactly the bug the number was measured
 *  to kill. The frame carries the body's own since c858568 — prefer it. */
const FOOTPRINT_M: Record<string, number> = { mars: 0.12 };
export function robotFootprintM(robot: string | undefined | null): number {
  return (robot && FOOTPRINT_M[robot]) || 0;
}

/** The blind disc to draw: what THIS device said, else the fallback above. */
export function lidarMinRange(f: Pick<LidarPayload, "minRange"> | null | undefined): number {
  return f?.minRange && f.minRange > 0 ? f.minRange : LIDAR_MIN_RANGE_M;
}

/** How much of the scan is the robot's own arm: what the BODY declared on
 *  the frame, else this build's table for a lab that predates the field.
 *
 *  `??` and not `||`, because **0.0 is an answer**: the lab sends it for a
 *  body that declared no footprint, meaning "keep every return"
 *  (`tof_from_lidar`'s own default). Falling through to the robot-id table
 *  there would drop returns the lab deliberately kept. */
export function lidarFootprintM(
  f: Pick<LidarPayload, "footprint"> | null | undefined,
  robot?: string | null,
): number {
  return f?.footprint ?? robotFootprintM(robot);
}

/** Which range sensor a body HAS — the question the inspector must answer
 *  before it draws a placeholder, because "no ToF frame yet" on a robot with
 *  no ToF is a lie about the hardware and not a note about the stream.
 *
 *  The body's own DECLARATION first (`SimDuck.tof` / `SimDuck.lidar`, the
 *  noise preset per device, which the lab sends whether or not a frame has
 *  arrived), then the live `sensors` keys, so a lab that predates the preset
 *  fields still lands on the right block. */
export type RangeChannel = "tof" | "lidar";
export function rangeChannel(d: SimDuck | null | undefined): RangeChannel | null {
  if (!d) return null;
  if (d.tof) return "tof";
  if (d.lidar) return "lidar";
  if (d.sensors?.tof) return "tof";
  if (d.sensors?.lidar) return "lidar";
  return null;
}

/** The label on the overlay toggle: it draws whatever the body's range sensor
 *  is, so it says so. A body with neither keeps the duck's wording — the
 *  toggle also owns the detector rays and the chase beliefs, and a button
 *  that renames itself to nothing reads as broken. */
export function sensorOverlayLabel(chan: RangeChannel | null): string {
  return chan === "lidar" ? "LiDAR overlay" : "ToF overlay";
}

/** One decoded ray of a scan. `mm === 0` is skipped upstream of this. */
export interface LidarPoint {
  /** Index into `mm`, so a caller can mark the ray a bin picked. */
  i: number;
  /** Bearing in the MOUNT frame, rad CCW from the robot's +x. */
  a: number;
  /** Range in metres, as measured from the LASER. */
  r: number;
  /** px in the plot box, forward UP. */
  x: number;
  y: number;
  /** True where the range saturated the plot's scale rather than fitting it. */
  clamped: boolean;
}

/** Where a (bearing, range) lands in a `size` px plot box, robot at the
 *  centre, a return at `maxRange` on the inscribed circle.
 *
 *  The bearings are the ROBOT's, CCW from its own +x (`sensors/ray.planar_fan`
 *  with `ccw`, Innate's convention), so the plot is drawn in the robot's
 *  frame: **forward is UP**, its left is left, and what is behind it is
 *  below. Plotting +x rightward instead would be a world-frame picture of a
 *  robot-frame measurement — it would only look right while the robot faced
 *  +x, and would lie silently the moment it turned. */
export function lidarPolar(a: number, rM: number, maxRange: number, size: number): { x: number; y: number } {
  const r0 = size / 2;
  const r = (Math.min(Math.max(rM, 0), maxRange) / maxRange) * r0;
  // Forward (a = 0) is up; +a is CCW, which on a forward-up plot is left.
  return { x: r0 - r * Math.sin(a), y: r0 - r * Math.cos(a) };
}

/** The scan as points in a `size` px box, forward up.
 *
 *  Skips every **0**, which is the payload's "no reading" and not a
 *  zero-range one (`sensors/lidar.LidarFrame.as_payload`): drawing those puts
 *  a dense blob of false contacts on the robot's own origin, which is the
 *  most misleading thing a range drawing can do. A reading longer than
 *  `maxRange` is CLAMPED rather than dropped — the direction is real even
 *  where the distance saturates. */
export function lidarPlotPoints(
  f: Pick<LidarPayload, "a0" | "da" | "mm" | "maxRange">,
  size: number,
): LidarPoint[] {
  const max = lidarMaxRange(f);
  const out: LidarPoint[] = [];
  f.mm.forEach((mm, i) => {
    if (!mm) return; // 0 = no reading
    const r = mm / 1000;
    const a = f.a0 + i * f.da;
    out.push({ i, a, r, ...lidarPolar(a, r, max, size), clamped: r > max });
  });
  return out;
}

/** The scale a plot is drawn at: what the lab said, else the device default. */
export function lidarMaxRange(f: Pick<LidarPayload, "maxRange">): number {
  return f.maxRange && f.maxRange > 0 ? f.maxRange : LIDAR_MAX_RANGE_M;
}

/** The nearest real return: bearing and range in the LASER's frame.
 *
 *  A ray at `maxRange` is NOT a hit — it is the LaserScan convention's
 *  "nothing within range" (`LidarSensor.scan`: a miss reports `max_range`
 *  and stays valid, which is information and not a wall). Reporting one as
 *  the nearest contact would put a 6 m obstacle in the readout of a robot
 *  standing in an empty field. */
export function lidarNearest(
  f: Pick<LidarPayload, "a0" | "da" | "mm" | "maxRange">,
): { i: number; bearing: number; rangeM: number } | null {
  const max = lidarMaxRange(f);
  let best = -1;
  let bestR = Infinity;
  for (let i = 0; i < f.mm.length; i++) {
    const mm = f.mm[i];
    if (!mm) continue;
    const r = mm / 1000;
    if (r >= max - 1e-9) continue; // a miss, not a contact
    if (r < bestR) {
      bestR = r;
      best = i;
    }
  }
  return best < 0 ? null : { i: best, bearing: f.a0 + best * f.da, rangeM: bestR };
}

/** The (cols + 1) bearing EDGES of a ToF frame's columns, rad, DESCENDING
 *  (column 0 is the leftmost, so its edges are the largest bearings).
 *
 *  A port of `sensors/lidar.tof_column_bearings`, which reads them off
 *  `sensors/ray.tof_fan`'s own geometry rather than assuming even angles:
 *  the zone centres lie on the TANGENT PLANE at x = 1, evenly in y between
 *  ±tan(fov/2), so a column's edges are `atan` of the plane coordinates that
 *  bound it. That is NOT evenly spaced in angle — the outer columns of a 45°
 *  matrix are ~0.4° narrower than the middle ones — and using even angles
 *  would put a hit in the wrong column near the edges, which is the whole
 *  thing the adapter exists to get right. */
export function tofColumnEdges(cols = TOF_COLS, fovDeg = TOF_FOV_DEG): number[] {
  const half = Math.tan((fovDeg * Math.PI) / 360);
  const out: number[] = [];
  for (let k = 0; k <= cols; k++) out.push(Math.atan(half - (2 * half * k) / cols));
  return out;
}

/** What one adapted column reports. `ray` is which scan ray won it, so the
 *  plot can mark the eight returns the brain is actually steering on. */
export interface TofColumn {
  /** Millimetres in the ToF's convention — **0 = no target**. */
  mm: number;
  /** The same in metres, null where the column is empty. */
  rangeM: number | null;
  /** The winning ray's bearing re-read in the BASE frame, rad. */
  bearing: number | null;
  /** Index into the scan's `mm`, or -1. */
  ray: number;
}

export interface LidarTof {
  /** One row of the 8x8 — a planar scan has nothing to say about elevation,
   *  so every row of the frame the brain gets is this row repeated. */
  cols: TofColumn[];
  /** The column edges these bins were cut with, rad, descending. */
  edges: number[];
  /** Per scan ray: the column it was binned into, or -1 (dropped as invalid,
   *  out of the ToF's range band, inside the footprint, or outside the FOV).
   *  The overlay tints the sector with exactly this, so the picture in the
   *  room and the strip in the panel cannot disagree. */
  rayCol: number[];
  /** How many rays landed in a column at all — the sector's own ray count. */
  used: number;
}

/**
 * A planar scan, as the 8 bearing columns the duck's ToF brains read.
 *
 * A PORT of `sensors.lidar.tof_from_lidar` (see the module note above), with
 * the wire's convention for validity: **`mm === 0` is no reading**, an
 * invalid ray and not a zero-range one, which is exactly the `frame.valid`
 * mask the Python reduces over.
 *
 * The three things it does, in the Python's order:
 *
 * 1. **The 76 mm is applied.** MARS's scanner sits 76.4 mm BEHIND the base
 *    origin (`mount`, in the base's HEADING frame), so a ray's (bearing,
 *    range) is in the LASER's frame and a consumer reading it as the base's
 *    would place every obstacle further away than it is — a 1.92 m wall reads
 *    2.00 m. Each ray becomes a point, is offset by `mount`, and is re-read
 *    as a base-frame bearing and range. No `mount` = already base-framed.
 * 2. **The band is the ToF's**, not the scanner's: a return outside
 *    [0.02, 4.0] m is "no target", which is why a 6 m miss fills nothing and
 *    a 4.5 m wall inside the scan is invisible to these brains.
 * 3. **The nearest valid return in each column wins**, and `footprintM`
 *    drops what is inside the robot first (the folded arm — the caller
 *    reads it off the frame with `lidarFootprintM`).
 *
 * Columns are half-open `(lo, hi]` so a ray exactly on an internal edge lands
 * in exactly one of them; column 0's upper edge is nudged instead, because a
 * ray at exactly +fov/2 belongs in the frame rather than nowhere.
 */
export function tofFromLidar(
  f: Pick<LidarPayload, "a0" | "da" | "mm" | "mount">,
  opts: { footprintM?: number; cols?: number; fovDeg?: number; minRangeM?: number; maxRangeM?: number } = {},
): LidarTof {
  const cols = opts.cols ?? TOF_COLS;
  const edges = tofColumnEdges(cols, opts.fovDeg ?? TOF_FOV_DEG);
  const minR = opts.minRangeM ?? TOF_MIN_RANGE_M;
  const maxR = opts.maxRangeM ?? TOF_MAX_RANGE_M;
  const foot = opts.footprintM ?? 0;
  const mx = f.mount ? f.mount[0] : 0;
  const my = f.mount ? f.mount[1] : 0;
  const n = f.mm.length;
  const rayCol = new Array<number>(n).fill(-1);
  const out: TofColumn[] = Array.from({ length: cols }, () => ({ mm: 0, rangeM: null, bearing: null, ray: -1 }));
  let used = 0;
  for (let i = 0; i < n; i++) {
    const mm = f.mm[i];
    if (!mm) continue; // 0 = no reading
    const a = f.a0 + i * f.da;
    const r = mm / 1000;
    const x = mx + r * Math.cos(a);
    const y = my + r * Math.sin(a);
    const bear = Math.atan2(y, x);
    const rng = Math.hypot(x, y);
    if (rng < minR || rng > maxR) continue;
    if (foot > 0 && rng <= foot) continue;
    // Which column: half-open (lo, hi], column 0's hi nudged out.
    let col = -1;
    for (let c = 0; c < cols; c++) {
      const hi = c === 0 ? edges[0] + 1e-12 : edges[c];
      if (bear > edges[c + 1] && bear <= hi) {
        col = c;
        break;
      }
    }
    if (col < 0) continue;
    rayCol[i] = col;
    used++;
    const slot = out[col];
    if (slot.rangeM === null || rng < slot.rangeM) {
      slot.rangeM = rng;
      // The wire's own rounding, so the strip reads the millimetres the
      // brain's frame carries (`depth[:, c] = round(rng * 1000)`).
      slot.mm = Math.round(rng * 1000);
      slot.bearing = bear;
      slot.ray = i;
    }
  }
  return { cols: out, edges, rayCol, used };
}

/** The claw's bar: where the needle sits, where the threshold mark goes, and
 *  whether this counts as holding.
 *
 *  `holding` is the LAB's predicate when it sends one — `|load| >= 1.0 N·m`
 *  **and** a blade contact with a body that is not part of the robot
 *  (`MarsDriver.held_body`). The threshold alone cannot answer it: closing on
 *  air drives the blade into its own hard stop with zero position error and
 *  reads 0.0 N·m, identical to an open claw, so a bar past the mark is
 *  necessary and not sufficient. Without a `holding` flag this falls back to
 *  the torque alone and says so in the panel's title. */
export function gripperBar(
  g: { load: number; holding?: boolean; limit?: number; hold?: number },
): { frac: number; markFrac: number; holding: boolean; inferred: boolean; limit: number; hold: number } {
  const limit = g.limit && g.limit > 0 ? g.limit : GRIPPER_LIMIT_NM;
  const hold = g.hold && g.hold > 0 ? g.hold : GRIPPER_HOLD_NM;
  const mag = Math.abs(g.load);
  return {
    frac: Math.max(0, Math.min(1, mag / limit)),
    markFrac: Math.max(0, Math.min(1, hold / limit)),
    holding: g.holding ?? mag >= hold,
    inferred: g.holding === undefined,
    limit,
    hold,
  };
}

/** One arm joint's bar: the achieved angle as a fraction of its travel, with
 *  the commanded target as a second mark — the pair IS the sag the brain
 *  pre-compensates (`Senses.arm`: 0.067 rad at joint2 is 28.5 mm of claw
 *  height). No limits: ±π, which is honest about being a default. */
export function armBar(q: number, limits?: [number, number]): number {
  const [lo, hi] = limits && limits[1] > limits[0] ? limits : [-Math.PI, Math.PI];
  return Math.max(0, Math.min(1, (q - lo) / (hi - lo)));
}

/** How far two neighbouring returns may differ in range and still be the same
 *  SURFACE, m. Under it the scan is drawn as a continuous edge; over it the
 *  two are separate things and the outline breaks.
 *
 *  MEASURED in the follow-me room: the neighbour-to-neighbour range step is
 *  under 4 cm along a wall (a 1° step at 3 m across a flat surface is 5 cm at
 *  60° of incidence) and jumps by 0.4 m or more at a doorway, a chair leg or
 *  the far side of the person. 0.25 m sits in that gap with room either side,
 *  so a wall reads as one line and a person does not get joined to the wall
 *  behind them. */
export const SCAN_GAP_M = 0.25;

/** The room OUTLINE a scan draws: index pairs to join, wrapping the turn.
 *
 *  This is what a planar scanner looks like in every viewer that has one
 *  (RViz's LaserScan, a robot vacuum's map): not a fan of rays, which reads
 *  as fog and hides the shape, but the SILHOUETTE the beam traced — the walls
 *  and the legs of things, closing around the robot.
 *
 *  A pair is joined only when both ends are real returns AND their ranges
 *  agree within `gapM`. That is the whole cleverness and it is load-bearing:
 *  joining every neighbour draws a chord straight across a doorway and makes
 *  a room look sealed, which is a lie about the one thing this instrument is
 *  for. Misses (`maxRange`) and no-readings (0) are not ends of anything.
 *
 *  Returns flat pairs `[i0, i1, i0, i1, …]` so the caller can walk them into
 *  a segment buffer without allocating per edge. */
export function scanOutline(
  mm: readonly number[],
  maxRange: number,
  gapM: number = SCAN_GAP_M,
): number[] {
  const n = mm.length;
  const out: number[] = [];
  if (n < 2) return out;
  const real = (i: number) => {
    const v = mm[i];
    return v > 0 && v / 1000 < maxRange - 1e-9;
  };
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    if (!real(i) || !real(j)) continue;
    if (Math.abs(mm[i] - mm[j]) / 1000 > gapM) continue;
    out.push(i, j);
  }
  return out;
}
