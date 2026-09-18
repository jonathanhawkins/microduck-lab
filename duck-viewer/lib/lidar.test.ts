// The LiDAR instrument's arithmetic: the polar mapping the /sim inspector
// plots with, the client-side port of the lab's lidar → 8×8 ToF adapter, the
// channel selection that decides WHICH sensor block a body gets, and the
// gripper bar's threshold.
//
// THE HAND TABLE IS THE PYTHON'S OWN. Every expected bin below was printed by
// running the real `sensors.lidar.tof_from_lidar` on the same synthetic scan
// (`uv run python` against `microduck_local`, scans built the way the WIRE
// gives them: millimetre ints, 0 = no reading). A port that drifts from the
// adapter draws a picture of a controller that does not exist, so the numbers
// are pinned rather than recomputed from the same formula on both sides —
// which is the mistake `docs/mars-roadmap.md` 4b caught four times in one day.

import { describe, expect, it } from "vitest";

import {
  armBar,
  gripperBar,
  lidarNearest,
  lidarPlotPoints,
  lidarPolar,
  LIDAR_MIN_RANGE_M,
  rangeChannel,
  robotFootprintM,
  sensorOverlayLabel,
  tofColumnEdges,
  tofFromLidar,
} from "./lidar";
import type { SimDuck } from "./sim";

const scan = (mm: number[], a0 = 0, da = Math.PI / 2, maxRange = 6) => ({ a0, da, mm, maxRange });

describe("lidarPolar / lidarPlotPoints: the plot is in the ROBOT's frame", () => {
  it("puts the robot's FORWARD at the top of the drawing", () => {
    // MuJoCo bearings are CCW from +x and a canvas's y grows downward. A
    // return dead ahead at half range must land ABOVE the centre; getting the
    // sign wrong draws a room mirrored front-to-back, which reads as
    // plausible and is the worst kind of wrong.
    const [p] = lidarPlotPoints(scan([3000]), 100);
    expect(p.x).toBeCloseTo(50, 6);
    expect(p.y).toBeCloseTo(25, 6); // above centre (y = 50)
  });

  it("rebuilds every bearing from a0 + i·da", () => {
    // 4 rays a quarter turn apart: ahead, left, behind, right.
    const pts = lidarPlotPoints(scan([6000, 6000, 6000, 6000]), 100);
    expect(pts).toHaveLength(4);
    expect(pts[0].x).toBeCloseTo(50, 6); // ahead → top
    expect(pts[0].y).toBeCloseTo(0, 6);
    expect(pts[1].x).toBeCloseTo(0, 6); // the robot's LEFT → left
    expect(pts[1].y).toBeCloseTo(50, 6);
    expect(pts[2].y).toBeCloseTo(100, 6); // behind → bottom
    expect(pts[3].x).toBeCloseTo(100, 6); // the robot's right → right
    expect(pts.map((p) => p.i)).toEqual([0, 1, 2, 3]);
  });

  it("drops a 0 — that is NO READING, not a contact on the robot", () => {
    // The payload's own convention (sensors/lidar.LidarFrame.as_payload).
    // Drawing them puts a blob of false contacts on the origin.
    expect(lidarPlotPoints(scan([0, 0, 0, 0]), 100)).toEqual([]);
    expect(lidarPlotPoints(scan([0, 3000]), 100)).toHaveLength(1);
    expect(lidarPlotPoints(scan([0, 3000]), 100)[0].i).toBe(1); // …and keeps its INDEX
  });

  it("clamps past max range instead of dropping the direction, and says it clamped", () => {
    const [p] = lidarPlotPoints(scan([99000]), 100);
    expect(p.y).toBeCloseTo(0, 6); // on the inscribed circle, not beyond it
    expect(p.clamped).toBe(true);
    expect(lidarPlotPoints(scan([3000]), 100)[0].clamped).toBe(false);
  });

  it("falls back to a 6 m scale for a lab that sends no maxRange", () => {
    const [p] = lidarPlotPoints({ a0: 0, da: 0, mm: [6000] }, 100);
    expect(p.y).toBeCloseTo(0, 6);
  });

  it("scales the blind disc by the same map the points use", () => {
    // The device's floor is 0.15 m (`sensors/lidar.DEFAULT_MIN_RANGE_M`), and
    // it is pinned as a LITERAL: the first cut of this case read the constant
    // on both sides of the comparison, so it agreed with itself whatever the
    // constant said and was the one planted break of 24 that got through.
    expect(LIDAR_MIN_RANGE_M).toBe(0.15);
    const q = lidarPolar(0, LIDAR_MIN_RANGE_M, 6, 200);
    expect(100 - q.y).toBeCloseTo((0.15 / 6) * 100, 6); // 2.5 px on a 6 m plot
  });
});

describe("lidarNearest: the nearest CONTACT, not the nearest number", () => {
  it("finds the closest real return and reports its bearing", () => {
    const n = lidarNearest(scan([5000, 1200, 3000, 900]));
    expect(n).not.toBeNull();
    expect(n!.i).toBe(3);
    expect(n!.rangeM).toBeCloseTo(0.9, 6);
    expect(n!.bearing).toBeCloseTo(3 * (Math.PI / 2), 6);
  });

  it("never reports a max-range MISS as a contact", () => {
    // `LidarSensor.scan`: a ray that hit nothing reports max_range and stays
    // VALID — "nothing within 6 m" is information, not a wall at 6 m. A
    // readout that called it the nearest obstacle would put a phantom wall in
    // front of a robot standing in an empty field.
    expect(lidarNearest(scan([6000, 6000, 6000, 6000]))).toBeNull();
    expect(lidarNearest(scan([6000, 5999, 6000, 6000]))!.i).toBe(1);
    expect(lidarNearest(scan([0, 0]))).toBeNull();
  });
});

describe("tofColumnEdges: the ToF's own columns, not even angles", () => {
  it("matches sensors/lidar.tof_column_bearings to six places", () => {
    // Printed by the Python itself for TofSpec() (8 columns, 45°):
    const py = [0.392699, 0.301208, 0.20422, 0.103186, 0.0, -0.103186, -0.20422, -0.301208, -0.392699];
    const ts = tofColumnEdges();
    expect(ts).toHaveLength(9);
    ts.forEach((v, i) => expect(v).toBeCloseTo(py[i], 6));
  });

  it("is NOT evenly spaced — the outer columns are narrower", () => {
    // The zone centres lie on the tangent plane at x = 1, evenly in y, so a
    // 45° matrix's outermost column spans 5.24° against the middle one's
    // 5.91°. Even angles (5.625° each) would put a hit in the wrong column
    // near the edges, which is the whole thing the adapter exists to get
    // right — so the test pins the asymmetry, not just the total.
    const e = tofColumnEdges();
    const outer = (e[0] - e[1]) * 57.2958;
    const middle = (e[3] - e[4]) * 57.2958;
    expect(outer).toBeCloseTo(5.242, 3);
    expect(middle).toBeCloseTo(5.912, 3);
    expect(middle - outer).toBeCloseTo(0.67, 2);
  });
});

describe("tofFromLidar: the port equals the lab's adapter", () => {
  // A 360-ray scan of an infinite wall 2 m ahead, exactly as the wire sends
  // it: 1 mm ints, 0 = no reading, a ray that reaches nothing at 6 m.
  const N = 360;
  const wall = (): number[] =>
    Array.from({ length: N }, (_, i) => {
      const a = (i * 2 * Math.PI) / N;
      const ca = Math.cos(a);
      if (ca <= 1e-6) return 6000;
      const d = 2 / ca;
      return d <= 6 ? Math.round(d * 1000) : 6000;
    });
  const a0 = 0;
  const da = (2 * Math.PI) / N;

  it("bins a wall 2 m ahead the way the Python does, to the millimetre", () => {
    // `uv run python` on sensors.lidar.tof_from_lidar, mount None:
    const py = [2103, 2045, 2011, 2000, 2000, 2011, 2045, 2103];
    const out = tofFromLidar({ a0, da, mm: wall() });
    expect(out.cols.map((c) => c.mm)).toEqual(py);
    // The middle columns read the wall's own 2 m; the outer ones read the
    // wall where it is FURTHER, because a wall at a bearing is 2/cos θ away —
    // not "max", which is what an evenly-spaced guess would give.
    expect(out.cols[3].mm).toBe(2000);
    expect(out.cols[0].mm).toBeGreaterThan(out.cols[3].mm);
  });

  it("applies the 76 mm mount offset — the third of a robot length", () => {
    // Same scan, mount = MARS's own [-0.0764, 0, 0.1716] (the laser sits
    // BEHIND the base origin, so the wall is NEARER in the base frame):
    const py = [2018, 1970, 1935, 1924, 1924, 1935, 1970, 2018];
    const out = tofFromLidar({ a0, da, mm: wall(), mount: [-0.0764, 0, 0.1716] });
    expect(out.cols.map((c) => c.mm)).toEqual(py);
    expect(2000 - out.cols[3].mm).toBe(76); // the offset, to the millimetre
  });

  it("reports a scan of nothing but misses as NO TARGET, not as 6 m", () => {
    // A miss is a valid reading at max_range, and max_range is outside the
    // ToF's own 4 m band — so these brains correctly see an empty frame.
    const out = tofFromLidar({ a0, da, mm: Array(N).fill(6000) });
    expect(out.cols.map((c) => c.mm)).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
    expect(out.cols.every((c) => c.ray === -1 && c.rangeM === null)).toBe(true);
    expect(out.used).toBe(0);
  });

  it("drops what is outside the ToF's band, both ends", () => {
    // 4.5 m is inside the 6 m scan and outside the 4 m ToF; 1 cm is under the
    // 0.02 m floor. Python prints all zeros for this frame.
    const mm = Array(N).fill(0);
    mm[0] = 4500;
    mm[1] = 10;
    expect(tofFromLidar({ a0, da, mm }).cols.map((c) => c.mm)).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it("keeps the robot's own arm out of the frame only when a footprint is given", () => {
    // MEASURED on MARS (`sensors/lidar.py`): the folded arm blocks 4 rays at
    // 7-10° from ~0.156 m, which in the base frame is ~0.08 m. It is NEARER
    // than the 1 m post in the same column, so without the footprint the
    // column reads the elbow and `wander` steers away from its own arm —
    // exactly what `FOOTPRINT_M` is for. Both numbers are the Python's.
    const mm = Array(N).fill(0);
    mm[10] = 1000; // a post 1 m away, 10° left
    mm[8] = 90; // the arm, 8° left, inside the chassis
    expect(tofFromLidar({ a0, da, mm }).cols.map((c) => c.mm)).toEqual([0, 0, 90, 0, 0, 0, 0, 0]);
    expect(tofFromLidar({ a0, da, mm }, { footprintM: 0.12 }).cols.map((c) => c.mm))
      .toEqual([0, 0, 1000, 0, 0, 0, 0, 0]);
    expect(robotFootprintM("mars")).toBe(0.12);
    expect(robotFootprintM("microduck")).toBe(0); // no declaration = keep every return
  });

  it("puts a ray on a column edge in exactly one column", () => {
    // Half-open (lo, hi]: a ray exactly ON an internal edge belongs to the
    // column BELOW it, and a hair above it belongs to the next one out. The
    // edge is taken from `tofColumnEdges` rather than written as a rounded
    // degree, because the Python's `angles` are float32 and a decimal
    // degree's float32 rounding decides the case on its own (5.9121° bins one
    // way there and the other way here, at 4e-7 rad — a property of the cast,
    // not of the rule).
    const atRad = (a: number) => tofFromLidar({ a0: a, da: 0, mm: [1500] }).cols.map((c) => c.mm);
    const e = tofColumnEdges();
    expect(atRad(e[3])).toEqual([0, 0, 0, 1500, 0, 0, 0, 0]);
    expect(atRad(e[3] + 1e-6)).toEqual([0, 0, 1500, 0, 0, 0, 0, 0]);
    // Dead ahead is column 4, NOT column 3 — the Python's own answer for a
    // single-ray frame at 0°, and the FOV's edge keeps the ray in the frame.
    expect(atRad(0)).toEqual([0, 0, 0, 0, 1500, 0, 0, 0]);
    expect(atRad(e[0])).toEqual([1500, 0, 0, 0, 0, 0, 0, 0]);
    expect(atRad((22.6 * Math.PI) / 180)).toEqual([0, 0, 0, 0, 0, 0, 0, 0]); // outside the FOV
  });

  it("marks which rays the sector claimed, so plot, strip and overlay agree", () => {
    const out = tofFromLidar({ a0, da, mm: wall() });
    // 45° of a 1°-per-ray fan, and every column's winner is a real ray.
    expect(out.used).toBe(45);
    // Every column's winning ray is tagged with that same column, which is
    // what lets the room's overlay tint exactly the rays the strip shows.
    for (const [c, col] of out.cols.entries()) {
      expect(col.ray).toBeGreaterThanOrEqual(0);
      expect(out.rayCol[col.ray]).toBe(c);
    }
    // Everything behind the robot is dropped, not silently binned.
    expect(out.rayCol[180]).toBe(-1);
    expect(out.rayCol.filter((c) => c >= 0)).toHaveLength(45);
  });
});

// ONE LIVE SCAN, as the wire gave it: the MARS in `mars-playroom` on
// 2026-09-18, `GET /replay/ring` frame t=975.18. Its 8 adapted columns were
// computed by the LAB's own Python (`sensors.lidar.tof_from_lidar` with
// `footprint_m=0.12`, the value `World.senses_tof` passes for a MARS) and are
// pinned below. The synthetic cases above prove the rules one at a time; this
// one proves the whole port against the real adapter on real data, including
// the 4 rays the folded arm blocks and the two dropped returns.
const LIVE_MM = (
  "552,526,510,512,511,490,477,471,164,167,150,0,0,428,415,406,381,398,386,368,371,371,370,360," +
  "357,357,346,347,357,349,338,338,339,341,322,326,317,309,311,319,319,322,310,308,300,304,312," +
  "306,314,311,305,288,297,290,310,303,295,302,303,301,296,304,300,307,306,295,303,303,304,313," +
  "303,312,303,307,316,309,312,324,313,317,321,325,327,340,324,339,348,333,335,357,339,374,377," +
  "373,366,386,376,397,385,383,410,418,427,428,433,467,461,471,486,487,495,510,526,518,544,573," +
  "577,606,621,630,652,683,710,735,730,788,833,857,919,943,1002,1081,1140,1240,1320,1443,1546,1" +
  "745,1893,2110,2423,2540,2539,2527,2532,2532,2538,2510,2516,2532,2527,2538,2519,2571,2532,255" +
  "2,2565,2588,2556,2587,2613,2610,2621,2645,2622,2637,2673,2686,2711,2741,2755,2810,2800,2763," +
  "2842,2864,2912,2896,2956,2976,3013,3041,3102,3139,3143,3162,3252,3308,3303,3264,3203,3159,30" +
  "93,3029,3004,2969,2915,2882,2785,2763,2758,2713,2666,2643,2596,2558,2545,2499,2483,2466,2451" +
  ",2444,2429,2385,2358,2386,2332,2320,2317,2339,2278,2275,2258,2284,2254,2219,2214,2224,2220,2" +
  "204,2177,2193,2186,2188,2165,2189,2195,2196,2190,2186,2181,2170,2199,2200,2207,2213,2250,223" +
  "5,2209,2194,2034,1870,1745,1643,1541,1459,1395,1339,1277,1217,1167,1096,1071,1036,1000,966,9" +
  "17,908,895,839,836,807,795,770,744,738,727,698,691,676,672,652,639,626,625,616,614,588,593,5" +
  "69,567,552,561,533,534,539,529,523,520,521,516,492,505,476,483,481,483,486,469,468,466,460,4" +
  "62,462,461,461,454,468,460,458,459,461,458,443,454,452,451,452,452,462,452,463,447,449,442,4" +
  "51,459,466,468,467,470,465,468,470,472,475,482,480,478,483,478,499,486,498,499,512,518,526,5" +
  "18,527"
).split(",").map(Number);
const LIVE_SCAN = { t: 975.18, a0: 0.0, da: 0.017453, mm: LIVE_MM,
  mount: [-0.0764, -0.0, 0.1716] as [number, number, number],
  maxRange: 6.0 };
const LIVE_PYTHON_COLS = [308, 341, 395, 434, 436, 410, 403, 396];

describe("the port on a live /sim scan", () => {
  it("bins it into the same 8 columns the lab's Python did", () => {
    expect(LIVE_MM).toHaveLength(360);
    const out = tofFromLidar(LIVE_SCAN, { footprintM: 0.12 });
    expect(out.cols.map((c) => c.mm)).toEqual(LIVE_PYTHON_COLS);
    // …and WITHOUT the footprint the robot's own arm wins two columns: the
    // reason `World.senses_tof` passes one at all.
    const bare = tofFromLidar(LIVE_SCAN).cols.map((c) => c.mm);
    expect(bare).not.toEqual(LIVE_PYTHON_COLS);
    expect(Math.min(...bare.filter((x) => x))).toBeLessThan(120);
  });
});

describe("rangeChannel: which instrument does this body HAVE", () => {
  const duck = (over: Partial<SimDuck>): SimDuck =>
    ({ id: "d0", name: "d0", policy: null, falls: 0, step: 0, rew: 0, speed: 0, cmdSpeed: 0,
       steerable: true, tof: null, detector: null, holding: null, skill: null, beak: "open",
       brain: { kind: "wander", state: "steer", cmd: [0, 0, 0], inputs: {} }, headApplied: false,
       bodies: [], sensors: null, ...over }) as SimDuck;

  it("gives a MARS the lidar block and NO ToF placeholder", () => {
    const mars = duck({
      robot: "mars", tof: null, lidar: "datasheet",
      sensors: { lidar: { t: 1, a0: 0, da: 0.017453, mm: [600], maxRange: 6 } },
    });
    expect(rangeChannel(mars)).toBe("lidar");
  });

  it("gives a duck the ToF block, scan or no scan", () => {
    const d = duck({ tof: "datasheet", sensors: { tof: { t: 1, mm: Array(64).fill(500), age: 0 } } });
    expect(rangeChannel(d)).toBe("tof");
    // …and BEFORE the first frame: the preset is the body's declaration, so
    // the grid (and its "no ToF frame yet") is up from the first render.
    expect(rangeChannel(duck({ tof: "ideal", sensors: null }))).toBe("tof");
  });

  it("reads the live sensor keys when a lab sends no presets", () => {
    expect(rangeChannel(duck({ sensors: { lidar: { t: 1, a0: 0, da: 0.1, mm: [600] } } }))).toBe("lidar");
    expect(rangeChannel(duck({ sensors: { tof: { t: 1, mm: [500], age: 0 } } }))).toBe("tof");
  });

  it("says NEITHER for a body with no range sensor at all", () => {
    // A blind duck, or a Menagerie body with no sensors: the block must say
    // so rather than show an empty ToF grid it does not have.
    expect(rangeChannel(duck({ sensors: { det: { t: 1, age: 0, items: [] } } }))).toBeNull();
    expect(rangeChannel(duck({}))).toBeNull();
    expect(rangeChannel(null)).toBeNull();
  });

  it("names the overlay after the channel it draws", () => {
    expect(sensorOverlayLabel("lidar")).toBe("LiDAR overlay");
    expect(sensorOverlayLabel("tof")).toBe("ToF overlay");
    // Nothing selected: the toggle still owns the detector rays and the chase
    // beliefs, so it keeps the duck's wording rather than going blank.
    expect(sensorOverlayLabel(null)).toBe("ToF overlay");
  });
});

describe("gripperBar: closed is not holding", () => {
  it("marks the 1.0 N·m threshold on the servo's own 2.0 N·m scale", () => {
    const b = gripperBar({ load: 0.5 });
    expect(b.markFrac).toBeCloseTo(0.5, 6);
    expect(b.frac).toBeCloseTo(0.25, 6);
    expect(b.holding).toBe(false);
    expect(gripperBar({ load: 1.0 }).holding).toBe(true); // the threshold is inclusive
    expect(gripperBar({ load: -1.4 }).holding).toBe(true); // …and the load is signed
    expect(gripperBar({ load: -1.4 }).frac).toBeCloseTo(0.7, 6);
  });

  it("trusts the LAB's predicate over the threshold when it sends one", () => {
    // `MarsDriver.held_body` is torque past 1.0 N·m AND a blade contact with
    // something that is not the robot: closing on air reads 0.0 N·m,
    // identical to an open claw, so the torque alone cannot answer it.
    expect(gripperBar({ load: 1.8, holding: false }).holding).toBe(false);
    expect(gripperBar({ load: 1.8, holding: false }).inferred).toBe(false);
    expect(gripperBar({ load: 1.8 }).inferred).toBe(true);
  });

  it("clamps the bar and takes a body's own limits", () => {
    expect(gripperBar({ load: 99 }).frac).toBe(1);
    expect(gripperBar({ load: 0 }).frac).toBe(0);
    const b = gripperBar({ load: 2, limit: 8, hold: 4 });
    expect([b.frac, b.markFrac, b.holding]).toEqual([0.25, 0.5, false]);
  });
});

describe("armBar: an angle inside its own travel", () => {
  it("places a joint in its limits, and falls back to ±π without them", () => {
    expect(armBar(0, [-1, 1])).toBeCloseTo(0.5, 6);
    expect(armBar(-1, [-1, 1])).toBe(0);
    expect(armBar(1, [-1, 1])).toBe(1);
    expect(armBar(0)).toBeCloseTo(0.5, 6);
    expect(armBar(2, [-1, 1])).toBe(1); // clamped, never off the end of the bar
    // A degenerate pair is ignored rather than dividing by zero: a body that
    // sends lo == hi would otherwise paint every joint NaN% wide. It falls
    // back to ±π, so 0.5 rad sits at (0.5 + π) / 2π.
    expect(armBar(0.5, [1, 1])).toBeCloseTo((0.5 + Math.PI) / (2 * Math.PI), 6);
  });
});
