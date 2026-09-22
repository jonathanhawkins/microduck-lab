// Types + client for the lab's WORLD mode (microduck_local/world_server.py):
// the /sim page's backend. Same lab host as lib/lab.ts, second socket.

import type { BrainGraphInfo } from "./braingraph";
import type { BrainView } from "./brainview";
import { LAB_HTTP } from "@/lib/lab";

export const SIM_WS = LAB_HTTP.replace(/^http/, "ws") + "/ws/sim";

/** The three.js layer every /sim sensor overlay draws on: the orbit camera
 *  sees it, the head-camera inset does not — a robot does not see its own
 *  sensor drawings. One definition, because a second copy that said 2 would
 *  put the new LiDAR overlay into the robot's own camera view. */
export const OVERLAY_LAYER = 1;

/** The layer a robot's own camera HOUSING is parked on while the inset
 *  renders: the orbit camera has it enabled, the inset camera does not.
 *
 *  A camera cannot see the shell it is bolted inside, and MARS's proves it:
 *  MEASURED across its whole head-pitch range, its `head` is 1.5-2.8 cm from
 *  the lens while inside the frame, against the inset's 3 cm near plane — so
 *  the shell is clipped while everything is still, and swings into the
 *  picture the moment the drawn pose (lerped toward each frame) lags the
 *  pose the capture came from. Moving the camera would have fixed the
 *  picture by moving a SENSOR, which is where every bearing the detector
 *  reports is measured from.
 *
 *  Only the housing, and only the robot whose camera it is: the arm stays
 *  (`link5` is in the same frame at 15-26 cm, and a real MARS sees its own
 *  claw), and another robot in the room keeps its head. The lab names the
 *  body — `det.selfBody`, the one its detector already refuses to detect. */
export const SELF_LAYER = 2;

/** Scale a scene dump's vertices into metres IN PLACE, and say so.
 *
 *  A dump may carry millimetre ints with a `vertScale` (the G1's is 21 MB
 *  that way instead of 78 MB of floats). Whoever scales it must also reset
 *  `vertScale` to 1: `SimStage.RobotBody` scales by that field itself (the
 *  lab stage's `Duck.tsx` always has), and a scene scaled here that still
 *  said 0.001 was scaled AGAIN there — a G1 person drawn a thousand times
 *  too small, i.e. not at all, with nothing in the console. That is how the
 *  /sim G1 vanished on 2026-09-18 when the person renderer was generalised.
 *  Idempotent: scaling a scene that already says 1 changes nothing. */
export function scaleSceneToMetres<T extends { meshes: { v: number[] }[]; vertScale?: number }>(sc: T): T {
  const scale = sc.vertScale ?? 1;
  if (scale !== 1) {
    for (const m of sc.meshes) {
      const v = m.v;
      for (let i = 0; i < v.length; i++) v[i] *= scale;
    }
    sc.vertScale = 1;
  }
  return sc;
}

/** Visual scene for a G1 person (same shape as GET /scene). null if unfetched. */
export async function fetchG1Scene(): Promise<import("./lab").Scene | null> {
  const r = await fetch(`${LAB_HTTP}/scene/g1`);
  if (!r.ok) return null;
  const sc = await r.json() as import("./lab").Scene & { vertScale?: number };
  return scaleSceneToMetres(sc);
}

export type TofPreset = "ideal" | "datasheet" | "hostile";
export const TOF_PRESETS: TofPreset[] = ["ideal", "datasheet", "hostile"];

export interface ScenarioWall { from: [number, number]; to: [number, number]; height: number; thickness: number }
export interface ScenarioBox { pos: [number, number, number]; size: [number, number, number]; yaw: number; mass: number; rgba: [number, number, number, number] }
export interface ScenarioBall { pos: [number, number]; radius: number; mass: number }
export interface ScenarioDuck {
  id: string;
  spawn: [number, number, number];
  /** Which BODY this entry is — any registry id, "microduck" when absent
   *  (`world/scenario.Duck.robot`). A room can hold a duck and a MARS, and
   *  the stage draws each from `GET /scene?robot=<id>`. */
  robot?: string;
  policy: string | null;
  /** The entry's RANGE-SENSOR preset. On a duck it names the 8x8 ToF's
   *  noise; on a wheeled body it names the LIDAR's — one field, because the
   *  question it answers ("how noisy is this robot's range sense") is the
   *  same one (`world/scenario.Duck`'s own docstring). */
  tof: TofPreset | null;
  detector?: TofPreset | null;
  brain?: string | null;
  /** Odometry drift preset. Carried so the editor's save round-trips it. */
  odom?: string;
  /** Soccer: the team's colorway and the job this duck plays. */
  team?: TeamName | null;
  role?: RoleName | null;
}

/** A team IS a colorway (microduck_local/world/scenario.py TEAM_COLORWAYS):
 *  the four Pollen ships, each a shell colour and the trim-and-beak colour
 *  that goes with it. Two ducks of one colorway cannot be told apart on the
 *  robot either, which is what a team is. Keep in step with the Python table —
 *  a test cannot see across the two repos, so the values are duplicated on
 *  purpose and the comment is the link. */
export const TEAM_COLORWAYS = {
  cream: { shell: "#f7e6cb", trim: "#f28c21", label: "Cream" },
  graphite: { shell: "#6c6a68", trim: "#fac71a", label: "Graphite" },
  lavender: { shell: "#bfa9cf", trim: "#fac71a", label: "Lavender" },
  sky: { shell: "#a9dbe8", trim: "#f28c21", label: "Sky" },
} as const;
export type TeamName = keyof typeof TEAM_COLORWAYS;
export const TEAM_NAMES = Object.keys(TEAM_COLORWAYS) as TeamName[];
/** The pair a new pitch is dealt, home (−x) then away (+x) — the same pair
 *  `make_pitch` uses (world/scenario.py PITCH_TEAMS, which carries the whole
 *  table). Cream v graphite is the furthest apart of the four ships in both
 *  colour distance and LIGHTNESS, and lightness is what survives a duck being
 *  60 px tall, moving, and lit from one side. Their trims differ too (orange
 *  against yellow), so a duck that is a few pixels of leg still reads. */
export const PITCH_TEAMS: readonly [TeamName, TeamName] = ["cream", "graphite"];
/** The lab's boards (world_server.PITCH_COVE / PITCH_CORNER): a quarter-round
 *  of this radius along the base of every wall, and a 45° chamfer this far
 *  across each corner. The editor's "make a pitch" applies both, so a pitch
 *  drawn there is the pitch the lab plays on. */
export const PITCH_COVE = 0.15;
export const PITCH_CORNER = 0.3;
export const ROLE_NAMES = ["defender", "midfielder", "striker", "keeper"] as const;
export type RoleName = (typeof ROLE_NAMES)[number];

/** MJCF material names a colorway repaints (world/compose.py — keep the two
 *  lists in step). EVERY printed part takes one of the two colours: shells are
 *  the head, trunk, legs and hips; trim is the beak, feet, ankles and soles.
 *  Servos, PCBs, bearings and the lens are left alone — they are the same on
 *  every duck, as on the real robot. The lists are long because the CAD export
 *  is not tidy: a teal thigh plate and shoe rim, a pale-blue hip and a pink
 *  mouth are printed parts that carried colours no colorway claimed, so a duck
 *  meant to be one colour rendered as five.
 *
 *  The viewer builds its duck geometry from the single-robot scene, so it
 *  tints by these names client-side; the server paints the same ones in the
 *  composed model for MuJoCo's own renders. */
export const SHELL_MATERIALS = ["left_shell_material", "right_shell_material",
  "top_head_shell_material", "bottom_head_shell_material",
  "leg_material", "upper_leg_left_material", "upper_leg_right_material",
  "hip_l_material", "upper_leg_rigidity_plate_material",
  "yaw_roll_motion_material", "jaw_soft_material", "soft_mouth_top_material"];
export const TRIM_MATERIALS = ["jaw_material", "foot_left_material", "foot_right_material",
  "ankle_left_material", "ankle_right_material",
  "sole_left_material", "sole_right_material"];

/** The shell colour of a team, for a swatch or a label. */
export function teamColor(team: string | null | undefined): string | null {
  return team && team in TEAM_COLORWAYS ? TEAM_COLORWAYS[team as TeamName].shell : null;
}

/** A team chip's background: the SHELL colour alone. A two-tone chip that
 *  showed the trim as well was tried and reverted — the trim is shared between
 *  colorway pairs (cream and sky are both orange, graphite and lavender both
 *  yellow), so at 12 px the trim half swamped the chip and made cream and
 *  lavender the same amber block, destroying the one distinction a chip is
 *  for. `fallback` is what a duck with no team gets. */
export function teamSwatch(team: string | null | undefined, fallback = "#555"): string {
  return teamColor(team) ?? fallback;
}
export interface ScenarioPerson {
  id: string; pos: [number, number]; yaw: number; path: [number, number][];
  speed: number; radius: number; height: number; kind?: "capsule" | "g1";
}
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
  /** > 0: a pitch — goals this wide centred on both short walls (the World counts them). */
  goal_width?: number;
  /** > 0: a quarter-round cove of this radius (m) along the base of every wall,
   *  cut at the goal mouths — the ball rolls up it and back into play
   *  (world/scenario.py `Scenario.cove`). The stage draws it (SimStage.tsx). */
  cove?: number;
  /** Which goal MOUTH each team attacks, for a roster not all facing it. */
  attacks?: Partial<Record<TeamName, "left" | "right">>;
  collision: "walk" | "all";
}
/** One row of `GET /scenarios`.
 *
 *  `ducks` is the TOTAL number of robot entries and is named that for its
 *  history; `robots` breaks it down by body, commonest first, and is what a
 *  menu should show. A lab too old to send `robots` leaves it undefined, and
 *  the picker falls back to counting everything as ducks — which is what it
 *  used to do, and why `mars-follow` announced "1 ducks". */
export interface ScenarioListing {
  name: string;
  builtin: boolean;
  ducks: number;
  robots?: { id: string; n: number; noun: string }[];
  objects: number;
  modified: number | null;
}

/** THE RUG (SimStage draws it on every walled room that is not a pitch).
 *  A /sim room is drawn at HUMAN scale — 0.8 m plank tiles, eight boards to a
 *  tile, a 1.32 m G1 walking through it — so the rug is sized the way a real
 *  area rug is, not the way a 25 cm duck would see it: a little over half the
 *  room each way, capped at a 9 x 12 ft rug (2.74 x 3.66 m), the largest size
 *  sold off the roll. The caps are what matter — the old ones (1.6 x 1.2) are
 *  a 5 x 4 ft accent rug, which reads as a doormat in the middle of the 6 x 5 m
 *  follow-me / flock rooms. Returned [x, y]: the long side lies along the
 *  room's long axis, so a portrait room gets a portrait rug. */
export const RUG_LONG_MAX = 3.66;
export const RUG_SHORT_MAX = 2.74;
export function rugSize(roomW: number, roomH: number): [number, number] {
  const long = Math.min(Math.max(roomW, roomH) * 0.55, RUG_LONG_MAX);
  const short = Math.min(Math.min(roomW, roomH) * 0.6, RUG_SHORT_MAX);
  return roomW >= roomH ? [long, short] : [short, long];
}

/** Which goal MOUTH each team attacks, decided the way the World decides it
 *  (world/arena.py `World.goal_for`): the scenario's declaration when it
 *  carries one, else the mouth that team's ducks are spawned facing. "right"
 *  is the mouth at +x. Empty off a pitch. */
export function attackedMouths(scenario: Scenario | null): Record<string, "left" | "right"> {
  const out: Record<string, "left" | "right"> = {};
  if (!scenario || !(scenario.goal_width ?? 0)) return out;
  for (const [team, mouth] of Object.entries(scenario.attacks ?? {})) {
    if (mouth === "left" || mouth === "right") out[team] = mouth;
  }
  for (const d of scenario.ducks) {
    if (!d.team || out[d.team]) continue;      // a declaration outranks the heading
    out[d.team] = Math.cos(d.spawn[2]) >= 0 ? "right" : "left";
  }
  return out;
}

/** The team that DEFENDS each goal mouth — whose colours the stage paints that
 *  goal frame, so the ends read as "cream's end" and "graphite's end" the way
 *  the ducks do. A mouth belongs to a team only when exactly ONE other team
 *  attacks it; a one-team pitch (eval-striker's) leaves the far end unpainted,
 *  and so does a roster the server would refuse, rather than guessing. */
export function goalDefenders(scenario: Scenario | null): { left: string | null; right: string | null } {
  const attacks = attackedMouths(scenario);
  const defender = (mouth: "left" | "right") => {
    const attackers = Object.keys(attacks).filter((t) => attacks[t] !== mouth);
    return attackers.length === 1 ? attackers[0] : null;
  };
  return { left: defender("left"), right: defender("right") };
}

/** Wall-clock speed presets for the world, mirroring SPEED_CHOICES in
 *  world_server.py. Slow motion is the same knob from the other end: at
 *  0.25x the lab steps the sim once every fourth wall tick, which is how
 *  you watch a fall or a kick land. */
export const SIM_SPEEDS = [0.25, 0.5, 1, 2, 4, 8] as const;

export const SIM_SPEED_DEFAULT = 1;

/** The world's current wall-clock speed, republished on every frame.
 *
 *  A module store rather than a prop because its consumers are the per-frame
 *  pose smoothers inside `useFrame` (Duck.tsx, SimStage.tsx), which would
 *  otherwise need it threaded through every duck and every object. Read it
 *  inside the frame callback, never during render.
 *
 *  Those renderers are shared with the LAB pages, which stream at 1x and have
 *  no speed knob, so this has to be handed back: `SimClient.close()` restores
 *  it, and a socket that drops without closing is covered by the reconnect
 *  writing the live value again. */
export const simRate = { speed: SIM_SPEED_DEFAULT };

/** Base rate of the renderers' pose smoothing, in wall Hz (tau = 62 ms).
 *
 *  It is MULTIPLIED by `simRate.speed`, because the filter's lag is fixed in
 *  WALL time while the sim time a frame carries is not. Left unscaled, 8x
 *  put half a second of world behind the picture and attenuated the ~1.5 Hz
 *  sim gait (arriving at ~12 Hz of wall) against a 2.5 Hz corner, so the
 *  ducks glided with barely-moving legs; and 0.25x converged inside two
 *  frames and then held, rendering slow motion as 12.5 Hz stepping — the
 *  one case the speed knob exists to make watchable. */
export const POSE_SMOOTH_HZ = 16;

/** Label a speed the way a video player would: 1x, 0.25x, 2x. */
export function speedLabel(x: number): string {
  return `${Number(x.toFixed(2))}\u00d7`;
}

/** The [ and ] keys step through SIM_SPEEDS and stop at the ends — they do
 *  not wrap, because a key-repeat off 8x landing back on 0.25x is a trap.
 *  An off-preset speed (someone POSTed 3) steps to the neighbour it is
 *  heading towards rather than snapping backwards. */
export function stepSpeed(current: number, dir: -1 | 1): number {
  const xs = SIM_SPEEDS;
  if (dir > 0) return xs.find((x) => x > current + 1e-9) ?? xs[xs.length - 1];
  return [...xs].reverse().find((x) => x < current - 1e-9) ?? xs[0];
}

/** Is the lab keeping the promise? True when the measured RTF has fallen
 *  meaningfully short of what was asked — a 3v3 at 4x runs about 3. The
 *  slack absorbs the RTF's own one-second window jitter.
 *
 *  The "no measurement yet" guard is `rtf > 0`, not an absolute floor: the
 *  lab reports exactly 0 until a window closes, and it zeroes the window on
 *  every speed change for that reason. An absolute floor would have to be
 *  small enough not to swallow a genuine stall at 0.25x — at which a world
 *  managing 0.04x is a 6x shortfall, and any floor above it reports a frozen
 *  world as healthy. */
export function speedShortfall(rtf: number, asked: number): boolean {
  return rtf > 0 && rtf < asked * 0.85;
}

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
/** The ToF's own range limits (`sensors/tof.TofSpec`): a return outside them
 *  is "no target", 0 on the wire. They are the SENSOR's, not the duck's, so
 *  the lidar→ToF adapter a wheeled body's brains read applies the same pair
 *  (lib/lidar.ts ports it) — a 4.5 m wall is inside a 6 m scan and outside
 *  the 8x8 those brains were tuned on. */
export const TOF_MIN_RANGE_M = 0.02;
export const TOF_MAX_RANGE_M = 4.0;
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
/** A wxyz quaternion applied to a vector. Exported because the /sim overlays
 *  need it too (the LiDAR overlay turns a mount-frame bearing into a world
 *  ray off the `base_laser` pose the frame already carries) and two copies of
 *  a rotation are two chances to get one of them wrong. */
export function quatRotate(q: number[], v: [number, number, number]): [number, number, number] {
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
/** A detector frame: captured at `t`, `age` old now, the frustum it saw
 *  through, the camera's world pose at capture (x y z, w x y z quaternion of
 *  the site frame, x forward) and what it found. */
export interface DetPayload {
  t: number;
  age: number;
  fov?: [number, number];
  cam?: number[];
  /** Which of THIS robot's own bodies wraps the lens, as an index into its
   *  `bodies` list — the body the /sim inset must not draw when it renders
   *  from this camera (SELF_LAYER). -1, or absent on an older lab, means
   *  draw everything. */
  selfBody?: number;
  items: DetectionItem[];
}
export interface BrainInputs {
  tof?: { age: number | null; stale: boolean; max: number };
  /** The planar scan's own freshness, for a body that reads one.
   *
   *  Sent since c858568: `age_inputs` keys on `Senses.lidar` and names the
   *  row after the DEVICE the body has, so a wheeled body reports `lidar`
   *  and no `tof` at all. On a lab that predates it the `tof` entry's age IS
   *  the scan's age
   *  (`World.senses_tof` hands the brain an adapted 8x8 and deliberately
   *  keeps the SCAN's timestamp — a 6 Hz scanner's frame is up to 167 ms old
   *  and a brain gating on freshness must see that). The inspector therefore
   *  LABELS that row by the body's channel, and renders this one whenever a
   *  lab starts sending it. */
  lidar?: { age: number | null; stale: boolean; max: number };
  det?: { age: number | null; stale: boolean; max: number; n: number };
  target?: { bearing: number; range: number | null; since: number } | null;
  /** The brain's tracker, with each track's odometry-frame position and velocity once it has both. */
  tracks?: { id: number; cls: string; name: string; bearing: number; range: number; hits: number; age: number; xy?: [number, number]; vel?: [number, number] }[];
  /** A chase brain's plan, in its odometry frame: where it predicts the
   *  ball will stop (it looks and hunts that way), the ball memory its
   *  search walks to, and its line-up / push spot.
   *
   *  Two signals about its own body ride along. `bumped` is how long ago
   *  (s) its feet last touched another duck or a person — the contact list
   *  here, the IMU and the servo loads on the robot; null until it ever
   *  happens. Under half a second the duck is being bumped, and it stands
   *  instead of turning in place (3v3 falls: 5.00 → 1.75 a run). `tofBall`
   *  is a ball-sized blob the 8×8 ToF sees on the floor at its feet,
   *  [bearing, range] in the duck's HEADING frame (+left, metres) — the
   *  blind last 30 cm the head camera loses a floor ball in. It is measured
   *  OFF as a ball source for the brain (a blob at the feet is as often the
   *  other duck's foot) and computed either way. */
  chase?: {
    kicks: number; pushes: number;
    /** What the team board says this duck is doing RIGHT NOW: "attack" (it
     *  is the one on the ball) or "support". */
    role: string;
    /** …and the static job off the scenario, absent for a duck that has
     *  none: "defender" | "midfielder" | "striker". The two are different
     *  questions — a defender is the attacker whenever the ball is in its
     *  own third and it is the quickest there. */
    job?: string;
    memory: [number, number] | null; predicted: [number, number] | null; spot: [number, number, string] | null;
    bumped?: number | null; tofBall?: [number, number] | null;
  };
  /** The TEAM's blackboard, as this duck sees it. `ball` is the shared
   *  belief — the freshest teammate sighting, or the inverse-variance fusion
   *  of all of them when the board is fusing — which is what a supporter
   *  steers by. Absent when nobody on the team has seen the ball. */
  team?: {
    name: string;
    attacker: string | null;
    ball?: [number, number];
    ballVel?: [number, number];
    jobs?: Record<string, string>;
    claims?: Record<string, { dist: number | null; cost: number | null; age: number; pos?: [number, number, number] }>;
  };
}
export interface SimDuck {
  id: string;
  name: string;
  /** Which BODY to draw this entry as — any registry id, "microduck" when
   *  absent. `bodies` below is then ITS scene's list, in
   *  `GET /scene?robot=<id>` order (18 poses for a MARS, world first). */
  robot?: string;
  /** Soccer: the team's colorway (what the duck is painted) and its job. */
  team?: TeamName | null;
  role?: RoleName | null;
  policy: string | null;
  falls: number;
  step: number;
  rew: number;
  speed: number;
  cmdSpeed: number;
  steerable: boolean;
  /** The RANGE-SENSOR noise preset, by device. A duck carries the 8x8 ToF and
   *  `lidar` is null; a wheeled body carries the 360-degree scanner and `tof`
   *  is null (`world_server.duck_info`). Which of the two is set is what the
   *  inspector reads to know which instrument this body HAS — it is the
   *  body's declaration, so it answers before the first frame arrives. */
  tof: TofPreset | "custom" | null;
  lidar?: TofPreset | "custom" | null;
  detector: TofPreset | "custom" | null;
  holding: string | null;
  /** Odometry drift preset the brain's pose carries (roadmap 1.7), and the drifted estimate itself. */
  odom?: string;
  odomEst?: [number, number, number];
  skill: string | null;
  beak: "open" | "closed";
  /** The 15th servo as an opening fraction, 0 shut to 1 wide. `beak` above is
   *  the GRASP state; this is how far the bill actually is. Optional: a server
   *  older than the hinged jaw does not send it. */
  mouth?: number;
  /** The duck's OWN brain, from the registry — unlike `brain.kind` below,
   *  which reports who is steering this tick and so reads "manual" for every
   *  duck while a drive command is held. Anything that must survive taking
   *  the wheel (the state graph's trace identity) keys on this. */
  brainKind?: string | null;
  /** Who is steering this duck this tick: a brain from the lab's registry
   *  (auto mode), the demo script (blind ducks), or you (manual). */
  brain: {
    kind: string; state: string; cmd: [number, number, number]; head?: number[]; note?: string; beak?: string | null; skill?: string | null;
    /** Which declared state graph this duck's brain is drawn on
     *  (brain/graph.py). A key into WorldInfo.graphs; null for a brain
     *  nothing is declared for. */
    graph?: string | null;
    inputs: BrainInputs & { tidy?: { picked: number; delivered: number; givenUp: string[] } };
    /** A learned brain's last decision — what the network saw and said (runtime.brain_view). */
    view?: BrainView;
  };
  headApplied: boolean;
  bodies: number[][];
  /** What this body SENSES, one key per channel it has
   *  (`world_server.tof_payload`, `robots/body.Body.make_sensors`): the duck's
   *  8x8 `tof`, a wheeled body's 360-degree `lidar`, the head camera's `det`.
   *  The inspector renders one block per key present, which is why a body with
   *  no ToF never shows a ToF placeholder. `gripper` and `arm` are the arm
   *  channels — see their own types for what the lab still owes. */
  sensors: {
    tof?: TofPayload;
    det?: DetPayload;
    lidar?: LidarPayload;
    gripper?: GripperPayload;
    arm?: ArmPayload;
  } | null;
}

/** The claw, as a reading: the constraint torque at joint6 and whether that
 *  counts as holding something (`robots/mars_drive.MarsDriver.held_body` —
 *  `|load| >= mars.HOLD_LOAD_NM` AND a blade contact with a body that is not
 *  part of the robot; closing on AIR reads 0.0 N·m, identical to an open
 *  claw, which is why the contact conjunct exists).
 *
 *  Sent since c858568 (`world_server.gripper_payload`), beside the
 *  pickable's id on `SimDuck.holding`. The inspector draws this block when a
 *  lab sends it and nothing when it does not, so an older lab still works. */
export interface GripperPayload {
  /** N·m at joint6. Signed: closing is one direction, opening the other. */
  load: number;
  /** The driver's own predicate, not a threshold re-applied here. */
  holding?: boolean;
  /** Full scale for the bar, if the body has an opinion
   *  (`mars.GRIPPER_EFFORT_LIMIT`, 2.0 N·m — the servo's own clamp). */
  limit?: number;
  /** The hold threshold to mark (`mars.HOLD_LOAD_NM`, 1.0 N·m). */
  hold?: number;
}

/** The arm's ACHIEVED joint positions by name — what Innate's
 *  `/mars/arm/state` publishes, and NOT the commanded targets: the servo
 *  carries structural compliance and backlash, so a commanded pose is reached
 *  a few hundredths of a radian low (`brain/runtime.Senses.arm` measures 28.5
 *  mm of claw height at the pick pose).
 *
 *  Sent since c858568 (`world_server.arm_payload`). With `cmd` the inspector
 *  marks the command beside the achieved angle, and the GAP between them is
 *  the sag the brain pre-compensates — which is why both are on the wire. */
export interface ArmPayload {
  q: Record<string, number>;
  cmd?: Record<string, number>;
  /** Joint limits, for a bar that means something: [lo, hi] rad by name. */
  limits?: Record<string, [number, number]>;
}

/** One 360-degree planar scan (`sensors/lidar.LidarFrame.as_payload`).
 *
 *  Millimetre integers in the ToF's own convention — **0 means no reading**,
 *  an invalid ray and not a zero-range one — with the bearings rebuilt from
 *  `a0 + i * da` rather than shipped as 360 floats the consumer already
 *  knows. `mount` is where the scanner sits in the base's HEADING frame, so
 *  a drawing starts at the aperture and not at the chassis origin: 76 mm
 *  apart on MARS, which is a third of the robot's length. */
export interface LidarPayload {
  t: number;
  a0: number;
  da: number;
  mm: number[];
  age?: number;
  maxRange?: number;
  /** The device's own floor (`LidarSensor.min_range`): a return nearer than
   *  this is CLIPPED and marked invalid, so it arrives as a 0 and a plot must
   *  draw the disc it cannot see inside of. Read it with `lidarMinRange`,
   *  which falls back to the 0.15 m default for a lab that predates it. */
  minRange?: number;
  /** How far out a return is the robot looking at its own arm
   *  (`WorldRobot.footprint_m` -> `MarsBody.footprint_m`), dropped by the
   *  adapter every brain here reads.
   *
   *  **0 is a value, not a gap**: it means "declared none, every return
   *  kept", which is `tof_from_lidar`'s own default. Read it with `??` and
   *  never `||` — `lidarFootprintM` does. */
  footprint?: number;
  mount?: [number, number, number] | null;
}
export interface SimObject {
  id: string; kind: "ball" | "box" | "person" | "toy"; pose: number[];
  possessed?: boolean; toy?: string; held?: string | null; inBasket?: boolean;
  /** "g1" when this person is a Unitree G1, not the mocap capsule. */
  robot?: string;
  /** G1 body poses in GET /scene/g1 order, when robot is set. */
  bodies?: number[][];
}
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
  /** Soccer on a pitch scenario. `left`/`right` are goal MOUTHS, not team
   *  scores — the left team attacks +x and its goals land in `right`. The
   *  per-team rates are what the benchmark actually judges by: goals are
   *  ~2.5 a run and cannot resolve a change (146 seeds for a 25% shift),
   *  while possession takes 9 and ballAdvance 43. */
  soccer: ({ left: number; right: number; ball: [number, number]; lastGoal: "left" | "right" | null; kickoff: number; kicked?: number; bumped?: number;
    /** The game state (roadmap Track 4 §6 B.3): "set" during the hold, "kickoff" while the
     *  conceding side's ball is still on the spot, else "playing"; and who kicks off. */
    state?: "set" | "kickoff" | "playing"; kickoffTeam?: string | null;
    /** Goals neither the kicker nor the last touch could be pinned to. */
    goalsUnattributed?: number }
    & Partial<Record<"ballAdvance" | "ballProgress" | "possession" | "possessionWide"
      | "ballOwnHalf" | "spread" | "crowd" | "depth"
      | "goalsFor" | "goalsAgainst" | "ownGoals" | "kickCount" | "kicksBack" | "kickCarry",
      Record<string, number | null>>>) | null;
  /** Brain round-trip latency applied to every intent (roadmap 12.10), ms; 0 = onboard. */
  tetherMs?: number;
  /** Wall-clock speed the world was ASKED to run at. What it managed is
   *  `rtf` — the two part company once the scene costs more than its 20 ms
   *  tick (a 3v3 pitch stops climbing near 3x), and the HUD says so. */
  simSpeed?: number;
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
  /** The learned brains again, with what people read: the inspector's menu
   *  files them by group and shows the title; `learned:<name>` stays the value. */
  learned?: LearnedInfo[];
  /** The state graphs, keyed by the per-duck `brain.graph`. Static tables —
   *  they ride this message once, never the frame. */
  graphs?: Record<string, BrainGraphInfo>;
}

export interface LearnedInfo {
  name: string;
  title: string | null;
  group: string | null;
  description: string | null;
}

/** Group keys in menu order, with their headings — mirrors describe_brain.GROUPS. */
export const LEARNED_GROUPS: [key: string, label: string][] = [
  ["shipped-followers", "Followers (shipped)"],
  ["trainer-ab", "Trainer defect A/B (seed 7)"],
  ["early-stop", "Early stop"],
  ["paired-sweeps", "Paired-seed sweeps (seeds 11–14)"],
  ["capacity", "Network capacity (seeds 31–36)"],
  ["null-pair", "Null pair (seeds 81–84)"],
  ["other", "Other"],
];

/** The group a new user should meet: brains that ship, not experiments about the trainer. */
export const SHIPPED_GROUP = "shipped-followers";

/**
 * What the brain menu offers. By default only the shipped brains (plus
 * whatever the duck is on right now, so the select never shows blank);
 * with `showAll`, every run under its heading. `hidden` is the count the
 * toggle should promise — 44 experiment runs is a fact, not a menu.
 */
export function menuBrains(
  learned: LearnedInfo[],
  current: string | null,
  showAll: boolean,
): { groups: [label: string, brains: LearnedInfo[]][]; hidden: number } {
  if (showAll) return { groups: groupLearned(learned), hidden: 0 };
  const keep = learned.filter((b) => b.group === SHIPPED_GROUP || `learned:${b.name}` === current);
  return { groups: groupLearned(keep), hidden: learned.length - keep.length };
}

/** Learned brains filed under their group heading, in LEARNED_GROUPS order;
 *  an unknown or missing group files under "Other". */
export function groupLearned(learned: LearnedInfo[]): [label: string, brains: LearnedInfo[]][] {
  const known = new Map(LEARNED_GROUPS);
  const by = new Map<string, LearnedInfo[]>();
  for (const b of learned) {
    const k = b.group && known.has(b.group) ? b.group : "other";
    by.set(k, [...(by.get(k) ?? []), b]);
  }
  return LEARNED_GROUPS.flatMap(([k, label]) => (by.has(k) ? [[label, by.get(k)!] as [string, LearnedInfo[]]] : []));
}

// The head camera: the MJCF `head_camera` site, x-forward, on jaw_soft.
export const CAM_SITE_POS: [number, number, number] = [0.0155, -0.0000913778, -0.0733];
export const CAM_SITE_QUAT_WXYZ: [number, number, number, number] = [0.707107, 0, 0.707107, 0];

// The detector's frustum (sensors/detector.py DetectorSpec): horizontal, vertical degrees.
// The stream carries the live values in det.fov; this is the fallback.
export const CAM_FOV_DEG: [number, number] = [62, 48];

/** The head camera's world-frame pose from the jaw pose: where it is, the
 *  axis it looks down (site x) and its up (site z). What the /sim page's
 *  camera inset renders from. */
export function headCameraPose(jaw: number[]): { origin: [number, number, number]; forward: [number, number, number]; up: [number, number, number] } {
  const jq = [jaw[3], jaw[4], jaw[5], jaw[6]];
  const off = quatRotate(jq, CAM_SITE_POS);
  const sq = quatMul(jq, CAM_SITE_QUAT_WXYZ);
  return {
    origin: [jaw[0] + off[0], jaw[1] + off[1], jaw[2] + off[2]],
    forward: quatRotate(sq, [1, 0, 0]),
    up: quatRotate(sq, [0, 0, 1]),
  };
}

/** The camera pose a detector frame was captured from (`det.cam`, the site
 *  frame: x forward, z up), in the same shape as headCameraPose. */
export function capturePose(cam: number[]): { origin: [number, number, number]; forward: [number, number, number]; up: [number, number, number] } {
  const q = [cam[3], cam[4], cam[5], cam[6]];
  return { origin: [cam[0], cam[1], cam[2]], forward: quatRotate(q, [1, 0, 0]), up: quatRotate(q, [0, 0, 1]) };
}

/** Where a detection lands in the camera's image, as fractions of the frame
 *  (u right, v down, 0..1), with the box's size: the pinhole projection of
 *  the bearing (+left), elevation (+up) and apparent width the detector
 *  reports, at the given field of view. Boxes are as wide as the target
 *  looks; a person's is as tall as a person at that range. */
export function detectionBox(d: DetectionItem, fovDeg: [number, number] = CAM_FOV_DEG): { u: number; v: number; w: number; h: number } {
  const halfH = Math.tan((fovDeg[0] * Math.PI) / 360);
  const halfV = Math.tan((fovDeg[1] * Math.PI) / 360);
  const u = 0.5 - Math.tan(d.bearing) / (2 * halfH);
  const v = 0.5 - Math.tan(d.elevation) / (2 * halfV);
  const w = Math.tan(d.width / 2) / halfH;
  const widthM = 2 * d.range * Math.tan(d.width / 2);
  const heightM = d.cls === "person" ? 1.5 : d.cls === "duck" ? 0.2 : widthM;
  const h = Math.min(1.5, (heightM / Math.max(d.range, 0.05)) / (2 * halfV));
  return { u, v, w, h };
}

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
/** Remove a user scenario. Built-ins are read-only (the server answers 409). */
export async function deleteScenario(name: string): Promise<void> {
  const r = await fetch(`${LAB_HTTP}/scenarios/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `delete ${r.status}`);
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
      simRate.speed = frame.simSpeed ?? 1;
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
  sendSpeed(x: number) { this.send({ speed: x }); }
  close() {
    this.closed = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    // Hand the speed back. `simRate` is a module store and `Duck`/`SimStage`
    // are shared with the lab pages, which have no speed knob — leaving 8x
    // behind meant a client-side nav from /sim to / rendered the lab with
    // its pose smoothing effectively off until a full reload.
    simRate.speed = SIM_SPEED_DEFAULT;
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
