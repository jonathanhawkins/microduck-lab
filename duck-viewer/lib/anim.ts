// Keyframe animation editor: types, clip math, lab client, and the shared
// store that bridges the DOM panel (AnimPanel) to the in-Canvas preview duck
// (PoseDuck) — same module-level-mutable-object philosophy as lib/assign.ts,
// because a pose drag must update at pointer speed, not at React speed.
//
// The clip JSON is a CONTRACT with the imitation-RL side (it resamples a saved
// clip at 50 Hz and rewards the policy for tracking it) — see viz_server.py's
// docstring. Anything here that touches `Clip` must keep that shape.

import { LAB_HTTP, type RobotId } from "./lab";
// Type-only, so the rig ↔ anim import pair is erased at build time: the rig
// controls a lab serves have exactly the shape lib/rig.ts's constant does.
import type { RigControl } from "./rig";

export type { RobotId };

// ---------------------------------------------------------------- joint meta

export interface JointMeta {
  index: number;
  name: string;
  group: string; // the duck: "left leg" | "head + neck" | "right leg"; the G1 adds waist + arms
  min: number; // MJCF jnt_range — the servo's real travel
  max: number;
  default: number; // DEFAULT_POSE
  body: number; // MuJoCo body this joint drives (index into /scene bodies)
  bodyName: string;
  axis: [number, number, number]; // hinge axis, BODY frame
  pos: [number, number, number]; // hinge anchor, BODY frame
}

/** A point the IK can be asked to put somewhere: a sole, a hand, the head. */
export interface EffectorMeta {
  id: string;
  label: string;
  kind: "foot" | "hand" | "head";
  body: number; // /scene body index the point rides on
  bodyName: string;
  point: [number, number, number]; // BODY frame
  /** Joint indices the solver may move for this effector — its own limb. */
  chain: number[];
}

/** One row of GET /robots: every body the lab knows. `ready` is false for a
 *  body whose assets are not fetched on this lab (`uv run fetch-robot <id>`).
 *
 *  NOT every row is posable — see `animate`. The endpoint used to list only
 *  bodies the 🎬 editor could open because every body was a walker; with a
 *  wheeled one registered, `pose_scratch("mars")` died on a missing
 *  `base_body`, so the capability is a flag on the row now. */
export interface RobotInfo {
  id: RobotId;
  title: string;
  numJoints: number;
  ready: boolean;
  /** What a sentence calls it: "teach the {noun} a trick". Older labs omit it. */
  noun?: string;
  /** "legged" | "wheeled" | "generic" — what SHAPE of robot this is, so the
   *  copy can say "a trick" or "a task" without a table of ids. */
  kind?: string;
  /** Can the 🎬 pose editor open it? A CAPABILITY (effectors, soles, a base
   *  link), not a kind. Absent on a lab that predates the flag, where every
   *  listed body was posable by construction. */
  animate?: boolean;
  /** The 🎓 panel's suggestion chips, from this robot's own recipes. */
  teach?: TeachSuggestion[];
}

export interface TeachSuggestion {
  text: string; // sent as /teach text — the lab guarantees it matches `behavior`
  behavior: string;
  emoji: string;
  title: string;
}

export interface JointsMeta {
  joints: JointMeta[];
  bodies: string[];
  trunkBody: number;
  rootPitchRange: [number, number];
  rootPitchSign: string;
  /** Which body this metadata describes. A lab older than the robot switch
   *  serves none of these; `fetchJoints` fills them in as the duck's. */
  robot: RobotId;
  title: string;
  numJoints: number;
  /** Ordered section labels for the joint list. */
  groups: string[];
  /** Standing height of the trunk body, metres. */
  standHeight: number;
  /** This body's size against the duck (a measured width ratio, 1.0 for the
   *  duck, 2.89 for the G1): every metre-sized gizmo constant scales by it. */
  sizeScale: number;
  effectors: EffectorMeta[];
  /** The body's own rig controls. Absent from an older lab — lib/rig.ts's
   *  `rigControlsFor` falls back to the duck's constant then. */
  rig?: RigControl[];
}

// ------------------------------------------------------------------ the clip

export interface Key {
  t: number; // seconds from clip start, ascending, first key at 0
  joints: number[]; // the robot's joint count of ABSOLUTE radians, joint_names order
  /** Intended trunk pitch, radians. NEGATIVE = lean back (the trunk's
   *  projected gravity acquires -x) — the server documents and tests this. */
  rootPitch: number;
}

export interface Clip {
  version: 1;
  name: string;
  duration: number; // seconds
  loop: boolean;
  keys: Key[];
  /** The body the clip poses. Absent = the duck (every clip saved before
   *  there was a second body); the server validates the joint count against
   *  it and /teach routes the run to that body. */
  robot?: RobotId;
}

/** The body a clip poses — the duck when it predates the field. */
export function clipRobot(clip: Clip): RobotId {
  return clip.robot ?? "microduck";
}

/** A clip as it comes back from the server listing (mtime added). */
export type StoredClip = Clip & { modified?: number };

export interface Pose {
  joints: number[];
  rootPitch: number;
}

/** The duck's joint count — the fallback before any metadata has arrived.
 *  Everywhere a pose is built for a KNOWN robot uses `meta.numJoints`. */
export const NUM_JOINTS = 14;
/** `selected` sentinel for the trunk: it carries rootPitch, not a servo. */
export const ROOT_SEL = -1;

export function zeroPose(numJoints = NUM_JOINTS): Pose {
  return { joints: new Array(numJoints).fill(0), rootPitch: 0 };
}

export function defaultPose(meta: JointsMeta | null): Pose {
  return meta ? { joints: meta.joints.map((j) => j.default), rootPitch: 0 } : zeroPose();
}

export function newClip(meta: JointsMeta | null, name = "untitled"): Clip {
  const p = defaultPose(meta);
  return {
    version: 1,
    name,
    duration: 1.2,
    loop: false,
    keys: [{ t: 0, joints: p.joints, rootPitch: p.rootPitch }],
    robot: meta?.robot ?? "microduck",
  };
}

/** True while the editor holds nothing a person made: one key, and both it
 *  and the pose on screen still read as the body's default (or as the all-zero
 *  placeholder a clip restored before the limits were known carries). This is
 *  the ONLY state in which the editor may follow a robot switch made in
 *  another panel — switching bodies starts a fresh clip, and an authored pose
 *  is expensive to redo. Without the body's metadata nothing can be judged,
 *  so the answer is no. */
export function isUntouched(clip: Clip, pose: Pose, meta: JointsMeta | null): boolean {
  if (!meta || clip.keys.length !== 1) return false;
  const d = defaultPose(meta);
  const same = (p: Pose) =>
    (p.rootPitch ?? 0) === d.rootPitch &&
    p.joints.length === d.joints.length &&
    p.joints.every((v, i) => v === d.joints[i]);
  const zero = (p: Pose) => (p.rootPitch ?? 0) === 0 && p.joints.every((v) => v === 0);
  const key = clip.keys[0];
  return (same(key) || zero(key)) && (same(pose) || zero(pose));
}

export function clampJoint(meta: JointsMeta | null, i: number, v: number): number {
  const j = meta?.joints[i];
  if (!j) return v;
  return Math.min(j.max, Math.max(j.min, v));
}

/** Linear interpolation in joint space — exactly what the RL resampler does,
 *  so what the timeline shows is what the reward will track. Before the first
 *  key / after the last, the pose is held (no extrapolation). */
export function sampleClip(clip: Clip, t: number, numJoints = NUM_JOINTS): Pose {
  const keys = clip.keys;
  // A keyless clip cannot say how many joints it has — the caller can.
  if (!keys.length) return zeroPose(numJoints);
  // Looping wraps into [0, duration) and blends the last key back to the
  // first across the tail, so a cycle reads continuously while scrubbing.
  let time = t;
  if (clip.loop && clip.duration > 0) {
    time = ((t % clip.duration) + clip.duration) % clip.duration;
    const last = keys[keys.length - 1];
    if (time > last.t) {
      const span = clip.duration - last.t;
      const u = span > 1e-6 ? (time - last.t) / span : 0;
      return blend(last, keys[0], u);
    }
  }
  if (time <= keys[0].t) return { joints: [...keys[0].joints], rootPitch: keys[0].rootPitch };
  const last = keys[keys.length - 1];
  if (time >= last.t) return { joints: [...last.joints], rootPitch: last.rootPitch };
  let i = 0;
  while (i < keys.length - 1 && keys[i + 1].t <= time) i++;
  const a = keys[i];
  const b = keys[i + 1];
  const span = b.t - a.t;
  return blend(a, b, span > 1e-9 ? (time - a.t) / span : 0);
}

function blend(a: Key, b: Key, u: number): Pose {
  return {
    joints: a.joints.map((v, k) => v + (b.joints[k] - v) * u),
    rootPitch: a.rootPitch + (b.rootPitch - a.rootPitch) * u,
  };
}

/** Index of the key at `t` (within `eps`), or -1. Editing a joint while the
 *  playhead sits on a key auto-updates that key (animator muscle memory). */
export function keyAt(clip: Clip, t: number, eps = 0.008): number {
  return clip.keys.findIndex((k) => Math.abs(k.t - t) <= eps);
}

/** Insert/replace a key at `t`, keeping `keys` sorted and t=0 anchored. A
 *  key found within `keyAt`'s tolerance keeps ITS time: the playhead parked
 *  4 ms past the anchor is still editing the anchor, not moving it. */
export function withKey(clip: Clip, t: number, pose: Pose): Clip {
  const at = keyAt(clip, t);
  const key: Key = {
    t: at >= 0 ? clip.keys[at].t : round3(t),
    joints: [...pose.joints],
    rootPitch: pose.rootPitch,
  };
  const keys = at >= 0 ? clip.keys.map((k, i) => (i === at ? key : k)) : [...clip.keys, key];
  keys.sort((a, b) => a.t - b.t);
  return { ...clip, keys };
}

export function round3(v: number): number {
  return Math.round(v * 1000) / 1000;
}

/** The contract's invariants, checked client-side so the save button can say
 *  what is wrong before the server 422s. */
export function clipProblem(clip: Clip): string | null {
  if (!clip.keys.length) return "a clip needs at least one key";
  if (clip.keys[0].t !== 0) return "the first key must sit at t = 0";
  for (let i = 1; i < clip.keys.length; i++)
    if (clip.keys[i].t <= clip.keys[i - 1].t) return "key times must ascend";
  if (clip.duration <= 0) return "duration must be > 0";
  if (clip.duration < clip.keys[clip.keys.length - 1].t)
    return "duration would cut off the last key";
  if (!/^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$/.test(clip.name))
    return "name: letters, digits, space, . _ - (starting alphanumeric)";
  return null;
}

// ------------------------------------------------------------- lab requests

async function jsonOrThrow(res: Response) {
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      detail = (await res.json())?.detail ?? detail;
    } catch {
      // non-JSON error body — the status is all we have
    }
    throw new Error(String(detail));
  }
  return res.json();
}

/** `?robot=<id>` — omitted for the duck, so a lab older than the robot
 *  switch (which knows no query) still answers. Same rule as fetchScene. */
function robotQuery(robot: RobotId): string {
  return robot && robot !== "microduck" ? `?robot=${encodeURIComponent(robot)}` : "";
}

/** Every body the lab can pose. A lab older than the switch has no
 *  /robots: that reads as the duck alone, which is what it can do. */
export async function fetchRobots(): Promise<RobotInfo[]> {
  const data = await jsonOrThrow(await fetch(`${LAB_HTTP}/robots`));
  return (data.robots ?? []) as RobotInfo[];
}

/** Fill in what an older lab's /joints does not say, as the duck's values,
 *  so every consumer can read the fields without guarding each one. */
export function normalizeMeta(raw: Partial<JointsMeta> & Pick<JointsMeta, "joints" | "bodies" | "trunkBody" | "rootPitchRange" | "rootPitchSign">, robot: RobotId): JointsMeta {
  const groups =
    raw.groups && raw.groups.length
      ? raw.groups
      : raw.joints.reduce<string[]>((acc, j) => (acc.includes(j.group) ? acc : [...acc, j.group]), []);
  return {
    ...raw,
    robot: raw.robot ?? robot,
    title: raw.title ?? (robot === "microduck" ? "Microduck" : robot),
    numJoints: raw.numJoints ?? raw.joints.length,
    groups,
    standHeight: raw.standHeight ?? 0,
    sizeScale: raw.sizeScale ?? 1,
    effectors: raw.effectors ?? [],
  };
}

export async function fetchJoints(robot: RobotId = "microduck"): Promise<JointsMeta> {
  return normalizeMeta(await jsonOrThrow(await fetch(`${LAB_HTTP}/joints${robotQuery(robot)}`)), robot);
}

export interface PoseResult {
  bodies: number[][]; // per body [x, y, z, qw, qx, qy, qz] — /scene body order
  balance?: Balance; // absent from a lab older than the endpoint
  joints: number[]; // clamped to the servo limits
  rootPitch: number;
  /** Which body answered — absent from a lab older than the robot switch. */
  robot?: RobotId;
  /** World position of every draggable point for THIS pose, same frame as
   *  `bodies` — the IK handles. Absent from a lab older than /ik. */
  effectors?: Record<string, [number, number, number]>;
}

export interface IkTarget {
  pos: [number, number, number];
  /** Hold the body as level as it stands (a foot flat). The server defaults
   *  it on for feet. */
  level?: boolean;
  weight?: number;
}
export type IkTargets = Record<string, IkTarget>;

export interface IkInfo {
  iterations: number;
  converged: boolean;
  /** How far short of its target each effector ended, metres. */
  residual: Record<string, number>;
  /** The effectors the server held where they were. */
  pins: string[];
}

/** POST /ik's answer: a /pose answer for THE SOLVED JOINTS, plus the solve. */
export type IkResult = PoseResult & { ik: IkInfo };

export function isIkResult(r: PoseResult | IkResult): r is IkResult {
  return "ik" in r && r.ik != null;
}

export async function fetchPose(pose: Pose, robot: RobotId = "microduck", signal?: AbortSignal): Promise<PoseResult> {
  return jsonOrThrow(
    await fetch(`${LAB_HTTP}/pose${robotQuery(robot)}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ joints: pose.joints, rootPitch: pose.rootPitch }),
      signal,
    })
  );
}

/** Inverse kinematics for a dragged effector: the pose that puts each target
 *  where it was asked. `pins` omitted = the server pins every foot that is
 *  not a target (the planted foot stays put while the other is dragged).
 *  Targets are in the frame /pose reports bodies in; an unreachable one is
 *  answered with the closest pose and a nonzero residual, never an error. */
export async function fetchIk(
  pose: Pose,
  targets: IkTargets,
  robot: RobotId = "microduck",
  pins?: string[],
  signal?: AbortSignal
): Promise<IkResult> {
  return jsonOrThrow(
    await fetch(`${LAB_HTTP}/ik${robotQuery(robot)}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        joints: pose.joints,
        rootPitch: pose.rootPitch,
        targets,
        ...(pins ? { pins } : {}),
      }),
      signal,
    })
  );
}

export async function listClips(): Promise<StoredClip[]> {
  const data = await jsonOrThrow(await fetch(`${LAB_HTTP}/clips`));
  return data.clips ?? [];
}

export async function loadClip(name: string): Promise<Clip> {
  return jsonOrThrow(await fetch(`${LAB_HTTP}/clips/${encodeURIComponent(name)}`));
}

export async function putClip(clip: Clip): Promise<StoredClip> {
  return jsonOrThrow(
    await fetch(`${LAB_HTTP}/clips/${encodeURIComponent(clip.name)}`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(clip),
    })
  );
}

export async function removeClip(name: string): Promise<void> {
  await jsonOrThrow(
    await fetch(`${LAB_HTTP}/clips/${encodeURIComponent(name)}`, { method: "DELETE" })
  );
}

/** What the streamer has queued: a plain preview, or an IK solve. */
type StreamJob =
  | { kind: "pose"; pose: Pose; robot: RobotId }
  | { kind: "ik"; pose: Pose; targets: IkTargets; pins?: string[]; robot: RobotId };

/** POST /pose (or /ik) with at most ONE request in flight and always a
 *  trailing send, so a slider or handle drag stays responsive instead of
 *  queueing a backlog (the server answers in ~0.2 ms for FK, ~1 ms for IK;
 *  the round trip is the only real cost). The newest request of EITHER kind
 *  wins the trailing slot. An IK answer carries the SOLVED joints — the
 *  callback tells them apart with `isIkResult` and adopts those.
 *
 *  Every job remembers the robot it was sent for, and an answer for a body
 *  the streamer has since been switched away from is dropped: a duck's 17
 *  body poses landing on the G1's 45 groups would draw a heap of parts. */
export class PoseStreamer {
  private inflight = false;
  private pending: StreamJob | null = null;
  private closed = false;
  private robot: RobotId = "microduck";

  constructor(
    private onPose: (r: PoseResult | IkResult) => void,
    private onError?: (e: string) => void
  ) {}

  /** Which body the next requests are for. */
  setRobot(robot: RobotId) {
    this.robot = robot;
  }

  request(pose: Pose) {
    this.pending = { kind: "pose", pose: copyPose(pose), robot: this.robot };
    this.pump();
  }

  requestIk(pose: Pose, targets: IkTargets, pins?: string[]) {
    this.pending = { kind: "ik", pose: copyPose(pose), targets, pins, robot: this.robot };
    this.pump();
  }

  private pump() {
    if (this.closed || this.inflight || !this.pending) return;
    const job = this.pending;
    this.pending = null;
    this.inflight = true;
    const req =
      job.kind === "ik"
        ? fetchIk(job.pose, job.targets, job.robot, job.pins)
        : fetchPose(job.pose, job.robot);
    req
      .then((r) => {
        if (this.closed) return;
        // A lab older than the switch names no robot: it is the duck's.
        if ((r.robot ?? "microduck") !== this.robot || job.robot !== this.robot) return;
        this.onPose(r);
      })
      .catch((e) => {
        if (!this.closed) this.onError?.(String(e?.message ?? e));
      })
      .finally(() => {
        this.inflight = false;
        this.pump(); // trailing edge: the latest request always lands
      });
  }

  close() {
    this.closed = true;
  }
}

function copyPose(pose: Pose): Pose {
  return { joints: [...pose.joints], rootPitch: pose.rootPitch };
}

// ----------------------------------------------------------- the shared store

export type AnimMode = "joints" | "rig" | "ik";

/** The rig selection PoseDuck needs for highlighting/labeling — a mirror of
 *  lib/rig.ts's RigBodyPick essentials, kept here to avoid an import cycle. */
export interface RigSelection {
  id: string;
  label: string;
  bodies: number[];
}

/** Where the posed duck's centre of mass sits relative to each sole, from
 *  POST /pose. `marginMm` is the signed distance from the CoM's ground
 *  projection to the edge of that sole's footprint: positive inside it,
 *  negative outside. `grounded` is false for a foot held clear of the floor.
 *  A STATIC check; PoseScratch.balance in viz_server.py says what it does and
 *  does not mean. */
export interface FootBalance {
  grounded: boolean;
  marginMm: number;
  /** The sole's footprint on the floor: world xy in metres, counter-clockwise,
   *  not closed. What the margin was measured against, so the scene can draw
   *  it under the crosshair. */
  outline: number[][];
}
export type Side = "left" | "right";
/** The support polygon: the hull of every grounded sole. A body stands in
 *  it statically; standing square the CoM is well inside it and well outside
 *  both soles at once, which is why the per-sole read alone would call every
 *  square stance a fall. */
export interface SupportBalance {
  feet: Side[];
  marginMm: number;
  outline: number[][];
}
export interface Balance {
  com: number[];
  feet: Record<Side, FootBalance>;
  /** The grounded foot whose footprint holds the CoM, or null. */
  over: Side | null;
  support: SupportBalance;
}

/** The three answers the marker can give, in order of how much the pose
 *  asks of the duck: over a single sole (a one-legged hold is on), inside the
 *  two-foot stance (it stands, on both feet), or outside everything. */
export type BalanceState = "sole" | "stance" | "out";

export function balanceState(b: Balance): BalanceState {
  if (b.over) return "sole";
  return b.support.marginMm > 0 ? "stance" : "out";
}

const SIDES: Side[] = ["left", "right"];

/** The foot the CoM is nearest to standing on: the larger margin among the
 *  feet that are down (a foot in the air cannot be stood on, however well
 *  the CoM lines up with it). `over` names a foot only once the CoM is
 *  inside its footprint; this still has an answer when it is outside both. */
export function nearestFoot(b: Balance): { side: Side; marginMm: number } {
  const down = SIDES.filter((s) => b.feet[s].grounded);
  const side = (down.length ? down : SIDES).reduce((a, s) =>
    b.feet[s].marginMm > b.feet[a].marginMm ? s : a
  );
  return { side, marginMm: b.feet[side].marginMm };
}

/** Margins this close are the same margin: standing square the two agree to
 *  within float dust, and naming a foot there would invent a lean. */
const SAME_MARGIN_MM = 0.5;

/** The readout beside the toggle. Millimetres, because that is the scale a
 *  sole is on (its flat is ~45 x 34 mm). */
export function balanceLabel(b: Balance): string {
  if (b.over) return `${b.feet[b.over].marginMm.toFixed(1)} mm inside the ${b.over} sole`;
  const { side, marginMm } = nearestFoot(b);
  const other = side === "left" ? "right" : "left";
  const bothDown = b.feet.left.grounded && b.feet.right.grounded;
  const square = bothDown && Math.abs(b.feet.left.marginMm - b.feet.right.marginMm) <= SAME_MARGIN_MM;
  const where = square ? "both soles" : `the ${side} sole`;
  const air = bothDown ? "" : ` (${other} foot in the air)`;
  // Inside the stance but over neither sole: it stands on two feet, and the
  // one-legged question is how far the mass has to travel to reach a sole.
  if (bothDown && b.support.marginMm > 0) {
    const toward = square ? "a sole" : `the ${side} sole`;
    return `${b.support.marginMm.toFixed(1)} mm inside the stance, ${(-marginMm).toFixed(1)} mm short of ${toward}`;
  }
  return `${(-marginMm).toFixed(1)} mm outside ${where}${air}`;
}

/** True when the two would read the same on the panel, so a pose response
 *  that changes nothing visible costs no render. */
export function sameBalance(a: Balance | null, b: Balance | null): boolean {
  if (!a || !b) return a === b;
  return (
    a.over === b.over &&
    a.support.marginMm === b.support.marginMm &&
    SIDES.every(
      (s) => a.feet[s].grounded === b.feet[s].grounded && a.feet[s].marginMm === b.feet[s].marginMm
    )
  );
}

/** Green once the CoM is inside a sole, blue while it is inside the two-foot
 *  stance, amber once it is outside everything: the marker's colour is the
 *  answer at a glance and the readout is the detail. Amber rather than red
 *  on purpose, because a negative margin is a pose that is hard to hold
 *  statically, not a pose that is wrong. */
export const BALANCE_IN = "#7dd87d";
export const BALANCE_STANCE = "#6fb7f0";
export const BALANCE_OUT = "#e8b24a";

export function balanceColor(b: Balance | null): string {
  if (!b) return BALANCE_OUT;
  const state = balanceState(b);
  return state === "sole" ? BALANCE_IN : state === "stance" ? BALANCE_STANCE : BALANCE_OUT;
}

export interface AnimStore {
  /** Preview robot visible + interactive (the panel is open). */
  visible: boolean;
  /** Which body the editor is posing — PoseDuck draws this one. Set with
   *  the metadata (`setAnimMeta`), which notifies. */
  robot: RobotId;
  /** This body's size against the duck (JointsMeta.sizeScale): every
   *  metre-sized gizmo constant in the scene is multiplied by it. */
  sizeScale: number;
  /** Latest body poses from POST /pose — read per-frame by PoseDuck. */
  bodies: number[][] | null;
  /** Where every IK handle sits for that same pose, effector id → MuJoCo
   *  xyz in the ghost group's frame. Per-frame data like `bodies`: does not
   *  notify. */
  effectors: Record<string, number[]> | null;
  /** The IK handle a click picked, or null. Mutually exclusive with the
   *  joint and rig selections. */
  selectedEffector: string | null;
  /** How far short of its target each effector ended on the last /ik
   *  answer, metres — a handle past 1 cm (× sizeScale) turns amber. */
  ikResidual: Record<string, number>;
  /** Registered by the panel: a 3D handle drag asks the solver to put the
   *  effector at `pos` (ghost-group frame, the frame /pose reports in). */
  applyIkDrag: ((id: string, pos: [number, number, number]) => void) | null;
  /** CoM vs the soles for that same pose: the editor can show a shape but
   *  not whether it would STAND. Per-frame data like `bodies`: does not
   *  notify. */
  balance: Balance | null;
  /** Draw the CoM marker in the scene. Owned (and persisted) by the panel's
   *  ⊕ toggle, off by default so the editor's default view is unchanged. */
  showBalance: boolean;
  /** What a 3D click/drag edits: one servo, or the rig control mapped to the
   *  clicked body part. Owned (and persisted) by the panel's mode toggle. */
  mode: AnimMode;
  /** Selected joint index, ROOT_SEL for the trunk, or null. */
  selected: number | null;
  /** Selected rig control (mutually exclusive with `selected`). */
  selectedRig: RigSelection | null;
  /** body index → rig-mode pick (control id, gearing) — built by the panel
   *  from lib/rig.ts once the joint metadata is known. */
  rigForBody: ({ rigId: string; label: string; bodies: number[]; gearJoint: number; gearCoeff: number } | null)[];
  /** Body under the cursor in the 3D scene (highlight only). */
  hoveredBody: number | null;
  /** True while a 3D joint drag is in progress (suppresses OrbitControls). */
  dragging: boolean;
  /** body index → joint index, built from /joints. */
  jointForBody: (number | null)[];
  meta: JointsMeta | null;
  /** Registered by the panel: a 3D drag applies its delta through this so the
   *  clip/pose state stays single-sourced in React. */
  applyJointDelta: ((joint: number, deltaRad: number) => void) | null;
  /** Same idea for the rig handles (lib/rig.ts): the ⇕ squat gizmo nudges a
   *  rig control by id, and the panel owns turning that into a pose. */
  applyRigDelta: ((rigId: string, delta: number) => void) | null;
  /** Set by the panel's ◎ button, consumed once by the in-Canvas helper. */
  focusRequest: number;
  /** The panel's own element. ◎ focus reads its top edge so it frames the
   *  duck in the stage the panel is NOT covering — the panel's height is
   *  viewport-relative, so a hard-coded offset would be wrong half the time. */
  panelEl: HTMLElement | null;
}

export const animStore: AnimStore = {
  visible: false,
  robot: "microduck",
  sizeScale: 1,
  bodies: null,
  effectors: null,
  selectedEffector: null,
  ikResidual: {},
  applyIkDrag: null,
  balance: null,
  showBalance: false,
  mode: "joints",
  selected: null,
  selectedRig: null,
  rigForBody: [],
  hoveredBody: null,
  dragging: false,
  jointForBody: [],
  meta: null,
  applyJointDelta: null,
  applyRigDelta: null,
  focusRequest: 0,
  panelEl: null,
};

const listeners = new Set<() => void>();
let version = 0;

/** Bump the React-visible version (selection/visibility change). Per-frame
 *  data — `bodies`, `hoveredBody` — deliberately does NOT notify. */
export function animNotify() {
  version++;
  listeners.forEach((l) => l());
}

export function subscribeAnim(cb: () => void) {
  listeners.add(cb);
  return () => {
    listeners.delete(cb);
  };
}

export function animVersion() {
  return version;
}

export function setSelected(sel: number | null) {
  if (animStore.selected === sel && animStore.selectedRig === null && animStore.selectedEffector === null) return;
  animStore.selected = sel;
  animStore.selectedRig = null; // one selection at a time — joint XOR rig XOR effector
  animStore.selectedEffector = null;
  animNotify();
}

export function setSelectedRig(sel: RigSelection | null) {
  if (animStore.selectedRig?.id === sel?.id && animStore.selected === null && animStore.selectedEffector === null) return;
  animStore.selectedRig = sel;
  animStore.selected = null;
  animStore.selectedEffector = null;
  animNotify();
}

export function setSelectedEffector(id: string | null) {
  if (animStore.selectedEffector === id && animStore.selected === null && animStore.selectedRig === null) return;
  animStore.selectedEffector = id;
  animStore.selected = null;
  animStore.selectedRig = null;
  animNotify();
}

export function setShowBalance(v: boolean) {
  if (animStore.showBalance === v) return;
  animStore.showBalance = v;
  animNotify();
}

export function setAnimMode(mode: AnimMode) {
  if (animStore.mode === mode) return;
  animStore.mode = mode;
  animNotify();
}

export function setAnimVisible(v: boolean) {
  if (animStore.visible === v) return;
  animStore.visible = v;
  animNotify();
}

export function setAnimMeta(meta: JointsMeta) {
  // A different body: everything measured on the old one is meaningless on
  // it (its pose list is the wrong length for the new groups), and nothing
  // is selected on a robot that has just arrived.
  if (meta.robot !== animStore.robot || meta.bodies.length !== animStore.meta?.bodies.length) {
    animStore.bodies = null;
    animStore.balance = null;
    animStore.effectors = null;
    animStore.ikResidual = {};
    animStore.selected = null;
    animStore.selectedRig = null;
    animStore.selectedEffector = null;
    animStore.hoveredBody = null;
  }
  animStore.meta = meta;
  animStore.robot = meta.robot;
  animStore.sizeScale = meta.sizeScale;
  const map: (number | null)[] = new Array(meta.bodies.length).fill(null);
  for (const j of meta.joints) map[j.body] = j.index;
  map[meta.trunkBody] = ROOT_SEL; // clicking the body itself edits root pitch
  animStore.jointForBody = map;
  animNotify();
}

/** The effector an IK-mode click on a body part picks: the one whose chain
 *  the body's joint is on (a shin → that foot), else the one riding the body
 *  itself (the jaw → the head). Null for a part no effector can move. */
export function effectorForBody(meta: JointsMeta, body: number): EffectorMeta | null {
  const joint = meta.joints.find((j) => j.body === body)?.index;
  if (joint != null) {
    const onChain = meta.effectors.find((e) => e.chain.includes(joint));
    if (onChain) return onChain;
  }
  return meta.effectors.find((e) => e.body === body) ?? null;
}

/** Where the preview robot stands, in MuJoCo XY. Clear of the lab grid,
 *  which starts at y = 0 and grows toward +y (see Viewer's gridOffsets): the
 *  duck at [0, -1.05], a bigger body proportionally further out so its
 *  locator ring and feet clear the grid too. */
export function previewOffset(sizeScale: number): [number, number] {
  return [0, -1.05 * Math.max(1, sizeScale)];
}
