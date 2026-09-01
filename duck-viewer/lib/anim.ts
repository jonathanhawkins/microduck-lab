// Keyframe animation editor: types, clip math, lab client, and the shared
// store that bridges the DOM panel (AnimPanel) to the in-Canvas preview duck
// (PoseDuck) — same module-level-mutable-object philosophy as lib/assign.ts,
// because a pose drag must update at pointer speed, not at React speed.
//
// The clip JSON is a CONTRACT with the imitation-RL side (it resamples a saved
// clip at 50 Hz and rewards the policy for tracking it) — see viz_server.py's
// docstring. Anything here that touches `Clip` must keep that shape.

import { LAB_HTTP } from "./lab";

// ---------------------------------------------------------------- joint meta

export interface JointMeta {
  index: number;
  name: string;
  group: string; // "left leg" | "head + neck" | "right leg"
  min: number; // MJCF jnt_range — the servo's real travel
  max: number;
  default: number; // DEFAULT_POSE
  body: number; // MuJoCo body this joint drives (index into /scene bodies)
  bodyName: string;
  axis: [number, number, number]; // hinge axis, BODY frame
  pos: [number, number, number]; // hinge anchor, BODY frame
}

export interface JointsMeta {
  joints: JointMeta[];
  bodies: string[];
  trunkBody: number;
  rootPitchRange: [number, number];
  rootPitchSign: string;
}

// ------------------------------------------------------------------ the clip

export interface Key {
  t: number; // seconds from clip start, ascending, first key at 0
  joints: number[]; // 14 ABSOLUTE radians, JOINT_NAMES order
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
}

/** A clip as it comes back from the server listing (mtime added). */
export type StoredClip = Clip & { modified?: number };

export interface Pose {
  joints: number[];
  rootPitch: number;
}

export const NUM_JOINTS = 14;
/** `selected` sentinel for the trunk: it carries rootPitch, not a servo. */
export const ROOT_SEL = -1;

export function defaultPose(meta: JointsMeta | null): Pose {
  return {
    joints: meta ? meta.joints.map((j) => j.default) : new Array(NUM_JOINTS).fill(0),
    rootPitch: 0,
  };
}

export function newClip(meta: JointsMeta | null, name = "untitled"): Clip {
  const p = defaultPose(meta);
  return {
    version: 1,
    name,
    duration: 1.2,
    loop: false,
    keys: [{ t: 0, joints: p.joints, rootPitch: p.rootPitch }],
  };
}

export function clampJoint(meta: JointsMeta | null, i: number, v: number): number {
  const j = meta?.joints[i];
  if (!j) return v;
  return Math.min(j.max, Math.max(j.min, v));
}

/** Linear interpolation in joint space — exactly what the RL resampler does,
 *  so what the timeline shows is what the reward will track. Before the first
 *  key / after the last, the pose is held (no extrapolation). */
export function sampleClip(clip: Clip, t: number): Pose {
  const keys = clip.keys;
  if (!keys.length) return { joints: new Array(NUM_JOINTS).fill(0), rootPitch: 0 };
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

/** Insert/replace a key at `t`, keeping `keys` sorted and t=0 anchored. */
export function withKey(clip: Clip, t: number, pose: Pose): Clip {
  const at = keyAt(clip, t);
  const key: Key = { t: round3(t), joints: [...pose.joints], rootPitch: pose.rootPitch };
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

export async function fetchJoints(): Promise<JointsMeta> {
  return jsonOrThrow(await fetch(`${LAB_HTTP}/joints`));
}

export interface PoseResult {
  bodies: number[][]; // per body [x, y, z, qw, qx, qy, qz] — /scene body order
  joints: number[]; // clamped to the servo limits
  rootPitch: number;
}

export async function fetchPose(pose: Pose, signal?: AbortSignal): Promise<PoseResult> {
  return jsonOrThrow(
    await fetch(`${LAB_HTTP}/pose`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ joints: pose.joints, rootPitch: pose.rootPitch }),
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

/** POST /pose with at most ONE request in flight and always a trailing send,
 *  so a slider drag stays responsive instead of queueing a backlog (the
 *  server answers in ~0.2 ms; the round trip is the only real cost). */
export class PoseStreamer {
  private inflight = false;
  private pending: Pose | null = null;
  private closed = false;

  constructor(private onPose: (r: PoseResult) => void, private onError?: (e: string) => void) {}

  request(pose: Pose) {
    this.pending = { joints: [...pose.joints], rootPitch: pose.rootPitch };
    this.pump();
  }

  private pump() {
    if (this.closed || this.inflight || !this.pending) return;
    const pose = this.pending;
    this.pending = null;
    this.inflight = true;
    fetchPose(pose)
      .then((r) => {
        if (!this.closed) this.onPose(r);
      })
      .catch((e) => {
        if (!this.closed) this.onError?.(String(e?.message ?? e));
      })
      .finally(() => {
        this.inflight = false;
        this.pump(); // trailing edge: the latest pose always lands
      });
  }

  close() {
    this.closed = true;
  }
}

// ----------------------------------------------------------- the shared store

export type AnimMode = "joints" | "rig";

/** The rig selection PoseDuck needs for highlighting/labeling — a mirror of
 *  lib/rig.ts's RigBodyPick essentials, kept here to avoid an import cycle. */
export interface RigSelection {
  id: string;
  label: string;
  bodies: number[];
}

export interface AnimStore {
  /** Preview duck visible + interactive (the panel is open). */
  visible: boolean;
  /** Latest body poses from POST /pose — read per-frame by PoseDuck. */
  bodies: number[][] | null;
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
  bodies: null,
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
  if (animStore.selected === sel && animStore.selectedRig === null) return;
  animStore.selected = sel;
  animStore.selectedRig = null; // one selection at a time — joint XOR rig
  animNotify();
}

export function setSelectedRig(sel: RigSelection | null) {
  if (animStore.selectedRig?.id === sel?.id && animStore.selected === null) return;
  animStore.selectedRig = sel;
  animStore.selected = null;
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
  animStore.meta = meta;
  const map: (number | null)[] = new Array(meta.bodies.length).fill(null);
  for (const j of meta.joints) map[j.body] = j.index;
  map[meta.trunkBody] = ROOT_SEL; // clicking the body itself edits root pitch
  animStore.jointForBody = map;
  animNotify();
}

/** Where the preview duck stands, in MuJoCo XY. Clear of the lab grid, which
 *  starts at y = 0 and grows toward +y (see Viewer's gridOffsets). */
export const PREVIEW_OFFSET: [number, number] = [0, -1.05];
