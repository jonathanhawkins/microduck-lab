// 🎮 rig controls: game-style macro handles layered over the raw joint
// sliders. Each control is a DIRECTION in (joints + rootPitch) space — e.g.
// "squat" couples hip pitch, knee and ankle on both legs in the ratio that
// keeps the feet flat and the trunk upright while the legs fold. Editing a
// control projects the current pose onto that direction and slides along it,
// so rig controls compose with each other and with raw joint edits instead of
// stomping them (asymmetric tweaks live in the orthogonal complement and
// survive a rig drag untouched).
//
// The coefficient signs are derived from the WORLD hinge axes of the MJCF at
// the STAND keyframe (x forward, y left, z up; verified against the model):
//
//   left  hip_pitch +y   knee −y   ankle +y      hip_roll +x   hip_yaw −z
//   right hip_pitch −y   knee +y   ankle −y      hip_roll +x   hip_yaw −z
//   neck_pitch −y   head_pitch +y
//
// A flat foot needs the leg's world-pitch sum (root + hip + knee + ankle) to
// stay constant — DEFAULT_POSE sums to 0 per leg, and every control below
// preserves that sum, which is why the preview duck's auto-grounding reads as
// "feet planted" rather than "toes drilling into the floor".

import { ROOT_SEL, type JointsMeta, type Pose } from "./anim";

export interface RigControl {
  id: string;
  label: string;
  /** What + / − mean, animator-facing. */
  hint: string;
  title: string;
  /** joint name (or "root" for rootPitch) → coefficient along the control. */
  parts: Record<string, number>;
  /** Joint names whose BODIES select this control when the scene is in rig
   *  mode — clicking that body part picks this control and dragging drives
   *  it. A picked joint with a nonzero coefficient also gears the drag (the
   *  part under the cursor tracks it 1:1); a zero-coefficient pick (e.g. the
   *  head-yaw body selecting "look") gears through the control's first
   *  driven joint instead. */
  pick: string[];
  /** Where the ⇕ drag handle parks while this control is active: anchored to
   *  a body (by the joint that drives it; "root" = trunk) plus a MuJoCo-frame
   *  offset (x fwd, y left, z up) from that body's origin — so the handle
   *  sits AT the thing it moves and relocating it shows which control is
   *  armed. Dragging the handle down always increases the control. */
  handle: { joint: string; offset: [number, number, number] };
}

/** One unit of "squat" is one radian of thigh-from-vertical: the thigh pitches
 *  back by v, the shank forward by v (knee bends 2v between them), the ankle
 *  unwinds v so the foot stays flat. + = crouch down. */
export const RIG_CONTROLS: RigControl[] = [
  {
    id: "squat",
    handle: { joint: "root", offset: [-0.105, 0, 0.03] },
    pick: ["left_knee", "right_knee"],
    label: "squat",
    hint: "+ crouch",
    title: "fold both legs symmetrically, feet flat, trunk upright — the ⇕ handle drags this when no other control is selected",
    parts: {
      left_hip_pitch: -1, left_knee: -2, left_ankle: -1,
      right_hip_pitch: 1, right_knee: 2, right_ankle: 1,
    },
  },
  {
    id: "lean",
    handle: { joint: "root", offset: [-0.105, 0, 0.115] },
    pick: ["root"],
    label: "lean",
    hint: "+ fwd",
    title: "the trunk pitches while the legs counterbalance, feet flat",
    // The counter-pitch is spread evenly over hip/knee/ankle (−⅓ world each)
    // rather than taken at the hip alone: that makes this direction exactly
    // ORTHOGONAL to squat in joint space, so squatting never moves the lean
    // slider and vice versa. Every control pair here is orthogonal — rig
    // sliders read 0 until you actually use them.
    parts: {
      root: 1,
      left_hip_pitch: -1 / 3, left_knee: 1 / 3, left_ankle: -1 / 3,
      right_hip_pitch: 1 / 3, right_knee: -1 / 3, right_ankle: 1 / 3,
    },
  },
  // Per-leg stride: the whole leg swings about the hip while the ankle
  // unwinds to keep the foot level — key one forward and the other back for a
  // run stride, and the slider ends ARE the leg's max extensions (whichever
  // of the hip or ankle servo runs out of travel first). Orthogonal to squat
  // and lean, so a stride keyed over a crouch keeps the crouch.
  {
    id: "swingL",
    handle: { joint: "left_hip_pitch", offset: [0, 0.07, 0] },
    pick: ["left_hip_pitch"],
    label: "L swing",
    hint: "+ fwd",
    title: "swing the whole left leg forward/back about the hip, foot kept level — pair with R swing for a stride",
    parts: { left_hip_pitch: -1, left_ankle: 1 },
  },
  {
    id: "swingR",
    handle: { joint: "right_hip_pitch", offset: [0, -0.07, 0] },
    pick: ["right_hip_pitch"],
    label: "R swing",
    hint: "+ fwd",
    title: "swing the whole right leg forward/back about the hip, foot kept level — pair with L swing for a stride",
    parts: { right_hip_pitch: 1, right_ankle: -1 },
  },
  {
    id: "sway",
    handle: { joint: "root", offset: [0, 0.115, 0.01] },
    pick: ["left_hip_roll", "right_hip_roll"],
    label: "sway",
    hint: "hips ±",
    title: "both hip rolls together — swing the legs sideways under the trunk",
    parts: { left_hip_roll: 1, right_hip_roll: 1 },
  },
  {
    id: "stance",
    handle: { joint: "root", offset: [0, -0.115, 0.01] },
    pick: [],
    label: "stance",
    hint: "+ wide",
    title: "hip rolls apart — widen or narrow the stance",
    parts: { left_hip_roll: -1, right_hip_roll: 1 },
  },
  {
    id: "twist",
    handle: { joint: "root", offset: [-0.145, 0, -0.025] },
    pick: ["left_hip_yaw", "right_hip_yaw"],
    label: "twist",
    hint: "hips ±",
    title: "both hip yaws together — pivot the hips against the feet",
    parts: { left_hip_yaw: 1, right_hip_yaw: 1 },
  },
  {
    id: "toes",
    handle: { joint: "left_ankle", offset: [0.075, 0, 0.015] },
    pick: ["left_ankle", "right_ankle"],
    label: "toes",
    hint: "+ out",
    title: "hip yaws apart — duck-foot or pigeon-toe the stance",
    parts: { left_hip_yaw: -1, right_hip_yaw: 1 },
  },
  {
    id: "look",
    handle: { joint: "head_pitch", offset: [-0.02, 0, 0.115] },
    pick: ["neck_pitch", "head_pitch", "head_yaw", "head_roll"],
    label: "look",
    hint: "+ down",
    title: "neck and head pitch share the motion — one radian of control is one radian of gaze",
    parts: { neck_pitch: -0.5, head_pitch: 0.5 },
  },
];

/** Resolved (index, coefficient, limits, default) per part — ROOT_SEL carries
 *  rootPitch. Null when a joint name is missing from the lab's metadata. */
export interface RigVector {
  ctrl: RigControl;
  parts: { index: number; name: string; coeff: number; min: number; max: number; def: number }[];
  /** Σ coeff² — the projection denominator. */
  norm2: number;
}

export function rigVector(meta: JointsMeta, ctrl: RigControl): RigVector | null {
  const parts: RigVector["parts"] = [];
  let norm2 = 0;
  for (const [name, coeff] of Object.entries(ctrl.parts)) {
    if (name === "root") {
      parts.push({
        index: ROOT_SEL, name: "root pitch", coeff,
        min: meta.rootPitchRange[0], max: meta.rootPitchRange[1], def: 0,
      });
    } else {
      const j = meta.joints.find((x) => x.name === name);
      if (!j) return null; // lab model without this joint — hide the control
      parts.push({ index: j.index, name, coeff, min: j.min, max: j.max, def: j.default });
    }
    norm2 += coeff * coeff;
  }
  return { ctrl, parts, norm2 };
}

const at = (pose: Pose, index: number) =>
  index === ROOT_SEL ? pose.rootPitch : pose.joints[index] ?? 0;

/** Where the pose sits along the control (0 = DEFAULT_POSE). */
export function rigMeasure(v: RigVector, pose: Pose): number {
  let dot = 0;
  for (const p of v.parts) dot += p.coeff * (at(pose, p.index) - p.def);
  return dot / v.norm2;
}

export interface RigRange {
  min: number;
  max: number;
  /** The servo whose MJCF limit ends the travel at each end — THE rig-limits
   *  answer to "why won't it go further". */
  minBy: string;
  maxBy: string;
}

/** How far the control can travel from the CURRENT pose before some servo
 *  hits its real limit. Moving along the control keeps these bounds fixed
 *  (each involved joint is linear in the control value), so the slider does
 *  not squirm under the pointer mid-drag. */
export function rigRange(v: RigVector, pose: Pose): RigRange {
  const val = rigMeasure(v, pose);
  let lo = -Infinity, hi = Infinity, minBy = "", maxBy = "";
  for (const p of v.parts) {
    const cur = at(pose, p.index);
    const a = (p.min - cur) / p.coeff;
    const b = (p.max - cur) / p.coeff;
    const dLo = Math.min(a, b), dHi = Math.max(a, b);
    if (val + dLo > lo) { lo = val + dLo; minBy = p.name; }
    if (val + dHi < hi) { hi = val + dHi; maxBy = p.name; }
  }
  return { min: lo, max: hi, minBy, maxBy };
}

/** What a body part means in rig mode: which control a click on it selects,
 *  and how a drag of it gears into that control. */
export interface RigBodyPick {
  rigId: string;
  label: string;
  /** Every body the control drives — all highlighted while it is selected. */
  bodies: number[];
  /** The joint whose hinge anchors the 3D drag (ROOT_SEL = trunk pitch):
   *  circling the cursor around this hinge moves the control so the part
   *  under the cursor tracks 1:1. */
  gearJoint: number;
  gearCoeff: number;
}

const bodyOfPart = (meta: JointsMeta, index: number) =>
  index === ROOT_SEL ? meta.trunkBody : meta.joints[index]?.body ?? -1;

/** Every body a control drives — what lights up while it is selected. */
export function rigBodies(meta: JointsMeta, v: RigVector): number[] {
  return [
    ...new Set(v.parts.filter((p) => p.coeff !== 0).map((p) => bodyOfPart(meta, p.index))),
  ];
}

/** body index → rig pick, from each control's `pick` list. Bodies no control
 *  claims stay null (clicks on them fall back to plain joint editing). */
export function rigBodyMap(meta: JointsMeta, vectors: RigVector[]): (RigBodyPick | null)[] {
  const map: (RigBodyPick | null)[] = new Array(meta.bodies.length).fill(null);
  for (const v of vectors) {
    const driven = v.parts.filter((p) => p.coeff !== 0);
    const bodies = rigBodies(meta, v);
    for (const name of v.ctrl.pick) {
      const index = name === "root" ? ROOT_SEL : meta.joints.find((j) => j.name === name)?.index;
      if (index === undefined) continue;
      // Gear preference: the picked joint itself, else a driven joint on the
      // same side (right foot → right hip yaw), else whatever drives at all.
      const side = name.startsWith("left_") ? "left_" : name.startsWith("right_") ? "right_" : null;
      const gear =
        driven.find((p) => p.index === index) ??
        (side ? driven.find((p) => p.name.startsWith(side)) : undefined) ??
        driven[0];
      if (!gear) continue;
      map[bodyOfPart(meta, index)] = {
        rigId: v.ctrl.id,
        label: v.ctrl.label,
        bodies,
        gearJoint: gear.index,
        gearCoeff: gear.coeff,
      };
    }
  }
  return map;
}

/** Slide the pose along the control to `value` (clamped to the rig range);
 *  everything orthogonal to the control is left exactly as it was. */
export function rigApply(v: RigVector, pose: Pose, value: number): Pose {
  const r = rigRange(v, pose);
  const delta = Math.min(r.max, Math.max(r.min, value)) - rigMeasure(v, pose);
  const joints = [...pose.joints];
  let rootPitch = pose.rootPitch;
  for (const p of v.parts) {
    const next = at(pose, p.index) + p.coeff * delta;
    if (p.index === ROOT_SEL) rootPitch = next;
    else joints[p.index] = next;
  }
  return { joints, rootPitch };
}
