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
}

/** One unit of "squat" is one radian of thigh-from-vertical: the thigh pitches
 *  back by v, the shank forward by v (knee bends 2v between them), the ankle
 *  unwinds v so the foot stays flat. + = crouch down. */
export const RIG_CONTROLS: RigControl[] = [
  {
    id: "squat",
    label: "squat",
    hint: "+ crouch",
    title: "fold both legs symmetrically, feet flat, trunk upright — the ⇕ handle on the duck drags this",
    parts: {
      left_hip_pitch: -1, left_knee: -2, left_ankle: -1,
      right_hip_pitch: 1, right_knee: 2, right_ankle: 1,
    },
  },
  {
    id: "lean",
    label: "lean",
    hint: "+ fwd",
    title: "hinge at the hips: the trunk pitches while the legs (and feet) stay put",
    parts: { root: 1, left_hip_pitch: -1, right_hip_pitch: 1 },
  },
  {
    id: "sway",
    label: "sway",
    hint: "hips ±",
    title: "both hip rolls together — swing the legs sideways under the trunk",
    parts: { left_hip_roll: 1, right_hip_roll: 1 },
  },
  {
    id: "stance",
    label: "stance",
    hint: "+ wide",
    title: "hip rolls apart — widen or narrow the stance",
    parts: { left_hip_roll: -1, right_hip_roll: 1 },
  },
  {
    id: "twist",
    label: "twist",
    hint: "hips ±",
    title: "both hip yaws together — pivot the hips against the feet",
    parts: { left_hip_yaw: 1, right_hip_yaw: 1 },
  },
  {
    id: "toes",
    label: "toes",
    hint: "+ out",
    title: "hip yaws apart — duck-foot or pigeon-toe the stance",
    parts: { left_hip_yaw: -1, right_hip_yaw: 1 },
  },
  {
    id: "look",
    label: "look",
    hint: "+ down",
    title: "neck and head pitch share the motion — one radian of control is one radian of gaze",
    parts: { neck_pitch: -0.5, head_pitch: 0.5 },
  },
];

/** Resolved (index, coefficient, limits, default) per part — ROOT_SEL carries
 *  rootPitch. Null when a joint name is missing from the farm's metadata. */
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
      if (!j) return null; // farm model without this joint — hide the control
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
