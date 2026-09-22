// The rig's arithmetic: which controls a body gets, how a control resolves
// against the joint metadata, and the two properties the animator relies on
// — controls are orthogonal (squatting never moves the lean slider) and a
// slider drag lands exactly where it was asked.
//
// The fixture is the duck's joint layout (names, order, DEFAULT_POSE), with
// wide limits so the round-trips are not clipped by a servo; the real limits
// are the MJCF's and rigRange's answer about them is checked separately.

import { describe, expect, it } from "vitest";
import { type JointsMeta, type Pose, normalizeMeta } from "./anim";
import {
  RIG_CONTROLS,
  rigApply,
  rigBodyMap,
  rigControlsFor,
  rigMeasure,
  rigRange,
  rigVector,
  type RigControl,
} from "./rig";

const NAMES = [
  "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
  "neck_pitch", "head_pitch", "head_yaw", "head_roll",
  "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
];
const DEFAULT = [
  0.0, -0.0873, -0.4579, -0.0049, 0.453,
  0.3491, 0.3491, 0.0, 0.0,
  0.0, 0.0873, 0.4579, 0.0049, -0.453,
];
const GROUP = (i: number) => (i < 5 ? "left leg" : i < 9 ? "head + neck" : "right leg");

function duckMeta(limit = 3): JointsMeta {
  return normalizeMeta(
    {
      joints: NAMES.map((name, index) => ({
        index, name, group: GROUP(index), min: -limit, max: limit, default: DEFAULT[index],
        body: index + 2, bodyName: name, axis: [0, 1, 0], pos: [0, 0, 0],
      })),
      bodies: ["world", "trunk", ...NAMES],
      trunkBody: 1,
      rootPitchRange: [-6.283185, 6.283185],
      rootPitchSign: "negative = lean back",
    },
    "microduck"
  );
}
const META = duckMeta();
const STAND: Pose = { joints: [...DEFAULT], rootPitch: 0 };
const ctrl = (id: string) => RIG_CONTROLS.find((c) => c.id === id)!;
const vec = (id: string, meta = META) => rigVector(meta, ctrl(id))!;

describe("rigControlsFor", () => {
  it("is the duck's constant without metadata, or when the lab serves no rig", () => {
    expect(rigControlsFor(null)).toBe(RIG_CONTROLS);
    expect(rigControlsFor(META)).toBe(RIG_CONTROLS);
    expect(rigControlsFor({ ...META, rig: [] })).toBe(RIG_CONTROLS);
  });
  it("is the lab's own list when it serves one — the G1's names its own joints", () => {
    const g1rig: RigControl[] = [
      {
        id: "squat", label: "squat", hint: "+ crouch", title: "fold",
        parts: { left_knee_joint: 2, right_knee_joint: 2 },
        pick: ["left_knee_joint"], handle: { joint: "root", offset: [-0.3, 0, 0.1] },
      },
    ];
    expect(rigControlsFor({ ...META, robot: "g1", rig: g1rig })).toBe(g1rig);
  });
});

describe("rigVector", () => {
  it("resolves every part of a control against the metadata, root included", () => {
    const v = vec("lean");
    expect(v.parts).toHaveLength(7);
    const root = v.parts.find((p) => p.index === -1)!;
    expect(root.name).toBe("root pitch");
    expect(root.min).toBeCloseTo(-6.283185);
    const knee = v.parts.find((p) => p.name === "left_knee")!;
    expect(knee.index).toBe(3);
    expect(knee.def).toBeCloseTo(-0.0049);
    expect(v.norm2).toBeCloseTo(1 + 6 / 9);
  });
  it("is null for a control naming a joint this body lacks — the control is hidden, not broken", () => {
    const headless = { ...META, joints: META.joints.filter((j) => !j.name.startsWith("head")) };
    expect(rigVector(headless, ctrl("look"))).toBeNull();
    expect(rigVector(headless, ctrl("squat"))).not.toBeNull();
    const g1ish: RigControl = { ...ctrl("squat"), parts: { left_knee_joint: 2 } };
    expect(rigVector(META, g1ish)).toBeNull();
  });
});

describe("orthogonality", () => {
  it("squat and lean are orthogonal: using one never moves the other's slider", () => {
    const squat = vec("squat");
    const lean = vec("lean");
    const crouched = rigApply(squat, STAND, 0.4);
    expect(rigMeasure(squat, crouched)).toBeCloseTo(0.4);
    expect(rigMeasure(lean, crouched)).toBeCloseTo(0);
    const leaning = rigApply(lean, STAND, 0.3);
    expect(rigMeasure(lean, leaning)).toBeCloseTo(0.3);
    expect(rigMeasure(squat, leaning)).toBeCloseTo(0);
  });
  it("holds for every pair of the duck's controls", () => {
    const vs = RIG_CONTROLS.map((c) => rigVector(META, c)!);
    for (const a of vs)
      for (const b of vs) {
        if (a === b) continue;
        let dot = 0;
        for (const pa of a.parts) {
          const pb = b.parts.find((p) => p.index === pa.index);
          if (pb) dot += pa.coeff * pb.coeff;
        }
        expect(dot, `${a.ctrl.id} · ${b.ctrl.id}`).toBeCloseTo(0);
      }
  });
});

describe("rigApply / rigMeasure", () => {
  it("round-trips: the slider lands exactly where it was asked, from anywhere", () => {
    for (const c of RIG_CONTROLS) {
      const v = rigVector(META, c)!;
      expect(rigMeasure(v, STAND), c.id).toBeCloseTo(0);
      const moved = rigApply(v, STAND, 0.25);
      expect(rigMeasure(v, moved), c.id).toBeCloseTo(0.25);
      expect(rigMeasure(v, rigApply(v, moved, -0.1)), c.id).toBeCloseTo(-0.1);
      expect(rigMeasure(v, rigApply(v, moved, 0)), c.id).toBeCloseTo(0);
    }
  });
  it("leaves everything orthogonal to the control exactly as it was", () => {
    const squat = vec("squat");
    const lean = vec("lean");
    // A crouch with an asymmetric tweak on top of it.
    const tweaked = rigApply(squat, STAND, 0.3);
    tweaked.joints[2] += 0.05; // left hip pitch alone
    const then = rigApply(lean, tweaked, 0.2);
    expect(rigMeasure(squat, then)).toBeCloseTo(rigMeasure(squat, tweaked));
    // Put lean back where the tweaked pose had it (the tweak itself projects
    // a little onto lean) and the tweak is still there, to the bit.
    const back = rigApply(lean, then, rigMeasure(lean, tweaked));
    back.joints.forEach((q, i) => expect(q).toBeCloseTo(tweaked.joints[i], 10));
    expect(back.rootPitch).toBeCloseTo(tweaked.rootPitch, 10);
  });
  it("moves the trunk for lean and only the named joints for a swing", () => {
    const leaning = rigApply(vec("lean"), STAND, 0.3);
    expect(leaning.rootPitch).toBeCloseTo(0.3);
    const swung = rigApply(vec("swingL"), STAND, 0.2);
    expect(swung.rootPitch).toBe(0);
    swung.joints.forEach((q, i) => {
      if (i === 2) expect(q).toBeCloseTo(DEFAULT[2] - 0.2); // left hip pitch, coeff −1
      else if (i === 4) expect(q).toBeCloseTo(DEFAULT[4] + 0.2); // left ankle, coeff +1
      else expect(q).toBe(DEFAULT[i]);
    });
  });
  it("stops at the first servo's limit and names it", () => {
    const tight = duckMeta(0.5);
    const v = vec("squat", tight);
    const r = rigRange(v, STAND);
    // Standing, the hip pitch already sits at 0.458 of its 0.5 and folds
    // further with +squat (1 per unit), so it ends the + travel at 0.042;
    // the ankle (0.453 of 0.5) unwinds with −squat and ends that at −0.047.
    // The knee, 2 per unit but starting near 0, is nowhere near either.
    expect(r.max).toBeCloseTo(0.5 - 0.4579);
    expect(r.min).toBeCloseTo(-(0.5 - 0.453));
    expect(["left_hip_pitch", "right_hip_pitch"]).toContain(r.maxBy);
    expect(["left_ankle", "right_ankle"]).toContain(r.minBy);
    expect(rigMeasure(v, rigApply(v, STAND, 5))).toBeCloseTo(r.max);
    expect(rigMeasure(v, rigApply(v, STAND, -5))).toBeCloseTo(r.min);
  });
});

describe("rigBodyMap", () => {
  it("maps each picked body to its control, geared by the picked joint", () => {
    const map = rigBodyMap(META, RIG_CONTROLS.map((c) => rigVector(META, c)!));
    const knee = map[3 + 2]!; // left knee's body
    expect(knee.rigId).toBe("squat");
    expect(knee.gearJoint).toBe(3);
    expect(knee.gearCoeff).toBe(-2);
    expect(knee.bodies).toHaveLength(6);
    // The trunk picks lean (its "root" part).
    expect(map[1]!.rigId).toBe("lean");
    expect(map[1]!.gearJoint).toBe(-1);
    // A zero-coefficient pick (head yaw → look) gears through a driven joint.
    const headYaw = map[7 + 2]!;
    expect(headYaw.rigId).toBe("look");
    expect(headYaw.gearJoint).toBe(5); // neck_pitch, the first driven part
    // Nothing picks the hip roll bodies for "stance" (pick is empty): sway has them.
    expect(map[1 + 2]!.rigId).toBe("sway");
  });
});
