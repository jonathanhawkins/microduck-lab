// The clip arithmetic (what the timeline shows is what the RL resampler
// will track), the contract's client-side checks, the pose/IK streamer's
// queue, and the balance readout's wording and colour.
//
// The balance fixtures are shapes, not measurements: a margin is a signed
// distance in millimetres and the readout only formats it, so what these
// lock is the wording (which foot, inside or outside, what an airborne foot
// does to the answer) and the tests name the numbers they use. The
// measurement itself is the server's, and tests/test_clips.py holds it
// against the model.
//
// What a test cannot do is judge the marker. Whether a ball and a crosshair
// land where the eye expects is pixels, and pixels are looked at.

import { afterEach, describe, expect, it, vi } from "vitest";
import {
  animStore,
  animVersion,
  BALANCE_IN,
  BALANCE_OUT,
  BALANCE_STANCE,
  balanceColor,
  balanceLabel,
  balanceState,
  clipProblem,
  clipRobot,
  effectorForBody,
  isIkResult,
  keyAt,
  nearestFoot,
  newClip,
  normalizeMeta,
  NUM_JOINTS,
  PoseStreamer,
  previewOffset,
  sameBalance,
  sampleClip,
  setAnimMeta,
  setSelected,
  setSelectedEffector,
  setSelectedRig,
  setShowBalance,
  subscribeAnim,
  withKey,
  type Balance,
  type Clip,
  type FootBalance,
  type IkResult,
  type JointsMeta,
  type PoseResult,
} from "./anim";

// ------------------------------------------------------------- clip math

/** A two-joint clip: a fold at t = 0, straight at t = 1, over 2 s. */
const key = (t: number, a: number, b: number, rootPitch = 0) => ({ t, joints: [a, b], rootPitch });
const clip = (over: Partial<Clip> = {}): Clip => ({
  version: 1,
  name: "fold",
  duration: 2,
  loop: false,
  keys: [key(0, 1, 10, -0.2), key(1, 3, 30, 0.2)],
  ...over,
});

describe("sampleClip", () => {
  it("holds the first key before it and the last key after it", () => {
    expect(sampleClip(clip(), -1)).toEqual({ joints: [1, 10], rootPitch: -0.2 });
    expect(sampleClip(clip(), 0)).toEqual({ joints: [1, 10], rootPitch: -0.2 });
    expect(sampleClip(clip(), 1)).toEqual({ joints: [3, 30], rootPitch: 0.2 });
    expect(sampleClip(clip(), 1.7)).toEqual({ joints: [3, 30], rootPitch: 0.2 });
  });
  it("is linear in joint space between keys — what the resampler does", () => {
    expect(sampleClip(clip(), 0.5)).toEqual({ joints: [2, 20], rootPitch: 0 });
    const q = sampleClip(clip(), 0.25);
    expect(q.joints[0]).toBeCloseTo(1.5);
    expect(q.rootPitch).toBeCloseTo(-0.1);
  });
  it("blends the last key back to the first across a loop's tail, and wraps", () => {
    const c = clip({ loop: true });
    // 1.5 s is halfway from the last key (t = 1) to the end (t = 2): half
    // way back to the first key.
    expect(sampleClip(c, 1.5)).toEqual({ joints: [2, 20], rootPitch: 0 });
    // 2.5 s wraps to 0.5 s — the middle of the first span.
    expect(sampleClip(c, 2.5)).toEqual({ joints: [2, 20], rootPitch: 0 });
    // Wrapping is modular from below as well.
    expect(sampleClip(c, -0.5)).toEqual({ joints: [2, 20], rootPitch: 0 });
    // Without `loop` nothing wraps.
    expect(sampleClip(clip(), 2.5)).toEqual({ joints: [3, 30], rootPitch: 0.2 });
  });
  it("gives a keyless clip the joint count it is told, the duck's by default", () => {
    expect(sampleClip(clip({ keys: [] }), 0).joints).toHaveLength(NUM_JOINTS);
    expect(sampleClip(clip({ keys: [] }), 0, 29).joints).toHaveLength(29);
  });
  it("hands back copies, never the key's own arrays", () => {
    const c = clip();
    const p = sampleClip(c, 0);
    p.joints[0] = 99;
    expect(c.keys[0].joints[0]).toBe(1);
  });
});

describe("keyAt / withKey", () => {
  it("finds the key under the playhead within the snap tolerance", () => {
    expect(keyAt(clip(), 1)).toBe(1);
    expect(keyAt(clip(), 1.004)).toBe(1);
    expect(keyAt(clip(), 1.02)).toBe(-1);
  });
  it("inserts in time order wherever it is asked", () => {
    let c = withKey(clip(), 1.5, { joints: [4, 40], rootPitch: 0 });
    c = withKey(c, 0.5, { joints: [2, 20], rootPitch: 0 });
    expect(c.keys.map((k) => k.t)).toEqual([0, 0.5, 1, 1.5]);
    expect(c.keys[1].joints).toEqual([2, 20]);
    expect(c.keys).not.toBe(clip().keys); // the input clip is untouched
  });
  it("replaces the key it lands on rather than stacking a second one", () => {
    const c = withKey(clip(), 1, { joints: [5, 50], rootPitch: 0.5 });
    expect(c.keys).toHaveLength(2);
    expect(c.keys[1]).toEqual(key(1, 5, 50, 0.5));
  });
  it("keeps the t = 0 anchor at exactly zero when the playhead is a hair past it", () => {
    // A scrub can park the playhead at 0.004 s; editing there edits the
    // anchor, and must not move it (the contract wants a key AT t = 0).
    const c = withKey(clip(), 0.004, { joints: [7, 70], rootPitch: 0 });
    expect(c.keys[0].t).toBe(0);
    expect(c.keys[0].joints).toEqual([7, 70]);
    expect(clipProblem(c)).toBeNull();
  });
  it("rounds a new key's time to the millisecond", () => {
    expect(withKey(clip(), 0.33333, { joints: [0, 0], rootPitch: 0 }).keys[1].t).toBe(0.333);
  });
  it("copies the pose into the key", () => {
    const pose = { joints: [8, 80], rootPitch: 0 };
    const c = withKey(clip(), 0.5, pose);
    pose.joints[0] = 0;
    expect(c.keys[1].joints[0]).toBe(8);
  });
});

describe("clipProblem", () => {
  it("passes a clip that meets the contract", () => {
    expect(clipProblem(clip())).toBeNull();
    expect(clipProblem(clip({ name: "A1 b.c_d-e" }))).toBeNull();
  });
  it("names each rule it enforces", () => {
    expect(clipProblem(clip({ keys: [] }))).toBe("a clip needs at least one key");
    expect(clipProblem(clip({ keys: [key(0.1, 0, 0)] }))).toBe("the first key must sit at t = 0");
    expect(clipProblem(clip({ keys: [key(0, 0, 0), key(1, 0, 0), key(1, 0, 0)] }))).toBe("key times must ascend");
    expect(clipProblem(clip({ keys: [key(0, 0, 0), key(1, 0, 0), key(0.5, 0, 0)] }))).toBe("key times must ascend");
    expect(clipProblem(clip({ duration: 0 }))).toBe("duration must be > 0");
    expect(clipProblem(clip({ duration: 0.5 }))).toBe("duration would cut off the last key");
    const badName = "name: letters, digits, space, . _ - (starting alphanumeric)";
    expect(clipProblem(clip({ name: "" }))).toBe(badName);
    expect(clipProblem(clip({ name: "-lead" }))).toBe(badName);
    expect(clipProblem(clip({ name: "a/b" }))).toBe(badName);
    expect(clipProblem(clip({ name: "x".repeat(65) }))).toBe(badName);
    expect(clipProblem(clip({ name: "x".repeat(64) }))).toBeNull();
  });
  it("checks the rules in the order the server rejects them", () => {
    // Several things wrong at once: the emptier structural fault is named.
    expect(clipProblem(clip({ keys: [], duration: 0, name: "" }))).toBe("a clip needs at least one key");
  });
});

// ------------------------------------------------------------- metadata

const joint = (index: number, name: string, group: string, body: number, def = 0) => ({
  index, name, group, min: -1.5, max: 1.5, default: def, body, bodyName: `b${body}`,
  axis: [0, 1, 0] as [number, number, number], pos: [0, 0, 0] as [number, number, number],
});

/** A three-joint make-believe body: a leg (two joints on bodies 2, 3) and
 *  a neck (body 4), with a foot effector on the shin and a head on the neck. */
const TOY_RAW = {
  joints: [joint(0, "left_hip_pitch", "left leg", 2, 0.1), joint(1, "left_knee", "left leg", 3, -0.2), joint(2, "neck_pitch", "head + neck", 4)],
  bodies: ["world", "trunk", "thigh", "shin", "neck", "foot"],
  trunkBody: 1,
  rootPitchRange: [-6.283185, 6.283185] as [number, number],
  rootPitchSign: "negative = lean back",
};
const TOY: JointsMeta = {
  ...normalizeMeta(TOY_RAW, "microduck"),
  effectors: [
    { id: "left_foot", label: "left foot", kind: "foot", body: 5, bodyName: "foot", point: [0, 0, 0], chain: [0, 1] },
    { id: "head", label: "head", kind: "head", body: 4, bodyName: "neck", point: [0, 0, 0], chain: [2] },
  ],
};

describe("normalizeMeta", () => {
  it("fills in what an older lab's /joints does not say, as the duck's", () => {
    const m = normalizeMeta(TOY_RAW, "microduck");
    expect(m.robot).toBe("microduck");
    expect(m.numJoints).toBe(3);
    expect(m.groups).toEqual(["left leg", "head + neck"]); // in joint order, once each
    expect(m.sizeScale).toBe(1);
    expect(m.effectors).toEqual([]);
    expect(m.rig).toBeUndefined();
  });
  it("keeps what the lab did say", () => {
    const m = normalizeMeta({ ...TOY_RAW, robot: "g1", numJoints: 29, groups: ["waist"], sizeScale: 2.89 }, "g1");
    expect(m.robot).toBe("g1");
    expect(m.numJoints).toBe(29);
    expect(m.groups).toEqual(["waist"]);
    expect(m.sizeScale).toBe(2.89);
  });
});

describe("newClip", () => {
  it("carries the metadata's robot and its DEFAULT_POSE, and the duck without metadata", () => {
    const c = newClip(TOY, "toy");
    expect(c.robot).toBe("microduck");
    expect(c.keys).toEqual([{ t: 0, joints: [0.1, -0.2, 0], rootPitch: 0 }]);
    expect(clipProblem(c)).toBeNull();
    const g1 = newClip({ ...TOY, robot: "g1" });
    expect(g1.robot).toBe("g1");
    expect(g1.name).toBe("untitled");
    const blind = newClip(null);
    expect(blind.robot).toBe("microduck");
    expect(blind.keys[0].joints).toHaveLength(NUM_JOINTS);
  });
  it("reads a clip that predates the field as the duck's", () => {
    expect(clipRobot(clip())).toBe("microduck");
    expect(clipRobot(clip({ robot: "g1" }))).toBe("g1");
  });
});

describe("effectorForBody", () => {
  it("picks the effector whose chain the body's joint is on, else the one on the body, else none", () => {
    expect(effectorForBody(TOY, 2)?.id).toBe("left_foot"); // thigh: hip is on the foot's chain
    expect(effectorForBody(TOY, 3)?.id).toBe("left_foot"); // shin
    expect(effectorForBody(TOY, 4)?.id).toBe("head"); // neck: on the head's chain
    expect(effectorForBody(TOY, 5)?.id).toBe("left_foot"); // the foot body itself, no joint of its own
    expect(effectorForBody(TOY, 1)).toBeNull(); // the trunk
  });
});

describe("previewOffset", () => {
  it("keeps the duck where it was and stands a bigger body proportionally further out", () => {
    expect(previewOffset(1)).toEqual([0, -1.05]);
    expect(previewOffset(2.89)[1]).toBeCloseTo(-1.05 * 2.89);
    expect(previewOffset(0.5)).toEqual([0, -1.05]); // never closer than the duck
  });
});

describe("selection", () => {
  afterEach(() => {
    animStore.selected = null;
    animStore.selectedRig = null;
    animStore.selectedEffector = null;
  });
  it("is one of joint, rig, effector at a time, and notifies only on change", () => {
    let seen = 0;
    const stop = subscribeAnim(() => seen++);
    try {
      setSelectedEffector("left_foot");
      expect(animStore.selectedEffector).toBe("left_foot");
      expect(seen).toBe(1);
      setSelectedEffector("left_foot");
      expect(seen).toBe(1);
      setSelected(2);
      expect(animStore.selectedEffector).toBeNull();
      expect(animStore.selected).toBe(2);
      setSelectedRig({ id: "squat", label: "squat", bodies: [2, 3] });
      expect(animStore.selected).toBeNull();
      setSelectedEffector("head");
      expect(animStore.selectedRig).toBeNull();
      expect(seen).toBe(4);
    } finally {
      stop();
    }
  });
  it("is cleared, with the per-frame data, when the metadata is for another body", () => {
    const before = animStore.meta;
    try {
      setAnimMeta(TOY);
      animStore.bodies = [[0, 0, 0, 1, 0, 0, 0]];
      animStore.effectors = { left_foot: [0, 0, 0] };
      setSelectedEffector("head");
      setAnimMeta({ ...TOY, robot: "g1", sizeScale: 2.89 });
      expect(animStore.robot).toBe("g1");
      expect(animStore.sizeScale).toBe(2.89);
      expect(animStore.bodies).toBeNull();
      expect(animStore.effectors).toBeNull();
      expect(animStore.selectedEffector).toBeNull();
      expect(animStore.jointForBody[1]).toBe(-1); // the trunk maps to ROOT_SEL
      expect(animStore.jointForBody[3]).toBe(1);
    } finally {
      animStore.meta = before;
      animStore.robot = "microduck";
      animStore.sizeScale = 1;
      animStore.bodies = null;
      animStore.effectors = null;
      animStore.jointForBody = [];
    }
  });
});

// ------------------------------------------------------------- streamer

/** A fetch that answers when the test says: each call is recorded with its
 *  URL and parsed body, and resolved by hand. */
function stubFetch() {
  const calls: { url: string; body: Record<string, unknown>; answer: (v: unknown) => void }[] = [];
  vi.stubGlobal(
    "fetch",
    (url: string, init?: RequestInit) =>
      new Promise<Response>((resolve) => {
        calls.push({
          url,
          body: init?.body ? JSON.parse(init.body as string) : {},
          answer: (v) =>
            resolve(new Response(JSON.stringify(v), { status: 200, headers: { "content-type": "application/json" } })),
        });
      })
  );
  return calls;
}
const settle = async () => {
  for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0));
};
const POSE = { joints: [0, 0], rootPitch: 0 };
const answer = (joints: number[], robot?: string): PoseResult => ({ bodies: [], joints, rootPitch: 0, ...(robot ? { robot: robot as "microduck" } : {}) });
const ikAnswer = (joints: number[]): IkResult => ({
  ...answer(joints),
  ik: { iterations: 3, converged: true, residual: { left_foot: 0 }, pins: ["right_foot"] },
});

describe("PoseStreamer", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("keeps one request in flight and sends only the newest of either kind afterwards", async () => {
    const calls = stubFetch();
    const seen: (PoseResult | IkResult)[] = [];
    const s = new PoseStreamer((r) => seen.push(r));
    s.request({ joints: [1, 1], rootPitch: 0 });
    s.request({ joints: [2, 2], rootPitch: 0 }); // superseded before it is sent
    s.requestIk({ joints: [3, 3], rootPitch: 0 }, { left_foot: { pos: [0.1, 0, 0] } });
    expect(calls).toHaveLength(1);
    expect(calls[0].url).toMatch(/\/pose$/);
    calls[0].answer(answer([1, 1]));
    await settle();
    // The trailing send is the IK, not the pose it replaced.
    expect(calls).toHaveLength(2);
    expect(calls[1].url).toMatch(/\/ik$/);
    expect(calls[1].body).toEqual({ joints: [3, 3], rootPitch: 0, targets: { left_foot: { pos: [0.1, 0, 0] } } });
    calls[1].answer(ikAnswer([3.5, 3.5]));
    await settle();
    expect(seen).toHaveLength(2);
    expect(isIkResult(seen[0])).toBe(false);
    expect(isIkResult(seen[1])).toBe(true);
    expect(seen[1].joints).toEqual([3.5, 3.5]); // the solved joints, for the panel to adopt
    s.close();
  });

  it("names the robot in the query, and pins when asked", async () => {
    const calls = stubFetch();
    const s = new PoseStreamer(() => {});
    s.setRobot("g1");
    s.requestIk(POSE, { left_hand: { pos: [0, 0, 1] } }, ["right_foot"]);
    expect(calls[0].url).toMatch(/\/ik\?robot=g1$/);
    expect(calls[0].body.pins).toEqual(["right_foot"]);
    calls[0].answer(answer([0, 0], "g1"));
    await settle();
    s.request(POSE);
    expect(calls[1].url).toMatch(/\/pose\?robot=g1$/);
    s.close();
  });

  it("drops an answer for a body it has been switched away from", async () => {
    const calls = stubFetch();
    const seen: PoseResult[] = [];
    const errors: string[] = [];
    const s = new PoseStreamer((r) => seen.push(r), (e) => errors.push(e));
    s.request(POSE); // the duck's, in flight
    s.setRobot("g1");
    calls[0].answer(answer([0, 0], "microduck"));
    await settle();
    expect(seen).toHaveLength(0);
    expect(errors).toHaveLength(0);
    s.close();
  });

  it("reports an error and still sends the trailing request", async () => {
    vi.stubGlobal("fetch", (() => {
      let n = 0;
      return () =>
        Promise.resolve(
          n++ === 0
            ? new Response(JSON.stringify({ detail: "joints must be 2 numbers" }), { status: 422 })
            : new Response(JSON.stringify(answer([0, 0])), { status: 200 })
        );
    })());
    const seen: PoseResult[] = [];
    const errors: string[] = [];
    const s = new PoseStreamer((r) => seen.push(r), (e) => errors.push(e));
    s.request(POSE);
    s.request(POSE);
    await settle();
    expect(errors).toEqual(["joints must be 2 numbers"]);
    expect(seen).toHaveLength(1);
    s.close();
  });
});

// ------------------------------------------------------------- balance

const SOLE = [
  [-0.02, -0.02],
  [0.02, -0.02],
  [0.02, 0.02],
  [-0.02, 0.02],
];
const foot = (marginMm: number, grounded = true): FootBalance => ({ marginMm, grounded, outline: SOLE });
/** `support` defaults to what the server would send: the grounded feet, and
 *  the stance margin the pose's own story implies — well inside for the
 *  square stance, the stance sole's own margin once the other foot lifts. */
const balance = (
  left: FootBalance,
  right: FootBalance,
  over: Balance["over"] = null,
  supportMm?: number
): Balance => {
  const feet = { left, right };
  const down = (["left", "right"] as const).filter((s) => feet[s].grounded);
  const marginMm =
    supportMm ?? (down.length === 1 ? feet[down[0]].marginMm : Math.max(left.marginMm, right.marginMm) + 40);
  return { com: [0, 0, 0.14], feet, over, support: { feet: down, marginMm, outline: SOLE } };
};

/** Standing square: the CoM on the midline, the same distance outside both
 *  soles, and 16 mm inside the stance (what the model measures). */
const SQUARE = balance(foot(-25.0), foot(-25.0), null, 16.3);
/** Inside the left sole, the right foot lifted. */
const ON_LEFT = balance(foot(3.2), foot(-52.3, false), "left");
/** Leaning left, still outside both feet, both down, inside the stance. */
const LEANING = balance(foot(-4.2), foot(-33.1), null, 9.8);
/** Pitched forward past the toes: both feet down, outside the stance. */
const TOPPLING = balance(foot(-52.2), foot(-52.2), null, -40.5);

describe("nearestFoot", () => {
  it("names the foot with more room, standing or not", () => {
    expect(nearestFoot(ON_LEFT)).toEqual({ side: "left", marginMm: 3.2 });
    expect(nearestFoot(LEANING)).toEqual({ side: "left", marginMm: -4.2 });
    expect(nearestFoot(balance(foot(-33.1), foot(-4.2)))).toEqual({ side: "right", marginMm: -4.2 });
  });
  it("will not pick a foot that is in the air", () => {
    // The CoM is nearer the lifted right foot, but it cannot be stood on.
    expect(nearestFoot(balance(foot(-30.0), foot(-2.0, false)))).toEqual({ side: "left", marginMm: -30.0 });
  });
});

describe("balanceLabel", () => {
  it("says which sole the CoM is inside, and by how much", () => {
    expect(balanceLabel(ON_LEFT)).toBe("3.2 mm inside the left sole");
  });
  it("says the square stance stands, and how far a sole is", () => {
    expect(balanceLabel(SQUARE)).toBe("16.3 mm inside the stance, 25.0 mm short of a sole");
  });
  it("names the near foot once there is a real lean", () => {
    expect(balanceLabel(LEANING)).toBe("9.8 mm inside the stance, 4.2 mm short of the left sole");
  });
  it("says outside once the CoM has left the stance", () => {
    expect(balanceLabel(TOPPLING)).toBe("52.2 mm outside both soles");
  });
  it("says when the other foot is off the floor", () => {
    expect(balanceLabel(balance(foot(-0.5), foot(-52.0, false)))).toBe(
      "0.5 mm outside the left sole (right foot in the air)"
    );
  });
});

describe("balanceState / balanceColor", () => {
  it("is green over a sole, blue inside the stance, amber outside", () => {
    expect(balanceState(ON_LEFT)).toBe("sole");
    expect(balanceColor(ON_LEFT)).toBe(BALANCE_IN);
    expect(balanceState(SQUARE)).toBe("stance");
    expect(balanceColor(SQUARE)).toBe(BALANCE_STANCE);
    expect(balanceState(TOPPLING)).toBe("out");
    expect(balanceColor(TOPPLING)).toBe(BALANCE_OUT);
    expect(balanceColor(null)).toBe(BALANCE_OUT);
  });
  it("cannot be inside the stance while over a lifted foot's sole", () => {
    // The server's `over` is grounded-only; the stance margin follows the
    // stance foot, so a lifted foot the CoM lines up with reads as out.
    const b = balance(foot(-30.0), foot(3.0, false));
    expect(b.support.marginMm).toBe(-30.0);
    expect(balanceState(b)).toBe("out");
  });
});

describe("sameBalance", () => {
  it("is equal when the readout would be, whatever the CoM did", () => {
    expect(sameBalance(SQUARE, { ...SQUARE, com: [0.01, 0, 0.13] })).toBe(true);
    expect(sameBalance(SQUARE, balance(foot(-25.0), foot(-24.9)))).toBe(false);
    expect(sameBalance(SQUARE, balance(foot(-25.0), foot(-25.0, false)))).toBe(false);
    expect(sameBalance(SQUARE, balance(foot(-25.0), foot(-25.0), null, 16.2))).toBe(false);
    // The outlines move with every slider tick; the readout does not print them.
    const moved = balance(foot(-25.0), foot(-25.0), null, 16.3);
    moved.feet.left.outline = SOLE.map(([x, y]) => [x + 0.01, y]);
    expect(sameBalance(SQUARE, moved)).toBe(true);
    expect(sameBalance(null, SQUARE)).toBe(false);
    expect(sameBalance(null, null)).toBe(true);
  });
});

describe("setShowBalance", () => {
  it("notifies React, unlike the per-frame balance data itself", () => {
    let seen = 0;
    const stop = subscribeAnim(() => seen++);
    const v0 = animVersion();
    try {
      setShowBalance(true);
      expect(animStore.showBalance).toBe(true);
      expect(seen).toBe(1);
      setShowBalance(true); // no change, no render
      expect(seen).toBe(1);
      setShowBalance(false);
      expect(seen).toBe(2);
      expect(animVersion()).toBe(v0 + 2);
    } finally {
      stop();
      animStore.showBalance = false;
    }
  });
});
