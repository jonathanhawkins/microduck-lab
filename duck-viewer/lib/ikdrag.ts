// The arithmetic behind dragging an IK handle: where the pointer ray meets
// the plane the effector is being moved in, and what target that asks for.
// Pure tuples, no three.js, so it is pinned by tests; the pointer plumbing
// (ray from the camera, group-local conversion) lives in PoseDuck.tsx.
//
// A handle is dragged in the VIEW PLANE: the plane through the effector's
// current position whose normal is the camera's view direction. The pointer
// then moves the point at the same depth it already sits at, which is the
// one depth a 2D pointer cannot express — everything else it can.

export type V3 = [number, number, number];

const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];

/** Below this the ray runs along the plane and there is no usable hit. */
const MIN_COS = 1e-6;

/** Where the ray from `origin` along `dir` crosses the plane through
 *  `anchor` with normal `viewDir`. Null when the ray is parallel to the
 *  plane or the plane is behind the ray's origin (a hit behind the camera
 *  would fling the handle across the room). `dir` and `viewDir` need not be
 *  unit length. */
export function dragPointOnViewPlane(origin: V3, dir: V3, anchor: V3, viewDir: V3): V3 | null {
  const denom = dot(dir, viewDir);
  if (Math.abs(denom) < MIN_COS) return null;
  const t = dot(sub(anchor, origin), viewDir) / denom;
  if (t < 0) return null;
  return add(origin, mul(dir, t));
}

/** `v` with its component along `normal` removed. */
export function projectOntoPlane(v: V3, normal: V3): V3 {
  const n2 = dot(normal, normal);
  if (n2 < MIN_COS) return v;
  return sub(v, mul(normal, dot(v, normal) / n2));
}

/** One pointer move of an IK drag → the target to ask the solver for.
 *
 *  At full gain the target is the hit plus the offset the grab started
 *  with (`grabOffset` = effector − first hit), so the effector keeps its
 *  relationship to the pointer however far the solver lags: an unreachable
 *  target stalls the limb, and when the pointer comes back the limb catches
 *  up with no jump. At a reduced gain (shift = fine) the target inches from
 *  the effector's CURRENT position by that fraction of the pointer's
 *  in-plane travel since the last hit, and the offset is re-anchored so the
 *  next full-gain move continues from there instead of snapping back. */
export function ikDragStep(
  current: V3,
  hit: V3,
  lastHit: V3 | null,
  grabOffset: V3,
  viewDir: V3,
  gain: number
): { target: V3; grabOffset: V3 } {
  if (gain >= 1) return { target: add(hit, grabOffset), grabOffset };
  const travel = projectOntoPlane(sub(hit, lastHit ?? hit), viewDir);
  const target = add(current, mul(travel, gain));
  return { target, grabOffset: sub(target, hit) };
}
