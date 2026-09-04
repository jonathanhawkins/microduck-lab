"use client";

// The camera flight model, shared by the lab page and /sim so both pages fly
// the same way. Keys land in the lib/camera held-set; this integrates them ×
// dt every frame — smooth game-editor flow, not per-keypress steps.
//
// The move that matters: A/D and Q/E truck the camera AND the orbit target
// together, so the view SLIDES through the room instead of pivoting around
// one fixed point. W/S dolly, arrows orbit. Rates scale with distance, so the
// feel is the same nose-close and across the room.

import { useMemo } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { heldMotions, takeReset, takeTruckImpulse } from "@/lib/camera";

export type Vec3Tuple = [number, number, number];

// Structural slice of drei/three-stdlib OrbitControls — enough to drive it
// without importing its concrete class type.
export interface ControlsLike {
  enabled: boolean;
  target: THREE.Vector3;
  addEventListener: (type: "end", cb: () => void) => void;
  removeEventListener: (type: "end", cb: () => void) => void;
  update?: () => void;
}

export interface CameraHome {
  p: readonly [number, number, number];
  t: readonly [number, number, number];
}

export interface CameraKeysProps {
  /** Pose Shift+R flies back to. */
  home: CameraHome;
  /** Dolly clamp — the lab floor and a /sim room want different rooms to move in. */
  minDist?: number;
  maxDist?: number;
  /** Read every frame: while true something else owns the camera (a capture),
   *  so queued resets are dropped and held motions ignored. A callback, not a
   *  prop value, because the owner can change without a re-render. */
  paused?: () => boolean;
}

/** Inside-the-Canvas helper: integrates held camera motions × dt every frame. */
export default function CameraKeys({ home, minDist = 0.25, maxDist = 8, paused }: CameraKeysProps) {
  const controls = useThree((s) => s.controls) as unknown as ControlsLike | null;
  const camera = useThree((s) => s.camera);
  const sph = useMemo(() => new THREE.Spherical(), []);
  const offset = useMemo(() => new THREE.Vector3(), []);
  const right = useMemo(() => new THREE.Vector3(), []);
  useFrame((_, dtRaw) => {
    // Drain the swipe impulse EVERY frame (even when unused/discarded) so a
    // burst can never pool up and teleport the view later — idle stays idle.
    const swipePx = takeTruckImpulse();
    if (!controls) return;
    if (paused?.()) {
      takeReset();
      return;
    }
    if (takeReset()) {
      camera.position.set(...(home.p as Vec3Tuple));
      controls.target.set(...(home.t as Vec3Tuple));
      camera.lookAt(controls.target);
      controls.update?.();
      return;
    }
    const held = heldMotions();
    if (!held.size && swipePx === 0) return;
    const dt = Math.min(dtRaw, 0.05); // tab-stall guard: no teleport frames

    offset.copy(camera.position).sub(controls.target);
    sph.setFromVector3(offset);

    // Lateral/vertical truck moves camera AND target (the view slides).
    const truckSpeed = 0.9 * sph.radius * dt;
    if (held.has("truckLeft") || held.has("truckRight")) {
      right.set(1, 0, 0).applyQuaternion(camera.quaternion);
      right.y = 0; // keep trucking parallel to the floor
      right.normalize().multiplyScalar(held.has("truckRight") ? truckSpeed : -truckSpeed);
      camera.position.add(right);
      controls.target.add(right);
    }
    // Two-finger trackpad swipe → the same lateral truck, impulse-scaled.
    // Natural scrolling reports fingers-moving-LEFT as +deltaX, and that
    // gesture should slide the VIEW left (scene drifts right on screen) —
    // hence the negation. Distance scaling keeps the feel constant near and
    // far; the cap stops a momentum fling from delivering a teleport frame.
    if (swipePx !== 0) {
      const cap = 0.5 * sph.radius;
      const step = Math.max(-cap, Math.min(cap, -swipePx * 0.0015 * sph.radius));
      right.set(1, 0, 0).applyQuaternion(camera.quaternion);
      right.y = 0; // floor-parallel, like A/D
      right.normalize().multiplyScalar(step);
      camera.position.add(right);
      controls.target.add(right);
    }
    if (held.has("up") || held.has("down")) {
      const dy = (held.has("up") ? 1 : -1) * 0.6 * sph.radius * dt;
      camera.position.y += dy;
      controls.target.y = Math.max(0.02, controls.target.y + dy);
    }

    // Orbit/dolly work on the target-relative spherical frame.
    offset.copy(camera.position).sub(controls.target);
    sph.setFromVector3(offset);
    if (held.has("orbitLeft")) sph.theta += 1.7 * dt;
    if (held.has("orbitRight")) sph.theta -= 1.7 * dt;
    if (held.has("dollyIn")) sph.radius *= Math.exp(-1.5 * dt);
    if (held.has("dollyOut")) sph.radius *= Math.exp(1.5 * dt);
    sph.radius = Math.min(maxDist, Math.max(minDist, sph.radius));
    sph.phi = Math.min(1.53, Math.max(0.1, sph.phi));
    offset.setFromSpherical(sph);
    camera.position.copy(controls.target).add(offset);

    camera.lookAt(controls.target);
    controls.update?.();
  });
  return null;
}
