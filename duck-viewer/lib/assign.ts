// Shared mutable state for policy assignment, bridging the DOM PolicyPanel
// (which owns the pointer gestures) and the r3f helper inside the Canvas
// (which projects duck positions to screen space). Plain module-level object
// mutated outside React — consumers read it per-frame, so no events or
// re-renders are involved (same philosophy as LabClient.frame).

/** One duck's projected screen position, refreshed every rendered frame. */
export interface AssignTarget {
  id: string;
  x: number; // client (viewport) px
  y: number;
  visible: boolean; // false when behind the camera
}

export interface AssignDrag {
  /** "drag" while a chip is held, "armed" after a chip click, null when idle. */
  mode: "drag" | "armed" | null;
  policyId: string | null;
  policyLabel: string | null;
  /** True while the chain-level "whole trick" chip is in flight — the assign
   *  or spawn it lands ships showcase: true (full-arc rehearsal spawns). */
  showcase: boolean;
  /** Latest pointer position in client (viewport) pixels. */
  px: number;
  py: number;
  /** Duck id nearest the pointer (within range) while a drag/arm is active;
   *  written per-frame by the Canvas-side AssignTargets helper. Drives the
   *  highlight ring — drop handlers compute their own nearestDuck() from the
   *  event position instead, so a drop never races the render loop. */
  hoverDuck: string | null;
  /** Projected duck positions, written per-frame by AssignTargets. */
  targets: AssignTarget[];
}

export const assignDrag: AssignDrag = {
  mode: null,
  policyId: null,
  policyLabel: null,
  showcase: false,
  px: 0,
  py: 0,
  hoverDuck: null,
  targets: [],
};

export function clearAssignDrag() {
  assignDrag.mode = null;
  assignDrag.policyId = null;
  assignDrag.policyLabel = null;
  assignDrag.showcase = false;
  assignDrag.hoverDuck = null;
}

/** Screen-space radius (px) within which a duck counts as the drop target. */
export const ASSIGN_RADIUS_PX = 80;

/** True when (x, y) hits the WebGL canvas itself — i.e. the open stage, not an
 *  overlay panel or button. Used to tell "drop on empty floor" (spawn) apart
 *  from "drop on UI" (cancel). The drag ghost and the ducks' DOM labels are
 *  pointer-transparent, so elementFromPoint sees straight through them. */
export function isCanvasAt(x: number, y: number): boolean {
  return document.elementFromPoint(x, y) instanceof HTMLCanvasElement;
}

/** Closest on-screen duck to (px, py), or null if none within range. */
export function nearestDuck(px: number, py: number): string | null {
  let best: string | null = null;
  let bestD = ASSIGN_RADIUS_PX;
  for (const t of assignDrag.targets) {
    if (!t.visible) continue;
    const d = Math.hypot(t.x - px, t.y - py);
    if (d < bestD) {
      bestD = d;
      best = t.id;
    }
  }
  return best;
}
