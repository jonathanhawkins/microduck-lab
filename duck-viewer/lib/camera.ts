// Keyboard → camera motion (game-editor style). Keys set/clear entries in a
// held-set on keydown/keyup; the CameraKeys helper inside the Canvas applies
// velocity × dt every frame while a key is held — smooth continuous motion,
// not per-keypress steps (the first version stepped like a scroll wheel;
// Maya/UE-trained hands expect flow). The view reset (Shift+R) is the one
// discrete action.

export type CameraMotion =
  | "truckLeft"   // A — slide view left (mouse already owns orbit)
  | "truckRight"  // D
  | "dollyIn"     // W / ↑
  | "dollyOut"    // S / ↓
  | "orbitLeft"   // ←
  | "orbitRight"  // →
  | "up"          // Q — vertical truck
  | "down";       // E

const KEY_MOTIONS: Record<string, CameraMotion> = {
  a: "truckLeft",
  d: "truckRight",
  w: "dollyIn",
  s: "dollyOut",
  arrowup: "dollyIn",
  arrowdown: "dollyOut",
  arrowleft: "orbitLeft",
  arrowright: "orbitRight",
  q: "up",
  e: "down",
};

const held = new Set<CameraMotion>();
let resetPending = false;

/** keydown: begin the motion (or queue the view reset). True if it was a
 *  camera key. Plain R now restarts the SIM (Viewer owns that), so the view
 *  reset moved to Shift+R — passed as a flag rather than sniffed from the key
 *  case, which Caps Lock would lie about. */
export function cameraKeyDown(key: string, shift = false): boolean {
  const k = key.toLowerCase();
  if (k === "r") {
    if (!shift) return false; // plain R isn't ours — leave it to the caller
    resetPending = true;
    return true;
  }
  const m = KEY_MOTIONS[k];
  if (!m) return false;
  held.add(m);
  return true;
}

/** keyup: end the motion. Safe to call for any key. */
export function cameraKeyUp(key: string): void {
  const m = KEY_MOTIONS[key.toLowerCase()];
  if (m) held.delete(m);
}

/** Stuck-key guard: clear everything (window blur, page hide). */
export function cameraKeysClear(): void {
  held.clear();
}

export function heldMotions(): ReadonlySet<CameraMotion> {
  return held;
}

export function takeReset(): boolean {
  const r = resetPending;
  resetPending = false;
  return r;
}

// --- trackpad swipe impulses -------------------------------------------------
// Two-finger horizontal swipes arrive as a smooth high-rate wheel-event
// stream, not key holds — so they accumulate PIXEL IMPULSES here rather than
// entries in the held-set (held state would stick between events). CameraKeys
// drains the total once per frame and applies it as a lateral truck.

let truckImpulsePx = 0;

/** Accumulate a horizontal swipe impulse (wheel deltaX, in pixels). */
export function truckImpulse(dxPixels: number): void {
  if (Number.isFinite(dxPixels)) truckImpulsePx += dxPixels;
}

/** Per-frame drain: return the accumulated impulse and clear it. */
export function takeTruckImpulse(): number {
  const px = truckImpulsePx;
  truckImpulsePx = 0;
  return px;
}
