// Tiny cross-panel UI store (assign.ts pattern): panels are independent
// fixed-position siblings, but the TeachPanel's height budget depends on
// whether the PolicyPanel above it is open — localStorage alone isn't
// reactive, so the open flag is mirrored here for live layout coupling.

import { useSyncExternalStore } from "react";

// Mirrors the PolicyPanel's persisted default (collapsed) until it mounts
// and publishes the real value.
let policyOpen = false;
const listeners = new Set<() => void>();

export function setPolicyOpen(v: boolean) {
  if (v === policyOpen) return;
  policyOpen = v;
  listeners.forEach((l) => l());
}

export function usePolicyOpen(): boolean {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => policyOpen,
    () => false
  );
}

// Right edge of the top-left HUD panel in px (0 = unmounted). Published by
// Hud via a ResizeObserver; the top-center 🎥/📷 capture panel reads it to
// slide right when a wide HUD (long duck names) would otherwise sit under it.
let hudRight = 0;

export function setHudRight(px: number) {
  if (px === hudRight) return;
  hudRight = px;
  listeners.forEach((l) => l());
}

export function useHudRight(): number {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => hudRight,
    () => 0
  );
}

// Measured height of the bottom-right teach panel in px — its collapsed pill
// counts, 0 = not measured yet. Published by TeachPanel via a ResizeObserver
// so the policy list above it can claim whatever vertical space teach is not
// using (a collapsed or short teach panel used to leave a dead gap while the
// chip list scrolled).
//
// Deliberately ONE-WAY: teach sizes itself off the PolicyPanel's NOMINAL cap
// (the min(40vh, 380px) constant), never off the policy panel's measured
// height. Feeding a measured policy height back into teach would close the
// loop and let the two panels pump each other on every frame.
let teachHeight = 0;

export function setTeachHeight(px: number) {
  if (px === teachHeight) return;
  teachHeight = px;
  listeners.forEach((l) => l());
}

export function useTeachHeight(): number {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => teachHeight,
    () => 0
  );
}

// Floating duck name labels on/off. Toggled from the HUD (which owns the
// localStorage persistence); read PER-FRAME by every Duck inside the Canvas
// (assign.ts philosophy) — a useSyncExternalStore subscription inside the
// r3f tree flushed unreliably when the write came from the DOM tree, so the
// labels only reappeared on the next unrelated roster re-render.
let duckLabels = true;

export function setDuckLabels(v: boolean) {
  duckLabels = v;
}

export function getDuckLabels(): boolean {
  return duckLabels;
}

// Modal gate. Two window/capture keydown listeners cannot suppress each other
// (stopPropagation only affects OTHER elements; stopImmediatePropagation only
// affects listeners registered LATER), and the scene's listener is registered
// first — so a dialog can only be protected by the scene AGREEING to stand
// down. Dialogs increment on mount and decrement on unmount; a counter, not a
// boolean, so two dialogs open at once can't un-gate each other.
let modalDepth = 0;

export function pushModal(): () => void {
  modalDepth += 1;
  let released = false;
  return () => {
    if (released) return;
    released = true;
    modalDepth = Math.max(0, modalDepth - 1);
  };
}

/** Read per-event inside key handlers — not a hook, so no re-render coupling. */
export function modalIsOpen(): boolean {
  return modalDepth > 0;
}
