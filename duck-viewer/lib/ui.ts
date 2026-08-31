// Tiny cross-panel UI store (assign.ts pattern): panels are independent
// fixed-position siblings, but the TeachPanel's height budget depends on
// whether the PolicyPanel above it is open — localStorage alone isn't
// reactive, so the open flag is mirrored here for live layout coupling.

import { useSyncExternalStore } from "react";

let policyOpen = true;
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
    () => true
  );
}
