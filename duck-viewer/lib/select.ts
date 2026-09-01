// Duck selection store (ui.ts pattern): click a duck on the stage (or its HUD
// row) to select it; Delete/Backspace removes it. React consumers (the HUD
// row highlight) subscribe via useSyncExternalStore; per-frame consumers (the
// selection ring in Duck.tsx) read getSelectedDuck() directly — same
// split-audience philosophy as assign.ts.

import { useSyncExternalStore } from "react";

let selected: string | null = null;
const listeners = new Set<() => void>();

/** Frame-loop read — no subscription, no re-render. */
export function getSelectedDuck(): string | null {
  return selected;
}

export function setSelectedDuck(id: string | null) {
  if (id === selected) return;
  selected = id;
  listeners.forEach((l) => l());
}

export function useSelectedDuck(): string | null {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => selected,
    () => null
  );
}
