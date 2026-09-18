// The ONE robot the user is working with (ui.ts / select.ts pattern).
//
// The 🧠 palette, the 🎓 teach panel and the 🎬 animate panel each used to
// keep their own persisted answer to "which robot?" (policyRobot / teachRobot
// / animRobot), so the palette could sit on the duck while teach was about to
// train a G1 — three questions, three answers. They all read this one now,
// and selecting a robot on the stage sets it too (Viewer.tsx).

import { useSyncExternalStore } from "react";
import { loadJSON, saveJSON } from "./persist";
import { robotEmoji } from "./robots";

const KEY = "activeRobot";
const FALLBACK = "microduck";

// What the panels used to persist, most deliberate first: teach's choice
// starts a training run, the palette's only filters a list ("all" there is
// not a robot), and animate's is kept by animate for its own clip.
const LEGACY_KEYS = ["teachRobot", "policyRobot"];

/** The robot a returning user lands on: the saved choice, else whatever the
 *  old per-panel keys said, else the duck. `read` is injected for the test. */
export function initialActiveRobot(read: (key: string) => unknown): string {
  for (const key of [KEY, ...LEGACY_KEYS]) {
    const v = read(key);
    if (typeof v === "string" && v && v !== "all") return v;
  }
  return FALLBACK;
}

/** One label for a robot on every switch. The panels each spelled it their
 *  own way — "🤖 g1", "🤖 G1", "🤖 Unitree G1" — for the same button.
 *
 *  The emoji comes from `lib/robots.robotEmoji`, which is the one place that
 *  knows a face per body (🦆 duck, 🤖 G1, 🛸 MARS, and a kind's emoji for a
 *  body this build has never heard of). This function still owns the LABEL —
 *  two definitions of that is the bug it was written to fix. */
export function robotChipLabel(r: {
  id: string;
  noun?: string;
  title?: string;
  label?: string;
  kind?: string;
}): string {
  return `${robotEmoji(r.id, r.kind)} ${r.noun || r.title || r.label || r.id}`;
}

/** The robot to work with given what the lab lists: the active one when the
 *  lab has it, else the lab's first — a saved choice can outlive its robot
 *  (an older lab, assets deleted). Never writes; the choice stays the user's. */
export function resolveRobot<T extends { id: string }>(robots: T[], active: string): T | undefined {
  return robots.find((r) => r.id === active) ?? robots[0];
}

let active: string | null = null;
const listeners = new Set<() => void>();

/** Read outside React (event handlers, effects) — no subscription. */
export function getActiveRobot(): string {
  // Lazy, not at module scope: this file is imported during SSR, where there
  // is no storage, and Fast Refresh re-evaluates it — re-reading the saved
  // value is what keeps an edit from snapping the switch back to the duck.
  if (active == null) active = initialActiveRobot((k) => loadJSON<unknown>(k, null));
  return active;
}

export function setActiveRobot(id: string) {
  if (!id || id === getActiveRobot()) return;
  active = id;
  saveJSON(KEY, id);
  listeners.forEach((l) => l());
}

export function useActiveRobot(): string {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    getActiveRobot,
    () => FALLBACK
  );
}
