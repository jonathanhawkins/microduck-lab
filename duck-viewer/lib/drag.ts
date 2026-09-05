// The arithmetic behind a draggable panel: where a box may sit, and what
// counts as a drag rather than a click. Pure, so it is pinned by tests; the
// pointer plumbing lives in components/useDrag.ts.

export interface Pos {
  x: number;
  y: number;
}

/** Below this many pixels of travel a press-and-release is a click, not a drag. */
export const DRAG_SLOP = 4;

export function moved(from: Pos, to: Pos, slop = DRAG_SLOP): boolean {
  return Math.abs(to.x - from.x) > slop || Math.abs(to.y - from.y) > slop;
}

/**
 * Keep a box of `size` inside the viewport: never off the right/bottom edge,
 * never above `minY` (the top bar owns that strip) or left of `pad`. When
 * the viewport is smaller than the box the box hugs the top-left, so a
 * panel can never be dragged, or resized, out of reach.
 */
export function clampPos(p: Pos, size: { w: number; h: number }, view: { w: number; h: number }, minY = 0, pad = 0): Pos {
  const maxX = Math.max(pad, view.w - size.w - pad);
  const maxY = Math.max(minY, view.h - size.h - pad);
  return {
    x: Math.round(Math.min(maxX, Math.max(pad, p.x))),
    y: Math.round(Math.min(maxY, Math.max(minY, p.y))),
  };
}
