"use client";

// A panel the user can put where they like. `pos` is null until they drag
// it — the panel sits at its designed dock — and persists in localStorage
// once they do, clamped back into view on every resize. Double-click the
// handle to return to the dock.
//
// The hook owns only the pointer plumbing; the box's own positioning code
// reads `pos` (React state, for a panel laid out by render) or `posRef`
// (for one laid out from a requestAnimationFrame loop, like the head-camera
// inset) and falls back to its dock when that is null.

import { useCallback, useEffect, useRef, useState } from "react";
import { clampPos, moved, type Pos } from "@/lib/drag";
import { loadJSON, saveJSON } from "@/lib/persist";

export function useDrag(key: string, box: React.RefObject<HTMLElement | null>, minY: number, pad = 10) {
  const [pos, setPos] = useState<Pos | null>(() => loadJSON<Pos | null>(key, null));
  const posRef = useRef(pos);
  posRef.current = pos;
  useEffect(() => saveJSON(key, pos), [key, pos]);

  const clamp = useCallback(
    (p: Pos) => {
      const el = box.current;
      const size = el ? { w: el.offsetWidth, h: el.offsetHeight } : { w: 0, h: 0 };
      return clampPos(p, size, { w: window.innerWidth, h: window.innerHeight }, minY, pad);
    },
    [box, minY, pad],
  );

  // A window that shrinks must not strand a panel off-screen.
  useEffect(() => {
    const onResize = () => setPos((p) => (p ? clamp(p) : p));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [clamp]);

  /** Put on the handle. Returns true from pointerup handling when the gesture was a drag, so a click-to-toggle handle can tell the two apart. */
  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLElement>) => {
      // Only the primary button, and never the second press of a
      // double-click: that one belongs to the re-dock, and any pointer
      // travel riding on it must not turn into a drag.
      if (e.button !== 0 || e.detail > 1) return;
      const el = box.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      const origin = { x: r.left, y: r.top };
      const start = { x: e.clientX, y: e.clientY };
      let dragged = false;
      const handle = e.currentTarget;
      handle.setPointerCapture(e.pointerId);
      e.stopPropagation();   // the room below assigns policies on pointerdown
      const onMove = (ev: PointerEvent) => {
        const here = { x: ev.clientX, y: ev.clientY };
        if (!dragged && !moved(start, here)) return;
        dragged = true;
        setPos(clamp({ x: origin.x + here.x - start.x, y: origin.y + here.y - start.y }));
      };
      const onUp = () => {
        handle.releasePointerCapture(e.pointerId);
        handle.removeEventListener("pointermove", onMove);
        handle.removeEventListener("pointerup", onUp);
        handle.removeEventListener("pointercancel", onUp);
        // A click on a handle that also toggles must not fire after a drag.
        if (dragged) handle.dataset.dragged = "1";
        else delete handle.dataset.dragged;
      };
      handle.addEventListener("pointermove", onMove);
      handle.addEventListener("pointerup", onUp);
      handle.addEventListener("pointercancel", onUp);
    },
    [box, clamp],
  );

  const reset = useCallback(() => setPos(null), []);
  return { pos, posRef, onPointerDown, reset };
}

/** For a handle that is also a button: true (and cleared) when the last gesture on it was a drag. */
export function consumeDragged(el: HTMLElement): boolean {
  const was = el.dataset.dragged === "1";
  delete el.dataset.dragged;
  return was;
}

export const HANDLE: React.CSSProperties = { cursor: "grab", touchAction: "none", userSelect: "none" };
