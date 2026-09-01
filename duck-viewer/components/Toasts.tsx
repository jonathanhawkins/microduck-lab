"use client";

// Bottom-center toast lines: one-shot "events" drained from the lab stream
// (LabClient buffers them — each event lives in a single 25 Hz frame) plus
// local lines pushed via pushToast(). Each toast shows for ~3 s; event lines
// are deduped so a re-broadcast never shows twice.

import { useCallback, useEffect, useRef, useState } from "react";
import type { LabClient } from "@/lib/lab";

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";

// Module-level hook so non-React code (e.g. the policy drag handlers) can
// raise a toast; wired to the mounted Toasts instance.
let pushImpl: ((text: string) => void) | null = null;
export function pushToast(text: string) {
  pushImpl?.(text);
}

interface Toast {
  key: number;
  text: string;
}

export function Toasts({
  clientRef,
}: {
  clientRef: React.MutableRefObject<LabClient | null>;
}) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const seenEvents = useRef<Set<string>>(new Set());
  const nextKey = useRef(0);

  const add = useCallback((text: string) => {
    const key = nextKey.current++;
    setToasts((t) => [...t.slice(-4), { key, text }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.key !== key)), 3000);
  }, []);

  useEffect(() => {
    pushImpl = add;
    return () => {
      pushImpl = null;
    };
  }, [add]);

  useEffect(() => {
    const id = setInterval(() => {
      for (const ev of clientRef.current?.takeEvents() ?? []) {
        if (seenEvents.current.has(ev)) continue;
        seenEvents.current.add(ev);
        add(ev);
      }
    }, 250);
    return () => clearInterval(id);
  }, [clientRef, add]);

  if (!toasts.length) return null;
  return (
    <div
      style={{
        position: "absolute",
        bottom: 146, // clear of the bottom-center drive pad
        left: "50%",
        transform: "translateX(-50%)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 6,
        pointerEvents: "none",
        zIndex: 30,
      }}
    >
      {toasts.map((t) => (
        <div
          key={t.key}
          style={{
            background: "rgba(14, 16, 20, 0.86)",
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 8,
            padding: "5px 12px",
            color: "#d8e4d8",
            fontFamily: mono,
            fontSize: 12,
            lineHeight: 1.4,
            whiteSpace: "nowrap",
            backdropFilter: "blur(6px)",
          }}
        >
          {t.text}
        </div>
      ))}
    </div>
  );
}
