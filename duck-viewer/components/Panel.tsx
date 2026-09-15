"use client";

// The chrome every overlay panel on /sim shares: the frosted box, and the
// —/+ in its title bar. Extracted from SimViewer when the state-graph panel
// became the fifth user — a panel that re-declares the same twelve style
// properties inline is how the set drifts apart, and importing them back out
// of SimViewer would be a cycle (SimViewer renders the panels).

import React from "react";

export const PANEL: React.CSSProperties = {
  position: "absolute",
  background: "rgba(16,18,22,0.86)",
  border: "1px solid #2b313b",
  borderRadius: 6,
  color: "#e9edf1",
  fontFamily: "ui-monospace, Menlo, monospace",
  fontSize: 12,
  padding: "8px 10px",
  zIndex: 20,
  backdropFilter: "blur(6px)",
};

export function PanelToggle({ open, onToggle, what, hint }: {
  open: boolean;
  onToggle: () => void;
  what: string;                 // "the inspector" — reads out as "minimize the inspector"
  hint?: string;                // the keyboard shortcut, shown in the tooltip
}) {
  const verb = open ? "minimize" : "expand";
  return (
    <button
      onPointerDown={(e) => e.stopPropagation()}
      onClick={onToggle}
      title={hint ? `${verb} (${hint})` : verb}
      aria-label={`${verb} ${what}`}
      aria-expanded={open}
      style={{ background: "none", border: "none", color: "#9aa5b1", cursor: "pointer", fontFamily: "inherit", fontSize: 12, padding: "0 4px", marginLeft: 10, lineHeight: 1 }}
    >
      {open ? "—" : "+"}
    </button>
  );
}
