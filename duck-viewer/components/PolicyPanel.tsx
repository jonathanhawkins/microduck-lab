"use client";

// Policy palette (top-right): every brain the farm server knows about, as
// draggable chips grouped by origin. Drop a chip on a duck — or click to arm
// it, then click a duck — to hot-swap that duck's policy over the WebSocket.
// Drop (or armed-click) on EMPTY floor instead — farther than the assign
// radius from every duck — to spawn a fresh duck running that policy.
// Pointer events only (no HTML5 DnD): a ghost chip follows the cursor and the
// Canvas-side AssignTargets helper (Viewer.tsx) picks/highlights the nearest
// duck via lib/assign's shared store. Styling mirrors Hud.tsx.

import { useCallback, useEffect, useRef, useState } from "react";
import { fetchPolicies, type FarmClient, type Policy } from "@/lib/farm";
import { assignDrag, clearAssignDrag, isCanvasAt, nearestDuck } from "@/lib/assign";
import { loadJSON, saveJSON } from "@/lib/persist";
import { setPolicyOpen } from "@/lib/ui";
import { Tip } from "./TeachPanel";
import { pushToast } from "./Toasts";

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";

const GROUPS: { key: Policy["group"]; title: string }[] = [
  { key: "pollen", title: "Pollen (shipped)" },
  { key: "runs", title: "Our runs" },
  { key: "checkpoints", title: "Checkpoints" },
];

const DRAG_THRESHOLD_PX = 5; // less movement than this counts as a click

/** "2h ago"-style label from an epoch-seconds timestamp. Computed at render;
 *  the panel re-renders once a minute so it can't go stale. */
function relTime(epochS: number): string {
  const s = Math.max(0, Date.now() / 1000 - epochS);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

/** "Our runs" rows: curriculum chains (same teach-…-<hash> prefix, -sN
 *  suffixes) fold into one family of compact stage chips, positioned by
 *  their NEWEST stage (the list arrives newest-first from the server, so
 *  first-seen = newest); everything else stays a single chip + time. */
type RunRow =
  | { kind: "single"; p: Policy }
  | { kind: "chain"; chain: string; newest: number; stages: Policy[] };

function runRows(list: Policy[]): RunRow[] {
  const rows: RunRow[] = [];
  const chains = new Map<string, Extract<RunRow, { kind: "chain" }>>();
  for (const p of list) {
    if (p.chain) {
      let row = chains.get(p.chain);
      if (!row) {
        row = { kind: "chain", chain: p.chain, newest: p.mtime ?? 0, stages: [] };
        chains.set(p.chain, row);
        rows.push(row);
      }
      row.stages.push(p);
      row.newest = Math.max(row.newest, p.mtime ?? 0);
    } else {
      rows.push({ kind: "single", p });
    }
  }
  for (const row of chains.values())
    row.stages.sort((a, b) => (a.stage ?? 0) - (b.stage ?? 0));
  return rows;
}

function Chip({
  policy,
  display,
  armed,
  accent,
  title,
  onDown,
  onMove,
  onUp,
}: {
  policy: Policy;
  /** Chip text override (stage chips show "s2"); drag ghost, toasts and
   *  tooltips keep the full policy.label so nothing ambiguous ships over
   *  the wire. */
  display?: string;
  armed: boolean;
  /** Warm-tinted styling for the chain-level "whole trick" chip, so it
   *  stands out from the cool single-stage chips beside it. */
  accent?: boolean;
  /** Native-tooltip override — "" suppresses it entirely, for chips whose
   *  explainer is a styled <Tip> wrapper instead. */
  title?: string;
  onDown: (e: React.PointerEvent<HTMLButtonElement>) => void;
  onMove: (e: React.PointerEvent<HTMLButtonElement>) => void;
  onUp: (e: React.PointerEvent<HTMLButtonElement>) => void;
}) {
  const [hover, setHover] = useState(false);
  return (
    <button
      type="button"
      title={
        title ??
        (armed
          ? `armed — click a duck to assign ${policy.label}, or empty floor to spawn`
          : `drag onto a duck to assign — or empty floor to spawn (click to arm): ${policy.label}`)
      }
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerEnter={() => setHover(true)}
      onPointerLeave={() => setHover(false)}
      style={{
        background: armed ? "#2a3548" : hover ? "#232a3a" : "#1c2230",
        color: armed ? "#cfe4f5" : accent ? "#d8cfa0" : "#9fb4d8",
        border: `1px solid ${
          armed ? "#7db8d8" : accent ? "rgba(216, 198, 125, 0.45)" : "rgba(255,255,255,0.08)"
        }`,
        borderRadius: 12,
        padding: "3px 9px",
        fontFamily: mono,
        fontSize: 11,
        lineHeight: 1.4,
        cursor: "grab",
        userSelect: "none",
        touchAction: "none", // pointer-event drags must not turn into scrolls
        // Long run/checkpoint names must truncate, not spill over neighbors:
        // flex items refuse to shrink below content unless minWidth is 0.
        minWidth: 0,
        maxWidth: 186,
        boxSizing: "border-box",
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
      }}
    >
      {display ?? policy.label}
    </button>
  );
}

export function PolicyPanel({
  clientRef,
}: {
  clientRef: React.MutableRefObject<FarmClient | null>;
}) {
  const [open, setOpen] = useState(() => loadJSON("policyOpen", true));
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [err, setErr] = useState(false);
  // Armed chip. `showcase` disambiguates the chain-level "whole trick" chip
  // from the final-stage chip — they share a policy id (the whole trick IS
  // the final stage's policy), but must arm independently.
  const [armed, setArmed] = useState<{ id: string; showcase: boolean } | null>(null);
  const [dragging, setDragging] = useState<Policy | null>(null);
  const ghostRef = useRef<HTMLDivElement | null>(null);
  const ghostSubRef = useRef<HTMLDivElement | null>(null);
  const moved = useRef(false);
  const start = useRef({ x: 0, y: 0 });

  useEffect(() => {
    saveJSON("policyOpen", open);
    setPolicyOpen(open); // lets the TeachPanel reclaim the vertical space
  }, [open]);

  const refresh = useCallback(() => {
    let stale = false;
    fetchPolicies()
      .then((p) => {
        if (stale) return;
        setPolicies(p);
        setErr(false);
      })
      .catch(() => {
        if (!stale) setErr(true);
      });
    return () => {
      stale = true;
    };
  }, []);

  // (Re)fetch whenever the panel is expanded — new training runs appear over time.
  useEffect(() => {
    if (!open) return;
    return refresh();
  }, [open, refresh]);

  // Relative "2h ago" labels are computed at render — tick once a minute so
  // an open panel's labels age along with the runs instead of going stale.
  const [, setTimeTick] = useState(0);
  useEffect(() => {
    if (!open) return;
    const id = setInterval(() => setTimeTick((n) => n + 1), 60_000);
    return () => clearInterval(id);
  }, [open]);

  // A finished training run drops a new "Our runs" policy on disk: watch the
  // streamed training status and refetch on the training → done transition.
  const prevStatus = useRef<string | null>(null);
  useEffect(() => {
    const id = setInterval(() => {
      const status = clientRef.current?.frame?.training?.status ?? null;
      if (prevStatus.current === "training" && status === "done") refresh();
      prevStatus.current = status;
    }, 1000);
    return () => clearInterval(id);
  }, [clientRef, refresh]);

  const disarm = () => {
    clearAssignDrag();
    setArmed(null);
  };

  /** True when this exact chip (id + whole-trick-ness) is the armed one. */
  const chipArmed = (id: string, showcase = false) =>
    armed !== null && armed.id === id && armed.showcase === showcase;

  const assign = (duckId: string, policy: { id: string; label: string }, showcase = false) => {
    clientRef.current?.sendAssign(duckId, policy.id, showcase);
    pushToast(`⚡ assigning ${policy.label} → ${duckId}`);
  };

  // Empty-floor drop/click: spawn a fresh duck running the policy. The server
  // confirms (or refuses, e.g. at the 20-duck cap) with its own event toast.
  const spawn = (policy: { id: string; label: string }, showcase = false) => {
    clientRef.current?.sendSpawnDuck(policy.id, showcase);
    pushToast(`⚡ spawning ${policy.label}…`);
  };

  // --- drag gesture (pointer capture keeps every move/up on the chip) ---

  const chipDown =
    (p: Policy, showcase = false) =>
    (e: React.PointerEvent<HTMLButtonElement>) => {
      if (e.pointerType === "mouse" && e.button !== 0) return;
      try {
        e.currentTarget.setPointerCapture(e.pointerId);
      } catch {
        // synthetic/stale pointer ids can't be captured — drag still works
        // through bubbling for clicks, so don't let this abort the gesture
      }
      moved.current = false;
      start.current = { x: e.clientX, y: e.clientY };
      assignDrag.mode = "drag";
      assignDrag.policyId = p.id;
      assignDrag.policyLabel = p.label;
      assignDrag.showcase = showcase;
      assignDrag.px = e.clientX;
      assignDrag.py = e.clientY;
      setDragging(p);
    };

  const chipMove = (e: React.PointerEvent<HTMLButtonElement>) => {
    if (assignDrag.mode !== "drag") return;
    assignDrag.px = e.clientX;
    assignDrag.py = e.clientY;
    if (
      !moved.current &&
      Math.hypot(e.clientX - start.current.x, e.clientY - start.current.y) >
        DRAG_THRESHOLD_PX
    )
      moved.current = true;
    const g = ghostRef.current;
    if (g && moved.current) {
      g.style.display = "block";
      g.style.left = `${e.clientX}px`;
      g.style.top = `${e.clientY}px`;
      // Sublabel tells the drop outcome apart: duck in range → assign, open
      // canvas → spawn, anything else (panels/buttons) → cancel. Same checks
      // the chipUp drop handler runs, so the hint can't lie.
      const sub = ghostSubRef.current;
      if (sub) {
        sub.textContent = nearestDuck(e.clientX, e.clientY)
          ? "drop to assign"
          : isCanvasAt(e.clientX, e.clientY)
            ? "drop to spawn"
            : "release to cancel";
      }
    }
  };

  // The ghost is shown/positioned imperatively (chipMove), so it must be
  // hidden imperatively too — React never rewrites display since the ghost's
  // style prop is unchanged between renders. Without this the pill lingered
  // at the drop point after every completed drag.
  const hideGhost = () => {
    const g = ghostRef.current;
    if (g) g.style.display = "none";
  };

  const chipUp =
    (p: Policy, showcase = false) =>
    (e: React.PointerEvent<HTMLButtonElement>) => {
      setDragging(null);
      hideGhost();
      if (assignDrag.mode !== "drag") return; // canceled (Escape)
      if (!moved.current) {
        // No movement → click: toggle armed mode instead.
        if (chipArmed(p.id, showcase)) {
          disarm();
        } else {
          setArmed({ id: p.id, showcase });
          assignDrag.mode = "armed";
          assignDrag.policyId = p.id;
          assignDrag.policyLabel = p.label;
          assignDrag.showcase = showcase;
        }
        return;
      }
      // Real drag → drop. Compute the target from the event position directly
      // (not hoverDuck) so a fast drag can't outrun the render loop. No duck in
      // range but open canvas under the cursor → spawn; dropped on UI → cancel.
      const duck = nearestDuck(e.clientX, e.clientY);
      if (duck) assign(duck, p, showcase);
      else if (isCanvasAt(e.clientX, e.clientY)) spawn(p, showcase);
      disarm();
    };

  // Escape cancels an in-flight drag.
  useEffect(() => {
    if (!dragging) return;
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        clearAssignDrag();
        setArmed(null);
        setDragging(null);
        hideGhost();
      }
    };
    window.addEventListener("keydown", key, true);
    return () => window.removeEventListener("keydown", key, true);
  }, [dragging]);

  // Armed mode: track the pointer for the highlight, assign on duck click,
  // disarm on click-elsewhere or Escape.
  useEffect(() => {
    if (!armed) return;
    const move = (e: PointerEvent) => {
      assignDrag.px = e.clientX;
      assignDrag.py = e.clientY;
    };
    const down = (e: PointerEvent) => {
      const t = e.target as HTMLElement | null;
      if (t?.closest("[data-policy-ui]")) return; // panel clicks handled above
      if (assignDrag.policyId) {
        const policy = {
          id: assignDrag.policyId,
          label: assignDrag.policyLabel ?? assignDrag.policyId,
        };
        const duck = nearestDuck(e.clientX, e.clientY);
        if (duck) assign(duck, policy, assignDrag.showcase);
        // Armed click on open canvas away from every duck → spawn there;
        // clicks on other UI (HUD, pads) just disarm as before.
        else if (isCanvasAt(e.clientX, e.clientY)) spawn(policy, assignDrag.showcase);
      }
      disarm();
    };
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") disarm();
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerdown", down, true);
    window.addEventListener("keydown", key, true);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerdown", down, true);
      window.removeEventListener("keydown", key, true);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [armed]);

  const ghost = (
    <div
      ref={ghostRef}
      style={{
        display: "none",
        position: "fixed",
        left: 0,
        top: 0,
        transform: "translate(-50%, -140%)",
        background: "rgba(42, 53, 72, 0.92)",
        color: "#cfe4f5",
        border: "1px solid #7db8d8",
        borderRadius: 12,
        padding: "3px 9px",
        fontFamily: mono,
        fontSize: 11,
        whiteSpace: "nowrap",
        pointerEvents: "none",
        zIndex: 100,
        boxShadow: "0 4px 14px rgba(0,0,0,0.45)",
      }}
    >
      {dragging?.label}
      {/* outcome hint, retargeted per pointer move (assign/spawn/cancel) */}
      <div
        ref={ghostSubRef}
        style={{
          fontSize: 9,
          color: "#8fb8d0",
          textAlign: "center",
          marginTop: 1,
        }}
      >
        drop to spawn
      </div>
    </div>
  );

  if (!open)
    return (
      <button
        data-policy-ui
        onClick={() => setOpen(true)}
        style={{
          position: "absolute",
          right: 14,
          top: 14,
          background: "rgba(14,16,20,0.86)",
          color: "#e8e6e1",
          border: "1px solid rgba(255,255,255,0.12)",
          borderRadius: 10,
          padding: "8px 12px",
          fontFamily: mono,
          fontSize: 12,
          cursor: "pointer",
          backdropFilter: "blur(6px)",
        }}
      >
        🧠 policies
      </button>
    );

  return (
    <>
      <div
        data-policy-ui
        style={{
          position: "absolute",
          right: 14,
          top: 14,
          width: 230,
          // The TeachPanel owns the bottom-right: its maxHeight is derived
          // from this cap (see TeachPanel.tsx), so the two can never collide.
          maxHeight: "min(40vh, 380px)",
          display: "flex",
          flexDirection: "column",
          background: "rgba(14, 16, 20, 0.82)",
          border: "1px solid rgba(255,255,255,0.09)",
          borderRadius: 10,
          color: "#e8e6e1",
          fontFamily: mono,
          fontSize: 12,
          lineHeight: 1.55,
          backdropFilter: "blur(6px)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            padding: "8px 12px",
            fontWeight: 700,
            fontSize: 13,
            borderBottom: "1px solid rgba(255,255,255,0.08)",
            flexShrink: 0,
            display: "flex",
            alignItems: "center",
          }}
        >
          <span style={{ flex: 1 }}>🧠 policies</span>
          <button
            onClick={refresh}
            title="refresh policy list"
            style={{
              background: "none",
              border: "none",
              color: "#8b93a3",
              cursor: "pointer",
              fontFamily: mono,
              fontSize: 12,
              padding: "0 4px",
            }}
          >
            ↻
          </button>
          <button
            onClick={() => setOpen(false)}
            title="collapse"
            style={{
              // keep clear of the Next.js dev-tools badge that floats in
              // this corner during development
              marginRight: 30,
              background: "none",
              border: "none",
              color: "#8b93a3",
              cursor: "pointer",
              fontFamily: mono,
              fontSize: 12,
              padding: "0 4px",
            }}
          >
            —
          </button>
        </div>

        <div style={{ overflowY: "auto", padding: "4px 10px 10px" }}>
          {err && (
            <div style={{ color: "#e07a5f", margin: "6px 0" }}>
              ⚠ can&apos;t load policies from :8788
            </div>
          )}
          {GROUPS.map(({ key, title }) => {
            const list = policies.filter((p) => p.group === key);
            if (!list.length) return null;
            return (
              <div key={key}>
                <div style={{ color: "#8b93a3", fontSize: 10, margin: "7px 0 4px" }}>
                  {title}
                </div>
                {key === "runs" ? (
                  // Newest-first (server-sorted by mtime), curriculum chains
                  // folded into one family row of stage chips — every chip,
                  // stage chips included, drags/arms exactly like before.
                  runRows(list).map((row) => {
                    if (row.kind === "single")
                      return (
                        <div
                          key={row.p.id}
                          style={{ display: "flex", alignItems: "center", gap: 6, margin: "3px 0" }}
                        >
                          <Chip
                            policy={row.p}
                            armed={chipArmed(row.p.id)}
                            onDown={chipDown(row.p)}
                            onMove={chipMove}
                            onUp={chipUp(row.p)}
                          />
                          {row.p.mtime != null && (
                            <span style={{ color: "#8b93a3", fontSize: 9, flexShrink: 0 }}>
                              {relTime(row.p.mtime)}
                            </span>
                          )}
                        </div>
                      );
                    // The chain-level "whole trick" chip assigns the FINAL
                    // stage's policy — each stage fine-tunes the same network,
                    // so the last one carries the entire curriculum — flagged
                    // showcase so the duck's env rehearses the whole trick arc
                    // instead of only a standing start. Ghost/toast label is
                    // the chain's name (✨), matching the server's roster label.
                    const last = row.stages[row.stages.length - 1];
                    const whole: Policy = {
                      ...last,
                      label: `${row.chain.replace(/^teach-/, "")} ✨`,
                    };
                    return (
                      <div key={row.chain} style={{ margin: "4px 0" }}>
                        <div
                          style={{
                            color: "#aab3c0",
                            fontSize: 10,
                            display: "flex",
                            gap: 6,
                            alignItems: "baseline",
                          }}
                        >
                          <span
                            style={{
                              minWidth: 0,
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                              whiteSpace: "nowrap",
                            }}
                          >
                            {row.chain.replace(/^teach-/, "")}
                          </span>
                          <span style={{ color: "#8b93a3", fontSize: 9, flexShrink: 0 }}>
                            {row.stages.length} stages · {relTime(row.newest)}
                          </span>
                        </div>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 2 }}>
                          <Tip
                            tip={
                              <>
                                each stage trains the same brain — the final stage
                                carries all of it. ▶ assigns that finished policy in
                                showcase mode: the duck rehearses spawns across the
                                whole trick arc, so every section gets performed.
                              </>
                            }
                          >
                            <Chip
                              policy={whole}
                              display="▶ whole trick"
                              accent
                              title=""
                              armed={chipArmed(last.id, true)}
                              onDown={chipDown(whole, true)}
                              onMove={chipMove}
                              onUp={chipUp(whole, true)}
                            />
                          </Tip>
                          {row.stages.map((p) => (
                            <Chip
                              key={p.id}
                              policy={p}
                              display={`s${p.stage}`}
                              armed={chipArmed(p.id)}
                              onDown={chipDown(p)}
                              onMove={chipMove}
                              onUp={chipUp(p)}
                            />
                          ))}
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                    {list.map((p) => (
                      <Chip
                        key={p.id}
                        policy={p}
                        armed={chipArmed(p.id)}
                        onDown={chipDown(p)}
                        onMove={chipMove}
                        onUp={chipUp(p)}
                      />
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {armed && (
          <div
            style={{
              padding: "6px 12px",
              borderTop: "1px solid rgba(255,255,255,0.08)",
              color: "#7db8d8",
              fontSize: 10,
              flexShrink: 0,
            }}
          >
            armed — click a duck to assign, or empty floor to spawn · esc
            cancels
          </div>
        )}
      </div>
      {ghost}
    </>
  );
}
