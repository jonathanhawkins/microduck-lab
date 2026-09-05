"use client";

// What the brain sees. For a duck driven by a learned brain, the frame
// carries the network's last decision (brain.view): the 80 floats it read,
// the action it emitted before the intent clip, and the clipped action the
// duck was given. This panel draws that live — a strip of bars for the
// observation, one gauge per action — painted from client.frame on every
// animation frame, like the ToF heatmap, so nothing here is React state at
// 25 Hz.
//
// Two things it shows that nothing else on the page does: the observation
// AS THE NETWORK GETS IT (clipped, aged, in the tracker's terms, not the
// sensor's), and each action against its bounds — an action pinned on its
// edge every decision, target or no target, is the saturated-mean tell the
// reward curve never shows. (The exported graph clamps, so there is no
// visible "ask" past the edge; the pin IS the ask.)

import React, { useEffect, useRef, useState } from "react";
import { ACTION_NAMES, atBound, gauge, normalizeObs, obsGroups, obsSlots, targetReadout, type BrainView } from "@/lib/brainview";
import type { SimClient } from "@/lib/sim";

const GROUP_COLOR: Record<string, string> = {
  tof: "#43c2b8",
  age: "#5f6b78",
  target: "#f2b632",
  since: "#5f6b78",
  act: "#c4b5fd",
  speed: "#93c5fd",
  track: "#f9a8d4",
  goal: "#6ee7b7",
  line: "#6ee7b7",
  ball: "#fdba74",
  busy: "#5f6b78",
  obs: "#9aa5b1",
};

const DIM = "#9aa5b1";
const SAT = "#f2b632";

export function BrainPanel({ client, duckId }: { client: SimClient; duckId: string | null }) {
  // The shapes change only when the brain does (80 vs 88 slots, 3 vs 5
  // actions) — the one thing that goes through React state.
  const [shape, setShape] = useState<{ n: number; nAct: number }>({ n: 0, nAct: 0 });
  const bars = useRef<(HTMLDivElement | null)[]>([]);
  const fills = useRef<(HTMLDivElement | null)[]>([]);
  const raws = useRef<(HTMLDivElement | null)[]>([]);
  const labels = useRef<(HTMLSpanElement | null)[]>([]);
  const head = useRef<HTMLSpanElement>(null);
  const target = useRef<HTMLDivElement>(null);
  const hoverRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const hoverIdx = useRef<number | null>(null);
  hoverIdx.current = hover;

  useEffect(() => {
    let raf = 0;
    const paint = () => {
      raf = requestAnimationFrame(paint);
      const d = client.frame?.ducks.find((x) => x.id === duckId);
      const v: BrainView | undefined = d?.brain.view;
      if (!v) {
        if (shape.n) setShape({ n: 0, nAct: 0 });
        return;
      }
      if (v.obs.length !== shape.n || v.act.clipped.length !== shape.nAct) {
        setShape({ n: v.obs.length, nAct: v.act.clipped.length });
        return;
      }
      const norm = normalizeObs(v);
      for (let i = 0; i < norm.length; i++) {
        const b = bars.current[i];
        if (b) b.style.height = `${Math.max(1, Math.round(norm[i] * 100))}%`;
      }
      const sat = atBound(v.act);
      for (let i = 0; i < v.act.clipped.length; i++) {
        const g = gauge(v.act, i);
        const f = fills.current[i];
        if (f) {
          const a = Math.min(g.zero, g.value);
          f.style.left = `${a * 100}%`;
          f.style.width = `${Math.abs(g.value - g.zero) * 100}%`;
          f.style.background = sat[i] ? SAT : "#c4b5fd";
        }
        const r = raws.current[i];
        if (r) {
          r.style.left = `calc(${g.raw * 100}% - 1px)`;
          r.style.background = sat[i] ? SAT : "#e9edf1";
        }
        const l = labels.current[i];
        if (l) {
          l.textContent = `${ACTION_NAMES[i] ?? `a${i}`} ${v.act.clipped[i].toFixed(2)}${sat[i] ? " ⊣" : ""}`;
          l.style.color = sat[i] ? SAT : "#e9edf1";
        }
      }
      if (head.current) head.current.textContent = `${v.obs.length} in → ${v.act.clipped.length} out · every ${v.decide_every} ticks · obs v${v.obs_version}`;
      if (target.current) target.current.textContent = targetReadout(v);
      if (hoverRef.current) {
        const h = hoverIdx.current;
        hoverRef.current.textContent = h === null ? "hover a bar" : `[${h}] ${obsSlots(v.obs.length)[h].label} = ${v.obs[h].toFixed(3)}`;
      }
    };
    raf = requestAnimationFrame(paint);
    return () => cancelAnimationFrame(raf);
  }, [client, duckId, shape]);

  if (!shape.n) return null;
  const groups = obsGroups(shape.n);
  const slots = obsSlots(shape.n);

  return (
    <div style={{ marginTop: 8 }} data-policy-ui>
      <div style={{ color: DIM, letterSpacing: ".08em", textTransform: "uppercase", fontSize: 10, marginBottom: 4 }}>
        What the brain sees <span ref={head} style={{ textTransform: "none", letterSpacing: 0 }} />
      </div>
      {/* the observation, one bar per float, grouped by meaning */}
      <div
        style={{ display: "flex", alignItems: "flex-end", gap: 1, height: 36, background: "#0d1015", border: "1px solid #1f2937", borderRadius: 4, padding: "2px 2px 0" }}
        onMouseLeave={() => setHover(null)}
        title="each bar is one float of the observation, scaled by that slot's own range — hover for its value"
      >
        {slots.map((s, i) => (
          <div
            key={i}
            ref={(el) => {
              bars.current[i] = el;
            }}
            onMouseEnter={() => setHover(i)}
            style={{ flex: 1, minWidth: 1, height: "1%", background: GROUP_COLOR[s.group] ?? DIM, opacity: hover === null || hover === i ? 1 : 0.55, borderRadius: "1px 1px 0 0" }}
          />
        ))}
      </div>
      <div style={{ display: "flex", gap: 1, marginTop: 2, fontSize: 9, color: DIM }}>
        {groups.map((g) => (
          <div
            key={`${g.group}-${g.start}`}
            style={{ flex: g.end - g.start, minWidth: 0, overflow: "hidden", whiteSpace: "nowrap", textAlign: "center", borderTop: `2px solid ${GROUP_COLOR[g.group] ?? DIM}` }}
            title={`obs[${g.start}${g.end - g.start > 1 ? `:${g.end}` : ""}]`}
          >
            {g.end - g.start >= 3 ? g.group : ""}
          </div>
        ))}
      </div>
      <div ref={hoverRef} style={{ color: DIM, fontSize: 10, marginTop: 2 }}>hover a bar</div>
      <div ref={target} style={{ color: "#e9edf1", marginTop: 4 }} title="the target slots [65:71] and the track flags, in words" />

      {/* the action, one gauge per output: the value fills from zero; the tick marks its end (and the raw output, which the exported graph has already clamped) */}
      <div style={{ marginTop: 6, display: "grid", gridTemplateColumns: "auto 1fr", gap: "3px 8px", alignItems: "center" }}>
        {Array.from({ length: shape.nAct }, (_, i) => (
          <React.Fragment key={i}>
            <span
              ref={(el) => {
                labels.current[i] = el;
              }}
              style={{ fontSize: 11, whiteSpace: "nowrap" }}
            />
            <div
              style={{ position: "relative", height: 8, background: "#0d1015", border: "1px solid #1f2937", borderRadius: 4 }}
              title="the bar is the action from zero, on a track that spans this brain's bounds; ⊣ and amber mean it is pinned on a bound — every decision there, target or not, is the saturated-mean trap"
            >
              <div
                ref={(el) => {
                  fills.current[i] = el;
                }}
                style={{ position: "absolute", top: 0, bottom: 0, left: 0, width: 0, background: "#c4b5fd", borderRadius: 3 }}
              />
              <div
                ref={(el) => {
                  raws.current[i] = el;
                }}
                style={{ position: "absolute", top: -2, bottom: -2, left: 0, width: 2, background: "#e9edf1" }}
              />
            </div>
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
