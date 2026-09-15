"use client";

// The states a brain moves through, drawn.
//
// The nodes are declared by the lab (brain/graph.py) and arrive once in the
// world-info message; the edges are drawn from what this duck's brain
// actually does, live. That split is the point. None of the scripted brains
// is a declared state machine — `Chase` sets `self.state` from twenty-odd
// places in one long step — so an allowed-transition diagram would be a
// fiction. What you see here is this run's trajectory: an arc appears the
// first time the duck makes that move, thickens as it repeats, and fades as
// it goes stale.
//
// What it teaches, in order of how often it has actually been useful:
//   * a duck stuck in a two-node loop (search -> turn -> search) is obvious
//     as a pair of fat arcs long before it is obvious in the 3-D view;
//   * a dim chip is a state this scenario never reaches — `block` never
//     lights up without a keeper, `duel` never without an opponent;
//   * and switching the duck to a learned brain collapses the whole picture
//     to two chips, which is the most honest thing this page says about the
//     difference between the two kinds of brain.
//
// Painted per-frame off client.frame like the other live panels; React state
// holds only the SET being drawn, which changes on a new state or a new edge
// and not at 25 Hz.

import React, { useEffect, useId, useMemo, useRef, useState } from "react";
import {
  LAYOUT,
  clampScale,
  labelWidth,
  type BrainGraphInfo,
  type Placed,
  StateTrace,
  edgeAlpha,
  edgeKey,
  edgePath,
  layoutGraph,
  selfPath,
} from "@/lib/braingraph";
import type { SimClient } from "@/lib/sim";
import { loadJSON, saveJSON } from "@/lib/persist";
import { PANEL, PanelToggle } from "./Panel";
import { useDrag } from "./useDrag";

const DIM = "#9aa5b1";
const LIVE = "#f2b632";
const TOP = 16;          // room above the top band for a self-loop
const PAD = 10;          // panel inset from the window edge (SimViewer's PAD)
const INSPECTOR_W = 242; // the panel docks to the left of the inspector

const GROUP_COLOR: Record<string, string> = {
  find: "#43c2b8",
  go: "#93c5fd",
  strike: "#fdba74",
  safe: "#f87171",
  team: "#c4b5fd",
  grasp: "#fdba74",
  carry: "#c4b5fd",
  deliver: "#6ee7b7",
  end: "#6ee7b7",
  drive: "#43c2b8",
  stuck: "#f87171",
  manual: DIM,
  net: "#f9a8d4",
};

export function StateGraphPanel(
  { client, duckId, graphs, open, onToggle, minY }:
  {
    client: SimClient;
    duckId: string | null;
    graphs?: Record<string, BrainGraphInfo>;
    open: boolean;
    onToggle: () => void;
    /** Top of the room area — a dragged panel may not go above the top bar. */
    minY: number;
  },
) {
  const box = useRef<HTMLDivElement>(null);
  const drag = useDrag("simGraphPos", box, minY, PAD);
  // How big the drawing is. The SVG keeps ONE internal coordinate system and
  // is rendered at `LAYOUT.width * scale`, so the chips, the arcs and the
  // state names all grow together and no layout arithmetic changes — the
  // alternative, scaling the layout constants, re-wraps the rows at every
  // step and makes the chips move around under the pointer while you drag.
  const [scale, setScale] = useState(() => clampScale(loadJSON("simGraphScale", 1)));
  useEffect(() => saveJSON("simGraphScale", scale), [scale]);
  const resizing = useRef<{ x: number; s: number } | null>(null);
  const maskId = useId();
  // The graph being drawn. Changes when the selected duck changes or its
  // brain is swapped — not per frame.
  const [info, setInfo] = useState<BrainGraphInfo | null>(null);
  // Bumped whenever a new node or edge appears, to re-render the SVG.
  const [rev, setRev] = useState(0);
  const trace = useRef(new StateTrace());
  const svgRef = useRef<SVGSVGElement>(null);
  // Resolved AFTER each commit by querying the committed SVG, not collected
  // by ref callbacks. A ref callback on a node inside a subtree React
  // re-creates can be handed `null` for the old element after the new one
  // has already registered, which leaves the map holding dead nodes: the
  // header kept updating (its ref is outside the subtree) while not one chip
  // ever changed colour. Reading the DOM once per commit cannot get that
  // wrong, and it is 18 lookups per new edge, not per frame.
  const chips = useRef(new Map<string, { box: SVGRectElement; label: SVGTextElement }>());
  const arcs = useRef(new Map<string, SVGPathElement>());
  const head = useRef<HTMLSpanElement>(null);
  const trail = useRef<HTMLDivElement>(null);
  const noteRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<string | null>(null);
  const hoverRef = useRef<string | null>(null);
  hoverRef.current = hover;
  // Read through a ref, never a dependency: `open` in the paint effect's deps
  // tore the effect down on every collapse, and its `key` starts null again,
  // so the next frame took the reset branch and threw the run away.
  const openRef = useRef(open);
  openRef.current = open;

  useEffect(() => {
    const m = new Map<string, { box: SVGRectElement; label: SVGTextElement }>();
    svgRef.current?.querySelectorAll<SVGGElement>("g[data-state]").forEach((el) => {
      const box = el.querySelector("rect");
      const label = el.querySelector("text");
      if (box && label) m.set(el.dataset.state!, { box, label });
    });
    chips.current = m;
    const a = new Map<string, SVGPathElement>();
    svgRef.current?.querySelectorAll<SVGPathElement>("path[data-edge]").forEach((el) => a.set(el.dataset.edge!, el));
    arcs.current = a;
  });

  const layout = useMemo(() => (info ? layoutGraph(info) : null), [info]);
  const byName = useMemo(() => {
    const m: Record<string, Placed> = {};
    for (const n of layout?.nodes ?? []) m[n.name] = n;
    return m;
  }, [layout]);

  useEffect(() => {
    let raf = 0;
    let key: string | null = null;
    const paint = () => {
      raf = requestAnimationFrame(paint);
      // `client.live`, not `client.frame`: every other panel may read the
      // scrubbed frame, but this one accumulates history, and `frame` is
      // `scrub ?? live` — so pressing space to look back rewound `t` by up
      // to two minutes and reset the run it was there to inspect.
      const f = client.live;
      const d = f?.ducks.find((x) => x.id === duckId);
      const g = d?.brain.graph ?? null;
      // A different duck, or the same duck on a different brain: the trace
      // belongs to one brain's run and must not be carried across. Keyed on
      // the BRAIN and the graph, because every learned:<run> shares the
      // "learned" graph — keying on the graph alone let a swap between two
      // learned brains inherit the previous one's edges and dwell totals.
      // `brainKind` (the duck's OWN brain), not `brain.kind` (who is steering
      // this tick) — the latter reads "manual" for every duck while a drive
      // command is held, so one WASD tap threw the whole run away and a
      // second reset landed when the hold lapsed.
      const id = d ? `${d.brainKind ?? d.brain.kind}\u0000${g}` : null;
      if (id !== key) {
        key = id;
        trace.current.reset();
        setInfo(g ? (graphs?.[g] ?? null) : null);
        setRev((v) => v + 1);
        return;
      }
      if (!d || !f || !g) return;
      // The trace keeps running while the panel is collapsed — reopening it
      // to an empty graph would throw away exactly the history you collapsed
      // it to make room for. Only the painting below is skipped.
      if (trace.current.step(d.brain.state, f.t, f.simSpeed ?? 1)) setRev((v) => v + 1);
      if (!openRef.current) return;

      const tr = trace.current;
      for (const [name, { box, label }] of chips.current) {
        const v = tr.visits.get(name);
        const live = tr.current === name;
        box.setAttribute("stroke", live ? LIVE : v ? "#4b5563" : "#232a33");
        // Opaque, not rgba: the amber tint at 0.18 let the arcs show through
        // the one chip the eye is always on.
        box.setAttribute("fill", live ? "#39301e" : v ? "#161a20" : "#101419");
        label.setAttribute("opacity", live ? "1" : v ? "0.82" : "0.3");
      }
      for (const [k, el] of arcs.current) {
        const e = tr.edges.get(k);
        if (!e) continue;
        el.setAttribute("opacity", String(edgeAlpha(e, f.t)));
        el.setAttribute("stroke-width", String(Math.min(3.2, 1.0 + 0.35 * e.count)));
      }
      if (head.current) {
        const n = tr.visits.size;
        head.current.textContent =
          `${tr.current ?? "—"} · ${tr.dwell(f.t).toFixed(1)}s · ${n}/${info?.nodes.length ?? 0} states, ${tr.edges.size} moves seen`;
      }
      if (trail.current) trail.current.textContent = tr.trail.join(" → ") || "—";
      if (noteRef.current) {
        const h = hoverRef.current;
        const node = h ? byName[h] : tr.current ? byName[tr.current] : null;
        const v = node ? tr.visits.get(node.name) : undefined;
        noteRef.current.textContent = !node
          ? ""
          : `${node.name} — ${node.note}` + (v ? `  (${v.count}x, ${v.total.toFixed(1)}s)` : "  (not reached this run)");
      }
    };
    raf = requestAnimationFrame(paint);
    return () => cancelAnimationFrame(raf);
  }, [client, duckId, graphs, info, byName]);

  if (!info || !layout) return null;

  const tr = trace.current;
  const edges = [...tr.edges.values()];

  const pos = drag.pos;
  // The grip must sit on the corner that actually MOVES as the panel grows.
  // Docked, the panel is pinned by its right edge and grows leftwards, so a
  // grip on the right would stay under the cursor doing nothing; once the
  // user has dragged it somewhere it is pinned top-left and grows rightwards.
  const anchoredRight = pos === null;
  const onGripDown = (e: React.PointerEvent<HTMLDivElement>) => {
    e.stopPropagation();
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    resizing.current = { x: e.clientX, s: scale };
  };
  const onGripMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const r = resizing.current;
    if (!r) return;
    const dx = (e.clientX - r.x) * (anchoredRight ? -1 : 1);
    setScale(clampScale((LAYOUT.width * r.s + dx) / LAYOUT.width));
  };
  const onGripUp = (e: React.PointerEvent<HTMLDivElement>) => {
    resizing.current = null;
    e.currentTarget.releasePointerCapture(e.pointerId);
  };
  const px = (n: number) => Math.round(n * scale * 10) / 10;
  return (
    <div
      ref={box}
      data-graph-ui
      style={{
        ...PANEL,
        width: LAYOUT.width * scale + 20,
        boxSizing: "border-box",
        // Scaled up, the graph is taller than the window — and a panel that
        // grows off the top takes its own title bar with it, which is where
        // the live state and the counts are. Cap it and let the drawing
        // scroll inside instead; the title bar stays put.
        maxHeight: pos ? `calc(100vh - ${Math.round(pos.y) + 24}px)` : `calc(100vh - ${Math.round(minY) + 80}px)`,
        display: "flex",
        flexDirection: "column",
        ...(pos ? { top: pos.y, left: pos.x } : { bottom: 64, right: INSPECTOR_W + PAD * 2 }),
      }}
    >
      <div
        onPointerDown={drag.onPointerDown}
        onDoubleClick={drag.reset}
        style={{ display: "flex", alignItems: "flex-start", cursor: "grab", touchAction: "none", flex: "0 0 auto" }}
      >
        <div style={{ flex: 1, color: DIM, letterSpacing: ".08em", textTransform: "uppercase", fontSize: px(10), marginBottom: 2 }}>
          States <span ref={head} style={{ textTransform: "none", letterSpacing: 0 }} />
        </div>
        <PanelToggle open={open} onToggle={onToggle} what="the state graph" hint="G" />
      </div>
      {!open ? null : (
      <div style={{ overflowY: "auto", overflowX: "hidden", minHeight: 0 }}>
      <div style={{ color: DIM, fontSize: px(10), lineHeight: `${px(13)}px`, marginBottom: 4 }}>{info.note}</div>
      <svg
        ref={svgRef}
        data-rev={rev}
        width={layout.width * scale}
        height={(layout.height + TOP) * scale}
        viewBox={`0 ${-TOP} ${layout.width} ${layout.height + TOP}`}
        style={{ display: "block", maxWidth: "100%" }}
      >
        {/* Arcs are not merely painted UNDER the chips — they are masked out
            of them. Z-order alone left a hairball of 70-odd curves crossing
            the group headings, which are 9 px text with nothing behind them,
            and a 2 px line through a thin glyph wins however the z-order
            reads. Punching the chips and the headings out of the arc layer
            means a move is drawn only in the gaps between the rows, which is
            also where it is legible. */}
        <defs>
          <mask id={maskId} maskUnits="userSpaceOnUse" x={0} y={-TOP} width={layout.width} height={layout.height + TOP}>
            <rect x={0} y={-TOP} width={layout.width} height={layout.height + TOP} fill="#fff" />
            {layout.nodes.map((n) => (
              <rect key={n.name} x={n.x - 1.5} y={n.y - 1.5} width={n.w + 3} height={n.h + 3} rx={5} fill="#000" />
            ))}
            {layout.groups.map((g) => (
              <rect key={g.key} x={-2} y={g.y} width={labelWidth(g.label, 9) + 4} height={12} rx={2} fill="#000" />
            ))}
          </mask>
        </defs>
        {/* the moves this brain has actually made, in the gaps between them */}
        <g mask={`url(#${maskId})`} fill="none" stroke="#b6c0cc" strokeLinecap="round">
          {edges.map((e) => {
            const a = byName[e.from];
            const b = byName[e.to];
            if (!a || !b) return null;
            const k = edgeKey(e.from, e.to);
            return (
              <path key={k} data-edge={k} d={a === b ? selfPath(a) : edgePath(a, b)} opacity={0.3}>
                <title>{`${e.from} → ${e.to}  ×${e.count}`}</title>
              </path>
            );
          })}
        </g>
        {layout.groups.map((g) => (
          <text key={g.key} x={0} y={g.y + 10} fontSize={9} fill={GROUP_COLOR[g.key] ?? DIM} opacity={0.85}>
            {g.label}
          </text>
        ))}
        {layout.nodes.map((n) => (
          <g
            key={n.name}
            data-state={n.name}
            onMouseEnter={() => setHover(n.name)}
            onMouseLeave={() => setHover(null)}
            style={{ cursor: "default" }}
          >
            <rect x={n.x} y={n.y} width={n.w} height={n.h} rx={4} fill="#101419" stroke="#232a33" />
            <rect x={n.x} y={n.y} width={3} height={n.h} rx={1.5} fill={GROUP_COLOR[n.group] ?? DIM} opacity={0.9} />
            <text x={n.x + 7} y={n.y + n.h / 2 + 3.5} fontSize={9} fill="#e9edf1" opacity={0.3} fontFamily="ui-monospace, Menlo, monospace">
              {n.name}
            </text>
          </g>
        ))}
      </svg>
      {/* the note for whatever is under the pointer, else the state it is in */}
      <div
        ref={noteRef}
        style={{ color: "#c9d0d8", fontSize: px(10), lineHeight: `${px(13)}px`, height: px(39), marginTop: 4, overflow: "hidden", display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical" }}
        title="hover a state for what the duck is doing there, and how long it has spent there this run"
      />
      <div
        ref={trail}
        style={{ color: DIM, fontSize: px(9), lineHeight: `${px(12)}px`, height: px(24), marginTop: 2, overflow: "hidden", wordBreak: "break-word" }}
        title="the last dozen states, oldest first"
      />
      </div>
      )}
      {open && (
        <div
          onPointerDown={onGripDown}
          onPointerMove={onGripMove}
          onPointerUp={onGripUp}
          onDoubleClick={() => setScale(1)}
          title="drag to resize the graph · double-click for the default size"
          aria-label="resize the state graph"
          style={{
            position: "absolute",
            bottom: 2,
            ...(anchoredRight ? { left: 2 } : { right: 2 }),
            width: 14,
            height: 14,
            cursor: anchoredRight ? "nesw-resize" : "nwse-resize",
            touchAction: "none",
            opacity: 0.5,
            // two short rules in the corner — the usual grip, mirrored to
            // whichever corner is the live one
            background: `repeating-linear-gradient(${anchoredRight ? 45 : -45}deg, transparent 0 3px, ${DIM} 3px 4px)`,
          }}
        />
      )}
    </div>
  );
}
