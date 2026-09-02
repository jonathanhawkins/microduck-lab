"use client";

// Live view of brain training runs (train-brain). The lab's /brains endpoint
// reads the trainer's own artifacts off disk, so this page watches a run that
// knows nothing about it — including one started before the page was opened,
// and one already finished (its curve is still on disk).
//
// Layout: one viewport-height page that never scrolls itself. The run list is
// the only scrolling region, so the chart stays put however many runs pile up
// in brains/.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  BrainRun,
  CurvePoint,
  fetchBrains,
  humanDuration,
  humanSteps,
  runColor,
  smooth,
} from "@/lib/train";

const POLL_MS = 2000;

export default function TrainPanel() {
  const [runs, setRuns] = useState<BrainRun[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [metric, setMetric] = useState<"ep_rew" | "ep_len">("ep_rew");
  // A run the user explicitly picked stays pinned; otherwise follow whatever
  // is training right now, so opening the page mid-run shows the live one.
  const [pinned, setPinned] = useState<string | null>(null);
  const seeded = useRef(false);

  useEffect(() => {
    const ac = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let stopped = false;

    const tick = async () => {
      try {
        const b = await fetchBrains(ac.signal);
        if (stopped) return;
        setRuns(b);
        setErr(null);
        // First load: show the active run alone if there is one, else all.
        if (!seeded.current && b.length) {
          seeded.current = true;
          if (b.some((r) => r.active)) {
            setHidden(new Set(b.filter((r) => !r.active).map((r) => r.name)));
          }
        }
      } catch (e) {
        if (!stopped && (e as Error).name !== "AbortError") setErr((e as Error).message);
      } finally {
        if (!stopped) timer = setTimeout(tick, POLL_MS);
      }
    };
    tick();
    return () => {
      stopped = true;
      ac.abort();
      clearTimeout(timer);
    };
  }, []);

  const shown = useMemo(() => runs.filter((r) => !hidden.has(r.name)), [runs, hidden]);
  const live = useMemo(() => runs.filter((r) => r.active), [runs]);
  const focus = useMemo(
    () => runs.find((r) => r.name === pinned) ?? live[0] ?? shown[0] ?? runs[0],
    [runs, pinned, live, shown]
  );

  const toggle = useCallback((name: string) => {
    setHidden((h) => {
      const n = new Set(h);
      if (n.has(name)) n.delete(name);
      else n.add(name);
      return n;
    });
  }, []);

  const allOn = shown.length === runs.length;

  return (
    <main style={S.page}>
      <header style={S.header}>
        <Link href="/" style={S.back}>
          ← lab
        </Link>
        <span style={S.title}>/train</span>
        <span style={S.sub}>brain training · train-brain</span>
        <span style={{ flex: 1 }} />
        <span style={{ ...S.dot, background: err ? "#f87171" : live.length ? "#6ee7b7" : "#4b5563" }} />
        <span style={S.status}>
          {err ? `lab offline — ${err}` : live.length ? `${live.length} training` : "idle"}
        </span>
      </header>

      {runs.length === 0 && !err ? (
        <p style={S.empty}>
          No brain runs on disk yet. Start one and it appears here within a
          couple of seconds:
          <code style={S.code}>uv run train-brain --run-name follow-v4 --variety</code>
        </p>
      ) : (
        <div style={S.body}>
          <section style={S.listCol}>
            <div style={S.listHead}>
              <span style={S.colLabel}>
                runs · {shown.length}/{runs.length} charted
              </span>
              <span style={{ flex: 1 }} />
              <button
                style={S.tab}
                onClick={() =>
                  setHidden(allOn ? new Set(runs.map((r) => r.name)) : new Set())
                }
              >
                {allOn ? "none" : "all"}
              </button>
            </div>
            {/* The one scrolling region on the page. */}
            <div style={S.list}>
              {runs.map((r, i) => (
                <RunCard
                  key={r.name}
                  run={r}
                  color={runColor(r.name, i)}
                  hidden={hidden.has(r.name)}
                  focused={focus?.name === r.name}
                  onToggle={() => toggle(r.name)}
                  onFocus={() => setPinned(r.name)}
                />
              ))}
            </div>
          </section>

          <section style={S.chartCol}>
            <div style={S.chartHead}>
              <span style={S.colLabel}>
                {metric === "ep_rew" ? "episode reward" : "episode length"}
              </span>
              <span style={{ flex: 1 }} />
              {(["ep_rew", "ep_len"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMetric(m)}
                  style={{ ...S.tab, ...(metric === m ? S.tabOn : null) }}
                >
                  {m === "ep_rew" ? "reward" : "ep len"}
                </button>
              ))}
            </div>
            <Chart runs={shown} all={runs} metric={metric} />
            <p style={S.note}>
              The bold line is a 9-rollout trailing mean; the faint one is the raw
              per-rollout value. Hover the chart for exact values at a step. A run
              still climbing at its last point is undertrained, whatever its final
              number says.
            </p>
          </section>
        </div>
      )}
    </main>
  );
}

function RunCard({
  run,
  color,
  hidden,
  focused,
  onToggle,
  onFocus,
}: {
  run: BrainRun;
  color: string;
  hidden: boolean;
  focused: boolean;
  onToggle: () => void;
  onFocus: () => void;
}) {
  const pct = run.progress != null ? Math.round(run.progress * 100) : null;
  const tags = [
    run.variety ? "variety" : null,
    run.obs_version ? `obs v${run.obs_version}` : null,
    run.envs ? `${run.envs} envs` : null,
    run.seed != null ? `seed ${run.seed}` : null,
    run.fixed_preset ? run.fixed_preset : null,
  ].filter(Boolean) as string[];

  return (
    <div
      onClick={onFocus}
      style={{
        ...S.card,
        border: `1px solid ${focused ? color : "#1f2937"}`,
        opacity: hidden ? 0.45 : 1,
      }}
    >
      <div style={S.cardTop}>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onToggle();
          }}
          title={hidden ? "show on chart" : "hide from chart"}
          style={{
            ...S.swatch,
            background: hidden ? "transparent" : color,
            border: `1.5px solid ${color}`,
          }}
        />
        <span style={S.name}>{run.name}</span>
        {run.active && <span style={S.live}>● live</span>}
        {!run.active && run.shipped && <span style={S.ship}>shipped</span>}
      </div>

      {pct != null && (
        <div style={S.barTrack}>
          <div
            style={{
              ...S.barFill,
              width: `${pct}%`,
              background: color,
              opacity: run.active ? 1 : 0.5,
            }}
          />
        </div>
      )}

      {run.rollouts === 0 && (
        <p style={S.noCurve}>
          no training log on disk — only brain.onnx and brain.json are committed,
          so a brain from a clone has no curve to draw
        </p>
      )}

      <div style={S.stats}>
        <Stat k="steps" v={`${humanSteps(run.last?.steps)} / ${humanSteps(run.steps)}`} />
        <Stat k="reward" v={run.last ? run.last.ep_rew.toFixed(1) : "—"} />
        <Stat k="elapsed" v={humanDuration(run.last?.elapsed_s)} />
        {run.active ? (
          <Stat k="eta" v={humanDuration(run.eta_s)} />
        ) : (
          <Stat k="rollouts" v={String(run.rollouts)} />
        )}
        <Stat k="steps/s" v={run.steps_per_s != null ? String(run.steps_per_s) : "—"} />
      </div>

      {tags.length > 0 && (
        <div style={S.tags}>
          {tags.map((t) => (
            <span key={t} style={S.tag}>
              {t}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div style={S.stat}>
      <span style={S.statK}>{k}</span>
      <span style={S.statV}>{v}</span>
    </div>
  );
}

const W = 760;
const H = 380;
const PAD = { l: 56, r: 18, t: 16, b: 32 };

/** Nearest point to `steps` by x — the curve is downsampled, so snap to data. */
function nearest(pts: CurvePoint[], steps: number): CurvePoint | null {
  if (!pts.length) return null;
  let best = pts[0];
  let bd = Math.abs(best.steps - steps);
  for (const p of pts) {
    const d = Math.abs(p.steps - steps);
    if (d < bd) {
      bd = d;
      best = p;
    }
  }
  return best;
}

function Chart({
  runs,
  all,
  metric,
}: {
  runs: BrainRun[];
  all: BrainRun[];
  metric: "ep_rew" | "ep_len";
}) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null); // in viewBox units
  const series = useMemo(() => runs.filter((r) => r.curve.length > 1), [runs]);
  const val = useCallback(
    (p: CurvePoint) => (metric === "ep_rew" ? p.ep_rew : p.ep_len),
    [metric]
  );

  const bounds = useMemo(() => {
    let xMax = 0;
    let yMin = Infinity;
    let yMax = -Infinity;
    for (const r of series) {
      for (const p of r.curve) {
        if (p.steps > xMax) xMax = p.steps;
        const v = metric === "ep_rew" ? p.ep_rew : p.ep_len;
        if (v < yMin) yMin = v;
        if (v > yMax) yMax = v;
      }
    }
    if (!isFinite(yMin)) {
      yMin = 0;
      yMax = 1;
    }
    const pad = (yMax - yMin) * 0.08 || 1;
    return { xMax, yMin: yMin - pad, yMax: yMax + pad };
  }, [series, metric]);

  const x = useCallback(
    (s: number) => PAD.l + (s / (bounds.xMax || 1)) * (W - PAD.l - PAD.r),
    [bounds.xMax]
  );
  const y = useCallback(
    (v: number) =>
      H - PAD.b - ((v - bounds.yMin) / (bounds.yMax - bounds.yMin)) * (H - PAD.t - PAD.b),
    [bounds.yMin, bounds.yMax]
  );
  const invX = useCallback(
    (px: number) => ((px - PAD.l) / (W - PAD.l - PAD.r)) * (bounds.xMax || 1),
    [bounds.xMax]
  );

  const onMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const el = svgRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    // The SVG scales to its box; map client px into viewBox units.
    const vx = ((e.clientX - r.left) / r.width) * W;
    setHoverX(vx >= PAD.l && vx <= W - PAD.r ? vx : null);
  }, []);

  const readout = useMemo(() => {
    if (hoverX == null || !series.length) return null;
    const steps = invX(hoverX);
    // Anchor the rule to the LONGEST run's nearest sample. Anchoring it to
    // whichever run happens to be first snaps the whole crosshair back to
    // that run's final step as soon as it is shorter than the others — a
    // 24k-step probe would drag a 2M-step reading back with it.
    const anchor = series.reduce((a, b) =>
      (b.curve[b.curve.length - 1]?.steps ?? 0) > (a.curve[a.curve.length - 1]?.steps ?? 0) ? b : a
    );
    const anchorPt = nearest(anchor.curve, steps);
    if (!anchorPt) return null;

    const rows = series
      .map((r) => {
        const end = r.curve[r.curve.length - 1]?.steps ?? 0;
        const raw = nearest(r.curve, steps);
        const sm = nearest(smooth(r.curve), steps);
        if (!raw || !sm) return null;
        const i = all.findIndex((a) => a.name === r.name);
        return {
          name: r.name,
          color: runColor(r.name, i < 0 ? 0 : i),
          steps: raw.steps,
          raw: metric === "ep_rew" ? raw.ep_rew : raw.ep_len,
          mean: metric === "ep_rew" ? sm.ep_rew : sm.ep_len,
          elapsed: raw.elapsed_s,
          // This run had already stopped by the hovered step: its numbers are
          // its final ones, not a reading at this x. Say so instead of
          // presenting a stale value as if it were current.
          ended: steps > end * 1.02,
        };
      })
      .filter(Boolean) as {
      name: string;
      color: string;
      steps: number;
      raw: number;
      mean: number;
      elapsed: number;
      ended: boolean;
    }[];
    if (!rows.length) return null;
    return { rows, anchorPt, snapX: x(anchorPt.steps) };
  }, [hoverX, series, all, invX, x, metric]);

  if (!series.length) {
    return (
      <div style={{ ...S.chartBox, display: "grid", placeItems: "center", color: "#4b5563" }}>
        nothing selected
      </div>
    );
  }

  const yTicks = 4;
  const xTicks = 4;
  // Flip the tooltip to the left of the rule when it would overflow the box.
  const tipLeftPct = readout ? (readout.snapX / W) * 100 : 0;
  const flip = tipLeftPct > 62;

  return (
    <div style={S.chartBox}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        style={S.svg}
        role="img"
        aria-label="training curves"
        onMouseMove={onMove}
        onMouseLeave={() => setHoverX(null)}
      >
        {Array.from({ length: yTicks + 1 }, (_, i) => {
          const v = bounds.yMin + ((bounds.yMax - bounds.yMin) * i) / yTicks;
          return (
            <g key={i}>
              <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} stroke="#1f2937" strokeWidth={1} />
              <text x={PAD.l - 8} y={y(v) + 4} textAnchor="end" style={S.axis}>
                {v.toFixed(0)}
              </text>
            </g>
          );
        })}
        {Array.from({ length: xTicks + 1 }, (_, i) => {
          const s = (bounds.xMax * i) / xTicks;
          return (
            <text key={i} x={x(s)} y={H - 11} textAnchor="middle" style={S.axis}>
              {humanSteps(s)}
            </text>
          );
        })}

        {series.map((r) => {
          const i = all.findIndex((a) => a.name === r.name);
          const c = runColor(r.name, i < 0 ? 0 : i);
          const sm = smooth(r.curve);
          const last = sm[sm.length - 1];
          const path = (pts: CurvePoint[]) =>
            pts
              .map((p, j) => `${j ? "L" : "M"}${x(p.steps).toFixed(1)},${y(val(p)).toFixed(1)}`)
              .join("");
          return (
            <g key={r.name}>
              <path d={path(r.curve)} fill="none" stroke={c} strokeWidth={1} opacity={0.22} />
              <path
                d={path(sm)}
                fill="none"
                stroke={c}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
              <circle cx={x(last.steps)} cy={y(val(last))} r={r.active ? 4 : 2.5} fill={c} />
              {r.active && (
                <circle cx={x(last.steps)} cy={y(val(last))} r={4} fill="none" stroke={c}>
                  <animate attributeName="r" values="4;10;4" dur="1.6s" repeatCount="indefinite" />
                  <animate
                    attributeName="opacity"
                    values="0.9;0;0.9"
                    dur="1.6s"
                    repeatCount="indefinite"
                  />
                </circle>
              )}
            </g>
          );
        })}

        {/* Crosshair: a rule at the hovered step, with a dot on every series. */}
        {readout && (
          <g pointerEvents="none">
            <line
              x1={readout.snapX}
              x2={readout.snapX}
              y1={PAD.t}
              y2={H - PAD.b}
              stroke="#9ca3af"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            {readout.rows
              .filter((row) => !row.ended)
              .map((row) => (
                <circle
                  key={row.name}
                  cx={x(row.steps)}
                  cy={y(row.mean)}
                  r={3.5}
                  fill="#0b0f14"
                  stroke={row.color}
                  strokeWidth={2}
                />
              ))}
          </g>
        )}
      </svg>

      {readout && (
        <div
          style={{
            ...S.tip,
            left: `${tipLeftPct}%`,
            transform: flip ? "translate(calc(-100% - 12px), 0)" : "translate(12px, 0)",
          }}
        >
          <div style={S.tipStep}>{readout.anchorPt.steps.toLocaleString()} steps</div>
          {readout.rows.map((row) => (
            <div key={row.name} style={{ ...S.tipRow, opacity: row.ended ? 0.45 : 1 }}>
              <span style={{ ...S.tipDot, background: row.color }} />
              <span style={S.tipName}>{row.name}</span>
              {row.ended ? (
                <span style={S.tipEnded}>ended {humanSteps(row.steps)}</span>
              ) : (
                <>
                  <span style={S.tipMean}>{row.mean.toFixed(1)}</span>
                  <span style={S.tipRaw}>raw {row.raw.toFixed(1)}</span>
                </>
              )}
            </div>
          ))}
          <div style={S.tipFoot}>at {humanDuration(readout.anchorPt.elapsed_s)}</div>
        </div>
      )}
    </div>
  );
}

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";

const S: Record<string, React.CSSProperties> = {
  // Fixed to the viewport: the page itself never scrolls, only the run list.
  page: {
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    background: "#0b0f14",
    color: "#e5e7eb",
    fontFamily: mono,
    padding: 20,
  },
  header: { display: "flex", alignItems: "center", gap: 12, marginBottom: 16, flexShrink: 0 },
  back: { color: "#6b7280", fontSize: 12 },
  title: { fontSize: 15, fontWeight: 700, color: "#e5e7eb" },
  sub: { fontSize: 11, color: "#6b7280" },
  dot: { width: 8, height: 8, borderRadius: 8, display: "inline-block" },
  status: { fontSize: 11, color: "#9ca3af" },
  empty: { color: "#9ca3af", fontSize: 12, lineHeight: 1.9, maxWidth: 620 },
  code: {
    display: "block",
    marginTop: 8,
    padding: "8px 10px",
    background: "#111827",
    borderRadius: 6,
    color: "#6ee7b7",
    fontSize: 11,
  },
  // min-height:0 is what lets the flex children actually scroll instead of
  // stretching the page.
  body: { display: "flex", gap: 20, flex: 1, minHeight: 0 },
  listCol: { display: "flex", flexDirection: "column", width: 330, flexShrink: 0, minHeight: 0 },
  listHead: { display: "flex", alignItems: "center", gap: 6, marginBottom: 8, flexShrink: 0 },
  colLabel: { fontSize: 11, color: "#9ca3af", letterSpacing: 0.4 },
  list: {
    display: "flex",
    flexDirection: "column",
    gap: 10,
    overflowY: "auto",
    flex: 1,
    minHeight: 0,
    paddingRight: 6,
  },
  card: { borderRadius: 8, padding: "10px 12px", background: "#0f141b", cursor: "pointer", flexShrink: 0 },
  cardTop: { display: "flex", alignItems: "center", gap: 8, marginBottom: 8 },
  swatch: { width: 11, height: 11, borderRadius: 3, cursor: "pointer", padding: 0 },
  name: { fontSize: 12.5, fontWeight: 700 },
  live: { fontSize: 9.5, color: "#6ee7b7", letterSpacing: 0.4 },
  ship: { fontSize: 9.5, color: "#6b7280", letterSpacing: 0.4 },
  barTrack: { height: 3, background: "#1f2937", borderRadius: 3, overflow: "hidden", marginBottom: 9 },
  barFill: { height: "100%" },
  noCurve: { fontSize: 9.5, color: "#4b5563", lineHeight: 1.6, marginBottom: 9 },
  stats: { display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "7px 6px" },
  stat: { display: "flex", flexDirection: "column", gap: 1 },
  statK: { fontSize: 8.5, color: "#6b7280", textTransform: "uppercase", letterSpacing: 0.5 },
  statV: { fontSize: 11.5, color: "#e5e7eb" },
  tags: { display: "flex", flexWrap: "wrap", gap: 4, marginTop: 9 },
  tag: {
    fontSize: 9,
    color: "#9ca3af",
    background: "#111827",
    border: "1px solid #1f2937",
    borderRadius: 4,
    padding: "1px 5px",
  },
  chartCol: { flex: 1, minWidth: 420, display: "flex", flexDirection: "column", minHeight: 0 },
  chartHead: { display: "flex", alignItems: "center", gap: 6, marginBottom: 8, flexShrink: 0 },
  tab: {
    fontFamily: mono,
    fontSize: 10,
    color: "#9ca3af",
    background: "#111827",
    border: "1px solid #1f2937",
    borderRadius: 5,
    padding: "3px 9px",
    cursor: "pointer",
  },
  // Full shorthand, not borderColor: this object is spread OVER S.tab, and
  // mixing `border` with `borderColor` makes React warn on every rerender.
  tabOn: { color: "#0b0f14", background: "#6ee7b7", border: "1px solid #6ee7b7" },
  chartBox: {
    position: "relative",
    background: "#0f141b",
    border: "1px solid #1f2937",
    borderRadius: 8,
    flexShrink: 0,
  },
  svg: { width: "100%", height: "auto", display: "block", cursor: "crosshair" },
  axis: { fill: "#4b5563", fontSize: 9, fontFamily: mono },
  tip: {
    position: "absolute",
    top: 12,
    pointerEvents: "none",
    background: "rgba(11,15,20,0.96)",
    border: "1px solid #374151",
    borderRadius: 6,
    padding: "7px 9px",
    minWidth: 190,
    boxShadow: "0 6px 18px rgba(0,0,0,0.5)",
  },
  tipStep: { fontSize: 10, color: "#e5e7eb", marginBottom: 5, letterSpacing: 0.3 },
  tipRow: { display: "flex", alignItems: "center", gap: 6, marginTop: 3 },
  tipDot: { width: 7, height: 7, borderRadius: 7, flexShrink: 0 },
  tipName: { fontSize: 10, color: "#9ca3af", flex: 1, whiteSpace: "nowrap" },
  tipMean: { fontSize: 11, color: "#e5e7eb", fontWeight: 700 },
  tipRaw: { fontSize: 9, color: "#6b7280", whiteSpace: "nowrap" },
  tipEnded: { fontSize: 9.5, color: "#6b7280", whiteSpace: "nowrap" },
  tipFoot: { fontSize: 9, color: "#4b5563", marginTop: 5 },
  note: { fontSize: 10.5, color: "#6b7280", marginTop: 8, lineHeight: 1.7, maxWidth: 640 },
};
