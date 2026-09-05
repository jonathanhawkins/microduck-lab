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
import { loadJSON, saveJSON } from "@/lib/persist";
import { groupLearned } from "@/lib/sim";
import {
  BrainRun,
  CurvePoint,
  fetchBrains,
  humanDuration,
  humanSteps,
  displayName,
  matrixRows,
  type MatrixRow,
  recipeChips,
  recipeDiff,
  runColor,
  sharedKnobs,
  shippedStep,
  smooth,
  varyingKnobs,
} from "@/lib/train";

const POLL_MS = 2000;

export default function TrainPanel() {
  const [runs, setRuns] = useState<BrainRun[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [metric, setMetric] = useState<"ep_rew" | "ep_len">("ep_rew");
  // The right column is either the curves or the sweep matrix — the same
  // runs read two ways: over time, or across the knobs that changed.
  const [view, setView] = useState<"chart" | "matrix">("chart");
  // The card groups start folded: 49 cards under six headings is a wall, and
  // the headings alone already say what is on disk. What you open stays open.
  const [openGroups, setOpenGroups] = useState<Set<string>>(() => new Set(loadJSON<string[]>("trainGroupsOpen", [])));
  useEffect(() => saveJSON("trainGroupsOpen", [...openGroups]), [openGroups]);
  const toggleGroup = useCallback((label: string) => {
    setOpenGroups((g) => {
      const n = new Set(g);
      if (n.has(label)) n.delete(label);
      else n.add(label);
      return n;
    });
  }, []);
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
        // First load: chart ONE run, never the whole of brains/. Forty-odd
        // curves over the same axes is a solid band you cannot read — and
        // the palette only holds six colours, so they are not even
        // separable by eye. Start on the live run, else the first card, and
        // let the user click in the rest.
        if (!seeded.current && b.length) {
          seeded.current = true;
          const opening = b.filter((r) => r.active);
          if (!opening.length) opening.push(b[0]);
          const on = new Set(opening.map((r) => r.name));
          setHidden(new Set(b.filter((r) => !on.has(r.name)).map((r) => r.name)));
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
  // The first charted run is the baseline every other card diffs against:
  // it shows its whole recipe, the rest show only the knobs they changed.
  const baseline = shown[0] ?? null;
  // A run's colour is its position in the list. One lookup, built once per
  // poll, shared by the cards, the chart and the matrix.
  const colors = useMemo(() => new Map(runs.map((r, i) => [r.name, runColor(r.name, i)])), [runs]);
  const toggle = useCallback((name: string) => {
    setHidden((h) => {
      const n = new Set(h);
      if (n.has(name)) n.delete(name);
      else n.add(name);
      return n;
    });
  }, []);

  const toggleMany = useCallback((names: string[]) => {
    setHidden((h) => {
      const n = new Set(h);
      // Any member hidden → show them all; else hide them all.
      const show = names.some((x) => n.has(x));
      for (const x of names) if (show) n.delete(x); else n.add(x);
      return n;
    });
  }, []);

  const allOn = shown.length === runs.length;
  const noneOn = shown.length === 0;

  return (
    <main style={S.page}>
      <header style={S.header}>
        <Link href="/sim" style={S.back}>
          ← sim
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
              {/* Two buttons, not one flip-flop: a lone button reading
                  "none" cannot say whether that is the state or the thing
                  it does. These light up like the chart's metric tabs. */}
              <button
                style={{ ...S.tab, ...(allOn ? S.tabOn : null) }}
                title="chart every run"
                onClick={() => setHidden(new Set())}
              >
                all
              </button>
              <button
                style={{ ...S.tab, ...(noneOn ? S.tabOn : null) }}
                title="clear the chart — then add runs with their swatches"
                onClick={() => setHidden(new Set(runs.map((r) => r.name)))}
              >
                none
              </button>
            </div>
            {/* The one scrolling region on the page. */}
            <div style={S.list}>
              {/* Filed under the use case each run answers (brain.json `group`),
                  the same headings the /sim brain menu uses. */}
              {groupLearned(runs.map((r) => ({ name: r.name, title: r.title ?? null, group: r.group ?? null, description: null }))).map(
                ([label, members]) => {
                  const open = openGroups.has(label);
                  const charted = members.filter(({ name }) => !hidden.has(name)).length;
                  return (
                  <div key={label}>
                    <button
                      onClick={() => toggleGroup(label)}
                      aria-expanded={open}
                      style={S.groupHead}
                      title={open ? "fold this group" : "unfold this group"}
                    >
                      <span style={{ display: "inline-block", width: 12 }}>{open ? "▾" : "▸"}</span>
                      {label}
                      <span style={S.groupCount}>
                        {" "}· {members.length}{charted ? ` · ${charted} charted` : ""}
                      </span>
                    </button>
                    {open && members.map(({ name }) => {
                      const r = runs.find((x) => x.name === name)!;
                      return (
                        <RunCard
                          key={r.name}
                          run={r}
                          color={colors.get(r.name)!}
                          hidden={hidden.has(r.name)}
                          baseline={baseline}
                          onToggle={() => toggle(r.name)}
                        />
                      );
                    })}
                  </div>
                  );
                },
              )}
            </div>
          </section>

          <section style={S.chartCol}>
            <div style={S.chartHead}>
              <span style={S.colLabel}>
                {view === "matrix" ? "sweep matrix" : metric === "ep_rew" ? "episode reward" : "episode length"}
              </span>
              <span style={{ flex: 1 }} />
              {view === "chart" &&
                (["ep_rew", "ep_len"] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setMetric(m)}
                    style={{ ...S.tab, ...(metric === m ? S.tabOn : null) }}
                  >
                    {m === "ep_rew" ? "reward" : "ep len"}
                  </button>
                ))}
              <span style={{ width: 10 }} />
              {(["chart", "matrix"] as const).map((v) => (
                <button
                  key={v}
                  onClick={() => setView(v)}
                  style={{ ...S.tab, ...(view === v ? S.tabOn : null) }}
                  title={v === "matrix" ? "every run against the knobs that changed, with its benchmark score" : "reward over training steps"}
                >
                  {v}
                </button>
              ))}
            </div>
            {view === "chart" ? (
              <>
                <Chart runs={shown} colors={colors} metric={metric} />
                <p style={S.note}>
                  The bold line is a 9-rollout trailing mean; the faint one is the raw
                  per-rollout value. Hover the chart for exact values at a step. A run
                  still climbing at its last point is undertrained, whatever its final
                  number says.
                </p>
              </>
            ) : (
              <Matrix all={runs} colors={colors} hidden={hidden} onToggle={toggleMany} />
            )}
          </section>
        </div>
      )}
    </main>
  );
}

/**
 * A card IS the chart toggle — the whole thing, not just the swatch. It used
 * to carry a second selection ("focused", set by clicking the body) that
 * drew a border and did nothing else, so clicking a run lit it up without
 * ever putting it on the chart. One state, one meaning: bordered and bright
 * means charted.
 */
function RunCard({
  run,
  color,
  hidden,
  baseline,
  onToggle,
}: {
  run: BrainRun;
  color: string;
  hidden: boolean;
  baseline: BrainRun | null;
  onToggle: () => void;
}) {
  const pct = run.progress != null ? Math.round(run.progress * 100) : null;
  const isBase = baseline?.name === run.name;
  // What the chips say depends on the card's role. The baseline (and every
  // card while nothing is charted) shows its recipe; any other card shows
  // ONLY what it changed against the baseline — the answer to "what is
  // different about this one", which brain.json has always held and the
  // five fixed tags here used to hide (ab-batch vs ab-batch-lr is lr_end).
  const diffs = baseline && !isBase ? recipeDiff(run, baseline) : null;
  const recipe = recipeChips(run);
  const shippedAt = shippedStep(run);

  return (
    <div
      onClick={onToggle}
      title={hidden ? "add to the chart" : "remove from the chart"}
      style={{
        ...S.card,
        border: `1px solid ${hidden ? "#1f2937" : color}`,
        opacity: hidden ? 0.45 : 1,
      }}
    >
      <div style={S.cardTop}>
        <button
          onClick={(e) => {
            e.stopPropagation();   // the card handles it; don't toggle twice
            onToggle();
          }}
          title={hidden ? "add to the chart" : "remove from the chart"}
          style={{
            ...S.swatch,
            background: hidden ? "transparent" : color,
            border: `1.5px solid ${color}`,
          }}
        />
        <span style={S.name}>{displayName(run)}</span>
        {run.title && <span style={S.runId} title="the run's name on disk — what --init-from and learned:<name> address">{run.name}</span>}
        {run.active && <span style={S.live}>● live</span>}
        {!run.active && run.shipped && <span style={S.ship}>shipped</span>}
      </div>
      {run.description ? (
        <p style={S.desc}>{run.description}</p>
      ) : (
        <p style={{ ...S.desc, ...S.dim }} title={`uv run describe-brain ${run.name} --title ... --description ...`}>
          no description — describe-brain {run.name}
        </p>
      )}

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

      {run.selected && (
        <div style={S.shipped} title="select-brain probed every checkpoint on the follow benchmark and shipped the best one as brain.onnx — the number is that probe's score, not the curve's">
          <span style={{ color }}>◆</span> shipped from {shippedAt != null ? humanSteps(shippedAt) : run.selected.tag} ·{" "}
          {run.selected.metric} {run.selected.score.toFixed(3)}
          {run.selected.final_score !== run.selected.score && (
            <span style={S.dim}> (final {run.selected.final_score.toFixed(3)})</span>
          )}
        </div>
      )}

      <div style={S.tags}>
        {diffs === null ? (
          <>
            {isBase && (
              <span style={{ ...S.tag, ...S.tagBase, borderColor: color, color }} title="the first charted run — every other card shows what it changed against this one">
                baseline
              </span>
            )}
            {recipe.map((c) => (
              <span key={c.key} style={S.tag}>
                {c.text}
              </span>
            ))}
          </>
        ) : diffs.length === 0 ? (
          <span style={{ ...S.tag, ...S.dim }} title={`the same recipe as ${baseline!.name}`}>
            = {baseline!.name}
          </span>
        ) : (
          diffs.map((d) => (
            <span key={d.key} style={{ ...S.tag, ...S.tagDiff }} title={`${baseline!.name} has ${d.label} ${d.from}`}>
              {d.label} {d.value}
              <span style={S.dim}> ← {d.from}</span>
            </span>
          ))
        )}
      </div>
    </div>
  );
}

type SortCol = "name" | "score" | "final" | "shippedAt" | "reward" | `knob:${string}`;

/**
 * Every run against the knobs that differ between runs, with the benchmark
 * score of what each one shipped. Grouped by family (name minus seed) by
 * default, because that is the comparison that means anything: one seed's
 * in_band moves ±0.02 on its own, so a knob is only shown to matter by the
 * mean over seeds — and the ± column says how far to trust it.
 *
 * Rows are ALL runs, not just the charted ones; the swatch says which are
 * on the chart and clicking a row toggles it (a family row toggles every
 * seed), so the matrix doubles as the way to chart a whole sweep at once.
 */
function Matrix({
  all,
  colors,
  hidden,
  onToggle,
}: {
  all: BrainRun[];
  colors: Map<string, string>;
  hidden: Set<string>;
  onToggle: (names: string[]) => void;
}) {
  const [byFamily, setByFamily] = useState(true);
  const [sort, setSort] = useState<{ col: SortCol; dir: 1 | -1 }>({ col: "score", dir: -1 });
  const knobs = useMemo(() => varyingKnobs(all), [all]);
  const rows = useMemo(() => matrixRows(all, knobs, byFamily), [all, knobs, byFamily]);
  const sorted = useMemo(() => {
    // Knob columns sort on the RAW value, never the cell text: "1e-3" collates
    // before "3e-4" and "2.00M" before "750000", and a sorted lr column that
    // put the largest rate first was the bug. Booleans sort off < on.
    const v = (r: MatrixRow): unknown => {
      if (sort.col === "name") return r.key;
      if (sort.col.startsWith("knob:")) return r.raw[sort.col.slice(5)];
      return r[sort.col as "score" | "final" | "shippedAt" | "reward"];
    };
    const rank = (x: unknown): number | string | null => {
      if (x === null || x === undefined) return null;
      if (typeof x === "number") return x;
      if (typeof x === "boolean") return x ? 1 : 0;
      return String(x);
    };
    return [...rows].sort((a, b) => {
      const x = rank(v(a));
      const y = rank(v(b));
      if (x === null && y === null) return 0;
      if (x === null) return 1;   // blanks sink whatever the direction
      if (y === null) return -1;
      const c =
        typeof x === "number" && typeof y === "number"
          ? x - y
          : String(x).localeCompare(String(y), undefined, { numeric: true });
      return c * sort.dir;
    });
  }, [rows, sort]);
  const best = useMemo(() => Math.max(...rows.map((r) => r.score ?? -Infinity)), [rows]);
  // What every run shares — said once under the table instead of 49 times.
  const shared = useMemo(() => sharedKnobs(all), [all]);

  const click = (col: SortCol) =>
    setSort((s) => (s.col === col ? { col, dir: s.dir === 1 ? -1 : 1 } : { col, dir: col === "name" || col.startsWith("knob:") ? 1 : -1 }));
  // A plain render helper, not a component: a component declared inside
  // render is recreated every pass and React remounts it (lint: react-hooks).
  const th = (col: SortCol, label: string, title?: string, right = false) => (
    <th
      key={col}
      onClick={() => click(col)}
      title={title}
      style={{ ...S.th, textAlign: right ? "right" : "left", color: sort.col === col ? "#e5e7eb" : "#6b7280" }}
    >
      {label}
      {sort.col === col ? (sort.dir === 1 ? " ↑" : " ↓") : ""}
    </th>
  );

  return (
    <>
      <div style={S.matrixBox}>
        <table style={S.table}>
          <thead>
            <tr>
              {th("name", byFamily ? "family" : "run")}
              {/* Scores first: they are what you sort by, and with a wide
                  sweep the knobs run off the right edge — the numbers must not. */}
              {th("score", "in_band", "benchmark score of the checkpoint select-brain shipped — mean ± sd over the family's seeds", true)}
              {th("final", "final", "the same benchmark on the end-of-run model — how much selecting a checkpoint bought", true)}
              {th("shippedAt", "shipped@", "the step the shipped checkpoint came from", true)}
              {th("reward", "reward", "last training reward — the curve's number, not the benchmark's", true)}
              {knobs.map(([k, label]) => th(`knob:${k}`, label, `${label} differs between runs`))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => {
              const names = r.runs.map((x) => x.name);
              const on = names.filter((n) => !hidden.has(n)).length;
              const c = colors.get(names[0]) ?? "#9ca3af";
              const isBest = r.score != null && r.score === best;
              return (
                <tr
                  key={r.key}
                  onClick={() => onToggle(names)}
                  title={`${r.description ? r.description + "\n\n" : ""}${on === names.length ? "click: remove from the chart" : "click: add to the chart"}`}
                  style={{ ...S.tr, opacity: on ? 1 : 0.5 }}
                >
                  <td style={S.td}>
                    <span
                      style={{
                        ...S.swatch,
                        display: "inline-block",
                        verticalAlign: "middle",
                        marginRight: 6,
                        background: on === names.length ? c : on ? `linear-gradient(90deg, ${c} 50%, transparent 50%)` : "transparent",
                        border: `1.5px solid ${c}`,
                      }}
                    />
                    {r.label}
                    {r.label !== r.key && <span style={S.runId}>{r.key}</span>}
                    {byFamily && names.length > 1 && <span style={S.dim}> ×{names.length}</span>}
                  </td>
                  <td style={{ ...S.td, ...S.num, color: isBest ? "#6ee7b7" : "#e5e7eb", fontWeight: isBest ? 700 : 400 }}>
                    {r.score == null ? "—" : r.score.toFixed(3)}
                    {r.scoreSd != null && <span style={S.dim}> ±{r.scoreSd.toFixed(3)}</span>}
                  </td>
                  <td style={{ ...S.td, ...S.num }}>{r.final == null ? "—" : r.final.toFixed(3)}</td>
                  <td style={{ ...S.td, ...S.num }}>{r.shippedAt == null ? "—" : humanSteps(r.shippedAt)}</td>
                  <td style={{ ...S.td, ...S.num }}>{r.reward == null ? "—" : r.reward.toFixed(1)}</td>
                  {knobs.map(([k]) => (
                    <td key={k} style={{ ...S.td, color: r.raw[k] == null ? "#4b5563" : "#c9d0d8" }}>
                      {r.knobs[k]}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div style={{ ...S.chartHead, marginTop: 8, marginBottom: 0 }}>
        {(["by family", "by run"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setByFamily(m === "by family")}
            style={{ ...S.tab, ...(byFamily === (m === "by family") ? S.tabOn : null) }}
            title={m === "by family" ? "one row per name-minus-seed AND recipe, scores averaged over the seeds" : "one row per run"}
          >
            {m}
          </button>
        ))}
        <span style={{ flex: 1 }} />
        <span style={S.colLabel}>{rows.length} rows · click a header to sort, a row to chart it</span>
      </div>
      <p style={S.note}>
        Columns are the knobs that differ between runs; {shared.length > 0 && <>every run that records them shares {shared.join(" · ")}. </>}
        A — is a run from before the trainer wrote that knob down.{" "}
        in_band is the follow benchmark on the checkpoint that shipped, ± its spread over the family&apos;s seeds —
        one seed moves about ±0.02 on its own, so a knob has to beat that to have done anything. Runs that share a
        name but not a recipe are separate rows (&quot;p-de&quot;, &quot;p-de (2)&quot;), never averaged together.
      </p>
    </>
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
  colors,
  metric,
}: {
  runs: BrainRun[];
  colors: Map<string, string>;
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
        return {
          name: r.name,
          color: colors.get(r.name) ?? "#9ca3af",
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
  }, [hoverX, series, colors, invX, x, metric]);

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
          const c = colors.get(r.name) ?? "#9ca3af";
          const sm = smooth(r.curve);
          const last = sm[sm.length - 1];
          const shipAt = shippedStep(r);
          const ship = shipAt == null ? null : nearest(sm, shipAt);
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
              {/* Where brain.onnx actually came from. select-brain picks a
                  checkpoint by benchmark score, and it is routinely NOT the
                  end of the line — ab-batch ships step 751k of 2M — which
                  the curve alone never says. */}
              {ship && (
                <g>
                  <title>
                    {`${r.name}: shipped from ${humanSteps(ship.steps)} · ${r.selected!.metric} ${r.selected!.score.toFixed(3)}`}
                  </title>
                  <path
                    d={`M${x(ship.steps)},${y(val(ship)) - 6} l5,6 l-5,6 l-5,-6 z`}
                    fill="#0b0f14"
                    stroke={c}
                    strokeWidth={1.5}
                  />
                  <text x={x(ship.steps)} y={y(val(ship)) - 10} textAnchor="middle" style={{ ...S.axis, fill: c }}>
                    {r.selected!.score.toFixed(3)}
                  </text>
                </g>
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
  groupHead: {
    display: "block", width: "100%", textAlign: "left", background: "none", border: "none", cursor: "pointer",
    font: "inherit", fontSize: 9.5, color: "#9ca3af", letterSpacing: ".08em", textTransform: "uppercase", padding: "10px 2px 4px",
  },
  groupCount: { color: "#6b7280", letterSpacing: 0, textTransform: "none" },
  runId: { fontSize: 10, color: "#6b7280", fontFamily: "ui-monospace, Menlo, monospace", marginLeft: 6 },
  desc: { fontSize: 10.5, color: "#9ca3af", lineHeight: 1.4, margin: "0 0 8px" },
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
  // Full shorthand again (see tabOn): these are spread OVER S.tag.
  tagDiff: { color: "#e5e7eb", border: "1px solid #374151" },
  tagBase: { background: "transparent", fontWeight: 700 },
  dim: { color: "#6b7280" },
  shipped: { fontSize: 10, color: "#d1d5db", marginTop: 8 },
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
  matrixBox: {
    flex: 1,
    minHeight: 0,
    overflow: "auto",
    background: "#0f141b",
    border: "1px solid #1f2937",
    borderRadius: 8,
  },
  table: { borderCollapse: "collapse", width: "100%", fontSize: 10.5, whiteSpace: "nowrap" },
  th: {
    position: "sticky",
    top: 0,
    background: "#0f141b",
    padding: "6px 8px",
    fontWeight: 400,
    letterSpacing: 0.4,
    cursor: "pointer",
    borderBottom: "1px solid #1f2937",
    userSelect: "none",
  },
  tr: { cursor: "pointer", borderBottom: "1px solid #111827" },
  td: { padding: "4px 8px", color: "#c9d0d8" },
  num: { textAlign: "right", fontVariantNumeric: "tabular-nums" },
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
