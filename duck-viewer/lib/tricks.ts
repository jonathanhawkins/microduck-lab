// "Our runs", grouped by the trick each run practised.
//
// The palette listed every run newest-first, which after a seed battery is a
// wall of chips that all truncate to the same words ("Last metre, lan…" 24
// times) — a newcomer can't tell which one to use. Grouped by trick, each
// trick shows the measured pick and the newest run, and keeps the rest
// behind "N more". The lab says which recipe a run trained (`trick`,
// GET /policies); a lab too old to say so gets the flat list it always had.

import { chainPick, policyRobotId, type Policy } from "./lab";

/** "Our runs" rows: curriculum chains (same teach-…-<hash> prefix, -sN
 *  suffixes) fold into one family of compact stage chips, positioned by
 *  their NEWEST stage (the list arrives newest-first from the server, so
 *  first-seen = newest); everything else stays a single chip + time. */
export type RunRow =
  | { kind: "single"; p: Policy }
  | { kind: "chain"; chain: string; newest: number; stages: Policy[] };

export function runRows(list: Policy[]): RunRow[] {
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

export function rowKey(row: RunRow): string {
  return row.kind === "single" ? row.p.id : row.chain;
}

/** The policy a row hands out: a chain's measured pick (else its tail). */
export function rowPolicy(row: RunRow): Policy {
  return row.kind === "single" ? row.p : chainPick(row.stages);
}

function rowPicked(row: RunRow): boolean {
  return row.kind === "single" ? !!row.p.pick : row.stages.some((s) => s.pick);
}

function rowNewest(row: RunRow): number {
  return row.kind === "single" ? (row.p.mtime ?? 0) : row.newest;
}

/** The recipe a row practised. A chain's stages share one in practice; the
 *  handed-out stage speaks for the family, then any stage that knows. */
export function rowTrick(row: RunRow): string | null {
  if (row.kind === "single") return row.p.trick ?? null;
  return rowPolicy(row).trick ?? row.stages.find((s) => s.trick)?.trick ?? null;
}

export interface TrickGroup {
  /** Recipe id, or "" for runs the lab could not place. */
  trick: string;
  /** The rows shown unasked — the two a person ever wants: the newest run
   *  that holds a MEASURED pick, then the newest run. One row when those are
   *  the same, or when nothing was measured. Only a measurement outranks
   *  recency (the last run of a battery is not the best one, it is just the
   *  last), but the run that JUST finished training must not vanish behind
   *  "N more" because an older one carries a star. */
  lead: RunRow[];
  /** Every other row of the trick, newest first. */
  rest: RunRow[];
  newest: number;
}

/** Rows grouped by trick, in the order each trick was first seen (the rows
 *  arrive newest-first, so the trick worked on last leads). Null when NO row
 *  names a trick — an older lab — so the caller keeps the flat list instead
 *  of burying everything under one "other runs" heading. */
export function trickGroups(rows: RunRow[]): TrickGroup[] | null {
  if (!rows.some((r) => rowTrick(r))) return null;
  const groups = new Map<string, RunRow[]>();
  for (const row of rows) {
    const key = rowTrick(row) ?? "";
    const list = groups.get(key);
    if (list) list.push(row);
    else groups.set(key, [row]);
  }
  return [...groups].map(([trick, list]) => {
    const picked = list.find(rowPicked);
    const lead = picked && picked !== list[0] ? [picked, list[0]] : [list[0]];
    return {
      trick,
      lead,
      rest: list.filter((r) => !lead.includes(r)),
      newest: Math.max(...list.map(rowNewest)),
    };
  });
}

/** Split a label into a head that may be ellipsized and a tail that never
 *  is. The runs of a battery — and the tricks they belong to — share their
 *  opening words and differ at the END ("… (left, seed 2)", "… (right
 *  foot)"), so a plain tail ellipsis rendered them all identical. The caller
 *  lays the two out as flex items (head shrinks, tail doesn't), which fits
 *  whatever width the row really has: a fixed character budget was tried
 *  first and the chip turned out a third narrower than its 186 px cap. A
 *  label short enough to need no help comes back whole. */
export function splitTail(text: string, tail = 9, minLength = 16): [string, string] {
  const chars = [...text]; // by code point — never split an emoji in half
  if (chars.length <= minLength) return [text, ""];
  return [chars.slice(0, -tail).join(""), chars.slice(-tail).join("")];
}

/** The run a 🎓 chip's ▶ plays for a trick: the newest MEASURED pick of that
 *  recipe on that robot. Undefined when nothing was ever measured best —
 *  "watch the best one" must not quietly mean "watch the latest one". */
export function bestRunFor(policies: Policy[], trick: string, robot: string): Policy | undefined {
  if (!trick) return undefined;
  return policies.find(
    (p) => p.group === "runs" && p.pick && p.trick === trick && policyRobotId(p) === robot
  );
}
