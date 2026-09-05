// The card's "what differs from the baseline" line and the shipped-checkpoint
// marker rest on these two pure functions. Pinned against the shapes
// train-brain and select-brain actually write (brains/ab-batch/brain.json).

import { describe, expect, it } from "vitest";
import { type BrainRun, familyOf, fmtKnob, knobChip, matrixRows, recipeChips, recipeDiff, recipeKeys, sharedKnobs, shippedStep, varyingKnobs } from "./train";

const base: BrainRun = {
  name: "ab-batch",
  task: "follow",
  obs_version: 2,
  envs: 12,
  steps: 2_000_000,
  seed: 7,
  variety: true,
  fixed_preset: null,
  polite: 0.55,
  polite_mix: [],
  n_steps: 128,
  batch_size: 768,
  lr: 0.0003,
  lr_end: 0.0003,
  log_std_max: 10,
  legacy_hparams: false,
  shipped: true,
  active: false,
  rollouts: 1303,
  curve: [
    { steps: 1536, ep_rew: 40, ep_len: 300, elapsed_s: 1 },
    { steps: 751_104, ep_rew: 170, ep_len: 600, elapsed_s: 400 },
    { steps: 2_000_000, ep_rew: 171, ep_len: 600, elapsed_s: 1000 },
  ],
  selected: { tag: "000751104", metric: "in_band", score: 0.9376, final_score: 0.9235 },
};

describe("fmtKnob", () => {
  it("shows learning rates as short exponents and booleans as on/off", () => {
    expect(fmtKnob(0.0003)).toBe("3e-4");
    expect(fmtKnob(0.00003)).toBe("3e-5");
    expect(fmtKnob(0.55)).toBe("0.55");
    expect(fmtKnob(true)).toBe("on");
    expect(fmtKnob(null)).toBe("—");
    expect(fmtKnob([])).toBe("—");
    expect(fmtKnob(2_000_000)).toBe("2.00M");
  });
});

describe("recipeDiff", () => {
  it("is empty against itself and against an identical recipe", () => {
    expect(recipeDiff(base, base)).toEqual([]);
    expect(recipeDiff({ ...base, name: "twin" }, base)).toEqual([]);
  });

  it("names only the knobs that differ, in display order", () => {
    // ab-legacy vs ab-batch, per their brain.json files.
    const legacy = { ...base, name: "ab-legacy", batch_size: 1024, lr_end: 0.00003, legacy_hparams: true, log_std_max: null };
    expect(recipeDiff(legacy, base)).toEqual([
      { key: "batch_size", label: "batch", value: "1024", from: "768" },
      { key: "lr_end", label: "lr_end", value: "3e-5", from: "3e-4" },
      { key: "log_std_max", label: "log_std_max", value: "—", from: "10" },
      { key: "legacy_hparams", label: "legacy", value: "on", from: "off" },
    ]);
  });

  it("treats a missing key and null as the same value", () => {
    const noPreset = { ...base, name: "x" } as Partial<BrainRun>;
    delete noPreset.fixed_preset;
    expect(recipeDiff(noPreset as BrainRun, base)).toEqual([]);
  });

  it("diffs an unlisted scalar flag under its raw key, but never state fields", () => {
    const run = { ...base, name: "x", spot_prob: 0.3, rollouts: 5, active: true } as BrainRun;
    expect(recipeDiff(run, base)).toEqual([
      { key: "spot_prob", label: "spot_prob", value: "0.3", from: "—" },
    ]);
  });
});

describe("shippedStep", () => {
  it("reads a checkpoint tag as its step", () => {
    expect(shippedStep(base)).toBe(751_104);
  });
  it("puts 'final' at the end of the curve", () => {
    expect(shippedStep({ ...base, selected: { ...base.selected!, tag: "final" } })).toBe(2_000_000);
  });
  it("is null with no selection", () => {
    expect(shippedStep({ ...base, selected: undefined })).toBeNull();
  });
});

describe("sweep matrix", () => {
  const seeded = (name: string, seed: number, batch: number, score: number): BrainRun => ({
    ...base,
    name,
    seed,
    batch_size: batch,
    selected: { ...base.selected!, score, final_score: score - 0.01 },
  });
  const sweep = [
    seeded("p-n128-s31", 31, 768, 0.92),
    seeded("p-n128-s32", 32, 768, 0.94),
    seeded("p-n256-s31", 31, 1024, 0.90),
    seeded("p-n256-s32", 32, 1024, 0.96),
  ];

  it("strips the seed suffix to name the family", () => {
    expect(familyOf("p-n256-s34")).toBe("p-n256");
    expect(familyOf("ab-batch-lr")).toBe("ab-batch-lr");
    expect(familyOf("follow-v4")).toBe("follow-v4");
  });

  it("keeps only the knobs the runs disagree on, and none for a single run", () => {
    expect(varyingKnobs(sweep).map(([k]) => k)).toEqual(["seed", "batch_size"]);
    expect(varyingKnobs([sweep[0]])).toEqual([]);
  });

  it("ignores runs that do not record a knob — silence is not a value", () => {
    // follow-v1..v5 predate n_steps/batch/lr in brain.json. With them on
    // disk, every column lit up and the scores scrolled off the screen.
    const old = { ...sweep[0], name: "follow-v1" } as Partial<BrainRun>;
    delete old.n_steps;
    delete old.batch_size;
    expect(varyingKnobs([old as BrainRun, sweep[0]])).toEqual([]);
    expect(varyingKnobs([old as BrainRun, sweep[0], sweep[2]]).map(([k]) => k)).toEqual(["batch_size"]);
  });

  it("gives one row per run with its own numbers", () => {
    const rows = matrixRows(sweep, varyingKnobs(sweep), false);
    expect(rows.map((r) => r.key)).toEqual(sweep.map((r) => r.name));
    expect(rows[0].knobs).toEqual({ seed: "31", batch_size: "768" });
    expect(rows[0].score).toBeCloseTo(0.92);
    expect(rows[0].scoreSd).toBeNull();
    expect(rows[0].shippedAt).toBe(751_104);
  });

  it("collapses a family to mean ± sd over its seeds", () => {
    const rows = matrixRows(sweep, varyingKnobs(sweep), true);
    expect(rows.map((r) => r.key)).toEqual(["p-n128", "p-n256"]);
    const n256 = rows[1];
    expect(n256.runs).toHaveLength(2);
    expect(n256.knobs).toEqual({ seed: "×2", batch_size: "1024" });
    expect(n256.score).toBeCloseTo(0.93);
    expect(n256.scoreSd).toBeCloseTo(Math.sqrt(((0.9 - 0.93) ** 2 + (0.96 - 0.93) ** 2) / 1));
    expect(n256.final).toBeCloseTo(0.92);
  });

  it("splits a name-family whose seeds were run with different recipes", () => {
    // p-n128-s31 and -s32 share a name; -s32 was run with a different lr_end.
    // Averaging them would blend two experiments under one label.
    const odd = [sweep[0], { ...sweep[1], lr_end: 0.00003 }, sweep[2]];
    const rows = matrixRows(odd, varyingKnobs(odd), true);
    expect(rows.map((r) => r.key)).toEqual(["p-n128", "p-n128 (2)", "p-n256"]);
    expect(rows[0].knobs.lr_end).toBe("3e-4");
    expect(rows[1].knobs.lr_end).toBe("3e-5");
    expect(rows[1].runs.map((r) => r.name)).toEqual(["p-n128-s32"]);
  });

  it("sorts knob columns by value: raw carries the number, knobs the text", () => {
    const rows = matrixRows(sweep, varyingKnobs(sweep), true);
    const n256 = rows.find((r) => r.key === "p-n256")!;
    expect(n256.knobs.batch_size).toBe("1024");
    expect(n256.raw.batch_size).toBe(1024);
    expect(n256.raw.seed).toBe(2);
    const runs = [{ ...base, name: "a", lr_end: 0.001 }, { ...base, name: "b", lr_end: 0.0003 }];
    const byRun = matrixRows(runs, varyingKnobs(runs), false);
    expect(byRun.map((r) => r.raw.lr_end)).toEqual([0.001, 0.0003]);
    expect(byRun.map((r) => r.knobs.lr_end)).toEqual(["1e-3", "3e-4"]);
  });

  it("gives the matrix the same open-ended keys the cards diff", () => {
    // A striker sweep varies knobs KNOBS has never heard of.
    const a = { ...base, name: "kick-s1", w_progress: 1.0, spot_prob: 0.3 } as BrainRun;
    const b = { ...base, name: "kick-s2", w_progress: 2.0, spot_prob: 0.3 } as BrainRun;
    expect(recipeKeys([a, b]).map(([k]) => k)).toEqual(expect.arrayContaining(["spot_prob", "w_progress"]));
    expect(varyingKnobs([a, b]).map(([k]) => k)).toEqual(["w_progress"]);
    expect(recipeDiff(b, a).map((d) => d.key)).toEqual(["w_progress"]);
    expect(sharedKnobs([a, b])).toContain("spot_prob 0.3");
  });

  it("chips: one rule for the baseline card and the footer", () => {
    expect(knobChip("variety", true)).toBe("variety");
    expect(knobChip("variety", false)).toBeNull();
    expect(knobChip("polite_mix", [])).toBeNull();
    expect(knobChip("lr_end", 0.00003)).toBe("lr_end 3e-5");
    expect(recipeChips(base).map((c) => c.text)).toEqual(
      expect.arrayContaining(["task follow", "batch 768", "lr_end 3e-4", "variety", "polite 0.55"]),
    );
    expect(recipeChips(base).map((c) => c.text)).not.toContain("polite_mix —");
  });

  it("footer reads a shared knob from the first run that records it", () => {
    const old = { ...base, name: "follow-v1" } as Partial<BrainRun>;
    delete old.n_steps;
    // follow-v1 sorts first and never recorded n_steps; the others all agree.
    expect(sharedKnobs([old as BrainRun, sweep[0], sweep[1]])).toContain("n_steps 128");
  });

  it("does not read a member's silence as a disagreement", () => {
    // The seeds of one family agree about variety; the older one predates
    // the flag and simply omits it. varyingKnobs already ignores silence —
    // the cells have to as well, or the family reads "≠" for a knob nobody
    // changed. The third run is only there to make variety a column.
    const older = { ...sweep[0], name: "fam-s1" } as Partial<BrainRun>;
    delete older.variety;
    const runs = [
      older as BrainRun,
      { ...sweep[1], name: "fam-s2", variety: true },
      { ...sweep[2], name: "other-s1", variety: false },
    ];
    const rows = matrixRows(runs, varyingKnobs(runs), true);
    const fam = rows.find((r) => r.key === "fam")!;
    expect(fam.runs).toHaveLength(2);
    expect(fam.knobs.variety).toBe("on");
  });

  it("shows — for a knob no member of the family recorded", () => {
    const a = { ...sweep[0], name: "old-s1" } as Partial<BrainRun>;
    const b = { ...sweep[1], name: "old-s2" } as Partial<BrainRun>;
    delete a.variety;
    delete b.variety;
    // Two NEWER runs disagreeing is what makes variety a column at all; the
    // old family recorded it neither way, and must read as a blank.
    const runs = [
      a as BrainRun,
      b as BrainRun,
      { ...sweep[2], name: "new-s1", variety: true },
      { ...sweep[3], name: "new-s2", variety: false },
    ];
    const knobs = varyingKnobs(runs);
    expect(knobs.map(([k]) => k)).toContain("variety");
    const fam = matrixRows(runs, knobs, true).find((r) => r.key === "old")!;
    expect(fam.knobs.variety).toBe("—");
  });
});
