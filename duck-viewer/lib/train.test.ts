// The card's "what differs from the baseline" line and the shipped-checkpoint
// marker rest on these two pure functions. Pinned against the shapes
// train-brain and select-brain actually write (brains/ab-batch/brain.json).

import { describe, expect, it } from "vitest";
import { type BrainRun, fmtKnob, recipeDiff, shippedStep } from "./train";

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
