// The two pure helpers the policy palette reads a run's RECORD through.
//
// The palette used to print the run directory verbatim
// ("teach-g1_imitate-g1_front_kick-25ac56-s2") and hand out a chain's last
// stage as "the whole trick". Both were wrong on their own terms: a user
// can't tell which run is the good one from a hash, and the stage that
// measured best is usually NOT the tail — the later stages are kept for
// comparison. These pin the fallbacks, because most runs have no record and
// must keep reading exactly as they always did.

import { describe, expect, it } from "vitest";
import { chainPick, policyTitle, type Policy } from "./lab";

function stage(label: string, extra: Partial<Policy> = {}): Policy {
  return {
    id: `runs:${label}`,
    label,
    group: "runs",
    path: `runs/${label}/policy.onnx`,
    ...extra,
  };
}

describe("policyTitle", () => {
  it("shows the record's title when the run has one", () => {
    const p = stage("teach-g1_imitate-g1_front_kick-25ac56-s2", {
      title: "Front kick (G1) · perform “g1-front-kick”",
    });
    expect(policyTitle(p)).toBe("Front kick (G1) · perform “g1-front-kick”");
  });

  it("falls back to the directory name when there is no record", () => {
    expect(policyTitle(stage("p-n256-s31"))).toBe("p-n256-s31");
  });

  it("strips the trainer's teach- prefix, like the chain header does", () => {
    expect(policyTitle(stage("teach-g1_front_kick-25ac56-s2"))).toBe(
      "g1_front_kick-25ac56-s2"
    );
  });

  it("leaves the raw label alone — it is what --init-from and the wire use", () => {
    const p = stage("teach-back-b18a5c-s5", { title: "Backflip" });
    policyTitle(p);
    expect(p.label).toBe("teach-back-b18a5c-s5");
  });
});

describe("chainPick", () => {
  it("returns the stage the record measured best, not the tail", () => {
    const stages = [stage("c-s1"), stage("c-s2", { pick: true }), stage("c-s3")];
    expect(chainPick(stages).label).toBe("c-s2");
  });

  it("returns the last stage when no stage was picked", () => {
    const stages = [stage("c-s1"), stage("c-s2"), stage("c-s3")];
    expect(chainPick(stages).label).toBe("c-s3");
  });

  it("returns a pick that sits first, where the tail default is furthest off", () => {
    const stages = [stage("c-s1", { pick: true }), stage("c-s2"), stage("c-s3")];
    expect(chainPick(stages).label).toBe("c-s1");
  });

  it("returns the only stage of a one-stage chain either way", () => {
    expect(chainPick([stage("c-s1")]).label).toBe("c-s1");
    expect(chainPick([stage("c-s1", { pick: true })]).label).toBe("c-s1");
  });
});
