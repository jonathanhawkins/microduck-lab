// The brain menu files learned runs under their use case; this is the
// arithmetic behind the headings (lib/sim.ts groupLearned).

import { describe, expect, it } from "vitest";
import { groupLearned, LEARNED_GROUPS, type LearnedInfo } from "./sim";

const b = (name: string, group: string | null, title: string | null = null): LearnedInfo => ({
  name, group, title, description: null,
});

describe("groupLearned", () => {
  it("files by group in menu order and skips empty groups", () => {
    const out = groupLearned([b("z1-s81", "null-pair"), b("follow-v4", "shipped-followers", "Follower v4"), b("z2-s81", "null-pair")]);
    expect(out.map(([label]) => label)).toEqual(["Followers (shipped)", "Null pair (seeds 81–84)"]);
    expect(out[1][1].map((x) => x.name)).toEqual(["z1-s81", "z2-s81"]);
  });

  it("files an unknown or missing group under Other, never drops it", () => {
    const out = groupLearned([b("striker-v1", null), b("odd", "no-such-group")]);
    expect(out).toEqual([["Other", [b("striker-v1", null), b("odd", "no-such-group")]]]);
  });

  it("returns nothing for nothing", () => {
    expect(groupLearned([])).toEqual([]);
    expect(LEARNED_GROUPS.at(-1)![0]).toBe("other");
  });
});

import { menuBrains } from "./sim";

describe("menuBrains", () => {
  const runs: LearnedInfo[] = [
    b("follow-v4", "shipped-followers", "Follower v4"),
    b("follow-v1", "shipped-followers", "Follower v1"),
    b("p-n256-s31", "capacity"), b("p-n256-s32", "capacity"),
    b("z1-s81", "null-pair"),
  ];

  it("offers only the shipped brains by default, and says how many it hid", () => {
    const m = menuBrains(runs, "wander", false);
    expect(m.groups.map(([label, bs]) => [label, bs.length])).toEqual([["Followers (shipped)", 2]]);
    expect(m.hidden).toBe(3);
  });

  it("never hides the brain the duck is on", () => {
    const m = menuBrains(runs, "learned:z1-s81", false);
    expect(m.groups.map(([label]) => label)).toEqual(["Followers (shipped)", "Null pair (seeds 81–84)"]);
    expect(m.hidden).toBe(2);
  });

  it("shows everything when asked", () => {
    const m = menuBrains(runs, null, true);
    expect(m.groups.length).toBe(3);
    expect(m.hidden).toBe(0);
  });
});
