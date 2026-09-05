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
