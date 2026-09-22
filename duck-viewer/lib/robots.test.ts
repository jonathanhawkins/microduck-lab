import { describe, expect, it } from "vitest";
import { isDuck, robotCount, robotEmoji, robotLook, trickNoun } from "./robots";
import { robotChipLabel } from "./activeRobot";
import { OUR_GROUPS, robotTag, shippedGroups, type Policy } from "./lab";

describe("robotEmoji", () => {
  it("gives each built-in body its own face", () => {
    expect(robotEmoji("microduck")).toBe("🦆");
    expect(robotEmoji("g1")).toBe("🤖");
    expect(robotEmoji("mars")).toBe("🛸");
  });

  it("falls back to the KIND for a body it has never heard of", () => {
    // A plugin's wheeled body, or a second arm on a base: no entry here, and
    // it must still read as the kind of thing it is.
    expect(robotEmoji("some-cart", "wheeled")).toBe("🛸");
  });

  it("never returns an empty chip", () => {
    // A Menagerie model is `generic` — no id entry, no kind entry — and a
    // blank chip reads as a broken one rather than as an unknown robot.
    expect(robotEmoji("menagerie:unitree_go2", "generic")).toBe("🤖");
    expect(robotEmoji("")).toBe("🤖");
  });

  it("prefers the id over the kind when both would answer", () => {
    // MARS is `wheeled` and both tables say 🛸; the duck is `legged`, where
    // they disagree — and the id is the one that must win.
    expect(robotEmoji("microduck", "legged")).toBe("🦆");
  });
});

describe("robotLook", () => {
  it("names the three hand-built looks and sends everything else to generic", () => {
    expect(robotLook("microduck")).toBe("duck");
    expect(robotLook("")).toBe("duck"); // an older lab's rows carry no robot
    expect(robotLook("g1")).toBe("g1");
    // MARS was "generic" until 2026-09-18: the server's charcoal chassis sat
    // within a few percent of the lab stage's backdrop and only the orange
    // arm read (components/MarsLook.tsx).
    expect(robotLook("mars")).toBe("mars");
    expect(robotLook("menagerie:unitree_go2")).toBe("generic");
  });
});

describe("trickNoun", () => {
  it("is Innate's vocabulary for a non-legged body", () => {
    expect(trickNoun("legged")).toBe("trick");
    expect(trickNoun(undefined)).toBe("trick"); // an older lab sends no kind
    expect(trickNoun("wheeled")).toBe("task");
    expect(trickNoun("generic")).toBe("task");
  });
});

describe("robotChipLabel", () => {
  it("is the ONE label every switch shows", () => {
    // The palette, the 🎓 panel and the 🎬 panel each spelled this their own
    // way ("🤖 g1" beside "🤖 G1") before it had one home.
    expect(robotChipLabel({ id: "microduck", noun: "duck" })).toBe("🦆 duck");
    expect(robotChipLabel({ id: "g1", noun: "G1" })).toContain("G1");
    expect(robotChipLabel({ id: "mars", noun: "MARS" })).toContain("MARS");
  });
});

const chip = (id: string, group: string, robot: string): Policy =>
  ({ id, label: id, group, path: `/tmp/${id}`, robot }) as Policy;

describe("shippedGroups", () => {
  it("derives a lab's sections from its chips when the lab is too old to send them", () => {
    const groups = shippedGroups([
      chip("pollen:alpha_stand", "pollen", "microduck"),
      chip("run:my-run", "runs", "microduck"),
      chip("g1:walker", "g1", "g1"),
      chip("pollen:alpha_walking", "pollen", "microduck"),
      chip("ckpt:my-run@1k", "checkpoints", "microduck"),
    ]);
    expect(groups.map((g) => g.key)).toEqual(["pollen", "g1"]);
    expect(groups.map((g) => g.robot)).toEqual(["microduck", "g1"]);
  });

  it("leaves OUR sections out — they are not any body's drop", () => {
    expect(shippedGroups([chip("run:x", "runs", "microduck")])).toEqual([]);
    expect(OUR_GROUPS).toContain("runs");
    expect(OUR_GROUPS).toContain("checkpoints");
  });
});

describe("robotTag", () => {
  it("prefixes a non-duck chip in a mixed list", () => {
    expect(robotTag("mars", "runs")).toBe("mars · ");
    expect(robotTag("g1", "runs")).toBe("g1 · ");
    expect(robotTag("microduck", "runs")).toBe("");
  });

  it("is empty inside ANY shipped section, not just the G1's", () => {
    // A shipped group is one body's by construction, so repeating the body
    // on every chip is noise — and naming only "g1" here would have put a
    // "mars · " on every chip of a MARS drop.
    expect(robotTag("g1", "g1")).toBe("");
    expect(robotTag("mars", "mars")).toBe("");
  });
});

describe("robotCount", () => {
  it("pluralises a common noun and leaves a name alone", () => {
    // The nouns come from the lab (`robot_noun`), so this is a rule and not
    // a table: a plugin body brings its own noun with it. Case is the rule
    // — nobody writes "MARSs".
    expect(robotCount(1, "duck")).toBe("1 duck");
    expect(robotCount(3, "duck")).toBe("3 ducks");
    expect(robotCount(1, "MARS")).toBe("1 MARS");
    expect(robotCount(2, "MARS")).toBe("2 MARS");
    expect(robotCount(1, "G1")).toBe("1 G1");
    expect(robotCount(2, "G1")).toBe("2 G1");
    expect(robotCount(2, "so_arm100")).toBe("2 so_arm100s");
  });

  it("is what a scenario row reads as", () => {
    // The bug: `mars-follow` holds one MARS and no duck, and the picker
    // called it "1 ducks" — wrong number AND wrong animal.
    const mars = [{ id: "mars", n: 1, noun: "MARS" }];
    expect(mars.map((r) => robotCount(r.n, r.noun)).join(" + ")).toBe("1 MARS");
    const mixed = [{ id: "microduck", n: 2, noun: "duck" }, { id: "mars", n: 1, noun: "MARS" }];
    expect(mixed.map((r) => robotCount(r.n, r.noun)).join(" + ")).toBe("2 ducks + 1 MARS");
  });
});

describe("isDuck", () => {
  it("reads a missing robot as the duck, and nothing else as one", () => {
    // A scenario entry leaves `robot` off when it means the duck
    // (world/scenario.Duck defaults it), and an older lab sends no field at
    // all — so absent has to mean duck or every duck goes silent.
    expect(isDuck(undefined)).toBe(true);
    expect(isDuck(null)).toBe(true);
    expect(isDuck("")).toBe(true);
    expect(isDuck("microduck")).toBe(true);
    for (const r of ["mars", "g1", "unitree_go2", "trs_so_arm100"]) {
      expect(isDuck(r), r).toBe(false);
    }
  });
});
