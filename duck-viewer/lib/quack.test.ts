// The ducks' voices: the schedule, not the sound. Everything here is about
// WHEN a quack comes out — the part a listener notices when it is wrong (a
// chord at world load, a machine-gun on a flickering track, chatter over a
// paused world).

import { describe, expect, it } from "vitest";

import {
  OWN_GAP_S, pitchFor, SPACING_S, voiceFor, Voices, WOBBLE,
  type VoiceDuck, type VoiceEvent,
} from "./quack";

const f = (id: string, state: string): VoiceDuck => ({ id, graph: "follow", state });

/** Run the director over a wall clock, with the sim clock advancing unless
 *  `frozen`. Returns every event with the time it came out. */
function run(
  v: Voices,
  ducks: (t: number) => VoiceDuck[],
  seconds: number,
  opts: { from?: number; step?: number; frozen?: boolean } = {},
): (VoiceEvent & { t: number })[] {
  const step = opts.step ?? 0.05;
  const from = opts.from ?? 0;
  const out: (VoiceEvent & { t: number })[] = [];
  for (let t = from; t <= from + seconds + 1e-9; t += step) {
    const simT = opts.frozen ? from : t;
    for (const e of v.update(ducks(t), t, simT)) out.push({ ...e, t });
  }
  return out;
}

describe("voiceFor", () => {
  it("gives the follow graph a voice per state", () => {
    expect(voiceFor("follow", "approach")?.kind).toBe("chirp");
    expect(voiceFor("follow", "hold")?.kind).toBe("coo");
    expect(voiceFor("follow", "search")?.kind).toBe("question");
    expect(voiceFor("follow", "coast")?.kind).toBe("question");
    expect(voiceFor("follow", "blocked")?.kind).toBe("grumble");
  });

  it("speaks more slowly while standing in the band than while walking in", () => {
    expect(voiceFor("follow", "hold")!.gap).toBeGreaterThan(voiceFor("follow", "approach")!.gap);
  });

  it("reads a learned follower's two observation labels", () => {
    expect(voiceFor("learned", "tracking")?.kind).toBe("chirp");
    expect(voiceFor("learned", "lost")?.kind).toBe("question");
    expect(voiceFor("learned", "learned")).toBeNull();   // before the first decision
  });

  it("leaves every other brain silent", () => {
    for (const [g, s] of [["chase", "kick"], ["tidy", "carry"], ["wander", "cruise"], ["script", "script"]]) {
      expect(voiceFor(g, s)).toBeNull();
    }
    expect(voiceFor("follow", "sleeping")).toBeNull();
    expect(voiceFor(null, "approach")).toBeNull();
    expect(voiceFor("follow", null)).toBeNull();
  });
});

describe("pitchFor", () => {
  it("is deterministic in the id and bounded", () => {
    for (const id of ["d0", "d1", "duck-7", ""]) {
      expect(pitchFor(id)).toBe(pitchFor(id));
      expect(pitchFor(id)).toBeGreaterThanOrEqual(0.8);
      expect(pitchFor(id)).toBeLessThanOrEqual(1.3);
    }
  });

  it("gives different ducks different voices", () => {
    const ps = ["d0", "d1", "d2", "d3"].map(pitchFor);
    expect(new Set(ps.map((p) => p.toFixed(3))).size).toBe(4);
  });
});

describe("Voices", () => {
  it("says nothing on the first tick, then chatters at about the line's gap", () => {
    const v = new Voices();
    const first = v.update([f("d0", "approach")], 0, 0);
    expect(first).toEqual([]);
    const evs = run(v, () => [f("d0", "approach")], 10, { from: 0.05 });
    expect(evs.every((e) => e.kind === "chirp")).toBe(true);
    // 10 s at a 1.6 s gap, jittered ±25 %: 5 to 8 of them.
    expect(evs.length).toBeGreaterThanOrEqual(5);
    expect(evs.length).toBeLessThanOrEqual(8);
    for (let i = 1; i < evs.length; i++) expect(evs[i].t - evs[i - 1].t).toBeGreaterThan(1.1);
  });

  it("staggers a flock instead of opening with a chord", () => {
    const flock = ["d0", "d1", "d2", "d3", "d4", "d5"].map((id) => f(id, "approach"));
    const v = new Voices();
    const evs = run(v, () => flock, 12);
    for (let i = 1; i < evs.length; i++) {
      expect(evs[i].t - evs[i - 1].t).toBeGreaterThanOrEqual(SPACING_S - 1e-9);
    }
    // …and every duck does get a turn.
    expect(new Set(evs.map((e) => e.id)).size).toBe(6);
    // The spread is the point, not just the 0.12 s spacing: six ducks queued
    // behind one timer would all speak inside the first second and read as
    // one squawk. Each duck's FIRST voice lands somewhere across its gap.
    const firsts = [...new Set(evs.map((e) => e.id))].map((id) => evs.find((e) => e.id === id)!.t);
    expect(Math.min(...firsts)).toBeGreaterThan(0.4);            // nobody speaks AT the load
    expect(Math.max(...firsts) - Math.min(...firsts)).toBeGreaterThan(0.6);
  });

  it("goes quiet while the world is frozen, and carries the timer across the pause", () => {
    const v = new Voices();
    const ducks = [f("d0", "approach")];
    // Warm it up to the tick it actually speaks on, so the pause starts with
    // a full gap left to run.
    let spoke = 0;
    for (let t = 0; t < 10; t += 0.05) {
      if (v.update(ducks, t, t).length) { spoke = t; break; }
    }
    expect(spoke).toBeGreaterThan(0);
    // Six wall seconds with the sim clock stopped (paused, scrubbing, socket
    // gone): nothing at all.
    const frozen: VoiceEvent[] = [];
    for (let t = spoke + 0.05; t < spoke + 6; t += 0.05) frozen.push(...v.update(ducks, t, spoke));
    expect(frozen).toEqual([]);
    // Resume: the gap it had left is the gap it still has — a pause is not a
    // charge-up. The shortest jittered gap is 1.2 s.
    const after: VoiceEvent[] = [];
    for (let t = spoke + 6; t < spoke + 7; t += 0.05) after.push(...v.update(ducks, t, t));
    expect(after).toEqual([]);
    const later: VoiceEvent[] = [];
    for (let t = spoke + 7; t < spoke + 10; t += 0.05) later.push(...v.update(ducks, t, t));
    expect(later.length).toBeGreaterThan(0);
  });

  it("calls out the moment it finds them again", () => {
    const v = new Voices();
    run(v, () => [f("d0", "search")], 4);
    const found = run(v, () => [f("d0", "approach")], 0.2, { from: 4.05 });
    expect(found.map((e) => e.kind)).toEqual(["reunion"]);
  });

  it("never speaks over itself, even to say it found them", () => {
    const v = new Voices();
    // Search until it quacks, past the dwell and the cooldown, then give the
    // person back on the very next tick — the collision the rule is for.
    let spoke = 0;
    for (let t = 0; t < 40; t += 0.05) {
      if (v.update([f("d0", "search")], t, t).length && t > 8) { spoke = t; break; }
    }
    expect(spoke).toBeGreaterThan(8);
    const back: (VoiceEvent & { t: number })[] = [];
    for (let t = spoke + 0.05; t < spoke + 2; t += 0.05) {
      for (const e of v.update([f("d0", "approach")], t, t)) back.push({ ...e, t });
    }
    expect(back[0].kind).toBe("reunion");
    expect(back[0].t - spoke).toBeGreaterThanOrEqual(OWN_GAP_S - 1e-9);
  });

  it("ignores a dropped detector frame — a reunion needs a real absence", () => {
    const v = new Voices();
    run(v, () => [f("d0", "approach")], 20);          // well past the cooldown
    // The detector blinks for a third of a second, as it does constantly.
    run(v, () => [f("d0", "coast")], 0.3, { from: 20.05 });
    const back = run(v, () => [f("d0", "approach")], 3, { from: 20.4 });
    expect(back.map((e) => e.kind)).not.toContain("reunion");
    expect(back.length).toBeGreaterThan(0);            // it does keep chirping
  });

  it("keeps the fanfare rare: not twice inside the cooldown", () => {
    const v = new Voices();
    run(v, () => [f("d0", "search")], 3);
    const first = run(v, () => [f("d0", "approach")], 0.2, { from: 3.05 });
    expect(first.map((e) => e.kind)).toEqual(["reunion"]);
    // A second genuine loss (longer than the dwell) inside the cooldown is
    // still only a chirp when they come back.
    run(v, () => [f("d0", "search")], 2, { from: 3.3 });
    const again = run(v, () => [f("d0", "approach")], 2, { from: 5.35 });
    expect(again.map((e) => e.kind)).not.toContain("reunion");
  });

  it("wobbles the pitch a little so it does not sound like a loop", () => {
    const v = new Voices();
    const evs = run(v, () => [f("d0", "approach")], 40);
    const base = pitchFor("d0");
    expect(new Set(evs.map((e) => e.pitch)).size).toBeGreaterThan(evs.length - 2);
    for (const e of evs) expect(Math.abs(e.pitch / base - 1)).toBeLessThanOrEqual(WOBBLE / 2 + 1e-9);
  });

  it("keeps a soccer pitch silent and holds no timers for it", () => {
    const v = new Voices();
    const pitch = ["d0", "d1", "d2"].map((id) => ({ id, graph: "chase", state: "chase" }));
    expect(run(v, () => pitch, 20)).toEqual([]);
    expect(v.tracked).toBe(0);
  });

  it("forgets the ducks a world reload took away", () => {
    const v = new Voices();
    run(v, () => [f("d0", "approach"), f("d1", "approach")], 5);
    expect(v.tracked).toBe(2);
    run(v, () => [f("d9", "approach")], 1, { from: 5.05 });
    expect(v.tracked).toBe(1);
  });
});
