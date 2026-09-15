// The bill's timing against the voice: opens as the note starts, is widest
// for the loudest notes, shuts a beat after the sound, and a three-note
// reunion is three flaps, not one long gape.

import { describe, expect, it } from "vitest";

import { CLOSE_S, flapOpen, Mouths, RISE_S } from "./mouth";
import { SHAPES, voiceNotes } from "./quackaudio";

describe("flapOpen", () => {
  it("is shut before the note, opens within the rise, and shuts after the tail", () => {
    expect(flapOpen(0.14, 0.8, -0.01)).toBe(0);
    expect(flapOpen(0.14, 0.8, 0)).toBe(0);
    expect(flapOpen(0.14, 0.8, RISE_S)).toBeCloseTo(0.8, 6);
    expect(flapOpen(0.14, 0.8, 0.14 + CLOSE_S)).toBe(0);
    expect(flapOpen(0.14, 0.8, 0.14 + CLOSE_S + 1)).toBe(0);
  });

  it("stays mostly open through the body of the note and closes in the tail", () => {
    const mid = flapOpen(0.3, 1, 0.15), late = flapOpen(0.3, 1, 0.3), tail = flapOpen(0.3, 1, 0.3 + CLOSE_S * 0.8);
    expect(mid).toBeGreaterThan(0.75);
    expect(late).toBeGreaterThan(0.2);
    expect(late).toBeGreaterThan(tail);
    expect(tail).toBeGreaterThan(0);
  });

  it("never exceeds the note's peak", () => {
    for (let t = 0; t < 0.5; t += 0.005) expect(flapOpen(0.3, 0.6, t)).toBeLessThanOrEqual(0.6 + 1e-9);
  });
});

describe("Mouths", () => {
  it("is shut when nobody is talking, and holds nothing", () => {
    const m = new Mouths();
    expect(m.open("d0", 5)).toBe(0);
    expect(m.talking).toBe(0);
  });

  it("opens the bill as the chirp starts and shuts it after the note", () => {
    const m = new Mouths();
    m.say("d0", "chirp", 10);
    expect(m.open("d0", 10)).toBe(0);
    expect(m.open("d0", 10 + RISE_S)).toBeCloseTo(SHAPES.chirp.peak, 6);
    expect(m.open("d0", 10 + SHAPES.chirp.dur * 0.5)).toBeGreaterThan(0.5);
    expect(m.open("d0", 10 + SHAPES.chirp.dur + CLOSE_S + 0.01)).toBe(0);
    expect(m.talking).toBe(0);                                  // forgotten once done
  });

  it("opens wider for a chirp than for a coo, off the same table as the sound", () => {
    const m = new Mouths();
    m.say("d0", "chirp", 0);
    m.say("d1", "coo", 0);
    expect(m.open("d0", RISE_S)).toBeGreaterThan(m.open("d1", RISE_S));
    expect(m.open("d1", RISE_S)).toBeCloseTo(SHAPES.coo.peak, 6);
  });

  it("flaps three times for a reunion, dipping between the notes", () => {
    const m = new Mouths();
    m.say("d0", "reunion", 0);
    const notes = voiceNotes("reunion");
    expect(notes).toHaveLength(3);
    const trace: number[] = [];
    for (let t = 0; t < 0.6; t += 0.005) trace.push(m.open("d0", t));
    // Count the rises: a fresh flap opening is a strictly increasing run
    // that starts from a lower value than the last peak.
    let peaks = 0;
    for (let i = 1; i < trace.length - 1; i++) if (trace[i] > trace[i - 1] && trace[i] >= trace[i + 1]) peaks++;
    expect(peaks).toBe(3);
    // …and it is shut by the end.
    expect(trace[trace.length - 1]).toBe(0);
  });

  it("keeps each duck's mouth its own", () => {
    const m = new Mouths();
    m.say("d0", "question", 0);
    expect(m.open("d1", RISE_S)).toBe(0);
    expect(m.open("d0", RISE_S)).toBeGreaterThan(0);
  });

  it("takes the widest of overlapping notes, not their sum", () => {
    const m = new Mouths();
    m.say("d0", "chirp", 0);
    m.say("d0", "chirp", 0.01);
    m.say("d0", "question", 0.02);
    const widest = Math.max(SHAPES.chirp.peak, SHAPES.question.peak);
    for (let t = 0; t < 0.5; t += 0.005) expect(m.open("d0", t)).toBeLessThanOrEqual(widest + 1e-9);
  });
});
