// The ducks' MOUTHS, timed to their voices. A per-duck envelope — how open
// the bill is right now, 0 shut to 1 wide — driven by the notes the voice
// box is playing (lib/quackaudio.ts `voiceNotes`, the same table the sound
// is rendered from). Duck.tsx reads it once a frame and rotates the bill.
//
// Cosmetic, and viewer-side on purpose: the lab's own mouth servo is real
// physics (world/compose.py `split_jaw`, streamed as `mouth`) and a round
// trip through it would put the bill ~150 ms behind a 140 ms chirp. Here the
// bill and the sound share one clock, and a 🎥 take records both in step.
// The physics jaw still wins: Duck.tsx opens the bill to the LARGER of the
// two, so a duck holding a toy never clamps its mouth shut to chirp.

import { voiceNotes } from "./quackaudio";
import type { Voice } from "./quack";

/** How far the bill travels from shut to wide (world/compose.py:
 *  MOUTH_CLOSED -5° to MOUTH_OPEN 30°). An opening fraction times this is
 *  the hinge angle, for the voice exactly as for the servo. */
export const MOUTH_TRAVEL_RAD = (35 * Math.PI) / 180;

/** A flap opens fast — this long, or a quarter of the note if that is
 *  shorter — because the sound has already started when the bill parts. */
export const RISE_S = 0.03;
/** …and closes AFTER the note, over this long: a mouth that shut on the
 *  last sample would look like it bit the sound off. */
export const CLOSE_S = 0.07;

interface Flap {
  at: number;
  dur: number;
  peak: number;
}

/** The opening of one flap at `t` seconds after it started. */
export function flapOpen(dur: number, peak: number, t: number): number {
  if (t <= 0) return 0;
  const rise = Math.min(RISE_S, dur * 0.25);
  if (t < rise) return peak * (t / rise);
  const end = dur + CLOSE_S;
  if (t >= end) return 0;
  // Ease shut: the bill hangs open through the body of the note and closes
  // in the tail, the way a beak does — a straight line reads as a hinge.
  const u = (t - rise) / (end - rise);
  return peak * (1 - u * u);
}

/** One instance for the page (Duck.tsx reads it inside the frame loop,
 *  like `simRate`), fed by <DuckVoices> as it plays each voice. */
export class Mouths {
  private flaps = new Map<string, Flap[]>();

  /** This duck just started saying `kind`, at wall time `now`. */
  say(id: string, kind: Voice, now: number): void {
    const list = this.flaps.get(id) ?? [];
    // Notes carry their own loudness; the bill scales with it, so a coo is a
    // small opening and a chirp a wide one, off the one table.
    for (const n of voiceNotes(kind)) list.push({ at: now + n.at, dur: n.dur, peak: n.peak });
    this.flaps.set(id, list);
  }

  /** How open this duck's bill is right now, 0..1. Forgets finished flaps
   *  as it goes, so a page left open all day holds nothing. */
  open(id: string, now: number): number {
    const list = this.flaps.get(id);
    if (!list) return 0;
    let best = 0;
    let n = 0;
    for (const f of list) {
      if (now >= f.at + f.dur + CLOSE_S) continue;           // done: drop it
      list[n++] = f;
      best = Math.max(best, flapOpen(f.dur, f.peak, now - f.at));
    }
    list.length = n;
    if (n === 0) this.flaps.delete(id);
    // The widest flap, not their sum: two notes on top of each other are one
    // open mouth, and no note's peak is past 1, so nothing here can be.
    return best;
  }

  /** Ducks with a flap in flight — the tests' window on the map. */
  get talking(): number {
    return this.flaps.size;
  }
}

export const duckMouths = new Mouths();
