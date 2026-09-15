// The ducks' VOICES — the arithmetic half: which duck says something this
// tick, what it says, and in which voice. The sound itself is
// lib/quackaudio.ts (Web Audio, no assets); the wiring is <DuckVoices> in
// components/SimViewer.tsx.
//
// Scoped to the FOLLOWERS on purpose. A duck chirping while it walks after a
// person is the thing people want to hear; eighteen chase states quacking
// over a 3v3 is noise, and a soccer battery running all afternoon behind a
// browser tab is worse. So the table below is keyed by the state GRAPH a
// brain draws (brain/graph.py: "follow" for the scripted baseline, "learned"
// for the trained followers) and nothing else makes a sound.
//
// Times are WALL seconds, not sim seconds — a quack is heard by a person, so
// it is scheduled in their clock. The world's own clock only decides whether
// it is running at all (`update` goes quiet when the sim stalls: paused,
// scrubbing, a dropped socket), so a frozen picture is a silent one.

/** What a duck can say. */
export type Voice = "chirp" | "coo" | "question" | "grumble" | "reunion";

export interface VoiceLine {
  kind: Voice;
  /** Seconds between two of these, before jitter. */
  gap: number;
}

/** Voices that mean "I can see them" — the other half of a reunion. */
const HAPPY: ReadonlySet<Voice> = new Set<Voice>(["chirp", "coo", "reunion"]);
export const isHappy = (v: Voice) => HAPPY.has(v);

/** State → line, per brain graph. The state names are the graph's nodes
 *  (brain/graph.py FOLLOW and LEARNED); a node that gains a name upstream
 *  and is missing here is simply silent, never a crash.
 *
 *  The gaps are what a duck sounds like rather than a metronome: chatter
 *  while it is walking after someone (1.6 s), a slower contented note once
 *  it is in the band and standing (3.2 s), and a questioning quack while it
 *  has lost them, spaced far enough apart to read as "where did you go" and
 *  not as an alarm. */
const LINES: Record<string, Record<string, VoiceLine>> = {
  follow: {
    approach: { kind: "chirp", gap: 1.6 },
    hold: { kind: "coo", gap: 3.2 },
    search: { kind: "question", gap: 2.4 },
    coast: { kind: "question", gap: 2.4 },
    blocked: { kind: "grumble", gap: 2.6 },
    dodge: { kind: "grumble", gap: 2.6 },
  },
  // A learned follower has no states — these two labels are one slot of its
  // observation ("is the target visible"), which is exactly the distinction
  // the voice wants.
  learned: {
    tracking: { kind: "chirp", gap: 1.6 },
    lost: { kind: "question", gap: 2.4 },
  },
};

/** The line a duck on this graph and state says, or null for silence. */
export function voiceFor(graph: string | null | undefined, state: string | null | undefined): VoiceLine | null {
  if (!graph || !state) return null;
  return LINES[graph]?.[state] ?? null;
}

/** Per-duck pitch, so six ducks are six ducks and not one duck six times.
 *  Deterministic in the id: d0 sounds like d0 across reloads and across the
 *  recording you make of it. */
export function pitchFor(id: string): number {
  return 0.84 + (hash(id) % 1000) / 1000 * 0.42;
}

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** mulberry32 — a tiny deterministic PRNG, one per duck, so the jitter is
 *  reproducible (a test can assert a schedule) without every duck drawing
 *  from the same stream and drifting into unison. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface VoiceDuck {
  id: string;
  /** SimDuck.brain.graph and .state. */
  graph?: string | null;
  state?: string | null;
}

export interface VoiceEvent {
  id: string;
  kind: Voice;
  pitch: number;
}

/** A sim clock that has not moved for this long is stopped — paused, being
 *  scrubbed, or the socket is gone. Longer than a frame interval at the
 *  slowest speed the lab serves, short enough that hitting pause silences
 *  the ducks while your finger is still on the key. */
export const STALL_S = 0.3;
/** Never two voices closer together than this. Six ducks whose timers land
 *  on the same tick (they all start following at a world load) read as one
 *  loud squawk; spread, they read as a flock. */
export const SPACING_S = 0.12;
/** "There you are!" fires at most this often per duck. */
export const REUNION_COOLDOWN_S = 5;
/** …and only after the duck has actually been without them this long. The
 *  detector drops a frame or two constantly — measured against a live
 *  follow-me lab, a duck that held its person for 95 % of 30 s still went
 *  "lost" in 39 frames scattered through it, which fired four fanfares in
 *  half a minute and made the one moment worth marking routine. */
export const REUNION_MIN_LOST_S = 1.5;
/** Per-utterance pitch spread, as a fraction of the duck's own pitch. */
export const WOBBLE = 0.08;
/** A duck never speaks over itself: its own voices are at least this far
 *  apart, even when a reunion is jumping the queue. */
export const OWN_GAP_S = 0.6;

/**
 * The director: hand it the ducks and the clock, take back the voices to
 * play. Holds one timer per duck and no audio at all, which is what makes
 * it testable — the Web Audio half has no decisions in it.
 */
export class Voices {
  private due = new Map<string, number>();
  private said = new Map<string, Voice>();
  private ownVoiceAt = new Map<string, number>();
  private lostSince = new Map<string, number>();
  private pending = new Set<string>();
  private reunionAt = new Map<string, number>();
  private jitter = new Map<string, () => number>();
  private lastVoiceAt = -Infinity;
  private lastUpdate: number | null = null;
  private lastSimT: number | null = null;
  private simMovedAt = 0;

  /**
   * @param ducks the live ducks, with the graph and state their brain reports
   * @param now   wall seconds (any monotonic origin)
   * @param simT  the world's own clock, to tell running from frozen
   */
  update(ducks: VoiceDuck[], now: number, simT: number): VoiceEvent[] {
    const dt = this.lastUpdate === null ? 0 : Math.max(0, now - this.lastUpdate);
    this.lastUpdate = now;
    if (simT !== this.lastSimT) {
      this.lastSimT = simT;
      this.simMovedAt = now;
    }
    if (now - this.simMovedAt > STALL_S) {
      // Frozen world: carry every timer forward with wall time rather than
      // letting them all come due while it is stopped — unpausing would
      // otherwise fire the whole flock at once.
      for (const [id, t] of this.due) this.due.set(id, t + dt);
      this.lastVoiceAt += dt;
      return [];
    }

    const out: VoiceEvent[] = [];
    const live = new Set<string>();
    for (const d of ducks) {
      const line = voiceFor(d.graph, d.state);
      if (!line) {
        // Not a follower (or a brain between decisions): forget it, so it
        // re-staggers rather than firing the moment it becomes one.
        this.due.delete(d.id);
        this.said.delete(d.id);
        continue;
      }
      live.add(d.id);
      this.said.set(d.id, line.kind);
      // Has it just found them again, after being without them long enough
      // for that to mean anything? The answer is held as a PENDING fanfare
      // rather than fired on the spot, because it still has to wait its turn
      // behind this duck's own last voice.
      if (!isHappy(line.kind)) {
        if (!this.lostSince.has(d.id)) this.lostSince.set(d.id, now);
        this.pending.delete(d.id);
      } else {
        const lost = this.lostSince.get(d.id);
        if (lost !== undefined) {
          this.lostSince.delete(d.id);
          if (now - lost >= REUNION_MIN_LOST_S
            && now - (this.reunionAt.get(d.id) ?? -Infinity) >= REUNION_COOLDOWN_S) this.pending.add(d.id);
        }
      }
      const rand = this.jitterFor(d.id);
      if (!this.due.has(d.id)) {
        // First sight of this duck: somewhere inside the first gap, so a
        // freshly loaded room does not open with a chord.
        this.due.set(d.id, now + line.gap * (0.35 + 0.9 * rand()));
        continue;
      }
      // Found them again: say so at once, whatever the timer said — this is
      // the moment the whole feature is for.
      const reunion = this.pending.has(d.id)
        && now - (this.ownVoiceAt.get(d.id) ?? -Infinity) >= OWN_GAP_S;
      if (!reunion && now < this.due.get(d.id)!) continue;
      if (now - this.lastVoiceAt < SPACING_S) {
        // Someone else just spoke — wait a beat and try again next tick.
        this.due.set(d.id, this.lastVoiceAt + SPACING_S);
        continue;
      }
      // A few percent of pitch wobble per utterance, on top of the duck's own
      // voice: a chirp every 1.6 s at exactly the same pitch reads as a loop
      // playing, which is the one thing that would give the illusion away.
      out.push({
        id: d.id,
        kind: reunion ? "reunion" : line.kind,
        pitch: pitchFor(d.id) * (1 + WOBBLE * (rand() - 0.5)),
      });
      this.lastVoiceAt = now;
      this.ownVoiceAt.set(d.id, now);
      if (reunion) {
        this.reunionAt.set(d.id, now);
        this.pending.delete(d.id);
      }
      this.due.set(d.id, now + line.gap * (0.75 + 0.5 * rand()));
    }
    // A world reload replaces the roster; do not keep timers for ducks that
    // no longer exist (the /sim page loads scenarios all day).
    for (const id of [...this.due.keys()]) if (!live.has(id)) this.forget(id);
    for (const id of [...this.said.keys()]) if (!live.has(id)) this.forget(id);
    return out;
  }

  private forget(id: string) {
    this.due.delete(id);
    this.said.delete(id);
    this.ownVoiceAt.delete(id);
    this.lostSince.delete(id);
    this.pending.delete(id);
    this.reunionAt.delete(id);
    this.jitter.delete(id);
  }

  private jitterFor(id: string): () => number {
    let r = this.jitter.get(id);
    if (!r) {
      r = rng(hash(id));
      this.jitter.set(id, r);
    }
    return r;
  }

  /** Ducks currently holding a timer — the test's window on the map, and the
   *  proof a world reload does not leak them. */
  get tracked(): number {
    return this.due.size;
  }
}
