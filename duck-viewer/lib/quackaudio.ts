// The ducks' voices — the SOUND half. Web Audio only: a triangle with a sine
// an octave up, a trill on the held notes, a rounding lowpass and a fast
// attack — a duckling's peep. No files to fetch, nothing to decode, ~40
// nodes a second at worst; the viewer
// already refuses fetched assets (duck-viewer/AGENTS.md) and a sample pack
// would be the heaviest thing in the app.
//
// WHEN a duck speaks is lib/quack.ts. This file only knows how to make the
// noise, which is why `renderVoice` takes any BaseAudioContext: the same
// code runs into an OfflineAudioContext for tuning and measuring.

import type { Voice } from "./quack";

/** One voice. `f` is the pitch in Hz at the start / middle / end of the note
 *  (multiplied by the duck's own pitch), `vibrato` a trill depth as a
 *  fraction of that pitch, `dur` the length, `peak` its loudness against the
 *  master gain.
 *
 *  The reference is a DUCKLING, not a mallard. The first version was a
 *  sawtooth through a sweeping bandpass — a fair quack, and measured as one
 *  — and the note that came back was "cuter". Cute, in a sound, is small:
 *  a fundamental around 800 Hz rather than 400, a rounded tone (a triangle
 *  with a sine an octave up for sparkle, through a gentle lowpass) instead
 *  of a buzz, sweeps that BOUNCE up rather than growl down, and a little
 *  trill on the held notes. Every contour below rises somewhere except the
 *  grumble, which is now a small deflating "aww" rather than a growl. */
export interface VoiceShape {
  f: [number, number, number];
  vibrato: number;
  dur: number;
  peak: number;
}

/** Trill rate for the voices that have one. Seven a second is a peep, not a
 *  wobble (four) and not a buzz (twelve). */
export const VIBRATO_HZ = 7;
/** The sine an octave above the triangle, as a fraction of it. */
const SPARKLE = 0.35;
/** The lowpass that rounds the triangle's edge off, before the duck's pitch. */
const ROUND_HZ = 3200;

/** The chirp is a quick bounce up and back (a "peep"), the coo a soft held
 *  note with the trill on it, the question a dip and then a big rise with a
 *  trill, the grumble a little sigh downward — softer than everything else. */
export const SHAPES: Record<Exclude<Voice, "reunion">, VoiceShape> = {
  // Peaks are set so the rendered notes measure 0.5–0.8 — the triangle plus
  // its octave sums higher than the old filtered sawtooth did, and a high
  // note already reads louder than a low one at the same level.
  chirp: { f: [700, 1150, 950], vibrato: 0, dur: 0.14, peak: 0.66 },
  coo: { f: [560, 600, 520], vibrato: 0.025, dur: 0.3, peak: 0.42 },
  question: { f: [640, 580, 1050], vibrato: 0.02, dur: 0.32, peak: 0.62 },
  grumble: { f: [520, 400, 300], vibrato: 0.015, dur: 0.28, peak: 0.34 },
};

/** A reunion — "there you are!" — is three quick chirps climbing: the one
 *  voice here that is a phrase rather than a note, because it marks the one
 *  moment worth marking. Three rising notes is the shape every happy little
 *  creature in every cartoon uses; two was a doorbell. */
const REUNION: [Exclude<Voice, "reunion">, number, number][] = [
  ["chirp", 0, 1],
  ["chirp", 0.11, 1.12],
  ["chirp", 0.22, 1.3],
];

/** The notes a voice is made of — when each starts, how long it lasts and
 *  how loud it peaks — as one list, so the bill (lib/mouth.ts) and the sound
 *  read the same score and can never drift apart. A reunion is three notes;
 *  everything else is one. */
export function voiceNotes(kind: Voice): { at: number; dur: number; peak: number }[] {
  if (kind === "reunion") return REUNION.map(([k, at]) => ({ at, dur: SHAPES[k].dur, peak: SHAPES[k].peak }));
  return [{ at: 0, dur: SHAPES[kind].dur, peak: SHAPES[kind].peak }];
}

/** Schedule one voice into `out` at time `at`. Returns when it ends. */
export function renderVoice(ctx: BaseAudioContext, out: AudioNode, kind: Voice, pitch = 1, at = 0): number {
  if (kind === "reunion") {
    let end = at;
    for (const [k, delay, mul] of REUNION) end = Math.max(end, renderVoice(ctx, out, k, pitch * mul, at + delay));
    return end;
  }
  const s = SHAPES[kind];
  const t0 = at;
  const mid = t0 + s.dur * 0.4;
  const t1 = t0 + s.dur;
  const stop = t1 + 0.02;

  const contour = (osc: OscillatorNode, mul: number) => {
    osc.frequency.setValueAtTime(s.f[0] * pitch * mul, t0);
    osc.frequency.linearRampToValueAtTime(s.f[1] * pitch * mul, mid);
    osc.frequency.linearRampToValueAtTime(s.f[2] * pitch * mul, t1);
  };
  // The body: a triangle — odd harmonics only and falling off fast, which
  // is a round tone, where the sawtooth was a buzz.
  const body = ctx.createOscillator();
  body.type = "triangle";
  contour(body, 1);
  // …and a sine an octave up, quiet, so it sparkles a little at the top.
  const spark = ctx.createOscillator();
  spark.type = "sine";
  contour(spark, 2);
  const sparkGain = ctx.createGain();
  sparkGain.gain.value = SPARKLE;

  const nodes: AudioNode[] = [body, spark, sparkGain];
  if (s.vibrato > 0) {
    // A trill: a slow sine, scaled to a few percent of the pitch, added to
    // the body's frequency. It rides the octave too, or the two drift apart.
    const lfo = ctx.createOscillator();
    lfo.frequency.value = VIBRATO_HZ;
    const depth = ctx.createGain();
    depth.gain.value = s.vibrato * s.f[1] * pitch;
    const depth2 = ctx.createGain();
    depth2.gain.value = 2;
    lfo.connect(depth);
    depth.connect(body.frequency);
    depth.connect(depth2).connect(spark.frequency);
    lfo.start(t0);
    lfo.stop(stop);
    nodes.push(lfo, depth, depth2);
  }

  const round = ctx.createBiquadFilter();
  round.type = "lowpass";
  round.Q.value = 0.7;
  round.frequency.value = ROUND_HZ * Math.sqrt(pitch);

  const env = ctx.createGain();
  // 12 ms attack — fast enough to read as a note starting, slow enough not
  // to click. Then a KNEE at 55 % of the note rather than one exponential
  // to the floor: a single ramp to 0.0008 is inaudible a third of the way
  // in, which measured as a 43 ms click where a 160 ms chirp was asked
  // for. The tail is exponential to a floor and then cut; Web Audio cannot
  // ramp to exactly zero.
  env.gain.setValueAtTime(0.0001, t0);
  env.gain.linearRampToValueAtTime(s.peak, t0 + 0.012);
  env.gain.exponentialRampToValueAtTime(s.peak * 0.35, t0 + s.dur * 0.55);
  env.gain.exponentialRampToValueAtTime(0.0008, t1);

  body.connect(round);
  spark.connect(sparkGain).connect(round);
  round.connect(env).connect(out);
  nodes.push(round, env);
  body.start(t0);
  spark.start(t0);
  body.stop(stop);
  spark.stop(stop);
  body.onended = () => {
    for (const n of nodes) n.disconnect();
  };
  return t1;
}

/**
 * The live voice box for the page: one AudioContext, one master gain, and a
 * tap the screen recorder can pull the same audio out of (`stream`), so the
 * quacks land in the mp4 the 🎥 button makes and not only in the room.
 *
 * Built lazily and resumed on demand: a context created before the user has
 * touched the page starts suspended (autoplay policy) and stays that way
 * until a gesture, so `resume()` is called both on the toggle that turns the
 * voices on and on the first click or key after that.
 */
export class DuckAudio {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private tap: MediaStreamAudioDestinationNode | null = null;
  private volume: number;
  /** Is anything currently feeding it? The context outlives a mute (so the
   *  recorder's tap and the browser's autoplay permission survive), so
   *  "running" alone does not mean the ducks are talking. */
  private active = false;

  constructor(volume = 0.22) {
    this.volume = volume;
  }

  /** The context, made on first use. Null where Web Audio is absent (SSR). */
  private ensure(): AudioContext | null {
    if (this.ctx) return this.ctx;
    if (typeof window === "undefined") return null;
    const Ctor = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return null;
    const ctx = new Ctor();
    const master = ctx.createGain();
    master.gain.value = this.volume;
    master.connect(ctx.destination);
    this.ctx = ctx;
    this.master = master;
    return ctx;
  }

  /** Start (or wake) the voices. Call it from a user gesture. */
  resume(): void {
    const ctx = this.ensure();
    if (ctx?.state === "suspended") void ctx.resume();
  }

  /** True once the browser is actually letting sound out. */
  get running(): boolean {
    return this.ctx?.state === "running";
  }

  /** The voices component holds this while it is mounted. */
  setActive(on: boolean): void {
    this.active = on;
  }
  get voicing(): boolean {
    return this.active && this.running;
  }

  play(kind: Voice, pitch = 1): void {
    const ctx = this.ensure();
    if (!ctx || !this.master || ctx.state !== "running") return;
    // A hair in the future: scheduling exactly at currentTime clips the
    // attack on a busy main thread.
    renderVoice(ctx, this.master, kind, pitch, ctx.currentTime + 0.01);
  }

  /** An audio track of these voices, for MediaRecorder. Made once, on the
   *  first take that asks for it, and kept for the life of the page. */
  captureTrack(): MediaStreamTrack | null {
    const ctx = this.ensure();
    if (!ctx || !this.master) return null;
    if (!this.tap) {
      this.tap = ctx.createMediaStreamDestination();
      this.master.connect(this.tap);
    }
    return this.tap.stream.getAudioTracks()[0] ?? null;
  }

  close(): void {
    const ctx = this.ctx;
    this.ctx = null;
    this.master = null;
    this.tap = null;
    void ctx?.close();
  }
}

// The page has ONE voice box, not one per mount: React remounts <DuckVoices>
// on every toggle and every hot reload, and a new AudioContext per mount both
// leaks the old one (Chrome caps a page at ~50) and drops the recorder's tap.
let shared: DuckAudio | null = null;
export function duckAudio(): DuckAudio {
  if (!shared) shared = new DuckAudio();
  return shared;
}

/** The page's voice box only if it already exists AND is making sound — what
 *  the screen recorder asks, so a take on a muted page (or the lab page,
 *  which has no voices at all) gets no silent audio track bolted on. */
export function liveDuckAudio(): DuckAudio | null {
  return shared?.voicing ? shared : null;
}
