// 🎥 capture store (select.ts pattern): the record flow's shared state.
// React consumers (RecordPanel) subscribe via useSyncExternalStore; per-frame
// consumers (the follow-cam inside the Canvas) read getCapture() directly.
//
// Phase machine: idle → framing (camera glides to the shot) → recording →
// processing (upload + server-side ffmpeg) → done | error → idle. The duck is
// captured by ID at start, so deselecting mid-take doesn't lose the shot.

import { useSyncExternalStore } from "react";

export type CapturePhase =
  | "idle"
  | "framing"
  | "recording"
  | "processing"
  | "done"
  | "error";

/** POST /captures response — mp4/gif are server paths under LAB_HTTP. */
export interface CaptureResult {
  name: string;
  mp4: string;
  gif: string;
  mp4Kb: number;
  gifKb: number;
  dir: string;
}

export interface CaptureState {
  phase: CapturePhase;
  duckId: string | null;
  /** Bumps on every start — the follow-cam re-picks its angle per epoch. */
  epoch: number;
  /** Date.now() when the recorder started rolling (0 otherwise). */
  recordingSince: number;
  result: CaptureResult | null;
  error: string | null;
}

const IDLE: CaptureState = {
  phase: "idle",
  duckId: null,
  epoch: 0,
  recordingSince: 0,
  result: null,
  error: null,
};

let state: CaptureState = IDLE;
const listeners = new Set<() => void>();

function set(next: Partial<CaptureState>) {
  state = { ...state, ...next };
  listeners.forEach((l) => l());
}

/** Frame-loop read — no subscription, no re-render. */
export const getCapture = (): CaptureState => state;

export function useCapture(): CaptureState {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => state,
    () => IDLE
  );
}

export function captureFraming(duckId: string) {
  set({
    phase: "framing",
    duckId,
    epoch: state.epoch + 1,
    recordingSince: 0,
    result: null,
    error: null,
  });
}
export function captureRecording() {
  if (state.phase === "framing")
    set({ phase: "recording", recordingSince: Date.now() });
}
export function captureProcessing() {
  set({ phase: "processing", recordingSince: 0 });
}
export function captureDone(result: CaptureResult) {
  set({ phase: "done", result });
}
export function captureError(error: string) {
  set({ phase: "error", error, recordingSince: 0 });
}
export function captureReset() {
  set({ phase: "idle", duckId: null, recordingSince: 0, result: null, error: null });
}

// The WebGL canvas to record — registered by the r3f tree (which owns it),
// read by RecordPanel's MediaRecorder. Only the canvas is captured, so DOM
// labels and panels never appear in the footage.
let canvas: HTMLCanvasElement | null = null;
export const setCaptureCanvas = (el: HTMLCanvasElement | null) => {
  canvas = el;
};
export const getCaptureCanvas = () => canvas;

// Frame pump. captureStream(fps)'s automatic capture rides the browser's own
// compositing, which goes quiet in throttled/backgrounded tabs and produced
// near-empty webms — so the recorder uses captureStream(0) and the r3f frame
// loop pushes a frame per RENDERED frame via requestFrame() (RecordCamera
// calls pumpCaptureFrame()). Null track = browser without requestFrame
// (Safari), which falls back to automatic capture.
interface RequestFrameTrack {
  requestFrame: () => void;
}
let track: RequestFrameTrack | null = null;
let framesPushed = 0;
export const setCaptureTrack = (t: RequestFrameTrack | null) => {
  track = t;
  framesPushed = 0;
};
export function pumpCaptureFrame() {
  if (state.phase !== "recording" || !track) return;
  track.requestFrame();
  framesPushed++;
}
/** Frames actually pushed this take (0 forever on the Safari fallback). */
export const getFramesPushed = () => framesPushed;
export const hasCaptureTrack = () => track !== null;

// 📷 snapshot — SYNCHRONOUS by design. Snapshotter (Viewer.tsx, which owns
// gl/scene/camera) registers the implementation here and the button's click
// handler calls it directly, so render → read → anchor-click all happen
// inside the user gesture: Chrome treats a download triggered from an async
// callback as "automatic" and silently drops a page's second one (the first
// async-flavored snapshot of a page load landed, the next never did).
let snapImpl: ((name: string) => void) | null = null;
export const setSnapshotFn = (fn: ((name: string) => void) | null) => {
  snapImpl = fn;
};
/** Take + download a PNG right now. False if the scene isn't mounted yet. */
export function snapshotNow(name: string): boolean {
  if (!snapImpl) return false;
  snapImpl(name);
  return true;
}

/** True on any frame that must render clean for capture (ring hidden).
 *  Recording only — 📷 snapshots hide rings themselves via the
 *  `userData.hideInCapture` tag, since they render mid-gesture, not from the
 *  frame loop. */
export function captureWantsCleanFrame(): boolean {
  return state.phase === "framing" || state.phase === "recording";
}
