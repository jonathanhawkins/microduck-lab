"use client";

// 🎥 the take itself, shared by the lab page's RecordPanel and the /sim
// toolbar's SimRecord: MediaRecorder on the registered WebGL canvas
// (lib/record.ts), an optional framing delay first (the lab page glides its
// camera to the duck; /sim records whatever the user is looking at), then
// the upload to POST /captures where the lab's ffmpeg makes mp4 + gif.
// Phase state lives in the lib/record.ts store so the in-canvas helpers
// (frame pump, follow-cam, clean-frame rule) see it without props.

import { useEffect, useRef, useState } from "react";
import { LAB_HTTP } from "@/lib/lab";
import { liveDuckAudio } from "@/lib/quackaudio";
import {
  captureDone,
  captureError,
  captureFraming,
  captureProcessing,
  captureRecording,
  captureReset,
  getCapture,
  getCaptureCanvas,
  getFramesPushed,
  hasCaptureTrack,
  setCaptureTrack,
  snapshotNow,
  useCapture,
  type CaptureState,
} from "@/lib/record";
import { pushToast } from "./Toasts";

/** Safety cap: a forgotten recorder must not fill the disk. */
export const MAX_TAKE_MS = 60_000;

/** Duck / scenario names carry emoji, spaces and dashes — reduce to a safe
 *  filename stem (mirrors the server's capture_slug, so shots and takes
 *  sort together). */
export function slug(s: string): string {
  return (
    s
      .replace(/[^A-Za-z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .toLowerCase()
      .slice(0, 40)
      // Mirror capture_slug exactly: it strips again AFTER truncating, and
      // strips leading _ as well (a stem starting with _ is one the server's
      // own /captures route then refuses to serve). Without both, a 📷 shot
      // and a 🎥 take of the same duck get different stems.
      .replace(/-+$/, "")
      .replace(/^[_-]+/, "") || "duck"
  );
}

export function stamp(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}` +
    `-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
  );
}

function pickMime(audio: boolean): string | null {
  if (typeof MediaRecorder === "undefined") return null;
  // With the ducks talking the container carries a second stream, so the
  // codec string has to name an audio codec too — a video-only mime with an
  // audio track in the stream is up to the browser, and Chrome has shipped
  // both "encode it as opus anyway" and "drop it".
  const prefs = audio
    ? ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm", "video/mp4"]
    : ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];
  return prefs.find((m) => MediaRecorder.isTypeSupported(m)) ?? null;
}

export interface Take {
  cap: CaptureState;
  /** Seconds the recorder has been rolling (0 unless recording). */
  secs: number;
  /** Begin a take; `subject` is what the server names the file after and
   *  `duckId` (if any) is what an in-canvas follow-cam frames. */
  start: (subject: string, duckId?: string | null) => void;
  stop: () => void;
  /** Discard the take without uploading. */
  cancel: () => void;
  /** 📷 a PNG of the current view, named `<stem>.png`. Synchronous — call it
   *  straight from the click handler (lib/record.ts says why). */
  snap: (stem: string) => void;
}

export function useTake({ framingMs = 0 }: { framingMs?: number } = {}): Take {
  const cap = useCapture();
  const recRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timersRef = useRef<number[]>([]);
  const [, bump] = useState(0); // re-render tick for the elapsed timer

  useEffect(() => {
    if (cap.phase !== "recording") return;
    const id = setInterval(() => bump((t) => t + 1), 250);
    return () => clearInterval(id);
  }, [cap.phase]);

  const clearTimers = () => {
    timersRef.current.forEach((t) => clearTimeout(t));
    timersRef.current = [];
  };

  const fail = (msg: string) => {
    captureError(msg);
    pushToast(`🎥 ${msg}`);
  };

  const beginRecording = (subject: string) => {
    if (getCapture().phase !== "framing") return; // cancelled during the glide
    const canvas = getCaptureCanvas();
    // The ducks' voices ride along in the take when they are actually
    // audible — the point of the feature is a clip you can hear.
    const voice = liveDuckAudio()?.captureTrack() ?? null;
    const mime = pickMime(!!voice);
    if (!canvas) return fail("no canvas to record");
    if (!mime) return fail("this browser can't record video (no MediaRecorder)");
    // captureStream(0) + an explicit requestFrame() per rendered frame (the
    // pump lives in a useFrame inside the Canvas) — automatic capture rides
    // the compositor and delivered near-empty webms whenever the tab was
    // throttled. Browsers without requestFrame (Safari) fall back to auto.
    let stream: MediaStream;
    let pumpTrack: { requestFrame: () => void } | null = null;
    try {
      stream = canvas.captureStream(0);
      const t = stream.getVideoTracks()[0] as unknown as {
        requestFrame?: () => void;
      };
      if (typeof t?.requestFrame === "function") {
        pumpTrack = t as { requestFrame: () => void };
      } else {
        stream = canvas.captureStream(30);
      }
    } catch (e) {
      return fail(`canvas capture failed: ${e}`);
    }
    if (voice) stream.addTrack(voice);
    setCaptureTrack(pumpTrack);
    let rec: MediaRecorder;
    try {
      rec = new MediaRecorder(stream, {
        mimeType: mime,
        videoBitsPerSecond: 12_000_000,
      });
    } catch (e) {
      return fail(`recorder failed to start: ${e}`);
    }
    chunksRef.current = [];
    rec.ondataavailable = (e) => {
      if (e.data.size) chunksRef.current.push(e.data);
    };
    rec.onstop = () => upload(subject, new Blob(chunksRef.current, { type: mime }));
    rec.onerror = () => fail("recorder error mid-take");
    recRef.current = rec;
    rec.start(250);
    captureRecording();
    timersRef.current.push(window.setTimeout(stop, MAX_TAKE_MS));
  };

  const start = (subject: string, duckId: string | null = null) => {
    captureFraming(duckId);
    if (framingMs > 0) {
      timersRef.current.push(window.setTimeout(() => beginRecording(subject), framingMs));
    } else {
      beginRecording(subject);
    }
  };

  const stop = () => {
    const rec = recRef.current;
    if (rec?.state !== "recording") return;
    clearTimers();
    captureProcessing(); // before .stop(): onstop checks the phase
    rec.stop();
  };

  const upload = async (subject: string, blob: Blob) => {
    recRef.current = null;
    const pushed = getFramesPushed();
    const pumped = hasCaptureTrack();
    setCaptureTrack(null);
    if (getCapture().phase !== "processing") return; // cancelled
    if (!blob.size) return fail("empty recording — nothing captured");
    // A handful of frames means the scene never rendered during the take
    // (hidden/throttled tab) — a 0.1 s "video" out of ffmpeg would only
    // confuse; say what actually happened instead.
    if (pumped && pushed < 5)
      return fail("scene barely rendered during the take — keep the tab visible while recording");
    try {
      const res = await fetch(
        `${LAB_HTTP}/captures?name=${encodeURIComponent(subject)}`,
        { method: "POST", body: blob }
      );
      if (!res.ok) {
        const detail = (await res.json().catch(() => null))?.detail;
        throw new Error(detail ?? `HTTP ${res.status}`);
      }
      const result = await res.json();
      captureDone(result);
      pushToast(`🎥 saved ${result.name} (mp4 + gif) in captures/`);
    } catch (e) {
      fail(`capture failed: ${e instanceof Error ? e.message : e}`);
    }
  };

  const cancel = () => {
    clearTimers();
    const rec = recRef.current;
    if (rec && rec.state !== "inactive") {
      rec.onstop = null; // discard, don't upload
      rec.stop();
    }
    recRef.current = null;
    setCaptureTrack(null);
    captureReset();
  };

  // Unmount: drop timers and a still-rolling recorder without uploading.
  useEffect(() => () => cancel(), []); // eslint-disable-line react-hooks/exhaustive-deps

  const snap = (stem: string) => {
    if (!snapshotNow(stem)) pushToast("📷 scene still loading — try again in a moment");
  };

  const secs =
    cap.phase === "recording" && cap.recordingSince > 0
      ? Math.floor((Date.now() - cap.recordingSince) / 1000)
      : 0;

  return { cap, secs, start, stop, cancel, snap };
}
