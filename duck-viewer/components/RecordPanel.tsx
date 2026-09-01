"use client";

// 🎥 capture panel (top-center): select a duck, hit record, get content.
// The flow: camera glides to a ¾ shot of the duck (RecordCamera in
// Viewer.tsx), MediaRecorder captures the WebGL canvas — DOM labels and
// panels are not part of the canvas, so takes come out clean — then the lab
// server converts the upload to mp4 + gif (POST /captures) and the panel
// offers both as downloads.

import { useEffect, useRef, useState } from "react";
import { LAB_HTTP, type LabClient } from "@/lib/lab";
import { useSelectedDuck } from "@/lib/select";
import { useHudRight } from "@/lib/ui";
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
} from "@/lib/record";
import { pushToast } from "./Toasts";

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";

/** Camera glide before the recorder rolls (matches RecordCamera's damping —
 *  the shot has settled by then, so takes don't open with a swish). */
const FRAMING_MS = 1200;
/** Safety cap: a forgotten recorder must not fill the disk. */
const MAX_TAKE_MS = 60_000;

/** Duck names carry emoji/spaces — reduce to a safe filename stem (mirrors
 *  the server's capture_slug, so shots and takes sort together). */
function slug(s: string): string {
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

function stamp(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}` +
    `-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
  );
}

function pickMime(): string | null {
  if (typeof MediaRecorder === "undefined") return null;
  const prefs = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4", // Safari
  ];
  return prefs.find((m) => MediaRecorder.isTypeSupported(m)) ?? null;
}

// No `left` here — the component computes it per render: centered, but never
// under the top-left HUD panel (see the layout block in RecordPanel).
const panelStyle: React.CSSProperties = {
  position: "fixed",
  top: 10,
  zIndex: 20,
  display: "flex",
  alignItems: "center",
  gap: 8,
  background: "rgba(14, 16, 20, 0.86)",
  border: "1px solid rgba(255,255,255,0.12)",
  borderRadius: 8,
  padding: "6px 10px",
  color: "#d8dee8",
  fontFamily: mono,
  fontSize: 12,
  backdropFilter: "blur(6px)",
};

const btnStyle: React.CSSProperties = {
  background: "rgba(255,255,255,0.06)",
  border: "1px solid rgba(255,255,255,0.14)",
  borderRadius: 6,
  color: "#d8dee8",
  fontFamily: mono,
  fontSize: 12,
  padding: "3px 10px",
  cursor: "pointer",
};

const linkStyle: React.CSSProperties = {
  ...btnStyle,
  textDecoration: "none",
  color: "#7db8d8",
};

export function RecordPanel({
  clientRef,
}: {
  clientRef: React.MutableRefObject<LabClient | null>;
}) {
  const selected = useSelectedDuck();
  const cap = useCapture();
  const recRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timersRef = useRef<number[]>([]);
  const [, bump] = useState(0); // re-render tick for the elapsed timer

  // Layout inputs for the HUD-dodging `left` computed at the bottom: the
  // HUD's live right edge, this panel's own width, and the window width.
  const hudRight = useHudRight();
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [panelW, setPanelW] = useState(220);
  const [winW, setWinW] = useState(() =>
    typeof window === "undefined" ? 1200 : window.innerWidth
  );
  useEffect(() => {
    const onResize = () => setWinW(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    setPanelW(el.offsetWidth);
    const ro = new ResizeObserver(() => setPanelW(el.offsetWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

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

  const start = () => {
    if (!selected) return;
    const duckName =
      clientRef.current?.frame?.ducks.find((d) => d.id === selected)?.name ??
      "duck";
    captureFraming(selected);
    timersRef.current.push(
      window.setTimeout(() => beginRecording(duckName), FRAMING_MS)
    );
  };

  const beginRecording = (duckName: string) => {
    if (getCapture().phase !== "framing") return; // cancelled during the glide
    const canvas = getCaptureCanvas();
    const mime = pickMime();
    if (!canvas) return fail("no canvas to record");
    if (!mime) return fail("this browser can't record video (no MediaRecorder)");
    // captureStream(0) + an explicit requestFrame() per rendered frame (the
    // pump lives in RecordCamera's useFrame) — automatic capture rides the
    // compositor and delivered near-empty webms whenever the tab was
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
    rec.onstop = () =>
      upload(duckName, new Blob(chunksRef.current, { type: mime }));
    rec.onerror = () => fail("recorder error mid-take");
    recRef.current = rec;
    rec.start(250);
    captureRecording();
    timersRef.current.push(window.setTimeout(stop, MAX_TAKE_MS));
  };

  const stop = () => {
    const rec = recRef.current;
    if (rec?.state !== "recording") return;
    clearTimers();
    captureProcessing(); // before .stop(): onstop checks the phase
    rec.stop();
  };

  const upload = async (duckName: string, blob: Blob) => {
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
        `${LAB_HTTP}/captures?name=${encodeURIComponent(duckName)}`,
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

  // 📷 is instant and client-side: name the file after the selected duck (or
  // the whole lab) and capture SYNCHRONOUSLY — the download must fire inside
  // this click's user gesture (see lib/record.ts).
  const snap = () => {
    const duckName = selected
      ? clientRef.current?.frame?.ducks.find((d) => d.id === selected)?.name
      : null;
    if (!snapshotNow(`${slug(duckName ?? "duck-lab")}-${stamp()}`))
      pushToast("📷 scene still loading — try again in a moment");
  };

  let content: React.ReactNode;
  if (cap.phase === "idle") {
    content = (
      <>
        {selected && (
          <button style={btnStyle} onClick={start} title="film the selected duck — the camera frames it, then mp4 + gif land in captures/">
            🎥 record
          </button>
        )}
        <button style={btnStyle} onClick={snap} title="download a PNG of the current view (selection ring hidden for the shot)">
          📷 shot
        </button>
      </>
    );
  } else if (cap.phase === "framing" || cap.phase === "recording") {
    const secs =
      cap.recordingSince > 0
        ? Math.floor((Date.now() - cap.recordingSince) / 1000)
        : 0;
    content = (
      <>
        {cap.phase === "framing" ? (
          <span>🎥 framing…</span>
        ) : (
          <>
            <span style={{ color: "#e07a5f" }}>●</span>
            <span>{secs}s</span>
            <button style={btnStyle} onClick={stop}>
              ■ stop
            </button>
          </>
        )}
        <button style={btnStyle} onClick={cancel} title="discard the take">
          ✕
        </button>
      </>
    );
  } else if (cap.phase === "processing") {
    content = <span>⏳ making mp4 + gif…</span>;
  } else if (cap.phase === "done" && cap.result) {
    content = (
      <>
        <span>🎥 {cap.result.name}</span>
        <a style={linkStyle} href={`${LAB_HTTP}${cap.result.mp4}`}>
          ⬇ mp4 {Math.max(1, Math.round(cap.result.mp4Kb / 1024))}MB
        </a>
        <a style={linkStyle} href={`${LAB_HTTP}${cap.result.gif}`}>
          ⬇ gif {Math.max(1, Math.round(cap.result.gifKb / 1024))}MB
        </a>
        <button style={btnStyle} onClick={captureReset}>
          ✕
        </button>
      </>
    );
  } else {
    content = (
      <>
        <span style={{ color: "#e07a5f" }}>
          ⚠ {cap.error ?? "capture failed"}
        </span>
        <button style={btnStyle} onClick={captureReset}>
          ✕
        </button>
      </>
    );
  }

  // Centered at the top — but never UNDER the HUD: a wide duck-lab panel
  // (long duck names) used to reach right beneath these buttons. The HUD
  // publishes its right edge (useHudRight); slide right of it when centering
  // would collide, and keep a margin from the right edge as a backstop.
  const centered = (winW - panelW) / 2;
  const left = Math.round(
    Math.min(Math.max(centered, hudRight + 12), Math.max(12, winW - panelW - 12))
  );
  return (
    <div ref={wrapRef} data-policy-ui style={{ ...panelStyle, left }}>
      {content}
    </div>
  );
}
