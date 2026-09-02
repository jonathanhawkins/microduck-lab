// The head-camera inset's two silent failure modes, pinned.
//
// Both are silent because nothing throws and nothing looks broken in
// isolation: the DOM box keeps drawing its border, crosshair and detection
// rectangles whatever the canvas underneath does. The bug this file was
// written for shipped exactly that way — a panel of labels floating over the
// main orbit view, on retina only. Pixels can only be judged by looking
// (.claude/skills/sim-smoke); what a test CAN hold is the arithmetic that
// decides which pixels get asked for.

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { camAspect, insetHeight, insetRect, renderInset, type InsetCamera, type InsetTarget } from "@/lib/inset";
import { CAM_FOV_DEG, detectionBox, type DetectionItem } from "@/lib/sim";

/** three's WebGLRenderer, in the three calls the inset pass makes.
 *
 *  setViewport and setScissor take CSS pixels and scale by the renderer's own
 *  pixel ratio (`.multiplyScalar( _pixelRatio )` in both) — the contract
 *  `threeStillScalesInternally` below re-checks against the installed three.
 *  Modelling that here is what turns "which pixels did we ask for?" into a
 *  number, with no GL context anywhere near CI. */
class FakeRenderer implements InsetTarget {
  scissorTest = false;
  scissor: number[] | null = null;
  viewport: number[] | null = null;
  /** The state each render() was issued under — the only thing GL sees. */
  passes: { scissorTest: boolean; scissor: number[] | null; viewport: number[] | null }[] = [];

  constructor(readonly pixelRatio: number) {}

  private device(x: number, y: number, w: number, h: number) {
    return [x, y, w, h].map((v) => Math.round(v * this.pixelRatio));
  }
  setScissorTest(enable: boolean) { this.scissorTest = enable; }
  setScissor(x: number, y: number, w: number, h: number) { this.scissor = this.device(x, y, w, h); }
  setViewport(x: number, y: number, w: number, h: number) { this.viewport = this.device(x, y, w, h); }
  render() {
    this.passes.push({ scissorTest: this.scissorTest, scissor: this.scissor, viewport: this.viewport });
  }
}

/** A canvas filling the window, with the inset parked bottom-right of the
 *  inspector — the real layout, in CSS pixels. */
const CANVAS = { left: 0, bottom: 600, width: 900, height: 600 };
const BOX = { left: 638, bottom: 520, width: 242, height: 179 };

describe("insetRect", () => {
  it("puts the DOM box on the canvas with a bottom-left origin", () => {
    expect(insetRect(BOX, CANVAS)).toEqual({ x: 638, y: 80, w: 242, h: 179 });
  });

  it("measures from the canvas, not the window", () => {
    // The canvas is not always at the window's origin (an editor rail, a
    // header that pushes it down); the offset has to come out of both axes.
    const shifted = { left: 40, bottom: 560, width: 900, height: 600 };
    expect(insetRect(BOX, shifted)).toEqual({ x: 598, y: 40, w: 242, h: 179 });
  });
});

describe("renderInset", () => {
  const rect = insetRect(BOX, CANVAS);
  const size = { width: CANVAS.width, height: CANVAS.height };
  const scene = {};
  const camera = (): InsetCamera & { updates: number } =>
    ({ aspect: 0, updates: 0, updateProjectionMatrix() { this.updates++; } });

  // dpr 1 is the display the bug hid on; the Canvas clamps to dpr={[1, 1.5]},
  // and 2 and 3 stand in for whatever a future clamp allows.
  for (const dpr of [1, 1.5, 2, 3]) {
    it(`scissors the box's own device pixels at dpr ${dpr}`, () => {
      const gl = new FakeRenderer(dpr);
      expect(renderInset(gl, scene, camera(), BOX, CANVAS, size)).toEqual(rect);

      const want = [rect.x, rect.y, rect.w, rect.h].map((v) => Math.round(v * dpr));
      expect(gl.passes).toHaveLength(1);
      expect(gl.passes[0].scissorTest).toBe(true);
      expect(gl.passes[0].scissor).toEqual(want);
      expect(gl.passes[0].viewport).toEqual(want);
    });
  }

  it("does not apply the pixel ratio twice", () => {
    // The shipped bug, named. At dpr 1.5 the pass was scissored to 1.5x the
    // box's device rect, so it drew off the panel entirely and the inset
    // showed the main view straight through a transparent div.
    const gl = new FakeRenderer(1.5);
    renderInset(gl, scene, camera(), BOX, CANVAS, size);
    const doubled = [rect.x, rect.y, rect.w, rect.h].map((v) => Math.round(v * 1.5 * 1.5));
    expect(gl.passes[0].scissor).not.toEqual(doubled);
  });

  it("hands the whole canvas back", () => {
    // The same loop draws the main view on the next frame. A scissor test
    // left on, or a viewport left narrowed, breaks that pass and every one
    // after it — the doubled restore is what zoomed the main view 1.5x.
    const gl = new FakeRenderer(1.5);
    renderInset(gl, scene, camera(), BOX, CANVAS, size);
    expect(gl.scissorTest).toBe(false);
    expect(gl.viewport).toEqual([0, 0, CANVAS.width * 1.5, CANVAS.height * 1.5]);
  });

  it("shapes the camera to the rectangle it renders into", () => {
    // Not a convenience: a camera at any other aspect spans a different
    // horizontal fov than the frame detectionBox() draws in.
    const gl = new FakeRenderer(2);
    const cam = camera();
    renderInset(gl, scene, cam, BOX, CANVAS, size);
    expect(cam.aspect).toBeCloseTo(rect.w / rect.h, 12);
    expect(cam.updates).toBe(1);
  });

  it("skips a box too small to draw into", () => {
    // A collapsed panel, or a frame measured mid-layout. Nothing is rendered
    // and nothing is left set — a scissor left on here would take the main
    // view down with it.
    const gl = new FakeRenderer(1.5);
    const sliver = { left: 638, bottom: 520, width: 242, height: 4 };
    expect(renderInset(gl, scene, camera(), sliver, CANVAS, size)).toBeNull();
    expect(gl.passes).toEqual([]);
    expect(gl.scissorTest).toBe(false);
  });
});

describe("camAspect", () => {
  it("gives the box the shape of the detector's frustum", () => {
    // A PerspectiveCamera is told the VERTICAL fov; the horizontal one it
    // spans falls out of the aspect it is given. Real three, no GL needed.
    const w = 242, h = insetHeight(w, CAM_FOV_DEG);
    const cam = new THREE.PerspectiveCamera(CAM_FOV_DEG[1], w / h);
    const fovH = (2 * Math.atan(cam.aspect * Math.tan((cam.fov * Math.PI) / 360)) * 180) / Math.PI;
    expect(fovH).toBeCloseTo(CAM_FOV_DEG[0], 0);
    expect(camAspect(CAM_FOV_DEG)).toBeCloseTo(1.35, 2);
  });

  // The overlay and the render are two independent projections of the same
  // frustum: detectionBox() places rectangles in fractions of the frame,
  // three projects the geometry under them. They have to be the same map, or
  // every box sits beside its target — worst at the edges, which is where a
  // chase brain's ball spends its time.
  //
  // Detector frame (sensors/detector.py): x forward, y left, z up, bearing
  // +left. three's camera looks down -z with +x right, +y up. Bearing is
  // swept at elevation 0 and elevation at bearing 0: the detector's angles
  // are spherical (elevation is measured off the horizontal distance, not the
  // forward axis), so away from those two lines the two differ by a factor of
  // cos(bearing) — 4% of the frame in the extreme corner, and the detector's
  // own bearing noise is 1-3 degrees.
  const edgeH = (CAM_FOV_DEG[0] * Math.PI) / 360, edgeV = (CAM_FOV_DEG[1] * Math.PI) / 360;
  const AXES: [number, number][] = [
    [0, 0], [edgeH, 0], [-edgeH, 0], [edgeH / 2, 0], [-edgeH / 3, 0],
    [0, edgeV], [0, -edgeV], [0, edgeV / 2],
  ];
  /** Where a PerspectiveCamera of this aspect projects the detection, 0..1. */
  function projected(bearing: number, elevation: number, range: number, aspect: number) {
    const cam = new THREE.PerspectiveCamera(CAM_FOV_DEG[1], aspect, 0.03, 20);
    cam.updateProjectionMatrix();
    const p = new THREE.Vector3(
      -range * Math.cos(elevation) * Math.sin(bearing),   // +bearing is left, +x is right
      range * Math.sin(elevation),
      -range * Math.cos(elevation) * Math.cos(bearing),   // forward is -z
    ).applyMatrix4(cam.projectionMatrix);
    return { u: (p.x + 1) / 2, v: (1 - p.y) / 2 };
  }
  const ball = (bearing: number, elevation: number, range: number): DetectionItem =>
    ({ cls: "ball", name: "ball", bearing, elevation, width: 0.05, range, conf: 1 });

  it("is the same projection detectionBox draws in", () => {
    for (const [bearing, elevation] of AXES) {
      const want = projected(bearing, elevation, 1.4, camAspect(CAM_FOV_DEG));
      const box = detectionBox(ball(bearing, elevation, 1.4), CAM_FOV_DEG);
      expect(box.u).toBeCloseTo(want.u, 12);
      expect(box.v).toBeCloseTo(want.v, 12);
    }
  });

  it("stays sub-pixel once the box is rounded to whole CSS pixels", () => {
    // insetHeight rounds, so the box on screen is never at exactly the ideal
    // aspect. What matters is that the leftover lands inside a pixel.
    const w = 242, h = insetHeight(w, CAM_FOV_DEG);
    for (const [bearing, elevation] of AXES) {
      const want = projected(bearing, elevation, 1.4, w / h);
      const box = detectionBox(ball(bearing, elevation, 1.4), CAM_FOV_DEG);
      expect(Math.abs(box.u - want.u) * w).toBeLessThan(0.5);
      expect(Math.abs(box.v - want.v) * h).toBeLessThan(0.5);
    }
  });
});

describe("the three.js contract FakeRenderer stands in for", () => {
  it("still scales setViewport and setScissor by the pixel ratio itself", () => {
    // The premise of every assertion above, re-read from the three that is
    // actually installed. A renderer cannot be built without a GL context, so
    // this checks the source: if a future three ever takes device pixels
    // instead, this fails here rather than silently in the browser, and both
    // the fake and lib/inset.ts need revisiting.
    const src = readFileSync(
      createRequire(import.meta.url).resolve("three/src/renderers/WebGLRenderer.js"),
      "utf8",
    );
    for (const fn of ["setViewport", "setScissor"] as const) {
      const at = src.indexOf(`this.${fn} = function`);
      expect(at, `${fn} not found in three's WebGLRenderer`).toBeGreaterThan(-1);
      expect(src.slice(at, at + 600), `${fn} no longer mentions _pixelRatio`).toContain("_pixelRatio");
    }
  });
});
