// The head-camera inset on the /sim page: where the duck's-eye view goes on
// the canvas, and how big its DOM box has to be for the detector's boxes to
// land on what is rendered under them.
//
// The inset is NOT a render target. It is a second, scissored pass into the
// same canvas (SimViewer's InsetRender), with the DOM box over it supplying
// only the border, the crosshair and the detection rectangles. That makes the
// units at the three.js boundary the whole ballgame, which is what this
// module exists to pin down — see lib/inset.test.ts.

/** The fields of a DOMRect this module reads. */
export interface RectLike { left: number; bottom: number; width: number; height: number }

/** A rectangle in the units three's setScissor/setViewport take. */
export interface Rect { x: number; y: number; w: number; h: number }

/** The minimum of three's WebGLRenderer the inset pass drives. */
export interface InsetTarget {
  setScissorTest(enable: boolean): void;
  setScissor(x: number, y: number, w: number, h: number): void;
  setViewport(x: number, y: number, w: number, h: number): void;
  render(scene: object, camera: object): void;
}

/** The minimum of a PerspectiveCamera the inset pass reshapes. */
export interface InsetCamera {
  aspect: number;
  updateProjectionMatrix(): void;
}

/** Below this the box is a sliver (a collapsed panel, a mid-layout frame) and
 *  there is nothing worth a second pass. */
const MIN_INSET_PX = 8;

/** Where a DOM box sits inside the canvas, ready for setScissor/setViewport:
 *  CSS pixels, origin BOTTOM-left.
 *
 *  CSS pixels, and deliberately no pixel ratio in sight. three scales both
 *  calls by the renderer's own pixel ratio (WebGLRenderer.setViewport /
 *  setScissor: `.multiplyScalar( _pixelRatio )`), so a caller that helpfully
 *  pre-multiplies applies the ratio TWICE. That shipped: on a retina Mac at
 *  dpr 1.5 the pass landed 1.5x off the box, so the inset showed nothing but
 *  the main view through a transparent div, and the doubled restore left the
 *  main view itself zoomed by the same factor. */
export function insetRect(box: RectLike, canvas: RectLike): Rect {
  return {
    x: Math.round(box.left - canvas.left),
    y: Math.round(canvas.bottom - box.bottom),
    w: Math.round(box.width),
    h: Math.round(box.height),
  };
}

/** Draw `scene` from `camera` into the `box`'s rectangle of the canvas alone,
 *  then hand the whole canvas back. Returns the rectangle drawn, or null if
 *  the box was too small to bother with.
 *
 *  It takes the two DOMRects rather than a rectangle so there is no seam for
 *  a caller to do its own arithmetic in — measuring the box IS the part that
 *  went wrong. For the same reason it owns the camera's aspect: a camera
 *  shaped differently from the rectangle it renders into spans a different
 *  field of view than the one detectionBox() draws its boxes in, and every
 *  box drifts off its target.
 *
 *  Handing the canvas back matters as much as the pass: the same priority-1
 *  loop draws the main view on the next frame, so a scissor test left on or a
 *  viewport left narrowed breaks that and everything after it. `size` is the
 *  canvas in CSS pixels (r3f's useThree().size). */
export function renderInset(
  gl: InsetTarget,
  scene: object,
  camera: InsetCamera,
  box: RectLike,
  canvas: RectLike,
  size: { width: number; height: number },
): Rect | null {
  const rect = insetRect(box, canvas);
  if (rect.w < MIN_INSET_PX || rect.h < MIN_INSET_PX) return null;
  camera.aspect = rect.w / rect.h;
  camera.updateProjectionMatrix();
  gl.setScissorTest(true);
  gl.setScissor(rect.x, rect.y, rect.w, rect.h);
  gl.setViewport(rect.x, rect.y, rect.w, rect.h);
  gl.render(scene, camera);
  gl.setScissorTest(false);
  gl.setViewport(0, 0, size.width, size.height);
  return rect;
}

/** A pinhole's width/height at a field of view.
 *
 *  The inset renders with a PerspectiveCamera set to the detector's VERTICAL
 *  fov, so the horizontal one it actually spans comes out of the box's
 *  aspect. Size the box at anything else and the render is wider or narrower
 *  than the frame detectionBox() places its rectangles in — the boxes slide
 *  off their targets, worst at the edges. */
export function camAspect(fovDeg: [number, number]): number {
  return Math.tan((fovDeg[0] * Math.PI) / 360) / Math.tan((fovDeg[1] * Math.PI) / 360);
}

/** The height that makes a box of `width` show exactly `fovDeg`. */
export function insetHeight(width: number, fovDeg: [number, number]): number {
  return Math.round(width / camAspect(fovDeg));
}
