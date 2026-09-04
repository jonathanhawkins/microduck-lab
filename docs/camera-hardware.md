# The camera: what is known, what is assumed, and what it costs

Single home for the camera facts. `DetectorSpec` (`microduck_local/src/
microduck_local/sensors/detector.py`) points here; keep them in step.

Everything below is either **measured**, **vendor-quoted**, or **assumed** —
and it is labelled, because the sim's whole vision model rests on two numbers
(field of view and pixels per radian) and both have been wrong.

---

## 1. The camera on the robot today: IMX219, and it runs CROPPED

Upstream identifies it on hardware — `imx219 2-0010: Model ID 0x0219`
(`microduck/docs/project/media-bringup.md`). That is a Raspberry Pi Camera
v2: 1/4", 1.12 µm, 3280 × 2464, quoted **62.2° × 48.8°**, which is where
`DetectorSpec`'s 62 × 48 came from.

But `mediad`'s `pin_sensor_mode` pins the **sensor** to 1920 × 1080
(`mediad/src/pipeline.rs`), and on the IMX219 that mode is a **crop**, not a
downscale. libcamera reports it as
`1920x1080 [47.57 fps - (680, 692)/1920x1080 crop]` — a centred window
keeping 59% of the columns and **44% of the rows**.

With the stock f = 3.04 mm lens (confirmed by geometry — the full array
computes to 62.3 × 48.8, matching the published spec):

| | H | V |
|---|---|---|
| full array 3280 × 2464 | 62.3° | 48.8° |
| **pinned 1920 × 1080 mode** | **39.0°** | **22.5°** |

So the sim's frustum may be **~23° too wide and ~25° too tall**. The vertical
error is the one that bites: 22.5° is half the vertical view the near-field
reasoning assumes (a floor ball leaving the camera in the last 0.3 m; a
person's middle leaving the frustum at 1.2 m).

The crop cuts the other way on sharpness: 320 px over 39° is **471 px/rad**
against the 296 assumed, because a narrow view concentrates pixels.

**ASSUMED, and it is the open question:** that the module is a stock Pi
Camera v2 (f = 3.04 mm). Third-party IMX219 boards ship M12 lenses from 88°
to 160° and would change every number here. The repo names the driver
overlay, not the lens. Nothing in the sim has been re-baselined on
39 × 22.5 — the durable point is that **62 × 48 is the full-array figure and
the robot does not run the full array**.

## 2. The replacement sensor and lens (vendor datasheet, 2026-09)

| | |
|---|---|
| optical format | 1/2.9" |
| pixel | 2.75 µm × 2.75 µm BSI |
| active array | 1920 × 1080 — **native, no crop** |
| frame rate | up to 90 fps |
| **FOV** | **D 142.2°, H 116°, V 60°** (max DFOV 165°) |
| EFL | ~2.9 mm |

Derived active area: **5.28 × 2.97 mm**, diagonal 6.06 mm, 16:9.

**The lens is not rectilinear, and that is load-bearing.** Solving a pinhole
focal length from each axis gives 1.65 mm (H), 2.57 mm (V), 1.04 mm (D) —
they disagree, so a pinhole model cannot describe this lens. Under an
equidistant (fisheye, r = f·θ) model they agree to ~10%: 2.61 / 2.84 /
2.44 mm, near the quoted 2.9 mm EFL.

`Detector` maps pixel offset to bearing with a **pinhole** model and no
distortion term. On a 116° lens that is wrong at the frame edges — and the
edges are exactly where the extra field of view lives. **Gap:** the sim has
no distortion model, so it would overstate how usable a wide lens's periphery
is for bearing. Not built; recorded.

## 3. What our own lens sweep already says about this camera

The sweep (README, "the lens sweep made honest") measured 62 / 90 / 120 /
150° at two pixel budgets. **116° × 60° is essentially the 120° arm**, so
this is not a new question — it is measured:

| configuration | px/rad | measured tidy rate |
|---|---|---|
| 62° / 320 px — the sim's baseline | 296 | 0.889 |
| 120° / 320 px | 153 | **0.632, worse on 24 of 24 seeds (sign p = 1e-7)** |
| 120° / 640 px | 306 | 0.819, back inside the noise |
| **116° / 320 px — this camera, today's NPU input** | **158** | sits with the **bad** arm |
| **116° / 640 px** | **316** | sits with the **recovered** arm |

**So the deciding variable is the NPU's inference input, not the camera.** At
a 320 px YOLO input this lens is a serious downgrade for tidying — the sweep
found a toy at 1–1.5 m detected in 36% of frames at 62° and **2%** at 120°,
and the geometric floor for a 3.2 cm brick falling from 1.83 m to 0.95 m,
which in a 3 × 2.5 m room turns "scan the room" into "bump into things". At
640 px it lands back inside the shipped lens's noise.

Soccer wants the opposite: kicks scaled with lens width (51 → 94 → 105 → 130
at 62 / 90 / 120 / 150°, p < 0.001), because a big orange ball is easy to
resolve and finding it is the constraint.

**The vertical is a clear win either way.** 60° beats both the sim's assumed
48° and — by a wide margin — the current camera's cropped 22.5°. Vertical
view is what the near-field blindness lives on, so this should help the last
0.3 m onto a ball and the close-in follow case.

## 3b. What the frustum costs in the near field, and what rescues it

A floor object leaves the camera when it is nearer than where the bottom of
the frame meets the floor. Derived from the measured camera pose
(`walker-facts`: standing height 0.235 m at 0.193 rad depression; head down
0.202 m at 0.654 rad):

| vertical FOV | head level | head down |
|---|---|---|
| 48° — what the sim assumes | 0.33 m | 0.11 m |
| **22.5° — the IMX219 1080p crop** | **0.57 m** | 0.18 m |
| 60° — the replacement module | 0.27 m | 0.08 m |

So the crop pushes the head-level blind radius from 0.33 m out to 0.57 m —
a 73% increase, and worse than any lens in the sweep. **But the head dip
rescues most of it**: `tidy` already sets `head_down` for the close
approach, and at 0.654 rad the blind radius is 0.18 m against the sim's
0.11 m. That is a real degradation and a much smaller one than the
head-level figure suggests, which is worth knowing before anyone panics
about the crop. The replacement module improves on the sim on both rows.

## 3c. MEASURED: the sim's baseline is a camera neither candidate is

The audit §3 asked for. `eval-tidy`, 32 paired seeds x 6 toys x 300 s, the
frustum swept and nothing else changed:

| frustum | px/rad | tidied | falls a run | grasp | att/pick | scans |
|---|---|---|---|---|---|---|
| **62 x 48 — what the sim assumes** | 296 | **0.880** | **0.38** | 88% | 1.15 | 80 |
| 39.0 x 22.5 — the IMX219 1080p crop | 470 | 0.609 | **2.44** | 62% | 1.61 | 102 |
| 116 x 60 equidistant — the new module | 158 | 0.542 | 0.69 | 80% | 1.24 | 73 |

Against the baseline, paired: the crop is **−1.62 toys (p < 0.0001, worse
on 26 of 32)** with falls up **6.5x** (12 events against 78, p < 0.0001);
the new module is **−2.03 toys (p < 0.0001, worse on 29 of 32, better on
0)** with falls up but unresolved (p = 0.12).

**Every tidy result in this repo was measured on a camera better than
either real candidate.** That is the headline, and it is a caveat on the
whole Track 12 benchmark rather than a fact about any one knob.

The two fail in *opposite* ways, which is the useful part:

- **The crop fails on FIELD OF VIEW while having the best angular
  resolution of the three** (470 px/rad). It sees sharply through a
  keyhole: scans rise 80 → 102, grasp drops to 62%, and the falls explode —
  a duck with 22.5° of vertical and 39° of horizontal walks into what it
  cannot see.
- **The new module fails on RESOLUTION while having the best field of
  view.** 158 px/rad is the lens sweep's bad arm exactly. Grasp holds at
  80% and falls barely move; it is the *finding* that breaks, which is what
  the sweep predicted.

So the sim's 62 x 48 at 296 px/rad sits near a sweet spot that neither
option occupies, and the choice between them is not "which is better" but
"which failure is cheaper". For the new module the answer is already known
and it is §3: **at a 640 px inference input it is 316 px/rad and lands back
beside the baseline.** The crop has no such escape — no inference budget
fixes a 22.5° vertical.

*Caveats.* Seeds 0–31 are a discovery block; a confirmation on 32 fresh
layouts is running, as is a pinhole rerun of the 116° arm to separate "wide
and blurry" from "and the bearing is wrong by up to 9.7°". And the crop arm
still assumes the stock Pi lens (§1).

## 4. Frame rate is a non-issue

`DetectorSpec.rate_hz` is 10 and the brain decides at 10 Hz, against the
sensor's 90 — nine times the headroom. The developer's "not sure we have
enough internal compute to use the full potential" caveat does not bite this
workload: frames can be dropped freely. What the compute budget must cover is
the **inference input size** in §3, which is a different question and the one
that matters.

## 5. Open questions, in priority order

1. **What YOLO input size can the NPU sustain, and at what rate?** 320 or
   640? §3 says this decides whether the new camera helps or hurts tidying.
   Note the sensor is not the constraint — 1920 px across is six times the
   320 the detector consumes.
2. **Which IMX219 module and lens is on the robot today?** Stock Pi Camera v2
   (f = 3.04 mm) or a third-party wide M12 board? §1 depends on it entirely.
3. **Lens distortion coefficients / calibration for the new module.** §2: the
   pinhole bearing model is wrong at the edges of a 116° lens, and the edges
   are the point of buying one.
4. Is the 16:9 frame letterboxed into the square YOLO input (keeps FOV,
   wastes pixels) or centre-cropped (keeps pixels, loses horizontal FOV)?
   The two give different px/rad **and** a different effective HFOV.

## 6. What the sim now models, and what it still does not

`DetectorSpec.projection` (added 2026-09) models the bearing error a
pinhole-calibrated reader makes on an equidistant lens — the error is
systematic, zero on axis, zero at the calibrated edge, and worst in
between: **9.7° at 116° against 1.2° at 62°**, which is past the chase
brain's 3.4–6.9° aim tolerance. It defaults to `pinhole`, so nothing
measured before it shifted.

One thing that came out of building it: **the size gate was already
equidistant-shaped.** `px_per_rad` is uniform across the field, which is
exactly what r = f·θ gives and is *not* what a pinhole lens does (a pinhole
frame resolves more finely toward its edges). So the width thresholds were
always modelling the wide-lens case; only the bearing needed the new model.

Still not modelled: radial distortion of the *box* (an off-axis target's
apparent width under a fisheye differs from the on-axis case), rolling
shutter, and any per-unit calibration. None of these is worth building
before someone answers §5.1 and §5.2.
