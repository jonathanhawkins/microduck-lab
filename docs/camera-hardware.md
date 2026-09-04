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

### The crop, confirmed

On 32 layouts nobody had run: **−1.44 toys (p < 0.0001, worse on 22 of
32)**, falls 13 → 66 events. Pooled over all **64 distinct layouts**:
**−1.53 ± 0.17, p < 0.0001, worse on 48 and better on 3**, with falls
**25 → 144 events (+1.86 ± 0.19)**. It replicated, and it is about as solid
as anything measured in this repo.

### The 116° arm, decomposed — and nearly half of it is fixable

Running the same frustum with a **pinhole** reader separates "wide and
blurry" from "and the bearing is wrong":

| | tidied | falls | grasp |
|---|---|---|---|
| 62 × 48 baseline | 0.880 | 0.38 | 88% |
| 116 × 60, **pinhole** (geometry only) | 0.693 | 0.59 | 86% |
| 116 × 60, **equidistant** (the real lens) | 0.542 | 0.69 | 80% |

| step | cost |
|---|---|
| the wide, low-resolution geometry | **−1.12 toys** (p < 0.0001) |
| the bearing error *on top of it* | **−0.91 toys** (p = 0.0005) |
| together | −2.03 toys |

**The distortion is nearly half the total cost of the wide lens** — and it
is the half that a calibration fixes, which is a checkerboard and an
afternoon rather than new hardware. Note grasp barely moves between the two
(86% → 80%) while tidying falls a long way: a bearing error does not make
the duck fumble, it makes it walk to the wrong place.

So the recommendation for the replacement module is now concrete and has
two parts, each addressing one half:

1. **Calibrate the lens** and use a proper projection when mapping box →
   bearing. Recovers ~0.9 of the 2.03 toys.
2. **Raise the inference input to 640 px** (§3): 316 px/rad puts it back
   beside the baseline on the resolution half.

Do both and it should land near 62 × 48. Do neither and it is a 40%
downgrade on the tidy benchmark.

### CONFIRMED: the recommendation, measured as a recommendation

The two parts above were each measured *separately* and the "do both" was
arithmetic on two numbers — exactly the kind of composition this repo has
been wrong about before. So both halves were applied together and run as
one arm, 32 paired seeds, same layouts as everything above:

| arm | px/rad | tidied | falls a run |
|---|---|---|---|
| 62 × 48, 320 px, pinhole — the shipped baseline | 296 | 0.880 | 0.38 |
| 116 × 60, 320 px, equidistant — the module as-is | 158 | 0.542 | 0.69 |
| 116 × 60, 320 px, **calibrated** | 158 | 0.693 | 0.59 |
| 116 × 60, **640 px**, equidistant | 316 | 0.724 | 0.38 |
| 116 × 60, **640 px, calibrated** — the recommendation | 316 | **0.880** | 0.41 |

Paired against the 116 × 60 module as it ships:

| step | gain |
|---|---|
| calibrate the lens (still 320 px) | **+0.91 toys** (p = 0.0005, better on 18 of 32) |
| 640 px input (lens left uncalibrated) | **+1.09 toys** (p = 0.0001, better on 23 of 32) |
| **both** | **+2.03 toys** (p < 0.0001, better on 28 of 32) |

**The recommendation vs the shipped baseline: +0.00 toys, p = 1.0000**
(better on 11 seeds, worse on 8, tied on 13), falls 0.41 against 0.38. Not
"close to" the baseline — indistinguishable from it, on the same 32
layouts.

Two things worth keeping from this:

1. **The two costs are separable and additive.** 0.91 + 1.09 = 2.00 against
   the 2.03 measured jointly. Bearing error and angular resolution are
   independent failures here, so either fix alone buys about half and
   neither is wasted if the other is skipped. That is *not* something the
   separate arms could establish — it is why the joint arm was run.
2. **Either fix alone still leaves a real loss**: 640 px alone is −0.94
   against the baseline (p < 0.0001) and calibration alone −1.12
   (p < 0.0001). Both are needed to break even; one is a half-measure with
   a measured price.

So the hardware answer is: **the 116 × 60 module costs nothing on this
benchmark provided the lens is calibrated and the NPU runs a 640 px input.**
Neither is new hardware. §4 covers whether the NPU has the budget for the
second.

### And on a second benchmark it is better than break-even

Tidying is one measure and "costs nothing" is a weak conclusion to rest a
hardware decision on, so the same three cameras were put through a
*different* benchmark measured a *different* way: the soccer line-up, whose
2–3 cm scatter the roadmap already blames for the shots that miss. Every
ball sighting in 3 seeds × 180 s of 1v1, placement checked against truth:

| camera | ball sightings | placement error | line-up error |
|---|---|---|---|
| 62 × 48, 320 px — the baseline | 16 881 | 7.8 cm | 5.6 cm |
| 116 × 60, 320 px, uncalibrated — as it ships | 16 374 | 9.0 cm | **7.5 cm** |
| 116 × 60, **640 px, calibrated** | **21 045** | 7.0 cm | **5.4 cm** |

("line-up error" is the median for a ball inside 0.6 m and nearly still —
the case that scores. These are medians over sightings, not paired seeds:
sightings within a run are correlated, so read the sizes, not a p-value.)

Two things the tidy benchmark could not show:

1. **The wide lens's actual benefit appears once the resolution is
   restored** — 25% more ball sightings (16 881 → 21 045). That is what
   116° is *for*, and at 320 px it does not appear at all: the as-ships
   module logs *fewer* sightings than the baseline (16 374) while placing
   what it sees 34% less accurately.

   Read that second number carefully. **Sighting count is an outcome, not a
   frustum property.** A wider frustum cannot show a target *less* often
   for a fixed trajectory — `projection` changes the reported bearing, not
   whether a target is in frame or passes the size gate. What it changes is
   the *brain*: a duck fed bearings that are wrong by up to 9.7° drives
   somewhere else, and ends up with the ball in frame no more often than
   the narrow camera did. So the honest statement is that the extra field
   of view buys nothing **as played** at 320 px, not that the sensor sees
   less.
2. **The recommendation is slightly better than the baseline here**, not
   merely level: 5.4 cm against 5.6. So "costs nothing on tidy" understates
   it — on the perception measure the wide lens is a small net win.

Worth being clear about what limits that number: at 5.4–5.6 cm the ball's
placement error is dominated by the **detector's own bearing and range
noise**, not by the frustum and not by either known bias (the stale pose
contributes 1.7 cm of it; ball motion 0.6 cm — `brain/tracker.py`). A
better lens moves it a little. Closing the line-up's 2–3 cm scatter is a
detector-quality problem, and that is a different piece of work.

### What that noise is made of

"Detector noise" is not actionable, so it was split. 16 881 sightings, each
detection compared against truth **in the detector's own frame and
definition** (which took two attempts — see the traps below):

| band | bearing error → lateral | range error | range's share |
|---|---|---|---|
| the line-up band (< 0.6 m) | 3.35° → **2.00 cm** | **2.69 cm** | 57% |
| mid (0.6–1.5 m) | 1.87° → 2.63 cm | 5.66 cm | 68% |
| far (> 1.5 m) | 1.96° → 5.70 cm | 15.58 cm | 73% |

**Range dominates, and its share grows with distance.** Range here is
inferred from apparent width, which is a *pixel* quantity — so range error
is a resolution story, and that is why the 640 px arm moved the line-up
number at all (5.6 → 5.4 cm). Bearing error is roughly constant in angle
(1.9–3.4°) and so grows linearly in metres, which is why it overtakes
nothing but stays a third of the total.

Both components are **noise rather than bias** (mean +1.3 cm of range
against a 2.7 cm median absolute error in the line-up band). That matters
because it says which levers exist: temporal averaging *does* help, and the
tracker currently smooths at α = 0.6 — about 1.7 frames of averaging, which
is light for a ball that is stationary in half of all sightings. Heavier
averaging gated on the tracker's own velocity estimate is the obvious
cheap lever, and it is deliberately **not** taken here: AGENTS.md rule 7
applies (the line-up's offsets and tolerances were fitted with this noise
present), the soccer benchmark needs ~200 seeds to resolve a +0.3 goal
effect, and two previous line-up-precision changes measured *worse*. It is
written down as a lead with its numbers, not shipped on a hunch.

**Two traps, both the same mistake**, recorded because the first version of
this table was wrong in a way that looked like a discovery:

1. `range_est` is the **3-D slant range** from the camera *site* to the
   ball's centre, and the camera sits ~21 cm above a ball on the floor.
   Compared against a 2-D ground distance it shows a "+5.7 cm systematic
   bias" that is entirely the measurer's: at 0.35 m,
   `hypot(0.35, 0.21) − 0.35 = 5.8 cm`.
2. `bearing` is in the **camera's** frame, and the chase brain yaws the head
   to track the ball. Compared against a body-frame bearing it charges the
   detector for the head's rotation — inflating the line-up band's bearing
   error from 3.35° to 5.65° and inventing a −1.5° bias.
   `frame.cam_yaw` is what `Tracker._associate` adds for exactly this
   reason.

*Caveat that remains.* The crop arm assumes the stock Pi lens (§1). Nothing
above is affected if that assumption is wrong — the 39 × 22.5 arm would
simply be describing a camera the robot does not have, and the question of
what it *does* have would still be open.

## 3d. The VERTICAL field of view: how much floor the camera loses

The horizontal number gets all the attention because it is what the lens
sweep varied, but the *vertical* one decides something specific and
physical: **how close a floor object can be before the camera loses it.**
That is the "blind last 30 cm" that `tof_floor_ball`, `_gaze_pitch` and
half a dozen docstrings are written around.

Measured from the model rather than from the docstrings — the `head_camera`
site on a duck **standing on the walker**, driving the head-pitch command
and reading the site back:

```
 cmd  tilt deg    z cm  blind@48  blind@60
0.00     11.01    23.5     28.5cm     23.0cm
0.20     19.45    22.5     20.0cm     16.2cm
0.40     28.30    21.3     13.8cm     11.0cm
0.60     37.53    20.1      9.0cm      6.9cm

measured: tilt = 11.01 deg + 0.772 * cmd (rad/unit)
assumed : tilt = 11.29 deg + 0.750 * cmd   (striker._gaze_pitch)
```

Three results:

1. **`_gaze_pitch`'s constants are right.** `cam_level = 0.197` and
   `gain = 0.75` were fitted by hand; measured they are 0.192 and 0.772.
   The "~0.3 m level, ~0.2 m looking down" in its docstring is the 28.5 cm
   and 20.0 cm above — accurate.
2. **The camera is not level at rest — it sits 11° down when the duck
   stands.** Reading the site at the model's *default* qpos says 0.0° and
   24.8 cm, which is wrong by the whole 11°; the walker's standing pose is
   what tilts it. A first pass at this measurement made exactly that
   mistake and concluded the docstrings were wrong by 20 cm. **Settle the
   duck before reading a site.**
3. **60° vertical buys 5.5 cm of floor, not the 11 cm a level-camera
   calculation gives.** Level-head blind radius 28.5 → 23.0 cm; head fully
   down 9.0 → 6.9 cm. The gain shrinks the more the head is already
   dipped, because the dip and the half-FOV add.

So the replacement module's 60° vertical is a real improvement over 48° and
a small one. It does not remove the blind zone — nothing does, the camera
is 23 cm above the floor and pointing forward — it moves its edge in by
about a fifth. Anything that depends on the last 20 cm still cannot use the
camera for it.

**What that does to the ToF floor-ball.** `ChaseParams.tof_ball_m` — read
the ball out of the ToF's lower rows while the camera cannot see it — was
measured at 48° and shipped OFF. The wider vertical was the obvious reason
to re-open it, so it was, counting the blob's own events over 6 seeds ×
180 s of 1v1 rather than re-running the noisy goal difference:

| V FOV | blob ticks | camera already had the ball | blind-case ticks | of those, the ball |
|---|---|---|---|---|
| 48° | 4785 | 87.5% | 599 | **30.1%** |
| 60° | 5809 | 93.6% | 374 | **36.9%** |

It closes harder than it did at 48°, for two reasons that both move the
wrong way:

1. The blob is **almost always redundant** — on 88–94% of the ticks it
   fires, the camera already has the ball.
2. In the case it exists for, it is **wrong about two times in three**, and
   that case is the one that ends in a line-up on the other duck's foot.

The wider lens *does* raise blind-case precision (30% → 37%) while cutting
the opportunity by 38% (599 → 374 ticks), because it reaches 5.5 cm further
into the blind zone itself. **Better optics shrink this feature's job
faster than they improve it** — which is the general shape of the answer
for anything built to paper over a sensor limit.

## 4. Frame rate is a non-issue — and the detector rate is now a lever

`DetectorSpec.rate_hz` is 10 and the brain decides at 10 Hz, against the
sensor's 90 — nine times the headroom. The developer's "not sure we have
enough internal compute to use the full potential" caveat does not bite this
workload: frames can be dropped freely. What the compute budget must cover is
the **inference input size** in §3, which is a different question and the one
that matters.

### The two halves meet here

§3 says the replacement module needs a **640 px** inference input to break
even. The NPU's measured budget for the shipped 320 px YOLO11n is p50
25.7 ms / p95 58.4 ms per frame (upstream `npu-bringup.md`). 640 × 640 is
four times the pixels, so expect roughly **p50 ~100 ms / p95 ~230 ms**.

Put that against a frame budget:

| input | rate | period | p50 duty | p95 duty |
|---|---|---|---|---|
| 320 px | 10 Hz | 100 ms | 26% | 58% |
| 640 px | 10 Hz | 100 ms | ~100% | **over budget** |
| **640 px** | **5 Hz** | **200 ms** | **~50%** | **~115%, marginal** |

**640 px at 10 Hz does not fit. At 5 Hz it plausibly does.** So the camera
recommendation is only implementable if the detector can run at 5 Hz — and
until recently it could not: halving the rate cost **−0.55 toys** on the
tidy benchmark (64 paired layouts, p = 0.0003), breaking the grasp
specifically (85% → 67%).

That cost has since been removed in software, and for a reason unrelated to
the camera: a detection was being placed from the pose the duck had *now*
rather than the pose it had when the frame was taken, which at 5 Hz is ~6 cm
of error against a 3 cm toy. Fixing that — together with the stop distance
that had been hand-fitted around the bias — makes **5 Hz statistically
indistinguishable from 10 Hz** (−0.14, p = 0.32) with the grasp at 94%.
`TidyParams.stale_fix` / `reach_pad`; the measurement is in
`microduck_local/README.md` and the trap is AGENTS.md rule 7.

**So the chain closes:** the wide module needs 640 px → 640 px needs 5 Hz →
5 Hz needed a stale-pose fix, which is now in and costs nothing at 10 Hz.
The remaining hardware question is only whether the NPU's p95 at 640 px
really lands near 230 ms, which §5.1 should answer with a measurement rather
than the 4× scaling assumed here.

## 5. Open questions, in priority order

1. **What YOLO input size can the NPU sustain, and at what rate?** 320 or
   640? §3 says this decides whether the new camera helps or hurts tidying.
   Note the sensor is not the constraint — 1920 px across is six times the
   320 the detector consumes. **The specific number to measure: p50 and p95
   for a 640 px input.** §4 assumes 4× the 320 px figures (~100 / ~230 ms),
   which puts 640 px at 5 Hz just inside budget and 640 px at 10 Hz outside
   it. If the real p95 is better than 230 ms, 640 px at 10 Hz opens up and
   nothing else has to change; if it is worse, 5 Hz is the only way to get
   the resolution and that path is now clear.
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
