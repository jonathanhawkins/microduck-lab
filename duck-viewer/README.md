# duck-viewer

Next.js + Three.js (react-three-fiber) web viewer for Microduck policies — watch
many training runs walk side by side in the browser instead of the native MuJoCo
viewer. The pattern is lifted from jenga-stacker's web viewer (mesh geometry
extracted straight from the compiled MuJoCo model, no asset pipeline), upgraded
from recorded replays to live WebSocket streaming.

```
┌──────────────────────────┐  GET /scene (meshes + colors, once)  ┌────────────┐
│ duck-lab (Python)       │ ───────────────────────────────────▶ │ Next.js    │
│ microduck_local          │   WS /ws ~25 Hz body poses + stats   │ duck-viewer│
│ one CPU-MuJoCo env per   │ ◀─────────────────────────────────── │ r3f canvas │
│ policy, real-time 50 Hz  │   {"cmd": [vx,vy,wz]} / {"reset"}    │ + HUD      │
└──────────────────────────┘                                      └────────────┘
```

## Run it

```bash
# 1. the lab (from microduck_local/) — each arg is one duck
uv run duck-lab --checkpoints runs/first-gait ../microduck/policies/alpha_walking.onnx

# 2. the viewer
cd duck-viewer && npm run dev     # then open the printed localhost URL
```

```bash
npm test        # vitest: the canvas arithmetic (lib/*.test.ts). CI runs it.
```

Unit tests here cover the maths that decides *which pixels get asked for* —
they cannot tell you the page looks right. For that, look at it:
`.claude/skills/sim-smoke` brings the lab and viewer up and screenshots
`/sim`.

`?lab=host:port` on the page URL points it at a different lab (a scratch
server on another port, a scratch lab on another port — the lab binds loopback, so same machine); default `127.0.0.1:8788`.

Duck sources: a run dir (`runs/my-run`, uses/exports `policy.onnx`), any
`.onnx` file (shipped alphas work), or `--checkpoints <run>` to line up one
duck per training checkpoint — watching a policy learn across 500k-step
snapshots is the point of this thing.

**Ducks are driven only by their RL policies** — no teleop. Walking policies
follow the server's auto demo script (their velocity-command input, the same
interface the real robot's gamepad uses); trick policies (`teach-*`, 🎓, 🤝)
get zero commands and just do their trick. The **keyboard flies the camera**,
Maya/Blender-style: drag to orbit, scroll/two-finger-vertical to zoom,
**two-finger horizontal swipe** to slide laterally (natural-scrolling
direction; browser back-swipe is suppressed over the scene, and panels keep
native scrolling — one swipe is locked to one axis, `lib/swipe.ts`),
**A/D** slide, **W/S·↑↓** dolly, **←/→** orbit, **E** up / **Q** down,
**Shift+R** reset view — all held keys move smoothly (velocity × dt).
The one non-camera key is **R**, which **restarts the sim** (`{"reset": true}`
to the lab): every duck's episode drops back to step zero at the same moment,
which is what makes a side-by-side comparison legible. **Clicking a duck**
(or its HUD row) **selects it** — amber floor ring + highlighted row — and
**Delete/Backspace removes it** (same `remove_duck` message as the row's ✕);
**Esc** or an empty-floor click deselects. Selection uses the same projected
screen-radius hit test as policy assignment, not raycasting. Selecting a duck
that runs one of our trained runs also **loads that run into the 🎓 teach
panel** (`POST /teach/load` — recipe card + sliders in finished state, so
"✨ fine-tune" continues from that exact brain); shipped Pollen policies are
skipped quietly, and nothing is loaded while a job is actively training. HUD shows per-duck
episode time, fall count, reward-rate EMA (dimmed for trick ducks — it scores
the walking recipe), a system-stats strip (cpu/mem/training steps-per-second),
and collapses to a pill via its — button. Its 🏷 button toggles the floating
duck name labels (persisted; the selection ring stays either way). Labels
stack below every overlay panel (panels sit at z-index 20, labels top out
at 10), so a crowded farm can't scribble text over the HUD or chip lists. (Manual drive commands still exist
at the protocol level — `LabClient.sendCmd` — for a future gamepad page;
the UI deliberately doesn't expose them.)

## The panels

**One robot, everywhere.** With a second body in the lab (the Unitree G1) the
🧠 palette, 🎓 teach and 🎬 animate each carry a `🦆 duck` / `🤖 G1` switch —
and they are ONE switch (`lib/activeRobot.ts`): pick the G1 in any of them, or
click a G1 on the stage, and the palette lists G1 policies, teach offers G1
tricks and animate poses a G1. Two deliberate exceptions: the palette's `all`
is its own list-only override, and animate follows a switch made elsewhere
only while its clip is untouched — switching bodies starts a fresh clip, so
an authored pose pins the editor to its body. A one-robot lab shows no switch.

- **🧠 policies** (top-right): every assignable brain — shipped Pollen
  policies, local runs, checkpoints. **Our runs are grouped by the trick they
  practised** (the lab's `trick` field): each trick shows its measured pick
  (`describe-run --pick`) and its newest run, and keeps the rest behind
  "▸ N more" — a seed battery no longer buries everything else. Only a
  measurement outranks recency. Long titles keep their END visible
  ("Last …, seed 2)"), because that is where a battery's runs differ; a
  filter shows every match, folded or not. Drag a chip onto a duck (or click to arm,
  then click a duck) to hot-swap its brain mid-stride; drag it to empty floor
  — or just **double-click the chip** — to spawn a fresh duck running that
  policy; drop it (or armed-click) **on the 🎓 teach panel** to load that
  run's recipe there for refinement instead. Auto-refreshes when a training
  run finishes. Hovering one of **our runs** reveals a ✕ that deletes that
  run's training data from disk — the exported policy, its checkpoints and
  its progress log; a curriculum chain deletes as one family, all stages at
  once. It always confirms first (naming the run dirs and the space it
  frees), and the lab refuses outright while that run's job is still
  training. Shipped Pollen policies have no ✕ — they aren't ours to delete.
- **🎓 teach** (bottom-right): the front door — no stored run needed. The
  footer asks in the order a person decides: **who** (the robot switch),
  **what** (one chip per recipe that names a `suggest` phrase, with its
  emoji; a trick that already has a MEASURED best run grows a `▶` that spawns
  a robot running it, so "what does this look like?" never means digging
  through the palette), then **how long** (`⏱ practice:` — one folded line,
  because the recipe's own plan is right for nearly everyone). Or chat a
  trick ("stand on one leg"), see the reward recipe in plain English, watch the live score curve + per-term
  bars while the 🎓 trainee duck improves snapshot by snapshot. When a run
  ends, the recipe's **weight sliders unlock**: drag them and either
  "↻ retrain" (fresh) or "✨ fine-tune" (keep what it learned, adjust) —
  that's the reward-shaping loop with no Python involved.
- **🎬 animate** (bottom-center): keyframe editor for the robot — pose the
  translucent ghost duck (sliders, or drag body parts in the scene), key poses
  on the timeline, save the clip, ⚡ train a policy to track it. The **🎮 rig**
  section on top gives game-style macro controls over coupled joints — squat,
  lean, per-leg L/R swing (a stride, feet kept level), sway, stance, twist,
  toes, look — each a fixed coupling that keeps the
  feet flat (e.g. squat folds hip pitch + knee + ankle on both legs); the ⇕
  handle drags the selected rig control (squat when none is selected) and
  parks at that control's anchor on the duck — head for look, thigh for a
  swing — wearing the control's name. A
  🦴 joints / 🎮 rig toggle picks what clicking the duck edits: one servo, or
  the part's rig control (feet→toes, thigh→swing, shin→squat, trunk→lean,
  head→look, hip yaw→twist, hip roll→sway) — the whole coupling lights up and
  the drag is geared so the grabbed part tracks the cursor; selecting a rig
  slider highlights and arms that control the same way. Rig
  slider ranges are computed live from the MJCF servo limits: the slider ends
  exactly where the first servo runs out of travel, and the tooltip names it.
  Rig directions are mutually orthogonal in joint space, so controls never
  move each other's sliders, and asymmetric hand-tweaks survive a rig drag.
  **⊕ balance** adds a centre-of-mass marker to the ghost: a ball at the CoM,
  a plumb line, and a crosshair where that line lands on the floor, with the
  soles outlined under it and the margin in millimetres read out under the
  toggle. Green once the point is inside a sole (a one-legged hold is on),
  blue while it is inside the two-foot stance (the hull of the grounded soles,
  drawn in the same colour: standing square the CoM is ~25 mm outside both
  soles and ~16 mm inside the stance, which is why the square stance stands),
  amber once it is outside everything. `POST /pose` measures it against each
  sole's real outline (the flat of the sole mesh, projected onto the floor,
  which stays true when a foot turns; the mesh's bounding box did not) and
  flags a foot held off the floor, which the readout cannot be stood on. It is
  a STATIC check (no velocity, no momentum, no ankle torque), so it answers
  "how hard is this pose to hold", not "would the duck fall": a real policy
  holds a small negative margin routinely. Off by default, and persisted like
  the other panel toggles.
- **📷 shot** (top-center, always available): one click downloads a full-res
  PNG of the current view, named after the selected duck (or `duck-lab` for a
  crowd shot) — the selection ring is hidden for the capture render, and the
  whole render→read→download runs synchronously inside the click so Chrome
  never blocks it as an "automatic" download. The panel centers at the top
  but slides right of the duck-lab HUD when that panel is wide (long duck
  names) — the HUD publishes its right edge through the ui.ts store.
- **🎥 record** (same panel, appears when a duck is selected): one click films
  the selected duck for you — the camera glides to a ¾ front shot (chosen from
  the duck's heading, then held with a slow cinematic drift; OrbitControls and
  camera keys pause for the take) and MediaRecorder captures the WebGL canvas.
  Footage is automatically clean: DOM labels/panels aren't part of the canvas,
  and the amber selection ring hides itself while filming. ■ stop uploads the
  take to the lab (`POST /captures`), whose bundled ffmpeg writes a
  full-resolution h264 **mp4** and a 480 px palette **gif** into
  `microduck_local/captures/`; the panel then offers ⬇ downloads of both.
  Takes cap at 60 s. Frames are pushed per RENDERED frame
  (`captureStream(0)` + `requestFrame()` — automatic capture rides the
  compositor and records almost nothing in a throttled tab), so keep the tab
  visible while recording; a take where the scene never rendered is refused
  with a message instead of producing a 0.1 s "video".
- **🎥 record / 📷 shot on `/sim`** (top bar, `SimRecord.tsx`): the same
  take without the framing glide — it films whatever you are looking at, and
  the camera stays yours for the duration (orbit, fly keys, chase a duck by
  hand), so a soccer scrum or a fall at the basket is recorded from the angle
  you chose. Files are named `sim-<scenario>-<stamp>`; ■ stop hands you the
  same ⬇ mp4 / ⬇ gif links. The footage is the WebGL canvas only (no panels,
  labels or scrub bar); for a clip WITH per-duck state, falls and score burned
  in, run `uv run record-world <scenario>` in `microduck_local/` instead.
  With 🔊 quacks on, the ducks' voices go into the take as an opus track and
  come out as aac in the mp4 (the gif is silent, as gifs are) — which is the
  only way to get them into a file, since `record-world` renders headless in
  MuJoCo and never sees the browser's audio.
- **helpers**: ＋ on the training row spawns a 🤝 helper duck — another
  viewer of the same live policy. Helpers do **not** add trainer workers
  (that *lowered* steps/s while the lab was open). ✕ removes it.

Panel states, chat history, and the camera persist in localStorage; the duck
roster itself persists server-side (`microduck_local/lab-state.json`) across
lab restarts.

## `/sim`: the world page

`http://localhost:63317/sim` renders the lab's **world mode** (start the lab
with `uv run duck-lab --world living-room`, or load a scenario from the
page's picker). The lab page's `🦆 duck lab` panel carries a `sim →` link to
get here, and this page's `← lab` goes back. One room, many ducks, and what
each duck senses:

- **Scenario picker + load** (top bar): built-ins and anything saved under
  `microduck_local/scenarios/`. Walls, static boxes and the floor come from
  the scenario JSON; balls and free boxes stream their poses at 25 Hz.
- **Sensor overlay** (`T` — the button names the selected body's own range
  sensor): for a duck, **ToF overlay** — one dot per zone of its 8×8 depth
  matrix, at the depth the sensor *reports*, colored near→far amber→teal,
  plus the four corner rays from the aperture. For a wheeled body with a
  planar scanner (MARS), **LiDAR overlay** — all 360 rays from the
  `base_laser` aperture in the base's heading frame: a short tick at each
  hit coloured by range, a faint full-length ray where the beam reached
  nothing, and the 45° front sector its brains read drawn as a tinted fan
  with the eight winning rays brightest. One `BufferGeometry` for the lot.
  Select a body (click, or `1`–`9`) to see only its rays.
- **Pitch panel** (on a soccer scenario): the score, the kickoff countdown,
  goals split into kicked and walked in, and — under a rule — the three
  per-team rates the benchmark actually judges by. Goals are about 2.5 a
  run and cannot resolve a change (146 seeds for a 25% shift), so watching
  them tells you almost nothing: `possession` (seconds a minute one of ours
  is nearest the ball inside 0.25 m) is the cheap screen at 9 seeds,
  `advance` (metres a minute the ball is carried toward the goal that team
  attacks) is the discriminator at 43, and `signed` is the same thing with
  backward motion charged for — the one churn cannot inflate, shown in red
  when a team is losing ground. They come off the same `PitchMetrics` the
  battery uses, so a number on screen is the battery's number.
- **Chase overlay** (under `T` too, on a pitch): what each chase brain
  thinks about the ball, drawn on the floor in its own odometry frame like
  the map — an orange line from the ball track to where the brain predicts
  it will stop (its head yaws that way and its hunt aims there), a grey
  ring on the ball memory its search would walk to, a teal ring on its
  line-up / push spot, and a violet ring on the ball-sized blob its 8×8 ToF
  sees on the floor at its feet. That blob (`tofBall`, a bearing and a range
  in the duck's heading frame) covers the last 30 cm, where a floor ball
  drops out of the head camera's frame; it is measured *off* as a ball
  source for the brain — at the feet it is as often the other duck's foot,
  and a line-up on a foot is a fall — but it is computed, so it is drawn.
  It is placed from the duck's odometry pose, so a duck without one gets no
  ring. Every chase duck, the selected one bright. The inspector carries
  both as rows: that blob's bearing and range, and `bump` — how long
  since this duck's feet last touched another duck or a person
  (contacts here, the IMU and the servo loads on the robot), amber while it
  is inside half a second. That is the window a bumped duck stands through
  instead of turning in place, which took 3v3 falls from 5.00 to 1.75 a
  run. The pitch panel splits the goals into kicked (within 4 s of a kick)
  and walked in, the same attribution `eval-pitch` prints.
- **Inspector** (right): the selected body's senses, **one block per channel
  it actually has** (`sensors` keys on the frame — never a ToF placeholder on
  a robot with no ToF), painted straight off the stream with the frame age
  (amber when stale), a noise preset select (`ideal` / `datasheet` /
  `hostile`) and which brain is steering it.
  - `tof` → the duck's 8×8 heatmap, hover a zone for its range.
  - `lidar` → the **polar plot**: nose UP, all 360 returns coloured by range,
    range rings to the device's 6 m, the 0.15 m disc it cannot resolve
    inside, the footprint circle whose returns are dropped as the robot's own
    arm, and the 45° front sector out to the ToF's own 4 m. It is centred on
    the LASER, with the chassis drawn 76 mm ahead of it, because the ranges
    are the laser's — reading them off the chassis origin puts every obstacle
    76 mm too far away. Under it, the **eight bearing columns** that sector is
    binned into: the 8×8 frame `wander` and `follow` actually steer on,
    computed in the browser the way `sensors/lidar.tof_from_lidar` computes it
    (`lib/lidar.ts`, whose vitest cases assert the Python's own numbers). The
    plot's note says what the scan is blind to: one horizontal slice at 0.17 m,
    so the basket's 6 cm rim and the toys on the floor exist only in the head
    camera's `sees:` list.
  - `det` → what the head camera found (class, bearing, range), and the
    camera inset's boxes.
  - `gripper` / `arm` → the claw's load against its 1.0 N·m hold threshold and
    the arm's achieved joint angles. Drawn when the frame carries them; the
    lab does not send them yet (`lib/sim.ts` `GripperPayload` / `ArmPayload`
    name the shape).
- **States** (`G`): the selected duck's brain as a graph — every state it
  could be in, grouped by what it is *for* (find the ball / go to it / hit
  it / stay safe / team duties), the one it is in now lit, the ones it has
  reached this run bright, and the rest dim. Hover a chip for what the duck
  does there and how long it has spent there. Under it, the last dozen
  states as a breadcrumb. **Drag the corner grip to resize it** — chips,
  names and arcs all scale together (0.7x–2.6x, remembered; double-click the
  grip for the default). The grip sits on whichever bottom corner actually
  moves as the panel grows, and past the window height the drawing scrolls
  inside rather than carrying its own title bar off the top of the screen.

  Arcs are *masked out* of the chips and the group headings, not merely
  drawn under them: z-order alone still left a 2 px curve crossing 9 px
  heading text, which wins however the painting order reads. A move is
  therefore drawn only in the gaps between rows — which is where it is
  legible anyway.

  The nodes are declared by the lab (`brain/graph.py`, shipped once in the
  world-info message and pinned to the code by `tests/test_brain_graph.py`,
  which walks each brain class's AST for every string it can assign to
  `self.state`). **The edges are not declared anywhere** — none of these
  brains is a written-down state machine; `Chase` sets `self.state` from
  twenty-odd places in one long `step`. So an arc is drawn the first time
  this duck actually makes that move, thickens as it repeats, and fades as
  it goes stale. What you are watching is this run's trajectory, not a spec.

  Three things it shows that the text readout does not: a duck stuck in a
  two-node loop is a pair of fat arcs long before it is obvious in the 3-D
  view (a 2v2 supporter oscillating `support ↔ retreat` is the usual one);
  a permanently dim chip is a state this scenario never reaches (`block`
  needs a keeper, `duel` needs an opponent); and switching the duck to a
  learned brain collapses the whole picture to two chips read off one slot
  of the observation, which is the most honest thing this page says about
  the difference between the two kinds of brain.
- **Panels go where you put them.** The inspector, the state graph and the
  head-camera inset drag by their title strip and remember where they were left
  (localStorage, clamped back into view on resize); double-click the strip
  to re-dock. Untouched, they sit at their designed spots below.
- **Cam** (`V`): top-left, under the top bar (under the pitch scoreboard on
  a pitch; hidden while the editor holds that corner), what the selected duck's head camera
  sees, rendered from the `head_camera` site at the detector's field of view
  (62°×48°), with the detector's output drawn over it as boxes — a bearing,
  an elevation and an apparent width per thing it found, and nothing else
  about the picture. That is what a brain gets, and why a floor ball
  vanishes from the frame in the last 0.3 m unless the head pitches down.
  The inset is rendered from the camera pose the frame was *captured* from
  (the stream carries it), not from where the head is now: at 10 Hz plus
  latency the walking head moves the picture by up to a fifth of its width
  before a brain gets the frame — measured, and the lag a brain acts on.
  It is a second, scissored pass of the same scene in the same canvas (a
  priority-1 `useFrame` takes the render loop over), not a render target and
  a readback; the sensor drawings (ToF dots, rays, map) sit on a layer the
  head camera does not see. The scissor rectangle goes to three in **CSS
  pixels** — `setScissor`/`setViewport` scale by the renderer's pixel ratio
  themselves, so measuring the box in device pixels applies it twice. That
  shipped: on a retina Mac the pass landed 1.5× off the panel, so the inset
  showed the main orbit view straight through a transparent div (labels and
  boxes intact, no picture) and the main view came back zoomed 1.5×. The
  arithmetic lives in `lib/inset.ts` and is pinned by `lib/inset.test.ts`.
- **Drive** (`P`, then WASD/arrows, Q/E strafe): every duck takes your twist
  for 6 s after the last key; otherwise ToF-equipped ducks wander on the
  lab's `Wander` brain and blind ducks follow a demo script. `R` restarts.
- **Speed** (`[` and `]`, or the menu in the top bar): run the world at
  0.25x–8x of the wall clock. Fast-forward to reach the part worth watching;
  slow down to see a fall or a kick land. It is a sim-time budget, not a
  faster clock — the wire keeps its 25 frames a second and a fast world
  simply jumps further between them, so the bandwidth does not move
  (`playroom`, 1 duck: 25.0 frames/s and ~105 kB/s at each of 1x, 2x and 8x;
  a 6-duck `pitch-3v3` sits at ~795 kB/s, equally flat).
  What a scene actually reaches is what its physics costs: a room runs 8x
  with room to spare, a 3v3 pitch costs ~7 ms of its 20 ms tick and tops out
  near 3x. Asking for more is not an error — the loop runs flat out and the
  **RTF turns amber and reads `3.04/8x`**, the measured speed against the
  one you asked for. The lab owns the setting (`POST /world/speed`), so a
  second tab follows along.
- **The camera flies exactly as on the lab page** — same keys, same
  **two-finger horizontal swipe** to slide laterally (the shared
  `components/useTruckSwipe.ts`; drive mode takes WASD away from the camera
  but never the swipe, since nothing on the trackpad steers a duck).
- **Inspector · brain**: which brain steers the selected duck (`wander`,
  `follow`, `tidy`, `script`, or a trained `learned:<run>`), switchable live,
  its inputs with their ages (ToF, detector, the target it is tracking), its
  current intent, and — for `tidy` — picked/delivered counts and the toys it
  gave up on. `head` toggles whether the brain's head intents are applied.
- **Persons + possess**: scenarios can carry walking persons (mocap capsules
  on waypoint paths). Possess one from the inspector and drive it with the
  same keys; the ducks keep their brains. This is how follow-me is tested.
- **Editor** (`E`): place walls (two clicks), boxes, balls, ducks, persons,
  toys and the basket on the floor, set each duck's brain, then save-and-load
  under a name (`PUT /scenarios/{name}`; built-ins are read-only, so a draft
  of one saves as a copy). `make a pitch` gives the room the lab's boards —
  the 15 cm cove and, for a rectangular room, 30 cm chamfered corners — and
  toggling back takes them away, so a pitch drawn here is the one `/sim`
  plays on. The scene menu in the top bar splits into
  `built in` and `saved by you`; each of your saved scenes carries a `✕`
  that deletes it in place (`DELETE /scenarios/{name}`, after a confirm) —
  built-ins have none, and the live world keeps running whatever it loaded.
- **What the brain sees** (in the inspector, for a duck on a `learned:*`
  brain): the network's last decision, live. A strip of 80 bars is the
  observation as the network gets it — ToF cells, ages, the tracker's
  target slots, the last action, speed, the coasting/yaw/confirmed flags —
  each scaled by its own range (hover a bar for the slot and value), with
  the target block also in words. Under it, one gauge per action on a track
  that spans the brain's own bounds; an action pinned on a bound shows `⊣`
  in amber. The exported graph clamps the network's output itself, so there
  is no visible "ask" past the edge — the pin is the ask, and a brain pinned
  every decision, target or no target, is the saturated-mean trap showing
  itself while the duck walks. Wire shape: `brain.view` in the frame
  (`runtime.brain_view`): the observation, the action before and after the
  intent clip, and the bounds.
- **Map** (`M`): the selected duck's occupancy grid, painted on the floor —
  what it believes the room is, from its ToF frames and its own odometry
  (amber occupied, teal free). Switch its `odom` preset in the inspector to
  `datasheet` or `hostile` and watch the map smear like a real robot's.
- **🔊 quacks** (top bar, `Shift+M`): the ducks talk. A follower chirps
  while it has its person in sight, coos when it is in the distance band and
  standing, asks a rising *where did you go* while it has lost them, and
  gives a two-note "there you are" when it finds them again after a real
  absence (1.5 s — the detector drops single frames constantly, and a
  fanfare on every dropped frame is just noise). Every duck has its own
  pitch off its id, and each utterance wobbles a few percent so a flock does
  not sound like one loop. **Only followers have anything to say**: the
  `follow` and learned-follower state graphs (`lib/quack.ts` maps graph +
  state → voice), so a soccer battery or a tidy room left running is silent.
  No assets — it is a triangle with a sine an octave up, a 7 Hz trill on the
  held notes and a rounding lowpass, ~8 nodes a voice (`lib/quackaudio.ts`);
  the reference is a duckling's peep, and the first version, a sawtooth
  through a sweeping bandpass, was a fair mallard and came back as "cuter". The toggle persists, a
  paused or scrubbed world is silent, and the voices ride into 🎥 takes as
  an audio track when they are on. **The bill moves with the voice**: the
  hinged `mouth` body the lab already streams (world/compose.py `split_jaw`)
  gets a viewer-side rotation on its mesh, timed off the same note table
  the sound is rendered from (`lib/mouth.ts` — opens in 30 ms, hangs open
  through the note, eases shut 70 ms after it; a reunion is three flaps).
  It opens to the WIDER of the voice and the real servo's `mouth` fraction,
  so a duck carrying a toy keeps its grip while it chirps and a silent duck
  shows exactly the physics. **Mute is the sound only** — the scheduler and
  the bills keep going, so a muted room still shows the ducks talking rather
  than reading as if they had stopped. Viewer-side on purpose: a round trip through
  the lab's servo would put the bill ~150 ms behind a 140 ms chirp, and
  this way a 🎥 take records mouth and sound in step.
- **Perf** (top bar): the lab's cost per 20 ms tick as physics+policies +
  sensors + frame encode, next to RTF and kB/s.
- **Pitch score** (top-left, `pitch` / `pitch-2v2` / `pitch-3v3`): goals per
  side while the `chase` brains go after one ball; in a team each duck's
  inspector shows its role (attack / support) and the team's blackboard.
- **Tidy score** (top-left, `playroom` scenario): toys in the basket, what the
  duck is carrying, picks and deliveries — the same numbers `eval-tidy`
  prints headless.
- **Timeline** (bottom): the lab keeps a ring buffer of recent frames; pause
  and scrub, or save the buffer as a recording.
- Protocol: `lib/sim.ts` (`/ws/sim` frames, `/scenarios`, `/world`).

The ducks are rendered by the main page's `Duck` component unchanged (merged
geoms per body, DOM labels), which is why world frames carry the world body
first, as `GET /scene` lists bodies. Everything else in the room is
`components/SimStage.tsx`, which dresses the scenario JSON without changing
what the lab simulates:

- **Pitch** (`goal_width > 0`): a mown grass floor with the markings sized to
  the room — touchlines, halfway line, centre circle, goal areas, penalty
  spots, corner arcs — a dark apron outside white rink boards with an amber
  stripe, and goal frames with nets on both short walls exactly where
  `World` counts a goal. A pitch with a `cove` (the lab's all have one,
  15 cm) gets a quarter-round along the base of the boards, cut flat at the
  goal mouths, so a ball rolls up it and back into play; its chamfered
  corners are walls like any other. The ball wears a 32-panel skin so you
  can see it roll.
- **Rooms** (`living-room`, `playroom`, `follow-me`, anything you draw in the
  editor): oak planks, plaster walls with a baseboard and a cap, a rug in the
  middle of the room, bevelled furniture on soft footprint shadows, a wicker
  basket with a bound rim, studded bricks / bevelled blocks / rolled socks.
- **Grounding without shadow maps**: one instanced mesh of multiply-blended
  contact blobs follows every duck, ball, toy, box and person (fading and
  spreading as the thing lifts off the floor), and a darkening strip runs
  along every wall base. A procedural `RoomEnvironment` PMREM on
  `scene.environment` gives the shells and the ball their highlights.
- Every texture is painted on a canvas at load (no image assets, no
  fetches); the scenario-sized ones are rebuilt only when the floor, the
  walls' extent or the goal width change, not on every edit click.

![the pitch](../docs/media/sim-pitch.jpg)
![the living room](../docs/media/sim-living-room.jpg)
![the playroom](../docs/media/sim-playroom.jpg)

## `/train`: brain training runs

`http://localhost:63317/train` charts `train-brain` runs live.

`train-brain` is a plain CLI process — it never talks to the lab — so a brain
run used to be invisible while a `/teach` job was watchable. The lab's
`GET /brains` reads the artifacts the trainer already writes
(`brains/<run>/brain.json` and `progress.jsonl`) and this page polls it every
2 s. Nothing here can start, steer or stop a run: it is a read of the disk.
That also means it picks up a run started before the page was opened, and
keeps the curve of one that has already finished.

- **Run list** (left, the only scrolling region on the page — the page itself
  is fixed to the viewport). One card per directory in `brains/`: progress
  bar, steps done against the budget, last reward, elapsed, ETA, steps/s, and
  the contract tags from `brain.json` (`variety`, `obs v2`, envs, seed). A
  live run is marked `● live`; a finished one that exported is `shipped`.
- **Chart** (right). Bold line is a 9-rollout trailing mean, faint line the
  raw per-rollout value, toggled between episode reward and episode length.
  Hover for a crosshair: a rule at the hovered step and a readout of every
  charted run's value there. A run that had already stopped by that step is
  greyed and labelled with the step it ended at, rather than showing its
  final value as though it were current.
- Click a card — anywhere on it, the swatch included — to add or remove that
  run from the chart; a bordered, bright card is one that is charted. `all`
  and `none` are separate buttons — the lit one is the current
  state. The page opens on ONE run (the live one, else the first card): the
  palette holds six colours, so forty-odd curves on the same axes come out
  as an unreadable band.
- **What is different.** The first charted run is the *baseline* and its
  card shows the whole recipe from `brain.json` (batch, lr → lr_end, epochs,
  arch, legacy hparams, polite, the git sha it trained at, …). Every other
  card shows **only the knobs it changed**, as `lr_end 3e-5 ← 3e-4`, or
  `= <baseline>` when the recipe is identical — so `ab-batch` against
  `ab-batch-lr` reads as one chip, not two identical tag rows. A flag the
  viewer has never heard of still diffs, under its raw key
  (`lib/train.ts`: `recipeDiff`).
- **Where the shipped brain came from.** `select-brain` probes every
  checkpoint on the follow benchmark and ships the best one, which is
  routinely not the end of the run (`ab-batch` ships step 751k of 2M). The
  card says `◆ shipped from 751.1k · in_band 0.938 (final 0.923)` and the
  chart puts a diamond at that step, labelled with the score. The curve
  is training reward; the score is the benchmark — they are different
  numbers, and the diamond is the one that decided what went on the robot.
- **Sweep matrix** (the `matrix` tab where the chart was). Every run — not
  just the charted ones — as a table: `in_band`, `final`, `shipped@`, last
  reward, then one column per knob that *differs* between runs (a knob every
  run shares is said once under the table; a run from before the trainer
  recorded a knob shows `—` and does not make the column appear). `by
  family` collapses `p-n256-s31..36` to one row with in_band as mean ± sd
  over the seeds — the number that decides whether a knob did anything,
  since one seed moves about ±0.02 on its own. A family is the name minus
  its seed *and* one recipe: runs that share a name but were run with
  different knobs split into `p-de` and `p-de (2)` rather than being
  averaged together. Click a header to sort (knob columns sort by value,
  not by their text), a row to put it (a family row: all its seeds) on the
  chart.

A brain cloned from the repo shows *no curve* — only `brain.onnx` and
`brain.json` are committed, `progress.jsonl` stays local — and the card says
so rather than reading as a broken run.

## Notes for future work

- The scene payload is ~20 MB raw (gzipped over the wire, one-time). If it ever
  matters: quantize verts or move to binary/Draco.
- Poses stream as JSON at 25 Hz (~16 bodies × 7 floats per duck) — binary
  framing is the next lever, far from needed at this scale.
- Rendering stays deliberately light: geoms merged per body (~16 draw calls per
  duck), no shadow maps, DOM labels. The first version (560 shadow-casting
  meshes + drei `Text` GPU glyph atlases) lost the WebGL context in the
  embedded browser — keep an eye on `THREE.WebGLRenderer: Context Lost` if you
  add GPU-heavy effects back. The `/sim` stage's fidelity (SimStage.tsx) is
  all canvas textures, one PMREM and one instanced blob mesh for that reason:
  a room costs a few dozen draw calls regardless of how many ducks are in it.
  Two three.js gotchas met on the way: `MultiplyBlending` needs
  `premultipliedAlpha` on the material, and `mergeGeometries` refuses a mix
  of indexed and non-indexed parts (flatten with `toNonIndexed()` first).
- Duck colors are the MJCF material rgba streamed per geom, carried through the
  per-body merge as a vertex-color channel (so per-part color costs zero extra
  draw calls). `Duck.tsx` keeps a by-name override table (`MATERIAL_FIX`)
  for materials an export gets wrong; the 2026-09 upstream CAD re-export
  carries the right colours itself, so the table is empty today. Against a
  lab too old to stream colors the viewer falls back to one guessed color
  per body.
- The lab loop is single-threaded Python: 8 ducks × 50 Hz ≈ 5% of one core.
  Dozens of ducks are fine; hundreds would want the envs in a worker pool.
