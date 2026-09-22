# Microduck local-training workspace

This repo is a community harness for training and watching RL policies for the
[Microduck](https://pollen-robotics.com/microduck) — Pollen Robotics' ~25 cm,
~800 g bipedal robot (14 Dynamixel XL330 servos, IMU, 50 Hz control) — **on an
ordinary Mac, no CUDA GPU required**. It is not affiliated with Pollen Robotics.

The workspace is four side-by-side checkouts. This repo tracks two of them; the
two upstream Pollen repos are cloned next to them (they are in `.gitignore`):

| Repo | What it is | Stack |
|---|---|---|
| `microduck_local/` | **This repo.** Local CPU-MuJoCo + Stable Baselines 3 PPO prototyping harness — same 61-obs contract and MJCF as microduck_rl. Includes `duck-lab`, the streaming backend for duck-viewer, and the 🎓 teach-a-trick training loop. | Python 3.12 / uv |
| `duck-viewer/` | **This repo.** Next.js + react-three-fiber browser viewer — many policies/checkpoints walking side by side, live over WebSocket from `duck-lab`. | Next.js / TS |
| `microduck/` | Upstream: the robot's onboard software, shipped ONNX policies, docs. Clone from `pollen-robotics/microduck`. | Rust workspace |
| `microduck_rl/` | Upstream: the official GPU training stack (MuJoCo Warp + mjlab + PPO), BAM actuator sim2real recipe, ONNX export. Clone from `pollen-robotics/microduck_rl`. **This is the sim2real recipe; this repo is the prototyping loop.** | Python 3.12 / uv |

## Setup

One command does all of the below (upstream clones at the pinned shas, the
shipped policies from the Hub, `uv sync`, `npm install`, a smoke test) on a
Mac or Linux:

```bash
git clone <this repo> microduck-workspace && cd microduck-workspace && ./scripts/setup.sh
```

By hand:

```bash
git clone <this repo> microduck-workspace && cd microduck-workspace
git clone https://github.com/pollen-robotics/microduck
git clone https://github.com/pollen-robotics/microduck_rl
# The contract, golden-bit and symmetry tests are measured against specific
# upstream models and policies — the same shas CI pins (.github/workflows/tests.yml).
# (The golden bits are per platform, tests/goldens/; a platform without a
# recording skips them — record yours with MICRODUCK_RECORD_GOLDENS=1.)
git -C microduck_rl checkout badc4e7ffe5507fd7acb1a21487bd2925c1afe5a
git -C microduck checkout 2c61dcc1f03440541cdc0729f7a375b2a9ea3005
cd microduck_local && uv sync            # needs https://docs.astral.sh/uv/
# The shipped policies left the microduck repo for the Hub on 2026-09-03
# (ef4becf); the pinned sha above still vendors them, any later one does not.
# This Hub revision is byte-identical to the vendored set — see setup.sh.
uv run hf download pollen-robotics/microduck-policies \
  alpha_walking.onnx alpha_stand.onnx alpha_sitstand.onnx alpha_ground_pick.onnx \
  ball_kick_left.onnx ball_kick_right.onnx roller.onnx roller_crouch.onnx roulade.onnx \
  --revision 088524a64e2557dc453256b6071dbb9d23888802 --local-dir ../microduck/policies --quiet
cd ../duck-viewer && npm install
```

`microduck_local` finds the MJCF models in `../microduck_rl` (override with
`MICRODUCK_RL_DIR`) and the shipped reference policies in
`../microduck/policies/` (downloaded from
[pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies)
by `setup.sh`; upstream no longer vendors them).

## Read the repo-local docs first — they are authoritative

- `microduck_local/README.md` — what the harness is/is not for, every command,
  the measured performance story.
- `microduck_local/AGENTS.md` — **the training playbook for agents**:
  invariants, reward-design rules, verification discipline. Read it before
  touching rewards, observations, or training code.
- `duck-viewer/README.md` — viewer architecture and the GPU pitfalls already hit.
- `microduck_rl/AGENTS.md` (upstream) — the full sim2real playbook the local
  harness mirrors.
- `.claude/skills/render-rollout/SKILL.md` — how to *look* at what a policy
  actually does (works as plain documentation for any agent, not just Claude).
- `.claude/skills/record-world/SKILL.md` — how to *record* a world scenario
  (living room, playroom, soccer pitch) to video + a contact sheet + an events
  log, headless under a seed, and read what the ducks did. Debug the `/sim`
  page with this, not by describing what a browser tab looked like.
- `docs/mars-roadmap.md` — the plan for a THIRD body, Innate's MARS (a
  wheeled base with a 6-DoF arm): what is measured, the `Body`/`RobotSpec`
  split that makes the next robot a registry entry, and the phases with
  the number that settles each. Read it before adding any robot.
- `docs/roadmap.md` — the working list of experiments: what to run next, the
  command for each, and the number that would settle it. Read it before
  starting anything open-ended, and **write the answer back into the item**
  when you finish one — a negative result is worth as much as a positive one,
  and this is where the next person finds out it was already tried.

## What runs where

- **A second body:** the Unitree G1 (29 joints, 99-obs) trains, exports,
  renders and appears in the lab beside the ducks — `robots/spec.py` is the
  seam, `uv run fetch-g1` the setup. A lab contract, not a sim2real one.
- **Any Mac (tuned on Apple Silicon), CPU-only:** everything in this repo —
  training (`train-walk`, `train-behavior`), eval, ONNX export, rendering,
  the lab + viewer. Linux works too (set `MUJOCO_GL=egl` for offscreen
  rendering); the MPS update path is Mac-only and auto-disables elsewhere.
- **Linux / cloud CPU boxes:** the same commands, under a different torch
  thread policy that `machine.py` detects (cores read from CPU affinity and
  the cgroup quota, not `os.cpu_count()`, so a container gets its real
  budget). Mac behavior is unchanged by it — see "Cloud and Linux
  training" in `microduck_local/README.md` for the measurements and the
  `--compare-profiles` A/B.
- **Needs a CUDA GPU:** the upstream `microduck_rl` MuJoCo Warp training —
  the final sim2real step once a behavior prototyped here is worth it.
- **Runs on the robot:** ONNX exported by `export-walk` is drop-in compatible
  with the deployment contract, but ship policies retrained on the official
  stack — see "sim2real honesty" in `microduck_local/AGENTS.md`.

## Command crib

```bash
# --- microduck_local (run from microduck_local/) ---
uv run --with pytest pytest tests/            # contract tests — run before training
uv run train-walk --envs 32 --steps 3_000_000 --run-name my-run
uv run export-walk runs/my-run && uv run eval-walk runs/my-run/policy.onnx
uv run train-behavior one_leg                 # teachable tricks (behaviors/)
uv run train-brain --run-name follow-v6 --steps 2_000_000 --variety \
    --title "Follower v6" --description "what it tests, and later what it found" --group shipped-followers
uv run describe-brain p-n256-s31 --title ... --description ... --group capacity   # name a BRAIN after the fact
uv run describe-run <run> --title "Front kick (G1)" --note "8/8 hold 20 s, apex 0.62 m" --pick  # name a WALK/TASK run
uv run describe-run <run> --backfill          # derive its title/description from run.json
uv run render-rollout --policy runs/my-run/policy.onnx --behavior stand --out /tmp/rr
uv run record-world pitch-2v2 --seconds 30 --out /tmp/rw   # world video + sheet + events.txt
uv run record-world playroom --brain d0=tidy --camera follow:d0 --seconds 60 --out /tmp/rw-tidy
uv run machine-facts                          # cores + thread profile for THIS machine
uv run bench-envs                             # the right --envs for THIS machine
uv run bench-envs --compare-profiles          # mac profile vs linux/cloud profile, interleaved

# a SECOND robot (the Unitree G1) — see microduck_local/README.md
uv run fetch-g1                               # MJCF + meshes + walker.onnx into .cache/
uv run bench-walk --robot g1                  # it costs ~2x the duck per step
uv run distill --robot g1 --teacher .cache/unitree_g1/walker.onnx --run-name g1-clone
uv run train-walk --robot g1 --envs 32 --init-from runs/g1-clone --run-name g1-walk
uv run export-walk runs/g1-walk               # obs[1,99] -> actions[1,29]
uv run duck-lab --checkpoints runs/my-run    # + viewer below → watch it in the browser
uv run duck-lab --world playroom              # world mode: rooms, sensors, brains (the /sim page)
uv run eval-tidy --seeds 3 --seconds 300      # Track 12 benchmark; trace-tidy / walker-facts to debug it
uv run eval-tidy --robot mars --seeds 3 --seconds 300   # the same room, tidied by MARS's ARM (brain tidy_arm)
uv run eval-pitch --seeds 4 --seconds 300     # soccer benchmark: two chase brains, shipped kicks, goals
uv run eval-pitch --seeds 12 --per-side 3 --out runs/poacher.jsonl --tag poacher   # resumable: re-run the SAME
                                              # command after any interruption and it continues (--seed0 extends)

# --- duck-viewer (run from duck-viewer/) ---
npm run dev                                   # then open the printed localhost URL
npm test                                      # vitest: the canvas arithmetic (lib/*.test.ts)

# --- upstream microduck_rl (run from microduck_rl/, GPU) ---
uv run train <TASK> --env.scene.num-envs 64 --agent.max_iterations 5   # smoke test
uv run scripts/infer_policy.py --walking ../microduck_local/runs/my-run/policy.onnx
```

## Conventions that matter

- Policies are hot-swapped on the robot behind a **shared 61-dim obs /
  14-action ONNX contract** — never change the obs layout per-task; zero-pad
  unused command slots.
- Always deploy ONNX from `export-walk` / upstream `scripts/export.py` — they
  bake the obs normalizer in. Never hand someone a raw checkpoint.
- Prototype here → port the env design to an mjlab cfg → retrain on GPU with
  the official stack. The local harness is for minutes-long iteration loops,
  not for the policy you put on hardware.
- Every trained run carries a `title`, `description` and `group` in its
  `brain.json` (`train-brain --title/--description/--group`, or
  `describe-brain` later). The run name is an identifier and never changes;
  the title is what the `/train` page and the `/sim` brain menu show. Write
  the finding into the description when the experiment resolves — see
  "Every run is a record" in `microduck_local/AGENTS.md`.
- A walk or task run (every `train-walk` / G1 run) carries the same thing in
  `runs/<name>/record.json` — written automatically by the trainer, edited
  with `describe-run`, and shown by the lab's policy palette as the chip's
  title and tooltip. **`--pick` names the stage of a curriculum chain worth
  using**: without it the palette's ▶ and ⤓ point at the LAST stage, which in
  the G1 kick chain was the worst one (3/8 seeds against 8/8 a rung earlier).
  Only a measurement may set it.
- **Start training through the lab, not the CLI** — `POST /teach` on
  `127.0.0.1:8788` runs the recipe's curriculum and streams it to the viewer
  at `localhost:63317`, so the person who asked can watch the robot practise
  and stop it. A bare `uv run train-*` is invisible to them. Batteries and
  paired A/Bs are the exception (use `MICRODUCK_RUNS_DIR`). Full rule in
  `microduck_local/AGENTS.md`.
- Before claiming anything about a trained policy, **render it and look**
  (`render-rollout`) — reward curves and eval sums have repeatedly lied here.
  The same rule for world mode: before claiming what happens in a room or on
  the pitch, `record-world` it and read `sheet.png` + `events.txt`.
- If rollouts never contain the skill you're paying for, **fix the physics
  curriculum, not the reward** — an unsampled state's value is never learned.
  The full lesson (and the servo-ladder pattern that solved it) is in
  `microduck_local/AGENTS.md` under "Reward design rules".
- When a new task subclasses an existing one, **retarget an inherited reward
  term, never switch it off** — and check it isn't flat where the policy
  starts. Zeroing a weight because its default target is wrong for the new
  task is the most repeated mistake here (five retrains in one day); it leaves
  a proxy that failure also satisfies, and the robot sags, folds or topples
  into it. Same section of `microduck_local/AGENTS.md`.
