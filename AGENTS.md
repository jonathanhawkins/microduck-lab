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

```bash
git clone <this repo> microduck-workspace && cd microduck-workspace
git clone https://github.com/pollen-robotics/microduck
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_local && uv sync            # needs https://docs.astral.sh/uv/
cd ../duck-viewer && npm install
```

`microduck_local` finds the MJCF models in `../microduck_rl` (override with
`MICRODUCK_RL_DIR`) and the shipped reference policies in
`../microduck/policies/`.

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

## What runs where

- **Any Mac (tuned on Apple Silicon), CPU-only:** everything in this repo —
  training (`train-walk`, `train-behavior`), eval, ONNX export, rendering,
  the lab + viewer. Linux works too (set `MUJOCO_GL=egl` for offscreen
  rendering); the MPS update path is Mac-only and auto-disables elsewhere.
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
uv run render-rollout --policy runs/my-run/policy.onnx --behavior stand --out /tmp/rr
uv run bench-envs                             # the right --envs for THIS machine
uv run duck-lab --checkpoints runs/my-run    # + viewer below → watch it in the browser

# --- duck-viewer (run from duck-viewer/) ---
npm run dev                                   # then open the printed localhost URL

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
- Before claiming anything about a trained policy, **render it and look**
  (`render-rollout`) — reward curves and eval sums have repeatedly lied here.
- If rollouts never contain the skill you're paying for, **fix the physics
  curriculum, not the reward** — an unsampled state's value is never learned.
  The full lesson (and the servo-ladder pattern that solved it) is in
  `microduck_local/AGENTS.md` under "Reward design rules".
