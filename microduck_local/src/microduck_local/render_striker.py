"""Look at a striker before believing any number (AGENTS.md verification
discipline #2; `.claude/skills/render-rollout/SKILL.md` is the same idea for
walk policies, which `render-rollout` renders and this does not).

    uv run python -m microduck_local.render_striker --brain striker-v1 --episodes 2 --out /tmp/rs
    uv run python -m microduck_local.render_striker --brain chase --out /tmp/rs-chase   # the scripted control

Writes `ep<N>.mp4` for a human and `ep<N>_sheet.png` for a model — a grid of
frames from a top-down camera over the pitch with the numbers that decide
whether the rollout contains the skill burned into each tile:

    t      seconds into the episode
    ball   the ball's (x, y); the goal this duck attacks is at x = +1.5
    bx     the ball's distance to that goal — the number that must fall
    d-b    duck-to-ball distance
    kick   a kick fired in this frame's window / how many so far
    prog   metres of ball progress toward the goal so far this episode
    R      reward so far

**Read the sheet, do not trust the totals.** The three failure patterns this
was written to catch: a striker that walks past the ball (d-b never small),
one that kick-spams on the spot (kick count climbs, bx flat), and one that
carries the ball the WRONG way (prog going negative while the churn-inflated
`ballAdvance` in a battery still looks respectable).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .brain import REGISTRY
from .brain.striker import LearnedStriker, StrikerEnv, StrikerTask
from .render_rollout import (  # noqa: F401  (grid_cols: build_sheet's layout)
    build_sheet,
    grid_cols,
    sheet_indices,
)


def action_of(intent) -> np.ndarray:
    """A brain's `Intent` as a striker action, so a scripted brain can be
    rendered — and scored — in exactly the env the learned one trains in."""
    a = np.zeros(5, np.float32)
    a[:3] = intent.twist
    if intent.skill == "kick_left":
        a[3] = 1.0
    elif intent.skill == "kick_right":
        a[4] = 1.0
    return a


def make(brain: str, env: StrikerEnv):
    if brain == "chase":
        return REGISTRY.make("chase", goal=env.goal, duck_id="d0", bounds=env.bounds,
                             goal_w=env.task.goal_width)
    return LearnedStriker(brain, goal=env.goal, duck_id="d0", bounds=env.bounds,
                          goal_w=env.task.goal_width)


def top_camera(distance: float):
    import mujoco
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation = 0.0, -70.0
    cam.distance = distance
    return cam


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brain", required=True, help="brains/<name> for a learned striker, or 'chase'")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--spot-prob", type=float, default=0.0,
                    help="drill spawns in the rendered episodes; 0 (the default) is the honest open pitch")
    ap.add_argument("--near-prob", type=float, default=0.0)
    ap.add_argument("--out", default="/tmp/rs")
    ap.add_argument("--sheet-frames", type=int, default=12)
    # One frame per DECISION (10 Hz), so the mp4 runs at real time and the
    # sheet's frame index is a decision index. Software GL (osmesa, which is
    # all a headless box has here) is the cost: keep the tiles small.
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1, help="render every Nth decision")
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--height", type=int, default=300)
    args = ap.parse_args()

    import imageio.v2 as imageio
    import mujoco

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = StrikerEnv(StrikerTask(episode_s=args.seconds, spot_prob=args.spot_prob,
                                 near_prob=args.near_prob), seed=args.seed)
    brain = make(args.brain, env)
    renderer = mujoco.Renderer(env.world.model, height=args.height, width=args.width)
    cam = top_camera(distance=max(env.task.pitch) * 1.35)
    stride = max(1, args.stride)

    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        brain.reset()
        frames: list[np.ndarray] = []
        caps: list[list[str]] = []
        R = prog = 0.0
        kicked_at: list[int] = []
        while True:
            intent = brain.step(env.senses())
            a = action_of(intent)
            before = env.kicks
            _, r, term, trunc, info = env.step(a)
            R += r
            prog += info["progress"]
            if env.kicks > before:
                kicked_at.append(env._steps)
            if (env._steps - 1) % stride == 0:
                renderer.update_scene(env.world.data, cam)
                frames.append(renderer.render().copy())
                ball = env.world.ball_xy()
                caps.append([
                    f"t {env._steps * env.decide_every * 0.02:5.1f}s  R {R:7.2f}",
                    f"ball {ball[0]:+.2f},{ball[1]:+.2f}  bx {info['ball_goal_dist']:.2f}",
                    f"d-b {info['ball_dist']:.2f}  prog {prog:+.2f}",
                    f"kick {info['kick'] or '-':10s} n{info['kicks']:d}"
                    + ("  FELL" if info["fell"] else "") + (f"  GOALS {info['goals']}" if info["goals"] else ""),
                ])
            if term or trunc:
                break
        idx = sheet_indices(len(frames), args.sheet_frames)
        header = [f"{args.brain} — episode {ep} (seed {args.seed + ep}), {args.seconds:g} s, "
                  f"return {R:.2f}, ball progress {prog:+.2f} m, {env.kicks} kicks, {env.goals} goals"]
        footer = ["goal this duck attacks: x = +1.50, mouth 0.70 m wide. bx = ball's distance to it (must FALL).",
                  "d-b = duck-to-ball distance. prog = signed ball progress so far (the reward's main term).",
                  "kick = the foot asked for in this frame's decision, n = kicks so far this episode."]
        build_sheet([frames[i] for i in idx], [caps[i] for i in idx], header, footer,
                    [caps[i][3].startswith("kick kick") for i in idx], out / f"ep{ep}_sheet.png")
        imageio.mimsave(out / f"ep{ep}.mp4", frames, fps=args.fps, macro_block_size=1)
        print(f"ep{ep}: return {R:.2f} · ball progress {prog:+.2f} m · kicks {env.kicks} · goals {env.goals} · "
              f"final ball-goal {math.hypot(env.goal[0] - env.world.ball_xy()[0], env.world.ball_xy()[1]):.2f} m")
        print(f"  wrote {out / f'ep{ep}_sheet.png'} and {out / f'ep{ep}.mp4'}")
    print("\nREAD the *_sheet.png files — bx must fall for the striker to be doing anything.")


if __name__ == "__main__":
    main()
