"""Distill a shipped ONNX policy into an SB3 checkpoint we can fine-tune.

Why this exists
---------------
Local training cannot learn a stable gait at our sample budget. Measured under
the BAM actuator, every policy trained here from scratch either falls (18/20)
or is stable and slow (0.13 m/s), while the shipped `alpha_walking` — ~1e9
samples on GPU with full domain randomisation — is stable AND 0.21 m/s.

We cannot fine-tune `alpha_walking` directly: it ships as ONNX, which has
weights but no optimizer state, no value function, and no VecNormalize stats.
So we clone its INPUT->OUTPUT behaviour into a fresh SB3 policy of our own
architecture, and hand that to PPO as a warm start. That turns "discover a
stable gait from nothing" into "make an existing stable gait faster", which is
the difference between a problem our sample budget cannot solve and one it
might.

What is and is not cloned
-------------------------
Only the actor's mean action is fitted. The critic starts untrained, so the
first PPO iterations after this will spend themselves learning a value
function — expect a dip before any gain. The teacher's own exploration noise
is not reproduced; we fit its DETERMINISTIC output.

Observation normalisation: the ONNX bakes its own normaliser in and takes RAW
observations, while an SB3 policy takes normalised ones. So we collect raw
observations, fit VecNormalize statistics from that sample, and train the
student on the normalised view. `export_onnx.export` later bakes those stats
back in, which closes the loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .train import RUNS_DIR


def collect(teacher: str, episodes: int, cmd_lo: float, cmd_hi: float,
            seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Roll the teacher out and record (raw obs, action) pairs.

    Collected in MicroduckWalkEnv with its OWN command sampler, not with a
    pinned forward command. The first version swept only twist_cmd[0]: the
    student then matched the teacher exactly on straight-line runs and fell
    16/16 in the walk env, which resamples lateral and yaw commands every 5 s.
    The teacher can turn; a student that has never seen a turn cannot. Cloning
    is only valid over the distribution you clone on, so the distribution has
    to be the deployment one — every command bucket the env produces (zero,
    turn-in-place, and the full vx/vy/wz mix), with domain randomisation and
    sensor noise on.

    cmd_lo/cmd_hi are unused now and kept for CLI compatibility.
    """
    import onnxruntime as ort

    from .walk_env import MicroduckWalkEnv

    sess = ort.InferenceSession(teacher, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    env = MicroduckWalkEnv(obs_noise=True, domain_rand=True, action_delay=True,
                           random_yaw=True, seed=seed)
    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        for _ in range(500):
            obs = np.asarray(obs, dtype=np.float32)
            act = sess.run(None, {inp: obs[None]})[0][0].astype(np.float32)
            obs_buf.append(obs.copy())
            act_buf.append(act)
            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break
    return np.asarray(obs_buf, np.float32), np.asarray(act_buf, np.float32)


def fit(obs: np.ndarray, act: np.ndarray, out: Path, epochs: int = 40,
        batch: int = 4096, lr: float = 1e-3, seed: int = 0) -> float:
    """Fit a fresh SB3 policy to the teacher's actions; save a warm-start run.

    Returns the final training MSE, in radians² of joint target.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from .symmetry import SymmetryPPO
    from .train_behavior import LR_END, LR_START, linear_decay
    from .walk_env import MicroduckWalkEnv

    venv = DummyVecEnv([lambda: MicroduckWalkEnv(seed=seed)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=100.0)
    # Statistics from the teacher's own state distribution — this is the view
    # the student is trained on, and export_onnx bakes it back in later.
    venv.obs_rms.mean = obs.mean(axis=0).astype(np.float64)
    venv.obs_rms.var = obs.var(axis=0).astype(np.float64) + 1e-8
    venv.obs_rms.count = float(len(obs))

    model = SymmetryPPO(
        "MlpPolicy", venv,
        policy_kwargs=dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                           activation_fn=torch.nn.ELU, log_std_init=0.0),
        n_steps=256, batch_size=1024, n_epochs=5,
        learning_rate=linear_decay(LR_START, LR_END),
        device="cpu", seed=seed, verbose=0, symmetry_coef=0.0, desired_kl=None,
    )

    norm = ((obs - venv.obs_rms.mean) / np.sqrt(venv.obs_rms.var)).astype(np.float32)
    norm = np.clip(norm, -venv.clip_obs, venv.clip_obs)
    x = torch.as_tensor(norm)
    y = torch.as_tensor(act)
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
    n = len(x)
    loss_val = float("nan")
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, seen = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = x[idx], y[idx]
            feats = model.policy.extract_features(xb)
            if isinstance(feats, tuple):        # SB3 returns (pi, vf) when the
                feats = feats[0]                # extractors are not shared
            latent = model.policy.mlp_extractor.forward_actor(feats)
            pred = model.policy.action_net(latent)
            loss = torch.nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            seen += len(idx)
        loss_val = tot / max(seen, 1)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  mse {loss_val:.5f} rad^2")

    # Exploration noise must be scaled to the CLONED policy, not to a fresh
    # one. log_std_init=0.0 means std 1.0, which utterly swamps a teacher whose
    # mean action is 0.19 rad — the first fine-tune attempt collapsed to
    # 31-step episodes because PPO was effectively sampling at random around a
    # perfectly good gait. Set it to ~a quarter of the action scale so
    # exploration perturbs the gait instead of replacing it.
    scale = float(np.abs(act).mean())
    model.policy.log_std.data.fill_(float(np.log(max(0.25 * scale, 1e-3))))
    print(f"  teacher |action| {scale:.3f} -> exploration std "
          f"{float(model.policy.log_std.exp().mean()):.3f}")

    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out / "model"))
    venv.save(str(out / "vecnormalize.pkl"))
    return loss_val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="path to the ONNX policy to clone")
    ap.add_argument("--run-name", required=True, help="run dir to write the warm start into")
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--cmd-lo", type=float, default=0.15)
    ap.add_argument("--cmd-hi", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"collecting from {args.teacher} ...")
    obs, act = collect(args.teacher, args.episodes, args.cmd_lo, args.cmd_hi,
                       seed=args.seed)
    print(f"  {len(obs)} transitions; teacher |action| mean {np.abs(act).mean():.3f}")
    out = RUNS_DIR / args.run_name
    mse = fit(obs, act, out, epochs=args.epochs, seed=args.seed)
    print(f"done: {out}  (final mse {mse:.5f} rad^2)")


if __name__ == "__main__":
    main()
