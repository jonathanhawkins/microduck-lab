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
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from .train import RUNS_DIR


def collect(teacher: str, episodes: int, cmd_lo: float, cmd_hi: float,
            seed: int = 0, gamma: float = 0.99
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll the teacher out and record (raw obs, action, discounted return).

    The returns are what lets `fit` initialise the CRITIC as well as the
    actor. They are computed per episode from the env's own reward with the
    trainer's `gamma`, bootstrapped at 0 — a truncated episode's tail is
    therefore slightly under-valued, which is the standard Monte-Carlo
    compromise and far closer to the truth than the zero-initialised critic
    it replaces.

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
    ret_buf: list[np.ndarray] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        rewards: list[float] = []
        n0 = len(obs_buf)
        for _ in range(500):
            obs = np.asarray(obs, dtype=np.float32)
            act = sess.run(None, {inp: obs[None]})[0][0].astype(np.float32)
            obs_buf.append(obs.copy())
            act_buf.append(act)
            obs, rew, term, trunc, _ = env.step(act)
            rewards.append(float(rew))
            if term or trunc:
                break
        # Discounted return from each state to the end of THIS episode.
        g = 0.0
        tail = np.empty(len(rewards), np.float32)
        for i in range(len(rewards) - 1, -1, -1):
            g = rewards[i] + gamma * g
            tail[i] = g
        assert len(tail) == len(obs_buf) - n0
        ret_buf.append(tail)
    return (np.asarray(obs_buf, np.float32), np.asarray(act_buf, np.float32),
            np.concatenate(ret_buf) if ret_buf else np.zeros(0, np.float32))


def fit(obs: np.ndarray, act: np.ndarray, out: Path, epochs: int = 40,
        batch: int = 4096, lr: float = 1e-3, seed: int = 0,
        returns: np.ndarray | None = None) -> float:
    """Fit a fresh SB3 policy to the teacher's actions; save a warm-start run.

    With `returns`, the CRITIC is fitted too, on the teacher's own discounted
    returns. That matters: this module's header used to note that "the critic
    starts untrained, so the first PPO iterations after this will spend
    themselves learning a value function — expect a dip before any gain", and
    the measurement showed the dip is what killed the idea. Cloning the actor
    alone and fine-tuning for 1M steps left every checkpoint of four arms
    falling in 100% of episodes, several at NEGATIVE ground speed: PPO's first
    updates are driven by a garbage advantage estimate, which wrecks a
    perfectly good gait before the critic catches up. An initialised critic
    is the targeted fix.

    Returns the final ACTION MSE, in radians² of joint target.
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
    v = None if returns is None else torch.as_tensor(returns.astype(np.float32))
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
    n = len(x)
    loss_val = v_loss = float("nan")
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, vtot, seen = 0.0, 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = x[idx], y[idx]
            feats = model.policy.extract_features(xb)
            if isinstance(feats, tuple):        # SB3 returns (pi, vf) when the
                pi_feats, vf_feats = feats      # extractors are not shared
            else:
                pi_feats = vf_feats = feats
            latent = model.policy.mlp_extractor.forward_actor(pi_feats)
            pred = model.policy.action_net(latent)
            loss = torch.nn.functional.mse_loss(pred, yb)
            a_loss = float(loss)
            if v is not None:
                # Same optimizer, one backward: the critic head is fitted on
                # the teacher's discounted returns over the SAME states, so
                # PPO's first advantage estimates are about right instead of
                # noise.
                v_lat = model.policy.mlp_extractor.forward_critic(vf_feats)
                v_pred = model.policy.value_net(v_lat).squeeze(-1)
                vl = torch.nn.functional.mse_loss(v_pred, v[idx])
                loss = loss + vl
                vtot += float(vl) * len(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += a_loss * len(idx)
            seen += len(idx)
        loss_val = tot / max(seen, 1)
        v_loss = vtot / max(seen, 1)
        if ep % 10 == 0 or ep == epochs - 1:
            extra = "" if v is None else f"  value mse {v_loss:.3f}"
            print(f"  epoch {ep:3d}  mse {loss_val:.5f} rad^2{extra}")

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


def default_teacher() -> Path:
    """The shipped walker: the only stable gait this workspace has."""
    from . import contract as C
    return C.MICRODUCK_RL_DIR.parent / "microduck" / "policies" / "alpha_walking.onnx"


def cache_is_valid(d: Path) -> bool:
    """Is this cache entry usable, not merely present?

    Checking `model.zip` EXISTS is not enough: a half-written one is exactly
    what a lost race leaves behind, and SB3 then fails deep inside training
    with "wasn't a zip-file" — after the vec-env workers have already forked.
    Validating on read means a corrupt entry is silently rebuilt instead.
    """
    import zipfile

    m, vn = d / "model.zip", d / "vecnormalize.pkl"
    if not (m.exists() and vn.exists() and vn.stat().st_size > 0):
        return False
    try:
        return zipfile.is_zipfile(m)
    except OSError:
        return False


# Fitting budget. 120 epochs, not 40, because cloning FIDELITY is what
# decides whether the clone stays upright — correlation(action MSE, fall
# rate) = +0.93 over five budgets, measured on the deterministic export:
#
#   episodes  epochs   mse rad^2   falls   ep_len   m/s
#        120      30     0.00030    0.75      289   0.238
#        250      40     0.00017    0.08      924   0.193   <- the old default
#        250     120     0.00011    0.00     1000   0.196
#        600     120     0.00006    0.00     1000   0.194
#
# Below ~0.00011 rad^2 the falling stops outright, at the teacher's own
# 0.196 m/s. The residual is not a diverging one — the clone tracks the
# teacher to a FLAT 0.021 rad through a whole episode, it does not drift —
# so this is approximation error costing robustness, not distribution shift,
# and more fitting is the direct fix. It costs 33 s once (17 s -> 50 s) and
# the result is cached.
DISTILL_EPISODES = 250
DISTILL_EPOCHS = 120


def ensure_distilled(teacher: Path | str | None = None, seed: int = 0,
                     episodes: int = DISTILL_EPISODES, epochs: int = DISTILL_EPOCHS,
                     cache_dir: Path | None = None, critic: bool = True) -> Path:
    """A distilled warm-start run dir, built once and reused.

    Cloning the teacher costs a few hundred episodes of rollout plus the fit,
    which is pure overhead to pay again on every launch — and the result is a
    deterministic function of (teacher bytes, seed, episodes, epochs), so it
    caches cleanly. The cache key includes the teacher's SIZE and MTIME so a
    swapped policy file cannot be silently reused.

    Returns a directory holding `model.zip` + `vecnormalize.pkl`, which is
    exactly what `--init-from` wants.
    """
    import hashlib

    t = Path(teacher) if teacher else default_teacher()
    if not t.exists():
        raise FileNotFoundError(
            f"no teacher policy at {t} — clone pollen-robotics/microduck, or pass one")
    st = t.stat()
    key = hashlib.sha256(
        f"{t.resolve()}|{st.st_size}|{int(st.st_mtime)}|{seed}|{episodes}|{epochs}|critic={critic}".encode()
    ).hexdigest()[:16]
    root = cache_dir or (RUNS_DIR / ".distill")
    out = root / key
    if cache_is_valid(out):
        return out
    print(f"[distill] cloning {t.name} -> {out} (once; cached by teacher+seed)", flush=True)
    obs, act, ret = collect(str(t), episodes, 0.15, 0.60, seed=seed)
    print(f"[distill]   {len(obs)} transitions; teacher |action| {np.abs(act).mean():.3f}; "
          f"return {ret.mean():.1f} +- {ret.std():.1f}", flush=True)
    # Build in a PRIVATE directory, then move it into place with a rename.
    # Both arms of a paired A/B share a cache key (same teacher, same seed)
    # and launch together, so without this they both miss, both `fit` into
    # the same path, and interleave their writes — which produced a
    # `model.zip` that "wasn't a zip-file" and killed two training runs.
    tmp = root / f".tmp-{os.getpid()}-{key}"
    mse = fit(obs, act, tmp, epochs=epochs, seed=seed,
              returns=ret if critic else None)
    (tmp / "distill.json").write_text(json.dumps({
        "teacher": str(t), "teacher_size": st.st_size, "seed": seed,
        "episodes": episodes, "epochs": epochs, "mse": mse,
        "critic": critic}, indent=2))
    try:
        os.rename(tmp, out)          # atomic while `out` does not exist
    except OSError:
        if cache_is_valid(out):
            # A sibling won the race. Drop our private copy — leaving it
            # behind would grow a junk directory per lost race — and use theirs.
            shutil.rmtree(tmp, ignore_errors=True)
            return out
        # `out` exists but is unusable (the interleaved-write case). Move it
        # aside rather than deleting, then land ours.
        os.rename(out, root / f".stale-{os.getpid()}-{key}")
        os.rename(tmp, out)
    print(f"[distill]   done, mse {mse:.5f} rad^2", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="path to the ONNX policy to clone")
    ap.add_argument("--run-name", required=True, help="run dir to write the warm start into")
    ap.add_argument("--episodes", type=int, default=DISTILL_EPISODES)
    ap.add_argument("--epochs", type=int, default=DISTILL_EPOCHS,
                    help="fitting epochs. Fidelity is what keeps the clone upright: "
                         "correlation(action MSE, fall rate) = +0.93, and falls stop "
                         "outright below ~0.00011 rad^2")
    ap.add_argument("--cmd-lo", type=float, default=0.15)
    ap.add_argument("--cmd-hi", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-critic", action="store_true",
                    help="clone only the actor, as before the critic fit existed "
                         "(the A/B baseline: that arm's fine-tune fell 100%% of episodes)")
    args = ap.parse_args()

    print(f"collecting from {args.teacher} ...")
    obs, act, ret = collect(args.teacher, args.episodes, args.cmd_lo, args.cmd_hi,
                            seed=args.seed)
    print(f"  {len(obs)} transitions; teacher |action| mean {np.abs(act).mean():.3f}; "
          f"return mean {ret.mean():.1f}")
    out = RUNS_DIR / args.run_name
    mse = fit(obs, act, out, epochs=args.epochs, seed=args.seed,
              returns=None if args.no_critic else ret)
    print(f"done: {out}  (final mse {mse:.5f} rad^2)")


if __name__ == "__main__":
    main()
