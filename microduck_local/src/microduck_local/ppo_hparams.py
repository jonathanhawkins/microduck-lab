"""PPO numbers shared by the trainers and the symmetry loss.

Torch-free on purpose: `train.py` / `train_behavior.py` fork SubprocVecEnv
workers BEFORE importing torch, and a module that pulled these constants
from `symmetry.py` (which imports torch) would put OpenMP thread pools in
the parent and deadlock the fork on macOS.
"""

import os

N_STEPS = 256
TARGET_BATCH = 1024
# rsl_rl's num_mini_batches. Below TARGET_BATCH * N_MINI_BATCHES we keep the
# historical "largest divisor <= 1024" rule (16 envs → 4 minibatches of 1024,
# matching upstream). Past that, grow the minibatch so the optimizer-step
# count stays at 4 * n_epochs instead of climbing with every helper duck —
# 28 envs used to mean 7 minibatches of 1024 (the serial update grew 75%
# for ~8% more rollout) and the cores sat idle at 40%.
N_MINI_BATCHES = 4
# Intra-op threads for the PPO update on this M5 Max. Measured: a 1024×512
# Linear peaks at 8 (0.38 ms) vs the PyTorch default of 6 P-cores (0.42 ms)
# vs 18 (0.42 ms, E-cores hurt). Set after fork, in the trainer only.
TORCH_INTRA_THREADS = 8

# rsl_rl's value_loss_coef. SB3 defaults to 0.5; leaving that in place made
# the critic half as loud as the GPU stack.
VF_COEF = 1.0

# microduck_rl's SYMMETRY_CFG ships mirror_loss_coeff: 0.5.
DEFAULT_SYMMETRY_COEF = 0.5

# Every official task cfg sets schedule="adaptive" with desired_kl=0.01.
# DEFAULT here is off: at our batch size that target pins the rate to its
# floor. See symmetry.py for the three-arm A/B.
UPSTREAM_DESIRED_KL = 0.01
DEFAULT_DESIRED_KL = None


def ppo_batch_size(n_steps: int, n_envs: int, target: int = TARGET_BATCH,
                   n_mini_batches: int = N_MINI_BATCHES) -> int:
    """A minibatch size that divides the rollout buffer.

    SB3 warns and truncates when `batch_size` does not divide `n_steps * n_envs`.
    Lab helpers change the env count at runtime (16, 18, 20, …), so this has
    to be computed, not hardcoded.

    Two regimes, so adding helpers buys rollout rather than a longer update:

    * Buffer smaller than `target * n_mini_batches` (the 16-env default and
      below): largest divisor of the buffer that is <= `target`. Same numbers
      as before — 16 envs is still 4 × 1024, matching rsl_rl.
    * Buffer at or above that: exactly `n_mini_batches` minibatches, so 24
      envs is 4 × 1536 rather than 6 × 1024, and 28 is 4 × 1792 rather than
      7 × 1024. Larger matmuls also keep the 8 intra-op threads busier.
    """
    n = int(n_steps) * int(n_envs)
    if n <= 0:
        raise ValueError(f"empty rollout buffer: n_steps={n_steps} n_envs={n_envs}")
    mini = int(n_mini_batches)
    cap = int(target)
    if mini > 0 and n >= cap * mini and n % mini == 0:
        return n // mini
    b = min(cap, n)
    while b > 1 and n % b:
        b -= 1
    return b


def configure_torch_cpu(torch_mod) -> None:
    """Trainer-side thread pool, called AFTER the vec-env workers fork.

    Intra-op: 8 on this machine (see TORCH_INTRA_THREADS). Inter-op: 1,
    because the PPO train loop is a single stream of ops — PyTorch's default
    of cpu_count() idle inter-op threads just contend for the same cores.
    Inter-op can only be set once per process; a RuntimeError means some
    earlier CPU work already froze it, which is fine.
    """
    try:
        torch_mod.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cpus = os.cpu_count() or TORCH_INTRA_THREADS
    torch_mod.set_num_threads(min(TORCH_INTRA_THREADS, cpus))
