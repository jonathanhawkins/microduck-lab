"""Vector envs that share ONE compiled mjModel instead of one per worker.

The problem this exists to fix, measured on this machine (scene_walk.xml):

    mjData  (the actual simulation state)         ~0.9 MB per env
    mjModel (compiled, read-only while stepping)  ~138 MB as a second copy in a
                                                  warm process; ~470 MB as the
                                                  FIRST compile in a fresh one

MuJoCo is explicitly built so a single read-only `mjModel` backs many `mjData`.
We were throwing that away: `SubprocVecEnv` defaults to the `forkserver` start
method, so every worker imported MuJoCo and compiled the MJCF *after* forking.
Measured private memory per worker: 632 MB, of which 0.9 MB is the simulation.
10 envs cost ~7 GB to simulate ~9 MB of state.

Backends (pick with ``MICRODUCK_VEC_ENV``, or the ``backend=`` argument):

``fork``    DEFAULT. A torch-free ``ForkVecEnv`` (SB3's package ``__init__``
            imports torch, which starts OpenMP pools that deadlock on a
            macOS fork). The model is compiled ONCE in the parent, then
            children inherit it copy-on-write. Same process-level
            parallelism as before, but the fleet costs one model instead of
            N: 632 MB of private memory per worker becomes 16 MB, and
            building a 64-env vec env drops from 7.8 s to 0.2 s because no
            child ever opens the MJCF. Domain randomization stays
            per-worker: a write to `body_mass`/`geom_friction` in a child
            copies those one or two pages for that child alone (verified in
            tests/test_vec_env.py). Trainers wrap the result with
            ``as_sb3_vec_env`` after importing torch.
``subproc`` The previous behavior: `forkserver`, a private compile per worker.
            Kept as the escape hatch, and as the honest baseline to measure
            against.
``dummy``   Every env in this process, stepped serially, sharing one model.
            The memory floor (one model, total), but no parallelism.
``thread``  Every env in this process, stepped by a thread pool, sharing one
            model. MuJoCo *does* release the GIL inside `mj_step` — measured
            3.5-4.2x on raw physics threads — but a full `env.step()` is only
            ~40% physics and the numpy half holds the GIL, so this measured
            ~1.0x end to end. It is here because the question deserves a
            measurable answer, not because it is fast.

Every backend keeps the caller's `env_fns` untouched: the model is injected by
building the envs inside `walk_env.shared_model_scope()`, so `MicroduckWalkEnv`
and `BehaviorEnv` need no signature change at the call site.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any

import cloudpickle
import gymnasium as gym
import numpy as np

from . import contract as C
from .walk_env import shared_model_scope

# Which backend `make_vec_env` picks when nobody says. "fork" shares the model
# and is otherwise the same N-process design as before; see the module
# docstring for the measurements behind the choice.
DEFAULT_BACKEND = "fork"
BACKENDS = ("fork", "subproc", "dummy", "thread")
ENV_VAR = "MICRODUCK_VEC_ENV"


def resolve_backend(backend: str | None = None) -> str:
    """Explicit argument wins, then $MICRODUCK_VEC_ENV, then DEFAULT_BACKEND."""
    name = (backend or os.environ.get(ENV_VAR) or DEFAULT_BACKEND).strip().lower()
    if name not in BACKENDS:
        raise ValueError(
            f"unknown vec-env backend {name!r}; expected one of {BACKENDS}"
        )
    return name


class _SharedModelEnvFn:
    """Run a caller's env factory inside a `shared_model_scope`.

    A class rather than a closure so the wrapper survives being pickled by a
    start method that pickles (it does not need to under `fork`, which is the
    whole point, but a wrapper that only works under one start method is a
    trap for the next person).
    """

    def __init__(self, fn: Callable[[], gym.Env], exclusive: bool):
        self.fn = fn
        self.exclusive = exclusive

    def __call__(self) -> gym.Env:
        with shared_model_scope(exclusive=self.exclusive):
            return self.fn()


def _prime_model_cache(env_fn: Callable[[], gym.Env]) -> None:
    """Compile the model in THIS process by building one throwaway env.

    The scene an env will open is a function of its kwargs (walk scene vs the
    full-collision scene, per behavior), and the factories are opaque closures
    — so the robust way to learn it is to build one env and let it say. The
    compile lands in `walk_env`'s per-process cache, which is what the forked
    children then inherit instead of compiling their own.
    """
    with shared_model_scope(exclusive=True):
        probe = env_fn()
    probe.close()


# --------------------------------------------------------------------------
# fork backend — torch-free. SB3's package __init__ imports PPO, which
# imports torch, which starts OpenMP pools; forking those deadlocks on macOS.
# --------------------------------------------------------------------------


class _CloudpickleFn:
    """cloudpickle an env factory so fork children can rebuild it. The fn
    itself is typically a tiny closure; the compiled mjModel is inherited
    via the address space, not this pickle."""

    def __init__(self, fn: Callable[[], gym.Env]):
        self.fn = fn

    def __getstate__(self) -> bytes:
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, state: bytes) -> None:
        self.fn = cloudpickle.loads(state)

    def __call__(self) -> gym.Env:
        return self.fn()


def _shm_view(buf, dtype, shape):
    """Numpy view of a multiprocessing.RawArray. Both parent and fork children
    see the same pages — that is the whole point (COW ndarrays would not)."""
    return np.frombuffer(buf, dtype=dtype).reshape(shape)


def _fork_worker(remote, parent_remote, env_fn_wrappers: list[_CloudpickleFn],
                 first_idx: int, widx: int, act_buf, obs_buf, rew_buf,
                 done_buf, ctrl_buf, go_sem, done_sem, pending,
                 pending_lock) -> None:
    parent_remote.close()
    envs = [fn() for fn in env_fn_wrappers]
    act = _shm_view(act_buf, np.float32, (-1, C.NUM_JOINTS))
    obs_s = _shm_view(obs_buf, np.float32, (-1, C.OBS_DIM))
    rew = _shm_view(rew_buf, np.float64, (-1,))
    done_s = _shm_view(done_buf, np.int8, (-1,))
    ctrl = _shm_view(ctrl_buf, np.int8, (-1,))
    while True:
        # One semaphore wake per step. The old protocol pickled a ("step",
        # None) tuple down the pipe and a None back up for EVERY worker on
        # EVERY step — 4 read/write syscalls each; at 32 envs the parent
        # spent more wall time in posix.read than the workers spent stepping
        # physics (vec-step floor 1.61 ms vs 0.34 ms of worker compute).
        # Steps now travel entirely through the shared buffers; the pipe
        # carries only control commands (ctrl flag raised) and the rare
        # episode-end info dict.
        #
        # A worker may own SEVERAL envs (envs_per_worker > 1), stepped
        # serially here: at high env counts that trades per-env parallel
        # latency for W-fewer semaphore ops and pipes per vec-step.
        go_sem.acquire()
        if not ctrl[widx]:
            for local, env in enumerate(envs):
                idx = first_idx + local
                o, reward, terminated, truncated, info = env.step(act[idx])
                is_done = terminated or truncated
                rew[idx] = reward
                done_s[idx] = int(is_done)
                if is_done:
                    info["TimeLimit.truncated"] = truncated and not terminated
                    info["terminal_observation"] = o
                    o, _ = env.reset()
                    obs_s[idx] = o
                    # Sent BEFORE the release so the bytes exist by the time
                    # the parent (which recvs only after collecting all W
                    # releases) asks this pipe for them. Multiple done envs
                    # in one worker send in local order; the parent reads
                    # this pipe back in the same order.
                    remote.send(info)
                else:
                    obs_s[idx] = o
            # Completion barrier: only the last worker out posts, so the
            # parent's step_wait is one acquire, not one per worker. The
            # lock is the memory fence that publishes this worker's shm
            # writes before the parent can wake.
            with pending_lock:
                pending.value -= 1
                last = pending.value == 0
            if last:
                done_sem.release()
            continue
        try:
            cmd, data = remote.recv()
        except EOFError:
            break
        if cmd == "reset":
            for local, env in enumerate(envs):
                o, _ = env.reset()
                obs_s[first_idx + local] = o
            remote.send(None)
        elif cmd == "close":
            for env in envs:
                env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send((envs[0].observation_space, envs[0].action_space))
        elif cmd == "get_attr":
            slots, name = data
            remote.send([getattr(envs[s].unwrapped, name) for s in slots])
        elif cmd == "set_attr":
            slots, name, value = data
            for s in slots:
                setattr(envs[s].unwrapped, name, value)
            remote.send(None)
        elif cmd == "env_method":
            slots, name, m_args, m_kwargs = data
            remote.send([getattr(envs[s].unwrapped, name)(*m_args, **m_kwargs)
                         for s in slots])
        else:
            raise NotImplementedError(cmd)


class ForkVecEnv:
    """SB3-shaped vec env (reset/step/get_attr) that forks WITHOUT importing
    torch. Wrap with ``as_sb3_vec_env`` after the workers exist so PPO's
    VecMonitor can see a real ``VecEnv`` subclass.

    Step traffic is shared-memory arrays (obs/rew/done/action) with semaphore
    wake/completion signalling — no per-step pipe traffic at all. Measured at
    32 envs (same live-lab load for both numbers): the old pickled pipe
    handshake put the no-policy vec-step floor at 1.61 ms against 0.34 ms of
    actual worker compute; semaphores took the floor to 1.08 ms. Pipes remain
    for control commands (guarded by the ctrl flag) and the rare episode-end
    info dict.
    """

    def __init__(self, env_fns: list[Callable[[], gym.Env]],
                 envs_per_worker: int = 1):
        n = len(env_fns)
        k = max(1, int(envs_per_worker))
        ctx = mp.get_context("fork")
        # env index -> (worker, local slot). Worker w owns the contiguous
        # block [w*k, min(n, (w+1)*k)).
        self._firsts = list(range(0, n, k))
        w_count = len(self._firsts)
        self._worker_of = [i // k for i in range(n)]
        self._slot_of = [i % k for i in range(n)]
        # Allocate BEFORE fork so children inherit the mappings.
        self._act_buf = mp.RawArray("f", n * C.NUM_JOINTS)
        self._obs_buf = mp.RawArray("f", n * C.OBS_DIM)
        self._rew_buf = mp.RawArray("d", n)
        self._done_buf = mp.RawArray("b", n)
        # ctrl[w] = 1 tells a woken worker "read a command from your pipe"
        # instead of the shared-memory fast step. Owned by the parent: raised
        # before the wake, cleared after the reply.
        self._ctrl_buf = mp.RawArray("b", w_count)
        self._act = _shm_view(self._act_buf, np.float32, (n, C.NUM_JOINTS))
        self._obs = _shm_view(self._obs_buf, np.float32, (n, C.OBS_DIM))
        self._rew = _shm_view(self._rew_buf, np.float64, (n,))
        self._done = _shm_view(self._done_buf, np.int8, (n,))
        self._ctrl = _shm_view(self._ctrl_buf, np.int8, (w_count,))
        # Per-worker wake ("go") plus a completion BARRIER: workers decrement
        # a shared counter and only the last one out posts the semaphore, so
        # the parent pays ONE acquire per vec-step instead of one per worker
        # (profiled at 64 envs: 32 sequential acquires were 36 us each —
        # syscall + blocking — the single largest parent-side line).
        self._go = [ctx.Semaphore(0) for _ in range(w_count)]
        self._done_sem = ctx.Semaphore(0)
        self._pending = mp.RawValue("i", 0)
        self._pending_lock = ctx.Lock()
        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(w_count)])
        self.processes = []
        for w, (work_remote, remote, first) in enumerate(
                zip(work_remotes, self.remotes, self._firsts)):
            fns = [_CloudpickleFn(fn) for fn in env_fns[first:first + k]]
            proc = ctx.Process(
                target=_fork_worker,
                args=(work_remote, remote, fns, first, w,
                      self._act_buf, self._obs_buf, self._rew_buf,
                      self._done_buf, self._ctrl_buf, self._go[w],
                      self._done_sem, self._pending, self._pending_lock),
                daemon=True,
            )
            proc.start()
            self.processes.append(proc)
            work_remote.close()
        self.observation_space, self.action_space = self._raw_command(
            0, "get_spaces", None)[0]
        self.num_envs = n
        self.num_workers = w_count
        self.closed = False
        self.waiting = False

    def _raw_command(self, workers, cmd: str, data) -> list[Any]:
        """Run a control command on the given worker(s); returns their replies.

        Fan-out then fan-in, like the old all-pipe protocol: every target gets
        its command before any reply is awaited, so workers execute
        concurrently rather than in serial round-trips.
        """
        if isinstance(workers, int):
            workers = [workers]
        for w in workers:
            self._ctrl[w] = 1
            self.remotes[w].send((cmd, data))
            self._go[w].release()
        replies = []
        for w in workers:
            replies.append(self.remotes[w].recv())
            self._ctrl[w] = 0
        return replies

    def _env_command(self, indices, cmd: str, payload: tuple) -> list[Any]:
        """Per-ENV control command: group env indices by worker, send each
        worker its local slot list, splice per-env replies back into the
        caller's index order."""
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        indices = list(indices)
        by_worker: dict[int, list[int]] = {}
        for i in indices:
            by_worker.setdefault(self._worker_of[i], []).append(self._slot_of[i])
        workers = sorted(by_worker)
        replies = self._raw_command_multi(workers, cmd, by_worker, payload)
        out: dict[int, Any] = {}
        for w, reply in zip(workers, replies):
            if reply is None:
                continue
            first = self._firsts[w]
            for slot, value in zip(by_worker[w], reply):
                out[first + slot] = value
        return [out.get(i) for i in indices]

    def _raw_command_multi(self, workers, cmd, by_worker, payload):
        """Like _raw_command but each worker gets its own slot list."""
        for w in workers:
            self._ctrl[w] = 1
            self.remotes[w].send((cmd, (by_worker[w], *payload)))
            self._go[w].release()
        replies = []
        for w in workers:
            replies.append(self.remotes[w].recv())
            self._ctrl[w] = 0
        return replies

    def reset(self) -> np.ndarray:
        self._raw_command(list(range(self.num_workers)), "reset", None)
        return self._obs.copy()

    def step_async(self, actions: np.ndarray) -> None:
        np.copyto(self._act, np.asarray(actions, dtype=np.float32))
        # Armed BEFORE any wake: a worker cannot decrement until released.
        self._pending.value = self.num_workers
        for sem in self._go:
            sem.release()
        self.waiting = True

    def step_wait(self) -> tuple:
        self._done_sem.acquire()
        # All completions collected, so every shared write (and any
        # episode-end info pickle) is in place before we read. A worker with
        # several done envs sent their infos in local order; reading its pipe
        # by ascending env index consumes them in that same order.
        infos: list[dict] = [{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(self._done):
            infos[i] = self.remotes[self._worker_of[i]].recv()
        self.waiting = False
        return (self._obs.copy(), self._rew.copy(),
                self._done.astype(bool), infos)

    def step(self, actions: np.ndarray) -> tuple:
        self.step_async(actions)
        return self.step_wait()

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            self.step_wait()
        for w in range(self.num_workers):
            self._ctrl[w] = 1
            self.remotes[w].send(("close", None))
            self._go[w].release()      # close sends no reply; just wake + join
        for proc in self.processes:
            proc.join()
        self.closed = True

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        return self._env_command(indices, "get_attr", (attr_name,))

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        self._env_command(indices, "set_attr", (attr_name, value))

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list[Any]:
        return self._env_command(indices, "env_method",
                                 (method_name, args, kwargs))

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        if indices is None:
            return [False] * self.num_envs
        if isinstance(indices, int):
            indices = [indices]
        return [False] * len(list(indices))


def as_sb3_vec_env(venv):
    """Promote a ``ForkVecEnv`` to an SB3 ``VecEnv`` after torch is imported.

    No-op if ``venv`` is already a VecEnv (dummy / thread / subproc backends).
    """
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv
    if isinstance(venv, VecEnv):
        return venv

    inner = venv

    class _Adapter(VecEnv):
        def __init__(self):
            super().__init__(inner.num_envs, inner.observation_space,
                             inner.action_space)
            self._inner = inner

        def reset(self):
            return self._inner.reset()

        def step_async(self, actions):
            return self._inner.step_async(actions)

        def step_wait(self):
            return self._inner.step_wait()

        def close(self):
            return self._inner.close()

        def get_attr(self, attr_name, indices=None):
            return self._inner.get_attr(attr_name, indices)

        def set_attr(self, attr_name, value, indices=None):
            return self._inner.set_attr(attr_name, value, indices)

        def env_method(self, method_name, *args, indices=None, **kwargs):
            return self._inner.env_method(method_name, *args, indices=indices,
                                          **kwargs)

        def env_is_wrapped(self, wrapper_class, indices=None):
            return self._inner.env_is_wrapped(wrapper_class, indices)

    return _Adapter()


def _threaded_cls():
    """DummyVecEnv subclass — imported lazily so this module stays torch-free."""
    from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

    class ThreadedVecEnv(DummyVecEnv):
        """`DummyVecEnv` whose envs are stepped by a thread pool.

        Worth trying because MuJoCo releases the GIL inside `mj_step`, so the
        physics of N envs really does run on N cores. Worth *measuring* before
        believing, because `env.step()` is only ~40% physics on this robot and
        the rest is GIL-held numpy — which is what the benchmark found (~1.0x).

        Domain randomization is refused here: the per-step re-assert that makes
        a shared model safe for serial envs (`MicroduckWalkEnv._sync_model`)
        races when siblings step concurrently, and silently shuffling which
        env gets which body mass is worse than not offering the backend.
        """

        def __init__(self, env_fns: list[Callable[[], gym.Env]],
                     max_workers: int | None = None):
            super().__init__(env_fns)
            shared = [e for e in self.envs
                      if getattr(e.unwrapped, "_model_shared", False)]
            randomized = [e for e in shared
                          if getattr(e.unwrapped, "domain_rand", False)]
            if randomized:
                raise ValueError(
                    "ThreadedVecEnv cannot combine domain_rand=True with a "
                    "shared mjModel: the per-step domain-randomization "
                    "re-assert races across threads. Pass domain_rand=False "
                    "(what train_behavior already does) or use the 'fork' "
                    "backend."
                )
            self._pool = ThreadPoolExecutor(
                max_workers=max_workers or len(self.envs),
                thread_name_prefix="duck-env",
            )

        def _step_one(self, env_idx: int) -> None:
            env = self.envs[env_idx]
            (obs, self.buf_rews[env_idx], terminated, truncated,
             self.buf_infos[env_idx]) = env.step(self.actions[env_idx])
            self.buf_dones[env_idx] = terminated or truncated
            self.buf_infos[env_idx]["TimeLimit.truncated"] = (
                truncated and not terminated)
            if self.buf_dones[env_idx]:
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = env.reset()
            self._save_obs(env_idx, obs)

        def step_wait(self):
            list(self._pool.map(self._step_one, range(self.num_envs)))
            return (self._obs_from_buf(), self.buf_rews.copy(),
                    self.buf_dones.copy(), deepcopy(self.buf_infos))

        def _reset_one(self, env_idx: int) -> None:
            options = self._options[env_idx]
            maybe_options = {"options": options} if options else {}
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(
                seed=self._seeds[env_idx], **maybe_options
            )
            self._save_obs(env_idx, obs)

        def reset(self):
            list(self._pool.map(self._reset_one, range(self.num_envs)))
            self._reset_seeds()
            self._reset_options()
            return self._obs_from_buf()

        def close(self) -> None:
            self._pool.shutdown(wait=True)
            super().close()

    return ThreadedVecEnv


# Public name for tests; the class itself is built on first thread-backend use
# so importing this module does not pull torch.
ThreadedVecEnv = None  # type: ignore[assignment]


def make_vec_env(env_fns: list[Callable[[], gym.Env]],
                 backend: str | None = None):
    """Build a vector env whose workers share one compiled mjModel.

    Drop-in for `SubprocVecEnv(env_fns)`: same factories, same step/reset API.
    The fork backend does NOT import torch (see ForkVecEnv). dummy / thread /
    subproc do, because they don't fork.
    """
    if not env_fns:
        raise ValueError("make_vec_env needs at least one env factory")
    name = resolve_backend(backend)

    if name == "fork":
        # Compile here, in the parent, THEN fork: the children inherit the
        # compiled model copy-on-write and never open the MJCF at all.
        _prime_model_cache(env_fns[0])
        wrapped = [_SharedModelEnvFn(fn, exclusive=True) for fn in env_fns]
        # >1 packs several envs per worker process (stepped serially there):
        # fewer semaphore ops and pipes per vec-step, at the cost of longer
        # per-worker latency. Worth it only at high env counts — measure with
        # `bench-envs` before changing.
        per_worker = int(os.environ.get("MICRODUCK_ENVS_PER_WORKER", "1") or 1)
        return ForkVecEnv(wrapped, envs_per_worker=per_worker)

    if name == "subproc":            # forkserver: children re-import, torch-safe
        from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
        return SubprocVecEnv(env_fns)

    # In-process backends share one model object across envs. exclusive=False
    # turns on the per-step domain-randomization re-assert that keeps siblings
    # from inheriting each other's body mass — and BAM cannot share (it
    # rewrites dof_frictionloss every substep). A single in-process env has
    # nobody to share with, so exclusive=True lets BAM boot under dummy
    # (how tests pin MICRODUCK_VEC_ENV).
    wrapped = [_SharedModelEnvFn(fn, exclusive=(len(env_fns) == 1))
               for fn in env_fns]
    if name == "dummy":
        from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
        return DummyVecEnv(wrapped)
    global ThreadedVecEnv
    if ThreadedVecEnv is None:
        ThreadedVecEnv = _threaded_cls()
    return ThreadedVecEnv(wrapped)
