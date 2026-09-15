"""Export a trained run to runtime-compatible ONNX.

    uv run export-walk runs/<run-name> [-o policy.onnx]

Bakes the VecNormalize observation statistics into the graph — actor(normalizer(obs))
— exactly the property microduck_rl's scripts/export.py guarantees, and for the
same reason: obs normalization is ON in training, so an un-baked checkpoint sees
unnormalized observations at deployment and silently misbehaves.

Output graph: input "obs" float32 [1, 61] -> output "actions" float32 [1, 14],
the same names/shapes as the shipped alpha policies, so the file drops into
microduck_rl/scripts/infer_policy.py --new-cmd-obs unchanged.

A run trained on another body (`train-walk --robot g1`) exports at THAT
robot's dimensions — read from the run's own `run.json`, so the shape can
never be guessed wrong — and the G1's file drops into the /sim world's
`G1Walker` in place of the shipped `walker.onnx`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


def run_robot(run_dir: Path) -> str:
    """Which body this run was trained on, from its own run.json."""
    meta = run_dir / "run.json"
    if meta.is_file():
        import json
        try:
            return str(json.loads(meta.read_text()).get("robot") or "microduck")
        except (ValueError, OSError):
            pass
    return "microduck"


class OnnxWalkPolicy(torch.nn.Module):
    def __init__(self, policy, obs_mean: np.ndarray, obs_var: np.ndarray, clip_obs: float):
        super().__init__()
        self.policy = policy
        self.register_buffer("obs_mean", torch.tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(np.sqrt(obs_var + 1e-8), dtype=torch.float32))
        self.clip_obs = clip_obs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.clamp((obs - self.obs_mean) / self.obs_std, -self.clip_obs, self.clip_obs)
        features = self.policy.extract_features(x, self.policy.features_extractor)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        return self.policy.action_net(latent_pi)  # deterministic mean action


def export(run_dir: Path, out_path: Path, model_path: Path | None = None,
           vn_path: Path | None = None, robot: str | None = None) -> Path:
    """Bake the normalizer into an ONNX policy.

    Defaults to the run's final `model.zip` + `vecnormalize.pkl`. The explicit
    paths are what `select-run` uses to export a numbered CHECKPOINT for
    deterministic scoring without disturbing the shipped `policy.onnx`.
    """
    model = PPO.load(str(model_path or (run_dir / "model")), device="cpu")
    # VecNormalize.load needs a venv only for stepping; stats load without one.
    import pickle
    with open(vn_path or (run_dir / "vecnormalize.pkl"), "rb") as f:
        vn: VecNormalize = pickle.load(f)

    wrapper = OnnxWalkPolicy(
        model.policy, vn.obs_rms.mean, vn.obs_rms.var, vn.clip_obs
    ).eval()

    # Dimensions come from the ROBOT this run trained on, and are then
    # cross-checked against the checkpoint itself: a mismatch here means the
    # run.json and the weights disagree, and exporting the wrong shape would
    # hand someone a policy that loads and does nothing sane.
    from .robots import spec as spec_mod
    rspec = spec_mod.get(robot or run_robot(Path(run_dir)))
    obs_dim, act_dim = rspec.obs_dim, rspec.num_actions
    ckpt_obs = int(model.policy.observation_space.shape[0])
    ckpt_act = int(model.policy.action_space.shape[0])
    if (ckpt_obs, ckpt_act) != (obs_dim, act_dim):
        raise ValueError(
            f"{run_dir}: run.json says robot={rspec.id} ({obs_dim} obs / "
            f"{act_dim} actions) but the checkpoint is {ckpt_obs} / {ckpt_act}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    torch.onnx.export(
        wrapper, (dummy,), str(out_path),
        input_names=["obs"], output_names=["actions"],
        opset_version=17, dynamo=False,
    )

    # Verify: ONNX output must match the torch policy on random observations.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path))
    rng = np.random.default_rng(0)
    for _ in range(5):
        obs = rng.normal(0, 1, (1, obs_dim)).astype(np.float32)
        with torch.no_grad():
            want = wrapper(torch.tensor(obs)).numpy()
        got = sess.run(["actions"], {"obs": obs})[0]
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--robot", default=None, choices=("microduck", "g1"),
                    help="override the robot recorded in the run's run.json")
    args = ap.parse_args()
    out = args.out or (args.run_dir / "policy.onnx")
    robot = args.robot or run_robot(args.run_dir)
    export(args.run_dir, out, robot=robot)
    from .robots import spec as spec_mod
    rspec = spec_mod.get(robot)
    print(f"exported {out} (obs[1,{rspec.obs_dim}] -> "
          f"actions[1,{rspec.num_actions}], normalizer baked, robot={rspec.id})")
    if rspec.id == "microduck":
        print("try it: cd ../microduck_rl && uv run scripts/infer_policy.py "
              f"--walking {out.resolve()} --new-cmd-obs")
    else:
        print("try it: uv run duck-lab --world follow-me  # the /sim person, "
              f"or MICRODUCK_G1_WALKER={out.resolve()}")


if __name__ == "__main__":
    main()
