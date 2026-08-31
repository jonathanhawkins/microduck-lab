"""Export a trained run to runtime-compatible ONNX.

    uv run export-walk runs/<run-name> [-o policy.onnx]

Bakes the VecNormalize observation statistics into the graph — actor(normalizer(obs))
— exactly the property microduck_rl's scripts/export.py guarantees, and for the
same reason: obs normalization is ON in training, so an un-baked checkpoint sees
unnormalized observations at deployment and silently misbehaves.

Output graph: input "obs" float32 [1, 61] -> output "actions" float32 [1, 14],
the same names/shapes as the shipped alpha policies, so the file drops into
microduck_rl/scripts/infer_policy.py --new-cmd-obs unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from . import contract as C


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


def export(run_dir: Path, out_path: Path) -> Path:
    model = PPO.load(str(run_dir / "model"), device="cpu")
    # VecNormalize.load needs a venv only for stepping; stats load without one.
    import pickle
    with open(run_dir / "vecnormalize.pkl", "rb") as f:
        vn: VecNormalize = pickle.load(f)

    wrapper = OnnxWalkPolicy(
        model.policy, vn.obs_rms.mean, vn.obs_rms.var, vn.clip_obs
    ).eval()

    dummy = torch.zeros(1, C.OBS_DIM, dtype=torch.float32)
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
        obs = rng.normal(0, 1, (1, C.OBS_DIM)).astype(np.float32)
        with torch.no_grad():
            want = wrapper(torch.tensor(obs)).numpy()
        got = sess.run(["actions"], {"obs": obs})[0]
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.run_dir / "policy.onnx")
    export(args.run_dir, out)
    print(f"exported {out} (obs[1,{C.OBS_DIM}] -> actions[1,{C.NUM_JOINTS}], normalizer baked)")
    print("try it: cd ../microduck_rl && uv run scripts/infer_policy.py "
          f"--walking {out.resolve()} --new-cmd-obs")


if __name__ == "__main__":
    main()
