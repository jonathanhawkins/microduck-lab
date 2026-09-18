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

**The file leaves here self-describing.** Every export stamps the body's
`PolicyContract` into the ONNX's own `metadata_props`
(`robots/policy_contract.py`): the id, the dims, the control rate, the slot
table naming all 61 (or 99, or 32) floats, and one sentence saying what
deploying it actually means. This is the only moment the harness can attach
that — the person running the exporter is the person about to hand the file
to someone — and it is what makes `eval-walk some.onnx`, a Hub download and a
palette chip agree about which body a policy drives without a run directory
anywhere near it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from .robots import registry
from .robots.policy_contract import PolicyContract, from_run_json, resolve


def run_robot(run_dir: Path) -> str:
    """Which body this run was trained on.

    The name this module has always exported, now answered by
    `policy_contract.resolve()` so that there is ONE precedence in the tree
    instead of one per reader: the policy's own ONNX metadata, then
    `run.json`'s recorded contract, then `run.json`'s `"robot"` (which is all
    this function used to read), then the duck. Every old answer is preserved
    — a run dir with no `run.json`, or an unreadable one, is still a duck —
    and a policy that knows its own body is now believed.
    """
    return resolve(Path(run_dir)).robot


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


def export_contract(run_dir: Path, robot: str | None = None) -> PolicyContract:
    """The contract this run's export must carry, or a refusal.

    Three sources have to agree before a file is written, and the checks are
    here rather than inline so `select-run`, `train_behavior` and the lab's
    export button all get them:

      * the BODY the export is shaped for — `--robot` if given, else the one
        `resolve()` reads off the run;
      * what the run RECORDED about itself — `run.json`'s `"contract"`, and
        only that (see `policy_contract.from_run_json`: a `policy.onnx`
        already in the directory is an earlier export's artefact, not the
        run's account of itself). A disagreement means the run trained one
        body and is being exported as another, which is the same failure the
        width check below catches — caught earlier, and named by CONTRACT
        rather than by width, because two bodies at one width are exactly
        what ids exist for;
      * the CHECKPOINT's own observation and action spaces (in `export`).

    Only a RECORDED contract can refuse, and there are exactly two ways it
    can: `--robot` overriding a run that recorded something else, and a body
    whose contract id has been deliberately bumped since the run was trained
    (then every old run's recorded `v1` stops matching, and a re-export waits
    for someone to decide what that means — which is what a version is for).
    With no override and no bump there is nothing to contradict: the recorded
    contract is itself what `resolve()` answers with.

    A run whose `run.json` names a body but declares no contract (every run
    trained before this existed) is exported at that body's current contract
    without complaint — the name is what it had to say, and `--robot` remains
    the override it has always been.
    """
    body = registry.get(robot or run_robot(Path(run_dir)))
    contract = body.contract()
    declared = from_run_json(Path(run_dir))
    if declared is not None and not contract.matches(declared):
        raise ValueError(
            f"{run_dir}: this run recorded contract {declared.id} "
            f"({declared.obs_dim} obs / {declared.act_dim} actions) but is "
            f"being exported as {contract.id} ({contract.obs_dim} obs / "
            f"{contract.act_dim} actions) — exporting it would hand someone "
            f"a file that names the wrong body")
    return contract


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
    contract = export_contract(run_dir, robot)
    obs_dim, act_dim = contract.obs_dim, contract.act_dim
    ckpt_obs = int(model.policy.observation_space.shape[0])
    ckpt_act = int(model.policy.action_space.shape[0])
    if (ckpt_obs, ckpt_act) != (obs_dim, act_dim):
        raise ValueError(
            f"{run_dir}: run.json says robot={contract.robot} ({obs_dim} obs / "
            f"{act_dim} actions) but the checkpoint is {ckpt_obs} / {ckpt_act}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    torch.onnx.export(
        wrapper, (dummy,), str(out_path),
        input_names=["obs"], output_names=["actions"],
        opset_version=17, dynamo=False,
    )
    # Stamp the contract BEFORE the cross-check below, so the check runs on
    # the file that actually ships. Metadata cannot change a graph, and this
    # is how every export proves that for itself rather than trusting it.
    contract.write_onnx_metadata(out_path)

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
    ap.add_argument("--robot", default=None, choices=registry.ids(),
                    help="override the robot recorded in the run's run.json")
    args = ap.parse_args()
    out = args.out or (args.run_dir / "policy.onnx")
    robot = args.robot or run_robot(args.run_dir)
    export(args.run_dir, out, robot=robot)
    contract = registry.get(robot).contract()
    # The contract line, not just the shapes: the deploy sentence is the part
    # that gets lost when the file is passed along, and this is the last
    # moment the harness is in the conversation.
    print(f"exported {out} (normalizer baked)")
    print(f"  contract {contract.describe()}")
    if contract.robot == "microduck":
        print("try it: cd ../microduck_rl && uv run scripts/infer_policy.py "
              f"--walking {out.resolve()} --new-cmd-obs")
    else:
        print("try it: uv run duck-lab --world follow-me  # the /sim person, "
              f"or MICRODUCK_G1_WALKER={out.resolve()}")


if __name__ == "__main__":
    main()
