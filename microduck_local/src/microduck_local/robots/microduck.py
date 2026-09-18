"""The Microduck's half of the lab's `Body` contract — the reference plugin.

`contract.MICRODUCK` is a `MicroduckBody`, and this file holds everything that
answer NAMES A DUCK ASSET: the viewer's mesh dump, Pollen's shipped policies,
`MicroduckWalkEnv`, the `compose()` attach sequence, the BAM servo default.

It lives here rather than on `RobotSpec` because `RobotSpec` is "a body the
walking env can walk", not "the duck". Those answers sat on it for one draft,
guarded by an id check, and that is the shape of the mistake this whole split
exists to avoid: a second walker would have inherited the duck's meshes and
the duck's env, and the failure would have arrived as a wrong picture and a
wrong policy rather than as an error. `robots/g1.G1Body` is the sibling.

The duck stays the reference body and is not an abstraction to generalise
(`docs/mars-roadmap.md` §7.3): the 61-obs deployment contract, BAM, the beak
and the colorways are its own, and hollowing them out would cost the one
sim2real-honest path in the repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .policy_contract import PolicyContract, Slot, declare
from .spec import RobotSpec

#: The duck's policy contract id (`robots/policy_contract.py`). `v1` is the
#: 61-float layout the robot itself runs and every shipped alpha policy
#: speaks; it is what a hot swap on hardware means, so the `v` bumps only if
#: upstream's deployment contract does — and then the robot's own software
#: bumps with it.
CONTRACT_ID = "microduck-61-v1"


def _obs_slots(num_joints: int) -> tuple[Slot, ...]:
    """The 61-float layout, as `walk_env.MicroduckWalkEnv._get_obs` writes it.

    Read off the slice assignments in that method, not off the docstring that
    describes them, and derived from `num_joints` rather than spelled as
    literals so the table cannot drift from the joint count that produced it:

        obs[0:3]   gyro          obs[34:48] last_action
        obs[3:6]   gravity       obs[48:51] twist_cmd
        obs[6:20]  joint_pos     obs[51:55] head_cmd
        obs[20:34] joint_vel     obs[55:61] body_cmd

    The NAMES are the deployment contract's (`contract.py`'s module
    docstring, which mirrors `microduck_rl`'s obs terms) rather than the
    method's local variables, because the table is read by someone holding
    the file and comparing it against upstream — `base_ang_vel` is what
    `infer_policy.py` calls the first three floats.

    `tests/test_policy_contract.py` pins every boundary here against a live
    env: it builds one, takes an observation, and reads each slot back
    against the state the env holds. A layout change without a contract
    change fails there, which is the whole point of the id.
    """
    n = int(num_joints)
    return (
        Slot("base_ang_vel", 0, 3),
        Slot("projected_gravity", 3, 6),
        Slot("joint_pos_rel", 6, 6 + n),
        Slot("joint_vel", 6 + n, 6 + 2 * n),
        Slot("last_action", 6 + 2 * n, 6 + 3 * n),
        Slot("twist_cmd", 6 + 3 * n, 9 + 3 * n),
        Slot("head_pose_cmd", 9 + 3 * n, 13 + 3 * n),
        Slot("body_pose_cmd", 13 + 3 * n, 19 + 3 * n),
    )


class MicroduckBody(RobotSpec):
    """Pollen's Microduck: 14 joints, 61-obs, the deployment contract.

    Not a dataclass of its own — it adds no fields, only answers. Declaring
    `@dataclass` again would re-emit `__init__` for no reason and invite a
    field to be added here instead of on the contract it belongs to.
    """

    def contract(self) -> PolicyContract:
        """The DEPLOYMENT contract: what the robot itself runs.

        Alone among the three bodies, this one is not a lab convention — the
        61 floats, the 14 actions and the 50 Hz are the shapes the Microduck's
        onboard software hot-swaps behind, so `deploy` says drop-in and names
        the one thing that still has to be true of the file (the normaliser
        baked in, which only `export-walk` does — `AGENTS.md`: never hand
        someone a raw checkpoint).

        Built from `contract.py`'s constants, imported here rather than at
        module scope because `contract.py` imports THIS module to construct
        `MICRODUCK`. Same lazy-import reason as `visual_scene` below.
        """
        from .. import contract as C

        return declare(
            self,
            id=CONTRACT_ID,
            # 1 / (PHYSICS_DT * DECIMATION) — upstream's infer_policy.py
            # clock, and the rate every shipped alpha policy was trained at.
            rate_hz=round(1.0 / C.CTRL_DT, 6),
            slots=_obs_slots(self.num_joints),
            deploy="drop-in: the robot's own 61-obs/14-action contract "
                   "(export-walk bakes the normaliser)")

    def fetch(self) -> Path:
        """A no-op that returns the directory the MJCF lives in.

        The duck's model comes from the pinned `microduck_rl` checkout that
        `scripts/setup.sh` clones, so there is nothing to download —
        `fetch-robot microduck` is legal and does nothing, and the palette's
        one download path needs no special case for a body whose assets
        arrive with the checkout.
        """
        return Path(self.scene_fn()).parent

    def setup_hint(self) -> str:
        """Nothing to run. `ready()` can still be False (a missing
        MICRODUCK_RL_DIR), which is a broken checkout rather than a download
        someone forgot, so inventing a `fetch-robot microduck` here would
        point at the wrong problem."""
        return ""

    def look(self) -> str:
        """The viewer's material set. "duck", not the id: the viewer has
        painted `duck` since before a second body existed."""
        return "duck"

    def visual_scene(self) -> dict:
        """The duck's mesh dump for the viewer.

        Re-exported, not reimplemented: `viz_server.extract_scene` builds it
        from `world.compose.scene_model()` — which carries the hinged mouth
        body the plain walk scene does not, and the world stream indexes that
        body list positionally. Two copies of that would be a silently
        mis-indexed duck.

        PHASE 1B: this import is the wrong direction (a body reaching into the
        4,500-line lab server). `extract_scene` belongs in `lab/robots.py`
        with the rest of the robot HTTP surface — `docs/mars-roadmap.md` §6.4.
        """
        from ..viz_server import extract_scene
        return extract_scene()

    def shipped_policies(self) -> tuple[dict, ...]:
        """Pollen's reference policies, as palette entries.

        PHASE 1B: `POLICIES_DIR` is read from `viz_server` for the same reason
        as `visual_scene` — one definition of where the Hub download landed,
        in the file that owns it today.
        """
        from ..viz_server import POLICIES_DIR
        if not POLICIES_DIR.exists():
            return ()
        return tuple({"id": f"pollen:{p.stem}", "label": p.stem,
                      "group": "pollen", "path": str(p),
                      "robot": self.id}
                     for p in sorted(POLICIES_DIR.glob("*.onnx")))

    def env_class(self, task: str = "walk") -> type:
        """The duck trains through recipes, so it has exactly one env.

        A `--task` for the duck is a category error worth saying out loud:
        its tricks are reward recipes in `behaviors/`, evaluated inside one
        env, not env subclasses.
        """
        if task not in (None, "", "walk"):
            raise SystemExit(
                f"--task {task!r} is a G1 env; the duck's tasks are reward "
                "recipes — use `train-behavior <name>`")
        from ..walk_env import MicroduckWalkEnv
        return MicroduckWalkEnv

    def train_env_kwargs(self, args: Any) -> dict:
        """`train-walk`'s actuator choice for the duck.

        An explicit flag is a per-run decision and beats the process env
        (MICRODUCK_ACTUATOR hard-overrides `actuator=`, not `actuator_force=`).
        """
        from ..train import DEFAULT_ACTUATOR
        actuator = getattr(args, "actuator", None)
        if actuator:
            return {"actuator_force": actuator}
        return {"actuator": DEFAULT_ACTUATOR}

    def attach(self, spec: Any, prefix: str, frame: Any,
               collision: str = "walk") -> None:
        """Attach one duck into a `/sim` world model.

        The `collision` variant is the duck's own knob: a variant is the walk
        robot PLUS collision meshes, and its mass properties are pinned back
        to the walk robot's to the bit (`compose._pin_mass_properties_to_walk`
        — the shipped walker's joints drifted 0.2-0.4 rad over 10 s without
        it). `split_jaw` is what makes the beak a hinge rather than a shell.

        PHASE 3: `world/compose.compose()` still holds this sequence inline;
        it calls this method once `WorldDuck` takes its driver from the body
        (`docs/mars-roadmap.md` §6.5). Both paths are the same three calls
        today, and `tests/test_arena.py` locks the composed duck step for step
        against the walk env.
        """
        import mujoco

        from ..world.compose import (
            ROBOT_XML,
            _pin_mass_properties_to_walk,
            split_jaw,
        )
        robot = mujoco.MjSpec.from_file(str(ROBOT_XML[collision]))
        if collision != "walk":
            _pin_mass_properties_to_walk(robot)
        split_jaw(robot)
        spec.attach(robot, prefix=prefix, frame=frame)


__all__ = ["MicroduckBody"]
