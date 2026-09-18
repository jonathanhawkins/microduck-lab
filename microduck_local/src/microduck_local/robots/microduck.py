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

from .spec import RobotSpec


class MicroduckBody(RobotSpec):
    """Pollen's Microduck: 14 joints, 61-obs, the deployment contract.

    Not a dataclass of its own — it adds no fields, only answers. Declaring
    `@dataclass` again would re-emit `__init__` for no reason and invite a
    field to be added here instead of on the contract it belongs to.
    """

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
