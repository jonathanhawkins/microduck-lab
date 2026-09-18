"""Teachable tasks for Innate's MARS — the panel's entry to the third body.

Same shape as `g1_tasks.py`, for the same reason: everything else in this
package is a MICRODUCK reward recipe (duck joints, duck feet, the 61-obs
command slots) evaluated inside `BehaviorEnv`, and another body's task is an
env subclass run by a different trainer. Registering it HERE anyway is what
gives MARS the 🎓 teach panel, the job card, the live-snapshot preview and the
trainee slot on stage without any of those learning what a MARS is.

The `terms` are what the panel SHOWS. They are display rows, not callables the
env evaluates — `robots/mars_env.MarsArmEnv` owns the reward, and `_env_owned`
exists so that is visible at a glance rather than implied. The weights here
mirror `mars_env`'s module constants; `tests/test_mars_env.py` pins the two
against each other, because two tables that must agree are exactly the
duplication that drifts.

Vocabulary, per `docs/mars-roadmap.md` §1 and §4: MARS is taught a **task**,
not a trick, and what comes out is a **code skill** — an Innate Python plugin
running onnxruntime — never one of their *learned skills*, which are ACT
checkpoints trained from teleop demonstrations. Untested on hardware: nothing
in this repo has ever driven a MARS.
"""

from .core import *  # noqa: F401,F403 — the package's namespace cascade
from .core import Behavior, RewardTerm


def _env_owned(env):  # noqa: ARG001 — see the module docstring
    """Placeholder: MARS rewards live in the env, not in this library."""
    return 0.0


MARS_REACH = Behavior(
    id="mars_reach",
    emoji="🎯",
    title="Reach a point (MARS)",
    description="Put the gripper on a point sampled in front of the robot and "
                "hold it there, without the arm hitting its own chassis.",
    how_it_learns=(
        "The big points are for CLOSING the gap: every control step it earns "
        "for however much nearer the gripper got to the target and loses the "
        "same for drifting away, so an arm that stalls halfway earns nothing "
        "at all and one that waves about earns nothing either — the total over "
        "an episode is simply how much ground it made up. On top of that there "
        "is a small bonus for every step the gripper spends inside two "
        "centimetres of the target, which is what makes it hold the pose "
        "instead of flying past. The arm starts folded across the chassis with "
        "the target out in front, so the first thing it has to solve is the "
        "route: driving a link into its own body ends the episode and forfeits "
        "the rest of the hold. Its wheels are switched off for this one — "
        "rolling the whole robot at the target would be the cheapest way to "
        "satisfy the score and not the thing being taught."
    ),
    keywords=("reach", "reach a point", "touch that", "point at it",
              "put your gripper there", "arm to the target", "reach out",
              "extend the arm", "touch the spot"),
    robot="mars",
    # `train-walk --robot mars --task reach`; TrainingJob appends --run-name,
    # --envs, --steps, --snap-steps and --init-from.
    trainer=("-m", "microduck_local.train", "--robot", "mars",
             "--task", "reach"),
    terms=(
        RewardTerm("reach_progress",
                   "big points for closing the gap to the target — and the "
                   "same taken back for drifting away from it",
                   40.0, _env_owned),
        RewardTerm("at_target",
                   "a bonus for every step the gripper spends within two "
                   "centimetres of the target, so holding it pays",
                   2.0, _env_owned),
        RewardTerm("action_rate_penalty",
                   "charged for twitchy joint commands", 0.002, _env_owned,
                   is_penalty=True),
        RewardTerm("action_mag_penalty",
                   "charged for asking for more than it is allowed — a "
                   "policy pinned against its own limits can only slam the "
                   "arm about, never hold it still", 0.002, _env_owned,
                   is_penalty=True),
        RewardTerm("joint_vel_penalty",
                   "charged a little for whipping the arm around", 0.0005,
                   _env_owned, is_penalty=True),
        RewardTerm("self_collision_penalty",
                   "the episode ends if a link drives into its own chassis — "
                   "which forfeits the rest of the hold", 2.0, _env_owned,
                   is_penalty=True),
    ),
    # No curriculum yet, and that is a measurement rather than an omission:
    # Phase 4a's first run is what says whether the shell needs laddering.
    # The rung to add, if it does, is a NARROWER TARGET SHELL — physics, not
    # pay (AGENTS.md: "Curriculum stages may ladder only physics, spawns and
    # strictness, never the reward").
    default_steps=1_000_000,
    # 8 s: 200 control steps at MARS's 25 Hz. The arm needs ~1-2 s to travel
    # and the success rule asks for the last 1 s inside 2 cm, so the hold is
    # most of the episode rather than a tail it barely reaches.
    episode_s=8.0,
    success_metric="gripper within 2 cm of the target for the last second of "
                   "the episode, on 8 of 8 deterministic seeds",
)

from .core import _register  # noqa: E402 — after the Behaviors are built

_register(MARS_REACH)
