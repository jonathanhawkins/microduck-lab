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
from .core import Behavior, CurriculumStage, RewardTerm


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
        "satisfy the score and not the thing being taught. What it steers are "
        "NUDGES to each joint rather than absolute angles, and that turned out "
        "to be the whole difference: given an angle to aim at, the arm reached "
        "the target but could never sit still on it — no command meant 'stay' — "
        "so it hovered a centimetre or two away for the rest of the episode. A "
        "nudge has a zero, and zero means hold."
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

MARS_PICK = Behavior(
    id="mars_pick",
    emoji="🧱",
    title="Pick a block up (MARS)",
    description="Find a 4 cm block on the floor in front of the robot, close "
                "the gripper on it and lift it clear of the ground.",
    how_it_learns=(
        "Two halves, and each is paid for GETTING somewhere rather than for "
        "being there. First the gripper earns for every centimetre it closes "
        "on the block — the total over an episode is simply how much ground "
        "it made up, so an arm that stalls halfway earns nothing and one that "
        "waves about earns nothing either — plus a small bonus for every step "
        "it spends within two centimetres of it, which is what makes it park "
        "the claw ON the block instead of sweeping past. Then, once it is "
        "actually holding the block, it earns for every centimetre the block "
        "RISES, and a steady bonus for every step it holds it more than five "
        "centimetres up. The lift pay is only ever for a block that is in the "
        "claw, so knocking one into the air earns nothing at all, and "
        "dropping one from height hands back every point the lift earned. "
        "The hard part is not the reward, it is finding a grasp at all, so "
        "the block starts in a six-centimetre square at one spot the arm is "
        "known to be able to pick from, and the square widens to the whole "
        "reach of the arm over three stages. Knocking the block out of reach "
        "ends the attempt, and so does driving a link into its own chassis."
    ),
    keywords=("pick", "pick it up", "pick up the block", "grab it", "grasp",
              "grab the block", "lift it", "pick that up", "take it",
              "close the gripper on it"),
    robot="mars",
    # `train-walk --robot mars --task pick`; TrainingJob appends --run-name,
    # --envs, --steps, --snap-steps and --init-from, and exports each stage's
    # `env` dict into the trainer subprocess.
    trainer=("-m", "microduck_local.train", "--robot", "mars",
             "--task", "pick"),
    terms=(
        RewardTerm("reach_progress",
                   "big points for closing the gap to the block — and the "
                   "same taken back for drifting away from it",
                   40.0, _env_owned),
        RewardTerm("at_target",
                   "a bonus for every step the gripper spends within two "
                   "centimetres of the block", 2.0, _env_owned),
        RewardTerm("lift_progress",
                   "big points for every centimetre the block RISES while it "
                   "is held — and all of them back if it is dropped",
                   100.0, _env_owned),
        RewardTerm("held_high",
                   "a bonus for every step it holds the block more than five "
                   "centimetres off the floor", 4.0, _env_owned),
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
                   "which forfeits the rest of the lift", 2.0, _env_owned,
                   is_penalty=True),
        RewardTerm("block_lost_penalty",
                   "the episode ends if the block is knocked out of the arm's "
                   "reach", 2.0, _env_owned, is_penalty=True),
    ),
    # The PHYSICS ladder, and the whole of it: every stage runs the identical
    # term set and changes only where the block and the ARM may spawn
    # (AGENTS.md, "curriculum stages may ladder only physics, spawns and
    # strictness, never the reward").
    #
    # The DRILL stage is here because of a measurement, not a hunch: rung 1
    # trained from scratch for 1.5 M steps parks the closed claw on the block
    # and collects the proximity bonus (`at_target` +263 of the +267 a
    # scripted success earns) while `lift_progress` and `held_high` stay at
    # 0.000 for the entire run. No rollout ever contains a grasp, so no weight
    # could have taught one — the fix is the world (`robots/mars_env.py`'s
    # `PICK_RUNGS`).
    curriculum=(
        CurriculumStage(
            "the block already in its claw", 1_500_000,
            {"MICRODUCK_MARS_PICK_RUNG": "0"},
            detail="A drill: the arm starts with its jaws already open around "
                   "the block, so all that is left is to squeeze and lift. "
                   "Without it nothing ever grasps at all — trained straight "
                   "on the next stage the robot learns to hover its shut claw "
                   "over the block and collect the proximity bonus forever."),
        CurriculumStage(
            "one spot", 1_500_000, {"MICRODUCK_MARS_PICK_RUNG": "1"},
            detail="Now it has to get to the block first: it lands in a "
                   "6 x 6 cm square at one place in front of the robot — the "
                   "place a scripted pick was measured to work from."),
        CurriculumStage(
            "a wider patch", 1_000_000, {"MICRODUCK_MARS_PICK_RUNG": "2"},
            detail="The same square, now 15 x 15 cm, so the approach has to "
                   "aim rather than repeat."),
        CurriculumStage(
            "anywhere it can reach", 1_500_000,
            {"MICRODUCK_MARS_PICK_RUNG": "3"},
            detail="The block can be anywhere on the floor inside the arm's "
                   "reach — a 120-degree arc from 15 to 40 cm out."),
    ),
    default_steps=1_500_000,
    # 8 s: 200 control steps at MARS's 25 Hz, the same as `reach`. The arm
    # needs ~1-2 s to travel and the success rule asks for the last 1 s
    # holding the block above 5 cm.
    episode_s=8.0,
    success_metric="the block held more than 5 cm off the floor for the last "
                   "second of the episode, on 80% of 20 deterministic seeds",
)

from .core import _register  # noqa: E402 — after the Behaviors are built

_register(MARS_REACH)
_register(MARS_PICK)
