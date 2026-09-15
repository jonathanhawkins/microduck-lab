"""Teachable tasks for the Unitree G1 — the panel's entry to another body.

Everything else in this package is a duck reward recipe evaluated inside
`BehaviorEnv`. A G1 task is different in one way only: its reward lives in
an env subclass (`robots/g1_env.G1StandEnv`) and a different trainer runs
it (`train-walk --robot g1 --task …`). It registers here so the 🎓 teach
panel, the job card, the live-snapshot preview and the trainee duck all
work for the G1 exactly as they do for a trick.

The `terms` below are what the panel SHOWS. They are display rows, not
callables the env evaluates — the env owns the reward, and `_env_owned`
exists so that is visible at a glance rather than implied.
"""

from .core import *  # noqa: F401,F403 — the package's namespace cascade
from .core import Behavior, CurriculumStage, RewardTerm


def _env_owned(env):  # noqa: ARG001 — see the module docstring
    """Placeholder: G1 rewards live in the env, not in this library."""
    return 0.0


G1_STAND = Behavior(
    id="g1_stand",
    emoji="🧍",
    title="Stand still (G1)",
    description="Hold a still, upright stance and ignore the drive command.",
    how_it_learns=(
        "It is scored for standing where it is: no body velocity, no spin, "
        "upright, at full standing height, both feet on the floor with the "
        "soles flat, its joints near their default pose, and its body still "
        "on the spot it started from. The last two matter more than they "
        "sound — a standing robot sways by rocking on its ankles, and "
        "scoring only velocity lets a sway that averages out go free. Half "
        "the practice episodes ask it to walk somewhere; it is paid for "
        "ignoring that too, so nothing in the lab can nudge it into "
        "wandering off."
    ),
    keywords=("stand still", "stand", "idle", "hold still", "stay", "stay still",
              "don't move", "do nothing", "at ease", "wait"),
    robot="g1",
    # `train-walk --robot g1 --task stand`; TrainingJob appends --run-name,
    # --envs, --steps, --snap-steps and --init-from.
    trainer=("-m", "microduck_local.train", "--robot", "g1", "--task", "stand"),
    terms=(
        RewardTerm("still", "stays where it is — no drifting", 3.0, _env_owned),
        RewardTerm("no_spin", "doesn't turn on the spot", 1.0, _env_owned),
        RewardTerm("upright", "keeps its chest level", 2.0, _env_owned),
        RewardTerm("height", "stands at full height instead of sagging", 2.0, _env_owned),
        RewardTerm("feet_planted", "keeps both feet on the floor", 1.0, _env_owned),
        RewardTerm("flat_feet", "keeps its soles flat instead of rocking on them",
                   1.0, _env_owned),
        RewardTerm("stay_home", "stays on the spot it started on", 1.0, _env_owned),
        RewardTerm("pose", "holds its joints near their resting pose", 1.0, _env_owned),
        RewardTerm("action_rate_penalty", "charged for twitchy joint commands",
                   1.0, _env_owned, is_penalty=True),
        RewardTerm("joint_vel_penalty", "charged for moving its joints at all",
                   0.002, _env_owned, is_penalty=True),
    ),
    curriculum=(
        CurriculumStage(
            "hold still", 700_000, {},
            detail="Practises holding its ground with nothing asked of it. "
                   "This is the stage that has to work first: an idle that "
                   "cannot stand cannot learn to ignore anything."),
        CurriculumStage(
            "ignore being driven", 500_000,
            {"MICRODUCK_G1_COMMAND_MIX": "0.5"},
            detail="Half the episodes now ASK it to walk somewhere and pay "
                   "it for staying put. Trained the other way round — mixing "
                   "commands in from step one — every commanded episode was a "
                   "one-second collapse and PPO learned from the wreckage."),
    ),
    default_steps=1_200_000,
    # 40 s, matching what train.py actually gives these two tasks. The 🎓
    # trainee previews THIS number, so 10 s here meant the watched preview
    # reset three times before reaching the settled regime the 40 s episode
    # exists to produce.
    episode_s=40.0,
    success_metric="stands 60 s without sagging below 0.5 m or being driven off",
)

G1_SQUAT = Behavior(
    id="g1_squat",
    emoji="🏋️",
    title="Hold a squat (G1)",
    description="Sink into a squat about 20 cm below standing and hold it.",
    how_it_learns=(
        "Same recipe as standing still, parked lower: the big points are for "
        "holding its hips at the squat height instead of standing height, and "
        "the reward for holding its joints in the normal standing pose is "
        "removed — a squat is precisely not that pose, and paying for both "
        "asks it to bend its knees and keep them straight at the same time. "
        "Everything that keeps the idle steady is kept: flat soles, both feet "
        "down, no drifting off the spot, no spin."
    ),
    keywords=("squat", "do a squat", "crouch", "crouch down", "sink down",
              "get low", "bend your knees", "hold a squat", "deep squat"),
    robot="g1",
    trainer=("-m", "microduck_local.train", "--robot", "g1", "--task", "squat"),
    terms=(
        RewardTerm("height", "big points for holding its hips at squat height",
                   6.0, _env_owned),
        RewardTerm("upright", "keeps its chest level while it is down there",
                   2.0, _env_owned),
        RewardTerm("still", "stays where it is — no drifting", 3.0, _env_owned),
        RewardTerm("no_spin", "doesn't turn on the spot", 1.0, _env_owned),
        RewardTerm("feet_planted", "keeps both feet on the floor", 1.0, _env_owned),
        RewardTerm("flat_feet", "keeps its soles flat instead of rocking on them",
                   1.0, _env_owned),
        RewardTerm("stay_home", "stays on the spot it started on", 1.0, _env_owned),
        RewardTerm("carriage", "holds its back straight and its arms evenly at "
                   "its sides instead of twisting", 2.0, _env_owned),
        RewardTerm("action_rate_penalty", "charged for twitchy joint commands",
                   1.0, _env_owned, is_penalty=True),
        RewardTerm("joint_vel_penalty", "charged for moving its joints at all",
                   0.002, _env_owned, is_penalty=True),
    ),
    curriculum=(
        CurriculumStage(
            "sink into it", 700_000, {},
            detail="Learns to leave standing height and hold the squat with "
                   "nothing asked of it."),
        CurriculumStage(
            "ignore being driven", 500_000,
            {"MICRODUCK_G1_COMMAND_MIX": "0.5"},
            detail="Half the episodes ask it to walk somewhere while it is "
                   "down there; it is paid for staying put."),
    ),
    default_steps=1_200_000,
    # 40 s, matching what train.py actually gives these two tasks. The 🎓
    # trainee previews THIS number, so 10 s here meant the watched preview
    # reset three times before reaching the settled regime the 40 s episode
    # exists to produce.
    episode_s=40.0,
    success_metric="holds the squat height for 60 s without standing back up "
                   "or toppling, and ignores a drive command",
)



# --- karate: strikes held at full extension ------------------------------
#
# Each is a HOLD of a solved pose, the shape the squat proved this harness
# learns. The pose is solved against the compiled model (robots/g1_karate.py),
# and for the kick the solve is a STATIC BALANCE problem, not a bit of
# posing: the CoM has to sit inside a 0.203 x 0.088 m support foot while the
# other leg is half a metre in the air.

_STRIKE_SHARED = (
    RewardTerm("upright", "keeps its chest level through the strike",
               2.0, _env_owned),
    RewardTerm("carriage", "holds everything it is NOT striking with "
               "composed — no twisting, no flailing", 2.0, _env_owned),
    RewardTerm("no_spin", "doesn't turn on the spot", 1.0, _env_owned),
    RewardTerm("stay_home", "strikes on the spot instead of travelling",
               1.0, _env_owned),
    RewardTerm("action_rate_penalty", "charged for twitchy joint commands",
               1.0, _env_owned, is_penalty=True),
    RewardTerm("joint_vel_penalty", "charged for moving its joints at all",
               0.002, _env_owned, is_penalty=True),
)

G1_FRONT_KICK = Behavior(
    id="g1_front_kick",
    emoji="🦵",
    title="Front kick (G1)",
    description="Mae geri — throw a high front kick, then come back to a "
                "normal standing stance, over and over.",
    how_it_learns=(
        "This one is a MOVE, not a pose: every two seconds it stands, throws "
        "the kicking leg up and forward to about waist height, holds it for a "
        "moment, and puts it back down on both feet. It is scored against "
        "where the kick should be at that instant, so the same points that "
        "pay for getting the leg up also pay for bringing it back — standing "
        "still on one leg earns nothing once the kick is over. It is told "
        "where it is in the cycle through its command slots, because a robot "
        "with no memory cannot time a repeating movement otherwise. The "
        "balance is worked out in advance: standing normally its weight sits "
        "12 cm to the side of either foot, so the kick pose leans it over the "
        "supporting foot before the leg ever leaves the floor."
    ),
    keywords=("front kick", "kick", "mae geri", "karate kick", "leg kick",
              "do a kick", "throw a kick", "high kick"),
    robot="g1",
    trainer=("-m", "microduck_local.train", "--robot", "g1",
             "--task", "front_kick"),
    terms=(
        RewardTerm("strike", "big points for the kicking leg being where the "
                   "kick should be at that moment — including back down",
                   6.0, _env_owned),
        RewardTerm("lift", "big points for the kicking foot actually being "
                   "off the floor, as high as the kick is meant to be at that "
                   "moment — standing still earns nothing here", 4.0, _env_owned),
        RewardTerm("stance", "one foot planted at full extension, BOTH feet "
                   "down once the kick is over", 2.0, _env_owned),
        RewardTerm("balance", "keeps its weight over whichever foot is "
                   "carrying it", 3.0, _env_owned),
        RewardTerm("height", "stays standing tall through the kick instead of "
                   "sagging into it", 3.0, _env_owned),
        RewardTerm("still", "settles between kicks (scored only when it is "
                   "meant to be idle)", 1.0, _env_owned),
        RewardTerm("spin_cost", "charged for building up spin — it has to "
                   "swing its arms to cancel the kick, not turn with it",
                   1.5, _env_owned, is_penalty=True),
        *_STRIKE_SHARED,
    ),
    curriculum=(
        # The apex is HOLDABLE — 13.2's held kick sustained 0.545 m for 20 s —
        # so a high kick that falls is a TRANSITION problem, not a pose
        # problem. Measured: freeing the arms and pricing real angular
        # momentum took the kick from 13% to 105% of target, and it still
        # toppled after one or two, with only 0.8 s to throw and recover.
        # Ladder the TIME (the world), not the pay.
        CurriculumStage(
            "a slow, high kick", 3_000_000,
            {"MICRODUCK_G1_KICK_HIP": "-1.5", "MICRODUCK_G1_KICK_CYCLE": "8.0"},
            detail="Full height with twice as long to throw it and get the "
                   "foot back down."),
        CurriculumStage(
            "tighten it up", 3_000_000,
            {"MICRODUCK_G1_KICK_HIP": "-1.5", "MICRODUCK_G1_KICK_CYCLE": "6.0"},
            detail="The same kick, thrown quicker."),
        CurriculumStage(
            "full speed", 3_000_000,
            {"MICRODUCK_G1_KICK_HIP": "-1.5", "MICRODUCK_G1_KICK_CYCLE": "4.0"},
            detail="The kick it keeps: waist height, thrown in 0.8 s."),
    ),
    default_steps=9_000_000,
    episode_s=40.0,
    success_metric="throws a waist-high kick and returns to both feet every "
                   "cycle for 20 s without toppling",
)

G1_PUNCH = Behavior(
    id="g1_punch",
    emoji="🥊",
    title="Straight punch (G1)",
    description="Choku zuki — drive one arm forward to full extension and "
                "hold the guard.",
    how_it_learns=(
        "The big points are for the punching arm being fully out in front "
        "with the elbow straight, while the other arm chambers back at the "
        "hip the way a karate guard does. Both feet stay down and the knees "
        "stay soft, so this is a bracing problem rather than a balancing "
        "one — the reward for the rest of the body keeps it from leaning "
        "into the punch or twisting away from it."
    ),
    keywords=("punch", "straight punch", "choku zuki", "jab", "throw a punch",
              "karate punch", "strike"),
    robot="g1",
    trainer=("-m", "microduck_local.train", "--robot", "g1",
             "--task", "punch"),
    terms=(
        RewardTerm("strike", "big points for the punching arm being where a "
                   "straight punch puts it", 6.0, _env_owned),
        RewardTerm("stance", "keeps both feet planted through the punch",
                   1.0, _env_owned),
        RewardTerm("balance", "keeps its weight centred between its feet",
                   3.0, _env_owned),
        RewardTerm("height", "stays standing tall through the strike instead of sagging into it", 2.0, _env_owned),
        RewardTerm("still", "punches without shoving itself around",
                   1.0, _env_owned),
        *_STRIKE_SHARED,
    ),
    curriculum=(
        # A LEVEL punch is the hardest arm position to hold: the moment the
        # fist is at shoulder height the arm's whole weight acts on the
        # longest possible moment arm, and the reaction goes into the torso.
        # Measured against an otherwise identical recipe whose fist sat 36 deg
        # high (a shorter moment arm): the raised one held 4/5 seeds for 20 s,
        # the level one held 0/5 — balancing cleanly for 3 s and then losing
        # it in half a second. Training here is deterministic, so re-running
        # reproduced it exactly; the answer is budget, not luck.
        CurriculumStage(
            "reach the punch", 1_500_000, {},
            detail="Learns to drive the arm out and hold it there without "
                   "leaning into it."),
        CurriculumStage(
            "hold it under pressure", 1_000_000,
            {"MICRODUCK_G1_COMMAND_MIX": "0.5"},
            detail="Half the runs ask it to walk off mid-punch; it is paid "
                   "for staying put."),
    ),
    default_steps=2_500_000,
    episode_s=40.0,
    success_metric="holds the punch for 20 s with both feet down and the "
                   "body square",
)

# --- imitation: perform a clip authored in the 🎬 panel -------------------
#
# The duck's `imitate` recipe, on the other body. The clip rides
# MICRODUCK_CLIP into the trainer and the preview exactly as it does for the
# duck (viz_server /teach); the reward lives in robots/g1_imitate.py and is
# the idle's every term retargeted at the clip's pose for the current instant,
# plus a Cartesian foot term — the one a joint-space pose match cannot supply
# for a kick (docs/roadmap.md 13.3 measured six reward formulations at a
# ~0.1 m ceiling; drawing the kick makes it a tracking problem instead).

G1_IMITATE = Behavior(
    id="g1_imitate",
    emoji="🎬",
    title="Copy the animation (G1)",
    description=(
        "Physically perform a keyframed motion clip: match the authored pose, "
        "lean and footwork at every instant."
    ),
    how_it_learns=(
        "This one is not asked to invent anything. An animation clip says "
        "where every joint should be at each moment, and it earns for being "
        "there — joints near the clip's, hips at the clip's height, leaning "
        "as the clip leans, each foot planted where the clip plants it and "
        "lifted as high as the clip lifts it. It is told where it is in the "
        "clip through its command slots, because a robot with no memory "
        "cannot time a motion otherwise. What it still has to solve is the "
        "physics: momentum, contact and balance are not in the clip."
    ),
    keywords=("copy the animation", "imitate", "follow the clip",
              "animation clip", "motion clip", "do the animation",
              "perform the clip", "track the clip"),
    robot="g1",
    trainer=("-m", "microduck_local.train", "--robot", "g1",
             "--task", "imitate"),
    terms=(
        RewardTerm("foot_track", "big points for each foot being where the "
                   "clip puts it, relative to the hips — the reach of a kick",
                   4.0, _env_owned),
        RewardTerm("carriage", "big points for every joint matching the clip's "
                   "pose right now, precisely", 3.0, _env_owned),
        RewardTerm("pose", "points for being roughly in the clip's pose — a "
                   "gradient from wherever it starts", 2.0, _env_owned),
        RewardTerm("height", "keeps its hips at the height the clip has them",
                   3.0, _env_owned),
        RewardTerm("upright", "leans exactly as much as the clip leans",
                   2.0, _env_owned),
        RewardTerm("feet", "a foot the clip plants is down and flat; a foot "
                   "the clip lifts is paid for getting up there", 3.0, _env_owned),
        RewardTerm("still", "doesn't drift while the clip stands still",
                   0.5, _env_owned),
        RewardTerm("no_spin", "doesn't turn on the spot", 1.0, _env_owned),
        RewardTerm("stay_home", "performs on the spot instead of travelling",
                   1.0, _env_owned),
        RewardTerm("action_rate_penalty", "charged for twitchy joint commands",
                   0.1, _env_owned, is_penalty=True),
        RewardTerm("joint_vel_penalty", "charged a little for joint speed",
                   0.0005, _env_owned, is_penalty=True),
    ),
    curriculum=(
        # A STRICTNESS ladder (AGENTS.md: stages may ladder physics, spawns
        # and strictness, never the pay). The env ends an episode that
        # stands through a lift the clip asks for — falling and not-kicking
        # are equally fatal, which is what removes the "standing is safe"
        # exploit six reward formulations died on — and each rung raises
        # how much lift counts. Measured: with no rule at all the policy
        # stood through every kick (apex 0.000 m at 6/6 survival); warm-
        # started from the 0.1 m cycle kick it clears the 5 cm rung at once.
        CurriculumStage(
            "any lift keeps it alive", 1_500_000,
            {"MICRODUCK_G1_LIFT_MIN": "0.05"},
            detail="Starts inside the clip more often than not, so every "
                   "moment of it is rehearsed from the first step; a foot "
                   "that stays down through a kick ends the episode."),
        CurriculumStage(
            "knee height", 1_500_000,
            {"MICRODUCK_G1_LIFT_MIN": "0.15"},
            detail="The kick must clear 15 cm to count, and the tracking "
                   "terms pull it the rest of the way."),
        CurriculumStage(
            "half the clip's height", 1_500_000,
            {"MICRODUCK_G1_LIFT_MIN": "0.30"},
            detail="30 cm or the episode ends — half of what the clip "
                   "draws, and three times what reward search ever found."),
    ),
    default_steps=4_500_000,
    episode_s=20.0,
    success_metric="performs the clip through 20 s without toppling, feet "
                   "reaching where the clip puts them",
)

from .core import _register  # noqa: E402 — after the Behaviors are built

_register(G1_STAND)
_register(G1_SQUAT)
_register(G1_FRONT_KICK)
_register(G1_PUNCH)
_register(G1_IMITATE)
