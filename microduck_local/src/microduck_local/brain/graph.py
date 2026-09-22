"""The states a brain moves through, as a graph the /sim inspector can draw.

**This is not a specification.** None of the scripted brains here is a
declared state machine — `Chase` assigns `self.state` from twenty-odd
places in one long `step`, and the order those branches fire in is an
emergent property of the pitch, not a table anyone wrote down. So this
module declares only the NODES: what states exist, what each one means,
and which ones belong together. The EDGES are drawn by the viewer from
the transitions it actually watches fire, live, this run.

That split is deliberate and it is what makes the picture honest:

  * a declared node that never lights up is visible as a dim orphan — a
    branch the scenario never reaches (or dead code);
  * a transition nothing declared still appears, because edges are
    observed rather than allowed;
  * and the picture cannot quietly drift from the code, because
    `tests/test_brain_graph.py` walks each brain class's AST for every
    string it can assign to `self.state` and asserts the node set here is
    exactly that set. Add a state and forget this file and the tests fail.

Keyed by CLASS, not by the registry kind: `learned:follow-v4` and
`learned:p-n256-s31` are different kinds and the same graph, and a kind
rename should not silently drop the picture.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    name: str
    group: str
    note: str                      # what the duck is doing here, in one line


@dataclass(frozen=True)
class BrainGraph:
    key: str
    title: str
    note: str                      # what kind of thing this brain is
    groups: list[tuple[str, str]]  # (key, heading) in reading order
    nodes: list[Node]

    def to_dict(self) -> dict:
        return {"key": self.key, "title": self.title, "note": self.note,
                "groups": [list(g) for g in self.groups],
                "nodes": [{"name": n.name, "group": n.group, "note": n.note} for n in self.nodes]}


def _g(key: str, title: str, note: str, groups: list[tuple[str, str]],
       nodes: dict[str, list[tuple[str, str]]]) -> BrainGraph:
    return BrainGraph(key, title, note, groups,
                      [Node(n, grp, t) for grp, _ in groups for n, t in nodes.get(grp, [])])


CHASE = _g(
    "chase", "Chase — the scripted soccer brain",
    "Eighteen states over the frozen walker. Everything below the twist — the gait, the "
    "kick — is a trained policy; everything here is hand-written.",
    [("find", "find the ball"), ("go", "go to it"), ("strike", "hit it"),
     ("safe", "stay safe"), ("team", "team duties")],
    {
        "find": [
            ("search", "nothing seen — turn on the spot in a walking circle, biased to the side the ball was last on"),
            ("hunt", "lost it on the way in — keep driving the last heading, bent toward where the ball is predicted to be"),
            ("seek", "walk to where the ball was remembered, and forget it on arrival"),
            ("look", "stand and sweep the head across the last known bearing"),
        ],
        "go": [
            ("turn", "the ball is too far off the nose to walk at — turn first"),
            ("chase", "walk at the ball, steering by its bearing"),
            ("lineup", "the camera lost the ball ~0.2 m out — dead-reckon in odometry to the planned spot behind it"),
            ("settle", "on the spot and squared up — stand still before the swing"),
        ],
        "strike": [
            ("kick", "the shipped kick policy owns the body for one cycle"),
            ("push", "dribble through the ball instead of kicking it — the planner's third action"),
        ],
        "safe": [
            ("retreat", "back off: the spot is unreachable, or we are stood against something"),
            ("avoid", "another duck ahead — turn out of it"),
            ("blocked", "the ToF says something is right there — turn at max rate"),
        ],
        "team": [
            ("support", "not the attacker — hold the support spot, back toward our own goal"),
            ("wait", "standing off the other side's kickoff"),
            ("block", "get between the ball's line and our goal (the keeper's job)"),
            ("duel", "an opponent wants the same ball — contest it"),
            ("yield", "a teammate has it — stand off"),
        ],
    })

TIDY = _g(
    "tidy", "Tidy — the playroom loop",
    "The one brain here that keeps its states honestly: every change goes through `_enter`, "
    "which stamps the time it started, so dwell is a real number rather than a guess.",
    [("find", "find a toy"), ("go", "go to it"), ("grasp", "pick it up"),
     ("carry", "carry it"), ("deliver", "put it in the basket"), ("end", "finish")],
    {
        "find": [
            ("scan", "turn on the spot looking for something to pick up"),
            ("explore", "nothing here — walk somewhere else and scan again"),
        ],
        "go": [
            ("approach", "walk to the toy"),
            ("blind", "the toy left the camera on the way in — close the last stretch on memory"),
            ("settle", "stand still before the grab"),
            ("backoff", "too close, or stood against something — back off and come again"),
        ],
        "grasp": [
            ("pick", "the ground-pick skill owns the body"),
            ("verify", "did the beak actually get it?"),
        ],
        "carry": [
            ("carry", "walk to the basket holding the toy"),
            ("carry_explore", "holding it, but the basket is not in sight — go and look for it"),
        ],
        "deliver": [
            ("aim", "square up on the basket"),
            ("deliver", "the last stretch to the basket"),
            ("drop", "open the beak"),
        ],
        "end": [
            ("done", "two clean scans with nothing seen — the room is tidy"),
        ],
    })

TIDY_ARM = _g(
    "tidy_arm", "TidyArm — the playroom loop, with an ARM",
    "A sibling of Tidy, not a subclass: a wheeled base with a gripper has no beak, no gait, "
    "no fall and no blind leg (its camera keeps a floor toy in frame all the way to the grasp), "
    "so the three legs the duck spends on those are four ARM phases instead.",
    [("find", "find a toy"), ("go", "go to it"), ("grasp", "pick it up"),
     ("carry", "carry it"), ("deliver", "put it in the basket"),
     ("stuck", "stuck"), ("end", "finish")],
    {
        "find": [
            ("search", "turn on the spot looking for something to pick up"),
            ("explore", "nothing here — drive somewhere else and look again"),
        ],
        "go": [
            ("approach", "drive so the toy lands inside the arm's reach shell"),
            ("settle", "stand still, re-measure, and solve the grasp pose"),
        ],
        "grasp": [
            ("hover", "jaws open 8 cm ABOVE the toy — and where the arm's sag is measured"),
            ("reach", "straight down onto the toy, sag pre-compensated"),
            ("close", "squeeze — the blades either load or they do not"),
            ("lift", "raise the grasp point clear of the floor"),
        ],
        "carry": [
            ("carry", "holding it — find the basket (the lidar cannot see a 6 cm rim)"),
            ("carry_explore", "holding it and the basket is nowhere in sight — go and look"),
        ],
        "deliver": [
            ("deliver", "drive to the standoff the arm can reach over the rim from"),
            ("place", "extend the arm over the tray"),
            ("drop", "open the jaw"),
            ("retract", "arm back to HOME"),
        ],
        "stuck": [
            ("unstick", "asked to move and did not — reverse and turn"),
        ],
        "end": [
            ("done", "six clean scans with nothing seen — the room is tidy"),
        ],
    })

WANDER = _g(
    "wander", "Wander — cruise on the ToF",
    "The smallest scripted brain: five states, and the only sensor is the 64-zone ToF.",
    [("drive", "drive"), ("stuck", "stuck")],
    {
        "drive": [
            ("cruise", "walk forward, nothing in the way"),
            ("steer", "turn toward the open side while walking"),
            ("spin", "turn in place — no open side ahead"),
        ],
        "stuck": [
            ("unstick", "we asked to move and did not — back out"),
            ("blind", "the ToF frame is too old to steer on"),
        ],
    })

FOLLOW = _g(
    "follow", "Follow — keep a person ahead",
    "The scripted baseline the learned followers are measured against, and the one benchmark "
    "where the learned brain wins.",
    [("find", "find them"), ("go", "hold the band"), ("safe", "stay safe")],
    {
        "find": [
            ("search", "no one in sight — turn and look"),
            ("coast", "the detection went stale — hold the last intent briefly"),
        ],
        "go": [
            ("approach", "walk toward them"),
            ("hold", "in the distance band and squared up — stand"),
        ],
        "safe": [
            ("blocked", "something in the way"),
            ("dodge", "step around it"),
        ],
    })

SCRIPT = _g(
    "script", "Script — no brain",
    "The drive script or the manual command steers; there is nothing to show.",
    [("manual", "driven")],
    {"manual": [("script", "someone else is steering")]})

LEARNED = _g(
    "learned", "A learned brain — no states at all",
    "This is the honest picture: a trained brain has no states to move between. These two "
    "labels are not a state machine — they are one slot of the observation (obs[65], "
    "\"is the target visible\") given a name so the page has something to print. Everything "
    "this brain knows is in the observation/action panel above; that is where to look.",
    [("net", "read off the observation")],
    {"net": [
        ("learned", "before the first decision"),
        ("tracking", "the network's input says the target is visible"),
        ("lost", "it does not"),
    ]})

STRIKER = _g(
    "striker", "A learned striker — no states at all",
    "The same non-state-machine as any learned brain, on the pitch. Measured against the "
    "eighteen-state scripted Chase on identical seeds, this one loses: 0.31 against 0.86 m/min "
    "of ball advance, and 43x less ball moved per touch.",
    [("net", "read off the observation")],
    {"net": [
        ("striker", "before the first decision"),
        ("ball", "the network's input says the ball is visible"),
        ("lost", "it does not"),
    ]})

GRAPHS: dict[str, BrainGraph] = {g.key: g for g in
                                 (CHASE, TIDY, TIDY_ARM, WANDER, FOLLOW, SCRIPT,
                                  LEARNED, STRIKER)}

# Which graph a brain OBJECT draws, by class name. The registry kind is not
# the key: every `learned:<run>` is its own kind and they all share one
# picture.
BY_CLASS: dict[str, str] = {
    "Chase": "chase", "Tidy": "tidy", "TidyArm": "tidy_arm",
    "Wander": "wander", "Follow": "follow",
    "Script": "script", "LearnedBrain": "learned", "LearnedStriker": "striker",
}


def graph_key(brain) -> str | None:
    """The graph this brain draws, or None for one nothing is declared for
    (a scenario's own controller, a test double) — the panel then simply
    shows the state text it always showed."""
    return None if brain is None else BY_CLASS.get(type(brain).__name__)


def payload() -> dict:
    """Every graph, for the world-info message. Sent once per connection,
    not per frame: the per-duck block carries only the key."""
    return {k: g.to_dict() for k, g in GRAPHS.items()}


__all__ = ["BY_CLASS", "BrainGraph", "GRAPHS", "Node", "graph_key", "payload"]
