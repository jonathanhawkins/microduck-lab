"""The 🎓 panel's per-robot suggestions come from the recipes, not the viewer.

A chip sends its phrase as /teach text for its robot, so the phrase has to
match the recipe that declared it — otherwise "stand still" on the G1 could
launch a duck trick, or a renamed keyword could silently re-route a chip.
"""

from microduck_local import behaviors as B
from microduck_local.viz_server import match_teach_text, teach_suggestions


def test_every_suggestion_matches_its_own_recipe():
    suggested = [b for b in B.BEHAVIORS.values() if b.suggest]
    assert suggested, "no recipe offers a suggestion chip"
    for b in suggested:
        got, _ = match_teach_text(b.suggest, b.robot)
        assert got is not None and got.id == b.id, (
            f"{b.robot} chip {b.suggest!r} matched {got and got.id!r}, not {b.id!r}")


def test_suggestions_are_scoped_to_the_robot():
    duck = teach_suggestions("microduck")
    g1 = teach_suggestions("g1")
    assert duck and g1
    assert {s["behavior"] for s in duck} <= {b.id for b in B.for_robot("microduck")}
    assert {s["behavior"] for s in g1} <= {b.id for b in B.for_robot("g1")}
    # The duck keeps the five chips the panel always offered.
    assert sorted(s["text"] for s in duck) == sorted([
        "stand still", "stand on one leg", "crouch down", "spin in place",
        "do a headstand"])
    assert teach_suggestions("no-such-robot") == []
