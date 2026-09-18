"""Which TRICK a run practised — what the palette groups "Our runs" by.

The palette listed 500 runs newest-first, so a seed battery buried everything
else under two dozen identical-looking chips. It now groups by the recipe each
run trained, and a 🎓 suggestion chip offers "▶ watch the best run" by looking
a measured pick up under its recipe id. Both rest on `run_trick` reading the
two files the two trainers write — and on an imitation run being filed under
the trick its clip is named after, or the G1's one measured pick (g1_imitate
tracking "g1-front-kick") would never be found under "front kick".
"""

import json

from microduck_local import behaviors as B
from microduck_local.viz_server import available_robots, robot_noun, run_trick, trick_names


def _run(tmp_path, name, **files):
    d = tmp_path / name
    d.mkdir()
    for fname, body in files.items():
        (d / f"{fname}.json").write_text(json.dumps(body))
    return d


def test_a_duck_trick_run_is_its_behavior(tmp_path):
    run = _run(tmp_path, "teach-backflip-abc123-s2", behavior={"behavior": "backflip"})
    assert run_trick(run, "microduck") == "backflip"


def test_a_task_run_resolves_through_that_robots_recipes(tmp_path):
    run = _run(tmp_path, "g1-kick", run={"robot": "g1", "task": "front_kick"})
    assert run_trick(run, "g1") == "g1_front_kick"
    assert B.BEHAVIORS["g1_front_kick"].task == "front_kick"   # what it matched on


def test_a_task_with_no_recipe_keeps_its_name(tmp_path):
    run = _run(tmp_path, "g1-odd", run={"robot": "g1", "task": "moonwalk"})
    assert run_trick(run, "g1") == "moonwalk"


def test_an_imitation_run_is_filed_under_the_trick_its_clip_names(tmp_path):
    # The G1's measured-best front kick: task `imitate`, clip "g1-front-kick".
    by_run_json = _run(tmp_path, "teach-g1_imitate-g1_front_kick-25ac56-s2",
                       run={"robot": "g1", "task": "imitate",
                            "env_kwargs": {"clip_name": "g1-front-kick"}})
    assert run_trick(by_run_json, "g1") == "g1_front_kick"
    by_behavior_json = _run(tmp_path, "teach-imitate-backflip-ffffff",
                            behavior={"behavior": "imitate", "clip": "backflip"})
    assert run_trick(by_behavior_json, "microduck") == "backflip"


def test_a_clip_named_after_nothing_stays_an_imitation_run(tmp_path):
    run = _run(tmp_path, "teach-imitate-wiggle-000000",
               behavior={"behavior": "imitate", "clip": "my-wiggle"})
    assert run_trick(run, "microduck") == "imitate"


def test_a_clip_never_crosses_bodies(tmp_path):
    # A duck run tracking a clip that happens to share a G1 recipe's name is
    # not a G1 front kick.
    run = _run(tmp_path, "teach-imitate-x-111111",
               behavior={"behavior": "imitate", "clip": "g1-front-kick"})
    assert run_trick(run, "microduck") == "imitate"


def test_a_run_with_neither_file_is_unplaced(tmp_path):
    assert run_trick(_run(tmp_path, "ab-kl-05"), "microduck") is None
    broken = _run(tmp_path, "half-written")
    (broken / "behavior.json").write_text("{not json")
    assert run_trick(broken, "microduck") is None


def test_trick_names_cover_recipes_and_skip_strangers():
    names = trick_names([{"trick": "backflip"}, {"trick": "g1_front_kick"},
                         {"trick": "moonwalk"}, {"label": "no trick at all"}])
    assert set(names) == {"backflip", "g1_front_kick"}
    assert names["g1_front_kick"] == {"title": B.BEHAVIORS["g1_front_kick"].title,
                                      "emoji": B.BEHAVIORS["g1_front_kick"].emoji}


def test_every_robot_switch_is_sent_the_same_noun():
    # GET /policies (the palette) and GET /robots (teach, animate) both read
    # robot_noun — the palette said "🤖 g1" beside teach's "🤖 G1".
    #
    # Over the REGISTRY, not over a pair of ids: the switch lists every body
    # this install knows, and pinning the pair here would have to be edited
    # for each new one — which is the pattern the registry exists to delete.
    # The three built-ins' nouns are then pinned by literal, because "the two
    # endpoints agree" is satisfied by two endpoints that are both wrong.
    assert {r["id"]: r["noun"] for r in available_robots()} == {
        r["id"]: robot_noun(r["id"]) for r in available_robots()}
    assert {r["id"] for r in available_robots()} >= {"microduck", "g1", "mars"}
    assert robot_noun("microduck") == "duck"
    assert robot_noun("g1") == "G1"
    assert robot_noun("mars") == "MARS"
