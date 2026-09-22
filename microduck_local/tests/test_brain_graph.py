"""The declared state graph cannot drift from the code that assigns states.

`brain/graph.py` declares, per brain class, the nodes the /sim inspector
draws. Nothing enforces that by construction — the brains set `self.state`
from wherever they like — so this walks each class's AST for every string
literal it can assign to `self.state` (or hand to Tidy's `_enter`) and
asserts the declared node set is exactly that set.

Add a state and forget graph.py and this fails, naming the state. Delete
one and leave it declared and this fails too, so the picture never shows a
node the duck can no longer be in.

The scan has to walk the assigned EXPRESSION, not just match a bare
constant: `Chase` writes `self.state = "wait" if x else "support"` and
`Tidy` writes `self._enter("done" if n >= k else "explore", t)`, and a
scanner that only accepts `ast.Constant` silently misses four real states —
which is exactly the drift this test exists to catch.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from microduck_local.brain import graph

SRC = Path(graph.__file__).parent


def states_assigned(cls: ast.ClassDef) -> set[str]:
    """Every string this class can put in `self.state`."""
    out: set[str] = set()

    def literals(node: ast.AST) -> set[str]:
        return {v.value for v in ast.walk(node)
                if isinstance(v, ast.Constant) and isinstance(v.value, str)}

    for node in ast.walk(cls):
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Attribute) and t.attr == "state"
                   and isinstance(t.value, ast.Name) and t.value.id == "self"
                   for t in node.targets):
                out |= literals(node.value)
        # Tidy's one entry point, which also stamps the dwell clock.
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_enter" and node.args):
            out |= literals(node.args[0])
    return out


def find_class(name: str) -> ast.ClassDef:
    for f in sorted(SRC.glob("*.py")):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
    raise AssertionError(f"no class {name} under {SRC} — graph.BY_CLASS names a class that moved")


@pytest.mark.parametrize("cls_name,key", sorted(graph.BY_CLASS.items()))
def test_declared_nodes_are_exactly_the_states_the_class_assigns(cls_name: str, key: str) -> None:
    declared = {n.name for n in graph.GRAPHS[key].nodes}
    actual = states_assigned(find_class(cls_name))
    assert declared == actual, (
        f"{cls_name} ({key}): "
        f"undeclared {sorted(actual - declared)}, stale {sorted(declared - actual)}")


def test_every_node_has_a_group_the_graph_lists() -> None:
    for key, g in graph.GRAPHS.items():
        keys = {k for k, _ in g.groups}
        bad = sorted({n.group for n in g.nodes} - keys)
        assert not bad, f"{key}: nodes in undeclared groups {bad}"
        empty = sorted(keys - {n.group for n in g.nodes})
        assert not empty, f"{key}: groups with no nodes {empty} — the layout would show a gap"


def test_every_node_says_what_it_means() -> None:
    """The notes ARE the feature: a node with no line teaches nothing.

    A word count, not a character count — "walk to the toy" is a complete
    explanation and a longer sentence is not automatically a better one."""
    for key, g in graph.GRAPHS.items():
        for n in g.nodes:
            assert len(n.note.split()) >= 3, f"{key}.{n.name}: note is not a sentence"
            assert n.note.strip() != n.name, f"{key}.{n.name}: note just repeats the state name"


def test_payload_is_json_and_keyed_the_way_the_viewer_indexes_it() -> None:
    import json

    p = graph.payload()
    assert json.loads(json.dumps(p)) == p
    for key, g in p.items():
        assert g["key"] == key
        assert [n["name"] for n in g["nodes"]] == [n.name for n in graph.GRAPHS[key].nodes]


def test_graph_key_reads_the_class_not_the_kind() -> None:
    """Every `learned:<run>` is its own registry kind and one picture."""
    class Fake:
        pass

    Fake.__name__ = "LearnedBrain"
    assert graph.graph_key(Fake()) == "learned"
    assert graph.graph_key(None) is None

    class Stranger:
        pass

    assert graph.graph_key(Stranger()) is None
