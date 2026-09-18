"""What a policy file SAYS IT IS: a contract id, not an observation width.

Named `policy_contract.py` and not `contract.py` because that name is taken,
and taken by the thing this module is not: `microduck_local/contract.py` is
the DUCK's deployment contract — 61 floats in a fixed order, 14 servos, the
constants mirrored from `microduck_rl`. This file is the generic record any
body's policy carries, and the duck's is one instance of it.

**The problem it removes.** Today a run says which body it trained on by
NAME, in its own `run.json`, and the lab refuses a policy on the wrong body
by observation WIDTH — 61 against 99, at four sites in `viz_server.py`. Width
is a proxy, and a proxy is exactly what this repo has been bitten by before:

  * two bodies at the same width would cross in silence. MARS declares 32 and
    the duck 61 today, but nothing stops the fourth body from declaring 61,
    and the failure mode is not an exception — it is a policy that loads,
    runs and emits plausible-looking nonsense, which is the hardest kind of
    wrong to see (`AGENTS.md`, "Verification discipline").
  * an ONNX file handed around without its run directory knows nothing about
    itself. `eval-walk some.onnx` two directories away from where it was
    exported had to be TOLD its body, and `--robot` has been the answer.
  * a width says nothing about what the floats MEAN. Same 61, different
    order, is a silent cross too — and the layout is the whole hot-swap
    contract.

The brain layer already got this right: `brain.json` carries `obs_version`,
and `brain/learned.py` reads it back (a brain written before the key existed
is version 1). This is the same idea for the reflex tier, with the layout and
the deployment caveat carried along because a policy leaves the building and
a brain does not.

**What a contract is.** An id that is a VERSION (`microduck-61-v1`), the
dims, the control rate, a slot table naming every float, and one sentence of
`deploy` honesty saying what running this policy on real hardware would
actually mean. The id is the comparison; everything else is what makes the id
readable by a person who has only the file.

**Where it is written** (§6.3 of `docs/mars-roadmap.md`): into the ONNX's own
`metadata_props` by `export_onnx.export`, into `run.json` by `train.py` and
`distill.py`, and its id into `record.json` (what a person reads). **Where it
is read**: `resolve()`, below — the one function every reader should call.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

#: The ONNX `metadata_props` keys. `contract_id` is the cheap one — it is
#: readable with `strings policy.onnx | grep` and by any tool that does not
#: want to parse JSON — and `contract` is the whole record. Both are written;
#: `from_onnx` reads the JSON, because a reader that trusted the bare id
#: would have to look the layout up in a registry that may not have the body.
KEY_ID = "contract_id"
KEY_JSON = "contract"

#: The body a policy with nothing to say belongs to. See `resolve`.
DEFAULT_ROBOT = "microduck"


class Slot(NamedTuple):
    """One named run of floats in the observation: `[start, stop)`.

    A tuple, so a slot round-trips through JSON as the three values it is and
    a test can compare it against a literal. `name` is the term's name in the
    env that fills it (`joint_pos_rel`, `twist_cmd`) rather than a prettified
    label, so the table can be read straight against `_get_obs`.
    """

    name: str
    start: int
    stop: int

    @property
    def width(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class PolicyContract:
    """What one body's policies speak: dims, rate, layout, and the id.

    Frozen and comparable: `from_onnx(p) == body.contract()` is the check a
    test wants, and a record that could be mutated after it was read is a
    record two readers can disagree about.
    """

    #: The comparison, and a VERSION. `<body>-<width>-v<n>`: the width is in
    #: the id on purpose, so a mismatch is legible in a log line without
    #: looking anything up, and the `v` is what a deliberate layout change
    #: bumps. Two contracts with the same id must agree on everything else —
    #: `matches()` rests on that, and `tests/test_policy_contract.py` pins it.
    id: str
    #: The registry id of the body this policy drives (`robots/registry.py`).
    robot: str
    obs_dim: int
    act_dim: int
    #: Control rate, Hz. Part of the contract because a policy trained at
    #: 50 Hz and run at 25 is a different controller — MARS's skills tick at
    #: 25 Hz and the walkers at 50, and nothing else in a policy file says so.
    rate_hz: float
    #: Every float in the observation, named, in order. See `tiles()`.
    slots: tuple[Slot, ...] = ()
    #: One sentence: what DEPLOYING this policy means. The duck's is the
    #: robot's own contract; the G1's is a lab contract (it observes base
    #: linear velocity, which no humanoid has without state estimation);
    #: MARS's is an Innate code skill. This travels with the file because the
    #: caveat is the thing that gets lost when a policy is passed along, and
    #: `AGENTS.md`'s "sim2real honesty" is otherwise a sentence in a README
    #: nobody reading `policy.onnx` will ever see.
    deploy: str = ""

    def __post_init__(self) -> None:
        # Normalise first: `from_dict` hands in lists of lists (JSON has no
        # tuples), and a contract whose slots are sometimes lists and
        # sometimes Slots is one that compares unequal to itself.
        object.__setattr__(self, "slots", tuple(
            s if isinstance(s, Slot) else Slot(str(s[0]), int(s[1]), int(s[2]))
            for s in self.slots))
        object.__setattr__(self, "obs_dim", int(self.obs_dim))
        object.__setattr__(self, "act_dim", int(self.act_dim))
        object.__setattr__(self, "rate_hz", float(self.rate_hz))
        if not str(self.id):
            raise ValueError("a PolicyContract needs an id — it IS the check")
        if self.obs_dim <= 0 or self.act_dim <= 0:
            raise ValueError(f"{self.id}: obs_dim/act_dim must be positive, "
                             f"got {self.obs_dim}/{self.act_dim}")
        if self.rate_hz <= 0:
            raise ValueError(f"{self.id}: rate_hz must be positive, "
                             f"got {self.rate_hz}")
        problems = self.tiles()
        if problems:
            raise ValueError(f"{self.id}: the slot table does not describe "
                             f"{self.obs_dim} floats — " + "; ".join(problems))

    # ------------------------------------------------------------- the layout

    def tiles(self) -> tuple[str, ...]:
        """Why the slots do NOT partition `[0, obs_dim)` — empty when they do.

        Problems, not a bool, for the reason `robots/body.conforms()` returns
        names instead of False: a caller reading "does not tile" learns
        nothing, and `__post_init__` needs the sentence for its error. A test
        asserting `tiles() == ()` reads as the predicate it wants.

        An exact partition is the point of having a table at all. A GAP is an
        undeclared float — the reader cannot tell whether it is padding or a
        term someone forgot to write down, which is precisely the ambiguity
        that makes a width a bad contract. An OVERLAP is two names for one
        float, so the table no longer says what a float means. MARS's four
        reserved zeros are therefore a declared slot (`reserved`) rather than
        a gap: "room for a task's own slots" is information, and absence is
        not.

        An EMPTY table is refused for the same reason: a contract that names
        none of its floats is a width wearing an id.
        """
        if not self.slots:
            return (f"no slots declared (a width is not a contract; "
                    f"{self.obs_dim} floats are undescribed)",)
        problems: list[str] = []
        ordered = sorted(self.slots, key=lambda s: (s.start, s.stop))
        cursor = 0
        for s in ordered:
            if s.stop <= s.start:
                problems.append(f"{s.name} is empty or inverted "
                                f"[{s.start}, {s.stop})")
                continue
            if s.start < cursor:
                problems.append(f"{s.name} [{s.start}, {s.stop}) overlaps the "
                                f"slot before it (which ended at {cursor})")
            elif s.start > cursor:
                problems.append(f"nothing describes [{cursor}, {s.start}) "
                                f"before {s.name}")
            cursor = max(cursor, s.stop)
        if cursor != self.obs_dim:
            problems.append(f"the table covers {cursor} floats, obs_dim is "
                            f"{self.obs_dim}")
        names = [s.name for s in self.slots]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            problems.append(f"repeated slot name(s): {', '.join(dupes)}")
        return tuple(problems)

    def slot(self, name: str) -> Slot:
        """The slot called `name`. Raises `KeyError` naming what there is.

        For a reader that wants one term out of an observation vector
        (`obs[c.slot("joint_pos_rel").start : ...]`) without re-deriving the
        offsets the body already published.
        """
        for s in self.slots:
            if s.name == name:
                return s
        raise KeyError(f"{self.id} has no slot {name!r} — has "
                       f"{', '.join(s.name for s in self.slots)}")

    # -------------------------------------------------------- the comparison

    def matches(self, other: Any) -> bool:
        """Same contract? The id, and only the id.

        The id carries the version, so two contracts that share one must
        agree on everything else — which is a property, not a hope:
        `tests/test_policy_contract.py` pins each body's id against its own
        dims and its env's real layout, so an id that stayed still while the
        layout moved fails there rather than here.

        Accepts a `PolicyContract`, anything with an `.id`, or the bare id
        string, because the callers that have only a string (a `record.json`
        field, an HTTP query, a log line) are the ones a width guard grows
        out of.
        """
        if other is None:
            return False
        other_id = other if isinstance(other, str) else getattr(other, "id", None)
        return bool(other_id) and str(other_id) == self.id

    def describe(self) -> str:
        """One line, for `export-walk`'s output and for an error message.

        `export-walk` prints it so the deployment caveat leaves the building
        with the file: the person who runs the exporter is the person who is
        about to hand the .onnx to someone, and this is the last moment the
        harness can say what it is.
        """
        line = (f"{self.id}: obs[1,{self.obs_dim}] -> actions[1,{self.act_dim}] "
                f"at {self.rate_hz:g} Hz on the {self.robot}")
        return line + (f" — {self.deploy}" if self.deploy else "")

    def layout(self) -> str:
        """The slot table as one readable line: `name[start:stop]` in order."""
        return "  ".join(f"{s.name}[{s.start}:{s.stop}]"
                         for s in sorted(self.slots, key=lambda s: s.start))

    # ------------------------------------------------------------ the record

    def as_dict(self) -> dict[str, Any]:
        """JSON-able, for `run.json` and for the ONNX metadata blob."""
        return {"id": self.id, "robot": self.robot,
                "obs_dim": self.obs_dim, "act_dim": self.act_dim,
                "rate_hz": self.rate_hz,
                "slots": [[s.name, s.start, s.stop] for s in self.slots],
                "deploy": self.deploy}

    @classmethod
    def from_dict(cls, data: Any) -> PolicyContract:
        """The inverse of `as_dict`. Raises on anything it cannot read.

        Loud here, quiet in `from_onnx` / `recorded`: a caller that HAS a
        dict and asks for a contract has made a claim about that dict, while
        a reader scraping a file is asking a question. The scrapers catch.
        """
        if not isinstance(data, dict):
            raise ValueError(f"not a contract record: {type(data).__name__}")
        return cls(
            id=str(data["id"]), robot=str(data["robot"]),
            obs_dim=int(data["obs_dim"]), act_dim=int(data["act_dim"]),
            rate_hz=float(data["rate_hz"]),
            slots=tuple(Slot(str(s[0]), int(s[1]), int(s[2]))
                        for s in (data.get("slots") or ())),
            deploy=str(data.get("deploy") or ""))

    # ------------------------------------------------------ the ONNX metadata

    def write_onnx_metadata(self, path: Path | str) -> Path:
        """Stamp this contract into `path`'s `metadata_props`, in place.

        `metadata_props` is ONNX's own key/value side-channel (§7.2 item 2 of
        the roadmap: standards over invention) — the runtime ignores it, every
        ONNX reader can print it, and it is the only part of the file that can
        say something the graph cannot. So a policy downloaded from the Hub
        lands on the right body in someone else's lab, with no run directory
        in sight.

        The graph is not touched: this loads the ModelProto, edits two string
        entries and writes the same proto back. `export_onnx.export` runs its
        onnxruntime cross-check AFTER this call, so every export proves that
        for itself, and `tests/test_policy_contract.py` compares the outputs
        either side of the stamp.

        Entries with these keys are REPLACED, not appended: re-exporting a
        run must not leave two contracts in one file for a reader to choose
        between, and `onnx.helper.set_model_props` is not used because it
        clears every other property with them.
        """
        import onnx

        path = Path(path)
        model = onnx.load(str(path))
        keep = [p for p in model.metadata_props if p.key not in (KEY_ID, KEY_JSON)]
        del model.metadata_props[:]
        for prop in keep:
            model.metadata_props.add().CopyFrom(prop)
        for key, value in ((KEY_ID, self.id),
                           (KEY_JSON, json.dumps(self.as_dict(), sort_keys=True))):
            entry = model.metadata_props.add()
            entry.key, entry.value = key, value
        onnx.save(model, str(path))
        return path

    @classmethod
    def from_onnx(cls, path: Path | str) -> PolicyContract | None:
        """The contract stamped in `path`, or None if it has none.

        None rather than an exception, including for a file that carries a
        metadata block which will not parse. This is a LABEL, and the rungs
        below it in `resolve()` answer exactly as well as they did before
        this module existed — while a raise here would land inside the lab's
        50 Hz loop, which is the failure this repo has paid for repeatedly
        (a stopped loop streams nothing and the viewer goes blank for every
        duck, not just the bad one).
        """
        try:
            import onnx

            model = onnx.load(str(path), load_external_data=False)
            for prop in model.metadata_props:
                if prop.key == KEY_JSON:
                    return cls.from_dict(json.loads(prop.value))
        except Exception:
            return None
        return None


# --------------------------------------------------------------- declaring one


def declare(body: Any, *, id: str, rate_hz: float,
            slots: Sequence[Any], deploy: str) -> PolicyContract:
    """A body's own contract, with the dims taken FROM the body.

    Every `Body.contract()` is built through here, so `obs_dim` and `act_dim`
    cannot drift from `body.obs_dim` / `body.num_actions` — they are not
    copied, they are read. That matters because those two numbers are what
    the exporter shapes the graph from, what the conformance suite checks and
    what the lab has refused a wrong policy by: a contract that disagreed
    with its own body would be a third opinion in a place that needs one.

    The slot table is then validated against that width by
    `PolicyContract.__post_init__`, so a layout that does not add up to the
    body's own `obs_dim` is a CONSTRUCTION error — raised the first time
    anything asks the body for its contract (the registry does, on every
    `fetch-robot` listing, and `tests/test_policy_contract.py` does for every
    registered body), not a mislabelled file discovered later.
    """
    return PolicyContract(
        id=id, robot=str(body.id), obs_dim=int(body.obs_dim),
        act_dim=int(body.num_actions), rate_hz=rate_hz,
        slots=tuple(slots), deploy=deploy)


# --------------------------------------------------------------- reading one


def _run_dir(path: Path) -> Path:
    """The directory a policy path belongs to (`viz_server.policy_robot`'s
    rule, which has always accepted either shape)."""
    return path if path.is_dir() else path.parent


def _run_json(run_dir: Path) -> dict[str, Any]:
    """`run.json`, or {} when there is none or it will not parse."""
    try:
        data = json.loads((run_dir / "run.json").read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def from_run_json(run_dir: Path | str) -> PolicyContract | None:
    """The contract `run_dir/run.json` RECORDS, or None.

    Rung 2 of `resolve()` on its own, and the run's own account of itself —
    written by `train.py` / `distill.py` from the body, before the first
    step. `export_onnx.export_contract` refuses a disagreement with THIS and
    not with `recorded()`, because a `policy.onnx` sitting in the same
    directory is an artefact of an earlier export rather than the run's
    record: believing it would refuse a legitimate re-export the day a body
    deliberately bumps its contract's `v`.

    Quiet on a malformed record (None, and the next rung answers) for the
    reason `from_onnx` is — see there.
    """
    data = _run_json(Path(run_dir)).get(KEY_JSON)
    if not data:
        return None
    try:
        return PolicyContract.from_dict(data)
    except Exception:
        return None


def recorded(run_dir_or_onnx: Path | str) -> PolicyContract | None:
    """The contract this policy DECLARES about itself, or None.

    The self-describing rungs of `resolve()` only — the ONNX's own metadata
    and `run.json`'s `"contract"` — never the two fallbacks that infer a
    contract from a body NAME or from the duck being the only body there used
    to be.

    Separate from `resolve()` because the difference decides whether a
    disagreement is an error. `eval_onnx` refuses a `--robot` that contradicts
    a contract the file RECORDED (the file cannot be wrong about itself, and
    the flag would build the wrong env); it still honours the flag for a
    policy that says nothing, because that is what `--robot` was added for —
    an old .onnx, moved away from its run directory, whose body only its
    owner knows.

    Directory before file, or file before directory, exactly as `resolve()`
    orders them; see there.
    """
    path = Path(run_dir_or_onnx)
    if path.is_dir():
        declared = from_run_json(path)
        if declared is not None:
            return declared
        onnx_path = path / "policy.onnx"
        return PolicyContract.from_onnx(onnx_path) if onnx_path.is_file() else None
    stamped = PolicyContract.from_onnx(path)
    return stamped if stamped is not None else from_run_json(path.parent)


def resolve(run_dir_or_onnx: Path | str) -> PolicyContract:
    """The contract a policy speaks. The one function a reader should call.

    Four rungs, most specific about THE THING YOU NAMED first:

    1. **the ONNX's own `metadata_props`** — the only rung that survives the
       file being copied out of its run directory, and the only one written
       from the body itself at export time. A file cannot be wrong about
       itself unless somebody rewrote it; its surroundings can be wrong about
       it the moment it is moved, and a `policy.onnx` dropped into another
       run's directory would otherwise inherit that run's identity.
    2. **`run.json`'s `"contract"`** — the run recorded what it trained
       against. This is the rung that covers a `policy.onnx` exported before
       the metadata existed but re-exported since, and any checkpoint ONNX
       (`select-run`) beside a run that knows its own contract.
    3. **`run.json`'s `"robot"`, mapped through the registry** — every run
       trained before this module recorded only a name. Mapping it to that
       body's contract TODAY is exactly what `export_onnx.run_robot` plus
       `spec.get` did, so an old run answers as it always has.
    4. **the duck** — a run with nothing at all has always been a duck.
       `run_robot` has returned `"microduck"` for a missing or unreadable
       `run.json` since it was written, because when those files were made the
       duck was the only body there was. Changing that answer would relabel
       every trick run in the palette (they carry `behavior.json`, not
       `run.json`), so the fallback stays.

    **A directory is asked in the other order** — `run.json`, then
    `policy.onnx` — for two reasons. Naming the directory is asking what the
    RUN drives, and the run's own record is the direct answer rather than an
    inference from one artefact inside it. And the lab resolves every run in
    the palette on a timer: `run.json` is a few hundred bytes, while parsing a
    policy's protobuf per run per poll is a cost nobody asked for. The two
    disagree only if a foreign .onnx was copied into a run directory, which
    `export()`'s own contract check and the lab's assign guard both catch.

    Raises `KeyError` (naming the fetch command) when rung 3 names a body
    this install cannot load — the honest answer, and the one `spec.get`
    already gives. Silently calling it a duck is the cross this file exists
    to prevent.
    """
    declared = recorded(run_dir_or_onnx)
    if declared is not None:
        return declared
    from . import registry

    robot = _run_json(_run_dir(Path(run_dir_or_onnx))).get("robot")
    return registry.get(str(robot) if robot else DEFAULT_ROBOT).contract()


def robot_of(run_dir_or_onnx: Path | str) -> str:
    """Which body drives this policy — `resolve(...).robot`.

    The narrow question `export_onnx.run_robot` has always answered, kept as
    a name because a caller that only needs the body id should not have to
    know that a contract is what answers it.
    """
    return resolve(run_dir_or_onnx).robot


__all__ = ["DEFAULT_ROBOT", "KEY_ID", "KEY_JSON", "PolicyContract", "Slot",
           "declare", "from_run_json", "recorded", "resolve", "robot_of"]
