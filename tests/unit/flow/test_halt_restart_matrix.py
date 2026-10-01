# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Stop, then resume, is equivalent to an uninterrupted run.

Each case runs one generated flow shape three ways:

1. Uninterrupted — the baseline, supplied by a pure-Python model.
2. Stopped at one leaf: an ambient halt, a simulated crash (the process
   dies; the store sees no further write), or an exception.
3. ``run(resume="latest")`` against the same store.

Dimensions: shape (chain, iterate, map, subflow, branch, ``state=``
scope, up to three levels deep; leaves that are plain verbs or Loop
turns), the leaf where the run stops and whether before or after its
work (before a Loop turn's work is mid-turn, where the halt pauses the
turn), the checkpoint policy (none,
``on_iterate``, ``on_map_item``, or ``ctx.checkpoint()`` in every leaf),
sequential or parallel maps, and the store (in-memory; a subset against
the file store).

Leaves are not state-driven: a leaf does its work every time it runs,
so a leaf that runs again shows up in the executed list. They honour the
halt contract: a leaf halted before its work raises ``Interrupted``, one
halted after its work returns. The resume point is the newest commit
with state — the commit ``resume="latest"`` checks out. A run that
raises writes none: its resume point is its last save.

Some cases stop twice: the resume stops again (a halt or a crash at the
first or second leaf it runs) and a final resume finishes. Each resume
is checked against its own resume point.

Properties checked per case:

- The stopped run ends as its stop dictates: a halt returns, a crash or
  an exception raises. Nothing is written after it returns or raises.
- The resume point holds only work the runs so far did, with the
  baseline's values.
- A halt writes a halt checkpoint — unless it was set after the last
  leaf's work, when the run completes — and the resume point holds every
  leaf the stopped run completed; with a checkpoint in every leaf, so
  does a crash's or an exception's resume point.
- Resume ends with the baseline's result, ``done`` map and scoped-map
  merges, marks the history complete, and executes every leaf the resume
  point lacks, in the baseline's order, once. Of the leaves the resume
  point holds, at most one runs again: the one that was running when the
  checkpoint was taken — and the leaf that crashed or raised. A Loop turn
  the halt paused continues from its saved half; it never starts over.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest

from llm_gent.flow import Context, Flow, FlowFactory, History, Interrupted, Loop, Role, verb
from llm_gent.flow.checkpoint import CheckpointStore
from llm_gent.flow.stores import InMemoryCheckpointStore, JsonFileCheckpointStore

from .conftest import make_test_logger


pytestmark = pytest.mark.unit

LG = make_test_logger()
FLOW_NAME = "matrix"
RUN_INPUT = 1

Stop = Literal["halt", "crash", "exception"]
Policy = Literal["none", "on_iterate", "on_map_item", "leaf", "named"]


class SimulatedCrash(BaseException):
    """The process died: not an ``Exception``, so nothing handles it on the way out."""


class LeafError(Exception):
    """An ordinary exception raised by a leaf."""


# --- Shapes -----------------------------------------------------------------


@dataclass(frozen=True)
class Leaf:
    """A verb computing ``3 * x + c``; ``name`` is unique within a shape."""

    name: str
    c: int


@dataclass(frozen=True)
class Turn(Leaf):
    """A leaf whose work is a Loop turn: half of it, then (unless paused there) the rest."""


@dataclass(frozen=True)
class BareTurn(Turn):
    """A :class:`Turn` whose Loop has no ConversationFactory: a paused turn starts over."""


@dataclass(frozen=True)
class Seq:
    """A chain: the top-level flow, or a subflow step inside it."""

    children: tuple[Node, ...]


@dataclass(frozen=True)
class Iter:
    """``.iterate(body, max_iters=n)``; with ``until``, also stops on :func:`_until`."""

    body: Node
    n: int
    until: bool = False


def _until(result: int) -> bool:
    """The state-driven stop of an ``Iter(until=True)``: a pass result divisible by 4."""
    return result % 4 == 0


FanMode = Literal["strict", "lenient", "scoped"]


@dataclass(frozen=True)
class Fan:
    """``.map(body)`` over ``[10 * x + i for i in range(width)]``, summed.

    ``lenient``: ``strict=False`` with a guard that skips the last item.
    ``scoped``: each item runs under its own ``state=`` (an empty ``done``)
    and merges back, appending the keys it merged to ``merges``, so a
    merge applied twice shows.
    """

    body: Node
    width: int
    mode: FanMode = "strict"


@dataclass(frozen=True)
class Branch:
    """``.branch``: ``then`` for an even input, ``else_`` for an odd one."""

    then: Node
    else_: Node


@dataclass(frozen=True)
class Scope:
    """A subflow run under ``state=`` (a copy of ``done``), merged back after."""

    body: Node


Node = Leaf | Seq | Iter | Fan | Branch | Scope

_L = Leaf("", 0)
_T = Turn("", 0)
_B = BareTurn("", 0)


def _label(node: Node, counter: list[int]) -> Node:
    """Return ``node`` with its leaves named ``l1, l2, ...`` in depth-first order."""
    if isinstance(node, Leaf):
        counter[0] += 1
        return type(node)(f"l{counter[0]}", counter[0])
    if isinstance(node, Seq):
        return Seq(tuple(_label(child, counter) for child in node.children))
    if isinstance(node, Iter):
        return Iter(_label(node.body, counter), node.n, node.until)
    if isinstance(node, Fan):
        return Fan(_label(node.body, counter), node.width, node.mode)
    if isinstance(node, Branch):
        return Branch(_label(node.then, counter), _label(node.else_, counter))
    return Scope(_label(node.body, counter))


_INNER: dict[str, Node] = {
    "iter": Iter(_L, 2),
    "fan": Fan(_L, 2),
    "sub": Seq((_L, _L)),
    "branch": Branch(_L, _L),
    "scope": Scope(Seq((_L, _L))),
    "iter(sub)": Iter(Seq((_L, _L)), 2),
    "iter(fan)": Iter(Fan(_L, 2), 2),
    "iter(branch)": Iter(Branch(_L, _L), 2),
    "iter(scope)": Iter(Scope(_L), 2),
    "fan(iter)": Fan(Iter(_L, 2), 2),
    "fan(scope)": Fan(Scope(_L), 2),
    "scope(fan)": Scope(Fan(_L, 2)),
    "sub(fan,iter)": Seq((Fan(_L, 2), Iter(_L, 2))),
    "iter(fan(sub))": Iter(Fan(Seq((_L, _L)), 2), 2),
    "fan(iter(fan))": Fan(Iter(Fan(_L, 2), 2), 2),
    "sub(iter(branch))": Seq((Iter(Branch(_L, _L), 2), _L)),
    "turn": _T,
    "sub(turn,turn)": Seq((_T, _T)),
    "iter(turn)": Iter(_T, 2),
    "fan(turn)": Fan(_T, 2),
    "branch(turn)": Branch(_T, _T),
    "scope(turn)": Scope(Seq((_T, _L))),
    "iter(fan(turn))": Iter(Fan(_T, 2), 2),
    "until": Iter(_L, 4, until=True),
    "until(fan)": Iter(Fan(_L, 2), 4, until=True),
    "until(sub)": Iter(Seq((_L, _L)), 4, until=True),
    "lenient": Fan(_L, 3, "lenient"),
    "lenient(iter)": Fan(Iter(_L, 2), 3, "lenient"),
    "lenient(turn)": Fan(_T, 3, "lenient"),
    "scoped": Fan(_L, 2, "scoped"),
    "scoped(sub)": Fan(Seq((_L, _L)), 2, "scoped"),
    "scoped(scoped)": Fan(Fan(_L, 2, "scoped"), 2, "scoped"),
    "iter(scoped)": Iter(Fan(_L, 2, "scoped"), 2),
    "scope(scoped)": Scope(Fan(_L, 2, "scoped")),
    "scoped(turn)": Fan(_T, 2, "scoped"),
    "bare": _B,
    "iter(bare)": Iter(_B, 2),
    "fan(bare)": Fan(_B, 2),
}

SHAPES: dict[str, Seq] = {}
for _key, _inner in _INNER.items():
    SHAPES[f"leaf>{_key}"] = Seq((_label(_L, [0]), _label(_inner, [1])))
    SHAPES[f"{_key}>leaf"] = Seq((_label(_inner, [0]), Leaf("l99", 99)))


def _has(node: Node, kind: type, mode: FanMode | None = None) -> bool:
    """True when ``node`` contains a node of ``kind`` (a ``Fan`` in ``mode``, when given)."""
    if isinstance(node, kind) and (mode is None or getattr(node, "mode", None) == mode):
        return True
    if isinstance(node, Seq):
        return any(_has(child, kind, mode) for child in node.children)
    if isinstance(node, Branch):
        return _has(node.then, kind, mode) or _has(node.else_, kind, mode)
    if isinstance(node, Iter | Fan | Scope):
        return _has(node.body, kind, mode)
    return False


def _has_scope(node: Node) -> bool:
    """True when ``node`` runs anything under a ``state=`` scope: a ``Scope`` or a scoped map."""
    return _has(node, Scope) or _has(node, Fan, "scoped")


# --- Reference model --------------------------------------------------------


def _model_node(node: Node, x: int, out: dict[str, int], merges: list[str]) -> int:
    """Evaluate ``node`` on ``x``, recording each leaf execution in ``out`` in order.

    ``merges`` collects the keys each scoped map item merges back.
    """
    if isinstance(node, Leaf):
        key = f"{node.name}:{x}"
        assert key not in out, f"model keys collide: {key}"
        out[key] = 3 * x + node.c
        return out[key]
    if isinstance(node, Seq):
        for child in node.children:
            x = _model_node(child, x, out, merges)
        return x
    if isinstance(node, Iter):
        for _ in range(node.n):
            x = _model_node(node.body, x, out, merges)
            if node.until and _until(x):
                break
        return x
    if isinstance(node, Fan):
        return _model_fan(node, x, out, merges)
    if isinstance(node, Branch):
        return _model_node(node.then if x % 2 == 0 else node.else_, x, out, merges)
    return _model_node(node.body, x, out, merges)


def _model_fan(node: Fan, x: int, out: dict[str, int], merges: list[str]) -> int:
    """A map: the lenient guard skips the last item; a scoped item merges its keys."""
    width = node.width - 1 if node.mode == "lenient" else node.width
    total = 0
    for i in range(width):
        before = set(out)
        total += _model_node(node.body, 10 * x + i, out, merges)
        if node.mode == "scoped":
            merges.extend(sorted(set(out) - before))
    return total


def model(shape: Seq) -> tuple[int, dict[str, int], list[str]]:
    """Return the uninterrupted run's result, its ``done`` map in execution order, and merges."""
    out: dict[str, int] = {}
    merges: list[str] = []
    return _model_node(shape, RUN_INPUT, out, merges), out, sorted(merges)


# --- Store and probe --------------------------------------------------------

_WRITES = frozenset({"bind_flow_id", "put_object", "set_ref", "gc_history"})


class CrashableStore:
    """Delegates to a store and counts writes; once dead, every write raises.

    A crashed process writes nothing more; raising :class:`SimulatedCrash`
    from each write also stops any task of the dead run that is still
    scheduled.
    """

    def __init__(self, inner: CheckpointStore) -> None:
        self._inner = inner
        self.dead = False
        self.writes = 0

    @property
    def retention(self) -> Any:
        return self._inner.retention

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name not in _WRITES:
            return attr

        def write(*args: Any, **kwargs: Any) -> Any:
            if self.dead:
                raise SimulatedCrash(f"{name} after the crash")
            self.writes += 1
            return attr(*args, **kwargs)

        return write


@dataclass
class Probe:
    """Per-run leaf instrumentation: where the run stops, what executed."""

    policy: Policy = "none"
    stop: Stop | None = None
    stop_at: int = 0
    mode: str = "before"
    halt: asyncio.Event | None = None
    store: CrashableStore | None = None
    crashed: bool = False
    stopped: str | None = None
    entered: int = 0
    executed: list[str] = field(default_factory=list)
    checkpointed: list[str] = field(default_factory=list)
    mid_turn: str | None = None
    turns_started: list[str] = field(default_factory=list)
    turns_paused: list[str] = field(default_factory=list)
    turns_resumed: list[str] = field(default_factory=list)

    def stop_here(self) -> None:
        """Stop the run at the current leaf.

        A halt is set and the leaf carries on: before its work it raises
        :class:`Interrupted` (it stops without doing it), after its work it
        returns normally (it completed). A crash or an exception raises.
        """
        if self.stop == "halt":
            assert self.halt is not None
            self.halt.set()
            return
        if self.stop == "crash":
            assert self.store is not None
            self.crashed = self.store.dead = True
            raise SimulatedCrash("process died")
        raise LeafError("leaf failed")


# --- Framework flows --------------------------------------------------------


def _leaf_verb(leaf: Leaf, probe: Probe) -> Any:
    """Return the verb for ``leaf``, reporting to ``probe``.

    It does its work every time it runs (not state-driven) and honours the
    halt contract: a leaf that sees the halt before its work raises
    :class:`Interrupted`.
    """

    async def body(ctx: Context[dict[str, Any]], x: Any = None) -> Any:
        await asyncio.sleep(0)  # lets parallel map items interleave
        if probe.crashed:
            raise SimulatedCrash("process already dead")
        if ctx.halt is not None and ctx.halt.is_set():
            raise Interrupted()
        key = f"{leaf.name}:{x}"
        done = ctx.state.data.setdefault("done", {})
        probe.entered += 1
        fire = probe.stop is not None and probe.entered == probe.stop_at
        if fire:
            probe.stopped = key
        if fire and probe.mode == "before":
            probe.stop_here()
            raise Interrupted()  # a halt: stops before its work
        done[key] = 3 * x + leaf.c
        probe.executed.append(key)
        await _save(ctx, probe, key)
        if fire:
            probe.stop_here()  # sets the halt, or raises for a crash / an exception
        return done[key]

    body.__name__ = body.__qualname__ = leaf.name
    return verb(body)


async def _save(ctx: Context[Any], probe: Probe, key: str) -> None:
    """A leaf's own save after its work: ``leaf`` saves, ``named`` saves as a checkpoint ``key``."""
    if probe.policy == "leaf":
        await ctx.checkpoint()
        probe.checkpointed.append(key)
    elif probe.policy == "named":
        await ctx.checkpoint(key)


ROLE = Role(name="matrix", backend="openai", model="none")


@dataclass
class _Conv:
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"messages": list(self.messages)}


class _ConvFactory:
    def create(self) -> _Conv:
        return _Conv()

    def create_from_state(self, state: dict[str, Any]) -> _Conv:
        return _Conv(list(state["messages"]))


@dataclass
class _TurnResult:
    paused: bool


class _TurnSAIA:
    """A SAIA turn in two halves; it pauses between them when the halt is set.

    A fresh turn does the first half, then pauses if it is where the run
    stops mid-turn or if the halt is already set (as a real backend does
    through ``abort_signal``). A resumed turn must arrive with the saved
    first half and does only the second. A turn without a conversation (a
    :class:`BareTurn`) cannot be captured: its pause is not recorded as
    one that must resume.
    """

    def __init__(self, probe: Probe) -> None:
        self._probe = probe

    async def complete(self, task: str, **kwargs: Any) -> _TurnResult:
        conv: _Conv | None = kwargs["conversation"]
        probe = self._probe
        if kwargs.get("resume", False):
            assert conv is not None and conv.messages == [f"half:{task}"], f"resumed {task}"
            probe.turns_resumed.append(task)
        else:
            probe.turns_started.append(task)
            if conv is not None:
                conv.messages.append(f"half:{task}")
            if probe.mid_turn == task:
                probe.stop_here()  # sets the halt, or raises for a crash / an exception
            await asyncio.sleep(0)  # lets parallel map items interleave
            halt = kwargs.get("abort_signal")
            if halt is not None and halt.is_set():
                if conv is not None:
                    probe.turns_paused.append(task)
                return _TurnResult(paused=True)
        if conv is not None:
            conv.messages.append(f"rest:{task}")
        return _TurnResult(paused=False)


def _turn_verb(leaf: Turn, probe: Probe) -> Any:
    """Return the verb for a :class:`Turn` leaf: its work is one Loop call.

    A stop "before" its work lands mid-turn, between the turn's halves; a
    paused turn leaves the step interrupted with nothing recorded. A
    :class:`BareTurn`'s Loop has no ConversationFactory: its paused turn is
    not captured, and its step's rerun starts the turn over.
    """
    factory = None if isinstance(leaf, BareTurn) else _ConvFactory()
    loop = Loop(ROLE, saia=_TurnSAIA(probe), conversation_factory=factory)

    async def body(ctx: Context[dict[str, Any]], x: Any = None) -> Any:
        await asyncio.sleep(0)
        if probe.crashed:
            raise SimulatedCrash("process already dead")
        if ctx.halt is not None and ctx.halt.is_set():
            raise Interrupted()
        key = f"{leaf.name}:{x}"
        probe.entered += 1
        fire = probe.stop is not None and probe.entered == probe.stop_at
        if fire:
            probe.stopped = key
            if probe.mode == "before":
                probe.mid_turn = key
        if (await loop(ctx, key)).paused:
            return None
        done = ctx.state.data.setdefault("done", {})
        done[key] = 3 * x + leaf.c
        probe.executed.append(key)
        await _save(ctx, probe, key)
        if fire and probe.mode == "after":
            probe.stop_here()
        return done[key]

    body.__name__ = body.__qualname__ = leaf.name
    return verb(body)


def _add(node: Node, flow: Flow, probe: Probe, parallel: bool) -> Flow:
    """Append ``node`` to ``flow`` as one chain step."""
    if isinstance(node, Turn):
        return flow.call(_turn_verb(node, probe))
    if isinstance(node, Leaf):
        return flow.call(_leaf_verb(node, probe))
    if isinstance(node, Seq | Scope):
        sub = FlowFactory(LG).create()
        for child in node.children if isinstance(node, Seq) else (node.body,):
            _add(child, sub, probe, parallel)
        if isinstance(node, Seq):
            return flow.call(sub)
        return flow.call(sub, state=lambda p: {"done": dict(p.get("done", {}))}, merge=_merge_scope)
    if isinstance(node, Iter):
        until = (lambda r, _ctx: _until(r)) if node.until else None
        return flow.iterate(
            lambda b: _add(node.body, b, probe, parallel), max_iters=node.n, until=until
        )
    if isinstance(node, Branch):
        return flow.branch(
            when=lambda prev, _ctx: prev % 2 == 0,
            then=lambda b: _add(node.then, b, probe, parallel),
            else_=lambda b: _add(node.else_, b, probe, parallel),
        )
    return _add_fan(node, flow, probe, parallel)


def _merge_scope(parent: dict[str, Any], child: dict[str, Any]) -> None:
    """Merge a ``Scope`` back: its ``done``, and the merges scoped maps inside it recorded."""
    parent.setdefault("done", {}).update(child["done"])
    parent.setdefault("merges", []).extend(child.get("merges", []))


def _merge_item(parent: dict[str, Any], child: dict[str, Any]) -> None:
    """Merge a scoped map item back, recording the keys it merged."""
    _merge_scope(parent, child)
    parent["merges"].extend(sorted(child["done"]))


def _add_fan(node: Fan, flow: Flow, probe: Probe, parallel: bool) -> Flow:
    """Append a ``.map`` step for ``node`` in its mode."""
    width = node.width
    scoped: dict[str, Any] = (
        {"state": lambda _p: {"done": {}}, "merge": _merge_item} if node.mode == "scoped" else {}
    )
    flow = flow.map(
        lambda b: _add(node.body, b, probe, parallel),
        items=lambda prev, _ctx: [10 * prev + i for i in range(width)],
        aggregate=lambda results: sum(r for r in results if isinstance(r, int)),
        strict=node.mode != "lenient",
        max_concurrency=None if parallel else 1,
        **scoped,
    )
    if node.mode == "lenient":
        flow = flow.guard(lambda item, _ctx: item % 10 != width - 1)
    return flow


def _flow(
    shape: Seq, probe: Probe, store: Any, *, parallel: bool, halt: asyncio.Event | None = None
) -> Flow:
    """Build the top-level flow for ``shape`` under ``probe.policy``."""
    flow = FlowFactory(LG).create(state={}).with_checkpointer(store, FLOW_NAME)
    if probe.policy in ("on_iterate", "on_map_item"):
        flow = flow.with_checkpoint_policy(**{probe.policy: True})
    if halt is not None:
        flow = flow.with_halt(halt)
    for step in shape.children:
        _add(step, flow, probe, parallel)
    return flow


async def _root_done(history: History, commit: Any) -> dict[str, int] | None:
    """The ``done`` map in ``commit``'s root scope; ``None`` when it holds no state."""
    snapshot = await history.snapshot(commit)
    return dict(snapshot.root.get("done", {})) if snapshot.has_state else None


async def _snapshot_done(history: History, commit: Any) -> dict[str, int] | None:
    """Every leaf ``commit``'s snapshot records: the root's and each saved scope's ``done``."""
    snapshot = await history.snapshot(commit)
    if not snapshot.has_state:
        return None
    done = dict(snapshot.root.get("done", {}))
    for scope in snapshot.scopes.values():
        done.update(scope.get("done", {}))
    return done


async def _resume_point_done(history: History) -> dict[str, int]:
    """The ``done`` map resume continues from: the newest commit with state."""
    async for commit in history.commits():
        done = await _snapshot_done(history, commit)
        if done is not None:
            return done
    return {}


async def _settle() -> None:
    """Let every task the stopped run left scheduled finish."""
    for _ in range(20):
        await asyncio.sleep(0)


# --- Cases ------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One point of the matrix."""

    shape: str
    stop: Stop
    stop_at: int
    mode: str
    policy: Policy
    parallel: bool
    store: str
    # A second stop, while the first resume runs: (stop, leaf), after its work.
    second: tuple[Stop, int] | None = None

    @property
    def id(self) -> str:
        run = "par" if self.parallel else "seq"
        then = "" if self.second is None else f"-then-{self.second[0]}{self.second[1]}"
        return (
            f"{self.shape}-{self.stop}-at{self.stop_at}-{self.mode}"
            f"-{self.policy}-{run}-{self.store}{then}"
        )

    def stops(self) -> list[tuple[Stop, int, str]]:
        """Each stopped run's (stop, leaf, mode), in order."""
        first = [(self.stop, self.stop_at, self.mode)]
        return first if self.second is None else [*first, (*self.second, "after")]


def _stop_kinds(shape: Seq) -> tuple[Stop, ...]:
    """Halt, crash, exception — without exception where a ``strict=False`` map absorbs it."""
    if _has(shape, Fan, "lenient"):
        return ("halt", "crash")
    return ("halt", "crash", "exception")


def _memory_cases() -> Iterator[Case]:
    """Every shape × stop point × stop × policy × map mode, in memory."""
    for name, shape in SHAPES.items():
        leaves = len(model(shape)[1])
        for parallel in (False, True) if _has(shape, Fan) else (False,):
            for policy in ("none", "on_iterate", "on_map_item", "leaf"):
                for stop in _stop_kinds(shape):
                    for stop_at in range(1, leaves + 1):
                        for mode in ("before", "after"):
                            yield Case(name, stop, stop_at, mode, policy, parallel, "mem")


SECOND_STOPS: tuple[tuple[Stop, int], ...] = (("halt", 1), ("halt", 2), ("crash", 1))
"""Where a resumed run stops again: the first or second leaf it runs."""


def _cycle_cases() -> Iterator[Case]:
    """Two stops: a halt or crash at every leaf, then another while the resume runs."""
    for name, shape in SHAPES.items():
        leaves = len(model(shape)[1])
        for parallel in (False, True) if _has(shape, Fan) else (False,):
            for policy in ("none", "leaf"):
                for stop in ("halt", "crash"):
                    for stop_at in range(1, leaves + 1):
                        for second in SECOND_STOPS:
                            yield Case(
                                name, stop, stop_at, "after", policy, parallel, "mem", second
                            )


def _file_cases() -> Iterator[Case]:
    """A slice against the file store: halt and crash, no policy or a checkpoint per leaf."""
    for name, shape in SHAPES.items():
        leaves = len(model(shape)[1])
        for policy in ("none", "leaf"):
            for stop in ("halt", "crash"):
                for stop_at in range(1, leaves + 1):
                    yield Case(name, stop, stop_at, "after", policy, False, "file")


def _cases() -> list[Case]:
    return [*_memory_cases(), *_cycle_cases(), *_file_cases()]


def _params() -> Iterator[Any]:
    for case in _cases():
        yield pytest.param(case, id=case.id)


# --- Tests ------------------------------------------------------------------


def _store(kind: str, tmp_path: Path) -> CheckpointStore:
    if kind == "mem":
        return InMemoryCheckpointStore()
    return JsonFileCheckpointStore(LG, tmp_path / "cp")


@pytest.mark.parametrize("parallel", [False, True], ids=["seq", "par"])
@pytest.mark.parametrize("shape_name", list(SHAPES))
async def test_uninterrupted_run_matches_model(shape_name: str, parallel: bool) -> None:
    shape = SHAPES[shape_name]
    expected_result, baseline, merges = model(shape)
    store = InMemoryCheckpointStore()
    probe = Probe()

    assert await _flow(shape, probe, store, parallel=parallel).run(RUN_INPUT) == expected_result
    assert sorted(probe.executed) == sorted(baseline)
    if not parallel:
        assert probe.executed == list(baseline)
    await _check_finished(History(store, FLOW_NAME), baseline, merges)


_RAISED_BY: dict[Stop, type[BaseException]] = {"crash": SimulatedCrash, "exception": LeafError}


async def _run_stopping(
    shape: Seq,
    case: Case,
    stop: Stop,
    stop_at: int,
    mode: str,
    store: CrashableStore,
    *,
    resume: bool,
) -> Probe:
    """Run ``shape`` — resuming when ``resume`` — until it stops at leaf ``stop_at``.

    A stop point past the leaves the run reaches never fires: the run
    finishes. Checks the run ended as its stop dictates and wrote nothing
    after it ended.
    """
    halt = asyncio.Event()
    probe = Probe(policy=case.policy, stop=stop, stop_at=stop_at, mode=mode, halt=halt, store=store)
    flow = _flow(shape, probe, store, parallel=case.parallel, halt=halt)
    raised: BaseException | None = None
    try:
        await flow.run(RUN_INPUT, resume="latest" if resume else "off")
    except (SimulatedCrash, LeafError) as e:
        raised = e
    expected = _RAISED_BY.get(stop) if probe.stopped is not None else None
    if expected is None:
        assert raised is None, f"the run raised {raised!r}"
    else:
        assert isinstance(raised, expected), f"expected {expected.__name__}, got {raised!r}"
    writes = store.writes
    await _settle()
    assert store.writes == writes, "the stopped run wrote after run() ended"
    return probe


@dataclass
class _Point:
    """Where a stopped run left the history: what the next resume continues from."""

    done: dict[str, int]
    raised: set[str]
    paused: set[str]


async def _check_stopped(
    shape: Seq, stop: Stop, probe: Probe, history: History, prior: set[str]
) -> _Point | None:
    """Check the resume point a stopped run left; ``None`` when the run finished instead.

    ``prior`` is every leaf the runs so far executed.
    """
    _, baseline, merges = model(shape)
    captured = await _resume_point_done(history)
    assert captured.items() <= baseline.items(), "resume point holds work never done"
    assert set(captured) <= prior, "resume point holds work no run did"
    if await history.is_complete():
        # The stop never fired, or a halt came after the last leaf's work: every
        # step completed, so the run finished and there is nothing to resume.
        assert probe.stopped is None or stop == "halt"
        await _check_finished(history, baseline, merges)
        return None
    if stop == "halt":
        lost = [k for k in probe.executed if k not in captured]
        assert not lost, f"completed work missing from the halt checkpoint: {lost}"
        # A save point written after the halt checkpoint (a sibling map item
        # finishing its iteration) is kept: it carries more progress.
        outcomes = [commit.meta.outcome async for commit in history.commits()]
        assert "halted" in outcomes, f"no halt checkpoint; outcomes {outcomes}"
    # The leaf that crashed or raised did not complete its step: it runs again.
    raised = {probe.stopped} if stop != "halt" and probe.stopped else set()
    if probe.policy == "leaf":
        # A leaf that raised inside a state= scope takes its work down with the
        # scope: the block never merges it, and a later save point (a sibling
        # map item carrying on) no longer holds it. Resume runs that leaf again.
        discarded = raised if stop == "exception" and _has_scope(shape) else set()
        lost = [k for k in probe.checkpointed if k not in captured and k not in discarded]
        assert not lost, f"checkpointed work missing from the resume point: {lost}"
    return _Point(captured, raised, set(probe.turns_paused))


def _check_rerun(probe: Probe, point: _Point) -> None:
    """A resumed run reruns at most the step running at its resume point.

    A leaf that crashed or raised runs again too: its step did not
    complete. Other map items keep running after it, so a later save can
    hold its work with another step running. A Loop turn the halt paused
    resumes; it never starts over.
    """
    assert len(probe.executed) == len(set(probe.executed)), "a leaf ran twice in one run"
    again = [k for k in probe.executed if k in point.done]
    rerun = [k for k in again if k not in point.raised]
    assert len(rerun) <= 1, f"more than the interrupted step ran again: {again}"
    restarted = point.paused & set(probe.turns_started)
    assert not restarted, f"paused Loop turns started over instead of resuming: {restarted}"


async def _check_finished(history: History, baseline: dict[str, int], merges: list[str]) -> None:
    """The history is complete with the uninterrupted run's state: its leaves and merges."""
    assert await history.is_complete()
    head = await history.head()
    assert head is not None
    root = (await history.snapshot(head)).root
    assert root.get("done", {}) == baseline
    assert sorted(root.get("merges", [])) == merges, "a scoped map merged an item twice or never"


@pytest.mark.parametrize("case", list(_params()))
async def test_resume_after_stop_matches_uninterrupted_run(case: Case, tmp_path: Path) -> None:
    shape = SHAPES[case.shape]
    expected_result, baseline, merges = model(shape)
    inner = _store(case.store, tmp_path)
    history = History(inner, FLOW_NAME)

    prior: set[str] = set()
    point: _Point | None = None
    for stop, stop_at, mode in case.stops():
        store = CrashableStore(inner)
        probe = await _run_stopping(
            shape, case, stop, stop_at, mode, store, resume=point is not None
        )
        if point is not None:
            _check_rerun(probe, point)
        prior |= set(probe.executed)
        point = await _check_stopped(shape, stop, probe, history, prior)
        if point is None:
            return

    assert point is not None
    resumed = Probe(policy=case.policy)
    flow = _flow(shape, resumed, inner, parallel=case.parallel)
    assert await flow.run(RUN_INPUT, resume="latest") == expected_result
    await _check_finished(history, baseline, merges)
    _check_rerun(resumed, point)
    missing = [k for k in baseline if k not in point.done]
    assert [k for k in resumed.executed if k not in point.done] == missing or case.parallel
    assert set(resumed.executed) - set(point.done) == set(missing)


def _named_params() -> Iterator[Any]:
    """Every shape × every leaf, sequential: the leaf whose named checkpoint resume checks out."""
    for name, shape in SHAPES.items():
        for leaf in range(len(model(shape)[1])):
            yield pytest.param(name, leaf, id=f"{name}-leaf{leaf + 1}")


@pytest.mark.parametrize(("shape_name", "leaf"), list(_named_params()))
async def test_resume_from_a_named_checkpoint_matches_uninterrupted_run(
    shape_name: str, leaf: int
) -> None:
    """A finished run's history, checked out at any leaf's named checkpoint, finishes again.

    Every leaf takes a checkpoint named after itself, after its work.
    ``run(resume=<name>)`` moves ``HEAD`` back there and continues: the
    result and state are the uninterrupted ones, the leaves the
    checkpoint lacks run once, and of the ones it holds only the leaf
    that took it runs again (its step had not completed).
    """
    shape = SHAPES[shape_name]
    expected_result, baseline, merges = model(shape)
    store = InMemoryCheckpointStore()
    first = Probe(policy="named")
    assert await _flow(shape, first, store, parallel=False).run(RUN_INPUT) == expected_result

    name = first.executed[leaf]
    history = History(store, FLOW_NAME)
    commit = await history.checkpoint(name)
    assert commit is not None
    captured = await _snapshot_done(history, commit)
    assert captured is not None and name in captured

    resumed = Probe()
    flow = _flow(shape, resumed, store, parallel=False)
    assert await flow.run(RUN_INPUT, resume=name) == expected_result
    await _check_finished(history, baseline, merges)
    _check_rerun(resumed, _Point(captured, {name}, set()))
    assert set(resumed.executed) & set(captured) <= {name}, (
        "a completed leaf before the named checkpoint ran again"
    )
    missing = [k for k in baseline if k not in captured]
    assert [k for k in resumed.executed if k not in captured] == missing
