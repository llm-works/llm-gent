# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Stop, then resume, is equivalent to an uninterrupted run.

Each case runs one generated flow shape three ways:

1. Uninterrupted — the baseline, supplied by a pure-Python model.
2. Stopped at one leaf: an ambient halt, a simulated crash (the process
   dies; the store sees no further write), or an exception.
3. ``run(resume="latest")`` against the same store.

Dimensions: shape (chain, iterate, map, subflow, branch, ``state=``
scope, up to three levels deep), the leaf where the run stops and
whether before or after its work, the checkpoint policy (none,
``on_iterate``, ``on_map_item``, or ``ctx.checkpoint()`` in every leaf),
sequential or parallel maps, and the store (in-memory; a subset against
the file store).

Leaves are not state-driven: a leaf does its work every time it runs,
so a leaf that runs again shows up in the executed list. They honour the
halt contract: a leaf halted before its work raises ``Interrupted``, one
halted after its work returns. The resume point is the newest commit
with state that is not a failure record — the commit
``resume="latest"`` checks out.

Properties checked per case:

- The stopped run ends as its stop dictates: a halt returns, a crash or
  an exception raises. Nothing is written after it returns or raises.
- The resume point holds only work the stopped run did, with the
  baseline's values.
- A halt writes a halt checkpoint — unless it was set after the last
  leaf's work, when the run completes — and the resume point holds every
  leaf the stopped run completed; with a checkpoint in every leaf, so
  does a crash's or an exception's resume point.
- Resume ends with the baseline's result and ``done`` map, marks the
  history complete, and executes every leaf the resume point lacks, in
  the baseline's order, once. Of the leaves the resume point holds, at
  most one runs again: the one that was running when the checkpoint was
  taken.

Cases that fail because of a known defect are listed by id in
``halt_restart_known_defects.json`` and run as ``xfail(strict=True)``
with the defect as the reason: a fix turns them into failures until
their ids are removed.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest

from llm_gent.flow import Context, Flow, FlowFactory, History, Interrupted, verb
from llm_gent.flow.checkpoint import FAILED_NODE_PATH, CheckpointStore
from llm_gent.flow.stores import InMemoryCheckpointStore, JsonFileCheckpointStore

from .conftest import make_test_logger


pytestmark = pytest.mark.unit

LG = make_test_logger()
FLOW_NAME = "matrix"
RUN_INPUT = 1

Stop = Literal["halt", "crash", "exception"]
Policy = Literal["none", "on_iterate", "on_map_item", "leaf"]


class SimulatedCrash(BaseException):
    """The process died: not an ``Exception``, so no failure commit is written."""


class LeafError(Exception):
    """An ordinary exception raised by a leaf."""


# --- Shapes -----------------------------------------------------------------


@dataclass(frozen=True)
class Leaf:
    """A verb computing ``3 * x + c``; ``name`` is unique within a shape."""

    name: str
    c: int


@dataclass(frozen=True)
class Seq:
    """A chain: the top-level flow, or a subflow step inside it."""

    children: tuple[Node, ...]


@dataclass(frozen=True)
class Iter:
    """``.iterate(body, max_iters=n)``."""

    body: Node
    n: int


@dataclass(frozen=True)
class Fan:
    """``.map(body)`` over ``[10 * x + i for i in range(width)]``, summed."""

    body: Node
    width: int


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


def _label(node: Node, counter: list[int]) -> Node:
    """Return ``node`` with its leaves named ``l1, l2, ...`` in depth-first order."""
    if isinstance(node, Leaf):
        counter[0] += 1
        return Leaf(f"l{counter[0]}", counter[0])
    if isinstance(node, Seq):
        return Seq(tuple(_label(child, counter) for child in node.children))
    if isinstance(node, Iter):
        return Iter(_label(node.body, counter), node.n)
    if isinstance(node, Fan):
        return Fan(_label(node.body, counter), node.width)
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
}

SHAPES: dict[str, Seq] = {}
for _key, _inner in _INNER.items():
    SHAPES[f"leaf>{_key}"] = Seq((_label(_L, [0]), _label(_inner, [1])))
    SHAPES[f"{_key}>leaf"] = Seq((_label(_inner, [0]), Leaf("l99", 99)))


def _has(node: Node, kind: type) -> bool:
    """True when ``node`` contains a node of ``kind``."""
    if isinstance(node, kind):
        return True
    if isinstance(node, Seq):
        return any(_has(child, kind) for child in node.children)
    if isinstance(node, Branch):
        return _has(node.then, kind) or _has(node.else_, kind)
    if isinstance(node, Iter | Fan | Scope):
        return _has(node.body, kind)
    return False


# --- Reference model --------------------------------------------------------


def _model_node(node: Node, x: int, out: dict[str, int]) -> int:
    """Evaluate ``node`` on ``x``, recording each leaf execution in ``out`` in order."""
    if isinstance(node, Leaf):
        key = f"{node.name}:{x}"
        assert key not in out, f"model keys collide: {key}"
        out[key] = 3 * x + node.c
        return out[key]
    if isinstance(node, Seq):
        for child in node.children:
            x = _model_node(child, x, out)
        return x
    if isinstance(node, Iter):
        for _ in range(node.n):
            x = _model_node(node.body, x, out)
        return x
    if isinstance(node, Fan):
        return sum(_model_node(node.body, 10 * x + i, out) for i in range(node.width))
    if isinstance(node, Branch):
        return _model_node(node.then if x % 2 == 0 else node.else_, x, out)
    return _model_node(node.body, x, out)


def model(shape: Seq) -> tuple[int, dict[str, int]]:
    """Return the uninterrupted run's result and its ``done`` map, in execution order."""
    out: dict[str, int] = {}
    return _model_node(shape, RUN_INPUT, out), out


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
        if probe.policy == "leaf":
            await ctx.checkpoint()
            probe.checkpointed.append(key)
        if fire:
            probe.stop_here()  # sets the halt, or raises for a crash / an exception
        return done[key]

    body.__name__ = body.__qualname__ = leaf.name
    return verb(body)


def _add(node: Node, flow: Flow, probe: Probe, parallel: bool) -> Flow:
    """Append ``node`` to ``flow`` as one chain step."""
    if isinstance(node, Leaf):
        return flow.call(_leaf_verb(node, probe))
    if isinstance(node, Seq | Scope):
        sub = FlowFactory(LG).create()
        for child in node.children if isinstance(node, Seq) else (node.body,):
            _add(child, sub, probe, parallel)
        if isinstance(node, Seq):
            return flow.call(sub)
        return flow.call(
            sub,
            state=lambda p: {"done": dict(p.get("done", {}))},
            merge=lambda p, c: p.setdefault("done", {}).update(c["done"]),
        )
    if isinstance(node, Iter):
        return flow.iterate(lambda b: _add(node.body, b, probe, parallel), max_iters=node.n)
    if isinstance(node, Branch):
        return flow.branch(
            when=lambda prev, _ctx: prev % 2 == 0,
            then=lambda b: _add(node.then, b, probe, parallel),
            else_=lambda b: _add(node.else_, b, probe, parallel),
        )
    width = node.width
    return flow.map(
        lambda b: _add(node.body, b, probe, parallel),
        items=lambda prev, _ctx: [10 * prev + i for i in range(width)],
        aggregate=sum,
        max_concurrency=None if parallel else 1,
    )


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
    """The ``done`` map resume continues from: newest commit with state, not a failure."""
    async for commit in history.commits():
        if commit.meta.node_path == FAILED_NODE_PATH:
            continue
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

    @property
    def id(self) -> str:
        run = "par" if self.parallel else "seq"
        return (
            f"{self.shape}-{self.stop}-at{self.stop_at}-{self.mode}"
            f"-{self.policy}-{run}-{self.store}"
        )


KNOWN_DEFECTS_FILE = Path(__file__).with_name("halt_restart_known_defects.json")


_RAISES: dict[str, type[BaseException]] = {"TypeError": TypeError, "AssertionError": AssertionError}


def _known_defects() -> dict[str, tuple[str, type[BaseException]]]:
    """Case id → (reason, exception the defect raises), for every case failing on a known defect.

    The file lists exact case ids per defect: whether a halted map hands
    ``aggregate`` a partial result depends on which item was running, so
    no rule over the dimensions matches the failing cases exactly. The
    exception type keeps a listed case from passing as xfail when it
    fails for another reason.
    """
    defects = json.loads(KNOWN_DEFECTS_FILE.read_text(encoding="utf-8"))
    return {
        case_id: (d["reason"], _RAISES[d["raises"]])
        for d in defects.values()
        for case_id in d["cases"]
    }


def _memory_cases() -> Iterator[Case]:
    """Every shape × stop point × stop × policy × map mode, in memory."""
    for name, shape in SHAPES.items():
        leaves = len(model(shape)[1])
        for parallel in (False, True) if _has(shape, Fan) else (False,):
            for policy in ("none", "on_iterate", "on_map_item", "leaf"):
                for stop in ("halt", "crash", "exception"):
                    for stop_at in range(1, leaves + 1):
                        for mode in ("before", "after"):
                            yield Case(name, stop, stop_at, mode, policy, parallel, "mem")


def _file_cases() -> Iterator[Case]:
    """A slice against the file store: halt and crash, no policy or a checkpoint per leaf."""
    for name, shape in SHAPES.items():
        leaves = len(model(shape)[1])
        for policy in ("none", "leaf"):
            for stop in ("halt", "crash"):
                for stop_at in range(1, leaves + 1):
                    yield Case(name, stop, stop_at, "after", policy, False, "file")


def _cases() -> list[Case]:
    return [*_memory_cases(), *_file_cases()]


def _params() -> Iterator[Any]:
    known = _known_defects()
    for case in _cases():
        defect = known.get(case.id)
        marks = (
            [pytest.mark.xfail(reason=defect[0], raises=defect[1], strict=True)] if defect else []
        )
        yield pytest.param(case, id=case.id, marks=marks)


# --- Tests ------------------------------------------------------------------


def _store(kind: str, tmp_path: Path) -> CheckpointStore:
    if kind == "mem":
        return InMemoryCheckpointStore()
    return JsonFileCheckpointStore(LG, tmp_path / "cp")


def test_known_defects_name_existing_cases() -> None:
    """Every listed case id exists in the matrix, so the list cannot go stale unnoticed."""
    stale = set(_known_defects()) - {case.id for case in _cases()}
    assert not stale, f"{KNOWN_DEFECTS_FILE.name} lists cases the matrix no longer has: {stale}"


@pytest.mark.parametrize("parallel", [False, True], ids=["seq", "par"])
@pytest.mark.parametrize("shape_name", list(SHAPES))
async def test_uninterrupted_run_matches_model(shape_name: str, parallel: bool) -> None:
    shape = SHAPES[shape_name]
    expected_result, baseline = model(shape)
    store = InMemoryCheckpointStore()
    probe = Probe()

    assert await _flow(shape, probe, store, parallel=parallel).run(RUN_INPUT) == expected_result
    assert sorted(probe.executed) == sorted(baseline)
    if not parallel:
        assert probe.executed == list(baseline)
    history = History(store, FLOW_NAME)
    head = await history.head()
    assert head is not None
    assert await _root_done(history, head) == baseline


async def _run_stopped(case: Case, shape: Seq, store: CrashableStore) -> Probe:
    """Run ``shape`` until it stops as ``case`` dictates; check how it ended."""
    halt = asyncio.Event()
    probe = Probe(
        policy=case.policy,
        stop=case.stop,
        stop_at=case.stop_at,
        mode=case.mode,
        halt=halt,
        store=store,
    )
    flow = _flow(shape, probe, store, parallel=case.parallel, halt=halt)
    expected: type[BaseException] | None = {
        "halt": None,
        "crash": SimulatedCrash,
        "exception": LeafError,
    }[case.stop]
    if expected is None:
        await flow.run(RUN_INPUT)
    else:
        with pytest.raises(expected):
            await flow.run(RUN_INPUT)
    writes = store.writes
    await _settle()
    assert store.writes == writes, "the stopped run wrote after run() ended"
    return probe


@pytest.mark.parametrize("case", list(_params()))
async def test_resume_after_stop_matches_uninterrupted_run(case: Case, tmp_path: Path) -> None:
    shape = SHAPES[case.shape]
    expected_result, baseline = model(shape)
    inner = _store(case.store, tmp_path)
    history = History(inner, FLOW_NAME)

    first = await _run_stopped(case, shape, CrashableStore(inner))
    captured = await _resume_point_done(history)
    assert captured.items() <= baseline.items(), "resume point holds work never done"
    assert set(captured) <= set(first.executed), "resume point holds work this run never did"
    if case.stop == "halt":
        lost = [k for k in first.executed if k not in captured]
        assert not lost, f"completed work missing from the halt checkpoint: {lost}"
        if await history.is_complete():
            # The halt was set after the last leaf's work: every step completed,
            # so the run finished and there is nothing to resume.
            assert captured == baseline
            return
        # A save point written after the halt checkpoint (a sibling map item
        # finishing its iteration) is kept: it carries more progress.
        outcomes = [commit.meta.outcome async for commit in history.commits()]
        assert "halted" in outcomes, f"no halt checkpoint; outcomes {outcomes}"
    if case.policy == "leaf":
        # A leaf that raised inside a state= scope takes its work down with the
        # scope: the block never merges it, and a later save point (a sibling
        # map item carrying on) no longer holds it. Resume runs that leaf again.
        discarded = {first.stopped} if case.stop == "exception" and _has(shape, Scope) else set()
        lost = [k for k in first.checkpointed if k not in captured and k not in discarded]
        assert not lost, f"checkpointed work missing from the resume point: {lost}"

    resumed = Probe(policy=case.policy)
    flow = _flow(shape, resumed, inner, parallel=case.parallel)
    assert await flow.run(RUN_INPUT, resume="latest") == expected_result
    head = await history.head()
    assert head is not None and await _root_done(history, head) == baseline
    assert await history.is_complete()
    assert len(resumed.executed) == len(set(resumed.executed)), "a leaf ran twice on resume"
    missing = [k for k in baseline if k not in captured]
    assert [k for k in resumed.executed if k not in captured] == missing or case.parallel
    assert set(resumed.executed) - set(captured) == set(missing)
    again = [k for k in resumed.executed if k in captured]
    assert len(again) <= 1, f"more than the interrupted step ran again: {again}"
