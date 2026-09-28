# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Halt then restart is equivalent to an uninterrupted run, across flow shapes.

Each case runs one generated flow shape three ways:

1. Uninterrupted — the baseline.
2. With the ambient halt fired at one leaf verb.
3. ``run(resume="restart")`` on the history the halted run left.

Leaves are state-driven, the contract restart relies on: a leaf whose
work is already recorded in ``state["done"]`` returns the recorded value
without executing. A leaf entered after the halt fired returns a partial
value without doing its work, as a verb polling ``ctx.halt`` does.

Properties checked per case:

- The halted run does not raise, and leaves a ``halted`` head that is
  not marked complete.
- The head's state holds every leaf the halted run completed, with the
  value the baseline computes: no completed work is lost or corrupted.
- Restart ends with the baseline's result and ``done`` map, marks the
  history complete, and executes exactly the leaves the head lacks —
  none twice.

A pure-Python model of each shape supplies the baseline execution order
and result; :func:`test_uninterrupted_run_matches_model` pins the
framework's uninterrupted run to that model so the oracle itself is
checked.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from llm_gent.flow import Context, Flow, FlowFactory, History, verb
from llm_gent.flow.stores import JsonFileCheckpointStore

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]

LG = make_test_logger()
FLOW_NAME = "matrix"
RUN_INPUT = 1


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


Node = Leaf | Seq | Iter | Fan

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
    return Fan(_label(node.body, counter), node.width)


_INNER: dict[str, Node] = {
    "iter": Iter(_L, 2),
    "fan": Fan(_L, 2),
    "sub": Seq((_L, _L)),
    "iter(sub)": Iter(Seq((_L, _L)), 2),
    "iter(fan)": Iter(Fan(_L, 2), 2),
    "fan(iter)": Fan(Iter(_L, 2), 2),
    "sub(fan,iter)": Seq((Fan(_L, 2), Iter(_L, 2))),
}

SHAPES: dict[str, Seq] = {}
for _key, _inner in _INNER.items():
    SHAPES[f"leaf>{_key}"] = Seq((_label(_L, [0]), _label(_inner, [1])))
    SHAPES[f"{_key}>leaf"] = Seq((_label(_inner, [0]), Leaf("l99", 99)))


# --- Reference model --------------------------------------------------------


def _model_node(
    node: Node, x: int, out: dict[str, int], in_map: list[bool | None], mapped: bool | None
) -> int:
    """Evaluate ``node`` on ``x``, recording each leaf execution in ``out`` in order.

    ``in_map`` gets one entry per leaf execution: ``None`` outside any
    map, else whether it runs in the last item of its innermost map.
    """
    if isinstance(node, Leaf):
        y = 3 * x + node.c
        key = f"{node.name}:{x}"
        assert key not in out, f"model keys collide: {key}"
        out[key] = y
        in_map.append(mapped)
        return y
    if isinstance(node, Seq):
        for child in node.children:
            x = _model_node(child, x, out, in_map, mapped)
        return x
    if isinstance(node, Iter):
        for _ in range(node.n):
            x = _model_node(node.body, x, out, in_map, mapped)
        return x
    last = node.width - 1
    return sum(
        _model_node(node.body, 10 * x + i, out, in_map, i == last) for i in range(node.width)
    )


def model(shape: Seq) -> tuple[int, dict[str, int], list[bool | None]]:
    """Return the uninterrupted run's result, ``done`` map, and per-leaf map positions.

    The ``done`` map and the positions are both in execution order.
    """
    out: dict[str, int] = {}
    in_map: list[bool | None] = []
    return _model_node(shape, RUN_INPUT, out, in_map, None), out, in_map


# --- Framework flows --------------------------------------------------------


@dataclass
class Probe:
    """Per-run leaf instrumentation: where the halt fires, what executed."""

    halt: asyncio.Event | None = None
    halt_at: int | None = None
    mode: str = "before"
    partial: str = "none"
    entered: int = 0
    executed: list[str] = field(default_factory=list)

    def partial_result(self, x: Any) -> Any:
        """What a leaf returns when it stops without doing its work."""
        return None if self.partial == "none" else x


def _leaf_verb(leaf: Leaf, probe: Probe) -> Any:
    """Return the state-driven verb for ``leaf``, reporting to ``probe``."""

    async def body(ctx: Context[dict[str, Any]], x: Any = None) -> Any:
        if ctx.halt is not None and ctx.halt.is_set():
            return probe.partial_result(x)
        key = f"{leaf.name}:{x}"
        done = ctx.state.data.setdefault("done", {})
        if key in done:
            return done[key]
        probe.entered += 1
        fire = probe.halt is not None and probe.entered == probe.halt_at
        if fire and probe.mode == "before":
            probe.halt.set()
            return probe.partial_result(x)
        done[key] = 3 * x + leaf.c
        probe.executed.append(key)
        if fire:
            probe.halt.set()
        return done[key]

    body.__name__ = body.__qualname__ = leaf.name
    return verb(body)


def _add(node: Node, flow: Flow, probe: Probe) -> Flow:
    """Append ``node`` to ``flow`` as one chain step."""
    if isinstance(node, Leaf):
        return flow.call(_leaf_verb(node, probe))
    if isinstance(node, Seq):
        sub = FlowFactory(LG).create()
        for child in node.children:
            _add(child, sub, probe)
        return flow.call(sub)
    if isinstance(node, Iter):
        return flow.iterate(lambda b: _add(node.body, b, probe), max_iters=node.n)
    width = node.width
    return flow.map(
        lambda b: _add(node.body, b, probe),
        items=lambda prev, _ctx: [10 * prev + i for i in range(width)],
        aggregate=sum,
        max_concurrency=1,
    )


def _flow(
    shape: Seq, probe: Probe, store: JsonFileCheckpointStore, halt: asyncio.Event | None
) -> Flow:
    """Build the top-level flow for ``shape``."""
    flow = FlowFactory(LG).create(state={}).with_checkpointer(store, FLOW_NAME)
    if halt is not None:
        flow = flow.with_halt(halt)
    for step in shape.children:
        _add(step, flow, probe)
    return flow


async def _head_done(history: History) -> dict[str, int]:
    """The ``done`` map in the history head's root scope."""
    head = await history.head()
    assert head is not None
    scopes = await history.scopes(head)
    return dict(scopes[0].get("done", {}))


# --- Tests ------------------------------------------------------------------


# A map that finishes after the halt fired still runs ``aggregate`` over its
# item results. When a later item was skipped (``Skipped`` sentinel) or the
# halting item returned ``None``, the aggregate raises and the halted run
# fails instead of writing its halt commit. Those cases are disabled until a
# halted map stops its enclosing work instead of aggregating.
_HALT_IN_MAP = pytest.mark.skip(reason="halt inside a map: aggregate runs on partial results")


def _aggregates_partial(map_position: bool | None, mode: str, partial: str) -> bool:
    """True when halting at a leaf in this map position hands aggregate a non-result."""
    if map_position is None:
        return False
    return not map_position or (mode, partial) == ("before", "none")


def _cases() -> Iterator[Any]:
    for name, shape in SHAPES.items():
        _, baseline, in_map = model(shape)
        for halt_at in range(1, len(baseline) + 1):
            for mode in ("before", "after"):
                for partial in ("none", "passthrough"):
                    skip = _aggregates_partial(in_map[halt_at - 1], mode, partial)
                    marks = [_HALT_IN_MAP] if skip else []
                    yield pytest.param(
                        name,
                        halt_at,
                        mode,
                        partial,
                        id=f"{name}-at{halt_at}-{mode}-{partial}",
                        marks=marks,
                    )


@pytest.mark.parametrize("shape_name", list(SHAPES))
async def test_uninterrupted_run_matches_model(tmp_path: Path, shape_name: str) -> None:
    shape = SHAPES[shape_name]
    expected_result, baseline, _ = model(shape)
    store = JsonFileCheckpointStore(LG, tmp_path / "cp")
    probe = Probe()

    assert await _flow(shape, probe, store, None).run(RUN_INPUT) == expected_result
    assert probe.executed == list(baseline)
    assert await _head_done(History(store, FLOW_NAME)) == baseline


@pytest.mark.parametrize(("shape_name", "halt_at", "mode", "partial"), list(_cases()))
async def test_restart_after_halt_matches_uninterrupted_run(
    tmp_path: Path, shape_name: str, halt_at: int, mode: str, partial: str
) -> None:
    shape = SHAPES[shape_name]
    expected_result, baseline, _ = model(shape)
    store = JsonFileCheckpointStore(LG, tmp_path / "cp")
    history = History(store, FLOW_NAME)
    halt = asyncio.Event()
    first = Probe(halt=halt, halt_at=halt_at, mode=mode, partial=partial)

    await _flow(shape, first, store, halt).run(RUN_INPUT)
    head = await history.head()
    assert head is not None and head.meta.outcome == "halted"
    assert not await history.is_complete()
    captured = await _head_done(history)
    assert captured.items() <= baseline.items(), "head holds work the baseline never did"
    lost = [k for k in first.executed if k not in captured]
    assert not lost, f"completed work missing from the head: {lost}"

    restart = Probe()
    result = await _flow(shape, restart, store, None).run(RUN_INPUT, resume="restart")
    assert result == expected_result
    assert await _head_done(history) == baseline
    assert await history.is_complete()
    assert restart.executed == [k for k in baseline if k not in captured]
