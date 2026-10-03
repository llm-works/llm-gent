# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Resume around rescue policies and typed state.

A structure that stops early keeps its position registered for the run's
halt checkpoint; whatever carries on past it drops that position — a
chain moving past a step a rescue policy recovered. Positions dropped
that way must not reach a later checkpoint, or resume would continue a
step that already ended.
Typed values — dataclass state, pydantic models between steps, in an
iterate's carry and as map items — come back as the types they were.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from llm_gent.flow import (
    HALTED,
    Context,
    FlowFactory,
    History,
    Interrupted,
    StateDataclass,
    TypeStateFactory,
    verb,
)
from llm_gent.flow.stores import JsonFileCheckpointStore

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


@pytest.fixture
def store(tmp_path: Path) -> JsonFileCheckpointStore:
    return JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")


async def _paths(store: JsonFileCheckpointStore, name: str) -> list[str]:
    """Paths of every cursor and child scope in the head commit's snapshot."""
    history = History(store, name)
    head = await history.head()
    assert head is not None
    snapshot = await history.snapshot(head)
    return sorted([*snapshot.cursors, *snapshot.scopes])


def _stop_then_halt(halt: asyncio.Event, ran: list[str], arm: bool) -> Any:
    """A last step that sets the halt and stops before its work while ``arm``; else finishes."""

    @verb
    async def last(ctx: Context[dict[str, Any]], x: Any) -> Any:
        ran.append("last")
        if arm:
            halt.set()
            raise Interrupted()
        return x

    return last


class TestRescue:
    async def test_rescued_step_leaves_no_position_behind(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """An iterate that raised in pass 1 is rescued; the later halt checkpoint has none of it."""
        ran: list[str] = []

        def build(halt: asyncio.Event, arm: bool) -> Any:
            @verb
            async def step(ctx: Context[dict[str, Any]], x: int) -> int:
                ran.append(f"step:{x}")
                if x == 1:
                    raise RuntimeError("boom")
                return x + 1

            return (
                FlowFactory(make_test_logger())
                .create(state={})
                .with_checkpoint_store(store, "rescue")
                .with_checkpointer()
                .with_halt(halt)
                .iterate(lambda b: b.call(step), max_iters=3, name="loop")
                .rescue(lambda _e, _p, _c: -1)
                .then(_stop_then_halt(halt, ran, arm))
            )

        assert await build(asyncio.Event(), arm=True).run(0) is HALTED
        assert ran == ["step:0", "step:1", "last"]
        assert await _paths(store, "rescue") == [""]  # only the top chain, at `last`

        ran.clear()
        assert await build(asyncio.Event(), arm=False).run(0, resume="latest") == -1
        assert ran == ["last"]


class Item(BaseModel):
    """A map item and the value passed between steps."""

    n: int


@dataclass
class Totals(StateDataclass):
    """Typed run state."""

    seen: list[int] = field(default_factory=list)
    last: Item | None = None


def _typed_flow(store: Any, halt: asyncio.Event, ran: list[str], stop_at: int | None) -> Any:
    """``seed -> iterate(bump) -> map(double) -> total`` over pydantic values, typed state.

    The ``stop_at``-th step to run sets the halt and stops before its work.
    """
    count = [0]

    def stopping(label: str) -> bool:
        count[0] += 1
        ran.append(label)
        if count[0] == stop_at:
            halt.set()
            return True
        return False

    @verb
    async def seed(ctx: Context[Totals], n: int) -> Item:
        if stopping("seed"):
            raise Interrupted()
        return Item(n=n)

    @verb
    async def bump(ctx: Context[Totals], item: Item) -> Item:
        if stopping(f"bump:{item.n}"):
            raise Interrupted()
        ctx.state.data.last = item
        return Item(n=item.n + 1)

    @verb
    async def double(ctx: Context[Totals], item: Item) -> Item:
        if stopping(f"double:{item.n}"):
            raise Interrupted()
        return Item(n=item.n * 2)

    @verb
    async def total(ctx: Context[Totals], items: list[Item]) -> int:
        if stopping("total"):
            raise Interrupted()
        ctx.state.data.seen = [i.n for i in items]
        return sum(i.n for i in items)

    return (
        FlowFactory(make_test_logger(), state_factory=TypeStateFactory(Totals))
        .create(state=Totals())
        .with_checkpoint_store(store, "typed")
        .with_checkpointer()
        .with_halt(halt)
        .call(seed)
        .iterate(lambda b: b.call(bump), max_iters=2)
        .map(
            lambda b: b.call(double),
            items=lambda prev, _c: [prev, Item(n=prev.n + 10)],
            max_concurrency=1,
        )
        .then(total)
    )


_TYPED_STEPS = ["seed", "bump:1", "bump:2", "double:3", "double:13", "total"]


class TestTypedValues:
    @pytest.mark.parametrize("stop_at", range(1, len(_TYPED_STEPS) + 1))
    async def test_halt_and_resume_keep_the_types(
        self, store: JsonFileCheckpointStore, stop_at: int
    ) -> None:
        """Halted at each step in turn, resume runs that step and the rest, over typed values."""
        ran: list[str] = []
        assert await _typed_flow(store, asyncio.Event(), ran, stop_at).run(1) is HALTED
        assert ran == _TYPED_STEPS[:stop_at]

        ran.clear()
        assert await _typed_flow(store, asyncio.Event(), ran, None).run(1, resume="latest") == 32
        assert ran == _TYPED_STEPS[stop_at - 1 :]
        history = History(store, "typed")
        head = await history.head()
        assert head is not None
        state = await history.root_state(head, TypeStateFactory(Totals))
        assert state is not None
        assert state.seen == [6, 26] and state.last == Item(n=2)
