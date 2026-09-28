# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""A halted run leaves exactly one correct resume point.

- Halt that arrives after all work completed is a clean exit: the final
  state is committed at ``$end`` and the history is complete.
- Halt that skips map items anchors at the map step, so resume runs the
  skipped items instead of moving past them.
- Once the halt commit is written, later work does not supersede it as
  the history's head.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from llm_gent.flow import Context, FlowFactory, History, verb
from llm_gent.flow.stores import JsonFileCheckpointStore

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


@pytest.fixture
def store(tmp_path: Path) -> JsonFileCheckpointStore:
    """A fresh store under an isolated temp root."""
    return JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")


def _with_halt(flow: Any, halt: asyncio.Event | None) -> Any:
    return flow.with_halt(halt) if halt is not None else flow


class TestHaltAfterAllWork:
    async def test_halt_in_last_step_completes_the_run(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """No step was skipped, so the run is complete and restart continues from its end."""
        calls: list[str] = []

        def build(halt: asyncio.Event | None) -> Any:
            @verb
            async def s1(ctx: Context[dict[str, Any]], _p: Any = None) -> None:
                calls.append("s1")
                ctx.state.data["s1"] = ctx.state.data.get("s1", 0) + 1

            @verb
            async def s2(ctx: Context[dict[str, Any]], _p: Any = None) -> None:
                calls.append("s2")
                ctx.state.data["s2"] = ctx.state.data.get("s2", 0) + 1
                if halt is not None:
                    halt.set()

            flow = (
                FlowFactory(make_test_logger())
                .create(state={})
                .with_checkpointer(store, "late-halt")
                .call(s1)
                .call(s2)
            )
            return _with_halt(flow, halt)

        await build(asyncio.Event()).run()
        history = History(store, "late-halt")
        assert await history.is_complete()
        head = await history.head()
        assert head is not None and await history.scopes(head) == [{"s1": 1, "s2": 1}]

        calls.clear()
        await build(None).run(resume="restart")
        assert calls == ["s1", "s2"]
        head = await history.head()
        assert head is not None and await history.scopes(head) == [{"s1": 2, "s2": 2}]


class TestHaltSkippedMapItems:
    @staticmethod
    def _map_flow(
        store: JsonFileCheckpointStore,
        name: str,
        halt: asyncio.Event | None,
        ran: list[int],
        received: list[Any],
        *,
        with_next_step: bool,
    ) -> Any:
        @verb
        async def item(ctx: Context[dict[str, Any]], x: int) -> int:
            ran.append(x)
            if halt is not None and x == 1:
                halt.set()
            await asyncio.sleep(0)
            return x

        @verb
        async def after(ctx: Context[dict[str, Any]], prev: Any = None) -> None:
            received.append(prev)

        flow = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpointer(store, name)
            .map(lambda b: b.call(item), items=lambda _p, _c: [1, 2, 3], max_concurrency=1)
        )
        if with_next_step:
            flow = flow.call(after)
        return _with_halt(flow, halt)

    async def test_replay_runs_items_the_halt_skipped(self, store: JsonFileCheckpointStore) -> None:
        """The halt anchors at the map, not the next step, so no item is silently lost."""
        ran: list[int] = []
        received: list[Any] = []
        await self._map_flow(
            store, "map-then-call", asyncio.Event(), ran, received, with_next_step=True
        ).run()
        assert ran == [1] and received == []

        ran.clear()
        replay = self._map_flow(store, "map-then-call", None, ran, received, with_next_step=True)
        await replay.run(resume="replay")
        # Positional replay re-runs the whole map (completed item 1 included).
        assert ran == [1, 2, 3]
        assert received == [[1, 2, 3]]

    async def test_map_as_last_step_is_not_complete(self, store: JsonFileCheckpointStore) -> None:
        ran: list[int] = []
        await self._map_flow(
            store, "map-last", asyncio.Event(), ran, [], with_next_step=False
        ).run()

        history = History(store, "map-last")
        head = await history.head()
        assert head is not None and head.meta.outcome == "halted"
        assert not await history.is_complete()


class TestHaltCommitStaysHead:
    async def test_later_item_saves_do_not_supersede_the_halt_commit(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A map item's per-item save after the halt commit must not become the head."""
        halt = asyncio.Event()

        @verb
        async def bump(ctx: Context[dict[str, Any]], _p: Any = None) -> None:
            ctx.state.data["n"] = ctx.state.data.get("n", 0) + 1
            halt.set()
            await asyncio.sleep(0)

        body = FlowFactory(make_test_logger()).create()
        body.iterate(lambda b: b.call(bump), max_iters=3)
        flow = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpointer(store, "halt-head")
            .with_checkpoint_policy(on_map_item=True)
            .with_halt(halt)
            .map(body, items=lambda _p, _c: [1, 2])
        )
        await flow.run()

        head = await History(store, "halt-head").head()
        assert head is not None
        assert head.meta.outcome == "halted"
