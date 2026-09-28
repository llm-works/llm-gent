# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""A halted run leaves exactly one halt commit, carrying the latest state.

- Halt set during the last top-level step writes a halt commit there
  with the run's final state; the history is not marked complete and is
  never deleted under ``gc_on_success``.
- Once the halt commit is written, later work does not supersede it as
  the history's head.
- A subflow's own ``.with_halt`` stops that subtree only; the run's
  later saves and its completion are unaffected.
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


class TestHaltInLastStep:
    async def test_halt_commit_carries_the_final_state(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """The head is a halt commit with the final state; restart continues from it."""
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
        assert not await history.is_complete()
        head = await history.head()
        assert head is not None and head.meta.outcome == "halted"
        assert await history.scopes(head) == [{"s1": 1, "s2": 1}]

        calls.clear()
        await build(None).run(resume="restart")
        assert calls == ["s1", "s2"]
        head = await history.head()
        assert head is not None and await history.scopes(head) == [{"s1": 2, "s2": 2}]

    async def test_late_halt_never_deletes_history_under_gc_on_success(
        self, tmp_path: Path
    ) -> None:
        store = JsonFileCheckpointStore(
            make_test_logger(), tmp_path / "cp", retention="gc_on_success"
        )
        halt = asyncio.Event()

        @verb
        async def s1(ctx: Context[dict[str, Any]], _p: Any = None) -> None:
            ctx.state.data["s1"] = 1
            halt.set()

        flow = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpointer(store, "gc-late-halt")
            .with_halt(halt)
            .call(s1)
        )
        await flow.run()

        history = History(store, "gc-late-halt")
        head = await history.head()
        assert head is not None and head.meta.outcome == "halted"
        assert await history.scopes(head) == [{"s1": 1}]
        assert not await history.is_complete()

    async def test_map_as_last_step_is_not_complete(self, store: JsonFileCheckpointStore) -> None:
        """Items the halt skipped leave the run halted, not complete."""
        halt = asyncio.Event()
        ran: list[int] = []

        @verb
        async def item(ctx: Context[dict[str, Any]], x: int) -> int:
            ran.append(x)
            halt.set()
            return x

        flow = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpointer(store, "map-last")
            .with_halt(halt)
            .map(lambda b: b.call(item), items=lambda _p, _c: [1, 2, 3], max_concurrency=1)
        )
        await flow.run()

        assert ran == [1]
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

    async def test_checkpoint_after_halt_commit_is_not_written(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A ``ctx.checkpoint()`` from work still running under the halt leaves the head alone."""
        halt = asyncio.Event()
        history = History(store, "halt-ckpt")
        entered: list[int] = []
        both_in = asyncio.Event()

        async def halt_committed() -> None:
            while (head := await history.head()) is None or head.meta.outcome != "halted":
                await asyncio.sleep(0.01)

        @verb
        async def work(ctx: Context[dict[str, Any]], _p: Any = None) -> None:
            n = len(entered)
            entered.append(n)
            if len(entered) == 2:
                both_in.set()
            await both_in.wait()
            if n == 0:
                halt.set()  # this item's iterate boundary writes the halt commit
                return
            await asyncio.wait_for(halt_committed(), timeout=5)
            await ctx.checkpoint()

        body = FlowFactory(make_test_logger()).create()
        body.iterate(lambda b: b.call(work), max_iters=2)
        flow = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpointer(store, "halt-ckpt")
            .with_halt(halt)
            .map(body, items=lambda _p, _c: [1, 2])
        )
        await flow.run()

        head = await history.head()
        assert head is not None and head.meta.outcome == "halted"


class TestSubflowLocalHalt:
    """A subflow's own ``.with_halt`` stops that subtree; the run carries on normally."""

    async def test_later_checkpoint_outside_the_halted_subflow_is_written(
        self, store: JsonFileCheckpointStore
    ) -> None:
        local = asyncio.Event()

        @verb
        async def a(ctx: Context[dict[str, Any]], _p: Any = None) -> None:
            ctx.state.data["a"] = ctx.state.data.get("a", 0) + 1
            local.set()

        @verb
        async def b(ctx: Context[dict[str, Any]], _p: Any = None) -> None:
            ctx.state.data["b"] = 1
            await ctx.checkpoint()

        sub = FlowFactory(make_test_logger()).create()
        sub.iterate(lambda x: x.call(a), max_iters=3)
        sub.with_halt(local)
        flow = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpointer(store, "local-ckpt")
            .call(sub)
            .call(b)
        )
        await flow.run()

        history = History(store, "local-ckpt")
        head = await history.head()
        assert head is not None and head.meta.outcome == "ok"
        assert await history.scopes(head) == [{"a": 1, "b": 1}]

    async def test_map_skip_under_subflow_halt_does_not_leave_run_incomplete(
        self, store: JsonFileCheckpointStore
    ) -> None:
        local = asyncio.Event()
        ran: list[int] = []

        @verb
        async def item(ctx: Context[dict[str, Any]], x: int) -> int:
            ran.append(x)
            local.set()
            return x

        sub = FlowFactory(make_test_logger()).create()
        sub.map(lambda b: b.call(item), items=lambda _p, _c: [1, 2, 3], max_concurrency=1)
        sub.with_halt(local)
        flow = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpointer(store, "local-map")
            .call(sub)
        )
        await flow.run()

        assert ran == [1]
        assert await History(store, "local-map").is_complete()
