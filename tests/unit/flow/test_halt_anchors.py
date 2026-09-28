# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""A halted run leaves exactly one correct resume point.

- Halt that arrives after all work completed is a clean exit: the final
  state is committed at ``$end`` and the history is complete, and never
  deleted under ``gc_on_success``.
- A step the halt cut short (map items skipped, a Loop turn paused, a
  verb's ``ctx.mark_cut_short()``) anchors the halt commit, so resume
  re-runs it instead of moving past it.
- A resumed top-level step gets back the input it had, when that input
  survives a JSON round trip; otherwise a cut-short step anchors at the
  step before it.
- Once the halt commit is written, later work does not supersede it as
  the history's head.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from llm_gent.flow import Context, FlowFactory, History, Loop, Role, verb
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
        assert head is not None and await history.scopes(head) == [{"s1": 1}]
        assert await history.is_complete()


class TestMarkCutShort:
    async def test_marked_last_step_reruns_with_its_input(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A verb that stops early on halt and marks itself re-runs, handed its input again."""
        received: list[Any] = []

        def build(halt: asyncio.Event | None) -> Any:
            @verb
            async def produce(ctx: Context[dict[str, Any]], _p: Any = None) -> dict[str, int]:
                received.append("produce")
                return {"n": 7}

            @verb
            async def consume(ctx: Context[dict[str, Any]], prev: Any = None) -> None:
                received.append(prev)
                if halt is not None:
                    halt.set()  # halt arrives mid-step; the verb stops before its work
                    ctx.mark_cut_short()
                    return
                ctx.state.data["done"] = prev["n"]

            flow = (
                FlowFactory(make_test_logger())
                .create(state={})
                .with_checkpointer(store, "marked")
                .call(produce)
                .call(consume)
            )
            return _with_halt(flow, halt)

        await build(asyncio.Event()).run()
        history = History(store, "marked")
        head = await history.head()
        assert head is not None and head.meta.outcome == "halted"
        assert not await history.is_complete()

        received.clear()
        await build(None).run(resume="replay")
        assert received == [{"n": 7}]
        head = await history.head()
        assert head is not None and await history.scopes(head) == [{"done": 7}]
        assert await history.is_complete()


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


class TestResumedStepInput:
    """A map fed by the previous step's result gets its items back on replay."""

    @staticmethod
    def _flow(
        store: JsonFileCheckpointStore,
        halt: asyncio.Event | None,
        calls: list[str],
        *,
        items: Any,
        halt_in: str,
    ) -> Any:
        @verb
        async def produce(ctx: Context[dict[str, Any]], _p: Any = None) -> Any:
            calls.append("produce")
            if halt is not None and halt_in == "produce":
                halt.set()
            return items

        @verb
        async def item(ctx: Context[dict[str, Any]], x: int) -> int:
            calls.append(f"item:{x}")
            if halt is not None and halt_in == "item":
                halt.set()
            return x

        @verb
        async def after(ctx: Context[dict[str, Any]], prev: Any = None) -> None:
            ctx.state.data["after"] = prev

        flow = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpointer(store, "input")
            .call(produce)
            .map(lambda b: b.call(item), max_concurrency=1)
            .call(after)
        )
        return _with_halt(flow, halt)

    async def _run_then_replay(
        self, store: JsonFileCheckpointStore, *, items: Any, halt_in: str
    ) -> list[str]:
        calls: list[str] = []
        await self._flow(store, asyncio.Event(), calls, items=items, halt_in=halt_in).run()
        calls.clear()
        await self._flow(store, None, calls, items=items, halt_in=halt_in).run(resume="replay")
        head = await History(store, "input").head()
        assert head is not None and await History(store, "input").scopes(head) == [
            {"after": [1, 2, 3]}
        ]
        return calls

    async def test_map_cut_short_replays_with_its_items(
        self, store: JsonFileCheckpointStore
    ) -> None:
        calls = await self._run_then_replay(store, items=[1, 2, 3], halt_in="item")
        assert calls == ["item:1", "item:2", "item:3"]

    async def test_halt_before_map_replays_with_its_items(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Halt during the step before the map anchors at the map, which gets its input back."""
        calls = await self._run_then_replay(store, items=[1, 2, 3], halt_in="produce")
        assert calls == ["item:1", "item:2", "item:3"]

    async def test_unstorable_input_anchors_at_the_previous_step(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A tuple does not survive JSON, so the step producing it re-runs instead."""
        calls = await self._run_then_replay(store, items=(1, 2, 3), halt_in="item")
        assert calls == ["produce", "item:1", "item:2", "item:3"]


class TestLoopPauseWithoutFactory:
    async def test_paused_last_step_reruns_with_its_task(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """No ConversationFactory means no paused turn to save; the step still re-runs."""
        role = Role(name="r", backend="openai", model="gpt-4o-mini")
        tasks: list[str] = []

        @dataclass
        class _Result:
            paused: bool

        class _SAIA:
            def __init__(self, halt: asyncio.Event | None) -> None:
                self.role = role
                self.halt = halt

            async def complete(self, task: str, **kwargs: Any) -> _Result:
                tasks.append(task)
                if self.halt is not None:
                    self.halt.set()
                    return _Result(paused=True)
                return _Result(paused=False)

        def build(halt: asyncio.Event | None) -> Any:
            loop = Loop(role, saia=_SAIA(halt))

            @verb(role=role)
            async def plan(ctx: Context[dict[str, Any]], _p: Any = None) -> str:
                return "the task"

            @verb(role=role)
            async def act(ctx: Context[dict[str, Any]], task: Any = None) -> Any:
                return await loop(ctx, task)

            flow = (
                FlowFactory(make_test_logger())
                .create(state={})
                .with_checkpointer(store, "loop-pause")
                .call(plan)
                .call(act)
            )
            return _with_halt(flow, halt)

        await build(asyncio.Event()).run()
        history = History(store, "loop-pause")
        head = await history.head()
        assert head is not None and head.meta.outcome == "halted"
        assert not await history.is_complete()

        tasks.clear()
        await build(None).run(resume="replay")
        assert tasks == ["the task"]
        assert await history.is_complete()


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
