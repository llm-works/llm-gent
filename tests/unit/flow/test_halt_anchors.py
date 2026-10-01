# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""A halt set during the last top-level step.

- When the last step completed, the run finished: the history is marked
  complete with the final state.
- When the last step was interrupted (it raised ``Interrupted``), the
  halt commit keeps the cursor on it: the history is not complete, is
  never deleted under ``gc_on_success``, and resume runs that step again.
- A map as the last step with items the halt stopped is interrupted.
- A subflow's own ``.with_halt`` does not count as the run's halt: the
  run completes normally.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from llm_gent.flow import Context, FlowFactory, History, Interrupted, verb
from llm_gent.flow.stores import JsonFileCheckpointStore

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


@pytest.fixture
def store(tmp_path: Path) -> JsonFileCheckpointStore:
    """A fresh store under an isolated temp root."""
    return JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")


def _with_halt(flow: Any, halt: asyncio.Event | None) -> Any:
    return flow.with_halt(halt) if halt is not None else flow


def _two_steps(
    store: Any, name: str, halt: asyncio.Event | None, calls: list[str], mode: str
) -> Any:
    """``s1 → s2``; with ``halt``, s2 sets it — then completes (``finish``) or not (``bail``)."""

    @verb
    async def s1(ctx: Context[dict[str, Any]], _p: Any = None) -> None:
        calls.append("s1")
        ctx.state.data["s1"] = ctx.state.data.get("s1", 0) + 1

    @verb
    async def s2(ctx: Context[dict[str, Any]], _p: Any = None) -> None:
        calls.append("s2")
        if halt is not None:
            halt.set()
            if mode == "bail":
                raise Interrupted()
        ctx.state.data["s2"] = ctx.state.data.get("s2", 0) + 1

    flow = (
        FlowFactory(make_test_logger())
        .create(state={})
        .with_checkpoint_store(store, name)
        .with_checkpointer()
        .call(s1)
        .call(s2)
    )
    return _with_halt(flow, halt)


class TestHaltInLastStep:
    async def test_completed_last_step_finishes_the_run(
        self, store: JsonFileCheckpointStore
    ) -> None:
        calls: list[str] = []
        assert await _two_steps(store, "finish", asyncio.Event(), calls, "finish").run() is None
        history = History(store, "finish")
        assert await history.is_complete()
        head = await history.head()
        assert head is not None
        assert (await history.snapshot(head)).root == {"s1": 1, "s2": 1}

    async def test_interrupted_last_step_runs_again_on_resume(
        self, store: JsonFileCheckpointStore
    ) -> None:
        calls: list[str] = []
        await _two_steps(store, "bail", asyncio.Event(), calls, "bail").run()
        history = History(store, "bail")
        assert not await history.is_complete()
        head = await history.head()
        assert head is not None and head.meta.outcome == "halted"
        snapshot = await history.snapshot(head)
        assert (snapshot.root, snapshot.scopes) == ({"s1": 1}, {})

        calls.clear()
        await _two_steps(store, "bail", None, calls, "bail").run(resume="latest")
        assert calls == ["s2"]
        head = await history.head()
        assert head is not None and await history.is_complete()
        assert (await history.snapshot(head)).root == {"s1": 1, "s2": 1}

    async def test_interrupted_run_is_never_deleted_under_gc_on_success(
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
            raise Interrupted()

        flow = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpoint_store(store, "gc-late-halt")
            .with_checkpointer()
            .with_halt(halt)
            .call(s1)
        )
        await flow.run()

        history = History(store, "gc-late-halt")
        head = await history.head()
        assert head is not None and head.meta.outcome == "halted"
        snapshot = await history.snapshot(head)
        assert (snapshot.root, snapshot.scopes) == ({"s1": 1}, {})
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
            .with_checkpoint_store(store, "map-last")
            .with_checkpointer()
            .with_halt(halt)
            .map(lambda b: b.call(item), items=lambda _p, _c: [1, 2, 3], max_concurrency=1)
        )
        await flow.run()

        assert ran == [1]
        history = History(store, "map-last")
        head = await history.head()
        assert head is not None and head.meta.outcome == "halted"
        assert not await history.is_complete()


class TestSubflowLocalHalt:
    """A subflow's own ``.with_halt`` stops that subtree; the run carries on normally."""

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
            .with_checkpoint_store(store, "local-map")
            .with_checkpointer()
            .call(sub)
        )
        await flow.run()

        assert ran == [1]
        assert await History(store, "local-map").is_complete()
