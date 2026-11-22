# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""One Flow, many runs at once: each run keeps its own run state."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_gent.flow import HALTED, Context, Factory, Flow, History, verb
from llm_gent.flow.hooks import Hook
from tests.helpers import memory_store

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


class _Quiet(Hook):
    """A hook that does nothing; with one attached every run gets a run id."""


def _probe_flow(barrier: asyncio.Barrier, seen: dict[str, list[str | None]]) -> Flow:
    """One step that reads ``ctx.run_id`` before and after both runs reach ``barrier``."""

    @verb
    async def probe(ctx: Context[dict[str, Any]], tag: str) -> None:
        seen[tag].append(ctx.run_id)
        await barrier.wait()
        seen[tag].append(ctx.run_id)

    return Factory(make_test_logger()).create(state={}).call(probe).with_hooks(_Quiet())


class TestConcurrentRuns:
    async def test_each_run_sees_its_own_run_id(self) -> None:
        seen: dict[str, list[str | None]] = {"a": [], "b": []}
        flow = _probe_flow(asyncio.Barrier(2), seen)

        await asyncio.gather(flow.run("a"), flow.run("b"))

        a_before, a_after = seen["a"]
        b_before, b_after = seen["b"]
        assert a_before is not None and b_before is not None
        assert a_before != b_before
        assert a_after == a_before
        assert b_after == b_before

    async def test_each_run_keeps_its_own_history_and_halt(self) -> None:
        """Run ``a`` halts while ``b`` is mid-step: ``b`` finishes; each history holds its own run."""
        store = memory_store()
        flow = _two_histories_flow(store, asyncio.Barrier(2))
        halt_a, halt_b = asyncio.Event(), asyncio.Event()

        results = await asyncio.gather(
            flow.run("a", state={}, halt=halt_a, history="a"),
            flow.run("b", state={}, halt=halt_b, history="b"),
        )
        assert results == [HALTED, "b:b"]
        assert not halt_b.is_set()

        a, b = History(store, "a"), History(store, "b")
        assert not await a.is_complete() and await b.is_complete()
        for history, tag in ((a, "a"), (b, "b")):
            roots = [(await history.snapshot(c)).root async for c in history.commits()]
            assert roots and all(root == {"tag": tag} for root in roots)

        assert await flow.run(state={}, history="a", resume="latest") == "a:a"
        assert await a.is_complete()


def _two_histories_flow(store: Any, barrier: asyncio.Barrier) -> Flow:
    """``first`` saves, then (in a run with a halt) waits for the other run; ``a`` sets its halt.

    The second wait keeps ``b`` inside ``first`` while ``a``'s halt is set.
    """

    @verb
    async def first(ctx: Context[dict[str, Any]], tag: str) -> str:
        ctx.state.data["tag"] = tag
        await ctx.checkpoint()
        if ctx.halt is not None:
            await barrier.wait()  # both runs have saved
            if tag == "a":
                ctx.halt.set()
            await barrier.wait()
        return tag

    @verb
    async def second(ctx: Context[dict[str, Any]], tag: str) -> str:
        return f"{tag}:{ctx.state.data['tag']}"

    ff = Factory(make_test_logger(), checkpoint_store=store)
    return ff.create().with_checkpointer().call(first).then(second)
