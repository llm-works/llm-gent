# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""One repo per run: the store on the top-level flow, checkpointers declared anywhere.

``with_checkpoint_store(store, client_flow_id)`` on the top-level flow is
the run's repo; ``with_checkpointer(name=None)`` on any flow makes saves
inside it write commits there, each holding the whole run. A save belongs
to the innermost checkpointer; a named one moves its repo-global tag.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_gent.flow import Context, FlowFactory, History, Interrupted, verb
from llm_gent.flow.stores import InMemoryCheckpointStore

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]

NAME = "repo"


class Crash(BaseException):
    """The process died."""


def _ff() -> FlowFactory:
    return FlowFactory(make_test_logger())


def _nested(store: Any, ran: list[str], *, crash: bool, halt: asyncio.Event | None = None) -> Any:
    """``sub(a1 → a2 → b) → c``; only ``sub`` declares a checkpointer; ``a2`` saves first thing.

    With ``crash``, ``b`` dies; with ``halt``, ``b`` sets it and stops before its work.
    """

    @verb
    async def a1(ctx: Context[Any], x: int) -> int:
        ran.append("a1")
        ctx.state.data["a1"] = x
        return x + 1

    @verb
    async def a2(ctx: Context[Any], x: int) -> int:
        ran.append("a2")
        await ctx.checkpoint()
        ctx.state.data["a2"] = x
        return x + 1

    @verb
    async def b(ctx: Context[Any], x: int) -> int:
        ran.append("b")
        if crash:
            raise Crash()
        if halt is not None and not halt.is_set():
            halt.set()
            raise Interrupted()
        return x * 10

    @verb
    async def c(ctx: Context[Any], x: int) -> int:
        ran.append("c")
        return x + 3

    sub = _ff().create().with_checkpointer().call(a1).then(a2).then(b)
    flow = _ff().create(state={}).with_checkpoint_store(store, NAME).call(sub).then(c)
    return flow if halt is None else flow.with_halt(halt)


class TestNestedCheckpointer:
    async def test_nested_save_goes_to_the_run_repo_and_resume_keeps_its_work(self) -> None:
        store = InMemoryCheckpointStore()
        ran: list[str] = []
        with pytest.raises(Crash):
            await _nested(store, ran, crash=True).run(1)
        head = await History(store, NAME).head()
        assert head is not None and head.meta.outcome == "ok"
        assert (await History(store, NAME).snapshot(head)).root == {"a1": 1}

        ran.clear()
        assert await _nested(store, ran, crash=False).run(1, resume="latest") == 33
        assert ran == ["a2", "b", "c"]  # a1's checkpointed work does not run again

    async def test_halt_inside_a_checkpointed_subtree_resumes_as_uninterrupted(self) -> None:
        store = InMemoryCheckpointStore()
        ran: list[str] = []
        halt = asyncio.Event()
        assert await _nested(store, ran, crash=False, halt=halt).run(1) is None
        head = await History(store, NAME).head()
        assert head is not None and head.meta.outcome == "halted"

        ran.clear()
        assert await _nested(store, ran, crash=False).run(1, resume="latest") == 33
        assert ran == ["b", "c"]

    async def test_without_a_checkpointer_a_save_writes_nothing(self) -> None:
        store = InMemoryCheckpointStore()
        written: list[str | None] = []

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            written.append(await ctx.checkpoint("x"))
            return x

        await _ff().create(state={}).with_checkpoint_store(store, NAME).call(a).run(1)
        assert written == [None]
        history = History(store, NAME)
        assert await history.checkpoint_names() == []
        assert [c.meta.node_path async for c in history.commits()] == ["$end"]

    async def test_a_flow_run_on_its_own_uses_its_own_repo(self) -> None:
        store = InMemoryCheckpointStore()

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            await ctx.checkpoint()
            return x

        sub = _ff().create(state={}).with_checkpoint_store(store, "sub").with_checkpointer().call(a)
        await sub.run(1)
        assert await History(store, "sub").head() is not None
        assert await History(store, NAME).head() is None


class TestOneRepoRules:
    async def test_checkpointer_without_a_store_raises_at_run_start(self) -> None:
        ran: list[str] = []

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            ran.append("a")
            return x

        sub = _ff().create().with_checkpointer().call(a)
        flow = _ff().create(state={}).call(a).call(sub)
        with pytest.raises(RuntimeError, match="checkpointers require a checkpoint store"):
            await flow.run(1)
        assert ran == []

    async def test_store_on_a_nested_flow_raises_at_run_start(self) -> None:
        store = InMemoryCheckpointStore()

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            return x

        sub = _ff().create().with_checkpoint_store(store, "sub").call(a)
        flow = _ff().create(state={}).with_checkpoint_store(store, NAME).call(sub)
        with pytest.raises(RuntimeError, match="a run has one repo"):
            await flow.run(1)
        assert await History(store, NAME).flow_id() is None  # nothing was written

    @pytest.mark.parametrize("bad", ["latest", "complete", "a" * 64, ""])
    async def test_checkpointer_name_follows_checkpoint_name_rules(self, bad: str) -> None:
        with pytest.raises(ValueError):
            _ff().create().with_checkpointer(bad)


class TestNamedCheckpointer:
    async def test_named_map_body_resumes_at_its_latest_item_checkpoint(self) -> None:
        store = InMemoryCheckpointStore()
        ran: list[int] = []

        def flow() -> Any:
            @verb
            async def item(ctx: Context[Any], n: int) -> int:
                ran.append(n)
                await ctx.checkpoint()
                return n * 2

            return (
                _ff()
                .create(state={})
                .with_checkpoint_store(store, NAME)
                .map(
                    lambda b: b.with_checkpointer("research").call(item),
                    items=lambda _p, _c: [1, 2, 3],
                    max_concurrency=1,
                )
            )

        assert await flow().run() == [2, 4, 6]
        history = History(store, NAME)
        tagged = await history.checkpoint("research")
        assert tagged is not None
        cursors = (await history.snapshot(tagged)).cursors
        [done] = [c["done"] for c in cursors.values() if "done" in c]
        assert sorted(done) == ["0", "1"]  # the third item's own checkpoint: items 0 and 1 done

        ran.clear()
        assert await flow().run(resume="research") == [2, 4, 6]
        assert ran == [3]

    async def test_a_save_moves_only_its_innermost_checkpointer_tag(self) -> None:
        store = InMemoryCheckpointStore()

        @verb
        async def inner_step(ctx: Context[Any], x: int) -> int:
            await ctx.checkpoint()
            return x

        @verb
        async def outer_step(ctx: Context[Any], x: int) -> int:
            return x

        inner = _ff().create().with_checkpointer("inner").call(inner_step)
        outer = _ff().create().with_checkpointer("outer").call(outer_step).call(inner)
        await _ff().create(state={}).with_checkpoint_store(store, NAME).call(outer).run(1)
        assert await History(store, NAME).checkpoint_names() == ["inner"]

    async def test_one_name_on_two_flows_is_one_tag(self) -> None:
        store = InMemoryCheckpointStore()

        @verb
        async def save(ctx: Context[Any], x: int) -> int:
            await ctx.checkpoint()
            return x + 1

        first = _ff().create().with_checkpointer("phase").call(save)
        second = _ff().create().with_checkpointer("phase").call(save)
        flow = _ff().create(state={}).with_checkpoint_store(store, NAME).call(first).call(second)
        assert await flow.run(1) == 3
        history = History(store, NAME)
        assert await history.checkpoint_names() == ["phase"]
        tagged = await history.checkpoint("phase")
        line = [c.content_hash async for c in history.commits()]
        # The tag is on the later save (second's): the newest "ok" commit on the line.
        assert tagged is not None and tagged.content_hash == line[1]
