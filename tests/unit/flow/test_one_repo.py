# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""One history per run: the top-level flow's store, checkpointers declared anywhere.

``run(history=...)`` names the run's history in the top-level flow's
``checkpoint_store``; ``with_checkpointer(name=None)`` on any flow makes
saves inside it write commits there, each holding the whole run. A save
belongs to the innermost checkpointer; a named one moves its
history-global tag.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_gent.flow import HALTED, Context, Factory, History, Interrupted, verb
from tests.helpers import memory_store

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]

NAME = "repo"


class Crash(BaseException):
    """The process died."""


def _ff(store: Any = None) -> Factory:
    return Factory(make_test_logger(), checkpoint_store=store)


def _nested(store: Any, ran: list[str], *, crash: bool, halt: bool = False) -> Any:
    """``sub(a1 → a2 → b) → c``; only ``sub`` declares a checkpointer; ``a2`` saves first thing.

    With ``crash``, ``b`` dies; with ``halt``, ``b`` sets the run's halt and stops before its work.
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
        if halt:
            ctx.halt.set()  # type: ignore[union-attr]
            raise Interrupted()
        return x * 10

    @verb
    async def c(ctx: Context[Any], x: int) -> int:
        ran.append("c")
        return x + 3

    sub = _ff().create().with_checkpointer().call(a1).then(a2).then(b)
    return _ff(store).create().call(sub).then(c)


class TestNestedCheckpointer:
    async def test_nested_save_goes_to_the_run_repo_and_resume_keeps_its_work(self) -> None:
        store = memory_store()
        ran: list[str] = []
        with pytest.raises(Crash):
            await _nested(store, ran, crash=True).run(1, history=NAME)
        head = await History(store, NAME).head()
        assert head is not None and head.meta.outcome == "ok"
        assert (await History(store, NAME).snapshot(head)).root == {"a1": 1}

        ran.clear()
        assert await _nested(store, ran, crash=False).run(1, history=NAME, resume="latest") == 33
        assert ran == ["a2", "b", "c"]  # a1's checkpointed work does not run again

    async def test_halt_inside_a_checkpointed_subtree_resumes_as_uninterrupted(self) -> None:
        store = memory_store()
        ran: list[str] = []
        flow = _nested(store, ran, crash=False, halt=True)
        assert await flow.run(1, halt=asyncio.Event(), history=NAME) is HALTED
        head = await History(store, NAME).head()
        assert head is not None and head.meta.outcome == "halted"

        ran.clear()
        assert await _nested(store, ran, crash=False).run(1, history=NAME, resume="latest") == 33
        assert ran == ["b", "c"]

    async def test_without_a_checkpointer_a_save_writes_nothing(self) -> None:
        store = memory_store()
        written: list[str | None] = []

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            written.append(await ctx.checkpoint("x"))
            return x

        await _ff(store).create().call(a).run(1, history=NAME)
        assert written == [None]
        history = History(store, NAME)
        assert await history.checkpoint_names() == []
        assert [c.meta.node_path async for c in history.commits()] == ["$end"]

    async def test_a_flow_run_on_its_own_uses_its_run_s_history(self) -> None:
        store = memory_store()

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            await ctx.checkpoint()
            return x

        sub = _ff(store).create().with_checkpointer().call(a)
        await sub.run(1, history="sub")
        assert await History(store, "sub").head() is not None
        assert await History(store, NAME).head() is None

    async def test_a_nested_flow_s_store_is_not_used(self) -> None:
        """Every flow a Factory builds carries its store; the run uses the top-level flow's."""
        store, other = memory_store(), memory_store()

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            await ctx.checkpoint()
            return x

        sub = _ff(other).create().with_checkpointer().call(a)
        await _ff(store).create().call(sub).run(1, history=NAME)
        assert await History(store, NAME).head() is not None
        assert other.list_repos() == []


class TestOneHistoryRules:
    async def test_a_checkpointer_without_a_history_raises_at_run_start(self) -> None:
        ran: list[str] = []

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            ran.append("a")
            return x

        sub = _ff().create().with_checkpointer().call(a)
        flow = _ff(memory_store()).create().call(a).call(sub)
        with pytest.raises(RuntimeError, match="has no history"):
            await flow.run(1)
        assert ran == []

    async def test_a_history_must_name_a_repository(self) -> None:
        from appinfra.cas import InvalidNameError

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            return x

        store = memory_store()
        flow = _ff(store).create().with_checkpointer().call(a)
        with pytest.raises(InvalidNameError):
            await flow.run(1, history="..")
        assert store.list_repos() == []

    @pytest.mark.parametrize("bad", ["latest", "complete", "a" * 64, ""])
    async def test_checkpointer_name_follows_checkpoint_name_rules(self, bad: str) -> None:
        with pytest.raises(ValueError):
            _ff().create().with_checkpointer(bad)


class TestNamedCheckpointer:
    async def test_named_map_body_resumes_at_its_latest_item_checkpoint(self) -> None:
        store = memory_store()
        ran: list[int] = []

        def flow() -> Any:
            @verb
            async def item(ctx: Context[Any], n: int) -> int:
                ran.append(n)
                await ctx.checkpoint()
                return n * 2

            return (
                _ff(store)
                .create()
                .map(
                    lambda b: b.with_checkpointer("items").call(item),
                    items=lambda _p, _c: [1, 2, 3],
                    max_concurrency=1,
                )
            )

        assert await flow().run(history=NAME) == [2, 4, 6]
        history = History(store, NAME)
        tagged = await history.checkpoint("items")
        assert tagged is not None
        cursors = (await history.snapshot(tagged)).cursors
        [done] = [c["done"] for c in cursors.values() if "done" in c]
        assert sorted(done) == ["0", "1"]  # the third item's own checkpoint: items 0 and 1 done

        ran.clear()
        assert await flow().run(history=NAME, resume="items") == [2, 4, 6]
        assert ran == [3]

    async def test_a_save_moves_only_its_innermost_checkpointer_tag(self) -> None:
        store = memory_store()

        @verb
        async def inner_step(ctx: Context[Any], x: int) -> int:
            await ctx.checkpoint()
            return x

        @verb
        async def outer_step(ctx: Context[Any], x: int) -> int:
            return x

        inner = _ff().create().with_checkpointer("inner").call(inner_step)
        outer = _ff().create().with_checkpointer("outer").call(outer_step).call(inner)
        await _ff(store).create().call(outer).run(1, history=NAME)
        assert await History(store, NAME).checkpoint_names() == ["inner"]

    async def test_one_name_on_two_flows_is_one_tag(self) -> None:
        store = memory_store()

        @verb
        async def save(ctx: Context[Any], x: int) -> int:
            await ctx.checkpoint()
            return x + 1

        first = _ff().create().with_checkpointer("phase").call(save)
        second = _ff().create().with_checkpointer("phase").call(save)
        flow = _ff(store).create().call(first).call(second)
        assert await flow.run(1, history=NAME) == 3
        history = History(store, NAME)
        assert await history.checkpoint_names() == ["phase"]
        tagged = await history.checkpoint("phase")
        line = [c.id async for c in history.commits()]
        # The tag is on the later save (second's): the newest "ok" commit on the line.
        assert tagged is not None and tagged.id == line[1]
