# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Named checkpoints — ``ctx.checkpoint(name)`` and ``run(resume=name)``.

A named checkpoint is the tag ``tags/<name>`` on the commit
``ctx.checkpoint(name)`` wrote. ``run(resume=name)`` checks that commit out
like ``resume="latest"`` checks out the newest one, and moves ``HEAD`` back
to it: the run's commits continue from there, and the commits written after
the checkpoint leave the history's line.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_gent.flow import Context, FlowFactory, History, verb
from llm_gent.flow.checkpoint import HEAD_REF
from llm_gent.flow.stores import InMemoryCheckpointStore

from .conftest import flow_id_for, make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]

NAME = "named"


def _flow(store: Any, calls: list[str], tag: str | None = "mid") -> Any:
    """``a → b → c``; ``b`` takes the named checkpoint ``tag`` before its work."""

    @verb
    async def a(ctx: Context[dict[str, Any]], x: int) -> int:
        calls.append("a")
        ctx.state.data["a"] = x
        return x + 1

    @verb
    async def b(ctx: Context[dict[str, Any]], x: int) -> int:
        calls.append("b")
        if tag is not None:
            await ctx.checkpoint(tag)
        ctx.state.data["b"] = x
        return x * 10

    @verb
    async def c(ctx: Context[dict[str, Any]], x: int) -> int:
        calls.append("c")
        ctx.state.data["c"] = x
        return x + 3

    return (
        FlowFactory(make_test_logger())
        .create(state={})
        .with_checkpointer(store, NAME)
        .call(a)
        .then(b)
        .then(c)
    )


class TestNamedCheckpoint:
    async def test_checkpoint_name_tags_its_commit(self) -> None:
        store = InMemoryCheckpointStore()
        await _flow(store, []).run(1)

        commit = await History(store, NAME).checkpoint("mid")
        assert commit is not None and commit.meta.outcome == "ok"
        snapshot = await History(store, NAME).snapshot(commit)
        assert snapshot.root == {"a": 1}
        assert snapshot.cursors[""]["chain"]["args"] == [2]

    async def test_resume_by_name_continues_at_the_checkpointed_step(self) -> None:
        store = InMemoryCheckpointStore()
        await _flow(store, []).run(1)

        calls: list[str] = []
        assert await _flow(store, calls).run(resume="mid") == 23
        assert calls == ["b", "c"]
        history = History(store, NAME)
        assert await history.is_complete()
        head = await history.head()
        assert head is not None
        assert (await history.snapshot(head)).root == {"a": 1, "b": 2, "c": 20}

    async def test_resume_by_name_resets_head_to_the_checkpoint(self) -> None:
        """The run's commits parent the checkpoint; later commits leave the history's line."""
        store = InMemoryCheckpointStore()
        await _flow(store, []).run(1)
        history = History(store, NAME)
        tagged = await history.checkpoint("mid")
        first_end = await history.head()
        assert tagged is not None and first_end is not None

        # The checkpoint step does not tag again, so the tag stays on run 1's commit.
        await _flow(store, [], tag=None).run(resume="mid")

        line = [commit.content_hash async for commit in history.commits()]
        assert tagged.content_hash in line
        assert first_end.content_hash not in line
        head = await history.head()
        assert head is not None and head.parent_hashes == (tagged.content_hash,)

    async def test_taking_the_checkpoint_again_moves_the_tag(self) -> None:
        store = InMemoryCheckpointStore()
        await _flow(store, []).run(1)
        history = History(store, NAME)
        before = await history.checkpoint("mid")

        await _flow(store, []).run(resume="mid")

        after = await history.checkpoint("mid")
        assert before is not None and after is not None
        assert after.content_hash != before.content_hash
        assert after.parent_hashes == (before.content_hash,)


class TestNamedCheckpointErrors:
    async def test_unknown_name_raises_and_writes_nothing(self) -> None:
        store = InMemoryCheckpointStore()
        await _flow(store, []).run(1)
        head = store.get_ref(flow_id_for(store, NAME), HEAD_REF)

        with pytest.raises(ValueError, match="no checkpoint named 'nope'"):
            await _flow(store, []).run(resume="nope")
        assert store.get_ref(flow_id_for(store, NAME), HEAD_REF) == head

    @pytest.mark.parametrize("name", ["off", "latest", "complete", ""])
    async def test_reserved_names_cannot_be_taken(self, name: str) -> None:
        with pytest.raises(ValueError, match="checkpoint"):
            await _flow(InMemoryCheckpointStore(), [], tag=name).run(1)

    async def test_resume_complete_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            await _flow(InMemoryCheckpointStore(), []).run(resume="complete")

    async def test_resume_by_name_requires_a_checkpointer(self) -> None:
        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            return x

        flow = FlowFactory(make_test_logger()).create().call(a)
        with pytest.raises(RuntimeError, match="no checkpointer"):
            await flow.run(1, resume="mid")


class TestNamedCheckpointInIterate:
    async def test_resume_continues_in_the_checkpointed_pass(self) -> None:
        """A checkpoint in pass 2 resumes in pass 2 with its carried value, not from pass 0."""
        store = InMemoryCheckpointStore()
        passes: list[int] = []

        def build() -> Any:
            @verb
            async def step(ctx: Context[dict[str, Any]], x: int) -> int:
                passes.append(x)
                if x == 3:
                    await ctx.checkpoint("third")
                return x + 1

            return (
                FlowFactory(make_test_logger())
                .create(state={})
                .with_checkpointer(store, NAME)
                .iterate(lambda body: body.call(step), max_iters=4)
            )

        assert await build().run(1) == 5
        passes.clear()
        assert await build().run(resume="third") == 5
        assert passes == [3, 4]
