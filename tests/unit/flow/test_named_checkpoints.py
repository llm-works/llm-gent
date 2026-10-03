# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Named checkpoints and resume by hash — ``ctx.checkpoint(name)``, ``run(resume=name|hash)``.

A named checkpoint is the tag ``tags/<name>`` on the commit
``ctx.checkpoint(name)`` wrote. ``run(resume=name)`` checks that commit out
like ``resume="latest"`` checks out the newest one, and the run's commits
continue from there: once it commits, the commits written after the
checkpoint leave the history's line; a run that fails first leaves ``HEAD``
where it was. ``run(resume=<hash>)`` does the same for any commit,
including one left off the line by a later resume.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_gent.flow import HALTED, Context, FlowFactory, History, verb
from llm_gent.flow.checkpoint import HEAD_REF
from llm_gent.flow.stores import InMemoryCheckpointStore

from .conftest import flow_id_for, make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]

NAME = "named"


class _Crash(Exception):
    pass


def _flow(store: Any, calls: list[str], tag: str | None = "mid", fail: bool = False) -> Any:
    """``a → b → c``; ``b`` takes the named checkpoint ``tag`` before its work.

    With ``fail``, ``b`` raises :class:`_Crash` after the checkpoint (before
    any when ``tag`` is ``None``).
    """

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
        if fail:
            raise _Crash()
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
        .with_checkpoint_store(store, NAME)
        .with_checkpointer()
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

    async def test_a_run_failing_before_its_first_commit_leaves_head(self) -> None:
        """Nothing was committed, so the history is as it was: ``latest`` still sees run 1's end."""
        store = InMemoryCheckpointStore()
        await _flow(store, []).run(1)
        history = History(store, NAME)
        first_end = await history.head()
        assert first_end is not None

        with pytest.raises(_Crash):
            await _flow(store, [], tag=None, fail=True).run(resume="mid")

        head = await history.head()
        assert head is not None and head.content_hash == first_end.content_hash

    async def test_a_removed_step_leaves_head(self) -> None:
        """The flow no longer has the checkpointed step: the resume fails and HEAD stays."""
        store = InMemoryCheckpointStore()
        await _flow(store, []).run(1)
        history = History(store, NAME)
        first_end = await history.head()
        assert first_end is not None

        @verb
        async def other(ctx: Context[dict[str, Any]], x: int) -> int:
            return x

        changed = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpoint_store(store, NAME)
            .with_checkpointer()
            .call(other)
        )
        with pytest.raises(RuntimeError, match="which this chain no longer has"):
            await changed.run(resume="mid")

        head = await history.head()
        assert head is not None and head.content_hash == first_end.content_hash

    async def test_a_run_failing_after_a_commit_has_moved_head(self) -> None:
        """Its commit is parented on the checkpoint and is the head; the old line is left."""
        store = InMemoryCheckpointStore()
        await _flow(store, []).run(1)
        history = History(store, NAME)
        tagged = await history.checkpoint("mid")
        first_end = await history.head()
        assert tagged is not None and first_end is not None

        with pytest.raises(_Crash):
            await _flow(store, [], fail=True).run(resume="mid")

        head = await history.head()
        assert head is not None and head.parent_hashes == (tagged.content_hash,)
        line = [commit.content_hash async for commit in history.commits()]
        assert first_end.content_hash not in line

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

        with pytest.raises(ValueError, match=r"no checkpoint named 'nope'; checkpoints: \['mid'\]"):
            await _flow(store, []).run(resume="nope")
        assert store.get_ref(flow_id_for(store, NAME), HEAD_REF) == head

    @pytest.mark.parametrize("name", ["off", "latest", "complete", "", "a" * 64])
    async def test_reserved_names_cannot_be_taken(self, name: str) -> None:
        with pytest.raises(ValueError, match="checkpoint"):
            await _flow(InMemoryCheckpointStore(), [], tag=name).run(1)

    async def test_resume_complete_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            await _flow(InMemoryCheckpointStore(), []).run(resume="complete")

    async def test_resume_by_name_requires_a_checkpoint_store(self) -> None:
        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            return x

        flow = FlowFactory(make_test_logger()).create().call(a)
        with pytest.raises(RuntimeError, match="no checkpoint store"):
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
                .with_checkpoint_store(store, NAME)
                .with_checkpointer()
                .iterate(lambda body: body.call(step), max_iters=4)
            )

        assert await build().run(1) == 5
        passes.clear()
        assert await build().run(resume="third") == 5
        assert passes == [3, 4]


def _hash_flow(
    store: Any,
    calls: list[str],
    hashes: list[str | None],
    *,
    take: bool,
    halt: asyncio.Event | None,
) -> Any:
    """``a → b → c → d``; with ``take``, ``a`` checkpoints unnamed, ``b`` as ``"y"``.

    With ``halt``, ``c`` sets it: the run stops before ``d``. Every hash
    ``ctx.checkpoint`` returns lands in ``hashes``.
    """

    @verb
    async def a(ctx: Context[dict[str, Any]], x: int) -> int:
        calls.append("a")
        if take:
            hashes.append(await ctx.checkpoint())
        ctx.state.data["a"] = x
        return x + 1

    @verb
    async def b(ctx: Context[dict[str, Any]], x: int) -> int:
        calls.append("b")
        if take:
            hashes.append(await ctx.checkpoint("y"))
        ctx.state.data["b"] = x
        return x * 10

    @verb
    async def c(ctx: Context[dict[str, Any]], x: int) -> int:
        calls.append("c")
        if halt is not None:
            halt.set()
        ctx.state.data["c"] = x
        return x + 3

    @verb
    async def d(ctx: Context[dict[str, Any]], x: int) -> int:
        calls.append("d")
        ctx.state.data["d"] = x
        return x - 1

    flow = (
        FlowFactory(make_test_logger())
        .create(state={})
        .with_checkpoint_store(store, NAME)
        .with_checkpointer()
    )
    if halt is not None:
        flow = flow.with_halt(halt)
    return flow.call(a).then(b).then(c).then(d)


class TestResumeByHash:
    async def test_checkpoint_returns_the_hash_it_wrote(self) -> None:
        store = InMemoryCheckpointStore()
        seen: list[tuple[str | None, str | None]] = []

        @verb
        async def a(ctx: Context[dict[str, Any]], x: int) -> int:
            written = await ctx.checkpoint()
            head = await History(store, NAME).head()
            seen.append((written, None if head is None else head.content_hash))
            return x

        flow = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpoint_store(store, NAME)
            .with_checkpointer()
        )
        await flow.call(a).run(1)
        assert len(seen) == 1 and seen[0][0] is not None and seen[0][0] == seen[0][1]

    async def test_checkpoint_without_a_checkpointer_returns_none(self) -> None:
        seen: list[str | None] = []

        @verb
        async def a(ctx: Context[dict[str, Any]], x: int) -> int:
            seen.append(await ctx.checkpoint("mid"))
            return x

        await FlowFactory(make_test_logger()).create(state={}).call(a).run(1)
        assert seen == [None]

    async def test_any_commit_stays_resumable_after_a_reset(self) -> None:
        """Unnamed checkpoint, then the halted head off the line, then a name: each resumes."""
        store = InMemoryCheckpointStore()
        history = History(store, NAME)
        calls: list[str] = []
        hashes: list[str | None] = []
        first = _hash_flow(store, calls, hashes, take=True, halt=asyncio.Event())
        assert await first.run(1) is HALTED
        h1 = hashes[0]
        assert h1 is not None
        halted = await history.head()
        assert halted is not None and halted.meta.outcome == "halted"

        calls.clear()
        assert await _hash_flow(store, calls, [], take=False, halt=None).run(resume=h1) == 22
        assert calls == ["a", "b", "c", "d"]
        line = [commit.content_hash async for commit in history.commits()]
        assert halted.content_hash not in line

        calls.clear()
        assert (
            await _hash_flow(store, calls, [], take=False, halt=None).run(
                resume=halted.content_hash
            )
            == 22
        )
        assert calls == ["d"]
        head = await history.head()
        assert head is not None and head.parent_hashes == (halted.content_hash,)
        assert (await history.snapshot(head)).root == {"a": 1, "b": 2, "c": 20, "d": 23}

        calls.clear()
        assert await _hash_flow(store, calls, [], take=False, halt=None).run(resume="y") == 22
        assert calls == ["b", "c", "d"]

    async def test_unknown_hash_raises_and_writes_nothing(self) -> None:
        store = InMemoryCheckpointStore()
        await _flow(store, []).run(1)
        head = store.get_ref(flow_id_for(store, NAME), HEAD_REF)

        with pytest.raises(ValueError, match=f"has no commit {'0' * 64}"):
            await _flow(store, []).run(resume="0" * 64)
        assert store.get_ref(flow_id_for(store, NAME), HEAD_REF) == head

    async def test_commit_without_state_is_refused(self) -> None:
        """A final state that could not be serialized left a stateless commit: nothing to resume."""
        store = InMemoryCheckpointStore()

        @verb
        async def a(ctx: Context[dict[str, Any]], x: int) -> int:
            ctx.state.data["handle"] = object()
            return x

        flow = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpoint_store(store, NAME)
            .with_checkpointer()
        )
        await flow.call(a).run(1)
        head = await History(store, NAME).head()
        assert head is not None

        with pytest.raises(ValueError, match="holds no state"):
            await _flow(store, []).run(resume=head.content_hash)


class TestHistoryReads:
    async def test_checkpoint_names_are_sorted_without_complete(self) -> None:
        store = InMemoryCheckpointStore()
        await _flow(store, [], tag="zeta").run(1)
        await _flow(store, [], tag="alpha").run(1)
        assert await History(store, NAME).checkpoint_names() == ["alpha", "zeta"]

    async def test_reads_on_a_missing_history(self) -> None:
        history = History(InMemoryCheckpointStore(), "never-existed")
        assert await history.checkpoint_names() == []
        assert await history.commit("0" * 64) is None

    async def test_commit_reads_a_commit_off_the_line(self) -> None:
        store = InMemoryCheckpointStore()
        await _flow(store, []).run(1)
        history = History(store, NAME)
        first_end = await history.head()
        assert first_end is not None
        await _flow(store, [], tag=None).run(resume="mid")

        found = await history.commit(first_end.content_hash)
        assert found is not None and found.content_hash == first_end.content_hash
        assert await history.commit("0" * 64) is None
