# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Pruning a history's repository deletes what no ref reaches and keeps the history intact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from appinfra.cas import Blob, ObjectNotFoundError, Store, Tree, TreeEntry

from llm_gent.flow import Context, Factory, FlowCommit, History, verb
from tests.helpers import file_store, memory_store

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]

NAME = "gc"


@pytest.fixture(params=["mem", "file"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Store:
    if request.param == "mem":
        return memory_store()
    return file_store(tmp_path / "cp")


def _flow(store: Store, tag: str | None) -> Any:
    """``a -> b -> c``; ``b`` takes the named checkpoint ``tag`` (when given) before its work."""

    @verb
    async def a(ctx: Context[dict[str, Any]], x: int) -> int:
        ctx.state.data["a"] = x
        return x + 1

    @verb
    async def b(ctx: Context[dict[str, Any]], x: int) -> int:
        if tag is not None:
            await ctx.checkpoint(tag)
        ctx.state.data["b"] = x
        return x * 10

    @verb
    async def c(ctx: Context[dict[str, Any]], x: int) -> int:
        ctx.state.data["c"] = x
        return x + 3

    return (
        Factory(make_test_logger())
        .create(state={})
        .with_checkpoint_store(store, NAME)
        .with_checkpointer()
        .call(a)
        .then(b)
        .then(c)
    )


async def _readable(history: History) -> None:
    """Every commit on the line, and every named ref, loads with its snapshot."""
    async for commit in history.commits():
        await history.snapshot(commit)
    last = await history.last_complete()
    assert last is not None
    await history.snapshot(last)


class TestPrune:
    async def test_prunes_what_a_named_reset_left_behind(self, store: Store) -> None:
        await _flow(store, "mid").run(1)
        history = History(store, NAME)
        first_end = await history.head()
        assert first_end is not None
        await _flow(store, None).run(resume="mid")

        repo = store.repo(NAME)
        assert repo.prune(0) > 0
        assert first_end.id not in repo
        await _readable(history)
        assert await history.checkpoint("mid") is not None
        # The history finished: latest starts again from its first step.
        assert await _flow(store, None).run(1, resume="latest") == 23
        assert repo.prune(0) == 0

    async def test_prunes_the_objects_of_a_commit_head_never_moved_to(self, store: Store) -> None:
        """A process that died before moving HEAD left a commit: kept while recent, then pruned."""
        await _flow(store, None).run(1)
        repo = store.repo(NAME)
        before = set(store.list_objects(NAME))
        head = await History(store, NAME).head()
        assert head is not None
        blob = repo.put(Blob(b'"orphan"'))
        tree = repo.put(Tree((TreeEntry("state", "blob", blob),)))
        repo.put(FlowCommit.build(tree, (head.id,), head.meta)[1])

        assert repo.prune(3600) == 0
        assert repo.prune(0) == 3
        assert set(store.list_objects(NAME)) == before

    async def test_a_history_without_garbage_loses_nothing(self, store: Store) -> None:
        await _flow(store, "mid").run(1)
        before = set(store.list_objects(NAME))
        assert store.repo(NAME).prune(0) == 0
        assert set(store.list_objects(NAME)) == before
        await _readable(History(store, NAME))

    async def test_a_corrupt_history_raises_and_deletes_nothing(self, store: Store) -> None:
        await _flow(store, None).run(1)
        repo = store.repo(NAME)
        orphan = repo.put(Blob(b"x"))
        head = await History(store, NAME).head()
        assert head is not None
        store.delete_objects(NAME, [head.tree])

        with pytest.raises(ObjectNotFoundError):
            repo.prune(0)
        assert orphan in repo

    async def test_unknown_name_prunes_nothing(self, store: Store) -> None:
        assert store.repo("never-existed").prune(0) == 0
