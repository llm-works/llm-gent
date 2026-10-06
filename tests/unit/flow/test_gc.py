# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""``collect_unreachable`` — deletes what no ref reaches, keeps the history intact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llm_gent.flow import Context, Factory, History, HistoryCorrupt, collect_unreachable, verb
from llm_gent.flow.state.cas import Blob, Commit, Tree, TreeEntry
from llm_gent.flow.stores import InMemoryCheckpointStore, JsonFileCheckpointStore

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]

NAME = "gc"


@pytest.fixture(params=["mem", "file"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    if request.param == "mem":
        return InMemoryCheckpointStore()
    return JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")


def _flow(store: Any, tag: str | None) -> Any:
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


async def _keys(store: Any) -> set[tuple[str, str]]:
    flow_id = await History(store, NAME).flow_id()
    assert flow_id is not None
    return set(store.list_objects(flow_id))


async def _readable(history: History) -> None:
    """Every commit on the line, and every named ref, loads with its snapshot."""
    async for commit in history.commits():
        await history.snapshot(commit)
    last = await history.last_complete()
    assert last is not None
    await history.snapshot(last)


class TestCollectUnreachable:
    async def test_collects_what_a_named_reset_left_behind(self, store: Any) -> None:
        await _flow(store, "mid").run(1)
        history = History(store, NAME)
        first_end = await history.head()
        assert first_end is not None
        await _flow(store, None).run(resume="mid")

        removed = await collect_unreachable(store, NAME)
        assert removed > 0
        flow_id = await history.flow_id()
        assert store.get_object(flow_id, "commit", first_end.content_hash) is None
        await _readable(history)
        assert await history.checkpoint("mid") is not None
        # The history finished: latest starts again from its first step.
        assert await _flow(store, None).run(1, resume="latest") == 23
        assert await collect_unreachable(store, NAME) == 0

    async def test_collects_the_objects_of_a_commit_head_never_moved_to(self, store: Any) -> None:
        await _flow(store, None).run(1)
        before = await _keys(store)
        flow_id = await History(store, NAME).flow_id()
        blob = Blob.from_bytes(b'"orphan"')
        tree = Tree.from_entries(
            [TreeEntry(scope_id="state", kind="blob", child_hash=blob.content_hash)]
        )
        head = await History(store, NAME).head()
        assert head is not None
        commit = Commit.build(
            root_tree_hash=tree.content_hash, parent_hashes=(head.content_hash,), meta=head.meta
        )
        store.put_object(flow_id, "blob", blob.content_hash, blob.payload)
        store.put_object(flow_id, "tree", tree.content_hash, tree.to_bytes())
        store.put_object(flow_id, "commit", commit.content_hash, commit.to_bytes())

        assert await collect_unreachable(store, NAME) == 3
        assert await _keys(store) == before

    async def test_a_history_without_garbage_loses_nothing(self, store: Any) -> None:
        await _flow(store, "mid").run(1)
        before = await _keys(store)
        assert await collect_unreachable(store, NAME) == 0
        assert await _keys(store) == before
        await _readable(History(store, NAME))

    async def test_a_corrupt_history_raises_and_deletes_nothing(self, store: Any) -> None:
        await _flow(store, None).run(1)
        history = History(store, NAME)
        flow_id = await history.flow_id()
        store.put_object(flow_id, "blob", "orphan", b"x")
        head = await history.head()
        assert head is not None
        store.delete_objects(flow_id, [("tree", head.root_tree_hash)])

        with pytest.raises(HistoryCorrupt):
            await collect_unreachable(store, NAME)
        assert store.get_object(flow_id, "blob", "orphan") == b"x"

    async def test_unknown_name_collects_nothing(self, store: Any) -> None:
        assert await collect_unreachable(store, "never-existed") == 0
