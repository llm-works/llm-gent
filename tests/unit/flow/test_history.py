# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Unit tests for :class:`llm_gent.flow.History` — the read API over one history."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from appinfra.cas import FileStore

from llm_gent.flow import FlowCommit, History, HistoryCorrupt, TypeStateFactory
from llm_gent.flow.testing.checkpoint import CanonicalCounter, build_canonical_flow
from tests.helpers import file_store

from .conftest import commit_ids, make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


@pytest.fixture
def store(tmp_path: Path) -> FileStore:
    """A fresh store under an isolated temp root."""
    return file_store(tmp_path / "cp")


async def _run(store: FileStore, name: str, *, halt_at: int | None = None) -> None:
    """Run the canonical flow (4 iterations) under ``name``; halt after ``halt_at`` if given."""
    halt = asyncio.Event() if halt_at is not None else None
    await build_canonical_flow(
        make_test_logger(),
        max_iters=4,
        halt=halt,
        halt_after_iteration=halt_at,
        store=store,
        client_flow_id=name,
    ).run(resume="latest")


async def _all(history: History) -> list[FlowCommit]:
    return [c async for c in history.commits()]


class TestEmptyHistory:
    async def test_unknown_name_reads_as_empty(self, store: FileStore) -> None:
        history = History(store, "nothing-here")
        assert await history.head() is None
        assert await history.last_complete() is None
        assert not await history.is_complete()
        assert await _all(history) == []


class TestCompletedHistory:
    async def test_head_is_the_tagged_final_state(self, store: FileStore) -> None:
        await _run(store, "done")
        history = History(store, "done")

        head = await history.head()
        assert head is not None
        assert History.is_final_state(head)
        assert await history.is_complete()
        assert await history.last_complete() == head

    async def test_root_state_restores_through_the_factory(self, store: FileStore) -> None:
        await _run(store, "state")
        history = History(store, "state")
        head = await history.head()
        assert head is not None

        state = await history.root_state(head, TypeStateFactory(CanonicalCounter))
        assert isinstance(state, CanonicalCounter)
        assert state.iterations_completed == 4
        snapshot = await history.snapshot(head)
        assert snapshot.has_state and snapshot.root == state.to_dict()


class TestCorruptHistory:
    """A hash pointing at a missing object raises instead of reading as a shorter history."""

    async def test_missing_parent_raises_in_walk(self, store: FileStore) -> None:
        await _run(store, "torn-chain")
        history = History(store, "torn-chain")
        head = await history.head()
        assert head is not None and head.parents
        store.delete_objects("torn-chain", [head.parents[0]])

        with pytest.raises(HistoryCorrupt) as err:
            await _all(history)
        assert (err.value.kind, err.value.object_id) == ("commit", head.parents[0])

    async def test_missing_blob_raises_in_snapshot(self, store: FileStore) -> None:
        await _run(store, "torn-state")
        history = History(store, "torn-state")
        head = await history.head()
        assert head is not None
        blobs = [
            i
            for i in store.list_objects("torn-state")
            if store.object_type("torn-state", i) == "blob"
        ]
        store.delete_objects("torn-state", blobs)

        with pytest.raises(HistoryCorrupt, match="blob"):
            await history.snapshot(head)

    async def test_ref_to_missing_commit_raises(self, store: FileStore) -> None:
        await _run(store, "torn-head")
        history = History(store, "torn-head")
        head = await history.head()
        assert head is not None
        store.delete_objects("torn-head", [head.id])

        with pytest.raises(HistoryCorrupt):
            await history.head()

    async def test_replay_on_corrupt_history_raises(self, store: FileStore) -> None:
        """Replay stops on store damage instead of starting over from ``state=``."""
        await _run(store, "torn-resume", halt_at=2)
        head = await History(store, "torn-resume").head()
        assert head is not None
        store.delete_objects("torn-resume", [head.id])

        flow = build_canonical_flow(
            make_test_logger(),
            state=CanonicalCounter(n=100),
            max_iters=4,
            store=store,
            client_flow_id="torn-resume",
        )
        with pytest.raises(HistoryCorrupt):
            await flow.run(resume="latest")
        # Nothing was written past the damage: the ref still names the missing commit.
        with pytest.raises(HistoryCorrupt):
            await History(store, "torn-resume").head()


class TestHistoryAcrossRuns:
    async def test_halt_after_completion(self, store: FileStore) -> None:
        """A halted follow-up run: head moves on, last_complete stays, the chain links both."""
        await _run(store, "sessions")
        first_end = await History(store, "sessions").head()
        await _run(store, "sessions", halt_at=2)
        history = History(store, "sessions")

        head = await history.head()
        assert head is not None and head.meta.outcome == "halted"
        assert not await history.is_complete()
        assert await history.last_complete() == first_end
        chain = await _all(history)
        assert chain[0] == head
        assert first_end in chain
        assert chain[-1].parents == ()

    async def test_commits_walk_every_stored_commit(self, store: FileStore) -> None:
        await _run(store, "walk", halt_at=2)
        await _run(store, "walk")
        history = History(store, "walk")
        assert {c.id for c in await _all(history)} == commit_ids(store, "walk")
