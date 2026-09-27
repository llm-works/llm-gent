# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Unit tests for :class:`llm_gent.flow.History` — the read API over one history."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from llm_gent.flow import History, HistoryCorrupt, TypeStateFactory
from llm_gent.flow.state.cas import Commit
from llm_gent.flow.stores import JsonFileCheckpointStore
from llm_gent.flow.testing.checkpoint import CanonicalCounter, build_canonical_flow

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


@pytest.fixture
def store(tmp_path: Path) -> JsonFileCheckpointStore:
    """A fresh store under an isolated temp root."""
    return JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")


async def _run(store: JsonFileCheckpointStore, name: str, *, halt_at: int | None = None) -> None:
    """Run the canonical flow (4 iterations) under ``name``; halt after ``halt_at`` if given."""
    halt = asyncio.Event() if halt_at is not None else None
    await build_canonical_flow(
        make_test_logger(),
        max_iters=4,
        halt=halt,
        halt_after_iteration=halt_at,
        store=store,
        client_flow_id=name,
    ).run(resume="replay")


async def _all(history: History) -> list[Commit]:
    return [c async for c in history.commits()]


class TestEmptyHistory:
    async def test_unknown_name_reads_as_empty(self, store: JsonFileCheckpointStore) -> None:
        history = History(store, "nothing-here")
        assert await history.flow_id() is None
        assert await history.head() is None
        assert await history.last_complete() is None
        assert not await history.is_complete()
        assert await history.replay_point() is None
        assert await _all(history) == []


class TestCompletedHistory:
    async def test_head_is_the_tagged_final_state(self, store: JsonFileCheckpointStore) -> None:
        await _run(store, "done")
        history = History(store, "done")

        head = await history.head()
        assert head is not None
        assert History.is_final_state(head)
        assert await history.is_complete()
        assert await history.replay_point() is None
        assert await history.last_complete() == head
        assert head.meta.flow_id == await history.flow_id()

    async def test_root_state_restores_through_the_factory(
        self, store: JsonFileCheckpointStore
    ) -> None:
        await _run(store, "state")
        history = History(store, "state")
        head = await history.head()
        assert head is not None

        state = await history.root_state(head, TypeStateFactory(CanonicalCounter))
        assert isinstance(state, CanonicalCounter)
        assert state.iterations_completed == 4
        scopes = await history.scopes(head)
        assert scopes is not None and scopes[0] == state.to_dict()


class TestCorruptHistory:
    """A hash pointing at a missing object raises instead of reading as a shorter history."""

    def _objects(self, store: JsonFileCheckpointStore, flow_id: str, kind: str) -> Path:
        return store._history_dir(flow_id) / "objects" / kind

    async def test_missing_parent_raises_in_walk(self, store: JsonFileCheckpointStore) -> None:
        await _run(store, "torn-chain")
        history = History(store, "torn-chain")
        head = await history.head()
        assert head is not None and head.parent_hashes
        flow_id = head.meta.flow_id
        (self._objects(store, flow_id, "commit") / head.parent_hashes[0]).unlink()

        with pytest.raises(HistoryCorrupt) as err:
            await _all(history)
        assert (err.value.kind, err.value.content_hash) == ("commit", head.parent_hashes[0])

    async def test_missing_blob_raises_in_scopes(self, store: JsonFileCheckpointStore) -> None:
        await _run(store, "torn-state")
        history = History(store, "torn-state")
        head = await history.head()
        assert head is not None
        for blob in self._objects(store, head.meta.flow_id, "blob").iterdir():
            blob.unlink()

        with pytest.raises(HistoryCorrupt, match="blob"):
            await history.scopes(head)

    async def test_ref_to_missing_commit_raises(self, store: JsonFileCheckpointStore) -> None:
        await _run(store, "torn-head")
        history = History(store, "torn-head")
        head = await history.head()
        assert head is not None
        (self._objects(store, head.meta.flow_id, "commit") / head.content_hash).unlink()

        with pytest.raises(HistoryCorrupt):
            await history.head()

    async def test_resume_on_corrupt_history_warns_and_starts_fresh(
        self, store: JsonFileCheckpointStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await _run(store, "torn-resume", halt_at=2)
        head = await History(store, "torn-resume").head()
        assert head is not None
        (self._objects(store, head.meta.flow_id, "commit") / head.content_hash).unlink()

        lg = make_test_logger()
        warnings: list[str] = []
        monkeypatch.setattr(lg, "warning", lambda msg, *_a, **_kw: warnings.append(msg))
        result = await build_canonical_flow(
            lg,
            state=CanonicalCounter(n=100),
            max_iters=4,
            store=store,
            client_flow_id="torn-resume",
        ).run(resume="replay")

        assert result["log"][0] == 101  # fresh from the fallback state, not resumed
        assert any("corrupt" in w for w in warnings)
        # The fresh run's commits form a new, fully readable chain.
        history = History(store, "torn-resume")
        chain = await _all(history)
        assert chain[0].meta.node_path == "$end"
        assert chain[-1].parent_hashes == ()
        assert {c.meta.flow_id for c in chain} == {await history.flow_id()}


class TestHistoryAcrossRuns:
    async def test_halt_after_completion(self, store: JsonFileCheckpointStore) -> None:
        """A halted follow-up run: head moves on, last_complete stays, the chain links both."""
        await _run(store, "sessions")
        first_end = await History(store, "sessions").head()
        await _run(store, "sessions", halt_at=2)
        history = History(store, "sessions")

        head = await history.head()
        assert head is not None and head.meta.outcome == "halted"
        assert not await history.is_complete()
        assert await history.replay_point() == head
        assert await history.last_complete() == first_end
        chain = await _all(history)
        assert chain[0] == head
        assert first_end in chain
        assert chain[-1].parent_hashes == ()

    async def test_commits_walk_every_stored_commit(self, store: JsonFileCheckpointStore) -> None:
        await _run(store, "walk", halt_at=2)
        await _run(store, "walk")
        history = History(store, "walk")
        flow_id = await history.flow_id()
        assert flow_id is not None

        stored = {f.name for f in (store._history_dir(flow_id) / "objects" / "commit").iterdir()}
        assert {c.content_hash for c in await _all(history)} == stored
