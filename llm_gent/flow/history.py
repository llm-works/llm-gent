# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""History — read API over one flow history in a :class:`CheckpointStore`.

A history is the chain of commits one flow instance writes, addressed by
the agent's ``client_flow_id``. :class:`History` resolves the name to the
internal ``flow_id`` and exposes the history as data: its head, the last
completed run, the commit chain, and the state each commit holds. It
never writes; the framework owns every write.

``None`` means absent — no history under the name, no commit yet, no
completed run. A hash the history holds (ref, tag, parent, tree entry)
that points at an object missing from the store raises
:class:`HistoryCorrupt` instead, so a damaged history never reads as a
shorter or empty one.

Every method works with sync and async stores alike.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TypeVar

from .checkpoint import (
    COMPLETE_TAG,
    END_NODE_PATH,
    FAILED_NODE_PATH,
    HEAD_REF,
    CheckpointStore,
    Kind,
    maybe_await,
)
from .state import StateFactory, restore_state_data
from .state.cas import Commit
from .state.snapshot import Snapshot, read_snapshot


T = TypeVar("T")


class HistoryCorrupt(ValueError):
    """A hash in the history points at an object the store does not hold."""

    def __init__(self, flow_id: str, kind: Kind, content_hash: str) -> None:
        super().__init__(f"history {flow_id}: {kind} {content_hash} is missing from the store")
        self.flow_id = flow_id
        self.kind = kind
        self.content_hash = content_hash


class History:
    """Read-only view of the history named ``client_flow_id`` in ``store``.

    Holds no cached state: every call reads the store, so a view built once
    observes later writes.
    """

    def __init__(self, store: CheckpointStore, client_flow_id: str) -> None:
        self.store = store
        self.client_flow_id = client_flow_id

    async def flow_id(self) -> str | None:
        """Internal id of the history, or ``None`` when none exists under the name."""
        result: str | None = await maybe_await(self.store.get_flow_id(self.client_flow_id))
        return result

    async def head(self) -> Commit | None:
        """Latest commit, or ``None`` for an empty history."""
        flow_id = await self.flow_id()
        if flow_id is None:
            return None
        commit_hash = await maybe_await(self.store.get_ref(flow_id, HEAD_REF))
        return None if commit_hash is None else await self._commit(flow_id, commit_hash)

    async def is_complete(self) -> bool:
        """True when the last run finished: the head is its final-state commit.

        A history whose head is a ``$failed`` commit is not complete, yet
        replay may still start fresh on it — :meth:`replay_point` is the
        rule ``run(resume="replay")`` applies.
        """
        head = await self.head()
        return head is not None and self.is_final_state(head)

    async def replay_point(self) -> Commit | None:
        """Commit ``run(resume="replay")`` resumes from, or ``None`` for a fresh run.

        The newest commit that is not a ``$failed`` commit (the state at a
        failure may be half-updated); ``None`` when that commit is a
        final-state commit (the run finished) or the history holds none.
        """
        async for commit in self.commits():
            if not self.is_failed(commit):
                return None if self.is_final_state(commit) else commit
        return None

    @staticmethod
    def is_final_state(commit: Commit) -> bool:
        """True for a final-state commit — written when a run finished cleanly."""
        return commit.meta.node_path == END_NODE_PATH

    @staticmethod
    def is_failed(commit: Commit) -> bool:
        """True for a failure commit — written when a run raised."""
        return commit.meta.node_path == FAILED_NODE_PATH

    async def last_complete(self) -> Commit | None:
        """Final-state commit of the most recent run that finished, or ``None``.

        Unlike :meth:`head`, stays on that commit while a later run appends
        past it — e.g. the state a halted follow-up session started from.
        """
        flow_id = await self.flow_id()
        if flow_id is None:
            return None
        commit_hash = await maybe_await(self.store.get_ref(flow_id, COMPLETE_TAG))
        return None if commit_hash is None else await self._commit(flow_id, commit_hash)

    async def commits(self) -> AsyncIterator[Commit]:
        """Walk the chain from the head through parent links, newest first.

        Parents are in write order across runs; a run's commits end at a
        ``halted``, ``$end`` or ``$failed`` commit.
        """
        commit = await self.head()
        while commit is not None:
            yield commit
            if not commit.parent_hashes:
                return
            commit = await self._commit(commit.meta.flow_id, commit.parent_hashes[0])

    async def snapshot(self, commit: Commit) -> Snapshot:
        """The run state ``commit`` holds, as stored (JSON-compatible values).

        The root scope, every child scope live at the save point keyed by
        its path, and each running iterate's pass counter. ``has_state`` is
        ``False`` for a commit that carries no state.
        """
        flow_id = commit.meta.flow_id

        async def load(kind: Kind, content_hash: str) -> bytes:
            return await self._object(flow_id, kind, content_hash)

        return await read_snapshot(commit.root_tree_hash, load)

    async def root_state(self, commit: Commit, factory: StateFactory[T]) -> T | None:
        """Top-level state ``commit`` holds, restored through ``factory``.

        ``None`` for a commit that carries no state. For dict-state flows,
        read ``(await snapshot(commit)).root`` instead.
        """
        snapshot = await self.snapshot(commit)
        if not snapshot.has_state:
            return None
        return restore_state_data(factory, snapshot.root)

    async def _commit(self, flow_id: str, commit_hash: str) -> Commit:
        """Load and parse one commit object."""
        return Commit.from_bytes(await self._object(flow_id, "commit", commit_hash))

    async def _object(self, flow_id: str, kind: Kind, content_hash: str) -> bytes:
        """Load one object the history references; :class:`HistoryCorrupt` if absent."""
        payload: bytes | None = await maybe_await(
            self.store.get_object(flow_id, kind, content_hash)
        )
        if payload is None:
            raise HistoryCorrupt(flow_id, kind, content_hash)
        return payload
