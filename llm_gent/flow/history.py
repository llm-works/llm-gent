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
    HEAD_REF,
    TAG_PREFIX,
    CheckpointStore,
    Kind,
    checkpoint_tag,
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
        """True when the head is a final-state commit: the last run that wrote one finished.

        A run that raises writes nothing, so after a run that raised this
        still reports what the run before it left.
        """
        head = await self.head()
        return head is not None and self.is_final_state(head)

    @staticmethod
    def is_final_state(commit: Commit) -> bool:
        """True for a final-state commit — written when a run finished cleanly."""
        return commit.meta.node_path == END_NODE_PATH

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

    async def checkpoint(self, name: str) -> Commit | None:
        """Commit of the named checkpoint ``name`` (``ctx.checkpoint(name)``), or ``None``.

        Raises:
            ValueError: ``name`` cannot name a checkpoint (see
                :func:`~llm_gent.flow.checkpoint.checkpoint_tag`).
        """
        tag = checkpoint_tag(name)
        flow_id = await self.flow_id()
        if flow_id is None:
            return None
        commit_hash = await maybe_await(self.store.get_ref(flow_id, tag))
        return None if commit_hash is None else await self._commit(flow_id, commit_hash)

    async def checkpoint_names(self) -> list[str]:
        """Names of the history's named checkpoints, sorted; empty when there is no history."""
        flow_id = await self.flow_id()
        if flow_id is None:
            return []
        refs: dict[str, str] = await maybe_await(self.store.list_refs(flow_id))
        return sorted(
            ref.removeprefix(TAG_PREFIX)
            for ref in refs
            if ref.startswith(TAG_PREFIX) and ref != COMPLETE_TAG
        )

    async def commit(self, commit_hash: str) -> Commit | None:
        """The history's commit ``commit_hash``, or ``None`` when the history holds no such commit.

        Any commit the history holds, on its line or off it — e.g. one
        written after a checkpoint that ``resume=<name>`` moved ``HEAD``
        back to, until :func:`~llm_gent.flow.collect_unreachable` deletes it.
        """
        flow_id = await self.flow_id()
        if flow_id is None:
            return None
        payload: bytes | None = await maybe_await(
            self.store.get_object(flow_id, "commit", commit_hash)
        )
        return None if payload is None else Commit.from_bytes(payload)

    async def commits(self) -> AsyncIterator[Commit]:
        """Walk the chain from the head through parent links, newest first.

        Parents are in write order across runs; a run's commits end at a
        ``halted`` or ``$end`` commit, or at its last save when it raised.
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
