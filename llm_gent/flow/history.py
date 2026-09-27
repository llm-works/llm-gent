# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""History — read API over one flow history in a :class:`CheckpointStore`.

A history is the chain of commits one flow instance writes, addressed by
the agent's ``client_flow_id``. :class:`History` resolves the name to the
internal ``flow_id`` and exposes the history as data: its head, the last
completed run, the commit chain, and the state each commit holds. It
never writes; the framework owns every write.

Every method works with sync and async stores alike.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from .checkpoint import COMPLETE_TAG, END_NODE_PATH, CheckpointStore, maybe_await
from .state import StateFactory
from .state.cas import Commit, Tree


T = TypeVar("T")


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
        """Latest commit, or ``None`` for an empty (or unreadable) history."""
        flow_id = await self.flow_id()
        if flow_id is None:
            return None
        commit_hash = await maybe_await(self.store.resolve_ref(flow_id))
        return None if commit_hash is None else await self._commit(flow_id, commit_hash)

    async def is_complete(self) -> bool:
        """True when the last run finished: the head is its final-state commit.

        ``run(resume=True)`` on a complete history starts fresh; otherwise
        it resumes from the head.
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
        commit_hash = await maybe_await(self.store.resolve_tag(flow_id, COMPLETE_TAG))
        return None if commit_hash is None else await self._commit(flow_id, commit_hash)

    async def commits(self) -> AsyncIterator[Commit]:
        """Walk the chain from the head through parent links, newest first.

        Stops early if a parent object is missing from the store.
        """
        commit = await self.head()
        while commit is not None:
            yield commit
            if not commit.parent_hashes:
                return
            commit = await self._commit(commit.meta.flow_id, commit.parent_hashes[0])

    async def scopes(self, commit: Commit) -> list[Any] | None:
        """State payloads ``commit`` holds, root scope first, as stored (JSON).

        One entry per state scope on the path to the save point. ``None``
        when the tree or a blob is missing.
        """
        flow_id = commit.meta.flow_id
        tree_bytes = await maybe_await(
            self.store.get_object(flow_id, "tree", commit.root_tree_hash)
        )
        if tree_bytes is None:
            return None
        payloads: list[Any] = []
        for entry in Tree.from_bytes(tree_bytes).entries:
            blob = await maybe_await(self.store.get_object(flow_id, "blob", entry.child_hash))
            if blob is None:
                return None
            payloads.append(json.loads(blob.decode("utf-8")))
        return payloads

    async def root_state(self, commit: Commit, factory: StateFactory[T]) -> T | None:
        """Top-level state ``commit`` holds, restored through ``factory``.

        ``None`` when the commit's objects are missing. For dict-state flows,
        read ``(await scopes(commit))[0]`` instead.
        """
        payloads = await self.scopes(commit)
        if not payloads:
            return None
        root = payloads[0]
        return factory.restore(root if isinstance(root, dict) else {})

    async def _commit(self, flow_id: str, commit_hash: str) -> Commit | None:
        """Load and parse one commit object, or ``None`` when absent."""
        payload = await maybe_await(self.store.get_object(flow_id, "commit", commit_hash))
        return None if payload is None else Commit.from_bytes(payload)
