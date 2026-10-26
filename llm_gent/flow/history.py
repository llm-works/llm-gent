# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""History — read API over one flow history in an :class:`appinfra.cas.Store`.

A history is the chain of commits one flow instance writes, held in the
store's repository named by the agent's ``client_flow_id``.
:class:`History` exposes it as data: its head, the last completed run,
the commit chain, and the state and flow structure each commit holds. It
never writes; the framework owns every write.

``None`` means absent — no history under the name, no commit yet, no
completed run. A hash the history holds (ref, tag, parent, tree entry)
that points at an object missing from the store raises
:class:`HistoryCorrupt` instead, so a damaged history never reads as a
shorter or empty one.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TypeVar

from appinfra.cas import Blob, Commit, Object, ObjectTypeError, Store, Tree

from .checkpoint import COMPLETE_TAG, END_NODE_PATH, HEAD_REF, TAG_PREFIX, checkpoint_tag
from .state import StateFactory, restore_state_data
from .state.cas import FlowCommit
from .state.snapshot import FLOW, Snapshot, read_snapshot
from .structure import Structure


T = TypeVar("T")
ObjT = TypeVar("ObjT", bound=Object)


class HistoryCorrupt(ValueError):
    """A hash in the history points at an object the store does not hold."""

    def __init__(self, client_flow_id: str, kind: str, object_id: str) -> None:
        """Name the history and the ``kind`` of object ``object_id`` it is missing."""
        super().__init__(
            f"history {client_flow_id!r}: {kind} {object_id} is missing from the store"
        )
        self.client_flow_id = client_flow_id
        self.kind = kind
        self.object_id = object_id


class History:
    """Read-only view of the history named ``client_flow_id`` in ``store``.

    Holds no cached state: every call reads the store, so a view built once
    observes later writes.
    """

    def __init__(self, store: Store, client_flow_id: str) -> None:
        """View the store's repository ``client_flow_id``; nothing is read until a call."""
        self.store = store
        self.client_flow_id = client_flow_id
        self._repo = store.repo(client_flow_id)

    async def head(self) -> FlowCommit | None:
        """Latest commit, or ``None`` for an empty history."""
        return self._at_ref(HEAD_REF)

    async def is_complete(self) -> bool:
        """True when the head is a final-state commit: the last run that wrote one finished.

        A run that raises writes nothing, so after a run that raised this
        still reports what the run before it left.
        """
        head = await self.head()
        return head is not None and self.is_final_state(head)

    @staticmethod
    def is_final_state(commit: FlowCommit) -> bool:
        """True for a final-state commit — written when a run finished cleanly."""
        return commit.meta.node_path == END_NODE_PATH

    async def last_complete(self) -> FlowCommit | None:
        """Final-state commit of the most recent run that finished, or ``None``.

        Unlike :meth:`head`, stays on that commit while a later run appends
        past it — e.g. the state a halted follow-up session started from.
        """
        return self._at_ref(COMPLETE_TAG)

    async def checkpoint(self, name: str) -> FlowCommit | None:
        """Commit of the named checkpoint ``name`` (``ctx.checkpoint(name)``), or ``None``.

        Raises:
            ValueError: ``name`` cannot name a checkpoint (see
                :func:`~llm_gent.flow.checkpoint.checkpoint_tag`).
        """
        return self._at_ref(checkpoint_tag(name))

    async def checkpoint_names(self) -> list[str]:
        """Names of the history's named checkpoints, sorted; empty when there is no history."""
        return sorted(
            ref.removeprefix(TAG_PREFIX)
            for ref in self._repo.refs(TAG_PREFIX)
            if ref != COMPLETE_TAG
        )

    async def commit(self, commit_hash: str) -> FlowCommit | None:
        """The history's commit ``commit_hash``, or ``None`` when the history holds no such commit.

        Any commit the history holds, on its line or off it — e.g. one
        left off the line when a run resumed from an earlier checkpoint,
        until a prune of the repository deletes it.
        """
        try:
            commit = self._repo.get(commit_hash, Commit)
        except ObjectTypeError:
            return None
        return None if commit is None else FlowCommit.from_commit(commit)

    async def commits(self) -> AsyncIterator[FlowCommit]:
        """Walk the chain from the head through first parents, newest first.

        Parents are in write order across runs; a run's commits end at a
        ``halted`` or ``$end`` commit, or at its last save when it raised.
        Reads one commit per step.
        """
        commit = await self.head()
        while commit is not None:
            yield commit
            commit = self._commit(commit.parents[0]) if commit.parents else None

    async def snapshot(self, commit: FlowCommit) -> Snapshot:
        """The run state ``commit`` holds, as stored (JSON-compatible values).

        The root scope, every child scope live at the save point keyed by
        its path, and each running iterate's pass counter. ``has_state`` is
        ``False`` for a commit that carries no state.
        """
        return read_snapshot(commit.tree, self._object)

    async def structure(self, commit: FlowCommit) -> Structure | None:
        """The structure of the flow that wrote ``commit``.

        Compare it with a flow today through
        :meth:`~llm_gent.flow.structure.Structure.diff`. ``None`` for a
        commit that carries no state.
        """
        entry = self._object(commit.tree, Tree).get(FLOW)
        if entry is None:
            return None
        return Structure.from_json(json.loads(self._object(entry.id, Blob).data))

    async def root_state(self, commit: FlowCommit, factory: StateFactory[T]) -> T | None:
        """Top-level state ``commit`` holds, restored through ``factory``.

        ``None`` for a commit that carries no state. For dict-state flows,
        read ``(await snapshot(commit)).root`` instead.
        """
        snapshot = await self.snapshot(commit)
        if not snapshot.has_state:
            return None
        return restore_state_data(factory, snapshot.root)

    def _at_ref(self, name: str) -> FlowCommit | None:
        """The commit ref ``name`` points at, or ``None`` when the ref does not exist."""
        commit_hash = self._repo.get_ref(name)
        return None if commit_hash is None else self._commit(commit_hash)

    def _commit(self, commit_hash: str) -> FlowCommit:
        """Load one commit the history references, with its metadata parsed."""
        return FlowCommit.from_commit(self._object(commit_hash, Commit))

    def _object(self, object_id: str, cls: type[ObjT]) -> ObjT:
        """Load one object the history references as ``cls``; :class:`HistoryCorrupt` if absent."""
        obj = self._repo.get(object_id, cls)
        if obj is None:
            raise HistoryCorrupt(self.client_flow_id, cls.type, object_id)
        return obj
