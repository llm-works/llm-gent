# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""CheckpointContext — CAS-store wiring bundled with save operations.

Introduced to replace the ``(checkpointer, client_flow_id)`` pair
that used to travel together on ``_RunEnv`` and Flow. Every save
site had the same guard shape — ``if env.checkpointer is None or
env.client_flow_id is None: return`` — and every store call re-
plumbed the client_flow_id argument. :class:`CheckpointContext`
holds the pair as one unambiguously-bound object and exposes the
save operations the framework needs.

Presence is the persistence gate. When a Flow was constructed with
``.with_checkpointer(store, client_flow_id)`` there is a context;
otherwise ``env.checkpoint_ctx is None`` and every save site skips.
No more Optional-pair guards.

The context is also the translation layer between the agent's name
(``client_flow_id``) and gent's internal history identity
(``flow_id``): reads look the ``flow_id`` up; the first save creates it.
Every store call below the context is keyed by ``flow_id``.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .checkpoint import FRAMEWORK_PRODUCER, CheckpointStore, Kind, Retention, maybe_await
from .state import serialize_state_data
from .state.cas import (
    Blob,
    Commit,
    CommitMeta,
    ProducedBy,
    Tree,
    TreeEntry,
    canonical_json,
)


if TYPE_CHECKING:
    from .state import State
    from .state.cas import CommitOutcome, TraceRef


class CheckpointContext:
    """CAS-store wiring bundled with the save operations that consume it.

    Constructed when a Flow is attached to a checkpointer via
    ``.with_checkpointer(store, client_flow_id)``. Absent when the
    flow runs without persistence — save sites check
    ``env.checkpoint_ctx is None`` and skip.

    Owns the ``(store, client_flow_id)`` pair unambiguously (both
    fields are always populated once the context exists — the
    Optional lives at the context level, not per-field) and caches
    the history's ``flow_id`` and head within one run
    (:meth:`begin_run` drops them). Exposes the put-object triad, ref
    put/resolve, get_object, put_tag, gc_history, :meth:`append_commit`,
    and the compound :meth:`save_scope_commit` that assembles a
    Blob→Tree→Commit chain and refs it at the boundary.
    """

    def __init__(
        self,
        store: CheckpointStore,
        client_flow_id: str,
        root_hash: Callable[[], str],
    ) -> None:
        """Bind the store and name; ``root_hash`` yields the owning flow's structure hash.

        ``root_hash`` is evaluated once per run, at the run's first commit,
        so nodes added to the flow after ``with_checkpointer`` are reflected
        and every commit of a run records the same structure hash.
        """
        self.store = store
        self.client_flow_id = client_flow_id
        self._root_hash = root_hash
        self._run_root_hash: str | None = None
        self._flow_id: str | None = None
        self._flow_id_lock = asyncio.Lock()
        # Head of the history: the newest commit, parent of the next one.
        # Loaded from the store on first append, then maintained here —
        # the context assumes it is the history's single writer for the run.
        self._head: str | None = None
        self._head_loaded = False
        self._commit_lock = asyncio.Lock()

    def begin_run(self) -> None:
        """Drop the cached ``flow_id``, head and root hash, and create fresh locks.

        Called at the start of every top-level run so each run re-reads
        the store: a history collected (or advanced) by someone else between
        runs is observed instead of written into under a stale id or head.
        Fresh locks keep a Flow reusable across event loops.
        """
        self._run_root_hash = None
        self._flow_id = None
        self._flow_id_lock = asyncio.Lock()
        self._head = None
        self._head_loaded = False
        self._commit_lock = asyncio.Lock()

    @property
    def retention(self) -> Retention:
        """The store's retention mode (``"retain"`` or ``"gc_on_success"``)."""
        return self.store.retention

    # --- flow_id resolution (client_flow_id → flow_id) ---

    async def lookup_flow_id(self) -> str | None:
        """Return this history's ``flow_id``, or ``None`` when none exists yet."""
        if self._flow_id is None:
            self._flow_id = await maybe_await(self.store.get_flow_id(self.client_flow_id))
        return self._flow_id

    async def ensure_flow_id(self) -> str:
        """Return this history's ``flow_id``, creating the history on first call.

        The lock keeps parallel map items in this process to one bind; the
        store's atomic bind-if-absent settles races with other processes.
        """
        async with self._flow_id_lock:
            flow_id = await self.lookup_flow_id()
            if flow_id is None:
                bound: str = await maybe_await(
                    self.store.bind_flow_id(self.client_flow_id, str(uuid.uuid4()))
                )
                self._flow_id = flow_id = bound
            return flow_id

    # --- store passthrough (kind-specific put_object variants) ---

    async def put_blob(self, content_hash: str, payload: bytes) -> None:
        """Put a blob under this history."""
        flow_id = await self.ensure_flow_id()
        await maybe_await(self.store.put_object(flow_id, "blob", content_hash, payload))

    async def put_tree(self, tree: Tree) -> None:
        """Serialize + put a Tree under this history."""
        flow_id = await self.ensure_flow_id()
        await maybe_await(
            self.store.put_object(flow_id, "tree", tree.content_hash, tree.to_bytes())
        )

    async def put_commit(self, commit: Commit) -> None:
        """Serialize + put a Commit under this history."""
        flow_id = await self.ensure_flow_id()
        await maybe_await(
            self.store.put_object(flow_id, "commit", commit.content_hash, commit.to_bytes())
        )

    async def put_ref(self, node_path: str, iteration: int, commit_hash: str) -> None:
        """Point ``(flow_id, node_path, iteration)`` at ``commit_hash``."""
        flow_id = await self.ensure_flow_id()
        await maybe_await(self.store.put_ref(flow_id, node_path, iteration, commit_hash))

    async def resolve_ref(self) -> str | None:
        """Latest commit hash across this history, or ``None`` (incl. no history)."""
        flow_id = await self.lookup_flow_id()
        if flow_id is None:
            return None
        result: str | None = await maybe_await(self.store.resolve_ref(flow_id))
        return result

    async def get_object(self, kind: Kind, content_hash: str) -> bytes | None:
        """Fetch an object under this history by kind + hash."""
        flow_id = await self.lookup_flow_id()
        if flow_id is None:
            return None
        result: bytes | None = await maybe_await(self.store.get_object(flow_id, kind, content_hash))
        return result

    async def put_tag(self, name: str, commit_hash: str) -> None:
        """Point tag ``name`` under this history at ``commit_hash``."""
        flow_id = await self.ensure_flow_id()
        await maybe_await(self.store.put_tag(flow_id, name, commit_hash))

    async def gc_history(self) -> None:
        """Remove this history (objects, refs, tags, name mapping); the next save starts a new one."""
        flow_id = await self.lookup_flow_id()
        if flow_id is None:
            return
        try:
            await maybe_await(self.store.gc_history(flow_id))
        finally:
            # Also on a failed gc: the next access re-reads the name binding
            # instead of trusting a flow_id the store may have half-removed.
            self._flow_id = None
            self._head = None
            self._head_loaded = False

    # --- history: append a commit on top of the head ---

    async def append_commit(self, root_tree_hash: str, meta: CommitMeta) -> Commit:
        """Build a commit whose parent is the current head, store it, and ref it.

        The ref goes to ``(meta.node_path, meta.iteration)``; the new commit
        becomes the head. Serialized so concurrent saves (parallel map items)
        form one linear history rather than sibling commits sharing a parent.
        The first commit of a history has no parent.
        """
        async with self._commit_lock:
            parent = await self._load_head()
            commit = Commit.build(
                root_tree_hash=root_tree_hash,
                parent_hashes=() if parent is None else (parent,),
                meta=meta,
            )
            await self.put_commit(commit)
            await self.put_ref(meta.node_path, meta.iteration, commit.content_hash)
            self._head = commit.content_hash
            return commit

    async def _load_head(self) -> str | None:
        """Return the cached head, reading the newest ref on first use."""
        if not self._head_loaded:
            self._head = await self.resolve_ref()
            self._head_loaded = True
        return self._head

    # --- compound: save a scope commit ---

    async def save_scope_commit(
        self,
        ancestor_chain: tuple[str, ...],
        iteration: int,
        node_id: str,
        current_state: State[Any],
        outcome: CommitOutcome,
        trace_ref: tuple[TraceRef, ...] = (),
    ) -> Commit:
        """Persist a content-addressed scope commit at ``node_id``; return it.

        Walks the scope stack from root to ``current_state``. For
        each scope: serialize its ``data`` via the state-data
        contract (dict passthrough or ``StateData.to_dict``) to
        canonical JSON bytes and :meth:`put_blob` a Blob keyed by
        content hash. Bundle every scope's blob hash into a Tree
        (one :class:`TreeEntry` per scope, ordered by depth via a
        two-digit ``scope_id``). Wrap the Tree in a Commit whose
        :class:`CommitMeta` pins ``(flow_id, node_path,
        iteration)`` and the provenance triple (``produced_by``,
        ``trace_ref``, ``outcome``). Finally :meth:`append_commit`
        chains it onto the head and points this boundary at it.

        ``node_path`` is the ``"/"``-joined ancestor chain (from run
        root to this node, inclusive). blake2b hex has no ``"/"``,
        so split round-trips on resume.

        ``outcome`` records why the commit fired — ``"ok"`` for a
        successful iterate boundary, ``"halted"`` when the
        halt-observation site triggered the save.
        """
        tree = await self.put_state_tree(current_state)
        node_path = "/".join(ancestor_chain + (node_id,))
        flow_id = await self.ensure_flow_id()
        meta = self._build_commit_meta(flow_id, node_path, iteration, node_id, outcome, trace_ref)
        return await self.append_commit(tree.content_hash, meta)

    async def put_state_tree(self, current_state: State[Any]) -> Tree:
        """Put one blob per scope from run root to ``current_state`` and their Tree."""
        entries = await self._put_scope_blobs(self._collect_scope_stack(current_state))
        tree = Tree.from_entries(entries)
        await self.put_tree(tree)
        return tree

    async def save_framework_commit(self, node_path: str, tree: Tree) -> Commit:
        """Commit an already-put ``tree`` at a reserved ``node_path`` the framework owns.

        No node produced it: ``produced_by.node_id`` is :data:`FRAMEWORK_PRODUCER`.
        """
        flow_id = await self.ensure_flow_id()
        meta = self._build_commit_meta(flow_id, node_path, 0, FRAMEWORK_PRODUCER, "ok", ())
        return await self.append_commit(tree.content_hash, meta)

    # --- private helpers used by save_scope_commit ---

    @staticmethod
    def _collect_scope_stack(current: State[Any]) -> list[State[Any]]:
        """Return the ``State`` chain from run-root down to ``current``."""
        scopes: list[State[Any]] = []
        node: State[Any] | None = current
        while node is not None:
            scopes.append(node)
            node = node._parent
        scopes.reverse()
        return scopes

    async def _put_scope_blobs(self, scopes: list[State[Any]]) -> list[TreeEntry]:
        """Serialize each scope's data, put_blob it, return tree entries.

        Uses a zero-padded two-digit index as :attr:`TreeEntry.scope_id`
        so canonical sort ordering matches root→leaf depth ordering.
        """
        entries: list[TreeEntry] = []
        for depth, scope in enumerate(scopes):
            blob_bytes = canonical_json(serialize_state_data(scope.data))
            blob = Blob.from_bytes(blob_bytes)
            await self.put_blob(blob.content_hash, blob.payload)
            entries.append(
                TreeEntry(scope_id=f"{depth:02d}", kind="blob", child_hash=blob.content_hash)
            )
        return entries

    def _build_commit_meta(
        self,
        flow_id: str,
        node_path: str,
        iteration: int,
        node_id: str,
        outcome: CommitOutcome,
        trace_ref: tuple[TraceRef, ...],
    ) -> CommitMeta:
        """Assemble :class:`CommitMeta` for one scope-commit save.

        ``produced_by`` records the node's ``node_id`` — verb-level
        attribution (``verb_name`` / ``role`` / ``result_hash``)
        lands with the SAIA-verb-wrapper wiring. ``flow_root_hash`` is
        the owning flow's structure hash, computed once per run.
        """
        from llm_gent import __version__

        if self._run_root_hash is None:
            self._run_root_hash = self._root_hash()
        return CommitMeta(
            flow_id=flow_id,
            node_path=node_path,
            iteration=iteration,
            produced_by=ProducedBy(node_id=node_id, verb_name=None, role=None, result_hash=None),
            trace_ref=trace_ref,
            outcome=outcome,
            flow_root_hash=self._run_root_hash,
            timestamp_iso=datetime.now(UTC).isoformat(),
            framework_version=__version__,
        )
