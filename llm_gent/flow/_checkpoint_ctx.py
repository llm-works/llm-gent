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

A run has one context: its repo, set on the top-level flow with
``.with_checkpoint_store(store, client_flow_id)``. Without one,
``env.checkpoint_ctx is None`` and every save site skips; with one, saves
inside the run also need a ``with_checkpointer()`` on the saving flow or
above it (``env.checkpointer``). :func:`check_one_repo` rejects a store on
a nested flow and a checkpointer in a run without a store.

The context is also the translation layer between the agent's name
(``client_flow_id``) and gent's internal history identity
(``flow_id``): reads look the ``flow_id`` up; the first save creates it.
Every store call below the context is keyed by ``flow_id``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .checkpoint import (
    COMPLETION_PRODUCER,
    END_NODE_PATH,
    HEAD_REF,
    CheckpointStore,
    ConcurrentWriteError,
    Kind,
    Retention,
    maybe_await,
)
from .state.cas import Blob, Commit, CommitMeta, ProducedBy, Tree
from .state.snapshot import ScopeRegistry, build_snapshot_tree, path_str
from .structure import FlowStructure


if TYPE_CHECKING:
    from .flow import Flow
    from .state import State
    from .state.cas import CommitOutcome, TraceRef


def check_one_repo(root: Flow) -> None:
    """Raise unless ``root``'s run has at most one repo, on ``root``, and checkpointers need one.

    Raises:
        RuntimeError: A flow inside the run sets a checkpoint store (a run
            has one repo, on its top-level flow), or a flow declares
            ``with_checkpointer()`` while the run has no store.
    """
    from ._node_id import iter_flows

    flows = iter_flows(root)
    for flow in flows[1:]:
        if flow._checkpoint_ctx is not None:
            raise RuntimeError(
                f"Flow {_label(flow)} sets a checkpoint store but runs inside "
                f"{_label(root)}: a run has one repo, set on its top-level flow — "
                f"use with_checkpointer() on the inner flow instead"
            )
    if root._checkpoint_ctx is None:
        declaring = next((f for f in flows if f._checkpointer is not None), None)
        if declaring is not None:
            raise RuntimeError(
                f"Flow {_label(declaring)} declares with_checkpointer() but the run has no "
                f"checkpoint store: checkpointers require a checkpoint store — call "
                f"with_checkpoint_store(store, client_flow_id) on {_label(root)}"
            )


def _label(flow: Flow) -> str:
    return repr(flow._name or "<anonymous>")


class CheckpointContext:
    """CAS-store wiring bundled with the save operations that consume it.

    Constructed by ``.with_checkpoint_store(store, client_flow_id)`` on a
    run's top-level flow. Absent when the run has no store — save sites
    check ``env.checkpoint_ctx is None`` and skip.

    Owns the ``(store, client_flow_id)`` pair unambiguously (both
    fields are always populated once the context exists — the
    Optional lives at the context level, not per-field) and caches
    the history's ``flow_id`` and head within one run
    (:meth:`begin_run` drops them). Exposes the put-object triad,
    get_ref / move_ref, get_object, put_tag, gc_history,
    :meth:`append_commit`, and the compound :meth:`save_scope_commit`
    that assembles a Blob→Tree→Commit chain and moves ``HEAD`` to it.
    """

    def __init__(
        self,
        store: CheckpointStore,
        client_flow_id: str,
        structure: Callable[[], FlowStructure],
    ) -> None:
        """Bind the store and name; ``structure`` yields the owning flow's structure.

        ``structure`` is evaluated once per run, at the run's first commit,
        so nodes added to the flow after ``with_checkpoint_store`` are
        reflected and every commit of a run holds the same structure and
        records its hash.
        """
        self.store = store
        self.client_flow_id = client_flow_id
        self._structure = structure
        self._run_structure: Blob | None = None
        self._flow_id: str | None = None
        self._flow_id_lock = asyncio.Lock()
        # _parent: parent of the next commit (normally HEAD, but an earlier
        # commit after continue_from). _head: where HEAD points, used as the
        # expected value for compare-and-set. The two are the same except
        # after continue_from, which parents on an earlier commit while HEAD
        # stays put until the run's first commit. Loaded from the store on
        # first append, then maintained here — the context assumes it is the
        # history's single writer for the run.
        self._parent: str | None = None
        self._head: str | None = None
        self._head_loaded: bool = False
        self._commit_lock = asyncio.Lock()
        # (kind, hash) of the blobs / trees this context already put during
        # the run. Scopes that did not change since the last commit are
        # content-identical, so their puts are skipped instead of repeated.
        # The kind is part of the key: a blob and a tree can have the same
        # bytes (the blob "[]" and the empty tree) and so the same hash.
        self._written: set[tuple[Kind, str]] = set()

    def begin_run(self) -> None:
        """Drop the cached ``flow_id``, head and structure, and create fresh locks.

        Called at the start of every top-level run so each run re-reads
        the store: a history collected (or advanced) by someone else between
        runs is observed instead of written into under a stale id or head.
        Fresh locks keep a Flow reusable across event loops.
        """
        self._run_structure = None
        self._flow_id = None
        self._flow_id_lock = asyncio.Lock()
        self._forget_head()
        self._commit_lock = asyncio.Lock()
        self._written = set()

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
        """Put a blob under this history; a no-op when this run already put it."""
        if ("blob", content_hash) in self._written:
            return
        flow_id = await self.ensure_flow_id()
        await maybe_await(self.store.put_object(flow_id, "blob", content_hash, payload))
        self._written.add(("blob", content_hash))

    async def put_tree(self, tree: Tree) -> None:
        """Serialize + put a Tree under this history; a no-op when this run already put it."""
        if ("tree", tree.content_hash) in self._written:
            return
        flow_id = await self.ensure_flow_id()
        await maybe_await(
            self.store.put_object(flow_id, "tree", tree.content_hash, tree.to_bytes())
        )
        self._written.add(("tree", tree.content_hash))

    async def put_commit(self, commit: Commit) -> None:
        """Serialize + put a Commit under this history."""
        flow_id = await self.ensure_flow_id()
        await maybe_await(
            self.store.put_object(flow_id, "commit", commit.content_hash, commit.to_bytes())
        )

    async def get_ref(self, name: str) -> str | None:
        """Commit hash ref ``name`` points at, or ``None`` (incl. no history)."""
        flow_id = await self.lookup_flow_id()
        if flow_id is None:
            return None
        result: str | None = await maybe_await(self.store.get_ref(flow_id, name))
        return result

    async def move_ref(self, name: str, commit_hash: str, expected: str | None) -> None:
        """Move ref ``name`` from ``expected`` to ``commit_hash``.

        Raises:
            ConcurrentWriteError: The ref no longer points at ``expected``
                — another writer moved it.
        """
        flow_id = await self.ensure_flow_id()
        moved = await maybe_await(self.store.set_ref(flow_id, name, commit_hash, expected))
        if not moved:
            raise ConcurrentWriteError(self.client_flow_id, name, expected)

    async def get_object(self, kind: Kind, content_hash: str) -> bytes | None:
        """Fetch an object under this history by kind + hash."""
        flow_id = await self.lookup_flow_id()
        if flow_id is None:
            return None
        result: bytes | None = await maybe_await(self.store.get_object(flow_id, kind, content_hash))
        return result

    async def put_tag(self, name: str, commit_hash: str) -> None:
        """Point tag ref ``name`` at ``commit_hash``, moving it from wherever it points now.

        Raises:
            ConcurrentWriteError: Another writer moved the tag between the
                read and the CAS. This is intentional: a history has one
                writer at a time, and the failure detects a violation.
        """
        await self.move_ref(name, commit_hash, await self.get_ref(name))

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
            self._forget_head()
            self._written.clear()

    # --- history: append a commit on top of the head ---

    async def append_commit(self, root_tree_hash: str, meta: CommitMeta) -> Commit:
        """Build a commit whose parent is the current head, store it, and move ``HEAD`` to it.

        ``HEAD`` moves by compare-and-set, so a concurrent writer raises
        :class:`ConcurrentWriteError` instead of forking the history
        silently. Serialized so concurrent saves (parallel map items)
        form one linear history rather than sibling commits sharing a
        parent. The first commit of a history has no parent. After
        :meth:`continue_from`, the parent is the commit it named, and
        ``HEAD`` moves from where it pointed then, not from the parent.
        """
        async with self._commit_lock:
            parent = await self._load_head()
            commit = Commit.build(
                root_tree_hash=root_tree_hash,
                parent_hashes=() if parent is None else (parent,),
                meta=meta,
            )
            await self.put_commit(commit)
            await self.move_ref(HEAD_REF, commit.content_hash, self._head)
            self._parent = self._head = commit.content_hash
            return commit

    async def continue_from(self, commit_hash: str) -> None:
        """Parent the run's commits on ``commit_hash``; ``HEAD`` moves from where it points now.

        Until the run commits, ``HEAD`` stays where it is: a run that
        fails first leaves the history as it was.
        """
        self._parent = commit_hash
        self._head = await self.get_ref(HEAD_REF)
        self._head_loaded = True

    async def _load_head(self) -> str | None:
        """Return the cached parent, reading ``HEAD`` on first use."""
        if not self._head_loaded:
            self._parent = self._head = await self.get_ref(HEAD_REF)
            self._head_loaded = True
        return self._parent

    def _forget_head(self) -> None:
        """Clear the cached head so the next access re-reads ``HEAD``."""
        self._parent = None
        self._head = None
        self._head_loaded = False

    # --- compound: save a scope commit ---

    async def save_scope_commit(
        self,
        ancestor_chain: tuple[str, ...],
        iteration: int,
        node_id: str,
        scopes: ScopeRegistry,
        current_state: State[Any],
        outcome: CommitOutcome,
        trace_ref: tuple[TraceRef, ...] = (),
    ) -> Commit:
        """Persist a full snapshot of the run as a commit at ``node_id``; return it.

        The snapshot holds every live scope of the run (see
        :func:`put_snapshot`), not only the ones above ``current_state``.
        :class:`CommitMeta` pins ``(flow_id, node_path, iteration)``, the
        path of the scope ``current_state`` is, and the provenance triple
        (``produced_by``, ``trace_ref``, ``outcome``). :meth:`append_commit`
        then chains the commit onto ``HEAD``.

        ``node_path`` is the ``"/"``-joined ancestor chain (from run
        root to this node, inclusive). blake2b hex has no ``"/"``,
        so split round-trips on resume.

        ``outcome`` records why the commit fired — ``"ok"`` for a
        policy save or ``ctx.checkpoint()``, ``"halted"`` for the run's
        halt checkpoint.
        """
        tree = await self.put_snapshot(scopes)
        node_path = "/".join(ancestor_chain + (node_id,))
        flow_id = await self.ensure_flow_id()
        meta = self._build_commit_meta(flow_id, node_path, iteration, node_id, outcome, trace_ref)
        meta = dataclasses.replace(meta, scope_path=path_str(scopes.path_of(current_state)))
        return await self.append_commit(tree.content_hash, meta)

    async def put_snapshot(self, scopes: ScopeRegistry) -> Tree:
        """Put the snapshot tree of every live scope in ``scopes``; return its root tree.

        Every scope is serialized before the first write
        (:meth:`ScopeRegistry.capture` is synchronous): the writes await
        the store, and other tasks (concurrent map items) change the
        scopes meanwhile, so serializing between writes could commit
        scopes from different moments.

        The root tree also holds the run's flow structure
        (:data:`~llm_gent.flow.state.snapshot.FLOW`), whose
        hash every commit of the run records as ``flow_root_hash``.
        """
        tree, objects = build_snapshot_tree(scopes.capture(), self._run_structure_blob())
        for obj in objects:
            if isinstance(obj, Blob):
                await self.put_blob(obj.content_hash, obj.payload)
            else:
                await self.put_tree(obj)
        return tree

    async def save_completion_commit(self, tree: Tree) -> Commit:
        """Commit an already-put ``tree`` as the final-state commit (``$end``, outcome ``ok``)."""
        return await self._save_framework_commit(END_NODE_PATH, COMPLETION_PRODUCER, "ok", tree)

    async def _save_framework_commit(
        self, node_path: str, producer: str, outcome: CommitOutcome, tree: Tree
    ) -> Commit:
        """Commit ``tree`` at a reserved ``node_path`` the framework owns.

        No node produced it: ``producer`` is a ``$framework/*`` pseudo node id.
        """
        flow_id = await self.ensure_flow_id()
        meta = self._build_commit_meta(flow_id, node_path, 0, producer, outcome, ())
        return await self.append_commit(tree.content_hash, meta)

    # --- private helpers used by save_scope_commit ---

    def _run_structure_blob(self) -> Blob:
        """The owning flow's structure as a blob, taken at the run's first use."""
        if self._run_structure is None:
            self._run_structure = self._structure().blob()
        return self._run_structure

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
        the hash of the owning flow's structure, taken once per run.
        """
        from llm_gent import __version__

        return CommitMeta(
            flow_id=flow_id,
            node_path=node_path,
            iteration=iteration,
            produced_by=ProducedBy(node_id=node_id, verb_name=None, role=None, result_hash=None),
            trace_ref=trace_ref,
            outcome=outcome,
            flow_root_hash=self._run_structure_blob().content_hash,
            timestamp_iso=datetime.now(UTC).isoformat(),
            framework_version=__version__,
        )
