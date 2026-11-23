# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""CheckpointContext — the run's history repository bundled with save operations.

Every save site would otherwise check the flow's checkpointer and the
history's name separately, and re-plumb the name into every store call.
:class:`CheckpointContext` holds the history's
:class:`appinfra.cas.Repo` and exposes the save operations the framework
needs.

A run has at most one context: the repository its ``history=``
names in the top-level flow's checkpoint store. Without one,
``frame.checkpoint_ctx is None`` and every save site skips; with one, saves
inside the run also need a ``with_checkpointer()`` on the saving flow or
above it (``frame.scope.checkpointer``). :func:`check_checkpointers` rejects a
checkpointer in a run without a history.

The repository is named by ``client_flow_id``; store calls are
synchronous, as :class:`appinfra.cas.Store` is.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from appinfra.cas import Blob, Repo, Store, Tree

from .._node_id import iter_flows
from ..checkpoint import (
    COMPLETION_PRODUCER,
    END_NODE_PATH,
    HEAD_REF,
    TAG_PREFIX,
    ConcurrentWriteError,
    Retention,
)
from ..flow import Flow
from ..state.cas import CommitMeta, FlowCommit, ProducedBy
from ..state.snapshot import ScopeRegistry, build_snapshot_tree, path_str
from ..structure import Structure


if TYPE_CHECKING:
    from ..state import State
    from ..state.cas import CommitOutcome, TraceRef


def check_checkpointers(root: Flow, history: str | None) -> None:
    """Raise when a flow in ``root``'s run declares a checkpointer but the run has no history.

    Raises:
        RuntimeError: A flow declares ``with_checkpointer()`` and the run
            was given no ``history``: its saves would have nowhere to go.
    """
    if history is not None:
        return
    declaring = next((f for f in iter_flows(root) if f._checkpointer is not None), None)
    if declaring is not None:
        raise RuntimeError(
            f"Flow {_label(declaring)} declares with_checkpointer() but the run of "
            f"{_label(root)} has no history: pass start(flow, history=...) to keep its "
            f"checkpoints"
        )


def _is_history_ref(name: str) -> bool:
    """True for the refs a checkpoint history owns: ``HEAD`` and its tags."""
    return name == HEAD_REF or name.startswith(TAG_PREFIX)


def _label(flow: Flow) -> str:
    return repr(flow._name or "<anonymous>")


class CheckpointContext:
    """The run's history repository bundled with the save operations that consume it.

    Constructed by :meth:`FlowRunner.start` for a run's ``history=``, on the
    top-level flow's checkpoint store. Absent when the run has no history —
    save sites check ``frame.checkpoint_ctx is None`` and skip.

    Each top-level run writes through its own, built at run start: it reads
    the head from the store on its first commit, so a history deleted or
    advanced by someone else between runs is observed instead of written
    into under a stale head, and runs of one flow at once keep their own
    head, lock and written set.
    Exposes get_ref / move_ref, put_tag, delete_history,
    :meth:`append_commit`, and the compound :meth:`save_scope_commit` that
    puts a snapshot, commits it and moves ``HEAD`` to it.
    """

    def __init__(
        self,
        store: Store,
        client_flow_id: str,
        structure: Callable[[], Structure],
        retention: Retention,
    ) -> None:
        """Bind the store's repository ``client_flow_id``; ``structure`` yields the flow's structure.

        ``structure`` is evaluated once per run, at the run's first commit,
        so every commit of a run holds the same structure and records its
        hash.

        Raises:
            appinfra.cas.InvalidNameError: ``client_flow_id`` cannot name a
                repository (empty, ``.``/``..``, NUL, over 255 UTF-8 bytes).
            ValueError: ``retention`` is not ``"retain"`` or ``"gc_on_success"``.
        """
        self.store = store
        self.client_flow_id = client_flow_id
        self.repo: Repo = store.repo(client_flow_id)
        if retention not in ("retain", "gc_on_success"):
            raise ValueError(f"retention must be 'retain' or 'gc_on_success'; got {retention!r}")
        self.retention = retention
        self._structure = structure
        self._run_structure: Blob | None = None
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
        # Ids of the blobs / trees this context already put during the run.
        # Scopes that did not change since the last commit are
        # content-identical, so their puts are skipped instead of repeated.
        self._written: set[str] = set()

    # --- objects and refs ---

    def put(self, obj: Blob | Tree) -> None:
        """Put a blob or tree into the history; a no-op when this run already put it."""
        if obj.id in self._written:
            return
        self.repo.put(obj)
        self._written.add(obj.id)

    async def get_ref(self, name: str) -> str | None:
        """Commit hash ref ``name`` points at, or ``None`` (incl. no history)."""
        return self.repo.get_ref(name)

    async def move_ref(self, name: str, commit_hash: str, expected: str | None) -> None:
        """Move ref ``name`` from ``expected`` to ``commit_hash``.

        Raises:
            ConcurrentWriteError: The ref no longer points at ``expected``
                — another writer moved it.
        """
        if not self.repo.set_ref(name, commit_hash, expected):
            raise ConcurrentWriteError(self.client_flow_id, name, expected)

    async def put_tag(self, name: str, commit_hash: str) -> None:
        """Point tag ref ``name`` at ``commit_hash``, moving it from wherever it points now.

        Raises:
            ConcurrentWriteError: Another writer moved the tag between the
                read and the CAS. This is intentional: a history has one
                writer at a time, and the failure detects a violation.
        """
        await self.move_ref(name, commit_hash, await self.get_ref(name))

    async def delete_history(self) -> None:
        """Delete the history: ``HEAD`` and its tags; the next save starts a new one.

        A repository holding no other refs is deleted whole. One that does
        (version-control records, an app's own refs) keeps them and what they
        reach; objects only the history reached stay until a prune of the
        repository deletes them. There the refs go one by one, tags before
        ``HEAD``, so a delete that fails partway leaves the history its head.

        Raises:
            ConcurrentWriteError: Another writer moved a history ref after
                it was read.
        """
        try:
            refs = self.repo.refs()
            if all(_is_history_ref(name) for name in refs):
                self.store.delete_repo(self.client_flow_id)
            else:
                history = [name for name in refs if _is_history_ref(name)]
                for name in sorted(history, key=lambda name: name == HEAD_REF):
                    if not self.repo.delete_ref(name, refs[name]):
                        raise ConcurrentWriteError(self.client_flow_id, name, refs[name])
        finally:
            # Also on a failed delete: the next access re-reads the store
            # instead of trusting a head the store may have half-removed.
            self._forget_head()
            self._written.clear()

    # --- history: append a commit on top of the head ---

    async def append_commit(self, tree: str, meta: CommitMeta) -> FlowCommit:
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
            commit, obj = FlowCommit.build(tree, () if parent is None else (parent,), meta)
            self.repo.put(obj)
            await self.move_ref(HEAD_REF, commit.id, self._head)
            self._parent = self._head = commit.id
            return commit

    async def continue_from(self, commit_hash: str, *, expect_head: str | None = None) -> None:
        """Parent the run's commits on ``commit_hash``; ``HEAD`` moves from where it points now.

        Until the run commits, ``HEAD`` stays where it is: a run that
        fails first leaves the history as it was.

        Args:
            commit_hash: The commit the run's commits will be parented on.
            expect_head: Expected current HEAD for the compare-and-set on
                the first commit. Pass the value captured when choosing
                ``commit_hash`` to detect concurrent writers; omit to read
                HEAD fresh (for intentional forks).
        """
        self._parent = commit_hash
        self._head = expect_head if expect_head is not None else await self.get_ref(HEAD_REF)
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
    ) -> FlowCommit:
        """Persist a full snapshot of the run as a commit at ``node_id``; return it.

        The snapshot holds every live scope of the run (see
        :func:`put_snapshot`), not only the ones above ``current_state``.
        :class:`CommitMeta` pins ``(node_path, iteration)``, the path of the
        scope ``current_state`` is, and the provenance triple
        (``produced_by``, ``trace_ref``, ``outcome``). :meth:`append_commit`
        then chains the commit onto ``HEAD``.

        ``node_path`` is the ``"/"``-joined ancestor chain (from run
        root to this node, inclusive). Node ids are hex, with no ``"/"``,
        so split round-trips on resume.

        ``outcome`` records why the commit fired — ``"ok"`` for a
        policy save or ``ctx.checkpoint()``, ``"halted"`` for the run's
        halt checkpoint.
        """
        tree = self.put_snapshot(scopes)
        node_path = "/".join(ancestor_chain + (node_id,))
        meta = self._build_commit_meta(node_path, iteration, node_id, outcome, trace_ref)
        meta = dataclasses.replace(meta, scope_path=path_str(scopes.path_of(current_state)))
        return await self.append_commit(tree.id, meta)

    def put_snapshot(self, scopes: ScopeRegistry) -> Tree:
        """Put the snapshot tree of every live scope in ``scopes``; return its root tree.

        Every scope is serialized in one synchronous pass
        (:meth:`ScopeRegistry.capture`) and the objects are stored without
        yielding to other tasks, so the snapshot holds the scopes as they
        were at one moment even while concurrent map items run. They are
        stored as one batch (:meth:`appinfra.cas.Repo.batch`): one link
        lookup and one store write for the whole snapshot. Blobs and trees
        this run already stored are left out of it.

        The root tree also holds the run's flow structure
        (:data:`~llm_gent.flow.state.snapshot.FLOW`), whose
        id every commit of the run records as ``flow_root_hash``.
        """
        flat = scopes.capture()
        with self.repo.batch() as batch:
            tree, queued = build_snapshot_tree(
                flat, self._run_structure_blob(), batch, self._written
            )
        # After the block: a batch that raised stored nothing.
        self._written |= queued
        return tree

    async def save_completion_commit(self, tree: Tree) -> FlowCommit:
        """Commit an already-put ``tree`` as the final-state commit (``$end``, outcome ``ok``)."""
        meta = self._build_commit_meta(END_NODE_PATH, 0, COMPLETION_PRODUCER, "ok", ())
        return await self.append_commit(tree.id, meta)

    # --- private helpers used by save_scope_commit ---

    def _run_structure_blob(self) -> Blob:
        """The owning flow's structure as a blob, taken at the run's first use."""
        if self._run_structure is None:
            self._run_structure = self._structure().blob()
        return self._run_structure

    def _build_commit_meta(
        self,
        node_path: str,
        iteration: int,
        node_id: str,
        outcome: CommitOutcome,
        trace_ref: tuple[TraceRef, ...],
    ) -> CommitMeta:
        """Assemble :class:`CommitMeta` for one commit.

        ``produced_by`` records the node's ``node_id`` — verb-level
        attribution (``verb_name`` / ``role`` / ``result_hash``)
        lands with the SAIA-verb-wrapper wiring. ``flow_root_hash`` is
        the id of the owning flow's structure, taken once per run.
        """
        from llm_gent import __version__

        return CommitMeta(
            node_path=node_path,
            iteration=iteration,
            produced_by=ProducedBy(node_id=node_id, verb_name=None, role=None, result_hash=None),
            trace_ref=trace_ref,
            outcome=outcome,
            flow_root_hash=self._run_structure_blob().id,
            timestamp_iso=datetime.now(UTC).isoformat(),
            framework_version=__version__,
        )
