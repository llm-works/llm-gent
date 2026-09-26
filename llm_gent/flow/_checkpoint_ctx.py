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
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .checkpoint import CheckpointStore, Kind, Retention, maybe_await
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
    Optional lives at the context level, not per-field). Exposes
    the put-object triad, ref put/resolve, get_object,
    gc_trajectory, and the compound :meth:`save_scope_commit` that
    assembles a Blob→Tree→Commit chain and refs it at the boundary.
    """

    def __init__(self, store: CheckpointStore, client_flow_id: str) -> None:
        self.store = store
        self.client_flow_id = client_flow_id

    @property
    def retention(self) -> Retention:
        """The store's retention mode (``"retain"`` or ``"gc_on_success"``)."""
        return self.store.retention

    # --- store passthrough (kind-specific put_object variants) ---

    async def put_blob(self, content_hash: str, payload: bytes) -> None:
        """Put a blob under this trajectory's ``client_flow_id``."""
        await maybe_await(self.store.put_object(self.client_flow_id, "blob", content_hash, payload))

    async def put_tree(self, tree: Tree) -> None:
        """Serialize + put a Tree under this trajectory."""
        await maybe_await(
            self.store.put_object(self.client_flow_id, "tree", tree.content_hash, tree.to_bytes())
        )

    async def put_commit(self, commit: Commit) -> None:
        """Serialize + put a Commit under this trajectory."""
        await maybe_await(
            self.store.put_object(
                self.client_flow_id, "commit", commit.content_hash, commit.to_bytes()
            )
        )

    async def put_ref(self, node_path: str, iteration: int, commit_hash: str) -> None:
        """Point ``(client_flow_id, node_path, iteration)`` at ``commit_hash``."""
        await maybe_await(
            self.store.put_ref(self.client_flow_id, node_path, iteration, commit_hash)
        )

    async def resolve_ref(self) -> str | None:
        """Latest commit hash across this trajectory, or ``None``."""
        result: str | None = await maybe_await(self.store.resolve_ref(self.client_flow_id))
        return result

    async def get_object(self, kind: Kind, content_hash: str) -> bytes | None:
        """Fetch an object under this trajectory by kind + hash."""
        result: bytes | None = await maybe_await(
            self.store.get_object(self.client_flow_id, kind, content_hash)
        )
        return result

    async def gc_trajectory(self) -> None:
        """Remove every object and ref under this ``client_flow_id``."""
        await maybe_await(self.store.gc_trajectory(self.client_flow_id))

    # --- compound: save a scope commit ---

    async def save_scope_commit(
        self,
        ancestor_chain: tuple[str, ...],
        iteration: int,
        node_id: str,
        current_state: State[Any],
        outcome: CommitOutcome,
        trace_ref: tuple[TraceRef, ...] = (),
    ) -> None:
        """Persist a content-addressed scope commit at ``node_id``.

        Walks the scope stack from root to ``current_state``. For
        each scope: serialize its ``data`` via the state-data
        contract (dict passthrough or ``StateData.to_dict``) to
        canonical JSON bytes and :meth:`put_blob` a Blob keyed by
        content hash. Bundle every scope's blob hash into a Tree
        (one :class:`TreeEntry` per scope, ordered by depth via a
        two-digit ``scope_id``). Wrap the Tree in a Commit whose
        :class:`CommitMeta` pins ``(client_flow_id, node_path,
        iteration)`` and the provenance triple (``produced_by``,
        ``trace_ref``, ``outcome``). Finally :meth:`put_ref` points
        this boundary at the commit hash.

        ``node_path`` is the ``"/"``-joined ancestor chain (from run
        root to this node, inclusive). blake2b hex has no ``"/"``,
        so split round-trips on resume.

        ``outcome`` records why the commit fired — ``"ok"`` for a
        successful iterate boundary, ``"halted"`` when the
        halt-observation site triggered the save.
        """
        scopes = self._collect_scope_stack(current_state)
        entries = await self._put_scope_blobs(scopes)
        tree = Tree.from_entries(entries)
        await self.put_tree(tree)
        node_path = "/".join(ancestor_chain + (node_id,))
        meta = self._build_commit_meta(node_path, iteration, node_id, outcome, trace_ref)
        commit = Commit.build(root_tree_hash=tree.content_hash, parent_hashes=(), meta=meta)
        await self.put_commit(commit)
        await self.put_ref(node_path, iteration, commit.content_hash)

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
        node_path: str,
        iteration: int,
        node_id: str,
        outcome: CommitOutcome,
        trace_ref: tuple[TraceRef, ...],
    ) -> CommitMeta:
        """Assemble :class:`CommitMeta` for one scope-commit save.

        ``produced_by`` records the node's ``node_id`` — verb-level
        attribution (``verb_name`` / ``role`` / ``result_hash``)
        lands with the SAIA-verb-wrapper wiring. ``flow_root_id`` is
        this trajectory's ``client_flow_id`` — a stable per-run
        identifier — until the framework computes a proper
        composition-tree root hash.
        """
        from llm_gent import __version__

        return CommitMeta(
            client_flow_id=self.client_flow_id,
            node_path=node_path,
            iteration=iteration,
            produced_by=ProducedBy(node_id=node_id, verb_name=None, role=None, result_hash=None),
            trace_ref=trace_ref,
            outcome=outcome,
            flow_root_id=self.client_flow_id,
            timestamp_iso=datetime.now(UTC).isoformat(),
            framework_version=__version__,
        )
