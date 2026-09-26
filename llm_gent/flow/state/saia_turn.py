# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Framework state for SAIA turn pause/resume.

Groups three concerns the halt-save + resume path shares across
``Flow`` / ``Loop`` / executor: the pending-side dict pair keyed by
Loop ``node_id`` (captured on pause, drained on halt-save), the
symmetric resume-side dict populated on resume hydration, and the
canonical ``{"task", "conversation"}`` envelope shape both sides
serialize through. Prior to this module those lived as ``dict``s on
:class:`Flow` with parallel writers/readers scattered across
modules; the classes here own the invariants (parallel-dict lockstep,
ancestry match on ``owns``, snapshot-during-iteration) that were
implicit before.

Not re-exported from :mod:`llm_gent.flow.state` — internal callers
import from :mod:`llm_gent.flow.state.saia_turn` explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .cas import Blob, Commit, TraceRef, canonical_json


if TYPE_CHECKING:
    from .._checkpoint_ctx import CheckpointContext


@dataclass
class PendingSaiaTurns:
    """Paused SAIA turns awaiting a halt-save.

    :class:`Loop` deposits an entry via :meth:`add` on pause; the
    halt-observation site drains via :meth:`snapshot` +
    :meth:`remove`. Two dicts are kept in lockstep — the payload
    keyed by the Loop's own ``node_id`` and the tuple of ancestor
    ``_Node`` ids captured at pause time so :meth:`owns` can match
    on ancestry for nested Loops (Loop paused inside an iterate
    body inside a chain step, where the pending entry's key is the
    Loop's descendant id computed under the chain step's descent
    context).
    """

    _bytes: dict[str, bytes] = field(default_factory=dict)
    _ancestors: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def add(self, node_id: str, payload: bytes, ancestors: tuple[str, ...]) -> None:
        """Record a paused Loop's payload + ancestor chain."""
        self._bytes[node_id] = payload
        self._ancestors[node_id] = ancestors

    def remove(self, node_id: str) -> None:
        """Drop both parallel entries for ``node_id`` if present."""
        self._bytes.pop(node_id, None)
        self._ancestors.pop(node_id, None)

    def owns(self, node_id: str) -> bool:
        """True when ``node_id`` is a pending Loop's own id or an ancestor of one.

        Direct match covers the ``.call(loop_verb)`` case. Ancestry
        match covers nested Loops — pending entry's key is the Loop's
        descendant id computed under the chain step's descent context.
        """
        if node_id in self._bytes:
            return True
        return any(node_id in chain for chain in self._ancestors.values())

    def snapshot(self) -> list[tuple[str, bytes]]:
        """Snapshot ``(node_id, payload)`` pairs for iteration.

        Callers awaiting ``put_object`` for each entry must NOT
        iterate the live dict — concurrent ``.map`` items can mutate
        it mid-iteration and raise :class:`RuntimeError`.
        """
        return list(self._bytes.items())

    def __bool__(self) -> bool:
        return bool(self._bytes)

    def clear(self) -> None:
        """Reset both dicts. Called at the top of :meth:`Flow.run`."""
        self._bytes.clear()
        self._ancestors.clear()

    async def stash_to_ctx(
        self,
        ctx: CheckpointContext,
    ) -> tuple[tuple[TraceRef, ...], tuple[str, ...]]:
        """Persist a snapshot via ``ctx``; return TraceRefs + stashed node_ids.

        Each entry becomes a standalone :class:`Blob` under the
        trajectory and yields one
        ``TraceRef(kind="saia_turn", id=f"{node_id}:{blob_hash}")``
        for the caller to stamp on the halt commit's meta; the
        compound id lets the resume side route each blob back to the
        Loop that produced it. Returns ``((), ())`` when nothing is
        pending.

        Iterates over :meth:`snapshot`, not the live dict, so
        concurrent ``.map`` items depositing new entries during the
        awaited put-blob calls don't raise :class:`RuntimeError`.
        Late arrivals stay pending; the caller drops only the
        returned ``stashed_ids`` after the halted commit is durable
        so a within-run retry can re-emit them. Blob puts are
        content-addressed — a repeat write for the same blob is a
        no-op.
        """
        if not self._bytes:
            return (), ()
        snapshot = self.snapshot()
        refs: list[TraceRef] = []
        for node_id, payload in snapshot:
            blob = Blob.from_bytes(payload)
            await ctx.put_blob(blob.content_hash, blob.payload)
            refs.append(TraceRef(kind="saia_turn", id=f"{node_id}:{blob.content_hash}"))
        return tuple(refs), tuple(node_id for node_id, _ in snapshot)


@dataclass
class ResumeSaiaTurns:
    """Reconstructed SAIA turn payloads awaiting Loop pickup on resume.

    :meth:`Flow._load_resume_saia_turn_bytes` populates via :meth:`add`
    from the halt commit's ``saia_turn`` :class:`TraceRef` entries;
    :class:`Loop` reads via :meth:`load` at dispatch and drops via
    :meth:`release` only after ``saia.complete`` returns (so a
    rescue-then-iterate-retry re-consumes the same envelope).
    """

    _bytes: dict[str, bytes] = field(default_factory=dict)

    def add(self, node_id: str, payload: bytes) -> None:
        """Register a reconstructed payload for the Loop at ``node_id``."""
        self._bytes[node_id] = payload

    def load(self, node_id: str) -> bytes | None:
        """Non-consuming read; ``None`` when no entry is registered."""
        return self._bytes.get(node_id)

    def release(self, node_id: str) -> None:
        """Drop the entry for ``node_id`` if present."""
        self._bytes.pop(node_id, None)

    def clear(self) -> None:
        """Reset the dict. Called at the top of :meth:`Flow.run`."""
        self._bytes.clear()

    async def load_from_commit(
        self,
        ctx: CheckpointContext,
        commit: Commit,
    ) -> None:
        """Populate resume entries from ``commit``'s saia_turn TraceRefs.

        Each :class:`TraceRef` with ``kind="saia_turn"`` on the
        halted commit was stamped by :meth:`PendingSaiaTurns.stash_to_ctx`
        with ``id=f"{node_id}:{blob_hash}"``. Split on the first
        colon, fetch the blob under ``blob_hash`` from ``ctx``, and
        register ``{node_id: blob_bytes}`` so the Loop at that node
        can pick its own entry up on first dispatch and hand the
        reconstructed conversation to SAIA with ``resume=True``.

        Silently skips entries whose blob is missing from the store
        — the run then falls back to a fresh dispatch at that Loop.
        Entries whose ``id`` is not in ``node_id:blob_hash`` shape
        are ignored (defensive against future ``saia_turn`` variants
        this framework does not understand).
        """
        for ref in commit.meta.trace_ref:
            if ref.kind != "saia_turn":
                continue
            node_id, sep, blob_hash = ref.id.partition(":")
            if not sep or not node_id or not blob_hash:
                continue
            payload = await ctx.get_object("blob", blob_hash)
            if payload is None:
                continue
            self.add(node_id, payload)


@dataclass(frozen=True)
class SaiaTurnEnvelope:
    """Canonical ``{"task", "conversation"}`` envelope for a paused SAIA turn.

    Single owner of the shape both :meth:`Loop._capture_paused` writes
    and :meth:`Loop._consume_resume_entry` reads. ``conversation`` is
    the ``to_dict()`` payload from the paused
    ``SerializableConversationLike``; reconstruction back into a
    ``Conversation`` is the :class:`ConversationFactory`'s job, not
    this envelope's.
    """

    task: str
    conversation: dict[str, Any]

    def to_bytes(self) -> bytes:
        """Serialize to canonical JSON bytes for CAS blob storage."""
        return canonical_json({"task": self.task, "conversation": self.conversation})

    @classmethod
    def from_bytes(cls, payload: bytes) -> SaiaTurnEnvelope:
        """Decode canonical JSON bytes; raises on missing keys."""
        data = json.loads(payload)
        return cls(task=data["task"], conversation=data["conversation"])
