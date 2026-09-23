# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Content-addressed state model — the CAS substrate.

Blob / Tree / Commit form a Git-shaped provenance store for gent
trajectories. State at every save-point becomes a :class:`Blob` keyed by
its content hash; an ordered :class:`Tree` of scope entries binds those
blobs to the state stack; a :class:`Commit` stamps the tree with its
position in the trajectory (``client_flow_id``, ``node_path``,
``iteration``) and provenance metadata (``produced_by``, ``trace_ref``,
``outcome``, ``flow_root_id``).

This module ships only the object model and the hash discipline.
Persistence (the :class:`CheckpointStore` Protocol redesign) and the
executor save/resume plumbing are separate concerns.

Hash discipline
---------------
- All content hashes are :func:`blake2b` with ``digest_size=32``.
- Content-only — no ``client_flow_id`` / ``node_path`` / ``iteration``
  salt in the hash. Two identical byte payloads across trajectories
  produce the same blob hash; cross-trajectory diff depends on this.
- Canonical serialization is versioned via
  :attr:`CommitMeta.framework_version` so a canonicalization change in a
  later gent version does not invalidate prior blobs — old commits stay
  valid within their own version.

Direct-save commits (consumer sites that save state outside a Flow
iterate boundary — e.g. an initial-plan write) use a ``$external/*``
prefix on :attr:`ProducedBy.node_id` so a downstream trajectory-walker
can filter without a schema-aware parser.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, Literal


_DIGEST_SIZE = 32
"""Byte length of every :func:`content_hash` output — blake2b truncated
to 256 bits. Matches the existing content-addressed node-id hash length
used throughout :mod:`llm_gent.flow`.
"""


def content_hash(data: bytes) -> str:
    """Return the blake2b hex digest of ``data`` at :data:`_DIGEST_SIZE`.

    Deterministic on ``data`` alone; no salt, no keying. Same bytes → same
    hash across trajectories, which is the invariant cross-trajectory
    diff depends on.
    """
    return blake2b(data, digest_size=_DIGEST_SIZE).hexdigest()


def canonical_json(obj: Any) -> bytes:
    """Serialize ``obj`` to canonical UTF-8 JSON bytes.

    Deterministic ordering via ``sort_keys=True`` and compact separators;
    the resulting bytes hash stably for any dict / list / primitive shape
    that survives :func:`json.dumps`. Used by :meth:`Tree.from_entries`
    and :meth:`Commit.build` for content-addressed hashing of their
    structural bodies.

    Not for :class:`Blob` payloads: Blobs are hashed over the caller's
    already-serialized state bytes (produced upstream via
    :func:`state_converter.unstructure` → :func:`json.dumps`), not by
    round-tripping through this helper. The layering keeps state-payload
    canonicalization owned by the state module.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


@dataclass(frozen=True)
class Blob:
    """A content-addressed payload — one scope's serialized data.

    :attr:`content_hash` MUST equal ``content_hash(payload)``; build only
    via :meth:`from_bytes` so the invariant holds.
    """

    content_hash: str
    payload: bytes

    @classmethod
    def from_bytes(cls, payload: bytes) -> Blob:
        """Build a Blob whose hash is derived from ``payload``."""
        return cls(content_hash=content_hash(payload), payload=payload)


TreeEntryKind = Literal["blob", "tree"]
"""Kind slot on :class:`TreeEntry` — a scope-level entry points at either
a leaf blob (that scope's serialized data) or a subtree (nested scope
tree). Closed enum: the recursion has two shapes, no third.
"""


@dataclass(frozen=True)
class TreeEntry:
    """One ``(scope_id, kind, child_hash)`` triple inside a :class:`Tree`.

    :attr:`scope_id` is the scope's stable identifier — typically the
    scope-defining node's content-addressed id (as computed elsewhere in
    :mod:`llm_gent.flow`). :attr:`kind` discriminates between a leaf blob
    and a subtree, mirroring git's tree-entry model.
    """

    scope_id: str
    kind: TreeEntryKind
    child_hash: str


@dataclass(frozen=True)
class Tree:
    """A content-addressed ordered map from ``scope_id`` to child hash.

    Entries are always sorted by :attr:`TreeEntry.scope_id` — canonical
    ordering is what lets two structurally identical trees produced by
    different runs hash the same. Build only via :meth:`from_entries` so
    the invariant holds.
    """

    content_hash: str
    entries: tuple[TreeEntry, ...]

    @classmethod
    def from_entries(cls, entries: Iterable[TreeEntry]) -> Tree:
        """Build a Tree from an unordered ``entries`` iterable.

        Sorts by :attr:`TreeEntry.scope_id` before hashing so the same
        entry set produced in any order hashes the same.
        """
        sorted_entries = tuple(sorted(entries, key=lambda e: e.scope_id))
        if any(
            prev.scope_id == curr.scope_id
            for prev, curr in zip(sorted_entries, sorted_entries[1:], strict=False)
        ):
            raise ValueError("Tree entries must have unique scope_id values")
        return cls(
            content_hash=content_hash(canonical_json(_tree_body(sorted_entries))),
            entries=sorted_entries,
        )

    def to_bytes(self) -> bytes:
        """Return the canonical byte payload — what a :class:`CheckpointStore`
        stores at ``kind="tree"``. Round-trips through :meth:`from_bytes`.
        """
        return canonical_json(_tree_body(self.entries))

    @classmethod
    def from_bytes(cls, payload: bytes) -> Tree:
        """Reconstruct a Tree from its canonical byte payload.

        Parses the JSON body, rebuilds :class:`TreeEntry` triples, and
        computes content_hash from the re-canonicalized body — same
        bytes in → same content_hash out as the writer produced.
        """
        body = json.loads(payload.decode("utf-8"))
        entries = tuple(TreeEntry(scope_id=e[0], kind=e[1], child_hash=e[2]) for e in body)
        return cls(
            content_hash=content_hash(canonical_json(_tree_body(entries))),
            entries=entries,
        )


@dataclass(frozen=True)
class ProducedBy:
    """Provenance attribution on one commit — which verb wrote this state.

    :attr:`node_id` is the content-addressed id of the scope-defining
    node whose body produced the commit, or a ``$external/*`` prefix for
    consumer-driven direct saves (e.g. an initial-plan write) that
    happen outside an iterate boundary.

    :attr:`verb_name` is the ``@verb`` callable's ``__name__`` when the
    commit came from an in-flow verb; ``None`` for direct saves.

    :attr:`role` is the verb's bound role, per-commit (not per-
    trajectory) — a run that dispatches multiple verbs under different
    roles attributes each commit to the role that owned that verb.

    :attr:`result_hash` is :func:`content_hash` of the verb's serialized
    return value (used by iterate bodies threading ``prev_result``).
    ``None`` when there is no result — direct saves, setup commits, verbs
    that return ``None``.
    """

    node_id: str
    verb_name: str | None
    role: str | None
    result_hash: str | None


@dataclass(frozen=True)
class TraceRef:
    """A cross-system trace pointer stamped into :class:`CommitMeta`.

    :attr:`kind` is an open string — starting values ``"verb_trace"``
    (a verb trace id) and ``"otel_span"`` (an OpenTelemetry span id,
    e.g. for EU AI Act cross-system correlation). Consumers may add new
    kinds; the framework does not inspect the value.

    :attr:`id` is the pointer itself — the id of the referenced record
    in whatever system :attr:`kind` names.
    """

    kind: str
    id: str


CommitOutcome = Literal["ok", "failed", "halted"]
"""The three iteration outcomes that MAY be committed:

- ``"ok"`` — iteration body ran to completion.
- ``"failed"`` — iteration raised.
- ``"halted"`` — iteration was interrupted by an ambient halt.

Closed enum; the accountability model needs exactly these three states.
"""


@dataclass(frozen=True)
class CommitMeta:
    """Framework-owned metadata stamped into every :class:`Commit`.

    See the module docstring for the hash-discipline rules that apply to
    every field participating in the commit's identity hash.
    """

    client_flow_id: str
    node_path: str
    iteration: int
    produced_by: ProducedBy
    trace_ref: tuple[TraceRef, ...]
    outcome: CommitOutcome
    flow_root_id: str
    timestamp_iso: str
    framework_version: str


@dataclass(frozen=True)
class Commit:
    """A trajectory timepoint — root tree, parent chain, provenance meta.

    :attr:`parent_hashes` is single-parent (linear history) in the common
    case; a multi-parent tuple is reserved for a future merge/branch
    surface and unused today.

    Build only via :meth:`build` so :attr:`content_hash` stays consistent
    with the canonical serialization of the body.
    """

    content_hash: str
    root_tree_hash: str
    parent_hashes: tuple[str, ...]
    meta: CommitMeta

    @classmethod
    def build(
        cls,
        *,
        root_tree_hash: str,
        parent_hashes: tuple[str, ...],
        meta: CommitMeta,
    ) -> Commit:
        """Build a Commit whose hash is derived from the canonical body."""
        body = _commit_body(root_tree_hash, parent_hashes, meta)
        return cls(
            content_hash=content_hash(canonical_json(body)),
            root_tree_hash=root_tree_hash,
            parent_hashes=parent_hashes,
            meta=meta,
        )

    def to_bytes(self) -> bytes:
        """Return the canonical byte payload — what a :class:`CheckpointStore`
        stores at ``kind="commit"``. Round-trips through :meth:`from_bytes`.
        """
        return canonical_json(_commit_body(self.root_tree_hash, self.parent_hashes, self.meta))

    @classmethod
    def from_bytes(cls, payload: bytes) -> Commit:
        """Reconstruct a Commit from its canonical byte payload.

        Parses the JSON body, rebuilds :class:`CommitMeta` +
        :class:`ProducedBy` + :class:`TraceRef`, and computes
        content_hash from the re-canonicalized body — same bytes in →
        same content_hash out as the writer produced.
        """
        body = json.loads(payload.decode("utf-8"))
        meta = _parse_commit_meta(body["meta"])
        root_tree_hash = body["root_tree_hash"]
        parent_hashes = tuple(body["parent_hashes"])
        return cls(
            content_hash=content_hash(
                canonical_json(_commit_body(root_tree_hash, parent_hashes, meta))
            ),
            root_tree_hash=root_tree_hash,
            parent_hashes=parent_hashes,
            meta=meta,
        )


def _parse_commit_meta(meta_body: dict[str, Any]) -> CommitMeta:
    """Rebuild :class:`CommitMeta` from its canonical JSON body."""
    produced = meta_body["produced_by"]
    return CommitMeta(
        client_flow_id=meta_body["client_flow_id"],
        node_path=meta_body["node_path"],
        iteration=meta_body["iteration"],
        produced_by=ProducedBy(
            node_id=produced["node_id"],
            verb_name=produced.get("verb_name"),
            role=produced.get("role"),
            result_hash=produced.get("result_hash"),
        ),
        trace_ref=tuple(
            TraceRef(kind=r["kind"], id=r["id"]) for r in meta_body.get("trace_ref", [])
        ),
        outcome=meta_body["outcome"],
        flow_root_id=meta_body["flow_root_id"],
        timestamp_iso=meta_body["timestamp_iso"],
        framework_version=meta_body["framework_version"],
    )


def _tree_body(entries: tuple[TreeEntry, ...]) -> list[list[str]]:
    """Canonicalize the entries list for hashing / serialization."""
    return [[e.scope_id, e.kind, e.child_hash] for e in entries]


def _commit_body(
    root_tree_hash: str,
    parent_hashes: tuple[str, ...],
    meta: CommitMeta,
) -> dict[str, Any]:
    """Canonicalize a commit's identity-determining fields for hashing.

    The body is ``(root_tree_hash, parent_hashes, meta)`` — every field on
    the commit that participates in identity. :attr:`Commit.content_hash`
    itself is derived from this body, so it is not in the body.
    """
    return {
        "root_tree_hash": root_tree_hash,
        "parent_hashes": list(parent_hashes),
        "meta": _meta_body(meta),
    }


def _meta_body(meta: CommitMeta) -> dict[str, Any]:
    """Flatten :class:`CommitMeta` to a canonical dict."""
    return {
        "client_flow_id": meta.client_flow_id,
        "node_path": meta.node_path,
        "iteration": meta.iteration,
        "produced_by": {
            "node_id": meta.produced_by.node_id,
            "verb_name": meta.produced_by.verb_name,
            "role": meta.produced_by.role,
            "result_hash": meta.produced_by.result_hash,
        },
        "trace_ref": [{"kind": r.kind, "id": r.id} for r in meta.trace_ref],
        "outcome": meta.outcome,
        "flow_root_id": meta.flow_root_id,
        "timestamp_iso": meta.timestamp_iso,
        "framework_version": meta.framework_version,
    }
