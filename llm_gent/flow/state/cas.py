# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Flow commits on :mod:`appinfra.cas` — the metadata a flow stamps into each commit.

A history lives in one repository of an :class:`appinfra.cas.Store`. At
every save point the run's live state becomes a tree of
:class:`~appinfra.cas.Tree` / :class:`~appinfra.cas.Blob` objects (the
layout is :mod:`llm_gent.flow.state.snapshot`'s), and an
:class:`~appinfra.cas.Commit` points at that tree. The commit's payload is
the :class:`CommitMeta` below as canonical JSON: the commit's position in
the history (``node_path``, ``iteration``, ``scope_path``) and its
provenance (``produced_by``, ``trace_ref``, ``outcome``,
``flow_root_hash``). :class:`FlowCommit` is a commit read back with its
metadata parsed.

Direct-save commits (consumer sites that save state outside a Flow
iterate boundary — e.g. an initial-plan write) use a ``$external/*``
prefix on :attr:`ProducedBy.node_id` so a downstream history-walker
can filter without a schema-aware parser.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from appinfra.cas import Commit, canonical_json


@dataclass(frozen=True)
class ProducedBy:
    """Provenance attribution on one commit — which verb wrote this state.

    :attr:`node_id` is the content-addressed id of the scope-defining
    node whose body produced the commit, a ``$external/*`` prefix for
    consumer-driven direct saves (e.g. an initial-plan write) that
    happen outside an iterate boundary, or ``$framework/*`` for commits
    the framework writes itself (the final-state commit on clean exit).

    :attr:`verb_name` is the ``@verb`` callable's ``__name__`` when the
    commit came from an in-flow verb; ``None`` for direct saves.

    :attr:`role` is the verb's bound role, per-commit (not per-
    history) — a run that dispatches multiple verbs under different
    roles attributes each commit to the role that owned that verb.

    :attr:`result_hash` is the content hash of the verb's serialized
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


CommitOutcome = Literal["ok", "halted"]
"""Why a commit was written:

- ``"ok"`` — a save point (policy, ``ctx.checkpoint()``) or the final state.
- ``"halted"`` — the run's halt checkpoint.

A run that raises writes no commit: resume continues from the last save.
"""


@dataclass(frozen=True)
class CommitMeta:
    """Framework-owned metadata stamped into every commit's payload.

    Serialized with :meth:`to_payload` as canonical JSON, so it takes part
    in the commit's id.
    """

    node_path: str
    iteration: int
    produced_by: ProducedBy
    trace_ref: tuple[TraceRef, ...]
    outcome: CommitOutcome
    flow_root_hash: str
    """Id of the blob holding the structure of the flow definition (the
    composition tree) — identical for every history running the same flow
    code. Equal ids guarantee identical node ids, so a ``node_path`` saved
    under one resolves under the other. Computed by the framework on every
    commit; empty only in hand-built commits."""
    timestamp_iso: str
    framework_version: str
    scope_path: str = ""
    """Snapshot path of the scope the saving node ran in (``""`` for the root
    scope); see :mod:`llm_gent.flow.state.snapshot`."""

    def to_payload(self) -> bytes:
        """Canonical JSON of the metadata: the payload of the commit it describes."""
        return canonical_json(
            {
                "node_path": self.node_path,
                "iteration": self.iteration,
                "produced_by": {
                    "node_id": self.produced_by.node_id,
                    "verb_name": self.produced_by.verb_name,
                    "role": self.produced_by.role,
                    "result_hash": self.produced_by.result_hash,
                },
                "trace_ref": [{"kind": r.kind, "id": r.id} for r in self.trace_ref],
                "outcome": self.outcome,
                "flow_root_hash": self.flow_root_hash,
                "timestamp_iso": self.timestamp_iso,
                "framework_version": self.framework_version,
                "scope_path": self.scope_path,
            }
        )

    @classmethod
    def from_payload(cls, payload: bytes) -> CommitMeta:
        """Rebuild the metadata from a commit's payload (:meth:`to_payload`'s output)."""
        body: dict[str, Any] = json.loads(payload.decode("utf-8"))
        produced = body["produced_by"]
        return cls(
            node_path=body["node_path"],
            iteration=body["iteration"],
            produced_by=ProducedBy(
                node_id=produced["node_id"],
                verb_name=produced.get("verb_name"),
                role=produced.get("role"),
                result_hash=produced.get("result_hash"),
            ),
            trace_ref=tuple(TraceRef(kind=r["kind"], id=r["id"]) for r in body["trace_ref"]),
            outcome=body["outcome"],
            flow_root_hash=body["flow_root_hash"],
            timestamp_iso=body["timestamp_iso"],
            framework_version=body["framework_version"],
            scope_path=body["scope_path"],
        )


@dataclass(frozen=True)
class FlowCommit:
    """One point in a flow's history: a commit with its :class:`CommitMeta` parsed.

    :attr:`parents` holds the history's previous head — the commit written
    immediately before this one, in time order — or is empty for a
    history's first commit. It records sequence, not derivation: a run
    started with ``resume="off"`` still parents on the previous head. A run
    that finishes or halts ends in an ``$end`` or ``halted`` commit; one
    that raises ends at its last save. A multi-parent tuple is reserved for
    a future fork/merge surface and unused today.
    """

    id: str
    """The commit's content hash: what ``ctx.checkpoint()`` returns and ``resume=`` takes."""
    tree: str
    """Id of the commit's snapshot tree."""
    parents: tuple[str, ...]
    meta: CommitMeta

    @classmethod
    def build(
        cls, tree: str, parents: tuple[str, ...], meta: CommitMeta
    ) -> tuple[FlowCommit, Commit]:
        """Return the flow commit of ``tree`` on ``parents`` and the store object that holds it."""
        commit = Commit(tree, parents, meta.to_payload())
        return cls(id=commit.id, tree=tree, parents=parents, meta=meta), commit

    @classmethod
    def from_commit(cls, commit: Commit) -> FlowCommit:
        """Read a stored commit back with its metadata parsed."""
        return cls(
            id=commit.id,
            tree=commit.tree,
            parents=commit.parents,
            meta=CommitMeta.from_payload(commit.payload),
        )
