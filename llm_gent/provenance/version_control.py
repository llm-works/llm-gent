# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Version control of agent runs on :mod:`appinfra.cas`: every step is a commit.

:class:`VersionControl` is a Flow hook (:mod:`llm_gent.flow.hooks`). Attached
with :meth:`Flow.with_hooks`, it commits a record of every chain step, every
model call of a Loop, every checkpoint and every run to one ref of one
repository: by default ``provenance`` in the repository a run's checkpoints
use, next to them. Verbs add records of their own with :meth:`VersionControl.record`
(``ctx.hook(VersionControl)``).

A record is a commit whose parent is the previous record, in the order they
were written:

- its tree holds the entities: ``inputs/<name>`` and ``outputs/<name>``. A
  value is a blob of its JSON form (the checkpoint codec's), so equal values
  are one object: a step's input that is the previous step's result names the
  same blob. A model call's context is a vector of its messages, so the
  messages two calls share are stored once.
- its payload (canonical JSON) says what happened: ``kind`` (``step``,
  ``llm_call``, ``checkpoint``, ``run`` or a verb's own), ``activity`` and
  ``parent`` (invocation ids), ``run``, ``ts``, ``actor``, ``where``,
  ``outcome``, ``decision``, ``error``, ``trace``, ``facets`` and, for a
  checkpoint, ``checkpoint`` (its commit id). A value with no JSON form is
  left out and named in ``omitted``.

The records are append-only and nothing in a run reads them: resume reads
only checkpoints. Records are written synchronously, so concurrent map items
never interleave one record's writes; a second process committing to the same
ref is retried on its new head.
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from appinfra.cas import (
    Blob,
    Commit,
    RefConflictError,
    Repo,
    Tree,
    TreeEntry,
    Vector,
    canonical_json,
)
from appinfra.cas.objects import TreeEntryType
from appinfra.cas.store import Batch
from appinfra.log import Logger

from ..flow.context import Context
from ..flow.hooks import (
    CheckpointEvent,
    Hook,
    HookEvent,
    LLMCallEvent,
    RunEvent,
    StepEvent,
    Where,
)
from ..flow.state.codec import encode


REF = "provenance"
"""The ref :class:`VersionControl` commits to unless told otherwise."""

INPUTS = "inputs"
OUTPUTS = "outputs"

_RETRIES = 3


@dataclass(frozen=True)
class Messages:
    """A list stored as a vector of one blob per element: a conversation's messages."""

    items: list[Any]


@dataclass(frozen=True)
class Record:
    """One record read back: its commit, its payload and its entities."""

    id: str
    parents: tuple[str, ...]
    meta: dict[str, Any]
    inputs: dict[str, TreeEntry] = field(default_factory=dict)
    outputs: dict[str, TreeEntry] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        """What the record is of: ``step``, ``llm_call``, ``checkpoint``, ``run`` or a verb's own."""
        kind: str = self.meta["kind"]
        return kind


class VersionControl(Hook):
    """Commits a record of every step, model call, checkpoint and run to ``ref`` in ``repo``."""

    def __init__(self, lg: Logger, repo: Repo, *, ref: str = REF) -> None:
        """Commit to ref ``ref`` of ``repo`` (``store.repo(name)``)."""
        self._lg = lg
        self.repo = repo
        self.ref = ref

    # --- hooks ---

    def after_run(self, event: RunEvent) -> None:
        """Record the run's end; ``resumed_from`` names the commit it started from."""
        extra = {"outcome": event.outcome, "resumed_from": event.run.resumed_from}
        self._commit(_meta("run", event, extra, error=event.error), {}, {})

    def after_step(self, event: StepEvent) -> None:
        """Record a step: its inputs, its result when it returned one, how it ended."""
        inputs = {str(i): arg for i, arg in enumerate(event.args)} | dict(event.kwargs)
        outputs = {"result": event.result} if event.outcome in ("ok", "denied") else {}
        decision = None if event.decision is None else dataclasses.asdict(event.decision)
        extra = {"outcome": event.outcome, "decision": decision}
        self._commit(_meta("step", event, extra, error=event.error), inputs, outputs)

    def after_llm_call(self, event: LLMCallEvent) -> None:
        """Record a model call: the context it was sent and its response.

        Runs while SAIA waits, before the response joins the conversation,
        so the context is what the model saw.
        """
        context = _messages(event.conversation)
        inputs = {} if context is None else {"context": context}
        self._commit(_meta("llm_call", event, {}), inputs, {"response": event.response})

    def after_checkpoint(self, event: CheckpointEvent) -> None:
        """Record that a checkpoint was written, naming its commit."""
        extra = {"checkpoint": event.commit_id, "outcome": event.outcome}
        self._commit(_meta("checkpoint", event, extra), {}, {})

    # --- explicit records ---

    def record(
        self,
        ctx: Context[Any],
        kind: str,
        *,
        inputs: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        facets: Mapping[str, Any] | None = None,
    ) -> str:
        """Commit a record of a verb's own, as an activity inside its step; return its id.

        ``inputs`` / ``outputs`` map names to values (a :class:`Messages`
        becomes a vector); ``facets`` are stored in the payload.
        """
        role = None if ctx.role is None else ctx.role.name
        meta = {
            "kind": kind,
            "activity": uuid.uuid4().hex,
            "parent": ctx.activity_id,
            "run": ctx.run_id,
            "ts": datetime.now(UTC).isoformat(),
            "actor": {"verb": None, "role": role, "model": None},
            "facets": _facets(facets or {}),
        }
        return self._commit(meta, dict(inputs or {}), dict(outputs or {}))

    # --- reading ---

    def head(self) -> str | None:
        """Id of the newest record, or ``None`` before the first."""
        return self.repo.get_ref(self.ref)

    def log(self) -> Iterator[Record]:
        """The records, newest first."""
        head = self.head()
        if head is None:
            return
        for commit in self.repo.first_parents(head):
            yield self._read(commit)

    def load(self, entry: TreeEntry) -> Any:
        """The value of one of a record's entities: a JSON value, or a list for a vector."""
        if entry.type == "vector":
            return [self._blob(element) for element in Vector(self.repo, entry.id)]
        return self._blob(entry.id)

    # --- internals ---

    def _commit(self, meta: dict[str, Any], inputs: dict[str, Any], outputs: dict[str, Any]) -> str:
        """Store the entities, then commit the record on the ref's head; return its id."""
        omitted: list[str] = []
        with self.repo.batch() as batch:
            tree = Tree(
                (
                    TreeEntry(INPUTS, "tree", _put_entities(batch, inputs, INPUTS, omitted)),
                    TreeEntry(OUTPUTS, "tree", _put_entities(batch, outputs, OUTPUTS, omitted)),
                )
            )
            tree_id = batch.put(tree)
        if omitted:
            meta = {**meta, "omitted": omitted}
        return self._commit_tree(tree_id, canonical_json(meta))

    def _commit_tree(self, tree_id: str, payload: bytes) -> str:
        """Commit ``tree_id`` on the ref; on a concurrent move, again on the new head."""
        for attempt in range(_RETRIES):
            try:
                return self.repo.commit(self.ref, tree_id, payload)
            except RefConflictError:
                if attempt == _RETRIES - 1:
                    raise
                self._lg.debug("provenance ref moved; committing on its new head")
        raise AssertionError("unreachable")

    def _read(self, commit: Commit) -> Record:
        """A stored record: its payload parsed and its entities by name."""
        tree = self.repo.get(commit.tree, Tree)
        assert tree is not None
        return Record(
            id=commit.id,
            parents=commit.parents,
            meta=json.loads(commit.payload),
            inputs=self._entries(tree, INPUTS),
            outputs=self._entries(tree, OUTPUTS),
        )

    def _entries(self, tree: Tree, name: str) -> dict[str, TreeEntry]:
        """The entries of the subtree ``name`` of a record's tree, by name."""
        entry = tree.get(name)
        sub = None if entry is None else self.repo.get(entry.id, Tree)
        return {} if sub is None else {e.name: e for e in sub.entries}

    def _blob(self, object_id: str) -> Any:
        """The JSON value a blob holds."""
        blob = self.repo.get(object_id, Blob)
        assert blob is not None
        return json.loads(blob.data)


# --- helpers ---


def _meta(
    kind: str, event: HookEvent, extra: dict[str, Any], error: BaseException | None = None
) -> dict[str, Any]:
    """The payload fields every hook-written record has, plus ``extra``."""
    return {
        "kind": kind,
        "activity": event.activity_id,
        "parent": event.parent_activity_id,
        "run": event.run.run_id,
        "ts": event.ts.isoformat(),
        "actor": dataclasses.asdict(event.actor),
        "where": _where(event.where),
        "trace": dataclasses.asdict(event.trace),
        "facets": _facets(event.facets),
        "error": None if error is None else f"{type(error).__name__}: {error}",
        **extra,
    }


def _where(where: Where) -> dict[str, Any]:
    """Where a record's event happened, as JSON; the flow by its name."""
    return {
        "flow": None if where.flow is None else (where.flow.name or None),
        "node_id": where.node_id,
        "node_path": list(where.node_path),
        "scope_path": where.scope_path,
        "kind": where.kind,
        "step_name": where.step_name,
    }


def _facets(facets: Mapping[str, Any]) -> dict[str, Any]:
    """Facets as JSON; one without a JSON form is stored as its ``repr``."""
    out: dict[str, Any] = {}
    for name, value in facets.items():
        try:
            out[str(name)] = _json(value)
        except TypeError:
            out[str(name)] = repr(value)
    return out


def _messages(conversation: Any) -> Messages | None:
    """A conversation's messages: a list, or the ``messages`` of its ``to_dict()``."""
    if conversation is None:
        return None
    if isinstance(conversation, list | tuple):
        return Messages(list(conversation))
    to_dict = getattr(conversation, "to_dict", None)
    data = to_dict() if callable(to_dict) else None
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        return Messages(data["messages"])
    return None


def _put_entities(batch: Batch, values: Mapping[str, Any], where: str, omitted: list[str]) -> str:
    """Put one subtree of named entities; names whose value has no JSON form go to ``omitted``."""
    entries: list[TreeEntry] = []
    for name, value in values.items():
        try:
            entry_type, object_id = _put_value(batch, value)
        except TypeError:
            omitted.append(f"{where}/{name}")
            continue
        entries.append(TreeEntry(name, entry_type, object_id))
    return batch.put(Tree(tuple(entries)))


def _put_value(batch: Batch, value: Any) -> tuple[TreeEntryType, str]:
    """Put a value: :class:`Messages` as a vector of blobs, anything else as one blob."""
    if isinstance(value, Messages):
        ids = [batch.put(Blob(canonical_json(_json(item)))) for item in value.items]
        return "vector", Vector.build(batch, ids).id
    return "blob", batch.put(Blob(canonical_json(_json(value))))


def _json(value: Any) -> Any:
    """The JSON form of ``value``: the checkpoint codec's, else a dataclass's fields.

    A dataclass the codec does not take (a model response) is stored as its
    fields, without ``raw``, the backend-native response.

    Raises:
        TypeError: ``value`` has no JSON form.
    """
    try:
        return encode(value, "provenance record")
    except TypeError:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                f.name: _json(getattr(value, f.name))
                for f in dataclasses.fields(value)
                if f.name != "raw"
            }
        if isinstance(value, list | tuple):
            return [_json(item) for item in value]
        if isinstance(value, dict):
            return {str(k): _json(v) for k, v in value.items()}
        raise
