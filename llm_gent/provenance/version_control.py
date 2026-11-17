# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Version control of agent runs on :mod:`appinfra.cas`: every step is a commit.

:class:`VersionControl` commits records of what an agent did to one ref of one
repository: by default ``provenance``, which can share a repository with a
Flow run's checkpoints. Commits are made two ways, and read back the same:

- **Explicitly**, with no Flow, SAIA or llm-infer required:
  :meth:`VersionControl.commit` commits one record;
  :meth:`VersionControl.activity` opens an :class:`Activity` that the records
  made inside it name as their parent, and that commits itself on exit;
  :meth:`VersionControl.llm_call` commits a model call from a message list (or
  a SAIA conversation) and any response value, and
  :meth:`VersionControl.llm_calls` is SAIA's ``on_iteration`` callback.
- **As a Flow hook** (:mod:`llm_gent.flow.hooks`): attached with
  :meth:`Flow.with_hooks`, it commits a record of every chain step, every model
  call of a Loop, every checkpoint and every run. A verb commits records of its
  own with ``within=ctx`` (``ctx.hook(VersionControl)``).

A record is a commit whose parent is the previous record, in the order they
were written:

- its tree holds the entities: ``inputs/<name>`` and ``outputs/<name>``. A
  value is a blob of its JSON form (the checkpoint codec's), so equal values
  are one object: a step's input that is the previous step's result names the
  same blob. The inputs of a step or an activity are encoded when it starts,
  so they are what it received even when it mutates them. A model call's
  context is a vector of its messages, so the messages two calls share are
  stored once.
- its payload (canonical JSON) says what happened: ``kind`` (``step``,
  ``llm_call``, ``checkpoint``, ``run`` or the caller's own), ``activity`` and
  ``parent`` (invocation ids), ``run`` (the root activity), ``ts``,
  ``actor``, ``facets``, ``trace`` and, where they apply, ``outcome``,
  ``error``, ``where``, ``decision`` and ``checkpoint`` (a checkpoint's commit
  id). A value with no JSON form is left out and named in ``omitted``, as is a
  step's keyword argument named like one of its positional indexes.

The records are append-only and nothing in a run reads them: resume reads
only checkpoints. Records are written synchronously, so concurrent map items
never interleave one record's writes; a second process committing to the same
ref is retried on its new head.
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import TracebackType
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
    Actor,
    CheckpointEvent,
    Hook,
    HookEvent,
    LLMCallEvent,
    RunEvent,
    StepEvent,
    TraceIds,
    Where,
)
from ..flow.state.codec import encode


REF = "provenance"
"""The ref :class:`VersionControl` commits to unless told otherwise."""

INPUTS = "inputs"
OUTPUTS = "outputs"

_RETRIES = 3

_NOT_STORED = frozenset({"raw", "request"})
"""Fields of a response dataclass left out of its record: the backend-native
response, and llm-infer's back-reference to the request (its messages are the
call's context, stored once as a vector)."""


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
        """What the record is of: ``step``, ``llm_call``, ``checkpoint``, ``run`` or the caller's."""
        kind: str = self.meta["kind"]
        return kind


@dataclass(frozen=True)
class _Scope:
    """Where an explicit record sits: its parent activity, its run, its role."""

    parent: str | None = None
    run: str | None = None
    role: str | None = None


class Activity:
    """An activity in progress, opened by :meth:`VersionControl.activity`.

    Records made through it (:meth:`commit`, :meth:`llm_call`,
    :meth:`llm_calls`, :meth:`activity`) name it as their parent. Used as a
    context manager it commits itself on exit, after the records made inside
    it: with the inputs it was opened with, encoded then, :attr:`outputs` and
    :attr:`facets` as they are on exit, and outcome ``ok``, ``failed`` (an
    exception escaped) or ``interrupted`` (a cancellation or other
    ``BaseException``).
    """

    def __init__(
        self, vc: VersionControl, kind: str, scope: _Scope, actor: Actor, inputs: _Entities
    ) -> None:
        """Open activity ``kind`` in ``scope``; the root of a run when it has no parent."""
        self._vc = vc
        self.kind = kind
        self.id = uuid.uuid4().hex
        self.parent = scope.parent
        self.run = scope.run or self.id
        self.actor = actor
        self.outputs: dict[str, Any] = {}
        self.facets: dict[str, Any] = {}
        self._inputs = inputs
        self._role = scope.role

    # --- records inside this activity ---

    def commit(self, kind: str, **fields: Any) -> str:
        """:meth:`VersionControl.commit` inside this activity."""
        return self._vc.commit(kind, within=self, **fields)

    def activity(self, kind: str, **fields: Any) -> Activity:
        """:meth:`VersionControl.activity` inside this activity."""
        return self._vc.activity(kind, within=self, **fields)

    def llm_call(self, context: Any, response: Any, **fields: Any) -> str:
        """:meth:`VersionControl.llm_call` inside this activity."""
        return self._vc.llm_call(context, response, within=self, **fields)

    def llm_calls(
        self, conversation: Any, *, role: str | None = None
    ) -> Callable[[int, Any], Awaitable[None]]:
        """:meth:`VersionControl.llm_calls` inside this activity."""
        return self._vc.llm_calls(conversation, role=role, within=self)

    # --- context manager ---

    def __enter__(self) -> Activity:
        """Return the activity; it commits itself on exit."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Commit the activity with how it ended; an exception from its body propagates.

        A record that fails to commit raises, unless the body raised: then it
        is logged, so it does not replace the body's exception.
        """
        if exc is None:
            outcome = "ok"
        elif isinstance(exc, Exception):
            outcome = "failed"
        else:
            outcome = "interrupted"
        try:
            self._vc._end(self, outcome, exc)
        except Exception as e:
            if exc is None:
                raise
            self._vc._lg.warning(
                "activity record not committed", extra={"activity": self.kind, "exception": e}
            )

    def _scope(self) -> _Scope:
        """The scope of records made inside this activity."""
        return _Scope(parent=self.id, run=self.run, role=self.actor.role or self._role)


class VersionControl(Hook):
    """Commits records of what an agent did to ``ref`` in ``repo``: explicitly, or as a Flow hook."""

    def __init__(self, lg: Logger, repo: Repo, *, ref: str = REF) -> None:
        """Commit to ref ``ref`` of ``repo`` (``store.repo(name)``)."""
        self._lg = lg
        self.repo = repo
        self.ref = ref
        # Inputs of the steps running now, encoded before each ran; by activity id.
        self._received: dict[str, _Entities] = {}

    # --- explicit records ---

    def commit(
        self,
        kind: str,
        *,
        inputs: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        actor: Actor | None = None,
        facets: Mapping[str, Any] | None = None,
        trace: TraceIds | None = None,
        within: Activity | Context[Any] | None = None,
    ) -> str:
        """Commit one record of ``kind``; return its commit id.

        ``inputs`` / ``outputs`` map names to values (a :class:`Messages`
        becomes a vector); ``facets`` are stored in the payload. ``within``
        places the record: inside an :class:`Activity`, inside a Flow step
        (its ``ctx``), or, when ``None``, on its own.
        """
        scope = _scope(within)
        meta = _explicit_meta(kind, uuid.uuid4().hex, scope, _actor(actor, scope), facets, trace)
        return self._commit(meta, dict(inputs or {}), dict(outputs or {}))

    def activity(
        self,
        kind: str,
        *,
        inputs: Mapping[str, Any] | None = None,
        actor: Actor | None = None,
        within: Activity | Context[Any] | None = None,
    ) -> Activity:
        """Open an :class:`Activity` of ``kind``, placed like :meth:`commit`'s ``within``.

        ``inputs`` are encoded now, so the activity's record holds them as
        received. With no ``within`` it is the root of a run: its id is the
        ``run`` of every record made inside it.
        """
        scope = _scope(within)
        encoded = _entities(dict(inputs or {}), INPUTS)
        return Activity(self, kind, scope, _actor(actor, scope), encoded)

    def llm_call(
        self,
        context: Any,
        response: Any,
        *,
        actor: Actor | None = None,
        facets: Mapping[str, Any] | None = None,
        within: Activity | Context[Any] | None = None,
    ) -> str:
        """Commit a model call: the ``context`` it was sent and its ``response``.

        ``context`` is the message list (any JSON values, as sent), or a
        conversation object: one whose ``to_dict()["messages"]`` is that list,
        or SAIA's ``ConversationLike`` (``as_messages()``). It is stored as a
        vector of its messages; one that cannot be read is named in
        ``omitted``. ``response`` is any value with a JSON form, or a
        dataclass (``raw`` and ``request`` left out). The actor's model
        defaults to the response's ``model``.
        """
        scope = _scope(within)
        actor = _actor(actor or Actor(model=_response_model(response)), scope)
        meta = _explicit_meta(
            "llm_call", uuid.uuid4().hex, scope, actor, facets, _response_trace(response)
        )
        outputs = _entities({"response": response}, OUTPUTS)
        return self._commit_entities(meta, _context_entities(context), outputs)

    def llm_calls(
        self,
        conversation: Any,
        *,
        role: str | None = None,
        within: Activity | Context[Any] | None = None,
    ) -> Callable[[int, Any], Awaitable[None]]:
        """SAIA's ``on_iteration`` callback: a :meth:`llm_call` per backend call.

        SAIA calls it after each model response, before appending the
        response to ``conversation``, so the context committed is what the
        model was sent. The iteration number is a facet.

        This holds for ``complete``, which appends to ``conversation`` as it
        goes. ``complete_structured`` runs on its own copy and merges into
        ``conversation`` only at the end, so its commits would miss the
        current turn's messages.
        """

        async def on_iteration(iteration: int, response: Any) -> None:
            actor = Actor(role=role, model=_response_model(response))
            self.llm_call(
                conversation, response, actor=actor, facets={"iteration": iteration}, within=within
            )

        return on_iteration

    # --- hooks ---

    def after_run(self, event: RunEvent) -> None:
        """Record the run's end; ``resumed_from`` names the commit it started from."""
        extra = {"outcome": event.outcome, "resumed_from": event.run.resumed_from}
        self._commit(_meta("run", event, extra, error=event.error), {}, {})

    def before_step(self, event: StepEvent) -> None:
        """Encode the step's inputs before it runs, so one it mutates is recorded as received.

        Never keeps the step from running: a failure is logged, and the step's
        record then holds its inputs as they are when it ends.
        """
        try:
            self._received[event.activity_id] = _step_entities(event)
        except Exception as e:
            self._lg.warning("step inputs not encoded", extra={"exception": e})

    def after_step(self, event: StepEvent) -> None:
        """Record a step: the inputs it received, its result when it returned one, how it ended.

        A step a hook stopped before this one's ``before_step`` never ran, so
        its inputs are read now.
        """
        inputs = self._received.pop(event.activity_id, None)
        if inputs is None:
            inputs = _step_entities(event)
        outputs = {"result": event.result} if event.outcome in ("ok", "denied") else {}
        decision = None if event.decision is None else dataclasses.asdict(event.decision)
        extra = {"outcome": event.outcome, "decision": decision}
        meta = _meta("step", event, extra, error=event.error)
        self._commit_entities(meta, inputs, _entities(outputs, OUTPUTS))

    def after_llm_call(self, event: LLMCallEvent) -> None:
        """Record a model call: the context it was sent and its response.

        Runs while SAIA waits, before the response joins the conversation,
        so the context is what the model saw. A conversation that cannot be
        read is named in ``omitted``; the response is still recorded.
        """
        outputs = _entities({"response": event.response}, OUTPUTS)
        self._commit_entities(
            _meta("llm_call", event, {}), _context_entities(event.conversation), outputs
        )

    def after_checkpoint(self, event: CheckpointEvent) -> None:
        """Record that a checkpoint was written, naming its commit."""
        extra = {"checkpoint": event.commit_id, "outcome": event.outcome}
        self._commit(_meta("checkpoint", event, extra), {}, {})

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

    def _end(self, activity: Activity, outcome: str, error: BaseException | None) -> str:
        """Commit ``activity`` itself, with how it ended."""
        scope = _Scope(parent=activity.parent, run=activity.run)
        meta = _explicit_meta(
            activity.kind, activity.id, scope, activity.actor, activity.facets, None
        )
        meta |= {"outcome": outcome, "error": _error(error)}
        return self._commit_entities(meta, activity._inputs, _entities(activity.outputs, OUTPUTS))

    def _commit(self, meta: dict[str, Any], inputs: dict[str, Any], outputs: dict[str, Any]) -> str:
        """Encode and store the entities, then commit the record; return its id."""
        return self._commit_entities(meta, _entities(inputs, INPUTS), _entities(outputs, OUTPUTS))

    def _commit_entities(self, meta: dict[str, Any], inputs: _Entities, outputs: _Entities) -> str:
        """Store encoded entities, then commit the record on the ref's head; return its id."""
        with self.repo.batch() as batch:
            tree = Tree(
                (
                    TreeEntry(INPUTS, "tree", _put_entities(batch, inputs)),
                    TreeEntry(OUTPUTS, "tree", _put_entities(batch, outputs)),
                )
            )
            tree_id = batch.put(tree)
        omitted = [*inputs.omitted, *outputs.omitted]
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


def _scope(within: Activity | Context[Any] | None) -> _Scope:
    """The scope ``within`` places a record in: an activity, a Flow step, or none."""
    if within is None:
        return _Scope()
    if isinstance(within, Activity):
        return within._scope()
    role = None if within.role is None else within.role.name
    return _Scope(parent=within.activity_id, run=within.run_id, role=role)


def _actor(actor: Actor | None, scope: _Scope) -> Actor:
    """``actor``, its role defaulting to the scope's."""
    actor = actor or Actor()
    if actor.role is None and scope.role is not None:
        actor = dataclasses.replace(actor, role=scope.role)
    return actor


def _explicit_meta(
    kind: str,
    activity: str,
    scope: _Scope,
    actor: Actor,
    facets: Mapping[str, Any] | None,
    trace: TraceIds | None,
) -> dict[str, Any]:
    """The payload of an explicitly made record."""
    return {
        "kind": kind,
        "activity": activity,
        "parent": scope.parent,
        "run": scope.run,
        "ts": datetime.now(UTC).isoformat(),
        "actor": dataclasses.asdict(actor),
        "trace": None if trace is None else dataclasses.asdict(trace),
        "facets": _facets(facets or {}),
    }


def _error(error: BaseException | None) -> str | None:
    """An error as ``"<type>: <message>"``."""
    return None if error is None else f"{type(error).__name__}: {error}"


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
        "error": _error(error),
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
        except Exception:
            out[str(name)] = repr(value)
    return out


def _context_entities(conversation: Any) -> _Entities:
    """A model call's context, encoded: its messages as a vector.

    A conversation of no known shape is stored as a plain value; one whose
    messages cannot be read (its ``to_dict()`` raises) is named in ``omitted``.
    """
    if conversation is None:
        return _Entities()
    try:
        messages = _messages(conversation)
    except Exception:
        return _Entities(omitted=(f"{INPUTS}/context",))
    return _entities({"context": conversation if messages is None else messages}, INPUTS)


def _messages(conversation: Any) -> Messages | None:
    """A conversation's messages: a list, the ``messages`` of its ``to_dict()``, or its
    ``as_messages()`` (SAIA's ``ConversationLike``)."""
    if isinstance(conversation, Messages):
        return conversation
    if isinstance(conversation, list | tuple):
        return Messages(list(conversation))
    to_dict = getattr(conversation, "to_dict", None)
    data = to_dict() if callable(to_dict) else None
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        return Messages(data["messages"])
    as_messages = getattr(conversation, "as_messages", None)
    if callable(as_messages):
        return Messages(list(as_messages()))
    return None


def _response_model(response: Any) -> str | None:
    """The model a response names: its ``model`` attribute or key."""
    model = (
        response.get("model") if isinstance(response, dict) else getattr(response, "model", None)
    )
    return model if isinstance(model, str) else None


def _response_trace(response: Any) -> TraceIds | None:
    """The ids a response carries: SAIA's ``call_id``, llm-infer's ``request.id``."""
    call_id = getattr(response, "call_id", None) or None
    request_id = getattr(getattr(response, "request", None), "id", None) or None
    if call_id is None and request_id is None:
        return None
    return TraceIds(call_id=call_id, llm_request_id=request_id)


@dataclass(frozen=True)
class _Encoded:
    """A value in its stored form, not yet put: one blob, or a vector's elements."""

    type: TreeEntryType
    blobs: tuple[bytes, ...]


@dataclass(frozen=True)
class _Entities:
    """Named values in their stored form; the names whose value has no JSON form in ``omitted``."""

    encoded: dict[str, _Encoded] = field(default_factory=dict)
    omitted: tuple[str, ...] = ()


def _step_entities(event: StepEvent) -> _Entities:
    """A step's inputs, encoded: positional arguments by index, keyword arguments by name.

    A keyword argument named like a positional index (``run(1, **{"0": 2})``)
    is left out and named in ``omitted`` as ``inputs/<name> (keyword)``.
    """
    inputs = {str(i): arg for i, arg in enumerate(event.args)}
    clashes = tuple(f"{INPUTS}/{name} (keyword)" for name in event.kwargs if name in inputs)
    inputs |= {name: value for name, value in event.kwargs.items() if name not in inputs}
    entities = _entities(inputs, INPUTS)
    return dataclasses.replace(entities, omitted=entities.omitted + clashes)


def _entities(values: Mapping[str, Any], where: str) -> _Entities:
    """Encode named values; a name whose value has no JSON form is named in ``omitted``."""
    encoded: dict[str, _Encoded] = {}
    omitted: list[str] = []
    for name, value in values.items():
        try:
            encoded[name] = _encode(value)
        except Exception:  # no JSON form, one that contains itself, a failing serializer
            omitted.append(f"{where}/{name}")
    return _Entities(encoded, tuple(omitted))


def _encode(value: Any) -> _Encoded:
    """A value's stored form: :class:`Messages` as a vector of blobs, anything else as one blob."""
    if isinstance(value, Messages):
        return _Encoded("vector", tuple(canonical_json(_json(item)) for item in value.items))
    return _Encoded("blob", (canonical_json(_json(value)),))


def _put_entities(batch: Batch, entities: _Entities) -> str:
    """Put one subtree of encoded entities; return its id."""
    entries: list[TreeEntry] = []
    for name, value in entities.encoded.items():
        ids = [batch.put(Blob(data)) for data in value.blobs]
        object_id = Vector.build(batch, ids).id if value.type == "vector" else ids[0]
        entries.append(TreeEntry(name, value.type, object_id))
    return batch.put(Tree(tuple(entries)))


def _json(value: Any) -> Any:
    """The JSON form of ``value``: the checkpoint codec's, else a dataclass's fields.

    A dataclass the codec does not take (a model response) is stored as its
    fields, without :data:`_NOT_STORED`.

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
                if f.name not in _NOT_STORED
            }
        if isinstance(value, list | tuple):
            return [_json(item) for item in value]
        if isinstance(value, dict):
            return {str(k): _json(v) for k, v in value.items()}
        raise
