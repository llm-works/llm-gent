# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Run snapshots — every live scope and cursor of a run, at a stable path, as one tree.

A checkpoint captures the whole working tree of a run, the way a git
commit captures a whole repository: the root scope, every child scope a
``state=`` projection opened and has not merged back, and the cursor of
every running primitive — where it is and the value it is working on.
Paths are built from position-independent node ids plus the coordinates
that tell repeated executions apart::

    state                             root scope
    chain                             cursor of the top-level chain
    n/<node>/state                    scope opened by a .call / .iterate step
    n/<node>/chain                    cursor of the chain a .call / .branch runs
    n/<node>/pass, carry, until       cursor of a running .iterate
    n/<node>/arm                      arm a running .branch took
    n/<node>/t/<k>/turn               paused turn of the step's <k>-th Loop call
    n/<node>/p/<pass>/n/<node>/...    positions inside iterate pass <pass>
    n/<node>/items, n/<node>/done     cursor of a running .map (no items over members)
    n/<node>/i/<key>/state            scope of map item <key>: its index, or a member's key
    n/<node>/i/<key>/chain            cursor of map item <key>'s body
    cost, <run path>/cost             a budgeted run's child tracker: spend so far
    tracker, <run path>/tracker       a cost tracker a flow declares: spend so far
    signals                           the run's signals that are set
    flow                              the structure of the flow that wrote it

A dict payload is stored as a tree with one blob per top-level key, so
keys that did not change keep their id from one commit to the next. A
paused turn's messages are stored as an :class:`appinfra.cas.Vector` of
one blob per message, so a turn paused again stores only what it added.

:class:`ScopeRegistry` tracks the live scopes and cursors during a run;
:func:`build_snapshot_tree` turns them into :mod:`appinfra.cas` objects;
:func:`read_snapshot` turns a stored tree back into a :class:`Snapshot`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from appinfra.cas import (
    Batch,
    Blob,
    Object,
    Tree,
    TreeEntry,
    Vector,
    VectorNode,
    canonical_json,
)
from appinfra.cas.objects import TreeEntryType

from . import codec, serialize_state_data
from .base import State


ScopePath = tuple[str, ...]
"""Position of a scope in the composition, e.g. ``("n", "<node>", "i", "0")``; ``()`` is the root."""

STATE = "state"
"""Tree entry holding a scope's payload."""

PASS = "pass"
"""Cursor entry: the pass a running iterate is in (0-based)."""

CARRY = "carry"
"""Cursor entry: the value a running iterate carries into its current pass."""

UNTIL = "until"
"""Cursor entry: a running iterate's ``until`` verdict on its carried value (``null``: unchecked)."""

CHAIN = "chain"
"""Cursor entry: the step a running chain is at, and that step's input."""

ARM = "arm"
"""Cursor entry: the arm a running branch took (``"then"`` or ``"else"``)."""

TURN = "turn"
"""Cursor entry: a Loop's paused SAIA turn — its task and conversation."""

ITEMS = "items"
"""Cursor entry: the items a running map runs over, resolved once."""

DONE = "done"
"""Cursor entry: a running map's completed items — index to result and whether it merged."""

SIGNALS = "signals"
"""Cursor entry at the root: the run's signals (``Flow.with_signal``) that are set."""

RESOURCES = "resources"
"""Cursor entry: the resources a flow declares (``with_resource``, ``with_cost_tracker``) — name to ``snapshot()``."""

RUN_RESOURCES = "run_resources"
"""Cursor entry: a running flow's per-run resource children (``with_budget``'s) — name to ``snapshot()``."""

FLOW = "flow"
"""Root tree entry: the structure of the flow that wrote the snapshot, as a blob.

Its id is the commit's ``flow_root_hash``
(:class:`~llm_gent.flow.structure.Structure`). Reachable from the
commit like every entry, so pruning the repository keeps it;
:func:`read_snapshot` does not read it.
"""

CURSOR_ENTRIES = frozenset(
    {
        PASS,
        CARRY,
        UNTIL,
        CHAIN,
        ARM,
        TURN,
        ITEMS,
        DONE,
        SIGNALS,
        RESOURCES,
        RUN_RESOURCES,
    }
)
"""Tree entries that hold cursor values rather than a scope."""


_NOT_SAVED = object()
"""Marker for "no saved cursor entry": a saved value can be ``None``."""


class Cursor(Protocol):
    """A running primitive that reports where it is.

    :meth:`cursor` returns its cursor entries (names from
    :data:`CURSOR_ENTRIES`) mapped to values :mod:`.codec` can store: everything the
    primitive needs to continue from this point.
    """

    def cursor(self) -> dict[str, Any]: ...


def path_str(path: ScopePath) -> str:
    """``"/"``-joined form of ``path`` used in commit meta and :class:`Snapshot` keys."""
    return "/".join(path)


def path_from_str(text: str) -> ScopePath:
    """Inverse of :func:`path_str`."""
    return tuple(text.split("/")) if text else ()


class ScopeRegistry:
    """The live scopes and cursors of one run, by path.

    Descent sites open a scope when a ``state=`` projection creates one and
    close it once it merged back (or was discarded). Running primitives
    register themselves as a :class:`Cursor` at their path while they run.
    Everything a checkpoint needs is here.
    """

    def __init__(self) -> None:
        self._root: State[Any] | None = None
        self._scopes: dict[ScopePath, State[Any]] = {}
        self._cursors: dict[tuple[ScopePath, int], Cursor] = {}
        self._saved: dict[ScopePath, Any] = {}
        self._saved_cursors: dict[tuple[ScopePath, str], Any] = {}
        self._turns: dict[ScopePath, int] = {}

    def begin(self, root: State[Any], saved: Snapshot | None = None) -> None:
        """Start a run whose root scope is ``root``; forget the previous run.

        ``saved`` is the snapshot the run continues from: its child scopes
        are handed out by :meth:`take_saved` and its cursor entries by
        :meth:`take_cursor` as the run reaches their paths.
        """
        self._root = root
        self._scopes.clear()
        self._cursors.clear()
        self._turns.clear()
        self._saved = (
            {} if saved is None else {path_from_str(p): v for p, v in saved.scopes.items()}
        )
        self._saved_cursors = (
            {}
            if saved is None
            else {
                (path_from_str(p), name): raw
                for p, entries in saved.cursors.items()
                for name, raw in entries.items()
            }
        )

    def take_saved(self, path: ScopePath) -> tuple[bool, Any]:
        """Pop the saved payload of the scope at ``path``; ``(False, None)`` when there is none.

        Each saved scope is handed out once: a block entered again at the
        same path projects a fresh scope.
        """
        if path not in self._saved:
            return False, None
        return True, self._saved.pop(path)

    def take_cursor(self, path: ScopePath, name: str) -> tuple[bool, Any]:
        """Pop the saved cursor entry ``name`` at ``path``, decoded; ``(False, None)`` when none.

        Like a saved scope, each entry is handed out once, to the first
        runner that reaches ``path``.

        Raises:
            TypeError: The stored value cannot be rebuilt (see
                :func:`.codec.decode`); the message names the path.
        """
        raw = self._saved_cursors.pop((path, name), _NOT_SAVED)
        if raw is _NOT_SAVED:
            return False, None
        return True, codec.decode(raw, f"cursor at {path_str((*path, name))!r}")

    def has_saved_under(self, prefix: ScopePath) -> bool:
        """True when a saved scope or cursor entry nothing has taken is at or below ``prefix``.

        A block that has some is one that was running when the snapshot was
        taken: it continues where it was.
        """
        n = len(prefix)
        return any(p[:n] == prefix for p in self._saved) or any(
            p[:n] == prefix for p, _ in self._saved_cursors
        )

    def drop_saved(self) -> list[ScopePath]:
        """Forget the saved scopes and cursor entries nothing has taken; return their paths."""
        dropped = [*self._saved, *((*path, name) for path, name in self._saved_cursors)]
        self._saved.clear()
        self._saved_cursors.clear()
        return dropped

    def open(self, path: ScopePath, scope: State[Any]) -> None:
        """Register ``scope`` as live at ``path``."""
        self._scopes[path] = scope

    def close(self, path: ScopePath) -> None:
        """Drop the scope at ``path``; a no-op when none is registered."""
        self._scopes.pop(path, None)

    def open_cursor(self, path: ScopePath, runner: Cursor) -> None:
        """Register ``runner`` as running at ``path``.

        Several runners can share a path when their entries differ: a
        branch's ``arm`` and the ``chain`` of the arm it runs.
        """
        self._cursors[(path, id(runner))] = runner

    def close_cursor(self, path: ScopePath, runner: Cursor) -> None:
        """Drop ``runner`` from ``path``; a no-op when it is not registered."""
        self._cursors.pop((path, id(runner)), None)

    def close_under(self, prefix: ScopePath) -> None:
        """Drop every scope and runner registered at ``prefix`` or below it.

        For a block that ended while things under it stayed registered —
        positions the halt stopped, which the run's halt checkpoint was
        not written from, because the block ended the interruption.
        """
        n = len(prefix)
        for path in [p for p in self._scopes if p[:n] == prefix]:
            del self._scopes[path]
        for key in [k for k in self._cursors if k[0][:n] == prefix]:
            del self._cursors[key]

    def holds_under(self, prefix: ScopePath, name: str) -> bool:
        """True when a runner at ``prefix`` or below it reports the cursor entry ``name``."""
        n = len(prefix)
        return any(
            path[:n] == prefix and name in runner.cursor()
            for (path, _), runner in self._cursors.items()
        )

    def next_turn(self, step: ScopePath) -> ScopePath:
        """Path of the next Loop call inside the step at ``step``: ``<step>/t/<k>``.

        A step runs at most once per run at a given path, so numbering the
        Loop calls in the order they start gives a rerun of the step the
        same path for the same call.
        """
        k = self._turns.get(step, 0)
        self._turns[step] = k + 1
        return (*step, "t", str(k))

    def path_of(self, scope: State[Any]) -> ScopePath:
        """Path of the live scope ``scope`` (identity); ``()`` for the root or an unknown scope."""
        return next((p for p, s in self._scopes.items() if s is scope), ())

    def capture(self) -> dict[ScopePath, Any]:
        """Serialize every live scope and cursor now, in one synchronous pass.

        Returns tree paths (ending in :data:`STATE` or a cursor entry)
        mapped to JSON-compatible values. Nothing awaits in between, so
        every value comes from the same moment. Saved scopes and cursor
        entries the run has not reached yet are included as saved, so a
        checkpoint taken early in a resumed run keeps them.

        Raises:
            TypeError: A scope's payload cannot be serialized, or a cursor
                holds a value :mod:`.codec` cannot store; the message names the
                path.
        """
        if self._root is None:
            raise RuntimeError("ScopeRegistry.capture() before begin()")
        flat: dict[ScopePath, Any] = {(STATE,): _to_json(self._root, ())}
        for path, value in self._saved.items():
            flat[(*path, STATE)] = value
        for (path, name), raw in self._saved_cursors.items():
            flat[(*path, name)] = raw
        for path, scope in self._scopes.items():
            flat[(*path, STATE)] = _to_json(scope, path)
        for (path, _), runner in self._cursors.items():
            for name, value in runner.cursor().items():
                where = f"cursor at {path_str((*path, name))!r}"
                flat[(*path, name)] = codec.encode(value, where)
        return flat


def _to_json(scope: State[Any], path: ScopePath) -> Any:
    """Canonical JSON round-trip of ``scope``'s payload — a deep, detached copy."""
    try:
        return json.loads(canonical_json(serialize_state_data(scope.data)))
    except (TypeError, ValueError) as e:
        where = path_str(path) or "root"
        raise TypeError(f"scope at {where!r} cannot be checkpointed: {e}") from e


_VECTOR_AT: dict[str, tuple[str, ...]] = {TURN: ("conversation", "messages")}
"""Leaf entries holding a list stored as a vector: entry name to the keys that lead to it.

A paused turn's messages, so a turn paused again stores only the messages
it added. The keys follow the ``to_dict()`` shape of kelt's
``Conversation``; a conversation of another shape is stored as one blob.
"""


def build_snapshot_tree(
    flat: dict[ScopePath, Any],
    structure: Blob,
    batch: Batch,
    stored: AbstractSet[str] = frozenset(),
) -> tuple[Tree, set[str]]:
    """Queue the snapshot tree for captured values on ``batch``; return it and what was queued.

    ``structure`` is the structure of the flow writing the snapshot
    (:meth:`~llm_gent.flow.structure.Structure.blob`), held at the
    root as :data:`FLOW`. Blobs and trees whose ids are in ``stored`` are
    not queued again; the returned set holds the ids of the blobs and
    trees queued. Vector nodes are queued by :class:`~appinfra.cas.Vector`
    and are in neither. Synchronous: nothing is stored before the batch's
    block ends.
    """
    nested: dict[str, Any] = {}
    for path, value in flat.items():
        node = nested
        for segment in path[:-1]:
            node = node.setdefault(segment, {})
        node[path[-1]] = _Leaf(value, _VECTOR_AT.get(path[-1], ()))
    assert FLOW not in nested, "no scope or cursor is registered at the root as 'flow'"
    nested[FLOW] = structure
    sink = _Sink(batch, stored)
    tree = _build_tree(nested, sink)
    sink.put(tree)
    return tree, sink.queued


@dataclass(frozen=True)
class _Leaf:
    """A captured value at a :data:`STATE` or cursor entry; ``vector_at`` as in :data:`_VECTOR_AT`."""

    value: Any
    vector_at: tuple[str, ...]


@dataclass(frozen=True)
class _Ref:
    """What a tree entry points at: its entry type and object id."""

    type: TreeEntryType
    id: str

    def named(self, name: str) -> TreeEntry:
        """The tree entry ``name`` pointing here."""
        return TreeEntry(name, self.type, self.id)


class _Sink:
    """Queues objects on a batch, skipping those already stored, and records what it queued."""

    def __init__(self, batch: Batch, stored: AbstractSet[str]) -> None:
        self.batch = batch
        self._stored = stored
        self.queued: set[str] = set()

    def put(self, obj: Blob | Tree) -> _Ref:
        """Queue ``obj`` unless it is stored; return a reference to it."""
        if obj.id not in self._stored:
            self.batch.put(obj)
            self.queued.add(obj.id)
        return _Ref("blob" if isinstance(obj, Blob) else "tree", obj.id)

    def blob(self, value: Any) -> _Ref:
        """Queue ``value`` as a blob of canonical JSON."""
        return self.put(Blob(canonical_json(value)))


def _build_tree(node: dict[str, Any], sink: _Sink) -> Tree:
    """Tree for one level of the nested path map; what it points at is queued, not the tree."""
    entries: list[TreeEntry] = []
    for name, child in node.items():
        if isinstance(child, _Leaf):
            ref = _leaf_ref(child.value, child.vector_at, sink)
        elif isinstance(child, Blob):
            ref = sink.put(child)
        else:
            ref = sink.put(_build_tree(child, sink))
        entries.append(ref.named(name))
    return Tree(tuple(entries))


def _leaf_ref(value: Any, vector_at: tuple[str, ...], sink: _Sink) -> _Ref:
    """A dict payload becomes a tree with one entry per key; anything else is one blob.

    A dict with an empty key is one blob too: a tree entry needs a name.
    The key ``vector_at`` names leads one level down: its value is a tree
    of its own when the keys after it lead to a list, which is stored as a
    vector of one blob per element. Without such a list it stays a blob.
    """
    if not isinstance(value, dict) or "" in value:
        return sink.blob(value)
    entries: list[TreeEntry] = []
    for key, item in value.items():
        if vector_at and key == vector_at[0] and _leads_to_list(item, vector_at[1:]):
            ref = _nested_ref(item, vector_at[1:], sink)
        else:
            ref = sink.blob(item)
        entries.append(ref.named(key))
    return sink.put(Tree(tuple(entries)))


def _leads_to_list(value: Any, keys: tuple[str, ...]) -> bool:
    """True when ``keys`` lead through dicts with named keys to a list inside ``value``."""
    if not keys:
        return isinstance(value, list)
    if not isinstance(value, dict) or "" in value:
        return False
    return keys[0] in value and _leads_to_list(value[keys[0]], keys[1:])


def _nested_ref(value: Any, keys: tuple[str, ...], sink: _Sink) -> _Ref:
    """The list at the end of ``keys`` (see :func:`_leads_to_list`) as a vector, in its trees."""
    if not keys:
        ids = [sink.blob(element).id for element in value]
        return _Ref("vector", Vector.build(sink.batch, ids).id)
    return _leaf_ref(value, keys, sink)


@dataclass(frozen=True)
class Snapshot:
    """The run state one commit holds, as stored (JSON-compatible values).

    ``root`` is ``None`` when the commit carries no state (``has_state`` is
    then ``False``). ``scopes`` and ``cursors`` are keyed by
    :func:`path_str` of the owner's path; a path's cursor maps its entry
    names (:data:`CURSOR_ENTRIES`) to their values.
    """

    has_state: bool
    root: Any = None
    scopes: dict[str, Any] = field(default_factory=dict)
    cursors: dict[str, dict[str, Any]] = field(default_factory=dict)


ObjT = TypeVar("ObjT", bound=Object)


class Loader(Protocol):
    """Fetches one stored object by id as ``cls``; raises when it is missing."""

    def __call__(self, object_id: str, cls: type[ObjT]) -> ObjT:
        """Return the object ``object_id`` as ``cls``."""
        ...


def read_snapshot(root_tree: str, load: Loader) -> Snapshot:
    """Read the snapshot tree ``root_tree`` back into a :class:`Snapshot`."""
    flat: dict[ScopePath, Any] = {}
    _read_level(root_tree, (), load, flat)
    if (STATE,) not in flat:
        return Snapshot(has_state=False)
    cursors: dict[str, dict[str, Any]] = {}
    for p, v in flat.items():
        if p[-1] in CURSOR_ENTRIES:
            cursors.setdefault(path_str(p[:-1]), {})[p[-1]] = v
    return Snapshot(
        has_state=True,
        root=flat.pop((STATE,)),
        scopes={path_str(p[:-1]): v for p, v in flat.items() if p[-1] == STATE},
        cursors=cursors,
    )


def _read_level(tree_id: str, path: ScopePath, load: Loader, flat: dict[ScopePath, Any]) -> None:
    """Collect the :data:`STATE` and cursor leaves under the tree at ``path``."""
    for entry in load(tree_id, Tree).entries:
        here = (*path, entry.name)
        if entry.name == STATE or entry.name in CURSOR_ENTRIES:
            flat[here] = _read_leaf(entry, load)
        elif entry.type == "tree":
            _read_level(entry.id, here, load, flat)


def _read_leaf(entry: TreeEntry, load: Loader) -> Any:
    """A leaf's value: a blob's JSON, a tree as a dict of its entries, a vector as a list.

    Trees nest: an entry of a leaf's tree is read the same way, so a key
    can hold a tree or a vector of its own. A vector's elements are blobs
    of JSON.
    """
    if entry.type == "blob":
        return json.loads(load(entry.id, Blob).data)
    if entry.type == "vector":
        return [json.loads(load(e, Blob).data) for e in _vector_elements(entry.id, load)]
    return {e.name: _read_leaf(e, load) for e in load(entry.id, Tree).entries}


def _vector_elements(node_id: str, load: Loader) -> Iterator[str]:
    """The element ids of the vector whose root node is ``node_id``, in order.

    A level-0 node lists elements; a node above it lists the nodes one
    level down. Reading through ``load`` keeps a missing node the
    loader's error, as for every other object.
    """
    node = load(node_id, VectorNode)
    if node.level == 0:
        yield from node.children
        return
    for child in node.children:
        yield from _vector_elements(child, load)
