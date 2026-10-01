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
    n/<node>/items, n/<node>/done     cursor of a running .map
    n/<node>/i/<index>/state          scope of map item <index>
    n/<node>/i/<index>/chain          cursor of map item <index>'s body

A dict payload is stored as a tree with one blob per top-level key, so
keys that did not change keep their hash from one commit to the next.

:class:`ScopeRegistry` tracks the live scopes and cursors during a run;
:func:`build_snapshot_tree` turns them into CAS objects; :func:`read_snapshot`
turns a stored tree back into a :class:`Snapshot`.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..checkpoint import Kind
from . import codec, serialize_state_data
from .base import State
from .cas import Blob, Tree, TreeEntry, TreeEntryKind, canonical_json


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

CURSOR_ENTRIES = frozenset({PASS, CARRY, UNTIL, CHAIN, ARM, TURN, ITEMS, DONE})
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


def build_snapshot_tree(flat: dict[ScopePath, Any]) -> tuple[Tree, list[Blob | Tree]]:
    """Build the snapshot tree for captured values; return it and every object to store.

    Pure and synchronous: hashing only, no store access.
    """
    nested: dict[str, Any] = {}
    for path, value in flat.items():
        node = nested
        for segment in path[:-1]:
            node = node.setdefault(segment, {})
        node[path[-1]] = _Leaf(value)
    objects: list[Blob | Tree] = []
    return _build_tree(nested, objects), objects


@dataclass(frozen=True)
class _Leaf:
    """A captured value at a :data:`STATE` or :data:`PASS` entry."""

    value: Any


def _build_tree(node: dict[str, Any], objects: list[Blob | Tree]) -> Tree:
    """Tree for one level of the nested path map; objects are appended children-first."""
    entries: list[TreeEntry] = []
    for name, child in node.items():
        if isinstance(child, _Leaf):
            obj: Blob | Tree = _leaf_object(child.value, objects)
        else:
            obj = _build_tree(child, objects)
        kind: TreeEntryKind = "blob" if isinstance(obj, Blob) else "tree"
        entries.append(TreeEntry(scope_id=name, kind=kind, child_hash=obj.content_hash))
    tree = Tree.from_entries(entries)
    objects.append(tree)
    return tree


def _leaf_object(value: Any, objects: list[Blob | Tree]) -> Blob | Tree:
    """A dict payload becomes a tree with one blob per key; anything else is one blob."""
    if not isinstance(value, dict):
        blob = Blob.from_bytes(canonical_json(value))
        objects.append(blob)
        return blob
    entries: list[TreeEntry] = []
    for key, item in value.items():
        blob = Blob.from_bytes(canonical_json(item))
        objects.append(blob)
        entries.append(TreeEntry(scope_id=key, kind="blob", child_hash=blob.content_hash))
    tree = Tree.from_entries(entries)
    objects.append(tree)
    return tree


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


Loader = Callable[[Kind, str], Awaitable[bytes]]
"""Fetches one stored object by kind and hash; raises when it is missing."""


async def read_snapshot(root_tree_hash: str, load: Loader) -> Snapshot:
    """Read the snapshot tree at ``root_tree_hash`` back into a :class:`Snapshot`."""
    flat: dict[ScopePath, Any] = {}
    await _read_level(root_tree_hash, (), load, flat)
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


async def _read_level(
    tree_hash: str, path: ScopePath, load: Loader, flat: dict[ScopePath, Any]
) -> None:
    """Collect the :data:`STATE` and cursor leaves under the tree at ``path``."""
    tree = Tree.from_bytes(await load("tree", tree_hash))
    for entry in tree.entries:
        here = (*path, entry.scope_id)
        if entry.scope_id == STATE or entry.scope_id in CURSOR_ENTRIES:
            flat[here] = await _read_leaf(entry, load)
        elif entry.kind == "tree":
            await _read_level(entry.child_hash, here, load, flat)


async def _read_leaf(entry: TreeEntry, load: Loader) -> Any:
    """A leaf's value: a blob's JSON, or a key tree rebuilt into a dict."""
    if entry.kind == "blob":
        return json.loads(await load("blob", entry.child_hash))
    keys = Tree.from_bytes(await load("tree", entry.child_hash))
    return {e.scope_id: json.loads(await load("blob", e.child_hash)) for e in keys.entries}
