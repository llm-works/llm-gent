# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Run snapshots — every live scope of a run, at a stable path, as one tree.

A checkpoint captures the whole working tree of a run, the way a git
commit captures a whole repository: the root scope, every child scope a
``state=`` projection opened and has not merged back, and each running
iterate's pass counter. Paths are built from position-independent node
ids plus the coordinates that tell repeated executions apart::

    state                             root scope
    n/<node>/state                    scope opened by a .call / .iterate step
    n/<node>/pass                     pass counter of a running .iterate
    n/<node>/p/<pass>/n/<node>/...    positions inside iterate pass <pass>
    n/<node>/i/<index>/state          scope of map item <index>

A scope's payload is stored as a tree with one blob per top-level key,
so keys that did not change keep their hash from one commit to the next.

:class:`ScopeRegistry` tracks the live scopes during a run;
:func:`build_snapshot_tree` turns it into CAS objects; :func:`read_snapshot`
turns a stored tree back into a :class:`Snapshot`.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..checkpoint import Kind
from . import serialize_state_data
from .base import State
from .cas import Blob, Tree, TreeEntry, TreeEntryKind, canonical_json


ScopePath = tuple[str, ...]
"""Position of a scope in the composition, e.g. ``("n", "<node>", "i", "0")``; ``()`` is the root."""

STATE = "state"
"""Tree entry holding a scope's payload."""

PASS = "pass"
"""Tree entry holding a running iterate's pass counter."""


def path_str(path: ScopePath) -> str:
    """``"/"``-joined form of ``path`` used in commit meta and :class:`Snapshot` keys."""
    return "/".join(path)


def path_from_str(text: str) -> ScopePath:
    """Inverse of :func:`path_str`."""
    return tuple(text.split("/")) if text else ()


class ScopeRegistry:
    """The live scopes and iterate pass counters of one run, by path.

    Descent sites open a scope when a ``state=`` projection creates one and
    close it once it merged back (or was discarded); iterates record their
    pass counter the same way. Everything a checkpoint needs is here.
    """

    def __init__(self) -> None:
        self._root: State[Any] | None = None
        self._scopes: dict[ScopePath, State[Any]] = {}
        self._passes: dict[ScopePath, int] = {}
        self._saved: dict[ScopePath, Any] = {}

    def begin(self, root: State[Any], saved: Snapshot | None = None) -> None:
        """Start a run whose root scope is ``root``; forget the previous run.

        ``saved`` is the snapshot a restart continues from: its child scopes
        are handed out by :meth:`take_saved` as the run reaches their paths.
        """
        self._root = root
        self._scopes.clear()
        self._passes.clear()
        self._saved = (
            {} if saved is None else {path_from_str(p): v for p, v in saved.scopes.items()}
        )

    def take_saved(self, path: ScopePath) -> tuple[bool, Any]:
        """Pop the saved payload of the scope at ``path``; ``(False, None)`` when there is none.

        Each saved scope is handed out once: a block entered again at the
        same path projects a fresh scope.
        """
        if path not in self._saved:
            return False, None
        return True, self._saved.pop(path)

    def open(self, path: ScopePath, scope: State[Any]) -> None:
        """Register ``scope`` as live at ``path``."""
        self._scopes[path] = scope

    def close(self, path: ScopePath) -> None:
        """Drop the scope at ``path``; a no-op when none is registered."""
        self._scopes.pop(path, None)

    def set_pass(self, path: ScopePath, count: int) -> None:
        """Record the iterate at ``path`` as running pass ``count`` (0-based)."""
        self._passes[path] = count

    def clear_pass(self, path: ScopePath) -> None:
        """Drop the pass counter of the iterate at ``path``."""
        self._passes.pop(path, None)

    def path_of(self, scope: State[Any]) -> ScopePath:
        """Path of the live scope ``scope`` (identity); ``()`` for the root or an unknown scope."""
        return next((p for p, s in self._scopes.items() if s is scope), ())

    def capture(self) -> dict[ScopePath, Any]:
        """Serialize every live scope and pass counter now, in one synchronous pass.

        Returns tree paths (ending in :data:`STATE` or :data:`PASS`) mapped
        to JSON-compatible values. Nothing awaits in between, so every
        value comes from the same moment.

        Raises:
            TypeError: A scope's payload cannot be serialized; the message
                names its path.
        """
        if self._root is None:
            raise RuntimeError("ScopeRegistry.capture() before begin()")
        flat: dict[ScopePath, Any] = {(STATE,): _to_json(self._root, ())}
        for path, scope in self._scopes.items():
            flat[(*path, STATE)] = _to_json(scope, path)
        for path, count in self._passes.items():
            flat[(*path, PASS)] = count
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
    then ``False``); ``scopes`` and ``passes`` are keyed by
    :func:`path_str` of the owner's path.
    """

    has_state: bool
    root: Any = None
    scopes: dict[str, Any] = field(default_factory=dict)
    passes: dict[str, int] = field(default_factory=dict)

    def chain(self, scope_path: str) -> list[Any]:
        """Payloads of the live scopes from the root's child down to ``scope_path``, in order.

        The scopes whose path is a prefix of ``scope_path`` (including it);
        the root itself is not included.
        """
        target = path_from_str(scope_path)
        owners = [path_from_str(p) for p in self.scopes]
        on_path = sorted((p for p in owners if target[: len(p)] == p), key=len)
        return [self.scopes[path_str(p)] for p in on_path]


Loader = Callable[[Kind, str], Awaitable[bytes]]
"""Fetches one stored object by kind and hash; raises when it is missing."""


async def read_snapshot(root_tree_hash: str, load: Loader) -> Snapshot:
    """Read the snapshot tree at ``root_tree_hash`` back into a :class:`Snapshot`."""
    flat: dict[ScopePath, Any] = {}
    await _read_level(root_tree_hash, (), load, flat)
    if (STATE,) not in flat:
        return Snapshot(has_state=False)
    return Snapshot(
        has_state=True,
        root=flat.pop((STATE,)),
        scopes={path_str(p[:-1]): v for p, v in flat.items() if p[-1] == STATE},
        passes={path_str(p[:-1]): int(v) for p, v in flat.items() if p[-1] == PASS},
    )


async def _read_level(
    tree_hash: str, path: ScopePath, load: Loader, flat: dict[ScopePath, Any]
) -> None:
    """Collect the :data:`STATE` / :data:`PASS` leaves under the tree at ``path``."""
    tree = Tree.from_bytes(await load("tree", tree_hash))
    for entry in tree.entries:
        here = (*path, entry.scope_id)
        if entry.scope_id in (STATE, PASS):
            flat[here] = await _read_leaf(entry, load)
        elif entry.kind == "tree":
            await _read_level(entry.child_hash, here, load, flat)


async def _read_leaf(entry: TreeEntry, load: Loader) -> Any:
    """A leaf's value: a blob's JSON, or a key tree rebuilt into a dict."""
    if entry.kind == "blob":
        return json.loads(await load("blob", entry.child_hash))
    keys = Tree.from_bytes(await load("tree", entry.child_hash))
    return {e.scope_id: json.loads(await load("blob", e.child_hash)) for e in keys.entries}
