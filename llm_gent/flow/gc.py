# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Remove the objects of a history that no ref reaches.

A history's objects are reachable from its refs — ``HEAD`` and every tag —
through commits, their parents, each commit's snapshot tree, its subtrees
and blobs. Two things leave objects that nothing reaches:
``run(resume=<name>)`` or ``run(resume=<hash>)`` moving ``HEAD`` back to an
earlier commit (the commits written after it, and what only they hold —
resumable by hash until collected), and a process that died after writing
a commit's objects but before moving ``HEAD`` to it.
:func:`collect_unreachable` deletes them; the framework never does on its
own.
"""

from __future__ import annotations

from collections.abc import Iterable

from .checkpoint import CheckpointStore, Kind, maybe_await
from .history import HistoryCorrupt
from .state.cas import Commit, Tree


async def collect_unreachable(store: CheckpointStore, client_flow_id: str) -> int:
    """Delete the objects of ``client_flow_id``'s history that no ref reaches; return how many.

    Refs, the name binding and every object a ref reaches stay. Call it
    while no run writes the history: a run puts a commit's blobs and
    trees before ``HEAD`` moves to the commit, so until then they are
    unreachable and would be deleted. Returns ``0`` for an unknown name.

    Raises:
        HistoryCorrupt: A ref, a parent or a tree entry points at an object
            the store does not hold. Nothing is deleted then.
    """
    flow_id: str | None = await maybe_await(store.get_flow_id(client_flow_id))
    if flow_id is None:
        return 0
    refs: dict[str, str] = await maybe_await(store.list_refs(flow_id))
    reachable = await _reachable(store, flow_id, refs.values())
    stored: list[tuple[Kind, str]] = await maybe_await(store.list_objects(flow_id))
    garbage = [key for key in stored if key not in reachable]
    if garbage:
        await maybe_await(store.delete_objects(flow_id, garbage))
    return len(garbage)


async def _reachable(
    store: CheckpointStore, flow_id: str, heads: Iterable[str]
) -> set[tuple[Kind, str]]:
    """Every ``(kind, hash)`` reachable from the commits ``heads``; each object must exist."""
    seen: set[tuple[Kind, str]] = set()
    pending: list[tuple[Kind, str]] = [("commit", head) for head in heads]
    while pending:
        key = pending.pop()
        if key in seen:
            continue
        seen.add(key)
        kind, content_hash = key
        if kind == "blob":
            if not await maybe_await(store.has_object(flow_id, kind, content_hash)):
                raise HistoryCorrupt(flow_id, kind, content_hash)
            continue
        payload: bytes | None = await maybe_await(store.get_object(flow_id, kind, content_hash))
        if payload is None:
            raise HistoryCorrupt(flow_id, kind, content_hash)
        pending.extend(_children(kind, payload))
    return seen


def _children(kind: Kind, payload: bytes) -> list[tuple[Kind, str]]:
    """The objects a commit (parents, root tree) or a tree (its entries) points at."""
    if kind == "commit":
        commit = Commit.from_bytes(payload)
        return [("tree", commit.root_tree_hash), *(("commit", p) for p in commit.parent_hashes)]
    return [(entry.kind, entry.child_hash) for entry in Tree.from_bytes(payload).entries]
