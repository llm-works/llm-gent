# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Run snapshots — every live scope at its path, captured at once, read back.

Unit tests of :mod:`llm_gent.flow.state.snapshot` (registry, tree layout,
read-back) and flow-level tests of what a checkpoint captures: sibling map
item scopes, iterate pass counters, and the saving scope's path.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_gent.flow import Context, FlowFactory, History, verb
from llm_gent.flow.state import State
from llm_gent.flow.state.cas import Blob, Tree
from llm_gent.flow.state.snapshot import (
    ScopeRegistry,
    Snapshot,
    build_snapshot_tree,
    read_snapshot,
)
from llm_gent.flow.stores import InMemoryCheckpointStore

from .conftest import make_test_logger


pytestmark = pytest.mark.unit

LG = make_test_logger()


async def _round_trip(scopes: ScopeRegistry) -> tuple[Snapshot, dict[str, Blob | Tree]]:
    """Build the snapshot tree for ``scopes`` and read it back from the objects it made."""
    tree, objects = build_snapshot_tree(scopes.capture())
    by_hash = {obj.content_hash: obj for obj in objects}

    async def load(kind: Any, content_hash: str) -> bytes:
        obj = by_hash[content_hash]
        return obj.payload if isinstance(obj, Blob) else obj.to_bytes()

    return await read_snapshot(tree.content_hash, load), by_hash


def _blobs(objects: dict[str, Blob | Tree]) -> set[str]:
    """Hashes of the blobs among ``objects``."""
    return {h for h, obj in objects.items() if isinstance(obj, Blob)}


def _registry(root: Any) -> tuple[ScopeRegistry, State[Any]]:
    scopes = ScopeRegistry()
    root_state: State[Any] = State(data=root)
    scopes.begin(root_state)
    return scopes, root_state


class TestSnapshotTree:
    async def test_round_trip_root_scopes_and_passes(self) -> None:
        scopes, root = _registry({"a": 1, "b": [1, 2]})
        scopes.open(("n", "call"), State(data={"x": "y"}, _parent=root))
        scopes.open(("n", "map", "i", "0"), State(data={"k": 0}, _parent=root))
        scopes.set_pass(("n", "it"), 3)

        snapshot, _ = await _round_trip(scopes)
        assert snapshot.has_state
        assert snapshot.root == {"a": 1, "b": [1, 2]}
        assert snapshot.scopes == {"n/call": {"x": "y"}, "n/map/i/0": {"k": 0}}
        assert snapshot.passes == {"n/it": 3}

    async def test_non_dict_root_round_trips(self) -> None:
        scopes, _ = _registry(None)
        snapshot, _ = await _round_trip(scopes)
        assert snapshot.has_state and snapshot.root is None

    async def test_one_blob_per_key_so_unchanged_keys_keep_their_hash(self) -> None:
        """Changing one key of a scope changes that key's blob only."""
        big = {"history": list(range(500))}
        _, before = await _round_trip(_registry({**big, "n": 1})[0])
        _, after = await _round_trip(_registry({**big, "n": 2})[0])
        assert len(_blobs(before) & _blobs(after)) == 1  # the unchanged "history" blob
        assert len(_blobs(after) - _blobs(before)) == 1  # only "n" is new

    async def test_capture_is_a_detached_copy(self) -> None:
        """Mutating state after capture does not change what was captured."""
        scopes, root = _registry({"items": [1]})
        flat = scopes.capture()
        root.data["items"].append(2)
        assert flat[("state",)] == {"items": [1]}

    async def test_unserializable_scope_names_its_path(self) -> None:
        scopes, root = _registry({})
        scopes.open(("n", "call"), State(data={"handle": object()}, _parent=root))
        with pytest.raises(TypeError, match="scope at 'n/call' cannot be checkpointed"):
            scopes.capture()

    async def test_chain_is_the_scopes_on_the_path_root_first(self) -> None:
        snapshot = Snapshot(
            has_state=True,
            root={},
            scopes={"n/a": {"d": 1}, "n/a/n/b": {"d": 2}, "n/c": {"d": 9}},
        )
        assert snapshot.chain("n/a/n/b") == [{"d": 1}, {"d": 2}]
        assert snapshot.chain("n/c") == [{"d": 9}]
        assert snapshot.chain("") == []


class TestRegistry:
    def test_path_of_finds_a_live_scope_by_identity(self) -> None:
        scopes, root = _registry({})
        child = State(data={}, _parent=root)
        scopes.open(("n", "x"), child)
        assert scopes.path_of(child) == ("n", "x")
        assert scopes.path_of(root) == ()

    def test_close_drops_the_scope(self) -> None:
        scopes, root = _registry({})
        scopes.open(("n", "x"), State(data={"v": 1}, _parent=root))
        scopes.close(("n", "x"))
        assert list(scopes.capture()) == [("state",)]


class TestFlowSnapshots:
    async def test_a_checkpoint_in_one_map_item_holds_every_live_item_scope(self) -> None:
        """Siblings' scopes are in the snapshot, not only the saving item's chain."""
        store = InMemoryCheckpointStore()
        started, saved = asyncio.Event(), asyncio.Event()
        seen: list[int] = []

        @verb
        async def work(ctx: Context[dict[str, Any]], item: int) -> int:
            ctx.state.data["item"] = item
            seen.append(item)
            if len(seen) == 3:
                started.set()
            await started.wait()
            if item == 0:
                await ctx.checkpoint()
                saved.set()
            await saved.wait()  # every item stays live until item 0 has saved
            return item

        await (
            FlowFactory(LG)
            .create(state={})
            .with_checkpointer(store, "siblings")
            .map(
                lambda b: b.call(work),
                items=lambda _p, _c: [0, 1, 2],
                state=lambda _p: {},
                merge=lambda _p, _c: None,
            )
            .run()
        )
        history = History(store, "siblings")
        saved = [c async for c in history.commits() if c.meta.node_path != "$end"]
        (commit,) = saved
        snapshot = await history.snapshot(commit)
        assert sorted(v["item"] for v in snapshot.scopes.values()) == [0, 1, 2]
        assert snapshot.scopes[commit.meta.scope_path] == {"item": 0}

    async def test_iterate_pass_counter_is_in_the_snapshot(self) -> None:
        store = InMemoryCheckpointStore()

        @verb
        async def step(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            ctx.state.data["n"] = ctx.state.data.get("n", 0) + 1
            if ctx.state.data["n"] == 3:
                await ctx.checkpoint()

        await (
            FlowFactory(LG)
            .create(state={})
            .with_checkpointer(store, "passes")
            .iterate(lambda b: b.call(step), max_iters=5)
            .run()
        )
        history = History(store, "passes")
        saved = [c async for c in history.commits() if c.meta.node_path != "$end"]
        (commit,) = saved
        snapshot = await history.snapshot(commit)
        assert list(snapshot.passes.values()) == [2]  # third pass, 0-based
        assert snapshot.root == {"n": 3}

    async def test_final_state_commit_holds_only_the_root(self) -> None:
        store = InMemoryCheckpointStore()

        @verb
        async def step(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            ctx.state.data["done"] = True

        await (
            FlowFactory(LG)
            .create(state={})
            .with_checkpointer(store, "end")
            .call(
                FlowFactory(LG).create().iterate(lambda b: b.call(step), max_iters=2),
                state=lambda _p: {},
            )
            .run()
        )
        history = History(store, "end")
        head = await history.head()
        assert head is not None
        snapshot = await history.snapshot(head)
        assert (snapshot.root, snapshot.scopes, snapshot.passes) == ({}, {}, {})
