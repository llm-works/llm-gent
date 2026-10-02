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
from llm_gent.flow.structure import FlowStructure

from .conftest import make_test_logger


pytestmark = pytest.mark.unit

LG = make_test_logger()


_STRUCTURE = FlowStructure(()).blob()


async def _round_trip(scopes: ScopeRegistry) -> tuple[Snapshot, dict[str, Blob | Tree]]:
    """Build the snapshot tree for ``scopes`` and read it back from the objects it made."""
    tree, objects = build_snapshot_tree(scopes.capture(), _STRUCTURE)
    by_hash = {obj.content_hash: obj for obj in objects}

    async def load(kind: Any, content_hash: str) -> bytes:
        obj = by_hash[content_hash]
        return obj.payload if isinstance(obj, Blob) else obj.to_bytes()

    return await read_snapshot(tree.content_hash, load), by_hash


def _blobs(objects: dict[str, Blob | Tree]) -> set[str]:
    """Hashes of the blobs among ``objects``, the flow structure's aside."""
    return {
        h for h, obj in objects.items() if isinstance(obj, Blob) and h != _STRUCTURE.content_hash
    }


class _FixedCursor:
    """A cursor reporting fixed entries."""

    def __init__(self, entries: dict[str, Any]) -> None:
        self.entries = entries

    def cursor(self) -> dict[str, Any]:
        return self.entries


def _registry(root: Any) -> tuple[ScopeRegistry, State[Any]]:
    scopes = ScopeRegistry()
    root_state: State[Any] = State(data=root)
    scopes.begin(root_state)
    return scopes, root_state


class TestSnapshotTree:
    async def test_round_trip_root_scopes_and_cursors(self) -> None:
        scopes, root = _registry({"a": 1, "b": [1, 2]})
        scopes.open(("n", "call"), State(data={"x": "y"}, _parent=root))
        scopes.open(("n", "map", "i", "0"), State(data={"k": 0}, _parent=root))
        scopes.open_cursor(("n", "it"), _FixedCursor({"pass": 3, "carry": {"v": [1]}}))
        scopes.open_cursor((), _FixedCursor({"chain": {"step": "s", "args": [], "kwargs": {}}}))

        snapshot, _ = await _round_trip(scopes)
        assert snapshot.has_state
        assert snapshot.root == {"a": 1, "b": [1, 2]}
        assert snapshot.scopes == {"n/call": {"x": "y"}, "n/map/i/0": {"k": 0}}
        assert snapshot.cursors == {
            "n/it": {"pass": 3, "carry": {"v": [1]}},
            "": {"chain": {"step": "s", "args": [], "kwargs": {}}},
        }

    async def test_closed_cursor_leaves_the_snapshot(self) -> None:
        scopes, _ = _registry({})
        runner = _FixedCursor({"arm": "then"})
        scopes.open_cursor(("n", "br"), runner)
        scopes.close_cursor(("n", "br"), runner)
        snapshot, _ = await _round_trip(scopes)
        assert snapshot.cursors == {}

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

    def test_saved_scope_is_handed_out_once(self) -> None:
        """A restart's saved scope goes to the first block at its path; a re-entry projects."""
        scopes = ScopeRegistry()
        saved = Snapshot(has_state=True, root={}, scopes={"n/x": {"v": 1}, "n/x/i/0": None})
        scopes.begin(State(data={}), saved)
        assert scopes.take_saved(("n", "x")) == (True, {"v": 1})
        assert scopes.take_saved(("n", "x")) == (False, None)
        assert scopes.take_saved(("n", "x", "i", "0")) == (True, None)  # a None payload is saved

    def test_drop_saved_forgets_the_scopes_no_block_took(self) -> None:
        scopes = ScopeRegistry()
        saved = Snapshot(has_state=True, root={}, scopes={"n/a": {"a": 1}, "n/b": {"b": 2}})
        scopes.begin(State(data={}), saved)
        scopes.take_saved(("n", "a"))
        assert scopes.drop_saved() == [("n", "b")]
        assert list(scopes.capture()) == [("state",)]

    def test_begin_forgets_the_previous_runs_saved_scopes(self) -> None:
        scopes = ScopeRegistry()
        scopes.begin(State(data={}), Snapshot(has_state=True, root={}, scopes={"n/x": {}}))
        scopes.begin(State(data={}))
        assert scopes.take_saved(("n", "x")) == (False, None)

    def test_untouched_saved_scopes_persist_through_capture(self) -> None:
        """Saved scopes not yet reached are included in the snapshot."""
        scopes = ScopeRegistry()
        saved = Snapshot(has_state=True, root={}, scopes={"n/a": {"a": 1}, "n/b": {"b": 2}})
        scopes.begin(State(data={"root": True}), saved)
        # Take path a, leaving b untouched
        scopes.take_saved(("n", "a"))
        scopes.open(("n", "a"), State(data={"a": 1, "touched": True}))
        flat = scopes.capture()
        # b is preserved from saved, a is from the live scope
        assert flat[("n", "a", "state")] == {"a": 1, "touched": True}
        assert flat[("n", "b", "state")] == {"b": 2}


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
            .with_checkpoint_store(store, "siblings")
            .with_checkpointer()
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

    async def test_iterate_cursor_holds_the_pass_and_the_carried_value(self) -> None:
        store = InMemoryCheckpointStore()

        @verb
        async def step(ctx: Context[dict[str, Any]], prev: int) -> int:
            ctx.state.data["n"] = ctx.state.data.get("n", 0) + 1
            if ctx.state.data["n"] == 3:
                await ctx.checkpoint()
            return prev + 10

        await (
            FlowFactory(LG)
            .create(state={})
            .with_checkpoint_store(store, "passes")
            .with_checkpointer()
            .iterate(lambda b: b.call(step), max_iters=5)
            .run(0)
        )
        history = History(store, "passes")
        saved = [c async for c in history.commits() if c.meta.node_path != "$end"]
        (commit,) = saved
        snapshot = await history.snapshot(commit)
        iterate = [c for c in snapshot.cursors.values() if "pass" in c]
        # Third pass, carrying the second's result, which until did not stop on.
        assert iterate == [{"pass": 2, "carry": 20, "until": False}]
        assert snapshot.root == {"n": 3}

    async def test_chain_cursor_holds_the_step_and_its_input(self) -> None:
        store = InMemoryCheckpointStore()

        @verb
        async def first(ctx: Context[dict[str, Any]], x: int) -> int:
            return x + 1

        @verb
        async def second(ctx: Context[dict[str, Any]], x: int) -> int:
            await ctx.checkpoint()
            return x

        flow = (
            FlowFactory(LG)
            .create(state={})
            .with_checkpoint_store(store, "chain")
            .with_checkpointer()
            .call(first)
            .call(second, project=lambda r: r * 10)
        )
        await flow.run(1)
        history = History(store, "chain")
        (commit,) = [c async for c in history.commits() if c.meta.node_path != "$end"]
        snapshot = await history.snapshot(commit)
        step = snapshot.cursors[""]["chain"]
        assert step["args"] == [20] and step["kwargs"] == {}  # input after project
        assert step["step"] == commit.meta.node_path.split("/")[-1]

    async def test_branch_cursor_holds_the_arm_it_took(self) -> None:
        store = InMemoryCheckpointStore()

        @verb
        async def save(ctx: Context[dict[str, Any]], x: int) -> int:
            await ctx.checkpoint()
            return x

        await (
            FlowFactory(LG)
            .create(state={})
            .with_checkpoint_store(store, "arm")
            .with_checkpointer()
            .branch(
                when=lambda prev, _c: prev > 0,
                then=lambda b: b.call(save),
                else_=lambda b: b.call(save),
            )
            .run(-1)
        )
        history = History(store, "arm")
        (commit,) = [c async for c in history.commits() if c.meta.node_path != "$end"]
        snapshot = await history.snapshot(commit)
        assert [c["arm"] for c in snapshot.cursors.values() if "arm" in c] == ["else"]

    async def test_unstorable_cursor_value_fails_the_checkpoint_naming_its_path(self) -> None:
        @verb
        async def save(ctx: Context[dict[str, Any]], x: Any) -> Any:
            await ctx.checkpoint()
            return x

        flow = (
            FlowFactory(LG)
            .create(state={})
            .with_checkpoint_store(InMemoryCheckpointStore(), "tuple")
            .with_checkpointer()
            .call(save)
        )
        with pytest.raises(TypeError, match=r"cursor at 'chain': a value of type tuple cannot be"):
            await flow.run((1, 2))

    async def test_a_blob_and_a_tree_with_the_same_bytes_are_both_stored(self) -> None:
        """The blob ``[]`` and the empty tree share a hash; neither put may skip the other."""
        store = InMemoryCheckpointStore()

        @verb
        async def save(ctx: Context[dict[str, Any]], _p: Any = None) -> None:
            await ctx.checkpoint()

        sub = FlowFactory(LG).create().call(save)
        await (
            FlowFactory(LG)
            .create(state={"items": []})
            .with_checkpoint_store(store, "same-bytes")
            .with_checkpointer()
            .call(sub, state=lambda _p: {}, merge=lambda _p, _c: None)
            .run()
        )
        history = History(store, "same-bytes")
        snapshots = [await history.snapshot(c) async for c in history.commits()]
        assert [s.root for s in snapshots] == [{"items": []}, {"items": []}]
        assert list(snapshots[1].scopes.values()) == [{}]

    async def test_final_state_commit_holds_only_the_root(self) -> None:
        store = InMemoryCheckpointStore()

        @verb
        async def step(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            ctx.state.data["done"] = True

        await (
            FlowFactory(LG)
            .create(state={})
            .with_checkpoint_store(store, "end")
            .with_checkpointer()
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
        assert (snapshot.root, snapshot.scopes, snapshot.cursors) == ({}, {}, {})
