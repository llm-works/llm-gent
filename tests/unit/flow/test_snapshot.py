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
from appinfra.cas import (
    Blob,
    MemoryStore,
    Object,
    Repo,
    Tree,
    TreeEntry,
    Vector,
    VectorNode,
    canonical_json,
)

from llm_gent.flow import Context, Factory, History, verb
from llm_gent.flow.state import State
from llm_gent.flow.state.snapshot import (
    ScopeRegistry,
    Snapshot,
    build_snapshot_tree,
    read_snapshot,
)
from llm_gent.flow.structure import Structure
from tests.helpers import memory_store

from .conftest import make_test_logger


pytestmark = pytest.mark.unit

LG = make_test_logger()


_STRUCTURE = Structure(()).blob()


async def _round_trip(scopes: ScopeRegistry) -> tuple[Snapshot, dict[str, Object]]:
    """Store the snapshot tree for ``scopes`` and read it back; also return the blobs and trees queued."""
    repo = _repo()
    tree, queued = _store(repo, scopes)
    objects = {oid: repo.get(oid) for oid in queued}
    return read_snapshot(tree.id, _loader(repo)), {k: v for k, v in objects.items() if v}


def _store(
    repo: Repo, scopes: ScopeRegistry, stored: frozenset[str] = frozenset()
) -> tuple[Tree, set[str]]:
    """Store the snapshot tree for ``scopes`` in ``repo`` as one batch."""
    with repo.batch() as batch:
        return build_snapshot_tree(scopes.capture(), _STRUCTURE, batch, stored)


def _blobs(objects: dict[str, Object]) -> set[str]:
    """Ids of the blobs among ``objects``, the flow structure's aside."""
    return {h for h, obj in objects.items() if isinstance(obj, Blob) and h != _STRUCTURE.id}


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


def _repo() -> Repo:
    """An empty in-memory repository."""
    return MemoryStore(LG).repo("snap")


def _loader(repo: Repo) -> Any:
    """A :class:`Loader` over ``repo``: every object it is asked for is stored."""

    def load(object_id: str, cls: type[Object]) -> Any:
        obj = repo.get(object_id, cls)
        assert obj is not None
        return obj

    return load


def _message_vector(repo: Repo, messages: list[Any]) -> Vector:
    """``messages`` stored as JSON blobs and listed in a vector."""
    return Vector.build(repo, [repo.put(Blob(canonical_json(m))) for m in messages])


def _root_with(repo: Repo, path: tuple[str, str], entry: TreeEntry) -> str:
    """A snapshot root holding an empty root scope and ``entry`` under ``path[0]``."""
    state = repo.put(Blob(canonical_json({})))
    owner = repo.put(Tree((TreeEntry(path[1], entry.type, entry.id),)))
    return repo.put(Tree((TreeEntry("state", "blob", state), TreeEntry(path[0], "tree", owner))))


class TestVectorLeaves:
    """Snapshot trees whose leaves hold vectors read back as lists."""

    def test_a_vector_inside_a_leaf_tree_reads_as_a_list(self) -> None:
        """A turn stored as nested trees with its messages in a vector reads as one dict."""
        repo = _repo()
        messages = [{"role": "user", "content": f"m{i}"} for i in range(3)]
        vector = _message_vector(repo, messages)
        conversation = repo.put(
            Tree(
                (
                    TreeEntry("messages", "vector", vector.id),
                    TreeEntry("token_count", "blob", repo.put(Blob(canonical_json(7)))),
                )
            )
        )
        turn = repo.put(
            Tree(
                (
                    TreeEntry("conversation", "tree", conversation),
                    TreeEntry("task", "blob", repo.put(Blob(canonical_json("do it")))),
                )
            )
        )
        root = _root_with(repo, ("t", "turn"), TreeEntry("turn", "tree", turn))

        snapshot = read_snapshot(root, _loader(repo))
        assert snapshot.cursors == {
            "t": {
                "turn": {
                    "conversation": {"messages": messages, "token_count": 7},
                    "task": "do it",
                }
            }
        }

    def test_a_leaf_that_is_a_vector_reads_as_a_list(self) -> None:
        repo = _repo()
        vector = _message_vector(repo, [1, 2, 3])
        root = _root_with(repo, ("n", "done"), TreeEntry("done", "vector", vector.id))
        assert read_snapshot(root, _loader(repo)).cursors == {"n": {"done": [1, 2, 3]}}

    def test_a_vector_over_several_nodes_reads_in_order(self) -> None:
        """Past one leaf node's 256 entries, the elements come back in order across nodes."""
        repo = _repo()
        values = list(range(1000))
        vector = _message_vector(repo, values)
        root_node = repo.get(vector.id, VectorNode)
        assert root_node is not None and root_node.level > 0
        root = _root_with(repo, ("n", "done"), TreeEntry("done", "vector", vector.id))
        assert read_snapshot(root, _loader(repo)).cursors == {"n": {"done": values}}

    def test_an_empty_vector_reads_as_an_empty_list(self) -> None:
        repo = _repo()
        root = _root_with(repo, ("n", "done"), TreeEntry("done", "vector", Vector.empty(repo).id))
        assert read_snapshot(root, _loader(repo)).cursors == {"n": {"done": []}}


_TURN_AT = ("n", "step", "t", "0")
"""Path of the paused turn the writer tests register."""


def _turn(messages: list[Any], conversation: dict[str, Any] | None = None) -> dict[str, Any]:
    """A paused turn's envelope dict, its conversation in kelt's ``to_dict()`` shape by default."""
    return {
        "task": "research",
        "conversation": conversation or {"messages": messages, "token_count": len(messages)},
        "used": {"iterations": 1, "total_tokens": 10},
    }


def _with_turn(turn: dict[str, Any]) -> ScopeRegistry:
    """A registry holding an empty root scope and ``turn`` paused at :data:`_TURN_AT`."""
    scopes, _ = _registry({})
    scopes.open_cursor(_TURN_AT, _FixedCursor({"turn": turn}))
    return scopes


def _entry(repo: Repo, tree_id: str, *names: str) -> TreeEntry:
    """The tree entry ``names`` lead to from the tree ``tree_id``."""
    entry = TreeEntry(names[0], "tree", tree_id)
    for name in names:
        tree = repo.get(entry.id, Tree)
        assert tree is not None
        found = tree.get(name)
        assert found is not None
        entry = found
    return entry


def _messages(n: int) -> list[dict[str, str]]:
    return [{"role": "user" if i % 2 else "assistant", "content": f"m{i}"} for i in range(n)]


class TestVectorWrites:
    """A paused turn's messages are stored as a vector; nothing else is."""

    def test_turn_messages_are_stored_as_a_vector(self) -> None:
        repo = _repo()
        turn = _turn(_messages(5))
        tree, _ = _store(repo, _with_turn(turn))

        conversation = _entry(repo, tree.id, *_TURN_AT, "turn", "conversation")
        assert conversation.type == "tree"
        assert _entry(repo, conversation.id, "messages").type == "vector"
        assert read_snapshot(tree.id, _loader(repo)).cursors == {"n/step/t/0": {"turn": turn}}

    def test_a_conversation_without_a_message_list_stays_one_blob(self) -> None:
        """A conversation of another shape is stored as it was before vectors."""
        repo = _repo()
        turn = _turn([], conversation={"history": ["a", "b"]})
        tree, _ = _store(repo, _with_turn(turn))

        assert _entry(repo, tree.id, *_TURN_AT, "turn", "conversation").type == "blob"
        assert read_snapshot(tree.id, _loader(repo)).cursors == {"n/step/t/0": {"turn": turn}}

    def test_a_list_outside_a_turn_stays_one_blob(self) -> None:
        repo = _repo()
        scopes, _ = _registry({"messages": [1, 2]})
        scopes.open_cursor(("n", "m"), _FixedCursor({"done": {"messages": [3]}}))
        tree, _ = _store(repo, scopes)
        assert _entry(repo, tree.id, "state", "messages").type == "blob"
        assert _entry(repo, tree.id, "n", "m", "done", "messages").type == "blob"

    def test_a_turn_paused_again_stores_only_the_messages_it_added(self) -> None:
        repo = _repo()
        first = _messages(40)
        _, stored = _store(repo, _with_turn(_turn(first)))

        second = _messages(42)
        _, queued = _store(repo, _with_turn(_turn(second)), frozenset(stored))
        message_blobs = {Blob(canonical_json(m)).id for m in second}
        assert queued & message_blobs == {Blob(canonical_json(m)).id for m in second[40:]}


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
        store = memory_store()
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
            Factory(LG, checkpoint_store=store)
            .create(state={})
            .with_checkpointer()
            .map(
                lambda b: b.call(work),
                items=lambda _p, _c: [0, 1, 2],
                state=lambda _p: {},
                merge=lambda _p, _c: None,
            )
            .run(history="siblings")
        )
        history = History(store, "siblings")
        saved = [c async for c in history.commits() if c.meta.node_path != "$end"]
        (commit,) = saved
        snapshot = await history.snapshot(commit)
        assert sorted(v["item"] for v in snapshot.scopes.values()) == [0, 1, 2]
        assert snapshot.scopes[commit.meta.scope_path] == {"item": 0}

    async def test_iterate_cursor_holds_the_pass_and_the_carried_value(self) -> None:
        store = memory_store()

        @verb
        async def step(ctx: Context[dict[str, Any]], prev: int) -> int:
            ctx.state.data["n"] = ctx.state.data.get("n", 0) + 1
            if ctx.state.data["n"] == 3:
                await ctx.checkpoint()
            return prev + 10

        await (
            Factory(LG, checkpoint_store=store)
            .create(state={})
            .with_checkpointer()
            .iterate(lambda b: b.call(step), max_iters=5)
            .run(0, history="passes")
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
        store = memory_store()

        @verb
        async def first(ctx: Context[dict[str, Any]], x: int) -> int:
            return x + 1

        @verb
        async def second(ctx: Context[dict[str, Any]], x: int) -> int:
            await ctx.checkpoint()
            return x

        flow = (
            Factory(LG, checkpoint_store=store)
            .create(state={})
            .with_checkpointer()
            .call(first)
            .call(second, project=lambda r: r * 10)
        )
        await flow.run(1, history="chain")
        history = History(store, "chain")
        (commit,) = [c async for c in history.commits() if c.meta.node_path != "$end"]
        snapshot = await history.snapshot(commit)
        step = snapshot.cursors[""]["chain"]
        assert step["args"] == [20] and step["kwargs"] == {}  # input after project
        assert step["step"] == commit.meta.node_path.split("/")[-1]

    async def test_branch_cursor_holds_the_arm_it_took(self) -> None:
        store = memory_store()

        @verb
        async def save(ctx: Context[dict[str, Any]], x: int) -> int:
            await ctx.checkpoint()
            return x

        await (
            Factory(LG, checkpoint_store=store)
            .create(state={})
            .with_checkpointer()
            .branch(
                when=lambda prev, _c: prev > 0,
                then=lambda b: b.call(save),
                else_=lambda b: b.call(save),
            )
            .run(-1, history="arm")
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
            Factory(LG, checkpoint_store=memory_store())
            .create(state={})
            .with_checkpointer()
            .call(save)
        )
        with pytest.raises(TypeError, match=r"cursor at 'chain': a value of type set cannot be"):
            await flow.run({1, 2}, history="unstorable")

    async def test_a_blob_and_a_tree_with_the_same_bytes_are_both_stored(self) -> None:
        """The blob ``[]`` and the empty tree share a hash; neither put may skip the other."""
        store = memory_store()

        @verb
        async def save(ctx: Context[dict[str, Any]], _p: Any = None) -> None:
            await ctx.checkpoint()

        sub = Factory(LG).create().call(save)
        await (
            Factory(LG, checkpoint_store=store)
            .create(state={"items": []})
            .with_checkpointer()
            .call(sub, state=lambda _p: {}, merge=lambda _p, _c: None)
            .run(history="same-bytes")
        )
        history = History(store, "same-bytes")
        snapshots = [await history.snapshot(c) async for c in history.commits()]
        assert [s.root for s in snapshots] == [{"items": []}, {"items": []}]
        assert list(snapshots[1].scopes.values()) == [{}]

    async def test_final_state_commit_holds_only_the_root(self) -> None:
        store = memory_store()

        @verb
        async def step(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            ctx.state.data["done"] = True

        await (
            Factory(LG, checkpoint_store=store)
            .create(state={})
            .with_checkpointer()
            .call(
                Factory(LG).create().iterate(lambda b: b.call(step), max_iters=2),
                state=lambda _p: {},
            )
            .run(history="end")
        )
        history = History(store, "end")
        head = await history.head()
        assert head is not None
        snapshot = await history.snapshot(head)
        assert (snapshot.root, snapshot.scopes, snapshot.cursors) == ({}, {}, {})
