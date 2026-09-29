# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""End-to-end Flow-level checkpoint/resume smoke tests.

Runs the canonical multi-stage flow (add_one → times_two → minus_three,
iterated) through a real :class:`JsonFileCheckpointStore` and asserts:

- fresh runs save at iterate boundaries;
- resume from a mid-run halt reaches the same final state as an
  uninterrupted run (determinism);
- the retention policy toggles gc-on-clean-exit;
- halt-triggered exit always preserves the history.

Direct-Protocol tests (objects, refs, gc) live in
:mod:`tests.unit.flow.test_stores_json` (and the PG equivalent in
:mod:`tests.integration.flow.test_stores_pg`).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from llm_gent.flow.checkpoint import HEAD_REF
from llm_gent.flow.stores import JsonFileCheckpointStore
from llm_gent.flow.testing.checkpoint import (
    assert_resume_determinism,
    build_canonical_flow,
)

from .conftest import flow_id_for, make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


@dataclass
class SaiaResult:
    """Stand-in for SAIA's task result, passed between Loop steps.

    Round-trips through ``to_dict`` / ``from_dict`` like SAIA's ``TaskResult``,
    so a checkpoint can hold it.
    """

    paused: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"paused": self.paused, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SaiaResult:
        return cls(**data)


@dataclass
class WaveTarget:
    """A map item, like an app's fan-out target."""

    qid: int


class AppHandle:
    """An app object with no JSON form (e.g. an ORM row), passed between steps."""

    def __init__(self, qid: int) -> None:
        self.qid = qid


class YieldingStore:
    """Delegates to a sync store, yielding to the event loop before every call.

    In-tree stores are synchronous; yielding lets other tasks run in the
    middle of a commit, as they do against an async store.
    """

    def __init__(self, inner: JsonFileCheckpointStore) -> None:
        self._inner = inner

    @property
    def retention(self) -> Any:
        return self._inner.retention

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        async def call(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0)
            return attr(*args, **kwargs)

        return call


def fail_first_commit(store: JsonFileCheckpointStore) -> None:
    """Make the store's first commit put raise ``OSError("disk full")``."""
    original = store.put_object
    failed: list[bool] = []

    def flaky(flow_id: str, kind: Any, content_hash: str, payload: bytes) -> None:
        if kind == "commit" and not failed:
            failed.append(True)
            raise OSError("disk full")
        original(flow_id, kind, content_hash, payload)

    store.put_object = flaky  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> JsonFileCheckpointStore:
    """A fresh store under an isolated temp root, default retention."""
    return JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")


# ---------------------------------------------------------------------------
# Fresh run — saves happen at iterate boundaries
# ---------------------------------------------------------------------------


class TestFreshRunSaves:
    async def test_saves_a_commit_per_iteration(self, store: JsonFileCheckpointStore) -> None:
        """Each successful iterate iteration writes a commit + ref."""
        await build_canonical_flow(
            make_test_logger(), max_iters=3, store=store, client_flow_id="freshrun"
        ).run()
        # A resolvable ref exists — the history reached at least one commit.
        assert store.get_ref(flow_id_for(store, "freshrun"), HEAD_REF) is not None

    async def test_default_retention_keeps_history_on_success(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """retention="retain" (default): clean-exit does NOT gc the history."""
        await build_canonical_flow(
            make_test_logger(), max_iters=2, store=store, client_flow_id="retain-1"
        ).run()
        # Successful run — but retention="retain" so the ref survives.
        assert store.get_ref(flow_id_for(store, "retain-1"), HEAD_REF) is not None


# ---------------------------------------------------------------------------
# Retention policy
# ---------------------------------------------------------------------------


class TestRetention:
    async def test_gc_on_success_prunes(self, tmp_path: Path) -> None:
        """retention="gc_on_success": clean-exit removes the history."""
        store = JsonFileCheckpointStore(
            make_test_logger(), tmp_path / "cp", retention="gc_on_success"
        )
        await build_canonical_flow(
            make_test_logger(), max_iters=2, store=store, client_flow_id="gc-1"
        ).run()
        # History collected, including the name binding: the next save starts a new one.
        assert store.get_flow_id("gc-1") is None

    async def test_halt_preserves_history_regardless_of_retention(self, tmp_path: Path) -> None:
        """A halt-triggered exit preserves the history even under gc_on_success —
        the framework only prunes on fully successful runs.
        """
        store = JsonFileCheckpointStore(
            make_test_logger(), tmp_path / "cp", retention="gc_on_success"
        )
        halt = asyncio.Event()
        await build_canonical_flow(
            make_test_logger(),
            max_iters=5,
            halt=halt,
            halt_after_iteration=2,
            store=store,
            client_flow_id="halt-preserve",
        ).run()
        assert store.get_ref(flow_id_for(store, "halt-preserve"), HEAD_REF) is not None


# ---------------------------------------------------------------------------
# Save on halt — halt-observation sites emit a commit at the halt position
# ---------------------------------------------------------------------------


class TestCheckpointPolicyIterate:
    """CheckpointPolicy.on_iterate gates the iterate-boundary auto-save."""

    async def test_default_policy_writes_only_the_final_state_commit(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Default CheckpointPolicy (halt-only) skips per-iteration saves.

        Under the default policy an iterate that runs cleanly to
        ``max_iters`` should leave the store with exactly one commit:
        the ``$end`` final-state commit written by the clean-exit
        retention path. No iterate-boundary commit should exist.
        """
        from llm_gent.flow import Context, FlowFactory, verb
        from llm_gent.flow.state.cas import Commit

        @verb
        async def bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["n"] = ctx.state.data.get("n", 0) + 1
            return ctx.state.data["n"]

        body = FlowFactory(make_test_logger()).create()
        body.call(bump)

        outer = (
            FlowFactory(make_test_logger())
            .create(state={"n": 0})
            .with_checkpointer(store, "policy-halt-only")
            .iterate(body, max_iters=5)
        )
        await outer.run()

        # Every commit written during the run has node_path == "$end"
        # (only the clean-exit final-state commit fires under the default
        # policy). Walking the commit objects on disk reveals no iterate-
        # boundary commits.
        commits_dir = (
            store._history_dir(flow_id_for(store, "policy-halt-only")) / "objects" / "commit"
        )
        node_paths: set[str] = set()
        for f in commits_dir.iterdir():
            payload = f.read_bytes()
            node_paths.add(Commit.from_bytes(payload).meta.node_path)
        assert node_paths == {"$end"}, (
            f"only the $end final-state commit should have been written; got {node_paths}"
        )

    async def test_on_iterate_true_writes_per_iteration_commits(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """CheckpointPolicy(on_iterate=True) restores per-iteration saves.

        Opting in should produce an iterate-boundary ref alongside the
        final-state commit — the iterate's own node_path directory carries
        one ref per iteration.
        """
        from llm_gent.flow import Context, FlowFactory, verb
        from llm_gent.flow.state.cas import Commit

        @verb
        async def bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["n"] = ctx.state.data.get("n", 0) + 1
            return ctx.state.data["n"]

        body = FlowFactory(make_test_logger()).create()
        body.call(bump)

        outer = (
            FlowFactory(make_test_logger())
            .create(state={"n": 0})
            .with_checkpointer(store, "policy-on-iter")
            .with_checkpoint_policy(on_iterate=True)
            .iterate(body, max_iters=3)
        )
        await outer.run()

        # Under on_iterate=True the iterate boundary emits a commit each
        # iteration + a $end final-state commit on clean exit. Group commit
        # objects by node_path: 3 boundary commits share the iterate's
        # node_path, plus one $end commit.
        commits_dir = (
            store._history_dir(flow_id_for(store, "policy-on-iter")) / "objects" / "commit"
        )
        by_node_path: dict[str, int] = {}
        for f in commits_dir.iterdir():
            commit = Commit.from_bytes(f.read_bytes())
            by_node_path[commit.meta.node_path] = by_node_path.get(commit.meta.node_path, 0) + 1
        assert by_node_path["$end"] == 1
        iterate_paths = {p: n for p, n in by_node_path.items() if p != "$end"}
        assert len(iterate_paths) == 1, f"expected one iterate node_path; got {list(iterate_paths)}"
        # 3 iterate-boundary commits (one per iteration).
        assert next(iter(iterate_paths.values())) == 3


class TestCheckpointPolicyMap:
    """CheckpointPolicy.on_map_item gates the map per-item auto-save."""

    async def test_default_policy_writes_no_map_item_commits(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Under the default policy a completed map writes only the $end final-state commit."""
        from llm_gent.flow import Context, FlowFactory, verb
        from llm_gent.flow.state.cas import Commit

        @verb
        async def touch(ctx: Context[dict[str, Any]], item: int) -> int:
            return item * 2

        body = FlowFactory(make_test_logger()).create()
        body.call(touch)

        outer = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpointer(store, "map-default")
            .map(body, items=lambda _p, _c: [1, 2, 3])
        )
        await outer.run()

        commits_dir = store._history_dir(flow_id_for(store, "map-default")) / "objects" / "commit"
        node_paths: set[str] = set()
        for f in commits_dir.iterdir():
            node_paths.add(Commit.from_bytes(f.read_bytes()).meta.node_path)
        assert node_paths == {"$end"}, (
            f"only the $end commit should exist under default policy; got {node_paths}"
        )

    async def test_on_map_item_true_writes_per_item_commits(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """CheckpointPolicy(on_map_item=True) saves after each successful item.

        Runs a 3-item map and asserts that three iteration-indexed
        commits exist under the map's node_path plus the $end
        final-state commit. Item order across saves is not asserted
        (concurrent).
        """
        from llm_gent.flow import Context, FlowFactory, verb
        from llm_gent.flow.state.cas import Commit

        @verb
        async def touch(ctx: Context[dict[str, Any]], item: int) -> int:
            return item * 2

        body = FlowFactory(make_test_logger()).create()
        body.call(touch)

        outer = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpointer(store, "map-on-item")
            .with_checkpoint_policy(on_map_item=True)
            .map(body, items=lambda _p, _c: [1, 2, 3])
        )
        await outer.run()

        commits_dir = store._history_dir(flow_id_for(store, "map-on-item")) / "objects" / "commit"
        iterations_at_map_path: dict[str, list[int]] = {}
        for f in commits_dir.iterdir():
            commit = Commit.from_bytes(f.read_bytes())
            iterations_at_map_path.setdefault(commit.meta.node_path, []).append(
                commit.meta.iteration
            )
        assert "$end" in iterations_at_map_path
        non_final = {p: v for p, v in iterations_at_map_path.items() if p != "$end"}
        assert len(non_final) == 1, f"expected one map node_path; got {list(non_final)}"
        iterations = sorted(next(iter(non_final.values())))
        assert iterations == [0, 1, 2], f"expected three item slots 0/1/2; got {iterations}"

    async def test_failed_item_commit_restores_the_state_before_the_merge(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A per-item commit that fails rolls the merge back to the prior state, not to empty."""
        from llm_gent.flow import Context, FlowFactory, verb

        fail_first_commit(store)

        @verb
        async def add(ctx: Context[dict[str, Any]], item: int) -> int:
            ctx.state.data["n"] += item
            return item

        @verb
        async def read_state(ctx: Context[dict[str, Any]]) -> dict[str, Any]:
            return dict(ctx.state.data)

        outer = (
            FlowFactory(make_test_logger())
            .create(state={"total": 1})
            .with_checkpointer(store, "map-rollback")
            .with_checkpoint_policy(on_map_item=True)
            .map(
                lambda b: b.call(add),
                items=lambda _p, _c: [5],
                state=lambda _p: {"n": 0},
                merge=lambda p, c: p.__setitem__("total", p["total"] + c["n"]),
                strict=False,
            )
            .call(read_state)
        )
        assert await outer.run() == {"total": 1}

    async def test_failed_item_commit_without_state_keeps_sibling_writes(
        self, tmp_path: Path
    ) -> None:
        """Without ``state=`` items write the shared state; a failed commit must not erase them.

        Item 1 writes while item 0's commit is in flight; item 0's commit
        fails. Restoring a pre-merge snapshot would drop item 1's write.
        """
        from llm_gent.flow import Context, FlowFactory, History, verb

        inner = JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")
        fail_first_commit(inner)
        store = YieldingStore(inner)

        @verb
        async def mark(ctx: Context[dict[str, Any]], x: int) -> int:
            if x == 1:
                for _ in range(3):
                    await asyncio.sleep(0)
            ctx.state.data[f"s{x}"] = True
            return x

        state: dict[str, Any] = {}
        outer = (
            FlowFactory(make_test_logger())
            .create(state=state)
            .with_checkpointer(store, "map-siblings")  # type: ignore[arg-type]
            .with_checkpoint_policy(on_map_item=True)
            .map(lambda b: b.call(mark), items=lambda _p, _c: [0, 1], strict=False)
        )
        await outer.run()
        assert state == {"s0": True, "s1": True}
        history = History(store, "map-siblings")  # type: ignore[arg-type]
        head = await history.head()
        assert head is not None
        assert (await history.snapshot(head)).root == {"s0": True, "s1": True}


class TestCommitConsistency:
    async def test_every_scope_in_a_commit_comes_from_one_moment(self, tmp_path: Path) -> None:
        """A commit's scopes are all serialized before its first store write.

        Each item bumps a counter in the root and in the subflow's scope in
        one step, so the two always agree. Items interleave while a
        commit awaits the store; serializing a scope after a write would
        commit one counter from before another item's step and one from after.
        """
        from llm_gent.flow import Context, FlowFactory, History, verb

        store = YieldingStore(JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp"))

        @verb
        async def bump(ctx: Context[dict[str, Any]], _item: int) -> None:
            await asyncio.sleep(0)
            ctx.state.data["n"] += 1
            ctx.state.root().data["n"] += 1
            await ctx.checkpoint()

        sub = FlowFactory(make_test_logger()).create()
        sub.map(lambda b: b.call(bump), items=lambda _p, _c: list(range(6)))
        await (
            FlowFactory(make_test_logger())
            .create(state={"n": 0})
            .with_checkpointer(store, "one-moment")  # type: ignore[arg-type]
            .call(sub, state=lambda p: {"n": p["n"]})
            .run()
        )
        history = History(store, "one-moment")  # type: ignore[arg-type]
        snapshots = [await history.snapshot(commit) async for commit in history.commits()]
        pairs = [[s.root["n"], *(c["n"] for c in s.scopes.values())] for s in snapshots]
        nested = [p for p in pairs if len(p) == 2]
        assert len(nested) == 6
        assert all(root == child for root, child in nested), nested


class TestCheckpointedRunValues:
    """A checkpointed run passes any value between steps, as an uncheckpointed one does.

    Only state is persisted: step results, map item outputs and failures
    never have to be JSON-serializable.
    """

    async def test_app_object_passed_between_steps(self, store: JsonFileCheckpointStore) -> None:
        from llm_gent.flow import Context, FlowFactory, verb

        @verb
        async def make(ctx: Context[dict[str, Any]]) -> AppHandle:
            return AppHandle(7)

        @verb
        async def read(ctx: Context[dict[str, Any]], handle: AppHandle) -> int:
            return handle.qid

        flow = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpointer(store, "app-object")
            .call(make)
            .call(read)
        )
        assert await flow.run() == 7

    async def test_non_strict_guarded_map_of_app_objects(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Items that fail, are skipped by the guard, or return app objects all reach the next step."""
        from llm_gent.flow import Context, FlowFactory, verb

        @verb
        async def dispatch(ctx: Context[dict[str, Any]], t: WaveTarget) -> AppHandle:
            if t.qid == 2:
                raise RuntimeError("inner run failed")
            return AppHandle(t.qid)

        @verb
        async def kinds(ctx: Context[dict[str, Any]], outcomes: list[Any]) -> list[str]:
            return [type(o).__name__ for o in outcomes]

        flow = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpointer(store, "wave")
            .map(
                lambda b: b.call(dispatch),
                items=lambda _p, _c: [WaveTarget(1), WaveTarget(2), WaveTarget(3)],
                strict=False,
            )
            .guard(lambda t, _c: t.qid != 3)
            .call(kinds)
        )
        assert await flow.run() == ["AppHandle", "Failure", "Skipped"]

    async def test_non_strict_map_with_a_failed_item(self, store: JsonFileCheckpointStore) -> None:
        from llm_gent.flow import Context, FlowFactory, verb

        @verb
        async def double(ctx: Context[dict[str, Any]], x: int) -> int:
            if x == 2:
                raise RuntimeError("item failed")
            return 2 * x

        @verb
        async def summarize(ctx: Context[dict[str, Any]], results: list[Any]) -> list[Any]:
            return [r if isinstance(r, int) else "failed" for r in results]

        flow = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpointer(store, "failed-item")
            .map(lambda b: b.call(double), items=lambda _p, _c: [1, 2, 3], strict=False)
            .call(summarize)
        )
        assert await flow.run() == [2, "failed", 6]


class TestCtxCheckpoint:
    """Explicit ctx.checkpoint() writes a commit regardless of policy."""

    async def test_ctx_checkpoint_from_chain_step(self, store: JsonFileCheckpointStore) -> None:
        """A verb calling ``ctx.checkpoint()`` writes a commit at that step's node.

        Runs under the default (halt-only) policy so no iterate-
        boundary save fires; the only commit besides the final-state one
        that exists is the one the verb explicitly requested.
        """
        from llm_gent.flow import Context, FlowFactory, verb
        from llm_gent.flow.state.cas import Commit

        @verb
        async def saver(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["n"] = 42
            await ctx.checkpoint()
            return 42

        outer = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpointer(store, "ctx-ckpt")
            .call(saver)
        )
        await outer.run()

        # Two commits total: one from the explicit ctx.checkpoint() (verb's
        # chain-step node) + one $end final-state commit on clean exit.
        commits_dir = store._history_dir(flow_id_for(store, "ctx-ckpt")) / "objects" / "commit"
        node_paths: list[str] = []
        for f in commits_dir.iterdir():
            node_paths.append(Commit.from_bytes(f.read_bytes()).meta.node_path)
        assert "$end" in node_paths
        non_final = [p for p in node_paths if p != "$end"]
        assert len(non_final) == 1, (
            f"expected exactly one explicit-checkpoint commit; got {non_final}"
        )

    async def test_unchanged_state_is_not_put_again(self, store: JsonFileCheckpointStore) -> None:
        """Two checkpoints of the same state put its blob and tree once; only the commits differ."""
        from llm_gent.flow import Context, FlowFactory, verb

        puts: list[str] = []
        original = store.put_object

        def counting(flow_id: str, kind: Any, content_hash: str, payload: bytes) -> None:
            puts.append(content_hash)
            original(flow_id, kind, content_hash, payload)

        store.put_object = counting  # type: ignore[method-assign]

        @verb
        async def save_twice(ctx: Context[dict[str, Any]]) -> None:
            await ctx.checkpoint()
            await ctx.checkpoint()

        await (
            FlowFactory(make_test_logger())
            .create(state={"n": 1})
            .with_checkpointer(store, "no-reput")
            .call(save_twice)
            .run()
        )
        assert len(puts) == len(set(puts))

    async def test_ctx_checkpoint_noop_without_checkpointer(self) -> None:
        """``ctx.checkpoint()`` under a flow with no checkpointer is a no-op."""
        from llm_gent.flow import Context, FlowFactory, verb

        called = 0

        @verb
        async def saver(ctx: Context) -> None:
            nonlocal called
            called += 1
            await ctx.checkpoint()  # no checkpointer wired — must not raise

        flow = FlowFactory(make_test_logger()).create().call(saver)
        await flow.run()
        assert called == 1


class TestPausedTurnTraceRef:
    """Loop's paused conversation lands as a Blob referenced by trace_ref."""

    async def test_halt_save_stamps_paused_turn_and_writes_blob(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Loop paused mid-turn → halt commit's trace_ref points at a blob equal to canonical to_dict."""
        from dataclasses import dataclass, field

        from llm_gent.flow import Context, FlowFactory, Loop, Role, verb
        from llm_gent.flow.state.cas import Commit, canonical_json

        role = Role(name="r", backend="openai", model="gpt-4o-mini")

        @dataclass
        class _Conv:
            messages: list[str] = field(default_factory=list)

            def to_dict(self) -> dict[str, Any]:
                return {"messages": list(self.messages), "n": len(self.messages)}

        class _ConvFactory:
            def create(self) -> _Conv:
                return _Conv()

            def create_from_state(self, state: dict[str, Any]) -> _Conv:
                c = _Conv()
                c.messages = list(state.get("messages", []))
                return c

        @dataclass
        class _Result:
            paused: bool = True
            reason: str = "halt"

        halt = asyncio.Event()
        conv = _Conv(messages=["hello", "world"])
        loop = Loop(role, conversation_factory=_ConvFactory())

        # SAIA stub simulates a mid-turn pause: sets halt during complete and
        # returns paused so Loop captures the conversation before returning.
        class _PausingSAIA:
            def __init__(self, role: Role) -> None:
                self.role = role

            async def complete(self, task: str, **kwargs: Any) -> _Result:
                halt.set()
                return _Result()

        class _PausingSAIAFactory:
            def build(self, role: Role) -> _PausingSAIA:
                return _PausingSAIA(role)

        @verb(role=role)
        async def run_loop(ctx: Context, _prev: Any = None) -> Any:
            return await loop(ctx, "t", conversation=conv)

        @verb(role=role)
        async def after(ctx: Context, _prev: Any = None) -> str:
            return "not run"

        ff = FlowFactory(make_test_logger(), saia_factory=_PausingSAIAFactory())
        flow = (
            ff.create(state={})
            .with_checkpointer(store, "paused-turn-1")
            .with_halt(halt)
            .call(run_loop)
            .then(after)
        )
        await flow.run()

        halted_hash = store.get_ref(flow_id_for(store, "paused-turn-1"), HEAD_REF)
        assert halted_hash is not None
        commit = Commit.from_bytes(
            store.get_object(flow_id_for(store, "paused-turn-1"), "commit", halted_hash) or b""
        )
        assert commit.meta.outcome == "halted"
        # trace_ref carries exactly one paused_turn entry whose id encodes the
        # Loop's node_id and the blob hash; the blob bytes match canonical_json
        # of the conversation's to_dict.
        assert len(commit.meta.trace_ref) == 1
        ref = commit.meta.trace_ref[0]
        assert ref.kind == "paused_turn"
        node_id, _, blob_hash = ref.id.partition(":")
        assert node_id and blob_hash
        expected = canonical_json({"task": "t", "conversation": conv.to_dict()})
        stored = store.get_object(flow_id_for(store, "paused-turn-1"), "blob", blob_hash)
        assert stored == expected

    async def test_sibling_non_paused_clear_does_not_erase_other_loops_bytes(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Loop A pauses; sibling Loop B completes non-paused; A's bytes still land."""
        from dataclasses import dataclass, field

        from llm_gent.flow import Context, FlowFactory, Loop, Role, verb
        from llm_gent.flow.state.cas import Commit, canonical_json

        role = Role(name="r", backend="openai", model="gpt-4o-mini")

        @dataclass
        class _Conv:
            messages: list[str] = field(default_factory=list)

            def to_dict(self) -> dict[str, Any]:
                return {"messages": list(self.messages), "n": len(self.messages)}

        class _ConvFactory:
            def create(self) -> _Conv:
                return _Conv()

            def create_from_state(self, state: dict[str, Any]) -> _Conv:
                c = _Conv()
                c.messages = list(state.get("messages", []))
                return c

        _Result = SaiaResult
        halt = asyncio.Event()
        conv_a = _Conv(messages=["from-a"])
        conv_b = _Conv(messages=["from-b"])

        class _PausingSAIA:
            def __init__(self, role: Role) -> None:
                self.role = role

            async def complete(self, task: str, **kwargs: Any) -> _Result:
                return _Result(paused=True, reason="tool")

        class _HaltingSAIA:
            def __init__(self, role: Role) -> None:
                self.role = role

            async def complete(self, task: str, **kwargs: Any) -> _Result:
                halt.set()
                return _Result(paused=False)

        loop_a = Loop(role, saia=_PausingSAIA(role), conversation_factory=_ConvFactory())
        loop_b = Loop(role, saia=_HaltingSAIA(role), conversation_factory=_ConvFactory())

        @verb(role=role)
        async def run_a(ctx: Context, _prev: Any = None) -> Any:
            return await loop_a(ctx, "t", conversation=conv_a)

        @verb(role=role)
        async def run_b(ctx: Context, _prev: Any = None) -> Any:
            return await loop_b(ctx, "t", conversation=conv_b)

        @verb(role=role)
        async def after(ctx: Context, _prev: Any = None) -> str:
            return "not run"

        ff = FlowFactory(make_test_logger())
        flow = (
            ff.create(state={})
            .with_checkpointer(store, "paused-turn-multi")
            .with_halt(halt)
            .call(run_a)
            .then(run_b)
            .then(after)
        )
        await flow.run()

        halted_hash = store.get_ref(flow_id_for(store, "paused-turn-multi"), HEAD_REF)
        assert halted_hash is not None
        commit = Commit.from_bytes(
            store.get_object(flow_id_for(store, "paused-turn-multi"), "commit", halted_hash) or b""
        )
        assert commit.meta.outcome == "halted"
        # Under the old single-slot design Loop B's non-paused clear would have
        # erased Loop A's bytes. With per-node_id keying A's entry survives and
        # halt-save stamps it.
        saia_refs = [r for r in commit.meta.trace_ref if r.kind == "paused_turn"]
        assert len(saia_refs) == 1
        _, _, blob_hash = saia_refs[0].id.partition(":")
        expected = canonical_json({"task": "t", "conversation": conv_a.to_dict()})
        assert (
            store.get_object(flow_id_for(store, "paused-turn-multi"), "blob", blob_hash) == expected
        )

    @pytest.mark.parametrize("mode", ["replay", "restart"])
    async def test_resume_round_trip_hands_reconstructed_conv_and_resume_true(
        self, store: JsonFileCheckpointStore, mode: str
    ) -> None:
        """Halt mid-Loop-turn on run 1 → replay dispatches SAIA with resume=True + rebuilt conv.

        Restart does not offer paused turns (a step's node id does not
        identify a map item across runs): its Loop dispatches start fresh.
        """
        from dataclasses import dataclass, field

        from llm_gent.flow import Context, FlowFactory, Loop, Role, verb

        role = Role(name="r", backend="openai", model="gpt-4o-mini")

        @dataclass
        class _Conv:
            messages: list[str] = field(default_factory=list)

            def to_dict(self) -> dict[str, Any]:
                return {"messages": list(self.messages)}

        class _ConvFactory:
            def create(self) -> _Conv:
                return _Conv()

            def create_from_state(self, state: dict[str, Any]) -> _Conv:
                c = _Conv()
                c.messages = list(state.get("messages", []))
                return c

        _Result = SaiaResult

        # Recording SAIA — captures every call so the test can assert what
        # the resume-side dispatch handed to complete().
        complete_calls: list[dict[str, Any]] = []

        class _RecordingSAIA:
            def __init__(self, role: Role, halt: asyncio.Event, phase: str) -> None:
                self.role = role
                self._halt = halt
                self._phase = phase

            async def complete(self, task: str, **kwargs: Any) -> _Result:
                complete_calls.append(
                    {
                        "phase": self._phase,
                        "conversation": kwargs.get("conversation"),
                        "resume": kwargs.get("resume", False),
                    }
                )
                if self._phase == "first":
                    self._halt.set()
                    return _Result(paused=True, reason="halt")
                return _Result(paused=False)

        class _PhaseFactory:
            def __init__(self, halt: asyncio.Event, phase: str) -> None:
                self._halt = halt
                self._phase = phase

            def build(self, role: Role) -> _RecordingSAIA:
                return _RecordingSAIA(role, self._halt, self._phase)

        # ONE Loop instance and ONE pair of verbs — same qualnames across both
        # runs so ctx._node_id is stable and the resume dict entry matches.
        loop = Loop(role, conversation_factory=_ConvFactory())

        @verb(role=role)
        async def run_loop(ctx: Context, _prev: Any = None) -> Any:
            return await loop(ctx, "t", conversation=ctx.extra["conv"])

        @verb(role=role)
        async def after(ctx: Context, _prev: Any = None) -> str:
            return "ran-after"

        # Iterate body so halt-save lands at the iterate's node with the
        # paused iteration index — resume re-runs that iteration, which is
        # when the Loop's re-dispatch consumes the paused_turn entry. A pure
        # chain would halt-save at the NEXT chain step, skipping the
        # paused Loop entirely.
        body_ff = FlowFactory(make_test_logger())
        body = body_ff.create()
        body.call(run_loop)

        # ---- Run 1: iterate iteration 0 halts mid-Loop-turn.
        halt1 = asyncio.Event()
        ff1 = FlowFactory(make_test_logger(), saia_factory=_PhaseFactory(halt1, "first"))
        flow1 = (
            ff1.create(state={})
            .with_checkpointer(store, "resume-round-trip")
            .with_halt(halt1)
            .iterate(body, max_iters=2)
        )
        await flow1.run(extra={"conv": _Conv(messages=["from-turn-1"])})
        assert [c["phase"] for c in complete_calls] == ["first"]

        # ---- Run 2: resume="replay" re-runs iteration 0. The Loop's re-dispatch
        # picks up the paused_turn entry and hands SAIA the reconstructed
        # conversation with resume=True. The caller supplies a different
        # conversation — the resume path must override it with the rebuilt
        # one from the halted commit.
        halt2 = asyncio.Event()
        ff2 = FlowFactory(make_test_logger(), saia_factory=_PhaseFactory(halt2, "resume"))
        flow2 = (
            ff2.create(state={})
            .with_checkpointer(store, "resume-round-trip")
            .with_halt(halt2)
            .iterate(body, max_iters=2)
        )
        await flow2.run(
            resume=mode,  # type: ignore[arg-type]
            extra={"conv": _Conv(messages=["caller-supplied-but-overridden"])},
        )

        resume_calls = [c for c in complete_calls if c["phase"] == "resume"]
        if mode == "restart":
            # Counter starts at 0 (two passes); neither dispatch resumes a turn.
            assert [c["resume"] for c in resume_calls] == [False, False]
            return
        # Replay resumes at the halted iteration with max_iters cumulative.
        assert len(resume_calls) == 1
        resumed = resume_calls[0]
        assert resumed["resume"] is True
        assert isinstance(resumed["conversation"], _Conv)
        assert resumed["conversation"].messages == ["from-turn-1"]

    async def test_chain_resume_re_dispatches_paused_loop_verb(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Chain flow with Loop verb: halt on turn 1 → resume re-runs the loop verb."""
        from dataclasses import dataclass, field

        from llm_gent.flow import Context, FlowFactory, Loop, Role, verb

        role = Role(name="r", backend="openai", model="gpt-4o-mini")

        @dataclass
        class _Conv:
            messages: list[str] = field(default_factory=list)

            def to_dict(self) -> dict[str, Any]:
                return {"messages": list(self.messages)}

        class _ConvFactory:
            def create(self) -> _Conv:
                return _Conv()

            def create_from_state(self, state: dict[str, Any]) -> _Conv:
                c = _Conv()
                c.messages = list(state.get("messages", []))
                return c

        _Result = SaiaResult
        after_calls: list[int] = []
        complete_calls: list[dict[str, Any]] = []

        class _RecordingSAIA:
            def __init__(self, role: Role, halt: asyncio.Event, phase: str) -> None:
                self.role = role
                self._halt = halt
                self._phase = phase

            async def complete(self, task: str, **kwargs: Any) -> _Result:
                complete_calls.append(
                    {
                        "phase": self._phase,
                        "task": task,
                        "resume": kwargs.get("resume", False),
                        "conversation": kwargs.get("conversation"),
                    }
                )
                if self._phase == "first":
                    self._halt.set()
                    return _Result(paused=True)
                return _Result(paused=False)

        class _PhaseFactory:
            def __init__(self, halt: asyncio.Event, phase: str) -> None:
                self._halt = halt
                self._phase = phase

            def build(self, role: Role) -> _RecordingSAIA:
                return _RecordingSAIA(role, self._halt, self._phase)

        loop = Loop(role, conversation_factory=_ConvFactory())

        @verb(role=role)
        async def run_loop(ctx: Context, _prev: Any = None) -> Any:
            return await loop(ctx, ctx.extra["task"], conversation=ctx.extra["conv"])

        @verb(role=role)
        async def after_step(ctx: Context, _prev: Any = None) -> str:
            after_calls.append(1)
            return "ran-after"

        # Run 1: chain halts mid-Loop-turn. With the chain-halt fix, the halt
        # commit lands at run_loop's node (not after_step's) so resume re-runs
        # the loop verb.
        halt1 = asyncio.Event()
        ff1 = FlowFactory(make_test_logger(), saia_factory=_PhaseFactory(halt1, "first"))
        flow1 = (
            ff1.create(state={})
            .with_checkpointer(store, "chain-resume")
            .with_halt(halt1)
            .call(run_loop)
            .then(after_step)
        )
        await flow1.run(extra={"task": "saved-task", "conv": _Conv(messages=["turn-1"])})
        assert after_calls == []  # halt observed before after_step ran

        # Run 2: resume. Loop is re-dispatched, consumes paused_turn, SAIA
        # completes non-paused, chain moves on to after_step.
        halt2 = asyncio.Event()
        ff2 = FlowFactory(make_test_logger(), saia_factory=_PhaseFactory(halt2, "resume"))
        flow2 = (
            ff2.create(state={})
            .with_checkpointer(store, "chain-resume")
            .with_halt(halt2)
            .call(run_loop)
            .then(after_step)
        )
        result = await flow2.run(
            resume="replay",
            # Caller passes DIFFERENT task + conv on run 2 to prove the resume
            # path forwards the SAVED values from the envelope, not the ones
            # the verb happened to hand this dispatch.
            extra={"task": "caller-task", "conv": _Conv(messages=["overridden"])},
        )

        resume_calls = [c for c in complete_calls if c["phase"] == "resume"]
        assert len(resume_calls) == 1
        assert resume_calls[0]["resume"] is True
        # SAIA must receive the checkpoint-restored Conversation, not the
        # caller-supplied _Conv(["overridden"]).
        assert isinstance(resume_calls[0]["conversation"], _Conv)
        assert resume_calls[0]["conversation"].messages == ["turn-1"]
        # SAIA must receive the SAVED task from the envelope, not the caller's
        # "caller-task" on run 2 — a regression that forwarded the caller task
        # instead would fail this line.
        assert resume_calls[0]["task"] == "saved-task"
        assert after_calls == [1]
        assert result == "ran-after"

    async def test_resume_restores_saved_task_from_paused_turn_envelope(self) -> None:
        """Loop._consume_resume_entry decodes both task and conversation from the envelope."""
        from dataclasses import dataclass, field
        from types import SimpleNamespace

        from llm_gent.flow import Loop, Role
        from llm_gent.flow.state.cas import canonical_json
        from llm_gent.flow.state.paused_turn import ResumePausedTurns

        role = Role(name="r", backend="openai", model="gpt-4o-mini")

        @dataclass
        class _Conv:
            messages: list[str] = field(default_factory=list)

            def to_dict(self) -> dict[str, Any]:
                return {"messages": list(self.messages)}

        class _ConvFactory:
            def create(self) -> _Conv:
                return _Conv()

            def create_from_state(self, state: dict[str, Any]) -> _Conv:
                c = _Conv()
                c.messages = list(state.get("messages", []))
                return c

        node_id = "node-abc"
        loop = Loop(role, conversation_factory=_ConvFactory())

        # Seed a runtime resume-map with a paused_turn envelope carrying BOTH the
        # task and the conversation state. This is exactly the shape
        # _hydrate_resume_state populates from a halt commit's paused_turn blob.
        resume_turns = ResumePausedTurns()
        resume_turns.add(
            node_id,
            canonical_json(
                {"task": "saved-task-string", "conversation": {"messages": ["mid-turn"]}}
            ),
        )
        env = SimpleNamespace(resume_paused_turns=resume_turns)
        ctx = SimpleNamespace(_env=env, _node_id=node_id)

        task, conversation, is_resume = loop._consume_resume_entry(ctx)  # type: ignore[arg-type]
        assert is_resume is True
        # Without this test the direct-Loop chain-step at index > 0 case CR
        # flagged would silently lose the task on resume; _walk_chain calls
        # Loop.__call__(ctx) and _prepare_dispatch has no task to hand SAIA.
        assert task == "saved-task-string"
        assert isinstance(conversation, _Conv)
        assert conversation.messages == ["mid-turn"]
        # Entry is NOT popped by _consume_resume_entry — release happens only
        # after saia.complete succeeds, so a rescue-then-iterate-retry can
        # re-consume the same envelope.
        assert resume_turns.load(node_id) is not None


class TestSaveOnHaltChain:
    """Chain-only halt-observation site writes a commit at the not-yet-run step."""

    async def test_chain_only_halt_saves_commit_and_resume_lands_on_halted_step(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Chain flow with no iterate: halt set by step b → commit at c; resume lands at c."""
        from llm_gent.flow import Context, FlowFactory, verb
        from llm_gent.flow.state.cas import Commit

        halt = asyncio.Event()

        @verb
        async def step_a(ctx: Context[dict[str, Any]], _prev: Any = None) -> int:
            ctx.state.data["a"] = 1
            return 1

        @verb
        async def step_b(ctx: Context[dict[str, Any]], prev: int) -> int:
            ctx.state.data["b"] = prev + 1
            halt.set()
            return ctx.state.data["b"]

        @verb
        async def step_c(ctx: Context[dict[str, Any]], _prev: Any = None) -> int:
            # Reads from state — the halted resume runs step_c with no prev_result
            # per _walk_chain's start_index > 0 contract.
            ctx.state.data["c"] = ctx.state.data["b"] + 1
            return ctx.state.data["c"]

        ff = FlowFactory(make_test_logger())
        pre = (
            ff.create(state={})
            .with_checkpointer(store, "chain-halt")
            .with_halt(halt)
            .call(step_a)
            .then(step_b)
            .then(step_c)
        )
        await pre.run()

        halted_hash = store.get_ref(flow_id_for(store, "chain-halt"), HEAD_REF)
        assert halted_hash is not None
        commit = Commit.from_bytes(
            store.get_object(flow_id_for(store, "chain-halt"), "commit", halted_hash) or b""
        )
        assert commit.meta.outcome == "halted"
        assert commit.meta.iteration == 0
        # node_path's last segment is the not-yet-run step's chain id — step_c's.
        # Independent of the exact id, the run must not have executed step_c yet:
        # the pre-halt state has only a and b, no c.

        halt.clear()
        resume = (
            ff.create(state={})
            .with_checkpointer(store, "chain-halt")
            .call(step_a)
            .then(step_b)
            .then(step_c)
        )
        result = await resume.run(resume="replay")
        # step_c ran on resume; a and b restored from the halt commit.
        assert result == 3

    async def test_halt_without_checkpointer_is_noop(self) -> None:
        """No checkpointer bound → halt observed → no store activity."""
        from llm_gent.flow import Context, FlowFactory, verb

        halt = asyncio.Event()

        @verb
        async def a(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            halt.set()

        @verb
        async def b(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            pass

        ff = FlowFactory(make_test_logger())
        flow = ff.create(state={}).with_halt(halt).call(a).then(b)
        # No .with_checkpointer — halt check fires between chain steps but
        # _save_halt_checkpoint short-circuits at env.checkpoint_ctx is None.
        # Run should complete without raising.
        await flow.run()

    async def test_halt_before_branch_with_plain_verb_resumes_at_branch(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Halt before a Branch containing only plain verbs → resume runs the branch."""
        from llm_gent.flow import Context, FlowFactory, verb
        from llm_gent.flow.state.cas import Commit

        halt = asyncio.Event()

        @verb
        async def step_a(ctx: Context[dict[str, Any]], _prev: Any = None) -> int:
            ctx.state.data["a"] = 1
            halt.set()
            return 1

        @verb
        async def branch_verb(ctx: Context[dict[str, Any]], _prev: Any = None) -> int:
            # On resume, prev is None — read from state instead (same contract as
            # _walk_chain's start_index > 0 behavior for plain chain steps).
            ctx.state.data["branch"] = ctx.state.data["a"] + 10
            return ctx.state.data["branch"]

        @verb
        async def step_c(ctx: Context[dict[str, Any]], prev: int) -> int:
            ctx.state.data["c"] = prev + 100
            return ctx.state.data["c"]

        ff = FlowFactory(make_test_logger())
        pre = (
            ff.create(state={})
            .with_checkpointer(store, "branch-halt")
            .with_halt(halt)
            .call(step_a)
            .branch(when=lambda _p, _c: True, then=lambda f: f.call(branch_verb))
            .then(step_c)
        )
        await pre.run()

        # Halt fired after step_a, commit saved at the branch position.
        halted_hash = store.get_ref(flow_id_for(store, "branch-halt"), HEAD_REF)
        assert halted_hash is not None
        commit = Commit.from_bytes(
            store.get_object(flow_id_for(store, "branch-halt"), "commit", halted_hash) or b""
        )
        assert commit.meta.outcome == "halted"

        # Resume: branch runs (with its plain verb), then step_c.
        halt.clear()
        resume = (
            ff.create(state={})
            .with_checkpointer(store, "branch-halt")
            .call(step_a)
            .branch(when=lambda _p, _c: True, then=lambda f: f.call(branch_verb))
            .then(step_c)
        )
        result = await resume.run(resume="replay")
        # 1 (from restored a) + 10 (branch_verb) + 100 (step_c) = 111
        assert result == 111

    async def test_pre_set_halt_with_checkpointer_resumes_at_second_step(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Pre-set halt + checkpointer: checkpoint saved before second step; resume runs it."""
        from llm_gent.flow import Context, FlowFactory, verb
        from llm_gent.flow.state.cas import Commit

        halt = asyncio.Event()
        halt.set()  # Pre-set before run

        @verb
        async def step_a(ctx: Context[dict[str, Any]], items: list[int]) -> list[int]:
            ctx.state.data["items"] = items
            return items

        @verb
        async def step_b(ctx: Context[dict[str, Any]], _prev: Any = None) -> int:
            # Reads from state — halted resume runs step_b with no prev_result
            return sum(ctx.state.data["items"])

        ff = FlowFactory(make_test_logger())
        pre = (
            ff.create(state={})
            .with_checkpointer(store, "pre-set-halt")
            .with_halt(halt)
            .call(step_a)
            .then(step_b)
        )
        await pre.run([1, 2, 3])

        # Checkpoint saved at step_b (halt observed after step_a)
        halted_hash = store.get_ref(flow_id_for(store, "pre-set-halt"), HEAD_REF)
        assert halted_hash is not None
        commit = Commit.from_bytes(
            store.get_object(flow_id_for(store, "pre-set-halt"), "commit", halted_hash) or b""
        )
        assert commit.meta.outcome == "halted"

        # Resume with halt cleared — step_b runs, reads items from state
        halt.clear()
        resume = (
            ff.create(state={}).with_checkpointer(store, "pre-set-halt").call(step_a).then(step_b)
        )
        result = await resume.run(resume="replay")
        assert result == 6  # sum([1, 2, 3])

    async def test_nested_flow_halt_does_not_clobber_outer_checkpoint(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Nested flow's halt doesn't emit checkpoint — only outer flow does."""
        from llm_gent.flow import Context, FlowFactory, verb
        from llm_gent.flow.state.cas import Commit

        halt = asyncio.Event()

        @verb
        async def outer_a(ctx: Context[dict[str, Any]], _prev: Any = None) -> int:
            ctx.state.data["outer_a"] = 1
            return 1

        @verb
        async def inner_a(ctx: Context[dict[str, Any]], prev: int) -> int:
            ctx.state.data["inner_a"] = prev + 10
            halt.set()  # Halt set inside nested flow
            return ctx.state.data["inner_a"]

        @verb
        async def inner_b(ctx: Context[dict[str, Any]], prev: int) -> int:
            ctx.state.data["inner_b"] = prev + 100
            return ctx.state.data["inner_b"]

        @verb
        async def outer_b(ctx: Context[dict[str, Any]], _prev: Any = None) -> int:
            # Reads from state — halted resume runs with no prev_result
            return ctx.state.data.get("inner_b", 0) + 1000

        ff = FlowFactory(make_test_logger())
        inner = ff.create().call(inner_a).then(inner_b)

        pre = (
            ff.create(state={})
            .with_checkpointer(store, "nested-halt")
            .with_halt(halt)
            .call(outer_a)
            .call(inner)
            .then(outer_b)
        )
        await pre.run()

        # Checkpoint saved at outer_b (halt observed between outer's call(inner) and outer_b)
        # NOT at inner_b (nested flow's chain-walk skips halt observation)
        halted_hash = store.get_ref(flow_id_for(store, "nested-halt"), HEAD_REF)
        assert halted_hash is not None
        commit = Commit.from_bytes(
            store.get_object(flow_id_for(store, "nested-halt"), "commit", halted_hash) or b""
        )
        assert commit.meta.outcome == "halted"

        # Resume — outer_b runs, inner flow doesn't re-run
        halt.clear()
        inner_resume = ff.create().call(inner_a).then(inner_b)
        resume = (
            ff.create(state={})
            .with_checkpointer(store, "nested-halt")
            .call(outer_a)
            .call(inner_resume)
            .then(outer_b)
        )
        result = await resume.run(resume="replay")
        # inner_b ran before halt (state has inner_b=111), outer_b adds 1000
        assert result == 1111


class TestSaveOnHaltIterate:
    """Iterate halt-observation site writes a commit at the halted iteration."""

    async def test_halt_before_first_iteration_saves_at_iteration_zero(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Halt already set on entry to the iterate → commit saved at iteration=0."""
        from llm_gent.flow.state.cas import Commit

        halt = asyncio.Event()
        halt.set()  # halt observed on the first loop-top check, before any body run.
        await build_canonical_flow(
            make_test_logger(),
            max_iters=5,
            halt=halt,
            halt_after_iteration=999,  # internal halt_check never fires; halt is pre-set externally.
            store=store,
            client_flow_id="halt-iter-0",
        ).run()

        halted_hash = store.get_ref(flow_id_for(store, "halt-iter-0"), HEAD_REF)
        assert halted_hash is not None
        commit = Commit.from_bytes(
            store.get_object(flow_id_for(store, "halt-iter-0"), "commit", halted_hash) or b""
        )
        assert commit.meta.outcome == "halted"
        assert commit.meta.iteration == 0

        # Resume with halt cleared — completes the full iterate.
        halt.clear()
        result = await build_canonical_flow(
            make_test_logger(), max_iters=5, store=store, client_flow_id="halt-iter-0"
        ).run(resume="replay")
        assert result["iterations_completed"] == 5

    async def test_halt_between_iterations_overwrites_ok_ref_with_halted_commit(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Halt after body N → halted commit overwrites the ok ref at iteration=N."""
        from llm_gent.flow.state.cas import Commit

        halt = asyncio.Event()
        await build_canonical_flow(
            make_test_logger(),
            max_iters=5,
            halt=halt,
            halt_after_iteration=2,
            store=store,
            client_flow_id="halt-iter-mid",
        ).run()

        halted_hash = store.get_ref(flow_id_for(store, "halt-iter-mid"), HEAD_REF)
        assert halted_hash is not None
        commit = Commit.from_bytes(
            store.get_object(flow_id_for(store, "halt-iter-mid"), "commit", halted_hash) or b""
        )
        # Halt fired inside body 2's halt_check (iteration counter=2 by the time the loop-top
        # check observed halt). Post-halt-save iteration matches.
        assert commit.meta.outcome == "halted"
        assert commit.meta.iteration == 2

        # Resume completes the remaining iterations.
        halt.clear()
        result = await build_canonical_flow(
            make_test_logger(), max_iters=5, store=store, client_flow_id="halt-iter-mid"
        ).run(resume="replay")
        assert result["iterations_completed"] == 5


# ---------------------------------------------------------------------------
# History lineage — every commit's parent is the previous head
# ---------------------------------------------------------------------------


def _chain_from_head(store: JsonFileCheckpointStore, client_flow_id: str) -> list[Any]:
    """Walk parent links from the newest ref; return commits newest-first."""
    from llm_gent.flow.state.cas import Commit

    flow_id = flow_id_for(store, client_flow_id)
    chain: list[Commit] = []
    commit_hash = store.get_ref(flow_id, HEAD_REF)
    while commit_hash is not None:
        commit = Commit.from_bytes(store.get_object(flow_id, "commit", commit_hash) or b"")
        chain.append(commit)
        assert len(commit.parent_hashes) <= 1
        commit_hash = commit.parent_hashes[0] if commit.parent_hashes else None
    return chain


def _stored_commit_hashes(store: JsonFileCheckpointStore, client_flow_id: str) -> set[str]:
    """Every commit object on disk under the history."""
    commits_dir = store._history_dir(flow_id_for(store, client_flow_id)) / "objects" / "commit"
    return {f.name for f in commits_dir.iterdir()}


class TestHistoryLineage:
    async def test_iterate_commits_form_one_chain(self, store: JsonFileCheckpointStore) -> None:
        """Per-iteration commits chain onto each other; the final-state commit is the head."""
        await build_canonical_flow(
            make_test_logger(), max_iters=3, store=store, client_flow_id="lineage-iter"
        ).run()

        chain = _chain_from_head(store, "lineage-iter")
        assert chain[0].meta.node_path == "$end"
        assert chain[-1].parent_hashes == ()
        # Every stored commit is reachable from the head — no orphaned siblings.
        assert {c.content_hash for c in chain} == _stored_commit_hashes(store, "lineage-iter")
        assert len(chain) == 4

    async def test_concurrent_map_items_chain_linearly(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Parallel map-item saves serialize into one chain, not siblings of one parent."""
        from llm_gent.flow import Context, FlowFactory, verb

        @verb
        async def touch(ctx: Context[dict[str, Any]], item: int) -> int:
            await asyncio.sleep(0)
            return item * 2

        body = FlowFactory(make_test_logger()).create()
        body.call(touch)
        outer = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpointer(store, "lineage-map")
            .with_checkpoint_policy(on_map_item=True)
            .map(body, items=lambda _p, _c: [1, 2, 3, 4])
        )
        await outer.run()

        chain = _chain_from_head(store, "lineage-map")
        assert {c.content_hash for c in chain} == _stored_commit_hashes(store, "lineage-map")
        assert len(chain) == 5
        assert sorted(c.meta.iteration for c in chain[1:]) == [0, 1, 2, 3]

    async def test_resume_continues_the_same_history(self, store: JsonFileCheckpointStore) -> None:
        """A resumed run's first commit takes the halted run's last commit as parent."""
        halt = asyncio.Event()
        await build_canonical_flow(
            make_test_logger(),
            max_iters=5,
            halt=halt,
            halt_after_iteration=2,
            store=store,
            client_flow_id="lineage-resume",
        ).run()
        halted_head = store.get_ref(flow_id_for(store, "lineage-resume"), HEAD_REF)
        assert halted_head is not None

        halt.clear()
        await build_canonical_flow(
            make_test_logger(), max_iters=5, store=store, client_flow_id="lineage-resume"
        ).run(resume="replay")

        chain = _chain_from_head(store, "lineage-resume")
        assert halted_head in {c.content_hash for c in chain[1:]}
        assert {c.content_hash for c in chain} == _stored_commit_hashes(store, "lineage-resume")
        assert sum(1 for c in chain if not c.parent_hashes) == 1

    async def test_gc_history_restarts_the_chain(self, store: JsonFileCheckpointStore) -> None:
        """After gc_history the next commit starts a new history with no parent."""
        from llm_gent.flow._checkpoint_ctx import CheckpointContext
        from llm_gent.flow.state.cas import Tree

        ctx = CheckpointContext(store, "lineage-gc", lambda: "")
        tree = Tree.from_entries([])
        await ctx.put_tree(tree)

        async def _append(iteration: int) -> Any:
            meta = ctx._build_commit_meta(
                await ctx.ensure_flow_id(), "node", iteration, "node", "ok", ()
            )
            return await ctx.append_commit(tree.content_hash, meta)

        first = await _append(0)
        second = await _append(1)
        assert first.parent_hashes == ()
        assert second.parent_hashes == (first.content_hash,)

        await ctx.gc_history()
        await ctx.put_tree(tree)
        restarted = await _append(0)
        assert restarted.parent_hashes == ()


class TestFlowRootHash:
    async def test_every_commit_records_the_flow_root_hash(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Halt, iterate-boundary and final-state commits all carry the flow's structure hash."""
        halt = asyncio.Event()
        flow = build_canonical_flow(
            make_test_logger(),
            max_iters=4,
            halt=halt,
            halt_after_iteration=2,
            store=store,
            client_flow_id="root-hash",
        )
        await flow.run()
        halt.clear()
        resumed = build_canonical_flow(
            make_test_logger(), max_iters=4, store=store, client_flow_id="root-hash"
        )
        await resumed.run(resume="replay")

        chain = _chain_from_head(store, "root-hash")
        assert {c.meta.outcome for c in chain} == {"ok", "halted"}
        assert {c.meta.flow_root_hash for c in chain} == {flow.root_hash()}
        assert resumed.root_hash() == flow.root_hash()


class TestRunScopedCache:
    async def test_reused_flow_sees_external_gc(self, store: JsonFileCheckpointStore) -> None:
        """A Flow reused after someone else collected its history writes a new, named history."""
        from llm_gent.flow import History

        halt = asyncio.Event()
        flow = build_canonical_flow(
            make_test_logger(),
            max_iters=4,
            halt=halt,
            halt_after_iteration=2,
            store=store,
            client_flow_id="reused",
        )
        await flow.run()
        old_flow_id = store.get_flow_id("reused")
        assert old_flow_id is not None
        store.gc_history(old_flow_id)

        halt.clear()
        await flow.run()  # halts again, into a fresh history

        history = History(store, "reused")
        new_flow_id = await history.flow_id()
        assert new_flow_id is not None and new_flow_id != old_flow_id
        head = await history.head()
        assert head is not None and head.meta.outcome == "halted"
        chain = [c async for c in history.commits()]
        assert {c.meta.flow_id for c in chain} == {new_flow_id}
        assert chain[-1].parent_hashes == ()
        assert not store._history_dir(old_flow_id).exists()


class TestCompletionTag:
    """Clean exit commits the final state at ``$end`` and moves the ``complete`` tag to it."""

    async def test_unserializable_final_state_still_completes(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A finished run whose state can't be serialized returns normally; no state is kept."""
        from llm_gent.flow import Context, FlowFactory, History, verb

        @verb
        async def attach(ctx: Context[dict[str, Any]], _prev: Any = None) -> str:
            ctx.state.data["handle"] = object()
            return "done"

        result = await (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpointer(store, "opaque-state")
            .call(attach)
            .run()
        )

        assert result == "done"
        history = History(store, "opaque-state")
        head = await history.head()
        assert head is not None and History.is_final_state(head)
        assert await history.is_complete()
        assert not (await history.snapshot(head)).has_state

    async def test_final_state_commit_is_tagged_head(self, store: JsonFileCheckpointStore) -> None:
        """A halt-only run with no save points still leaves its final state at the head."""
        from llm_gent.flow import Context, FlowFactory, History, verb
        from llm_gent.flow.checkpoint import COMPLETE_TAG, COMPLETION_PRODUCER

        @verb
        async def bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["n"] += 1
            return ctx.state.data["n"]

        await (
            FlowFactory(make_test_logger())
            .create(state={"n": 0})
            .with_checkpointer(store, "final-state")
            .iterate(lambda body: body.call(bump), max_iters=3)
            .run()
        )

        flow_id = flow_id_for(store, "final-state")
        head = store.get_ref(flow_id, HEAD_REF)
        assert head is not None
        assert store.get_ref(flow_id, COMPLETE_TAG) == head
        (commit,) = _chain_from_head(store, "final-state")
        assert commit.meta.node_path == "$end"
        assert commit.meta.produced_by.node_id == COMPLETION_PRODUCER
        snapshot = await History(store, "final-state").snapshot(commit)
        assert (snapshot.root, snapshot.scopes, snapshot.cursors) == ({"n": 3}, {}, {})

    async def test_rerun_after_completion_appends_to_same_history(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """resume="replay" on a complete history runs fresh; its commits extend the history."""
        from llm_gent.flow.checkpoint import COMPLETE_TAG

        await build_canonical_flow(
            make_test_logger(), max_iters=2, store=store, client_flow_id="rerun"
        ).run()
        flow_id = flow_id_for(store, "rerun")
        first_end = store.get_ref(flow_id, HEAD_REF)

        await build_canonical_flow(
            make_test_logger(), max_iters=2, store=store, client_flow_id="rerun"
        ).run(resume="replay")

        assert flow_id_for(store, "rerun") == flow_id
        chain = _chain_from_head(store, "rerun")
        assert store.get_ref(flow_id, COMPLETE_TAG) == chain[0].content_hash
        assert first_end in {c.content_hash for c in chain[1:]}
        assert [c.meta.node_path for c in chain].count("$end") == 2

    async def test_commit_after_completion_makes_history_resumable(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A halt after a completed run leaves the tag behind; the next resume replays the halt."""
        from llm_gent.flow.checkpoint import COMPLETE_TAG
        from llm_gent.flow.testing.checkpoint import CanonicalCounter

        await build_canonical_flow(
            make_test_logger(), max_iters=2, store=store, client_flow_id="tag-behind"
        ).run()
        halt = asyncio.Event()
        await build_canonical_flow(
            make_test_logger(),
            max_iters=5,
            halt=halt,
            halt_after_iteration=2,
            store=store,
            client_flow_id="tag-behind",
        ).run(resume="replay")
        flow_id = flow_id_for(store, "tag-behind")
        assert store.get_ref(flow_id, COMPLETE_TAG) != store.get_ref(flow_id, HEAD_REF)

        # A fresh run would start from this fallback state (n=100); resume ignores it.
        resumed = await build_canonical_flow(
            make_test_logger(),
            state=CanonicalCounter(n=100),
            max_iters=5,
            store=store,
            client_flow_id="tag-behind",
        ).run(resume="replay")
        assert resumed["iterations_completed"] == 5
        assert resumed["log"][0] == 1
        assert store.get_ref(flow_id, COMPLETE_TAG) == store.get_ref(flow_id, HEAD_REF)

    async def test_untagged_final_state_head_counts_as_complete(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A crash between the $end commit and the tag write must not replay "$end"."""
        from llm_gent.flow._checkpoint_ctx import CheckpointContext
        from llm_gent.flow.checkpoint import COMPLETE_TAG
        from llm_gent.flow.state import State
        from llm_gent.flow.state.snapshot import ScopeRegistry
        from llm_gent.flow.testing.checkpoint import CanonicalCounter

        ctx = CheckpointContext(store, "torn-completion", lambda: "")
        scopes = ScopeRegistry()
        scopes.begin(State(data={"n": 7}))
        tree = await ctx.put_snapshot(scopes)
        await ctx.save_completion_commit(tree)
        assert store.get_ref(flow_id_for(store, "torn-completion"), COMPLETE_TAG) is None

        result = await build_canonical_flow(
            make_test_logger(),
            state=CanonicalCounter(n=100),
            max_iters=1,
            store=store,
            client_flow_id="torn-completion",
        ).run(resume="replay")
        assert result["log"][0] == 101


# ---------------------------------------------------------------------------
# Resume determinism — baseline vs interrupt+resume must reach same final state
# ---------------------------------------------------------------------------


class TestResumeDeterminism:
    async def test_resume_matches_uninterrupted(self, store: JsonFileCheckpointStore) -> None:
        """The load-bearing invariant: resume state == uninterrupted state."""
        final = await assert_resume_determinism(
            make_test_logger(),
            store,
            halt_after_iteration=2,
            max_iters=5,
            client_flow_id="determinism-1",
        )
        # Final iterations_completed reflects the full max_iters run.
        assert final["iterations_completed"] == 5

    async def test_resume_with_no_prior_checkpoint_starts_fresh(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """resume="replay" on a history with no ref falls back to a fresh run."""
        flow = build_canonical_flow(
            make_test_logger(), max_iters=2, store=store, client_flow_id="never-saved"
        )
        result = await flow.run(resume="replay")
        assert result["iterations_completed"] == 2

    async def test_resume_after_clean_exit_does_not_replay_final_commit(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A completed run stamps a marker so `resume="replay"` after success is a no-op.

        Without the marker, `run(resume="replay")` after a successful run
        resolves to the final iterate commit, fast-forwards iteration to
        `max_iters`, and re-executes any chain steps after the iterate —
        firing their side effects twice.
        """
        from llm_gent.flow import Context, FlowFactory, verb

        tail_calls: list[int] = []

        @verb
        async def bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["counter"] += 1
            return ctx.state.data["counter"]

        @verb
        async def tail(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            tail_calls.append(ctx.state.data["counter"])
            return ctx.state.data["counter"]

        def _flow() -> Any:
            return (
                FlowFactory(make_test_logger())
                .create(state={"counter": 0})
                .with_checkpointer(store, "complete-1")
                .iterate(lambda body: body.call(bump), max_iters=3)
                .call(tail)
            )

        await _flow().run()
        assert tail_calls == [3]
        # Resume after completion — should be a no-op (fresh run since the
        # history is marked complete). Tail runs ONCE more from the
        # fresh state, not twice from the resumed one.
        await _flow().run(resume="replay")
        assert tail_calls == [3, 3], f"tail should have fired only twice total; got {tail_calls}"

    async def test_multiple_resume_boundaries(self, store: JsonFileCheckpointStore) -> None:
        """Interrupt at different iterations, resume each — final state matches uninterrupted."""
        for cut_at in (1, 2, 3, 4):
            cfid = f"multi-cut-{cut_at}"
            final = await assert_resume_determinism(
                make_test_logger(),
                store,
                halt_after_iteration=cut_at,
                max_iters=5,
                client_flow_id=cfid,
            )
            assert final["iterations_completed"] == 5

    async def test_scope_less_iterate_restores_carry_from_cursor(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A scope-less iterate (no state=) restores the carried value from its cursor on resume.

        The cursor is registered at ``n/<node_id>`` under the parent scope, not at
        ``scope_path`` (which equals the parent scope's path when the iterate has no
        scope of its own). Without the cursor-path fix, _restore_carry would look up
        the parent's path and find nothing, causing the carry to reset to the run() arg.
        """
        from llm_gent.flow import Context, FlowFactory, verb

        halt = asyncio.Event()
        calls: list[tuple[int, int]] = []

        @verb
        async def step(ctx: Context[dict[str, Any]], prev: int) -> int:
            calls.append((ctx.state.data.get("iter", 0), prev))
            ctx.state.data["iter"] = ctx.state.data.get("iter", 0) + 1
            if ctx.state.data["iter"] == 2:
                halt.set()
            return prev + 10

        def _flow() -> Any:
            return (
                FlowFactory(make_test_logger())
                .create(state={"iter": 0})
                .with_checkpointer(store, "scopeless-carry")
                .with_halt(halt)
                .iterate(lambda b: b.call(step), max_iters=4)
            )

        await _flow().run(0)
        assert calls == [(0, 0), (1, 10)]

        halt.clear()
        calls.clear()
        result = await _flow().run(resume="replay")
        assert calls == [(2, 20), (3, 30)]
        assert result == 40


# ---------------------------------------------------------------------------
# Resume error paths — no client_flow_id, corruption, structural drift
# ---------------------------------------------------------------------------


class TestResumeErrorPaths:
    async def test_resume_raises_when_commit_object_missing(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Ref points at a commit that isn't stored → HistoryCorrupt, not a fresh run."""
        from llm_gent.flow import HistoryCorrupt

        # Seed HEAD pointing at a non-existent commit hash.
        store.set_ref(flow_id_for(store, "orphan-ref"), HEAD_REF, "0" * 64, None)
        flow = build_canonical_flow(
            make_test_logger(), max_iters=1, store=store, client_flow_id="orphan-ref"
        )
        with pytest.raises(HistoryCorrupt, match="commit"):
            await flow.run(resume="replay")

    async def test_resume_raises_when_tree_object_missing(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Commit points at a tree that isn't stored → HistoryCorrupt, not a fresh run."""
        from llm_gent.flow import HistoryCorrupt
        from llm_gent.flow.state.cas import (
            Commit,
            CommitMeta,
            ProducedBy,
        )

        meta = CommitMeta(
            flow_id=flow_id_for(store, "orphan-tree"),
            node_path="root",
            iteration=0,
            produced_by=ProducedBy(node_id="x", verb_name=None, role=None, result_hash=None),
            trace_ref=(),
            outcome="ok",
            flow_root_hash="orphan-tree",
            timestamp_iso="1970-01-01T00:00:00+00:00",
            framework_version="test",
        )
        commit = Commit.build(root_tree_hash="deadbeef" * 8, parent_hashes=(), meta=meta)
        store.put_object(
            flow_id_for(store, "orphan-tree"), "commit", commit.content_hash, commit.to_bytes()
        )
        store.set_ref(flow_id_for(store, "orphan-tree"), HEAD_REF, commit.content_hash, None)
        flow = build_canonical_flow(
            make_test_logger(), max_iters=1, store=store, client_flow_id="orphan-tree"
        )
        with pytest.raises(HistoryCorrupt, match="tree"):
            await flow.run(resume="replay")

    async def test_resume_raises_when_blob_missing(self, store: JsonFileCheckpointStore) -> None:
        """Tree entry references a blob that isn't stored → HistoryCorrupt, not a fresh run."""
        from llm_gent.flow import HistoryCorrupt
        from llm_gent.flow.state.cas import (
            Commit,
            CommitMeta,
            ProducedBy,
            Tree,
            TreeEntry,
        )

        # Build a tree with an entry pointing at a nonexistent blob.
        entry = TreeEntry(scope_id="state", kind="blob", child_hash="c0ffee" * 10 + "1234")
        tree = Tree.from_entries([entry])
        meta = CommitMeta(
            flow_id=flow_id_for(store, "orphan-blob"),
            node_path="root",
            iteration=0,
            produced_by=ProducedBy(node_id="x", verb_name=None, role=None, result_hash=None),
            trace_ref=(),
            outcome="ok",
            flow_root_hash="orphan-blob",
            timestamp_iso="1970-01-01T00:00:00+00:00",
            framework_version="test",
        )
        commit = Commit.build(root_tree_hash=tree.content_hash, parent_hashes=(), meta=meta)
        store.put_object(
            flow_id_for(store, "orphan-blob"), "tree", tree.content_hash, tree.to_bytes()
        )
        store.put_object(
            flow_id_for(store, "orphan-blob"), "commit", commit.content_hash, commit.to_bytes()
        )
        store.set_ref(flow_id_for(store, "orphan-blob"), HEAD_REF, commit.content_hash, None)
        flow = build_canonical_flow(
            make_test_logger(), max_iters=1, store=store, client_flow_id="orphan-blob"
        )
        with pytest.raises(HistoryCorrupt, match="blob"):
            await flow.run(resume="replay")

    async def test_resume_with_stale_node_path_raises_structural_change(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A commit whose node_path lists ids not present in the current
        composition raises a structural-change error on resume — the
        head-pop pre-scan fails before any node runs.
        """
        from llm_gent.flow.state.cas import (
            Blob,
            Commit,
            CommitMeta,
            ProducedBy,
            Tree,
            TreeEntry,
            canonical_json,
        )

        # Fabricate a commit whose node_path names ids that don't exist in
        # the resume flow's tree.
        stale_path = "cafebabecafebabe/deadbeefdeadbeef"
        blob = Blob.from_bytes(canonical_json({}))
        store.put_object(flow_id_for(store, "stale-1"), "blob", blob.content_hash, blob.payload)
        tree = Tree.from_entries(
            [TreeEntry(scope_id="state", kind="blob", child_hash=blob.content_hash)]
        )
        store.put_object(flow_id_for(store, "stale-1"), "tree", tree.content_hash, tree.to_bytes())
        meta = CommitMeta(
            flow_id=flow_id_for(store, "stale-1"),
            node_path=stale_path,
            iteration=2,
            produced_by=ProducedBy(node_id="x", verb_name=None, role=None, result_hash=None),
            trace_ref=(),
            outcome="ok",
            flow_root_hash="stale-1",
            timestamp_iso="1970-01-01T00:00:00+00:00",
            framework_version="test",
        )
        commit = Commit.build(root_tree_hash=tree.content_hash, parent_hashes=(), meta=meta)
        store.put_object(
            flow_id_for(store, "stale-1"), "commit", commit.content_hash, commit.to_bytes()
        )
        store.set_ref(flow_id_for(store, "stale-1"), HEAD_REF, commit.content_hash, None)
        flow = build_canonical_flow(
            make_test_logger(), max_iters=3, store=store, client_flow_id="stale-1"
        )
        with pytest.raises(RuntimeError) as excinfo:
            await flow.run(resume="replay")
        message = str(excinfo.value)
        assert "structurally changed" in message
        # Both stale ids appear in the triage message (root→leaf full_path).
        assert "cafebabecafebabe" in message
        assert "deadbeefdeadbeef" in message

    @pytest.mark.parametrize(
        "edited",
        [["a", "c", "b"], ["a", "b", "x", "c"], ["a", "b", "c", "x"]],
        ids=["reorder", "insert-before-save-point", "append"],
    )
    async def test_replay_refuses_edited_chain(
        self, store: JsonFileCheckpointStore, edited: list[str]
    ) -> None:
        """Halt in ``[a, b, c]`` saves at c; replay of an edited chain raises before any step.

        The saved step's id survives each edit, so only the structure check
        stops replay from skipping ``x`` or running ``b`` twice.
        """
        from llm_gent.flow import Context, FlowFactory, verb

        halt = asyncio.Event()
        ran: list[str] = []

        @verb
        async def a(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            ran.append("a")

        @verb
        async def b(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            ran.append("b")
            halt.set()

        @verb
        async def c(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            ran.append("c")

        @verb
        async def x(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            ran.append("x")

        steps = {"a": a, "b": b, "c": c, "x": x}

        def build(names: list[str], halt_event: asyncio.Event) -> Any:
            flow = ff.create(state={}).with_checkpointer(store, "edited").with_halt(halt_event)
            flow = flow.call(steps[names[0]])
            for name in names[1:]:
                flow = flow.then(steps[name])
            return flow

        ff = FlowFactory(make_test_logger())
        await build(["a", "b", "c"], halt).run()
        assert ran == ["a", "b"]

        ran.clear()
        with pytest.raises(RuntimeError, match="structurally changed"):
            await build(edited, asyncio.Event()).run(resume="replay")
        assert ran == []


# ---------------------------------------------------------------------------
# Scoped state — .call(state=...) round-trip through the CAS commit tree
# ---------------------------------------------------------------------------


class TestScopedStateRoundTrip:
    async def test_call_scope_state_survives_resume(self, store: JsonFileCheckpointStore) -> None:
        """A ``.call(state=child)`` scope's mutations survive resume.

        Outer flow projects a child scope for its inner subflow; inner
        subflow's iterate mutates the projected scope and halts. On
        resume, the projected scope's payload restores from the CAS
        commit instead of being re-projected fresh — verified by
        inspecting the pre-resume commit's leaf blob and confirming the
        resume completes without a structural-drift error.
        """
        from llm_gent.flow import Context, FlowFactory, History, verb

        @verb
        async def bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["counter"] += 1
            return ctx.state.data["counter"]

        halt = asyncio.Event()

        @verb
        async def maybe_halt(ctx: Context[dict[str, int]], _prev: Any = None) -> Any:
            if ctx.state.data["counter"] >= 2:
                halt.set()
            return _prev

        ff = FlowFactory(make_test_logger())
        inner = ff.create().iterate(lambda body: body.call(bump).then(maybe_halt), max_iters=5)

        outer_pre = (
            ff.create(state={"outer": True})
            .with_checkpointer(store, "scoped-1")
            .with_halt(halt)
            .call(inner, state=lambda _p: {"counter": 0})
        )
        await outer_pre.run()

        # The halt commit's snapshot holds the .call scope with counter=2.
        halted = await History(store, "scoped-1").head()
        assert halted is not None
        snapshot = await History(store, "scoped-1").snapshot(halted)
        assert list(snapshot.scopes.values()) == [{"counter": 2}]
        assert snapshot.chain(halted.meta.scope_path) == [{"counter": 2}]

        # Resume — projected scope restores from the commit instead of
        # re-projecting to counter=0; run reaches max_iters=5 cleanly.
        outer_resume = (
            ff.create(state={"outer": True})
            .with_checkpointer(store, "scoped-1")
            .call(inner, state=lambda _p: {"counter": 0})
        )
        await outer_resume.run(resume="replay")

    async def test_three_level_nested_scopes_survive_resume(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """3-level nested ``.call(state=)`` chain — intermediate scope restores.

        Composition::

            outer(root scope)
              .call(mid_flow, state=lambda p: {...})    # depth 1 — middle scope
                mid_flow.iterate(body, ...)              # depth 2 — leaf iterate

        The commit's snapshot holds both child scopes, each at its own
        path. Replay has to restore the middle one too, not re-project it
        through the state factory.

        This test halts mid-run, resumes, and asserts a mutation stored
        in the middle scope during the pre-halt run persists — proving
        the intermediate blob is restored rather than re-projected.
        """
        from llm_gent.flow import Context, FlowFactory, verb

        # A witness marker written into the middle scope in the pre-halt
        # run; the resume run must observe the same value in state.data.
        # If the middle scope is re-projected fresh, this key is missing.
        WITNESS_KEY = "mid_scope_witness"

        @verb
        async def mark_middle(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            # Reach the middle scope (parent of the iterate body scope)
            # via the State._parent chain and mutate WITNESS_KEY on it.
            middle = ctx.state._parent
            assert middle is not None
            middle.data[WITNESS_KEY] = middle.data.get(WITNESS_KEY, 0) + 1

        halt = asyncio.Event()

        @verb
        async def maybe_halt(ctx: Context[dict[str, Any]], _prev: Any = None) -> Any:
            middle = ctx.state._parent
            if middle is not None and middle.data.get(WITNESS_KEY, 0) >= 2:
                halt.set()
            return _prev

        ff_inner = FlowFactory(make_test_logger())
        # Iterate has its own state=lambda so its body runs in a NEW scope
        # (leaf, depth 2). The middle .call scope sits between root and
        # the iterate body.
        mid_flow = ff_inner.create().iterate(
            lambda body: body.call(mark_middle).then(maybe_halt),
            state=lambda _p: {"leaf_iter": 0},
            max_iters=5,
        )

        ff = FlowFactory(make_test_logger())
        outer_pre = (
            ff.create(state={"outer": True})
            .with_checkpointer(store, "3-level-1")
            .with_checkpoint_policy(on_iterate=True)
            .with_halt(halt)
            .call(mid_flow, state=lambda _p: {})
        )
        await outer_pre.run()
        assert store.get_ref(flow_id_for(store, "3-level-1"), HEAD_REF) is not None

        # The head's snapshot holds the root, the middle (.call) scope and
        # the leaf (.iterate) scope; the middle one carries the witness.
        from llm_gent.flow import History

        history = History(store, "3-level-1")
        commit = await history.head()
        assert commit is not None
        snapshot = await history.snapshot(commit)
        middle_path, leaf_path = sorted(snapshot.scopes, key=len)
        assert leaf_path.startswith(middle_path + "/")
        middle_data = snapshot.scopes[middle_path]
        assert middle_data.get(WITNESS_KEY) == 2, (
            f"middle scope should carry the WITNESS mutation; got {middle_data!r}"
        )

        # Resume without halt — runs to max_iters=5, so 3 more iterations
        # under the restored middle scope. If _hydrate_resume_state
        # restores the middle scope's blob, WITNESS_KEY starts at 2 and
        # bumps to 5. If the middle scope re-projects from scratch (the
        # pre-fix behavior), WITNESS_KEY starts at 0 and only reaches 3.
        outer_resume = (
            ff.create(state={"outer": True})
            .with_checkpointer(store, "3-level-1")
            .with_checkpoint_policy(on_iterate=True)
            .call(mid_flow, state=lambda _p: {})
        )
        await outer_resume.run(resume="replay")
        # Inspect the newest commit under the iterate's node_path (the
        # final-state commit at "$end", written on clean exit, sits after it).
        final_commit = None
        async for candidate in history.commits():
            if candidate.meta.node_path == commit.meta.node_path:
                final_commit = candidate
                break
        assert final_commit is not None
        final_middle = (await history.snapshot(final_commit)).scopes[middle_path]
        assert final_middle.get(WITNESS_KEY) == 5, (
            f"resume should have restored the middle scope (WITNESS=2) and "
            f"continued for 3 more iterations to WITNESS=5; got {final_middle!r} "
            f"— re-projected middle would land at 3 instead"
        )

    async def test_leaf_iterate_with_state_survives_resume(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A leaf iterate with its own ``state=`` projection restores on resume.

        Regression test for the case where the leaf iterate (not just its
        parent .call) has a ``state=`` projection. The iterate's scope must
        be placed in ``child_state_data`` by ``_split_scopes`` so that
        ``_resume_iteration`` returns it correctly.
        """
        from llm_gent.flow import Context, FlowFactory, History, verb

        @verb
        async def bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["counter"] = ctx.state.data.get("counter", 0) + 1
            return ctx.state.data["counter"]

        halt = asyncio.Event()

        @verb
        async def maybe_halt(ctx: Context[dict[str, int]], _prev: Any = None) -> Any:
            if ctx.state.data["counter"] >= 2:
                halt.set()
            return _prev

        ff = FlowFactory(make_test_logger())

        # Leaf iterate with its own state= projection (not inherited from .call)
        flow_pre = (
            ff.create(state={"root": True})
            .with_checkpointer(store, "leaf-iterate-state")
            .with_halt(halt)
            .iterate(
                lambda body: body.call(bump).then(maybe_halt),
                max_iters=5,
                state=lambda _p: {"counter": 0},  # leaf iterate's own scope
            )
        )
        await flow_pre.run()

        # Verify the halt commit's snapshot has the iterate's scope
        history = History(store, "leaf-iterate-state")
        halted = await history.head()
        assert halted is not None
        snapshot = await history.snapshot(halted)
        assert list(snapshot.scopes.values()) == [{"counter": 2}]

        # Resume — the iterate's scope should restore from the commit,
        # not re-project to counter=0
        halt.clear()
        flow_resume = (
            ff.create(state={"root": True})
            .with_checkpointer(store, "leaf-iterate-state")
            .iterate(
                lambda body: body.call(bump).then(maybe_halt),
                max_iters=5,
                state=lambda _p: {"counter": 0},
            )
        )
        await flow_resume.run(resume="replay")

        # Final commit should exist (run completed without structural-drift error)
        final = await history.head()
        assert final is not None and History.is_final_state(final)


# ---------------------------------------------------------------------------
# Async-store round-trip — sync-or-async Protocol contract
# ---------------------------------------------------------------------------


class TestAsyncStore:
    async def test_async_wrapper_round_trips_via_flow_run(self, tmp_path: Path) -> None:
        """A store with async methods (declared via ``async def``) works
        end-to-end through Flow.run — the framework awaits every store
        call via :func:`maybe_await`.
        """
        import inspect

        from llm_gent.flow.stores import JsonFileCheckpointStore as _Sync

        # Wrap the sync store's methods in async ones so the executor
        # goes through maybe_await on every call.
        class _AsyncWrap:
            def __init__(self, inner: _Sync) -> None:
                self._inner = inner

            @property
            def retention(self) -> str:
                return self._inner.retention

            async def get_flow_id(self, *a: Any, **kw: Any) -> str | None:
                await asyncio.sleep(0)
                return self._inner.get_flow_id(*a, **kw)

            async def bind_flow_id(self, *a: Any, **kw: Any) -> str:
                await asyncio.sleep(0)
                return self._inner.bind_flow_id(*a, **kw)

            async def put_object(self, *a: Any, **kw: Any) -> None:
                await asyncio.sleep(0)
                self._inner.put_object(*a, **kw)

            async def get_object(self, *a: Any, **kw: Any) -> bytes | None:
                await asyncio.sleep(0)
                return self._inner.get_object(*a, **kw)

            async def has_object(self, *a: Any, **kw: Any) -> bool:
                await asyncio.sleep(0)
                return self._inner.has_object(*a, **kw)

            async def get_ref(self, *a: Any, **kw: Any) -> str | None:
                await asyncio.sleep(0)
                return self._inner.get_ref(*a, **kw)

            async def set_ref(self, *a: Any, **kw: Any) -> bool:
                await asyncio.sleep(0)
                return self._inner.set_ref(*a, **kw)

            async def gc_history(self, *a: Any, **kw: Any) -> None:
                await asyncio.sleep(0)
                self._inner.gc_history(*a, **kw)

        inner = _Sync(make_test_logger(), tmp_path / "cp")
        wrap = _AsyncWrap(inner)
        # Sanity: every method IS async.
        for m in (
            "get_flow_id",
            "bind_flow_id",
            "put_object",
            "get_object",
            "get_ref",
            "set_ref",
            "gc_history",
        ):
            assert inspect.iscoroutinefunction(getattr(wrap, m))

        # Round-trip via the canonical flow: fresh save + resume.
        halt = asyncio.Event()
        await build_canonical_flow(
            make_test_logger(),
            max_iters=5,
            halt=halt,
            halt_after_iteration=2,
            store=wrap,
            client_flow_id="async-round-trip",
        ).run()
        resumed = await build_canonical_flow(
            make_test_logger(),
            max_iters=5,
            store=wrap,
            client_flow_id="async-round-trip",
        ).run(resume="replay")
        assert resumed["iterations_completed"] == 5
