# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""End-to-end Flow-level checkpoint/resume smoke tests.

Runs the canonical multi-stage flow (add_one → times_two → minus_three,
iterated) through a real :class:`FileStore` and asserts:

- fresh runs save at iterate boundaries;
- resume from a mid-run halt reaches the same final state as an
  uninterrupted run (determinism);
- the retention policy toggles gc-on-clean-exit;
- halt-triggered exit always preserves the history.

Store tests (objects, refs, prune) are :mod:`appinfra.cas`'s conformance
suite; gent's Postgres store runs it in
:mod:`tests.integration.flow.test_stores_pg`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from appinfra.cas import FileStore

from llm_gent.flow import FlowCommit
from llm_gent.flow.checkpoint import HEAD_REF
from llm_gent.flow.testing.checkpoint import (
    assert_resume_determinism,
    build_canonical_flow,
)
from tests.helpers import file_store

from .conftest import commit_ids, head_commit, make_test_logger, read_commit


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
class Target:
    """A map item, like an app's fan-out target."""

    qid: int


class AppHandle:
    """An app object with no JSON form (e.g. an ORM row), passed between steps."""

    def __init__(self, qid: int) -> None:
        self.qid = qid


def fail_first_commit(store: FileStore) -> None:
    """Make the store's first commit write raise ``OSError("disk full")``."""
    original = store.write_object
    failed: list[bool] = []

    def flaky(repo: str, object_id: str, object_type: Any, data: bytes) -> None:
        if object_type == "commit" and not failed:
            failed.append(True)
            raise OSError("disk full")
        original(repo, object_id, object_type, data)

    store.write_object = flaky  # type: ignore[method-assign]


def stored_commits(store: FileStore, client_flow_id: str) -> list[FlowCommit]:
    """Every commit the history's repository holds, in no particular order."""
    return [read_commit(store, client_flow_id, cid) for cid in commit_ids(store, client_flow_id)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> FileStore:
    """A fresh store under an isolated temp root, default retention."""
    return file_store(tmp_path / "cp")


# ---------------------------------------------------------------------------
# Fresh run — saves happen at iterate boundaries
# ---------------------------------------------------------------------------


class TestFreshRunSaves:
    async def test_saves_a_commit_per_iteration(self, store: FileStore) -> None:
        """Each successful iterate iteration writes a commit + ref."""
        await build_canonical_flow(
            make_test_logger(), max_iters=3, store=store, client_flow_id="freshrun"
        ).run()
        # A resolvable ref exists — the history reached at least one commit.
        assert store.repo("freshrun").get_ref(HEAD_REF) is not None

    async def test_default_retention_keeps_history_on_success(self, store: FileStore) -> None:
        """retention="retain" (default): clean-exit does NOT delete the history."""
        await build_canonical_flow(
            make_test_logger(), max_iters=2, store=store, client_flow_id="retain-1"
        ).run()
        # Successful run — but retention="retain" so the ref survives.
        assert store.repo("retain-1").get_ref(HEAD_REF) is not None


# ---------------------------------------------------------------------------
# Retention policy
# ---------------------------------------------------------------------------


class TestRetention:
    async def test_gc_on_success_deletes_the_history(self, store: FileStore) -> None:
        """retention="gc_on_success": clean-exit deletes the history's repository."""
        await build_canonical_flow(
            make_test_logger(),
            max_iters=2,
            store=store,
            client_flow_id="gc-1",
            retention="gc_on_success",
        ).run()
        assert "gc-1" not in store.list_repos()

    async def test_halt_preserves_history_regardless_of_retention(self, store: FileStore) -> None:
        """A halt-triggered exit preserves the history even under gc_on_success —
        the framework only deletes it on fully successful runs.
        """
        halt = asyncio.Event()
        await build_canonical_flow(
            make_test_logger(),
            max_iters=5,
            halt=halt,
            halt_after_iteration=2,
            store=store,
            client_flow_id="halt-preserve",
            retention="gc_on_success",
        ).run()
        assert store.repo("halt-preserve").get_ref(HEAD_REF) is not None


# ---------------------------------------------------------------------------
# Save on halt — halt-observation sites emit a commit at the halt position
# ---------------------------------------------------------------------------


class TestCheckpointPolicyIterate:
    """CheckpointPolicy.on_iterate gates the iterate-boundary auto-save."""

    async def test_default_policy_writes_only_the_final_state_commit(
        self, store: FileStore
    ) -> None:
        """Default CheckpointPolicy (halt-only) skips per-iteration saves.

        Under the default policy an iterate that runs cleanly to
        ``max_iters`` should leave the store with exactly one commit:
        the ``$end`` final-state commit written by the clean-exit
        retention path. No iterate-boundary commit should exist.
        """
        from llm_gent.flow import Context, Factory, verb

        @verb
        async def bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["n"] = ctx.state.data.get("n", 0) + 1
            return ctx.state.data["n"]

        body = Factory(make_test_logger()).create()
        body.call(bump)

        outer = (
            Factory(make_test_logger())
            .create(state={"n": 0})
            .with_checkpoint_store(store, "policy-halt-only")
            .with_checkpointer()
            .iterate(body, max_iters=5)
        )
        await outer.run()

        # Every commit written during the run has node_path == "$end"
        # (only the clean-exit final-state commit fires under the default
        # policy). Walking the stored commits reveals no iterate-boundary
        # commits.
        node_paths = {c.meta.node_path for c in stored_commits(store, "policy-halt-only")}
        assert node_paths == {"$end"}, (
            f"only the $end final-state commit should have been written; got {node_paths}"
        )

    async def test_on_iterate_true_writes_per_iteration_commits(self, store: FileStore) -> None:
        """CheckpointPolicy(on_iterate=True) restores per-iteration saves.

        Opting in should produce an iterate-boundary ref alongside the
        final-state commit — the iterate's own node_path directory carries
        one ref per iteration.
        """
        from llm_gent.flow import Context, Factory, verb

        @verb
        async def bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["n"] = ctx.state.data.get("n", 0) + 1
            return ctx.state.data["n"]

        body = Factory(make_test_logger()).create()
        body.call(bump)

        outer = (
            Factory(make_test_logger())
            .create(state={"n": 0})
            .with_checkpoint_store(store, "policy-on-iter")
            .with_checkpointer()
            .with_checkpoint_policy(on_iterate=True)
            .iterate(body, max_iters=3)
        )
        await outer.run()

        # Under on_iterate=True the iterate boundary emits a commit each
        # iteration + a $end final-state commit on clean exit. Group commit
        # objects by node_path: 3 boundary commits share the iterate's
        # node_path, plus one $end commit.
        by_node_path: dict[str, int] = {}
        for commit in stored_commits(store, "policy-on-iter"):
            by_node_path[commit.meta.node_path] = by_node_path.get(commit.meta.node_path, 0) + 1
        assert by_node_path["$end"] == 1
        iterate_paths = {p: n for p, n in by_node_path.items() if p != "$end"}
        assert len(iterate_paths) == 1, f"expected one iterate node_path; got {list(iterate_paths)}"
        # 3 iterate-boundary commits (one per iteration).
        assert next(iter(iterate_paths.values())) == 3


class TestCheckpointPolicyMap:
    """CheckpointPolicy.on_map_item gates the map per-item auto-save."""

    async def test_default_policy_writes_no_map_item_commits(self, store: FileStore) -> None:
        """Under the default policy a completed map writes only the $end final-state commit."""
        from llm_gent.flow import Context, Factory, verb

        @verb
        async def touch(ctx: Context[dict[str, Any]], item: int) -> int:
            return item * 2

        body = Factory(make_test_logger()).create()
        body.call(touch)

        outer = (
            Factory(make_test_logger())
            .create(state={})
            .with_checkpoint_store(store, "map-default")
            .with_checkpointer()
            .map(body, items=lambda _p, _c: [1, 2, 3])
        )
        await outer.run()

        node_paths = {c.meta.node_path for c in stored_commits(store, "map-default")}
        assert node_paths == {"$end"}, (
            f"only the $end commit should exist under default policy; got {node_paths}"
        )

    async def test_on_map_item_true_writes_per_item_commits(self, store: FileStore) -> None:
        """CheckpointPolicy(on_map_item=True) saves after each successful item.

        Runs a 3-item map and asserts that three iteration-indexed
        commits exist under the map's node_path plus the $end
        final-state commit. Item order across saves is not asserted
        (concurrent).
        """
        from llm_gent.flow import Context, Factory, verb

        @verb
        async def touch(ctx: Context[dict[str, Any]], item: int) -> int:
            return item * 2

        body = Factory(make_test_logger()).create()
        body.call(touch)

        outer = (
            Factory(make_test_logger())
            .create(state={})
            .with_checkpoint_store(store, "map-on-item")
            .with_checkpointer()
            .with_checkpoint_policy(on_map_item=True)
            .map(body, items=lambda _p, _c: [1, 2, 3])
        )
        await outer.run()

        iterations_at_map_path: dict[str, list[int]] = {}
        for commit in stored_commits(store, "map-on-item"):
            iterations_at_map_path.setdefault(commit.meta.node_path, []).append(
                commit.meta.iteration
            )
        assert "$end" in iterations_at_map_path
        non_final = {p: v for p, v in iterations_at_map_path.items() if p != "$end"}
        assert len(non_final) == 1, f"expected one map node_path; got {list(non_final)}"
        iterations = sorted(next(iter(non_final.values())))
        assert iterations == [0, 1, 2], f"expected three item slots 0/1/2; got {iterations}"

    async def test_failed_item_commit_restores_the_state_before_the_merge(
        self, store: FileStore
    ) -> None:
        """A per-item commit that fails rolls the merge back to the prior state, not to empty."""
        from llm_gent.flow import Context, Factory, verb

        fail_first_commit(store)

        @verb
        async def add(ctx: Context[dict[str, Any]], item: int) -> int:
            ctx.state.data["n"] += item
            return item

        @verb
        async def read_state(ctx: Context[dict[str, Any]]) -> dict[str, Any]:
            return dict(ctx.state.data)

        outer = (
            Factory(make_test_logger())
            .create(state={"total": 1})
            .with_checkpoint_store(store, "map-rollback")
            .with_checkpointer()
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

        Item 0's commit fails while item 1 has not finished writing.
        Restoring a pre-merge snapshot would drop item 1's write.
        """
        from llm_gent.flow import Context, Factory, History, verb

        store = file_store(tmp_path / "cp")
        fail_first_commit(store)

        @verb
        async def mark(ctx: Context[dict[str, Any]], x: int) -> int:
            if x == 1:
                for _ in range(3):
                    await asyncio.sleep(0)
            ctx.state.data[f"s{x}"] = True
            return x

        state: dict[str, Any] = {}
        outer = (
            Factory(make_test_logger())
            .create(state=state)
            .with_checkpoint_store(store, "map-siblings")
            .with_checkpointer()
            .with_checkpoint_policy(on_map_item=True)
            .map(lambda b: b.call(mark), items=lambda _p, _c: [0, 1], strict=False)
        )
        await outer.run()
        assert state == {"s0": True, "s1": True}
        history = History(store, "map-siblings")
        head = await history.head()
        assert head is not None
        assert (await history.snapshot(head)).root == {"s0": True, "s1": True}


class TestCommitConsistency:
    async def test_every_scope_in_a_commit_comes_from_one_moment(self, tmp_path: Path) -> None:
        """A commit's scopes all come from one moment.

        Each item bumps a counter in the root and in the subflow's scope in
        one step, so the two always agree. Items interleave between their
        steps; a commit that took one scope before another item's step and
        one after would hold counters that disagree.
        """
        from llm_gent.flow import Context, Factory, History, verb

        store = file_store(tmp_path / "cp")

        @verb
        async def bump(ctx: Context[dict[str, Any]], _item: int) -> None:
            await asyncio.sleep(0)
            ctx.state.data["n"] += 1
            ctx.state.root().data["n"] += 1
            await ctx.checkpoint()

        sub = Factory(make_test_logger()).create()
        sub.map(lambda b: b.call(bump), items=lambda _p, _c: list(range(6)))
        await (
            Factory(make_test_logger())
            .create(state={"n": 0})
            .with_checkpoint_store(store, "one-moment")
            .with_checkpointer()
            .call(sub, state=lambda p: {"n": p["n"]})
            .run()
        )
        history = History(store, "one-moment")
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

    async def test_app_object_passed_between_steps(self, store: FileStore) -> None:
        from llm_gent.flow import Context, Factory, verb

        @verb
        async def make(ctx: Context[dict[str, Any]]) -> AppHandle:
            return AppHandle(7)

        @verb
        async def read(ctx: Context[dict[str, Any]], handle: AppHandle) -> int:
            return handle.qid

        flow = (
            Factory(make_test_logger())
            .create(state={})
            .with_checkpoint_store(store, "app-object")
            .with_checkpointer()
            .call(make)
            .call(read)
        )
        assert await flow.run() == 7

    async def test_non_strict_guarded_map_of_app_objects(self, store: FileStore) -> None:
        """Items that fail, are skipped by the guard, or return app objects all reach the next step."""
        from llm_gent.flow import Context, Factory, verb

        @verb
        async def dispatch(ctx: Context[dict[str, Any]], t: Target) -> AppHandle:
            if t.qid == 2:
                raise RuntimeError("inner run failed")
            return AppHandle(t.qid)

        @verb
        async def kinds(ctx: Context[dict[str, Any]], outcomes: list[Any]) -> list[str]:
            return [type(o).__name__ for o in outcomes]

        flow = (
            Factory(make_test_logger())
            .create(state={})
            .with_checkpoint_store(store, "targets")
            .with_checkpointer()
            .map(
                lambda b: b.call(dispatch),
                items=lambda _p, _c: [Target(1), Target(2), Target(3)],
                strict=False,
            )
            .guard(lambda t, _c: t.qid != 3)
            .call(kinds)
        )
        assert await flow.run() == ["AppHandle", "Failure", "Skipped"]

    async def test_non_strict_map_with_a_failed_item(self, store: FileStore) -> None:
        from llm_gent.flow import Context, Factory, verb

        @verb
        async def double(ctx: Context[dict[str, Any]], x: int) -> int:
            if x == 2:
                raise RuntimeError("item failed")
            return 2 * x

        @verb
        async def summarize(ctx: Context[dict[str, Any]], results: list[Any]) -> list[Any]:
            return [r if isinstance(r, int) else "failed" for r in results]

        flow = (
            Factory(make_test_logger())
            .create(state={})
            .with_checkpoint_store(store, "failed-item")
            .with_checkpointer()
            .map(lambda b: b.call(double), items=lambda _p, _c: [1, 2, 3], strict=False)
            .call(summarize)
        )
        assert await flow.run() == [2, "failed", 6]


class TestCtxCheckpoint:
    """Explicit ctx.checkpoint() writes a commit regardless of policy."""

    async def test_ctx_checkpoint_from_chain_step(self, store: FileStore) -> None:
        """A verb calling ``ctx.checkpoint()`` writes a commit at that step's node.

        Runs under the default (halt-only) policy so no iterate-
        boundary save fires; the only commit besides the final-state one
        that exists is the one the verb explicitly requested.
        """
        from llm_gent.flow import Context, Factory, verb

        @verb
        async def saver(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["n"] = 42
            await ctx.checkpoint()
            return 42

        outer = (
            Factory(make_test_logger())
            .create(state={})
            .with_checkpoint_store(store, "ctx-ckpt")
            .with_checkpointer()
            .call(saver)
        )
        await outer.run()

        # Two commits total: one from the explicit ctx.checkpoint() (verb's
        # chain-step node) + one $end final-state commit on clean exit.
        node_paths = [c.meta.node_path for c in stored_commits(store, "ctx-ckpt")]
        assert "$end" in node_paths
        non_final = [p for p in node_paths if p != "$end"]
        assert len(non_final) == 1, (
            f"expected exactly one explicit-checkpoint commit; got {non_final}"
        )

    async def test_unchanged_state_is_not_put_again(self, store: FileStore) -> None:
        """Two checkpoints of the same state put its blob and tree once; only the commits differ."""
        from llm_gent.flow import Context, Factory, verb

        puts: list[str] = []
        original = store.write_object

        def counting(repo: str, object_id: str, object_type: Any, data: bytes) -> None:
            puts.append(object_id)
            original(repo, object_id, object_type, data)

        store.write_object = counting  # type: ignore[method-assign]

        @verb
        async def save_twice(ctx: Context[dict[str, Any]]) -> None:
            await ctx.checkpoint()
            await ctx.checkpoint()

        await (
            Factory(make_test_logger())
            .create(state={"n": 1})
            .with_checkpoint_store(store, "no-reput")
            .with_checkpointer()
            .call(save_twice)
            .run()
        )
        assert len(puts) == len(set(puts))

    async def test_ctx_checkpoint_noop_without_checkpointer(self) -> None:
        """``ctx.checkpoint()`` under a flow with no checkpointer is a no-op."""
        from llm_gent.flow import Context, Factory, verb

        called = 0

        @verb
        async def saver(ctx: Context) -> None:
            nonlocal called
            called += 1
            await ctx.checkpoint()  # no checkpointer wired — must not raise

        flow = Factory(make_test_logger()).create().call(saver)
        await flow.run()
        assert called == 1


class TestPausedTurnInSnapshot:
    """A Loop's paused turn lands in the halt commit's snapshot at the call's path."""

    async def test_halt_commit_holds_paused_turn_at_the_call_path(self, store: FileStore) -> None:
        """Loop paused mid-turn → the snapshot holds its task + conversation at ``<step>/t/0``."""
        from dataclasses import dataclass, field

        from llm_gent.flow import Context, Factory, History, Loop, Role, verb

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

        ff = Factory(make_test_logger(), saia_factory=_PausingSAIAFactory())
        flow = (
            ff.create(state={})
            .with_checkpoint_store(store, "paused-turn-1")
            .with_checkpointer()
            .with_halt(halt)
            .call(run_loop)
            .then(after)
        )
        await flow.run()

        commit = head_commit(store, "paused-turn-1")
        assert commit.meta.outcome == "halted"
        assert commit.meta.trace_ref == ()
        snapshot = await History(store, "paused-turn-1").snapshot(commit)
        turns = {p: c["turn"] for p, c in snapshot.cursors.items() if "turn" in c}
        [(path, turn)] = turns.items()
        assert path.endswith("/t/0")
        nothing_used = {"iterations": 0, "total_tokens": 0}
        assert turn == {"task": "t", "conversation": conv.to_dict(), "used": nothing_used}

    async def test_turn_paused_before_the_run_halt_is_not_saved(self, store: FileStore) -> None:
        """Loop A pauses with the run's halt unset: its step completed, its turn is dropped.

        Loop B then sets the halt and completes, so the halt commit's
        cursor is at the step after B and holds no paused turn.
        """
        from dataclasses import dataclass, field

        from llm_gent.flow import Context, Factory, History, Loop, Role, verb

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

        ff = Factory(make_test_logger())
        flow = (
            ff.create(state={})
            .with_checkpoint_store(store, "paused-turn-multi")
            .with_checkpointer()
            .with_halt(halt)
            .call(run_a)
            .then(run_b)
            .then(after)
        )
        await flow.run()

        commit = head_commit(store, "paused-turn-multi")
        assert commit.meta.outcome == "halted"
        snapshot = await History(store, "paused-turn-multi").snapshot(commit)
        assert not any("turn" in c for c in snapshot.cursors.values())

    async def test_resume_round_trip_hands_reconstructed_conv_and_resume_true(
        self, store: FileStore
    ) -> None:
        """Halt mid-Loop-turn on run 1 → resume dispatches SAIA with resume=True + rebuilt conv.

        The paused Loop's step is interrupted, so the halt commit keeps the
        iterate in pass 0 and the chain at that step: resume re-dispatches
        it with the paused turn, then runs pass 1 fresh.
        """
        from dataclasses import dataclass, field

        from llm_gent.flow import Context, Factory, Loop, Role, verb

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
        body_ff = Factory(make_test_logger())
        body = body_ff.create()
        body.call(run_loop)

        # ---- Run 1: iterate iteration 0 halts mid-Loop-turn.
        halt1 = asyncio.Event()
        ff1 = Factory(make_test_logger(), saia_factory=_PhaseFactory(halt1, "first"))
        flow1 = (
            ff1.create(state={})
            .with_checkpoint_store(store, "resume-round-trip")
            .with_checkpointer()
            .with_halt(halt1)
            .iterate(body, max_iters=2)
        )
        await flow1.run(extra={"conv": _Conv(messages=["from-turn-1"])})
        assert [c["phase"] for c in complete_calls] == ["first"]

        # ---- Run 2: resume="latest" re-runs iteration 0. The Loop's re-dispatch
        # picks up the paused_turn entry and hands SAIA the reconstructed
        # conversation with resume=True. The caller supplies a different
        # conversation — the resume path must override it with the rebuilt
        # one from the halted commit.
        halt2 = asyncio.Event()
        ff2 = Factory(make_test_logger(), saia_factory=_PhaseFactory(halt2, "resume"))
        flow2 = (
            ff2.create(state={})
            .with_checkpoint_store(store, "resume-round-trip")
            .with_checkpointer()
            .with_halt(halt2)
            .iterate(body, max_iters=2)
        )
        await flow2.run(
            resume="latest",
            extra={"conv": _Conv(messages=["caller-supplied-but-overridden"])},
        )

        resume_calls = [c for c in complete_calls if c["phase"] == "resume"]
        # Pass 0 resumes the paused turn; pass 1 (max_iters=2) dispatches fresh.
        assert [c["resume"] for c in resume_calls] == [True, False]
        resumed = resume_calls[0]
        assert isinstance(resumed["conversation"], _Conv)
        assert resumed["conversation"].messages == ["from-turn-1"]

    async def test_chain_resume_re_dispatches_paused_loop_verb(self, store: FileStore) -> None:
        """Chain flow with Loop verb: halt on turn 1 → resume re-runs the loop verb."""
        from dataclasses import dataclass, field

        from llm_gent.flow import Context, Factory, Loop, Role, verb

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
        ff1 = Factory(make_test_logger(), saia_factory=_PhaseFactory(halt1, "first"))
        flow1 = (
            ff1.create(state={})
            .with_checkpoint_store(store, "chain-resume")
            .with_checkpointer()
            .with_halt(halt1)
            .call(run_loop)
            .then(after_step)
        )
        await flow1.run(extra={"task": "saved-task", "conv": _Conv(messages=["turn-1"])})
        assert after_calls == []  # halt observed before after_step ran

        # Run 2: resume. Loop is re-dispatched, consumes paused_turn, SAIA
        # completes non-paused, chain moves on to after_step.
        halt2 = asyncio.Event()
        ff2 = Factory(make_test_logger(), saia_factory=_PhaseFactory(halt2, "resume"))
        flow2 = (
            ff2.create(state={})
            .with_checkpoint_store(store, "chain-resume")
            .with_checkpointer()
            .with_halt(halt2)
            .call(run_loop)
            .then(after_step)
        )
        result = await flow2.run(
            resume="latest",
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
        """The Loop call rebuilds task and conversation from the turn saved at its path."""
        from dataclasses import dataclass, field
        from types import SimpleNamespace

        from llm_gent.flow import Loop, Role
        from llm_gent.flow.loop import _LoopTurn
        from llm_gent.flow.state import State
        from llm_gent.flow.state.snapshot import TURN, ScopeRegistry, Snapshot

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

        # A checked-out snapshot whose turn at the step's first Loop call
        # carries BOTH the task and the conversation state.
        saved = {"task": "saved-task-string", "conversation": {"messages": ["mid-turn"]}}
        scopes = ScopeRegistry()
        scopes.begin(
            State(data={}),
            Snapshot(has_state=True, root={}, cursors={f"n/{node_id}/t/0": {TURN: saved}}),
        )
        env = SimpleNamespace(scopes=scopes, owner_path=lambda nid: ("n", nid))
        ctx = SimpleNamespace(_env=env, _node_id=node_id)

        turn = _LoopTurn(ctx)  # type: ignore[arg-type]
        task, conversation, is_resume = loop._consume_saved_turn(ctx, turn)  # type: ignore[arg-type]
        assert is_resume is True
        # A direct Loop chain step resumes with no task of its own: the saved
        # task is what reaches SAIA.
        assert task == "saved-task-string"
        assert isinstance(conversation, _Conv)
        assert conversation.messages == ["mid-turn"]
        # The resuming turn stays in the run's snapshots until SAIA finishes it;
        # a saved turn without "used" counts as nothing used.
        nothing_used = {"iterations": 0, "total_tokens": 0}
        assert scopes.capture()[("n", node_id, "t", "0", TURN)] == {**saved, "used": nothing_used}


class TestSaveOnHaltChain:
    """Chain-only halt-observation site writes a commit at the not-yet-run step."""

    async def test_chain_only_halt_saves_commit_and_resume_lands_on_halted_step(
        self, store: FileStore
    ) -> None:
        """Chain flow with no iterate: halt set by step b → commit at c; resume lands at c."""
        from llm_gent.flow import Context, Factory, verb

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

        ff = Factory(make_test_logger())
        pre = (
            ff.create(state={})
            .with_checkpoint_store(store, "chain-halt")
            .with_checkpointer()
            .with_halt(halt)
            .call(step_a)
            .then(step_b)
            .then(step_c)
        )
        await pre.run()

        commit = head_commit(store, "chain-halt")
        assert commit.meta.outcome == "halted"
        assert commit.meta.iteration == 0
        # node_path's last segment is the not-yet-run step's chain id — step_c's.
        # Independent of the exact id, the run must not have executed step_c yet:
        # the pre-halt state has only a and b, no c.

        halt.clear()
        resume = (
            ff.create(state={})
            .with_checkpoint_store(store, "chain-halt")
            .with_checkpointer()
            .call(step_a)
            .then(step_b)
            .then(step_c)
        )
        result = await resume.run(resume="latest")
        # step_c ran on resume; a and b restored from the halt commit.
        assert result == 3

    async def test_halt_without_checkpointer_is_noop(self) -> None:
        """No checkpointer bound → halt observed → no store activity."""
        from llm_gent.flow import Context, Factory, verb

        halt = asyncio.Event()

        @verb
        async def a(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            halt.set()

        @verb
        async def b(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            pass

        ff = Factory(make_test_logger())
        flow = ff.create(state={}).with_halt(halt).call(a).then(b)
        # No .with_checkpointer — halt check fires between chain steps but
        # _save_halt_checkpoint short-circuits at env.checkpoint_ctx is None.
        # Run should complete without raising.
        await flow.run()

    async def test_halt_before_branch_with_plain_verb_resumes_at_branch(
        self, store: FileStore
    ) -> None:
        """Halt before a Branch containing only plain verbs → resume runs the branch."""
        from llm_gent.flow import Context, Factory, verb

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

        ff = Factory(make_test_logger())
        pre = (
            ff.create(state={})
            .with_checkpoint_store(store, "branch-halt")
            .with_checkpointer()
            .with_halt(halt)
            .call(step_a)
            .branch(when=lambda _p, _c: True, then=lambda f: f.call(branch_verb))
            .then(step_c)
        )
        await pre.run()

        # Halt fired after step_a, commit saved at the branch position.
        commit = head_commit(store, "branch-halt")
        assert commit.meta.outcome == "halted"

        # Resume: branch runs (with its plain verb), then step_c.
        halt.clear()
        resume = (
            ff.create(state={})
            .with_checkpoint_store(store, "branch-halt")
            .with_checkpointer()
            .call(step_a)
            .branch(when=lambda _p, _c: True, then=lambda f: f.call(branch_verb))
            .then(step_c)
        )
        result = await resume.run(resume="latest")
        # 1 (from restored a) + 10 (branch_verb) + 100 (step_c) = 111
        assert result == 111

    async def test_pre_set_halt_with_checkpointer_resumes_at_second_step(
        self, store: FileStore
    ) -> None:
        """Pre-set halt + checkpointer: checkpoint saved before second step; resume runs it."""
        from llm_gent.flow import Context, Factory, verb

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

        ff = Factory(make_test_logger())
        pre = (
            ff.create(state={})
            .with_checkpoint_store(store, "pre-set-halt")
            .with_checkpointer()
            .with_halt(halt)
            .call(step_a)
            .then(step_b)
        )
        await pre.run([1, 2, 3])

        # Checkpoint saved at step_b (halt observed after step_a)
        commit = head_commit(store, "pre-set-halt")
        assert commit.meta.outcome == "halted"

        # Resume with halt cleared — step_b runs, reads items from state
        halt.clear()
        resume = (
            ff.create(state={})
            .with_checkpoint_store(store, "pre-set-halt")
            .with_checkpointer()
            .call(step_a)
            .then(step_b)
        )
        result = await resume.run(resume="latest")
        assert result == 6  # sum([1, 2, 3])

    async def test_nested_flow_halt_does_not_clobber_outer_checkpoint(
        self, store: FileStore
    ) -> None:
        """Nested flow's halt doesn't emit checkpoint — only outer flow does."""
        from llm_gent.flow import Context, Factory, verb

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

        ff = Factory(make_test_logger())
        inner = ff.create().call(inner_a).then(inner_b)

        pre = (
            ff.create(state={})
            .with_checkpoint_store(store, "nested-halt")
            .with_checkpointer()
            .with_halt(halt)
            .call(outer_a)
            .call(inner)
            .then(outer_b)
        )
        await pre.run()

        # Checkpoint saved at outer_b (halt observed between outer's call(inner) and outer_b)
        # NOT at inner_b (nested flow's chain-walk skips halt observation)
        commit = head_commit(store, "nested-halt")
        assert commit.meta.outcome == "halted"

        # Resume — outer_b runs, inner flow doesn't re-run
        halt.clear()
        inner_resume = ff.create().call(inner_a).then(inner_b)
        resume = (
            ff.create(state={})
            .with_checkpoint_store(store, "nested-halt")
            .with_checkpointer()
            .call(outer_a)
            .call(inner_resume)
            .then(outer_b)
        )
        result = await resume.run(resume="latest")
        # inner_b ran before halt (state has inner_b=111), outer_b adds 1000
        assert result == 1111


class TestSaveOnHaltIterate:
    """Iterate halt-observation site writes a commit at the halted iteration."""

    async def test_halt_before_first_iteration_saves_at_iteration_zero(
        self, store: FileStore
    ) -> None:
        """Halt already set on entry to the iterate → commit saved at iteration=0."""

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

        commit = head_commit(store, "halt-iter-0")
        assert commit.meta.outcome == "halted"
        assert commit.meta.iteration == 0

        # Resume with halt cleared — completes the full iterate.
        halt.clear()
        result = await build_canonical_flow(
            make_test_logger(), max_iters=5, store=store, client_flow_id="halt-iter-0"
        ).run(resume="latest")
        assert result["iterations_completed"] == 5

    async def test_halt_between_iterations_overwrites_ok_ref_with_halted_commit(
        self, store: FileStore
    ) -> None:
        """Halt after body N → halted commit overwrites the ok ref at iteration=N."""

        halt = asyncio.Event()
        await build_canonical_flow(
            make_test_logger(),
            max_iters=5,
            halt=halt,
            halt_after_iteration=2,
            store=store,
            client_flow_id="halt-iter-mid",
        ).run()

        commit = head_commit(store, "halt-iter-mid")
        # Halt fired inside body 2's halt_check (iteration counter=2 by the time the loop-top
        # check observed halt). Post-halt-save iteration matches.
        assert commit.meta.outcome == "halted"
        assert commit.meta.iteration == 2

        # Resume completes the remaining iterations.
        halt.clear()
        result = await build_canonical_flow(
            make_test_logger(), max_iters=5, store=store, client_flow_id="halt-iter-mid"
        ).run(resume="latest")
        assert result["iterations_completed"] == 5


# ---------------------------------------------------------------------------
# History lineage — every commit's parent is the previous head
# ---------------------------------------------------------------------------


def _seed_dangling_commit(store: FileStore, client_flow_id: str, tree: str) -> None:
    """Store a commit of ``tree`` at ``HEAD`` without checking that ``tree`` is stored."""
    from llm_gent.flow.state.cas import CommitMeta, ProducedBy

    meta = CommitMeta(
        node_path="root",
        iteration=0,
        produced_by=ProducedBy(node_id="x", verb_name=None, role=None, result_hash=None),
        trace_ref=(),
        outcome="ok",
        flow_root_hash="",
        timestamp_iso="1970-01-01T00:00:00+00:00",
        framework_version="test",
    )
    commit, obj = FlowCommit.build(tree, (), meta)
    store.write_object(client_flow_id, commit.id, "commit", obj.serialize())
    store.write_ref(client_flow_id, HEAD_REF, commit.id, None)


def _chain_from_head(store: FileStore, client_flow_id: str) -> list[FlowCommit]:
    """Walk parent links from the newest ref; return commits newest-first."""
    chain: list[FlowCommit] = []
    commit_hash = store.repo(client_flow_id).get_ref(HEAD_REF)
    while commit_hash is not None:
        commit = read_commit(store, client_flow_id, commit_hash)
        chain.append(commit)
        assert len(commit.parents) <= 1
        commit_hash = commit.parents[0] if commit.parents else None
    return chain


class TestHistoryLineage:
    async def test_iterate_commits_form_one_chain(self, store: FileStore) -> None:
        """Per-iteration commits chain onto each other; the final-state commit is the head."""
        await build_canonical_flow(
            make_test_logger(), max_iters=3, store=store, client_flow_id="lineage-iter"
        ).run()

        chain = _chain_from_head(store, "lineage-iter")
        assert chain[0].meta.node_path == "$end"
        assert chain[-1].parents == ()
        # Every stored commit is reachable from the head — no orphaned siblings.
        assert {c.id for c in chain} == commit_ids(store, "lineage-iter")
        assert len(chain) == 4

    async def test_concurrent_map_items_chain_linearly(self, store: FileStore) -> None:
        """Parallel map-item saves serialize into one chain, not siblings of one parent."""
        from llm_gent.flow import Context, Factory, verb

        @verb
        async def touch(ctx: Context[dict[str, Any]], item: int) -> int:
            await asyncio.sleep(0)
            return item * 2

        body = Factory(make_test_logger()).create()
        body.call(touch)
        outer = (
            Factory(make_test_logger())
            .create(state={})
            .with_checkpoint_store(store, "lineage-map")
            .with_checkpointer()
            .with_checkpoint_policy(on_map_item=True)
            .map(body, items=lambda _p, _c: [1, 2, 3, 4])
        )
        await outer.run()

        chain = _chain_from_head(store, "lineage-map")
        assert {c.id for c in chain} == commit_ids(store, "lineage-map")
        assert len(chain) == 5
        assert sorted(c.meta.iteration for c in chain[1:]) == [0, 1, 2, 3]

    async def test_resume_continues_the_same_history(self, store: FileStore) -> None:
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
        halted_head = store.repo("lineage-resume").get_ref(HEAD_REF)
        assert halted_head is not None

        halt.clear()
        await build_canonical_flow(
            make_test_logger(), max_iters=5, store=store, client_flow_id="lineage-resume"
        ).run(resume="latest")

        chain = _chain_from_head(store, "lineage-resume")
        assert halted_head in {c.id for c in chain[1:]}
        assert {c.id for c in chain} == commit_ids(store, "lineage-resume")
        assert sum(1 for c in chain if not c.parents) == 1

    async def test_deleting_the_history_restarts_the_chain(self, store: FileStore) -> None:
        """After the history is deleted the next commit starts a new history with no parent."""
        from appinfra.cas import Tree

        from llm_gent.flow._checkpoint_ctx import CheckpointContext
        from llm_gent.flow.structure import Structure

        ctx = CheckpointContext(store, "lineage-gc", lambda: Structure(()), "retain")
        tree = Tree()
        ctx.put(tree)

        async def _append(iteration: int) -> FlowCommit:
            meta = ctx._build_commit_meta("node", iteration, "node", "ok", ())
            return await ctx.append_commit(tree.id, meta)

        first = await _append(0)
        second = await _append(1)
        assert first.parents == ()
        assert second.parents == (first.id,)

        await ctx.delete_history()
        ctx.put(tree)
        restarted = await _append(0)
        assert restarted.parents == ()


class TestFlowRootHash:
    async def test_every_commit_records_the_flow_root_hash(self, store: FileStore) -> None:
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
        await resumed.run(resume="latest")

        chain = _chain_from_head(store, "root-hash")
        assert {c.meta.outcome for c in chain} == {"ok", "halted"}
        assert {c.meta.flow_root_hash for c in chain} == {flow.root_hash()}
        assert resumed.root_hash() == flow.root_hash()


class TestRunScopedCache:
    async def test_reused_flow_sees_an_external_delete(self, store: FileStore) -> None:
        """A Flow reused after someone else deleted its history writes a new one."""
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
        old = commit_ids(store, "reused")
        assert old
        store.delete_repo("reused")

        halt.clear()
        await flow.run()  # halts again, into a fresh history

        history = History(store, "reused")
        head = await history.head()
        assert head is not None and head.meta.outcome == "halted"
        chain = [c async for c in history.commits()]
        assert chain[-1].parents == ()
        assert {c.id for c in chain} == commit_ids(store, "reused")


class TestCompletionTag:
    """Clean exit commits the final state at ``$end`` and moves the ``complete`` tag to it."""

    async def test_unserializable_final_state_still_completes(self, store: FileStore) -> None:
        """A finished run whose state can't be serialized returns normally; no state is kept."""
        from llm_gent.flow import Context, Factory, History, verb

        @verb
        async def attach(ctx: Context[dict[str, Any]], _prev: Any = None) -> str:
            ctx.state.data["handle"] = object()
            return "done"

        result = await (
            Factory(make_test_logger())
            .create(state={})
            .with_checkpoint_store(store, "opaque-state")
            .with_checkpointer()
            .call(attach)
            .run()
        )

        assert result == "done"
        history = History(store, "opaque-state")
        head = await history.head()
        assert head is not None and History.is_final_state(head)
        assert await history.is_complete()
        assert not (await history.snapshot(head)).has_state

    async def test_final_state_commit_is_tagged_head(self, store: FileStore) -> None:
        """A halt-only run with no save points still leaves its final state at the head."""
        from llm_gent.flow import Context, Factory, History, verb
        from llm_gent.flow.checkpoint import COMPLETE_TAG, COMPLETION_PRODUCER

        @verb
        async def bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["n"] += 1
            return ctx.state.data["n"]

        await (
            Factory(make_test_logger())
            .create(state={"n": 0})
            .with_checkpoint_store(store, "final-state")
            .with_checkpointer()
            .iterate(lambda body: body.call(bump), max_iters=3)
            .run()
        )

        repo = store.repo("final-state")
        head = repo.get_ref(HEAD_REF)
        assert head is not None
        assert repo.get_ref(COMPLETE_TAG) == head
        (commit,) = _chain_from_head(store, "final-state")
        assert commit.meta.node_path == "$end"
        assert commit.meta.produced_by.node_id == COMPLETION_PRODUCER
        snapshot = await History(store, "final-state").snapshot(commit)
        assert (snapshot.root, snapshot.scopes, snapshot.cursors) == ({"n": 3}, {}, {})

    async def test_rerun_after_completion_appends_to_same_history(self, store: FileStore) -> None:
        """resume="latest" on a complete history runs fresh; its commits extend the history."""
        from llm_gent.flow.checkpoint import COMPLETE_TAG

        await build_canonical_flow(
            make_test_logger(), max_iters=2, store=store, client_flow_id="rerun"
        ).run()
        repo = store.repo("rerun")
        first_end = repo.get_ref(HEAD_REF)

        await build_canonical_flow(
            make_test_logger(), max_iters=2, store=store, client_flow_id="rerun"
        ).run(resume="latest")

        chain = _chain_from_head(store, "rerun")
        assert repo.get_ref(COMPLETE_TAG) == chain[0].id
        assert first_end in {c.id for c in chain[1:]}
        assert [c.meta.node_path for c in chain].count("$end") == 2

    async def test_commit_after_completion_makes_history_resumable(self, store: FileStore) -> None:
        """A halt after a completed run leaves the tag behind; the next resume continues the halt."""
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
        ).run(resume="latest")
        repo = store.repo("tag-behind")
        assert repo.get_ref(COMPLETE_TAG) != repo.get_ref(HEAD_REF)

        # A fresh run would start from this fallback state (n=100); resume ignores it.
        resumed = await build_canonical_flow(
            make_test_logger(),
            state=CanonicalCounter(n=100),
            max_iters=5,
            store=store,
            client_flow_id="tag-behind",
        ).run(resume="latest")
        # First run: 2 passes. Second run continues from its final state with its
        # own max_iters=5 and halts after its first pass; the resume runs its other 4.
        assert resumed["iterations_completed"] == 7
        assert resumed["log"][0] == 1
        assert repo.get_ref(COMPLETE_TAG) == repo.get_ref(HEAD_REF)

    async def test_untagged_final_state_head_is_still_the_final_state(
        self, store: FileStore
    ) -> None:
        """A crash between the $end commit and the tag write: resume continues from $end."""
        from llm_gent.flow._checkpoint_ctx import CheckpointContext
        from llm_gent.flow.checkpoint import COMPLETE_TAG
        from llm_gent.flow.state import State
        from llm_gent.flow.state.snapshot import ScopeRegistry
        from llm_gent.flow.structure import Structure
        from llm_gent.flow.testing.checkpoint import CanonicalCounter

        ctx = CheckpointContext(store, "torn-completion", lambda: Structure(()), "retain")
        scopes = ScopeRegistry()
        scopes.begin(State(data={"n": 7}))
        tree = ctx.put_snapshot(scopes)
        await ctx.save_completion_commit(tree)
        assert store.repo("torn-completion").get_ref(COMPLETE_TAG) is None

        result = await build_canonical_flow(
            make_test_logger(),
            state=CanonicalCounter(n=100),
            max_iters=1,
            store=store,
            client_flow_id="torn-completion",
        ).run(resume="latest")
        assert result["log"][0] == 8  # n=7 from $end, not the fallback n=100


# ---------------------------------------------------------------------------
# Resume determinism — baseline vs interrupt+resume must reach same final state
# ---------------------------------------------------------------------------


class TestResumeDeterminism:
    async def test_resume_matches_uninterrupted(self, store: FileStore) -> None:
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

    async def test_resume_with_no_prior_checkpoint_starts_fresh(self, store: FileStore) -> None:
        """resume="latest" on a history with no ref falls back to a fresh run."""
        flow = build_canonical_flow(
            make_test_logger(), max_iters=2, store=store, client_flow_id="never-saved"
        )
        result = await flow.run(resume="latest")
        assert result["iterations_completed"] == 2

    async def test_resume_after_clean_exit_starts_a_new_run_on_the_final_state(
        self, store: FileStore
    ) -> None:
        """The final commit holds no cursors: resume continues from its state, not a position.

        Resuming from an earlier commit of the finished run — the last
        iterate commit — would re-enter the run at that position and run
        the steps after it a second time.
        """
        from llm_gent.flow import Context, Factory, History, verb

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
                Factory(make_test_logger())
                .create(state={"counter": 0})
                .with_checkpoint_store(store, "complete-1")
                .with_checkpointer()
                .iterate(lambda body: body.call(bump), max_iters=3)
                .call(tail)
            )

        await _flow().run()
        assert tail_calls == [3]
        history = History(store, "complete-1")
        head = await history.head()
        assert head is not None and (await history.snapshot(head)).cursors == {}
        # A new run on the final state (counter=3): three more passes, then tail once.
        await _flow().run(resume="latest")
        assert tail_calls == [3, 6], f"tail should fire once per run; got {tail_calls}"

    async def test_multiple_resume_boundaries(self, store: FileStore) -> None:
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

    async def test_scope_less_iterate_restores_carry_from_cursor(self, store: FileStore) -> None:
        """A scope-less iterate (no state=) restores the carried value from its cursor on resume.

        The cursor is registered at ``n/<node_id>`` under the parent scope, not at
        ``scope_path`` (which equals the parent scope's path when the iterate has no
        scope of its own). Without the cursor-path fix, _restore_carry would look up
        the parent's path and find nothing, causing the carry to reset to the run() arg.
        """
        from llm_gent.flow import Context, Factory, verb

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
                Factory(make_test_logger())
                .create(state={"iter": 0})
                .with_checkpoint_store(store, "scopeless-carry")
                .with_checkpointer()
                .with_halt(halt)
                .iterate(lambda b: b.call(step), max_iters=4)
            )

        await _flow().run(0)
        assert calls == [(0, 0), (1, 10)]

        halt.clear()
        calls.clear()
        result = await _flow().run(resume="latest")
        assert calls == [(2, 20), (3, 30)]
        assert result == 40


# ---------------------------------------------------------------------------
# Resume error paths — no client_flow_id, corruption, structural drift
# ---------------------------------------------------------------------------


class TestResumeErrorPaths:
    async def test_resume_raises_when_commit_object_missing(self, store: FileStore) -> None:
        """Ref points at a commit that isn't stored → HistoryCorrupt, not a fresh run."""
        from llm_gent.flow import HistoryCorrupt

        # Seed HEAD pointing at a non-existent commit hash.
        store.write_ref("orphan-ref", HEAD_REF, "0" * 64, None)
        flow = build_canonical_flow(
            make_test_logger(), max_iters=1, store=store, client_flow_id="orphan-ref"
        )
        with pytest.raises(HistoryCorrupt, match="commit"):
            await flow.run(resume="latest")

    async def test_resume_raises_when_tree_object_missing(self, store: FileStore) -> None:
        """Commit points at a tree that isn't stored → HistoryCorrupt, not a fresh run."""
        from llm_gent.flow import HistoryCorrupt

        _seed_dangling_commit(store, "orphan-tree", "d" * 64)
        flow = build_canonical_flow(
            make_test_logger(), max_iters=1, store=store, client_flow_id="orphan-tree"
        )
        with pytest.raises(HistoryCorrupt, match="tree"):
            await flow.run(resume="latest")

    async def test_resume_raises_when_blob_missing(self, store: FileStore) -> None:
        """Tree entry references a blob that isn't stored → HistoryCorrupt, not a fresh run."""
        from appinfra.cas import Tree, TreeEntry

        from llm_gent.flow import HistoryCorrupt

        # A tree with an entry pointing at a nonexistent blob.
        tree = Tree((TreeEntry("state", "blob", "c" * 64),))
        store.write_object("orphan-blob", tree.id, "tree", tree.serialize())
        _seed_dangling_commit(store, "orphan-blob", tree.id)
        flow = build_canonical_flow(
            make_test_logger(), max_iters=1, store=store, client_flow_id="orphan-blob"
        )
        with pytest.raises(HistoryCorrupt, match="blob"):
            await flow.run(resume="latest")


# ---------------------------------------------------------------------------
# Scoped state — .call(state=...) round-trip through the CAS commit tree
# ---------------------------------------------------------------------------


class TestScopedStateRoundTrip:
    async def test_call_scope_state_survives_resume(self, store: FileStore) -> None:
        """A ``.call(state=child)`` scope's mutations survive resume.

        Outer flow projects a child scope for its inner subflow; inner
        subflow's iterate mutates the projected scope and halts. On
        resume, the projected scope's payload restores from the CAS
        commit instead of being re-projected fresh — verified by
        inspecting the pre-resume commit's leaf blob and confirming the
        resume completes without a structural-drift error.
        """
        from llm_gent.flow import Context, Factory, History, verb

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

        ff = Factory(make_test_logger())
        inner = ff.create().iterate(lambda body: body.call(bump).then(maybe_halt), max_iters=5)

        outer_pre = (
            ff.create(state={"outer": True})
            .with_checkpoint_store(store, "scoped-1")
            .with_checkpointer()
            .with_halt(halt)
            .call(inner, state=lambda _p: {"counter": 0})
        )
        await outer_pre.run()

        # The halt commit's snapshot holds the .call scope with counter=2.
        halted = await History(store, "scoped-1").head()
        assert halted is not None
        snapshot = await History(store, "scoped-1").snapshot(halted)
        assert list(snapshot.scopes.values()) == [{"counter": 2}]
        assert snapshot.scopes[halted.meta.scope_path] == {"counter": 2}

        # Resume — projected scope restores from the commit instead of
        # re-projecting to counter=0; run reaches max_iters=5 cleanly.
        outer_resume = (
            ff.create(state={"outer": True})
            .with_checkpoint_store(store, "scoped-1")
            .with_checkpointer()
            .call(inner, state=lambda _p: {"counter": 0})
        )
        await outer_resume.run(resume="latest")

    async def test_three_level_nested_scopes_survive_resume(self, store: FileStore) -> None:
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
        from llm_gent.flow import Context, Factory, verb

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

        ff_inner = Factory(make_test_logger())
        # Iterate has its own state=lambda so its body runs in a NEW scope
        # (leaf, depth 2). The middle .call scope sits between root and
        # the iterate body.
        mid_flow = ff_inner.create().iterate(
            lambda body: body.call(mark_middle).then(maybe_halt),
            state=lambda _p: {"leaf_iter": 0},
            max_iters=5,
        )

        ff = Factory(make_test_logger())
        outer_pre = (
            ff.create(state={"outer": True})
            .with_checkpoint_store(store, "3-level-1")
            .with_checkpointer()
            .with_checkpoint_policy(on_iterate=True)
            .with_halt(halt)
            .call(mid_flow, state=lambda _p: {})
        )
        await outer_pre.run()
        assert store.repo("3-level-1").get_ref(HEAD_REF) is not None

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
            .with_checkpoint_store(store, "3-level-1")
            .with_checkpointer()
            .with_checkpoint_policy(on_iterate=True)
            .call(mid_flow, state=lambda _p: {})
        )
        await outer_resume.run(resume="latest")
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

    async def test_leaf_iterate_with_state_survives_resume(self, store: FileStore) -> None:
        """A leaf iterate with its own ``state=`` projection restores on resume.

        Regression test for the case where the leaf iterate (not just its
        parent .call) has a ``state=`` projection. The iterate's scope must
        be placed in ``child_state_data`` by ``_split_scopes`` so that
        ``_resume_iteration`` returns it correctly.
        """
        from llm_gent.flow import Context, Factory, History, verb

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

        ff = Factory(make_test_logger())

        # Leaf iterate with its own state= projection (not inherited from .call)
        flow_pre = (
            ff.create(state={"root": True})
            .with_checkpoint_store(store, "leaf-iterate-state")
            .with_checkpointer()
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
            .with_checkpoint_store(store, "leaf-iterate-state")
            .with_checkpointer()
            .iterate(
                lambda body: body.call(bump).then(maybe_halt),
                max_iters=5,
                state=lambda _p: {"counter": 0},
            )
        )
        await flow_resume.run(resume="latest")

        # Final commit should exist (run completed without structural-drift error)
        final = await history.head()
        assert final is not None and History.is_final_state(final)
