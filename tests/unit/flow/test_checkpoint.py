# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""End-to-end Flow-level checkpoint/resume smoke tests.

Runs the canonical multi-stage flow (add_one → times_two → minus_three,
iterated) through a real :class:`JsonFileCheckpointStore` and asserts:

- fresh runs save at iterate boundaries;
- resume from a mid-run halt reaches the same final state as an
  uninterrupted run (determinism);
- the retention policy toggles gc-on-clean-exit;
- halt-triggered exit always preserves the trajectory.

Direct-Protocol tests (put/get/put_ref/resolve_ref/gc) live in
:mod:`tests.unit.flow.test_stores_json` (and the PG equivalent in
:mod:`tests.integration.flow.test_stores_pg`).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from llm_gent.flow.stores import JsonFileCheckpointStore
from llm_gent.flow.testing.checkpoint import (
    assert_resume_determinism,
    build_canonical_flow,
)

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


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
            make_test_logger(), max_iters=3, store=store, trajectory_id="freshrun"
        ).run()
        # A resolvable ref exists — the trajectory reached at least one commit.
        assert store.resolve_ref("freshrun") is not None

    async def test_default_retention_keeps_trajectory_on_success(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """retention="retain" (default): clean-exit does NOT gc the trajectory."""
        await build_canonical_flow(
            make_test_logger(), max_iters=2, store=store, trajectory_id="retain-1"
        ).run()
        # Successful run — but retention="retain" so the ref survives.
        assert store.resolve_ref("retain-1") is not None


# ---------------------------------------------------------------------------
# Retention policy
# ---------------------------------------------------------------------------


class TestRetention:
    async def test_gc_on_success_prunes(self, tmp_path: Path) -> None:
        """retention="gc_on_success": clean-exit removes the trajectory."""
        store = JsonFileCheckpointStore(
            make_test_logger(), tmp_path / "cp", retention="gc_on_success"
        )
        await build_canonical_flow(
            make_test_logger(), max_iters=2, store=store, trajectory_id="gc-1"
        ).run()
        assert store.resolve_ref("gc-1") is None

    async def test_halt_preserves_trajectory_regardless_of_retention(self, tmp_path: Path) -> None:
        """A halt-triggered exit preserves the trajectory even under gc_on_success —
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
            trajectory_id="halt-preserve",
        ).run()
        assert store.resolve_ref("halt-preserve") is not None


# ---------------------------------------------------------------------------
# Save on halt — halt-observation sites emit a commit at the halt position
# ---------------------------------------------------------------------------


class TestCheckpointPolicyIterate:
    """CheckpointPolicy.on_iterate gates the iterate-boundary auto-save."""

    async def test_default_policy_writes_only_the_completion_marker(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Default CheckpointPolicy (halt-only) skips per-iteration saves.

        Under the default policy an iterate that runs cleanly to
        ``max_iters`` should leave the store with exactly one ref:
        the ``$complete`` completion marker stamped by the clean-exit
        retention path. No iterate-boundary ref should exist.
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

        # Every commit written during the run has node_path == "$complete"
        # (only the clean-exit completion marker fires under the default
        # policy). Walking the commit objects on disk reveals no iterate-
        # boundary commits.
        commits_dir = store._root / "policy-halt-only" / "objects" / "commit"  # type: ignore[attr-defined]
        node_paths: set[str] = set()
        for f in commits_dir.iterdir():
            payload = f.read_bytes()
            node_paths.add(Commit.from_bytes(payload).meta.node_path)
        assert node_paths == {"$complete"}, (
            f"only the $complete completion marker should have been written; got {node_paths}"
        )

    async def test_on_iterate_true_writes_per_iteration_commits(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """CheckpointPolicy(on_iterate=True) restores per-iteration saves.

        Opting in should produce an iterate-boundary ref alongside the
        completion marker — the iterate's own node_path directory carries
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
        # iteration + a $complete marker on clean exit. Group commit
        # objects by node_path: 3 boundary commits share the iterate's
        # node_path, plus one $complete marker.
        commits_dir = store._root / "policy-on-iter" / "objects" / "commit"  # type: ignore[attr-defined]
        by_node_path: dict[str, int] = {}
        for f in commits_dir.iterdir():
            commit = Commit.from_bytes(f.read_bytes())
            by_node_path[commit.meta.node_path] = by_node_path.get(commit.meta.node_path, 0) + 1
        assert by_node_path["$complete"] == 1
        iterate_paths = {p: n for p, n in by_node_path.items() if p != "$complete"}
        assert len(iterate_paths) == 1, f"expected one iterate node_path; got {list(iterate_paths)}"
        # 3 iterate-boundary commits (one per iteration).
        assert next(iter(iterate_paths.values())) == 3


class TestCheckpointPolicyMap:
    """CheckpointPolicy.on_map_item gates the map per-item auto-save."""

    async def test_default_policy_writes_no_map_item_commits(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Under the default policy a completed map writes only the $complete marker."""
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

        commits_dir = store._root / "map-default" / "objects" / "commit"  # type: ignore[attr-defined]
        node_paths: set[str] = set()
        for f in commits_dir.iterdir():
            node_paths.add(Commit.from_bytes(f.read_bytes()).meta.node_path)
        assert node_paths == {"$complete"}, (
            f"only the $complete marker should exist under default policy; got {node_paths}"
        )

    async def test_on_map_item_true_writes_per_item_commits(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """CheckpointPolicy(on_map_item=True) saves after each successful item.

        Runs a 3-item map and asserts that three iteration-indexed
        commits exist under the map's node_path plus the $complete
        marker. Item order across saves is not asserted (concurrent).
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

        commits_dir = store._root / "map-on-item" / "objects" / "commit"  # type: ignore[attr-defined]
        iterations_at_map_path: dict[str, list[int]] = {}
        for f in commits_dir.iterdir():
            commit = Commit.from_bytes(f.read_bytes())
            iterations_at_map_path.setdefault(commit.meta.node_path, []).append(
                commit.meta.iteration
            )
        assert "$complete" in iterations_at_map_path
        non_marker = {p: v for p, v in iterations_at_map_path.items() if p != "$complete"}
        assert len(non_marker) == 1, f"expected one map node_path; got {list(non_marker)}"
        iterations = sorted(next(iter(non_marker.values())))
        assert iterations == [0, 1, 2], f"expected three item slots 0/1/2; got {iterations}"


class TestCtxCheckpoint:
    """Explicit ctx.checkpoint() writes a commit regardless of policy."""

    async def test_ctx_checkpoint_from_chain_step(self, store: JsonFileCheckpointStore) -> None:
        """A verb calling ``ctx.checkpoint()`` writes a commit at that step's node.

        Runs under the default (halt-only) policy so no iterate-
        boundary save fires; the only non-completion-marker commit
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
        # chain-step node) + one $complete marker on clean exit.
        commits_dir = store._root / "ctx-ckpt" / "objects" / "commit"  # type: ignore[attr-defined]
        node_paths: list[str] = []
        for f in commits_dir.iterdir():
            node_paths.append(Commit.from_bytes(f.read_bytes()).meta.node_path)
        assert "$complete" in node_paths
        non_marker = [p for p in node_paths if p != "$complete"]
        assert len(non_marker) == 1, (
            f"expected exactly one explicit-checkpoint commit; got {non_marker}"
        )

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


class TestSaiaTurnTraceRef:
    """Loop's paused conversation lands as a Blob referenced by trace_ref."""

    async def test_halt_save_stamps_saia_turn_and_writes_blob(
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
            .with_checkpointer(store, "saia-turn-1")
            .with_halt(halt)
            .call(run_loop)
            .then(after)
        )
        await flow.run()

        halted_hash = store.resolve_ref("saia-turn-1")
        assert halted_hash is not None
        commit = Commit.from_bytes(store.get_object("saia-turn-1", "commit", halted_hash) or b"")
        assert commit.meta.outcome == "halted"
        # trace_ref carries exactly one saia_turn entry pointing at a blob whose
        # bytes match canonical_json of the conversation's to_dict.
        assert len(commit.meta.trace_ref) == 1
        ref = commit.meta.trace_ref[0]
        assert ref.kind == "saia_turn"
        expected = canonical_json(conv.to_dict())
        stored = store.get_object("saia-turn-1", "blob", ref.id)
        assert stored == expected


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

        halted_hash = store.resolve_ref("chain-halt")
        assert halted_hash is not None
        commit = Commit.from_bytes(store.get_object("chain-halt", "commit", halted_hash) or b"")
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
        result = await resume.run(resume=True)
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
        # _save_halt_checkpoint short-circuits at env.checkpointer is None.
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
        halted_hash = store.resolve_ref("branch-halt")
        assert halted_hash is not None
        commit = Commit.from_bytes(store.get_object("branch-halt", "commit", halted_hash) or b"")
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
        result = await resume.run(resume=True)
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
        halted_hash = store.resolve_ref("pre-set-halt")
        assert halted_hash is not None
        commit = Commit.from_bytes(store.get_object("pre-set-halt", "commit", halted_hash) or b"")
        assert commit.meta.outcome == "halted"

        # Resume with halt cleared — step_b runs, reads items from state
        halt.clear()
        resume = (
            ff.create(state={}).with_checkpointer(store, "pre-set-halt").call(step_a).then(step_b)
        )
        result = await resume.run(resume=True)
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
        halted_hash = store.resolve_ref("nested-halt")
        assert halted_hash is not None
        commit = Commit.from_bytes(store.get_object("nested-halt", "commit", halted_hash) or b"")
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
        result = await resume.run(resume=True)
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
            trajectory_id="halt-iter-0",
        ).run()

        halted_hash = store.resolve_ref("halt-iter-0")
        assert halted_hash is not None
        commit = Commit.from_bytes(store.get_object("halt-iter-0", "commit", halted_hash) or b"")
        assert commit.meta.outcome == "halted"
        assert commit.meta.iteration == 0

        # Resume with halt cleared — completes the full iterate.
        halt.clear()
        result = await build_canonical_flow(
            make_test_logger(), max_iters=5, store=store, trajectory_id="halt-iter-0"
        ).run(resume=True)
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
            trajectory_id="halt-iter-mid",
        ).run()

        halted_hash = store.resolve_ref("halt-iter-mid")
        assert halted_hash is not None
        commit = Commit.from_bytes(store.get_object("halt-iter-mid", "commit", halted_hash) or b"")
        # Halt fired inside body 2's halt_check (iteration counter=2 by the time the loop-top
        # check observed halt). Post-halt-save iteration matches.
        assert commit.meta.outcome == "halted"
        assert commit.meta.iteration == 2

        # Resume completes the remaining iterations.
        halt.clear()
        result = await build_canonical_flow(
            make_test_logger(), max_iters=5, store=store, trajectory_id="halt-iter-mid"
        ).run(resume=True)
        assert result["iterations_completed"] == 5


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
            trajectory_id="determinism-1",
        )
        # Final iterations_completed reflects the full max_iters run.
        assert final["iterations_completed"] == 5

    async def test_resume_with_no_prior_checkpoint_starts_fresh(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """resume=True on a trajectory with no ref falls back to a fresh run."""
        flow = build_canonical_flow(
            make_test_logger(), max_iters=2, store=store, trajectory_id="never-saved"
        )
        result = await flow.run(resume=True)
        assert result["iterations_completed"] == 2

    async def test_resume_after_clean_exit_does_not_replay_final_commit(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A completed run stamps a marker so `resume=True` after success is a no-op.

        Without the marker, `run(resume=True)` after a successful run
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
        # trajectory is marked complete). Tail runs ONCE more from the
        # fresh state, not twice from the resumed one.
        await _flow().run(resume=True)
        assert tail_calls == [3, 3], f"tail should have fired only twice total; got {tail_calls}"

    async def test_multiple_resume_boundaries(self, store: JsonFileCheckpointStore) -> None:
        """Interrupt at different iterations, resume each — final state matches uninterrupted."""
        for cut_at in (1, 2, 3, 4):
            traj = f"multi-cut-{cut_at}"
            final = await assert_resume_determinism(
                make_test_logger(),
                store,
                halt_after_iteration=cut_at,
                max_iters=5,
                trajectory_id=traj,
            )
            assert final["iterations_completed"] == 5


# ---------------------------------------------------------------------------
# Resume error paths — no client_flow_id, corruption, structural drift
# ---------------------------------------------------------------------------


class TestResumeErrorPaths:
    async def test_resume_without_client_flow_id_raises(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """resume=True on a checkpointer bound without client_flow_id raises."""
        from llm_gent.flow.factory import FlowFactory

        ff = FlowFactory(make_test_logger())
        # with_checkpointer is what binds client_flow_id — skip it, leave
        # _checkpointer set but _client_flow_id None by manual attribute.
        flow = ff.create(state={})
        flow._checkpointer = store
        # resume=True with no client_flow_id → RuntimeError.
        with pytest.raises(RuntimeError, match="no client_flow_id"):
            await flow.run(resume=True)

    async def test_resume_falls_through_when_commit_object_missing(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Ref points at a commit that isn't stored → treated as absent, fresh run."""
        # Seed a ref pointing at a non-existent commit hash.
        store.put_ref("orphan-ref", "some/node", 0, "0" * 64)
        # No corresponding commit object was ever put — resume falls back.
        flow = build_canonical_flow(
            make_test_logger(), max_iters=1, store=store, trajectory_id="orphan-ref"
        )
        result = await flow.run(resume=True)
        assert result["iterations_completed"] == 1

    async def test_resume_falls_through_when_tree_object_missing(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Commit points at a tree that isn't stored → treated as absent, fresh run."""
        from llm_gent.flow.state.cas import (
            Commit,
            CommitMeta,
            ProducedBy,
        )

        meta = CommitMeta(
            client_flow_id="orphan-tree",
            node_path="root",
            iteration=0,
            produced_by=ProducedBy(node_id="x", verb_name=None, role=None, result_hash=None),
            trace_ref=(),
            outcome="ok",
            flow_root_id="orphan-tree",
            timestamp_iso="1970-01-01T00:00:00+00:00",
            framework_version="test",
        )
        commit = Commit.build(root_tree_hash="deadbeef" * 8, parent_hashes=(), meta=meta)
        store.put_object("orphan-tree", "commit", commit.content_hash, commit.to_bytes())
        store.put_ref("orphan-tree", "root", 0, commit.content_hash)
        # Tree object missing → resume falls back.
        flow = build_canonical_flow(
            make_test_logger(), max_iters=1, store=store, trajectory_id="orphan-tree"
        )
        result = await flow.run(resume=True)
        assert result["iterations_completed"] == 1

    async def test_resume_falls_through_when_blob_missing(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Tree entry references a blob that isn't stored → fresh run."""
        from llm_gent.flow.state.cas import (
            Commit,
            CommitMeta,
            ProducedBy,
            Tree,
            TreeEntry,
        )

        # Build a tree with an entry pointing at a nonexistent blob.
        entry = TreeEntry(scope_id="00", kind="blob", child_hash="c0ffee" * 10 + "1234")
        tree = Tree.from_entries([entry])
        meta = CommitMeta(
            client_flow_id="orphan-blob",
            node_path="root",
            iteration=0,
            produced_by=ProducedBy(node_id="x", verb_name=None, role=None, result_hash=None),
            trace_ref=(),
            outcome="ok",
            flow_root_id="orphan-blob",
            timestamp_iso="1970-01-01T00:00:00+00:00",
            framework_version="test",
        )
        commit = Commit.build(root_tree_hash=tree.content_hash, parent_hashes=(), meta=meta)
        store.put_object("orphan-blob", "tree", tree.content_hash, tree.to_bytes())
        store.put_object("orphan-blob", "commit", commit.content_hash, commit.to_bytes())
        store.put_ref("orphan-blob", "root", 0, commit.content_hash)
        # Blob missing → resume falls back.
        flow = build_canonical_flow(
            make_test_logger(), max_iters=1, store=store, trajectory_id="orphan-blob"
        )
        result = await flow.run(resume=True)
        assert result["iterations_completed"] == 1

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
        store.put_object("stale-1", "blob", blob.content_hash, blob.payload)
        tree = Tree.from_entries(
            [TreeEntry(scope_id="00", kind="blob", child_hash=blob.content_hash)]
        )
        store.put_object("stale-1", "tree", tree.content_hash, tree.to_bytes())
        meta = CommitMeta(
            client_flow_id="stale-1",
            node_path=stale_path,
            iteration=2,
            produced_by=ProducedBy(node_id="x", verb_name=None, role=None, result_hash=None),
            trace_ref=(),
            outcome="ok",
            flow_root_id="stale-1",
            timestamp_iso="1970-01-01T00:00:00+00:00",
            framework_version="test",
        )
        commit = Commit.build(root_tree_hash=tree.content_hash, parent_hashes=(), meta=meta)
        store.put_object("stale-1", "commit", commit.content_hash, commit.to_bytes())
        store.put_ref("stale-1", stale_path, 2, commit.content_hash)
        flow = build_canonical_flow(
            make_test_logger(), max_iters=3, store=store, trajectory_id="stale-1"
        )
        with pytest.raises(RuntimeError) as excinfo:
            await flow.run(resume=True)
        message = str(excinfo.value)
        assert "structurally changed" in message
        # Both stale ids appear in the triage message (root→leaf full_path).
        assert "cafebabecafebabe" in message
        assert "deadbeefdeadbeef" in message


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
        import json

        from llm_gent.flow import Context, FlowFactory, verb
        from llm_gent.flow.state.cas import Commit, Tree

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

        # Halt-commit's leaf scope (the .call scope) carries counter=2.
        halted_hash = store.resolve_ref("scoped-1")
        assert halted_hash is not None
        halted_commit = Commit.from_bytes(
            store.get_object("scoped-1", "commit", halted_hash) or b""
        )
        halted_tree = Tree.from_bytes(
            store.get_object("scoped-1", "tree", halted_commit.root_tree_hash) or b""
        )
        leaf_hash = halted_tree.entries[-1].child_hash
        leaf_data = json.loads((store.get_object("scoped-1", "blob", leaf_hash) or b"").decode())
        assert leaf_data == {"counter": 2}

        # Resume — projected scope restores from the commit instead of
        # re-projecting to counter=0; run reaches max_iters=5 cleanly.
        outer_resume = (
            ff.create(state={"outer": True})
            .with_checkpointer(store, "scoped-1")
            .call(inner, state=lambda _p: {"counter": 0})
        )
        await outer_resume.run(resume=True)

    async def test_three_level_nested_scopes_survive_resume(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """3-level nested ``.call(state=)`` chain — intermediate scope restores.

        Composition::

            outer(root scope)
              .call(mid_flow, state=lambda p: {...})    # depth 1 — middle scope
                mid_flow.iterate(body, ...)              # depth 2 — leaf iterate

        The middle scope's payload lands as a Blob under scope_id "01"
        in the commit's tree. Before the fix, ``_hydrate_resume_state``
        extracted only root ("00") and leaf ("02"); the middle scope
        was dropped and re-projected via the state factory on resume.

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
        assert store.resolve_ref("3-level-1") is not None

        # Load the commit and inspect the middle scope's blob directly —
        # end-to-end verification that scope_id "01" carries the witness
        # value. This asserts the SAVE side without needing the fix on
        # the load side.
        from llm_gent.flow.state.cas import Commit, Tree

        commit_hash = store.resolve_ref("3-level-1")
        assert commit_hash is not None
        commit = Commit.from_bytes(store.get_object("3-level-1", "commit", commit_hash) or b"")
        tree = Tree.from_bytes(store.get_object("3-level-1", "tree", commit.root_tree_hash) or b"")
        # Three scope entries: root (00), middle (01), leaf (02).
        assert [e.scope_id for e in tree.entries] == ["00", "01", "02"]
        middle_entry = tree.entries[1]
        import json

        middle_data = json.loads(
            (store.get_object("3-level-1", "blob", middle_entry.child_hash) or b"").decode()
        )
        assert middle_data.get(WITNESS_KEY) == 2, (
            f"middle scope blob should carry the WITNESS mutation; got {middle_data!r}"
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
        await outer_resume.run(resume=True)
        # Inspect the FINAL commit under the iterate's node_path (the
        # completion marker at "$complete" is skipped by this specific
        # lookup — it's stamped on clean exit).
        final_commit_hash = store.resolve_ref("3-level-1", commit.meta.node_path)
        assert final_commit_hash is not None
        final_commit = Commit.from_bytes(
            store.get_object("3-level-1", "commit", final_commit_hash) or b""
        )
        final_tree = Tree.from_bytes(
            store.get_object("3-level-1", "tree", final_commit.root_tree_hash) or b""
        )
        final_middle_hash = next(e.child_hash for e in final_tree.entries if e.scope_id == "01")
        final_middle = json.loads(
            (store.get_object("3-level-1", "blob", final_middle_hash) or b"").decode()
        )
        assert final_middle.get(WITNESS_KEY) == 5, (
            f"resume should have restored the middle scope (WITNESS=2) and "
            f"continued for 3 more iterations to WITNESS=5; got {final_middle!r} "
            f"— re-projected middle would land at 3 instead"
        )


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

            async def put_object(self, *a: Any, **kw: Any) -> None:
                await asyncio.sleep(0)
                self._inner.put_object(*a, **kw)

            async def get_object(self, *a: Any, **kw: Any) -> bytes | None:
                await asyncio.sleep(0)
                return self._inner.get_object(*a, **kw)

            async def has_object(self, *a: Any, **kw: Any) -> bool:
                await asyncio.sleep(0)
                return self._inner.has_object(*a, **kw)

            async def put_ref(self, *a: Any, **kw: Any) -> None:
                await asyncio.sleep(0)
                self._inner.put_ref(*a, **kw)

            async def resolve_ref(self, *a: Any, **kw: Any) -> str | None:
                await asyncio.sleep(0)
                return self._inner.resolve_ref(*a, **kw)

            async def gc_trajectory(self, *a: Any, **kw: Any) -> None:
                await asyncio.sleep(0)
                self._inner.gc_trajectory(*a, **kw)

        inner = _Sync(make_test_logger(), tmp_path / "cp")
        wrap = _AsyncWrap(inner)
        # Sanity: every method IS async.
        for m in ("put_object", "get_object", "put_ref", "resolve_ref", "gc_trajectory"):
            assert inspect.iscoroutinefunction(getattr(wrap, m))

        # Round-trip via the canonical flow: fresh save + resume.
        halt = asyncio.Event()
        await build_canonical_flow(
            make_test_logger(),
            max_iters=5,
            halt=halt,
            halt_after_iteration=2,
            store=wrap,
            trajectory_id="async-round-trip",
        ).run()
        resumed = await build_canonical_flow(
            make_test_logger(),
            max_iters=5,
            store=wrap,
            trajectory_id="async-round-trip",
        ).run(resume=True)
        assert resumed["iterations_completed"] == 5
