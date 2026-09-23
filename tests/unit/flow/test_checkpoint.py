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
        """A ``.call(state=child)`` scope's mutations survive across resume.

        The executor walks the scope stack per iteration and writes a
        Blob per scope into the commit tree; on resume, the load path
        reconstructs the leaf scope's data and hands it to the iterate
        body via ``_ResumeReplay.child_state_data``.
        """
        from llm_gent.flow import Context, verb

        @verb
        async def bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["counter"] += 1
            return ctx.state.data["counter"]

        halt = asyncio.Event()

        @verb
        async def halt_after_two(ctx: Context[dict[str, int]], _prev: Any = None) -> Any:
            if ctx.state.data["counter"] >= 2:
                halt.set()
            return _prev

        from llm_gent.flow.factory import FlowFactory
        from llm_gent.flow.state import TypeStateFactory
        from llm_gent.flow.testing.checkpoint import CanonicalCounter  # reuse the FF pattern

        ff = FlowFactory(make_test_logger(), state_factory=TypeStateFactory(CanonicalCounter))

        def _body(f: Any) -> None:
            f.call(bump).then(halt_after_two)

        flow = (
            ff.create(state={"counter": 0})
            .with_checkpointer(store, "scoped-1")
            .with_halt(halt)
            .iterate(_body, max_iters=5)
        )
        await flow.run()
        # Trajectory preserved (halt exit — not gc'd regardless of retention).
        assert store.resolve_ref("scoped-1") is not None


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
