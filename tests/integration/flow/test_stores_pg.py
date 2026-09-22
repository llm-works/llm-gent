# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Integration tests for :class:`llm_gent.flow.stores.PgCheckpointStore`.

Exercises the same Protocol contract as the JSON-file backend but against
a real Postgres via appinfra's schema-isolated fixtures. Skips when
``APPINFRA_TEST_PG_URL`` is not set.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from appinfra.db.pg import PG
from appinfra.log import Logger
from sqlalchemy import delete

from llm_gent.flow.stores import PgCheckpointStore
from llm_gent.flow.stores.postgres import FlowCheckpoint
from llm_gent.flow.testing import build_canonical_flow, resume_in_subprocess


pytestmark = pytest.mark.integration


def _state(n: int) -> dict[str, Any]:
    """Small deterministic ``state_json`` fixture."""
    return {"data": {"n": n}, "children": []}


def _meta(iteration: int, path: list[str] | None = None) -> dict[str, Any]:
    """Small deterministic ``metadata_json`` fixture."""
    return {"path": path or ["node-a"], "iteration": iteration}


@pytest.fixture
def store(pg_migrated: PG, pg_test_logger: Logger) -> PgCheckpointStore:
    """PgCheckpointStore bound to the session's migrated schema."""
    return PgCheckpointStore(pg_test_logger, pg_migrated)


@pytest.fixture(autouse=True)
def _clean_table(pg_migrated: PG) -> None:
    """Wipe the checkpoint table between tests so cases are independent."""
    with pg_migrated.session() as session:
        session.execute(delete(FlowCheckpoint))


class TestPgCheckpointStore:
    """Save / load / delete Protocol conformance against Postgres."""

    def test_round_trip_single_iteration(self, store: PgCheckpointStore) -> None:
        """Save one row, load it back verbatim."""
        store.save_checkpoint("traj-1", "node-a", 1, _state(42), _meta(1))
        loaded = store.load_checkpoint("traj-1")
        assert loaded is not None
        state_json, meta_json = loaded
        assert state_json == _state(42)
        assert meta_json == _meta(1)

    def test_load_returns_none_when_absent(self, store: PgCheckpointStore) -> None:
        """A never-saved trajectory reads as ``None``."""
        assert store.load_checkpoint("nobody") is None

    def test_load_specific_iteration(self, store: PgCheckpointStore) -> None:
        """``iteration=N`` fetches exactly that row when it exists."""
        store.save_checkpoint("traj-1", "node-a", 1, _state(1), _meta(1))
        store.save_checkpoint("traj-1", "node-a", 2, _state(2), _meta(2))
        loaded = store.load_checkpoint("traj-1", iteration=1)
        assert loaded is not None
        assert loaded[0] == _state(1)

    def test_load_specific_iteration_missing(self, store: PgCheckpointStore) -> None:
        """A non-existent iteration under a live trajectory reads as ``None``."""
        store.save_checkpoint("traj-1", "node-a", 1, _state(1), _meta(1))
        assert store.load_checkpoint("traj-1", iteration=99) is None

    def test_load_latest_picks_last_save(self, store: PgCheckpointStore) -> None:
        """Both filters ``None`` returns the most recently inserted row (last db_id wins)."""
        store.save_checkpoint("traj-1", "node-a", 1, _state(1), _meta(1))
        store.save_checkpoint("traj-1", "node-a", 3, _state(3), _meta(3))
        store.save_checkpoint("traj-1", "node-a", 2, _state(2), _meta(2))
        loaded = store.load_checkpoint("traj-1")
        assert loaded is not None
        assert loaded[1]["iteration"] == 2

    def test_same_iteration_resave_upserts(self, store: PgCheckpointStore) -> None:
        """A second save at the same iteration replaces the earlier row (ON CONFLICT DO UPDATE)."""
        store.save_checkpoint("traj-1", "node-a", 1, _state(1), _meta(1))
        store.save_checkpoint("traj-1", "node-a", 1, _state(99), _meta(1))
        loaded = store.load_checkpoint("traj-1", iteration=1)
        assert loaded is not None
        assert loaded[0] == _state(99)

    def test_delete_wipes_trajectory(self, store: PgCheckpointStore) -> None:
        """Every row under the trajectory is gone after delete."""
        store.save_checkpoint("traj-1", "node-a", 1, _state(1), _meta(1))
        store.save_checkpoint("traj-1", "node-a", 2, _state(2), _meta(2))
        store.delete_checkpoint("traj-1")
        assert store.load_checkpoint("traj-1") is None
        assert store.load_checkpoint("traj-1", iteration=1) is None
        assert store.load_checkpoint("traj-1", iteration=2) is None

    def test_delete_idempotent_when_absent(self, store: PgCheckpointStore) -> None:
        """Deleting an unknown trajectory is a no-op, not an error."""
        store.delete_checkpoint("nobody")

    def test_delete_leaves_other_trajectories(self, store: PgCheckpointStore) -> None:
        """Deleting one trajectory does not affect a sibling under the same schema."""
        store.save_checkpoint("traj-a", "node-a", 1, _state(1), _meta(1))
        store.save_checkpoint("traj-b", "node-a", 1, _state(2), _meta(1))
        store.delete_checkpoint("traj-a")
        assert store.load_checkpoint("traj-a") is None
        loaded = store.load_checkpoint("traj-b")
        assert loaded is not None
        assert loaded[0] == _state(2)

    def test_ensure_schema_idempotent(self, pg_migrated: PG, pg_test_logger: Logger) -> None:
        """Re-running :func:`ensure_schema` on an already-migrated DB is a no-op."""
        from llm_gent.schema import SchemaManager, SchemaState

        mgr = SchemaManager(pg_test_logger, pg_migrated)
        status = mgr.ensure_schema()
        assert status.state == SchemaState.CURRENT


class TestPgCheckpointStoreCrossProcessResume:
    """A fresh Python subprocess resuming from PG matches the uninterrupted final state.

    Same-process resume can silently retain non-serializable references
    across the round-trip; a subprocess with only the PG-persisted
    checkpoint is the production gate for network-backed stores.
    """

    def test_cross_process_resume_pg_store(
        self,
        pg_migrated: PG,
        pg_test_config: dict[str, Any],
        pg_test_schema: str,
        pg_test_logger: Logger,
    ) -> None:
        """Baseline and subprocess resume against the same PG schema yield identical final state."""
        store = PgCheckpointStore(pg_test_logger, pg_migrated)
        trajectory_id = "pg-cross-proc-1"

        baseline = asyncio.run(build_canonical_flow(pg_test_logger, max_iters=5).run())

        halt = asyncio.Event()
        asyncio.run(
            build_canonical_flow(
                pg_test_logger,
                max_iters=5,
                halt=halt,
                halt_after_iteration=2,
                store=store,
                trajectory_id=trajectory_id,
            ).run()
        )

        resumed = resume_in_subprocess(
            store_module="llm_gent.flow.testing.checkpoint",
            store_factory="pg_checkpoint_store_from_config",
            store_kwargs={
                "url": str(pg_test_config["url"]),
                "schema": pg_test_schema,
            },
            flow_builder_kwargs={"max_iters": 5},
            trajectory_id=trajectory_id,
        )

        assert resumed == baseline
