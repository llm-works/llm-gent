# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Integration tests for :class:`llm_gent.flow.stores.PgCheckpointStore`.

Real Postgres round-trip against the migrated schema (object + ref
tables). Same surface as :mod:`tests.unit.flow.test_stores_json` — the
two implementations should behave identically at the Protocol level.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from appinfra.db.pg import PG
from appinfra.log import Logger
from sqlalchemy import delete

from llm_gent.flow.stores import PgCheckpointStore
from llm_gent.flow.stores.postgres import FlowObject, FlowRef


@pytest.fixture
def store(pg_migrated: PG, pg_test_logger: Logger) -> PgCheckpointStore:
    """PgCheckpointStore bound to the session's migrated schema."""
    return PgCheckpointStore(pg_test_logger, pg_migrated)


@pytest.fixture(autouse=True)
def clean_tables(pg_migrated: PG) -> Generator[None, None, None]:
    """Wipe the object + ref tables before and after every test.

    Fixtures run in module scope so state leaks between tests without
    this. Autouse keeps every test independent.
    """
    with pg_migrated.session() as session:
        session.execute(delete(FlowRef))
        session.execute(delete(FlowObject))
    yield
    with pg_migrated.session() as session:
        session.execute(delete(FlowRef))
        session.execute(delete(FlowObject))


# ---------------------------------------------------------------------------
# Object store surface
# ---------------------------------------------------------------------------


class TestObjectStore:
    def test_put_get_round_trip(self, store: PgCheckpointStore) -> None:
        store.put_object("traj-1", "blob", "hash-a", b"payload-a")
        assert store.get_object("traj-1", "blob", "hash-a") == b"payload-a"

    def test_get_returns_none_when_absent(self, store: PgCheckpointStore) -> None:
        assert store.get_object("traj-1", "blob", "missing") is None

    def test_has_object_reflects_presence(self, store: PgCheckpointStore) -> None:
        assert not store.has_object("traj-1", "blob", "hash-a")
        store.put_object("traj-1", "blob", "hash-a", b"payload")
        assert store.has_object("traj-1", "blob", "hash-a")

    def test_put_idempotent_same_hash(self, store: PgCheckpointStore) -> None:
        """ON CONFLICT DO NOTHING — same PK re-put is a no-op."""
        store.put_object("traj-1", "blob", "hash-a", b"payload")
        store.put_object("traj-1", "blob", "hash-a", b"payload")
        assert store.get_object("traj-1", "blob", "hash-a") == b"payload"

    def test_kinds_do_not_collide(self, store: PgCheckpointStore) -> None:
        store.put_object("traj-1", "blob", "hash", b"blob-bytes")
        store.put_object("traj-1", "tree", "hash", b"tree-bytes")
        store.put_object("traj-1", "commit", "hash", b"commit-bytes")
        assert store.get_object("traj-1", "blob", "hash") == b"blob-bytes"
        assert store.get_object("traj-1", "tree", "hash") == b"tree-bytes"
        assert store.get_object("traj-1", "commit", "hash") == b"commit-bytes"

    def test_trajectories_do_not_collide(self, store: PgCheckpointStore) -> None:
        store.put_object("traj-a", "blob", "hash", b"a-bytes")
        store.put_object("traj-b", "blob", "hash", b"b-bytes")
        assert store.get_object("traj-a", "blob", "hash") == b"a-bytes"
        assert store.get_object("traj-b", "blob", "hash") == b"b-bytes"


# ---------------------------------------------------------------------------
# Ref store surface
# ---------------------------------------------------------------------------


class TestRefStore:
    def test_put_resolve_exact_key(self, store: PgCheckpointStore) -> None:
        store.put_ref("traj-1", "node/x", 5, "commit-hash-5")
        assert store.resolve_ref("traj-1", "node/x", 5) == "commit-hash-5"

    def test_resolve_returns_none_when_absent(self, store: PgCheckpointStore) -> None:
        assert store.resolve_ref("traj-1") is None

    def test_resolve_latest_across_node_paths(self, store: PgCheckpointStore) -> None:
        """resolve_ref with both None → latest by created_at."""
        store.put_ref("traj-1", "node/a", 1, "hash-1")
        store.put_ref("traj-1", "node/b", 1, "hash-2")
        assert store.resolve_ref("traj-1") == "hash-2"

    def test_resolve_latest_under_node_path(self, store: PgCheckpointStore) -> None:
        store.put_ref("traj-1", "node/x", 1, "hash-1")
        store.put_ref("traj-1", "node/x", 3, "hash-3")
        store.put_ref("traj-1", "node/x", 2, "hash-2")
        assert store.resolve_ref("traj-1", "node/x") == "hash-3"

    def test_resolve_iteration_without_node_path_raises(self, store: PgCheckpointStore) -> None:
        with pytest.raises(ValueError, match="iteration requires node_path"):
            store.resolve_ref("traj-1", None, 5)

    def test_put_ref_overwrites_same_key(self, store: PgCheckpointStore) -> None:
        """ON CONFLICT DO UPDATE — same PK re-put refreshes commit_hash + created_at."""
        store.put_ref("traj-1", "node/x", 5, "hash-first")
        store.put_ref("traj-1", "node/x", 5, "hash-second")
        assert store.resolve_ref("traj-1", "node/x", 5) == "hash-second"

    def test_refs_do_not_leak_across_trajectories(self, store: PgCheckpointStore) -> None:
        store.put_ref("traj-a", "node/x", 1, "hash-a")
        store.put_ref("traj-b", "node/x", 1, "hash-b")
        assert store.resolve_ref("traj-a", "node/x", 1) == "hash-a"
        assert store.resolve_ref("traj-b", "node/x", 1) == "hash-b"


# ---------------------------------------------------------------------------
# gc_trajectory
# ---------------------------------------------------------------------------


class TestGcTrajectory:
    def test_removes_all_objects_and_refs(self, store: PgCheckpointStore) -> None:
        store.put_object("traj-1", "blob", "h1", b"payload")
        store.put_ref("traj-1", "node/x", 1, "commit-h")
        store.gc_trajectory("traj-1")
        assert store.get_object("traj-1", "blob", "h1") is None
        assert store.resolve_ref("traj-1", "node/x", 1) is None

    def test_idempotent_when_absent(self, store: PgCheckpointStore) -> None:
        store.gc_trajectory("never-existed")

    def test_leaves_other_trajectories_intact(self, store: PgCheckpointStore) -> None:
        store.put_object("traj-a", "blob", "h", b"a-bytes")
        store.put_object("traj-b", "blob", "h", b"b-bytes")
        store.gc_trajectory("traj-a")
        assert store.get_object("traj-a", "blob", "h") is None
        assert store.get_object("traj-b", "blob", "h") == b"b-bytes"


# ---------------------------------------------------------------------------
# Retention policy
# ---------------------------------------------------------------------------


class TestRetention:
    def test_default_is_retain(self, pg_migrated: PG, pg_test_logger: Logger) -> None:
        s = PgCheckpointStore(pg_test_logger, pg_migrated)
        assert s.retention == "retain"

    def test_explicit_gc_on_success(self, pg_migrated: PG, pg_test_logger: Logger) -> None:
        s = PgCheckpointStore(pg_test_logger, pg_migrated, retention="gc_on_success")
        assert s.retention == "gc_on_success"
