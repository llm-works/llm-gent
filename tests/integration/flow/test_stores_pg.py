# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Integration tests for :class:`llm_gent.flow.stores.PgCheckpointStore`.

Real Postgres round-trip against the migrated schema (name, object,
ref and tag tables). Same surface as :mod:`tests.unit.flow.test_stores_json` — the
two implementations should behave identically at the Protocol level.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from appinfra.db.pg import PG
from appinfra.log import Logger
from sqlalchemy import delete

from llm_gent.flow.stores import PgCheckpointStore
from llm_gent.flow.stores.postgres import FlowName, FlowObject, FlowRef, FlowTag


@pytest.fixture
def store(pg_migrated: PG, pg_test_logger: Logger) -> PgCheckpointStore:
    """PgCheckpointStore bound to the session's migrated schema."""
    return PgCheckpointStore(pg_test_logger, pg_migrated)


def _wipe(pg: PG) -> None:
    """Delete every row from the store's tables."""
    with pg.session() as session:
        for model in (FlowTag, FlowRef, FlowObject, FlowName):
            session.execute(delete(model))


@pytest.fixture(autouse=True)
def clean_tables(pg_migrated: PG) -> Generator[None, None, None]:
    """Wipe the store's tables before and after every test.

    Fixtures run in module scope so state leaks between tests without
    this. Autouse keeps every test independent.
    """
    _wipe(pg_migrated)
    yield
    _wipe(pg_migrated)


# ---------------------------------------------------------------------------
# Object store surface
# ---------------------------------------------------------------------------


class TestObjectStore:
    def test_put_get_round_trip(self, store: PgCheckpointStore) -> None:
        store.put_object("history-1", "blob", "hash-a", b"payload-a")
        assert store.get_object("history-1", "blob", "hash-a") == b"payload-a"

    def test_get_returns_none_when_absent(self, store: PgCheckpointStore) -> None:
        assert store.get_object("history-1", "blob", "missing") is None

    def test_has_object_reflects_presence(self, store: PgCheckpointStore) -> None:
        assert not store.has_object("history-1", "blob", "hash-a")
        store.put_object("history-1", "blob", "hash-a", b"payload")
        assert store.has_object("history-1", "blob", "hash-a")

    def test_put_idempotent_same_hash(self, store: PgCheckpointStore) -> None:
        """ON CONFLICT DO NOTHING — same PK re-put is a no-op."""
        store.put_object("history-1", "blob", "hash-a", b"payload")
        store.put_object("history-1", "blob", "hash-a", b"payload")
        assert store.get_object("history-1", "blob", "hash-a") == b"payload"

    def test_kinds_do_not_collide(self, store: PgCheckpointStore) -> None:
        store.put_object("history-1", "blob", "hash", b"blob-bytes")
        store.put_object("history-1", "tree", "hash", b"tree-bytes")
        store.put_object("history-1", "commit", "hash", b"commit-bytes")
        assert store.get_object("history-1", "blob", "hash") == b"blob-bytes"
        assert store.get_object("history-1", "tree", "hash") == b"tree-bytes"
        assert store.get_object("history-1", "commit", "hash") == b"commit-bytes"

    def test_histories_do_not_collide(self, store: PgCheckpointStore) -> None:
        store.put_object("history-a", "blob", "hash", b"a-bytes")
        store.put_object("history-b", "blob", "hash", b"b-bytes")
        assert store.get_object("history-a", "blob", "hash") == b"a-bytes"
        assert store.get_object("history-b", "blob", "hash") == b"b-bytes"


# ---------------------------------------------------------------------------
# Ref store surface
# ---------------------------------------------------------------------------


class TestRefStore:
    def test_put_resolve_exact_key(self, store: PgCheckpointStore) -> None:
        store.put_ref("history-1", "node/x", 5, "commit-hash-5")
        assert store.resolve_ref("history-1", "node/x", 5) == "commit-hash-5"

    def test_resolve_returns_none_when_absent(self, store: PgCheckpointStore) -> None:
        assert store.resolve_ref("history-1") is None

    def test_resolve_latest_across_node_paths(self, store: PgCheckpointStore) -> None:
        """resolve_ref with both None → latest by database sequence."""
        store.put_ref("history-1", "node/a", 1, "hash-1")
        store.put_ref("history-1", "node/b", 1, "hash-2")
        assert store.resolve_ref("history-1") == "hash-2"

    def test_latest_ignores_writer_clock(self, store: PgCheckpointStore, pg_migrated: PG) -> None:
        """A ref written later wins even if its writer's clock stamped an earlier created_at."""
        from datetime import UTC, datetime, timedelta

        from sqlalchemy import update

        store.put_ref("history-1", "node/a", 1, "hash-first")
        store.put_ref("history-1", "node/b", 1, "hash-second")
        # Simulate the second writer's clock lagging by an hour.
        with pg_migrated.session() as session:
            session.execute(
                update(FlowRef)
                .where(FlowRef.commit_hash == "hash-second")
                .values(created_at=datetime.now(UTC) - timedelta(hours=1))
            )
        assert store.resolve_ref("history-1") == "hash-second"

    def test_re_put_becomes_latest(self, store: PgCheckpointStore) -> None:
        """Re-putting an existing key draws a new seq, so it becomes the head again."""
        store.put_ref("history-1", "node/a", 1, "hash-a")
        store.put_ref("history-1", "node/b", 1, "hash-b")
        store.put_ref("history-1", "node/a", 1, "hash-a2")
        assert store.resolve_ref("history-1") == "hash-a2"

    def test_resolve_latest_under_node_path(self, store: PgCheckpointStore) -> None:
        store.put_ref("history-1", "node/x", 1, "hash-1")
        store.put_ref("history-1", "node/x", 3, "hash-3")
        store.put_ref("history-1", "node/x", 2, "hash-2")
        assert store.resolve_ref("history-1", "node/x") == "hash-3"

    def test_resolve_iteration_without_node_path_raises(self, store: PgCheckpointStore) -> None:
        with pytest.raises(ValueError, match="iteration requires node_path"):
            store.resolve_ref("history-1", None, 5)

    def test_put_ref_overwrites_same_key(self, store: PgCheckpointStore) -> None:
        """ON CONFLICT DO UPDATE — same PK re-put refreshes commit_hash, seq and created_at."""
        store.put_ref("history-1", "node/x", 5, "hash-first")
        store.put_ref("history-1", "node/x", 5, "hash-second")
        assert store.resolve_ref("history-1", "node/x", 5) == "hash-second"

    def test_refs_do_not_leak_across_histories(self, store: PgCheckpointStore) -> None:
        store.put_ref("history-a", "node/x", 1, "hash-a")
        store.put_ref("history-b", "node/x", 1, "hash-b")
        assert store.resolve_ref("history-a", "node/x", 1) == "hash-a"
        assert store.resolve_ref("history-b", "node/x", 1) == "hash-b"


# ---------------------------------------------------------------------------
# Name map
# ---------------------------------------------------------------------------


class TestNameMap:
    def test_bind_get_round_trip(self, store: PgCheckpointStore) -> None:
        assert store.bind_flow_id("campaign-1", "history-1") == "history-1"
        assert store.get_flow_id("campaign-1") == "history-1"

    def test_get_returns_none_when_unbound(self, store: PgCheckpointStore) -> None:
        assert store.get_flow_id("campaign-1") is None

    def test_second_bind_returns_existing(self, store: PgCheckpointStore) -> None:
        """ON CONFLICT DO NOTHING on client_flow_id — the first binding wins."""
        store.bind_flow_id("campaign-1", "history-1")
        assert store.bind_flow_id("campaign-1", "history-2") == "history-1"
        assert store.get_flow_id("campaign-1") == "history-1"

    def test_flow_id_cannot_name_two_histories(self, store: PgCheckpointStore) -> None:
        store.bind_flow_id("campaign-1", "history-1")
        with pytest.raises(ValueError, match="already names"):
            store.bind_flow_id("campaign-2", "history-1")

    def test_concurrent_binds_agree_on_one_winner(self, store: PgCheckpointStore) -> None:
        """Racing binds with distinct ids, each on its own connection, all return one id."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(lambda i: store.bind_flow_id("campaign-1", f"history-{i}"), range(16))
            )
        assert len(set(results)) == 1
        assert store.get_flow_id("campaign-1") == results[0]


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


class TestTags:
    def test_put_resolve_round_trip(self, store: PgCheckpointStore) -> None:
        store.put_tag("history-1", "complete", "commit-h")
        assert store.resolve_tag("history-1", "complete") == "commit-h"

    def test_resolve_returns_none_when_absent(self, store: PgCheckpointStore) -> None:
        assert store.resolve_tag("history-1", "complete") is None

    def test_re_put_moves_the_tag(self, store: PgCheckpointStore) -> None:
        """ON CONFLICT DO UPDATE — same (flow_id, name) re-put moves the tag."""
        store.put_tag("history-1", "complete", "commit-1")
        store.put_tag("history-1", "complete", "commit-2")
        assert store.resolve_tag("history-1", "complete") == "commit-2"

    def test_tags_do_not_leak_across_histories(self, store: PgCheckpointStore) -> None:
        store.put_tag("history-a", "complete", "hash-a")
        assert store.resolve_tag("history-b", "complete") is None


# ---------------------------------------------------------------------------
# gc_history
# ---------------------------------------------------------------------------


class TestGcHistory:
    def test_removes_all_objects_and_refs(self, store: PgCheckpointStore) -> None:
        store.put_object("history-1", "blob", "h1", b"payload")
        store.put_ref("history-1", "node/x", 1, "commit-h")
        store.gc_history("history-1")
        assert store.get_object("history-1", "blob", "h1") is None
        assert store.resolve_ref("history-1", "node/x", 1) is None

    def test_removes_tags_and_name_binding(self, store: PgCheckpointStore) -> None:
        store.bind_flow_id("campaign-1", "history-1")
        store.put_tag("history-1", "complete", "commit-h")
        store.gc_history("history-1")
        assert store.resolve_tag("history-1", "complete") is None
        assert store.get_flow_id("campaign-1") is None
        store.bind_flow_id("campaign-1", "history-2")  # name is free again
        assert store.get_flow_id("campaign-1") == "history-2"

    def test_idempotent_when_absent(self, store: PgCheckpointStore) -> None:
        store.gc_history("never-existed")

    def test_leaves_other_histories_intact(self, store: PgCheckpointStore) -> None:
        store.put_object("history-a", "blob", "h", b"a-bytes")
        store.put_object("history-b", "blob", "h", b"b-bytes")
        store.gc_history("history-a")
        assert store.get_object("history-a", "blob", "h") is None
        assert store.get_object("history-b", "blob", "h") == b"b-bytes"


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
