# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Integration tests for :class:`llm_gent.flow.stores.PgCheckpointStore`.

Real Postgres round-trip against the migrated schema (name, object,
ref and tag tables). Protocol behaviour comes from
:class:`CheckpointStoreConformance`, shared with the file and in-memory
stores; this module adds the ordering of refs by database sequence.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from appinfra.db.pg import PG
from appinfra.log import Logger
from sqlalchemy import delete

from llm_gent.flow.stores import PgCheckpointStore
from llm_gent.flow.stores.postgres import FlowName, FlowObject, FlowRef, FlowTag
from tests.checkpoint_store_conformance import CheckpointStoreConformance


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


class TestConformance(CheckpointStoreConformance):
    """The Protocol behaviour every store shares."""


class TestRefOrdering:
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


class TestRetention:
    def test_explicit_gc_on_success(self, pg_migrated: PG, pg_test_logger: Logger) -> None:
        s = PgCheckpointStore(pg_test_logger, pg_migrated, retention="gc_on_success")
        assert s.retention == "gc_on_success"
