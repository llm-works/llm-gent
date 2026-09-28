# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Integration tests for :class:`llm_gent.flow.stores.PgCheckpointStore`.

Real Postgres round-trip against the migrated schema (name, object and
ref tables). Protocol behaviour, including compare-and-set on refs from
concurrent threads, comes from :class:`CheckpointStoreConformance`,
shared with the file and in-memory stores.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from appinfra.db.pg import PG
from appinfra.log import Logger
from sqlalchemy import delete

from llm_gent.flow.stores import PgCheckpointStore
from llm_gent.flow.stores.postgres import FlowName, FlowObject, FlowRef
from tests.checkpoint_store_conformance import CheckpointStoreConformance


@pytest.fixture
def store(pg_migrated: PG, pg_test_logger: Logger) -> PgCheckpointStore:
    """PgCheckpointStore bound to the session's migrated schema."""
    return PgCheckpointStore(pg_test_logger, pg_migrated)


def _wipe(pg: PG) -> None:
    """Delete every row from the store's tables."""
    with pg.session() as session:
        for model in (FlowRef, FlowObject, FlowName):
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


class TestRetention:
    def test_explicit_gc_on_success(self, pg_migrated: PG, pg_test_logger: Logger) -> None:
        s = PgCheckpointStore(pg_test_logger, pg_migrated, retention="gc_on_success")
        assert s.retention == "gc_on_success"
