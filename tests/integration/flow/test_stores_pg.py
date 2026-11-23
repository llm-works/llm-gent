# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Integration tests for :class:`llm_gent.flow.stores.PgStore`.

Real Postgres against the migrated schema: :mod:`appinfra.cas`'s
:class:`~appinfra.cas.conformance.StoreConformance`, the behaviour every
store shares (compare-and-set refs and locks across threads included), and
a checkpointed run end to end.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator
from datetime import timedelta
from typing import Any

import pytest
from appinfra.cas import Blob, ObjectType
from appinfra.cas.conformance import StoreConformance
from appinfra.db.pg import PG
from appinfra.log import Logger
from sqlalchemy import delete, update

from llm_gent.flow import History, Resume
from llm_gent.flow.stores import PgStore
from llm_gent.flow.stores.postgres import _WRITE_BATCH, CasObject, CasRef
from llm_gent.flow.testing.checkpoint import build_canonical_flow
from tests.helpers import run_flow


def _wipe(pg: PG) -> None:
    """Delete every row from the store's tables."""
    with pg.session() as session:
        for model in (CasRef, CasObject):
            session.execute(delete(model))


@pytest.fixture
def store(pg_migrated: PG, pg_test_logger: Logger) -> Generator[PgStore, None, None]:
    """An empty PgStore bound to the session's migrated schema; wiped after the test too."""
    _wipe(pg_migrated)
    yield PgStore(pg_test_logger, pg_migrated)
    _wipe(pg_migrated)


@pytest.fixture
def age(pg_migrated: PG) -> Callable[[str, str, float], None]:
    """Move an object's ``written_at`` ``seconds`` into the past."""

    def age(repo: str, object_id: str, seconds: float) -> None:
        stmt = (
            update(CasObject)
            .where(CasObject.repo == repo, CasObject.id == object_id)
            .values(written_at=CasObject.written_at - timedelta(seconds=seconds))
        )
        with pg_migrated.session() as session:
            session.execute(stmt)

    return age


class TestConformance(StoreConformance):
    """The behaviour every :class:`appinfra.cas.Store` shares."""


def _blobs(n: int) -> list[tuple[str, ObjectType, bytes]]:
    """``n`` distinct blobs as ``write_objects`` takes them."""
    blobs = [Blob(f"blob {i}".encode()) for i in range(n)]
    return [(b.id, b.type, b.serialize()) for b in blobs]


class TestBulkWrites:
    """What the bulk overrides do beyond the conformance suite: one statement's limits."""

    def test_an_id_listed_twice_is_written_once(self, store: PgStore) -> None:
        """A repeated id in one call does not hit ON CONFLICT twice in one statement."""
        objects = _blobs(2)
        store.write_objects("r", [*objects, objects[0]])
        assert set(store.list_objects("r")) == {oid for oid, _, _ in objects}

    def test_more_objects_than_one_statement_holds(self, store: PgStore) -> None:
        """A call over the per-statement row count stores every object."""
        objects = _blobs(_WRITE_BATCH + 1)
        store.write_objects("r", objects)
        ids = [oid for oid, _, _ in objects]
        assert set(store.list_objects("r")) == set(ids)
        assert store.object_types("r", ids) == dict.fromkeys(ids, "blob")

    def test_rewriting_refreshes_the_write_time(
        self, store: PgStore, age: Callable[[str, str, float], None]
    ) -> None:
        """Writing a stored object again makes it recent, as write_object does."""
        objects = _blobs(1)
        store.write_objects("r", objects)
        oid = objects[0][0]
        age("r", oid, 3600)
        store.write_objects("r", objects)
        assert store.now() - store.list_objects("r")[oid] < 60


class TestCheckpointedRun:
    async def test_halt_and_resume_on_postgres(
        self, store: PgStore, pg_test_logger: Logger
    ) -> None:
        """A run halts into Postgres and a new flow resumes it to the uninterrupted result."""
        baseline = await run_flow(build_canonical_flow(pg_test_logger, max_iters=4))

        common: dict[str, Any] = {"max_iters": 4, "checkpointed": True}
        await run_flow(
            build_canonical_flow(pg_test_logger, halt_after_iteration=2, **common),
            halt=asyncio.Event(),
            history="pg-run",
            store=store,
        )
        head = await History(store, "pg-run").head()
        assert head is not None and head.meta.outcome == "halted"

        resumed = await run_flow(
            build_canonical_flow(pg_test_logger, **common),
            history="pg-run",
            start=Resume("latest"),
            store=store,
        )
        assert resumed == baseline
        assert await History(store, "pg-run").is_complete()
