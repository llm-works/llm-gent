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
from appinfra.cas.conformance import StoreConformance
from appinfra.db.pg import PG
from appinfra.log import Logger
from sqlalchemy import delete, update

from llm_gent.flow import History
from llm_gent.flow.stores import PgStore
from llm_gent.flow.stores.postgres import CasObject, CasRef
from llm_gent.flow.testing.checkpoint import build_canonical_flow


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


class TestCheckpointedRun:
    async def test_halt_and_resume_on_postgres(
        self, store: PgStore, pg_test_logger: Logger
    ) -> None:
        """A run halts into Postgres and a new flow resumes it to the uninterrupted result."""
        baseline = await build_canonical_flow(pg_test_logger, max_iters=4).run()

        common: dict[str, Any] = {"max_iters": 4, "store": store, "client_flow_id": "pg-run"}
        await build_canonical_flow(
            pg_test_logger, halt=asyncio.Event(), halt_after_iteration=2, **common
        ).run()
        head = await History(store, "pg-run").head()
        assert head is not None and head.meta.outcome == "halted"

        resumed = await build_canonical_flow(pg_test_logger, **common).run(resume="latest")
        assert resumed == baseline
        assert await History(store, "pg-run").is_complete()
