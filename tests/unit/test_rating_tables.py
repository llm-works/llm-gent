# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Rating SQL names kelt's tables as kelt's models do, schema-qualified when asked."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from appinfra.log import Logger
from llm_kelt.memory.atomic import Fact, FeedbackDetails

from llm_gent.core.memory.rating import AtomicFactsBackend
from llm_gent.core.memory.rating.batch import BatchRatingService
from llm_gent.core.memory.rating.models import ProviderType, Result


pytestmark = pytest.mark.unit

FACTS = Fact.__tablename__
FEEDBACK = FeedbackDetails.__tablename__


class _Rows:
    def __iter__(self) -> Iterator[Any]:
        return iter(())

    def scalar(self) -> int:
        return 0

    def fetchone(self) -> None:
        return None

    def fetchall(self) -> list[Any]:
        return []


class _Recorder:
    """Stands in for a kelt Database and an appinfra PG; keeps every statement's SQL."""

    def __init__(self) -> None:
        self.sql: list[str] = []

    def execute(self, statement: Any, params: Any = None) -> _Rows:
        self.sql.append(str(statement))
        return _Rows()

    def commit(self) -> None:
        pass

    @contextmanager
    def session(self) -> Iterator[_Recorder]:
        yield self

    connect = session


def _result() -> Result:
    return Result(
        fact=7,
        signal="positive",
        strength=0.8,
        stars=4,
        criteria_scores={},
        reasoning="r",
        model="m",
        provider="p",
        provider_type=ProviderType.LLM,
    )


def _backend_sql(lg: Logger, schema: str | None) -> list[str]:
    db: Any = _Recorder()
    backend = AtomicFactsBackend(lg, db, schema=schema)
    backend.save_rating(_result())
    list(backend.unrated_facts("ctx"))
    backend.get_unrated_count("ctx")
    backend.get_rated_unpaired_facts("ctx", "c", 1, 5)
    backend.get_fact_rating(7)
    backend.mark_facts_paired(1, 2, 3)
    return db.sql


def _batch_sql(lg: Logger, schema: str | None) -> list[str]:
    pg: Any = _Recorder()
    caller: Any = None
    service = BatchRatingService(
        lg, pg, llm_caller=caller, prompt_template="", context_key="ctx", schema=schema
    )
    service.get_unrated()
    service._save_rating(_result())
    return pg.sql


@pytest.mark.parametrize("run", [_backend_sql, _batch_sql])
class TestTableNames:
    def test_every_statement_names_kelts_tables(self, run: Any, test_logger: Logger) -> None:
        sql = run(test_logger, None)
        assert sql and all(FACTS in s or FEEDBACK in s for s in sql)
        assert not any("atomic_f" in s.replace(FACTS, "").replace(FEEDBACK, "") for s in sql)

    def test_a_schema_qualifies_them(self, run: Any, test_logger: Logger) -> None:
        sql = run(test_logger, "tenant")
        assert all(f"tenant.{FACTS}" in s or f"tenant.{FEEDBACK}" in s for s in sql)
