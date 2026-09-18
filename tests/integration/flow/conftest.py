# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Test fixtures for :mod:`llm_gent.flow` integration tests.

The top-level ``tests/conftest.py`` handles PG availability probing,
config resolution, and per-worker schema isolation. This file layers
llm-gent's alembic migrations on top of the isolated schema so the
Flow-level checkpoint table is present before any store hits the DB.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from appinfra.db.pg import PG
from appinfra.log import Logger

from llm_gent.schema import SchemaManager


@pytest.fixture(scope="session")
def pg_migrated(pg_isolated: PG, pg_test_logger: Logger) -> Generator[PG, None, None]:
    """A schema-isolated PG handle with llm-gent's alembic schema at head.

    The isolated schema is created + migrated once per test session;
    :func:`pg_isolated`'s teardown drops it.
    """
    SchemaManager(pg_test_logger, pg_isolated).ensure_schema()
    yield pg_isolated
