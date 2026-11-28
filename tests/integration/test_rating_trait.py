# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Rating on a fresh database: the platform prepares memory's schema before any agent rates."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from appinfra import DotDict
from appinfra.db.pg import PG
from appinfra.log import Logger
from llm_kelt.core import Database

from llm_gent.core.traits.builtin.memory import MemoryTrait
from llm_gent.core.traits.builtin.rating import RatingTrait


pytestmark = pytest.mark.integration


def _agent(lg: Logger, context_key: str) -> MagicMock:
    agent = MagicMock()
    agent.lg = lg
    agent.identity.context_key = context_key
    agent.config = DotDict({})
    return agent


def test_rating_queries_memory_tables_before_any_memory_access(
    pg_isolated: PG, pg_test_logger: Logger
) -> None:
    """Rating's SQL reads memory's tables; the memory trait created them when it was built."""
    database = Database(pg_test_logger, pg_isolated)
    memory = MemoryTrait(pg_test_logger, DotDict({}), database=database, chat_client=MagicMock())
    rating = RatingTrait(pg_test_logger, MagicMock(), memory)

    binding = rating.bind(_agent(pg_test_logger, "jokes"))

    assert binding.get_unrated_count() == 0
    assert binding.get_fact_rating(1) is None


def test_rating_counts_only_its_agents_unrated_facts(
    pg_isolated: PG, pg_test_logger: Logger
) -> None:
    """A fact one agent remembers is unrated for that agent and absent for another."""
    database = Database(pg_test_logger, pg_isolated)
    memory = MemoryTrait(pg_test_logger, DotDict({}), database=database, chat_client=MagicMock())
    rating = RatingTrait(pg_test_logger, MagicMock(), memory)
    jokes, forecasts = _agent(pg_test_logger, "jokes"), _agent(pg_test_logger, "forecasts")

    memory.bind(jokes).remember("Why did the chicken cross the road?", category="joke")

    assert rating.bind(jokes).get_unrated_count() == 1
    assert rating.bind(jokes).get_unrated_count(category="joke") == 1
    assert rating.bind(forecasts).get_unrated_count() == 0
