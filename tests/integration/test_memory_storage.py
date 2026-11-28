# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""An agent's own tables, through its memory binding, on a fresh Postgres schema."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from appinfra import DotDict
from appinfra.db.pg import PG
from appinfra.log import Logger
from llm_kelt.core import Database
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from llm_gent.core.agent import Identity
from llm_gent.core.traits.builtin.memory import MemoryTrait
from llm_gent.storage import AgentTable


pytestmark = pytest.mark.integration


class _JokeTable(AgentTable):
    __tablename__ = "agent_test_storage_jokes"

    text: Mapped[str] = mapped_column(String, nullable=False)


def _agent(lg: Logger, context_key: str) -> MagicMock:
    return MagicMock(lg=lg, identity=Identity(context_key, context_key=context_key))


def test_each_agent_reads_only_its_own_rows(pg_isolated: PG, pg_test_logger: Logger) -> None:
    """Two agents share one table on the platform schema; each sees only the rows it wrote."""
    database = Database(pg_test_logger, pg_isolated)
    memory = MemoryTrait(pg_test_logger, DotDict({}), database=database, chat_client=MagicMock())
    jokes = memory.bind(_agent(pg_test_logger, "jokes")).storage
    puns = memory.bind(_agent(pg_test_logger, "puns")).storage

    jokes.register_table(_JokeTable)
    puns.register_table(_JokeTable)  # the table exists: registering again creates nothing
    jokes.insert(_JokeTable, text="Why did the chicken cross the road?")
    puns.insert(_JokeTable, text="I used to be a banker, but I lost interest.")

    assert [row.text for row in jokes.select(_JokeTable)] == ["Why did the chicken cross the road?"]
    assert [row.context_key for row in puns.select(_JokeTable)] == ["puns"]
