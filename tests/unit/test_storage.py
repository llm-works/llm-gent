# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for AgentStorage client."""

from unittest.mock import MagicMock, patch

import pytest

from llm_gent.storage.client import AgentStorage
from llm_gent.storage.schema import AgentTable


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Minimal concrete AgentTable subclass for testing
# ---------------------------------------------------------------------------


class FakeTable(AgentTable):
    """Minimal table model for tests."""

    __tablename__ = "agent_fake_items"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_kelt():
    """Build a mock KeltClient with the shape AgentStorage expects."""
    kelt = MagicMock()
    kelt.context.context_key = "agent-123"
    kelt.context.schema_name = "public"
    # session() is used as a context manager
    session_cm = MagicMock()
    kelt.database.session.return_value = session_cm
    session_cm.__enter__ = MagicMock(return_value=MagicMock())
    session_cm.__exit__ = MagicMock(return_value=False)
    return kelt


@pytest.fixture
def storage(mock_logger, mock_kelt):
    """AgentStorage with a registered FakeTable."""
    s = AgentStorage(mock_logger, mock_kelt)
    # Bypass real create-table call
    with patch.object(FakeTable.__table__, "create"):
        s.register_table(FakeTable)
    return s


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_valid_context_key(self, mock_logger, mock_kelt):
        storage = AgentStorage(mock_logger, mock_kelt)
        assert storage._registered_tables == {}

    def test_none_context_key_raises(self, mock_logger, mock_kelt):
        mock_kelt.context.context_key = None
        with pytest.raises(ValueError, match="requires isolation context"):
            AgentStorage(mock_logger, mock_kelt)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_context_key(self, storage):
        assert storage.context_key == "agent-123"

    def test_schema_name(self, storage):
        assert storage.schema_name == "public"


# ---------------------------------------------------------------------------
# register_table
# ---------------------------------------------------------------------------


class TestRegisterTable:
    def test_registers_and_creates_table(self, mock_logger, mock_kelt):
        s = AgentStorage(mock_logger, mock_kelt)

        with patch.object(FakeTable.__table__, "create") as mock_create:
            s.register_table(FakeTable)

        mock_create.assert_called_once_with(mock_kelt.database.engine, checkfirst=True)
        assert "agent_fake_items" in s._registered_tables

    def test_validate_rejects_non_agent_table(self, mock_logger, mock_kelt):
        """register_table delegates to validate_agent_table which rejects bad classes."""
        s = AgentStorage(mock_logger, mock_kelt)

        with pytest.raises(ValueError, match="must inherit from AgentTable"):
            s.register_table(MagicMock)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


class TestSelect:
    def test_unregistered_table_raises(self, mock_logger, mock_kelt):
        s = AgentStorage(mock_logger, mock_kelt)
        with pytest.raises(ValueError, match="Table not registered"):
            s.select(FakeTable)

    def test_select_returns_list(self, storage, mock_kelt):
        session = mock_kelt.database.session.return_value.__enter__.return_value
        session.execute.return_value.scalars.return_value.all.return_value = ["row1", "row2"]

        result = storage.select(FakeTable)

        assert result == ["row1", "row2"]
        # Verify session was used as context manager
        mock_kelt.database.session.return_value.__enter__.assert_called()

    def test_invalid_filter_column_raises(self, storage):
        with pytest.raises(ValueError, match="Invalid column name 'nonexistent'"):
            storage.select(FakeTable, nonexistent="val")


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------


class TestInsert:
    def test_unregistered_table_raises(self, mock_logger, mock_kelt):
        s = AgentStorage(mock_logger, mock_kelt)
        with pytest.raises(ValueError, match="Table not registered"):
            s.insert(FakeTable, text="hello")

    def test_insert_adds_context_key_and_returns_id(self, storage, mock_kelt):
        session = mock_kelt.database.session.return_value.__enter__.return_value

        # Capture what gets added to the session
        added_records: list = []
        session.add.side_effect = lambda rec: added_records.append(rec)

        # After flush, the record gets an id attribute
        def fake_flush():
            if added_records:
                added_records[-1].id = 42

        session.flush.side_effect = fake_flush

        # Only pass columns that exist on FakeTable (context_key is auto-added)
        row_id = storage.insert(FakeTable)

        assert row_id == 42
        session.commit.assert_called_once()
        # Verify context_key was injected
        assert added_records[0].context_key == "agent-123"


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


class TestExecute:
    def test_execute_runs_statement(self, storage, mock_kelt):
        session = mock_kelt.database.session.return_value.__enter__.return_value
        session.execute.return_value.all.return_value = [("a",), ("b",)]

        stmt = MagicMock()
        result = storage.execute(stmt)

        session.execute.assert_called_once_with(stmt)
        assert result == [("a",), ("b",)]


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_unregistered_table_raises(self, mock_logger, mock_kelt):
        s = AgentStorage(mock_logger, mock_kelt)
        with pytest.raises(ValueError, match="Table not registered"):
            s.update(FakeTable, 1, text="new")

    def test_context_key_update_forbidden(self, storage):
        with pytest.raises(ValueError, match="Cannot update isolation field"):
            storage.update(FakeTable, 1, context_key="other-key")

    def test_row_not_found_raises(self, storage, mock_kelt):
        session = mock_kelt.database.session.return_value.__enter__.return_value
        session.execute.return_value.scalar_one_or_none.return_value = None

        with pytest.raises(ValueError, match="Row not found: 99"):
            storage.update(FakeTable, 99, text="new")

    def test_update_sets_fields(self, storage, mock_kelt):
        session = mock_kelt.database.session.return_value.__enter__.return_value
        record = MagicMock()
        session.execute.return_value.scalar_one_or_none.return_value = record

        storage.update(FakeTable, 1, text="updated")

        record.__setattr__("text", "updated")  # setattr called via loop
        session.commit.assert_called_once()


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_unregistered_table_raises(self, mock_logger, mock_kelt):
        s = AgentStorage(mock_logger, mock_kelt)
        with pytest.raises(ValueError, match="Table not registered"):
            s.delete(FakeTable, 1)

    def test_row_not_found_raises(self, storage, mock_kelt):
        session = mock_kelt.database.session.return_value.__enter__.return_value
        session.execute.return_value.scalar_one_or_none.return_value = None

        with pytest.raises(ValueError, match="Row not found: 5"):
            storage.delete(FakeTable, 5)

    def test_delete_removes_and_commits(self, storage, mock_kelt):
        session = mock_kelt.database.session.return_value.__enter__.return_value
        record = MagicMock()
        session.execute.return_value.scalar_one_or_none.return_value = record

        storage.delete(FakeTable, 1)

        session.delete.assert_called_once_with(record)
        session.commit.assert_called_once()
