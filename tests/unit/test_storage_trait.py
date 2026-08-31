# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for StorageTrait."""

from unittest.mock import MagicMock, patch

import pytest

from llm_gent.core.traits.builtin.storage import StorageTrait


pytestmark = pytest.mark.unit


class TestStorageTrait:
    """Tests for StorageTrait lifecycle and functionality."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return MagicMock()

    @pytest.fixture
    def mock_agent(self, mock_logger):
        """Create a mock agent with logger."""
        agent = MagicMock()
        agent.lg = mock_logger
        return agent

    @pytest.fixture
    def mock_memory_trait(self):
        """Create a mock MemoryTrait with kelt client."""
        memory_trait = MagicMock()
        memory_trait.kelt = MagicMock()
        return memory_trait

    def test_init(self, mock_agent):
        """StorageTrait initializes with None storage."""
        trait = StorageTrait(mock_agent)

        assert trait.agent == mock_agent
        assert trait._storage is None

    def test_storage_property_raises_before_start(self, mock_agent):
        """Accessing storage before on_start raises RuntimeError."""
        trait = StorageTrait(mock_agent)

        with pytest.raises(RuntimeError, match="StorageTrait not started"):
            _ = trait.storage

    @patch("llm_gent.core.traits.builtin.storage.AgentStorage")
    def test_on_start_creates_storage(
        self, mock_agent_storage_class, mock_agent, mock_memory_trait
    ):
        """on_start creates AgentStorage from MemoryTrait's client."""
        # Setup agent to return mocked MemoryTrait
        mock_agent.require_trait.return_value = mock_memory_trait

        # Create mock AgentStorage instance
        mock_storage_instance = MagicMock()
        mock_agent_storage_class.return_value = mock_storage_instance

        trait = StorageTrait(mock_agent)
        trait.on_start()

        # Verify MemoryTrait was required
        from llm_gent.core.traits.builtin.memory import MemoryTrait

        mock_agent.require_trait.assert_called_once_with(MemoryTrait)

        # Verify AgentStorage was created with logger and kelt client
        mock_agent_storage_class.assert_called_once_with(mock_agent.lg, mock_memory_trait.kelt)

        # Verify storage was set
        assert trait._storage == mock_storage_instance

        # Verify debug log
        mock_agent.lg.debug.assert_called_with(
            "storage trait started", extra={"agent": mock_agent.name}
        )

    @patch("llm_gent.core.traits.builtin.storage.AgentStorage")
    def test_storage_property_returns_storage_after_start(
        self, mock_agent_storage_class, mock_agent, mock_memory_trait
    ):
        """storage property returns AgentStorage after on_start."""
        mock_agent.require_trait.return_value = mock_memory_trait
        mock_storage_instance = MagicMock()
        mock_agent_storage_class.return_value = mock_storage_instance

        trait = StorageTrait(mock_agent)
        trait.on_start()

        # Access storage property
        storage = trait.storage

        assert storage == mock_storage_instance

    @patch("llm_gent.core.traits.builtin.storage.AgentStorage")
    def test_on_stop_clears_storage(self, mock_agent_storage_class, mock_agent, mock_memory_trait):
        """on_stop clears storage reference."""
        mock_agent.require_trait.return_value = mock_memory_trait
        mock_storage_instance = MagicMock()
        mock_agent_storage_class.return_value = mock_storage_instance

        trait = StorageTrait(mock_agent)
        trait.on_start()

        # Verify storage exists
        assert trait._storage is not None

        # Stop trait
        trait.on_stop()

        # Verify storage is cleared
        assert trait._storage is None

        # Verify debug log
        assert mock_agent.lg.debug.call_count == 2
        mock_agent.lg.debug.assert_any_call(
            "storage trait stopped", extra={"agent": mock_agent.name}
        )

    @patch("llm_gent.core.traits.builtin.storage.AgentStorage")
    def test_storage_property_raises_after_stop(
        self, mock_agent_storage_class, mock_agent, mock_memory_trait
    ):
        """storage property raises after on_stop."""
        mock_agent.require_trait.return_value = mock_memory_trait
        mock_storage_instance = MagicMock()
        mock_agent_storage_class.return_value = mock_storage_instance

        trait = StorageTrait(mock_agent)
        trait.on_start()
        trait.on_stop()

        # Accessing storage after stop should raise
        with pytest.raises(RuntimeError, match="StorageTrait not started"):
            _ = trait.storage

    def test_on_start_requires_memory_trait(self, mock_agent):
        """on_start raises if MemoryTrait is not attached."""
        from llm_gent.core.errors import TraitNotFoundError

        # Make require_trait raise TraitNotFoundError
        mock_agent.require_trait.side_effect = TraitNotFoundError("MemoryTrait not found")

        trait = StorageTrait(mock_agent)

        with pytest.raises(TraitNotFoundError, match="MemoryTrait not found"):
            trait.on_start()
