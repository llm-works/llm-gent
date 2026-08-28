# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for ConversationTrait."""

from unittest.mock import MagicMock

import pytest

from llm_gent.core.traits.builtin.conversation import ConversationTrait, ConversationTraitConfig


pytestmark = pytest.mark.unit


class TestConversationTraitConfig:
    """Tests for ConversationTraitConfig defaults."""

    def test_defaults(self):
        cfg = ConversationTraitConfig()
        assert cfg.max_tokens == 32000
        assert cfg.compact_threshold == 0.8
        assert cfg.preserve_system is True
        assert cfg.min_recent_messages == 4
        assert cfg.compactor == "sliding_window"

    def test_custom(self):
        cfg = ConversationTraitConfig(max_tokens=8000, compactor="sliding_window")
        assert cfg.max_tokens == 8000


class TestConversationTrait:
    """Tests for ConversationTrait."""

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.get_trait.return_value = None
        return agent

    def test_init_default_config(self, mock_agent):
        trait = ConversationTrait(mock_agent)
        assert trait.config.compactor == "sliding_window"
        assert trait.conversation is not None

    def test_init_custom_config(self, mock_agent):
        cfg = ConversationTraitConfig(max_tokens=4000)
        trait = ConversationTrait(mock_agent, config=cfg)
        assert trait.config.max_tokens == 4000

    def test_conversation_property(self, mock_agent):
        trait = ConversationTrait(mock_agent)
        assert trait.conversation is not None

    def test_create_compactor_sliding_window(self, mock_agent):
        """sliding_window is the default and works."""
        trait = ConversationTrait(mock_agent)
        # If it constructed without error, sliding_window worked
        assert trait.conversation is not None

    def test_create_compactor_summarizing_raises(self, mock_agent):
        cfg = ConversationTraitConfig(compactor="summarizing")
        with pytest.raises(NotImplementedError, match="SummarizingCompactor"):
            ConversationTrait(mock_agent, config=cfg)

    def test_create_compactor_unknown_raises(self, mock_agent):
        cfg = ConversationTraitConfig(compactor="nonexistent")
        with pytest.raises(ValueError, match="Unknown compactor"):
            ConversationTrait(mock_agent, config=cfg)

    def test_on_start_with_system_prompt(self, mock_agent):
        """on_start adds system prompt from SAIATrait if present."""
        saia_mock = MagicMock()
        saia_mock.config.system_prompt = "You are helpful."
        mock_agent.get_trait.return_value = saia_mock

        trait = ConversationTrait(mock_agent)
        trait.on_start()

        messages = trait.get_context()
        assert len(messages) == 1
        assert messages[0].content == "You are helpful."

    def test_on_start_no_saia_trait(self, mock_agent):
        """on_start does nothing if no SAIATrait."""
        mock_agent.get_trait.return_value = None
        trait = ConversationTrait(mock_agent)
        trait.on_start()
        assert len(trait.get_context()) == 0

    def test_on_start_empty_system_prompt(self, mock_agent):
        """on_start skips if system_prompt is empty."""
        saia_mock = MagicMock()
        saia_mock.config.system_prompt = ""
        mock_agent.get_trait.return_value = saia_mock

        trait = ConversationTrait(mock_agent)
        trait.on_start()
        assert len(trait.get_context()) == 0

    def test_on_stop(self, mock_agent):
        """on_stop is a no-op."""
        trait = ConversationTrait(mock_agent)
        trait.on_stop()  # Should not raise

    def test_get_context_empty(self, mock_agent):
        trait = ConversationTrait(mock_agent)
        assert trait.get_context() == []

    def test_add_turn(self, mock_agent):
        trait = ConversationTrait(mock_agent)
        trait.add_turn("Hello", "Hi there!")
        messages = trait.get_context()
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi there!"

    def test_add_multiple_turns(self, mock_agent):
        trait = ConversationTrait(mock_agent)
        trait.add_turn("Q1", "A1")
        trait.add_turn("Q2", "A2")
        assert len(trait.get_context()) == 4

    def test_reset_clears(self, mock_agent):
        mock_agent.get_trait.return_value = None
        trait = ConversationTrait(mock_agent)
        trait.add_turn("Q", "A")
        assert len(trait.get_context()) == 2

        trait.reset()
        assert len(trait.get_context()) == 0

    def test_reset_readds_system_prompt(self, mock_agent):
        """reset re-adds system prompt when preserve_system=True."""
        saia_mock = MagicMock()
        saia_mock.config.system_prompt = "System prompt."
        mock_agent.get_trait.return_value = saia_mock

        trait = ConversationTrait(mock_agent)
        trait.on_start()
        trait.add_turn("Q", "A")
        assert len(trait.get_context()) == 3

        trait.reset()
        messages = trait.get_context()
        assert len(messages) == 1
        assert messages[0].content == "System prompt."

    def test_reset_no_system_prompt_when_preserve_false(self, mock_agent):
        """reset doesn't re-add system prompt when preserve_system=False."""
        saia_mock = MagicMock()
        saia_mock.config.system_prompt = "System prompt."
        mock_agent.get_trait.return_value = saia_mock

        cfg = ConversationTraitConfig(preserve_system=False)
        trait = ConversationTrait(mock_agent, config=cfg)
        trait.on_start()
        trait.add_turn("Q", "A")

        trait.reset()
        assert len(trait.get_context()) == 0
