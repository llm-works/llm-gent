# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for traits."""

from unittest.mock import MagicMock

import pytest

from llm_gent import (
    BaseTrait,
    Directive,
    DirectiveTrait,
    MethodTrait,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    return MagicMock()


class TestBaseTrait:
    """Tests for BaseTrait."""

    def test_init(self, mock_agent):
        """BaseTrait requires agent in constructor."""
        trait = BaseTrait(mock_agent)

        assert trait._agent == mock_agent

    def test_agent_property_returns_agent(self, mock_agent):
        """Agent property returns the agent passed in constructor."""
        trait = BaseTrait(mock_agent)

        assert trait.agent == mock_agent

    def test_lifecycle_methods_exist(self, mock_agent):
        """Lifecycle methods can be called - no-op by default."""
        trait = BaseTrait(mock_agent)

        # Should not raise - these are no-op by default
        trait.on_start()
        trait.on_stop()


class TestCustomTrait:
    """Tests for custom traits using BaseTrait."""

    def test_custom_trait_inherits_behavior(self):
        """Custom traits can extend BaseTrait."""

        class MyTrait(BaseTrait):
            def __init__(self, agent, value: str) -> None:
                super().__init__(agent)
                self.value = value

            def get_value(self) -> str:
                return f"{self.agent.name}: {self.value}"

        mock_agent = MagicMock()
        mock_agent.name = "test-agent"
        trait = MyTrait(mock_agent, "test")

        assert trait.get_value() == "test-agent: test"


class TestDirective:
    """Tests for Directive."""

    def test_identity_with_prompt(self):
        identity = Directive(prompt="You are a code reviewer.")

        assert identity.prompt == "You are a code reviewer."
        assert identity.extensions == {}

    def test_identity_with_extensions(self):
        identity = Directive(
            prompt="You are a code reviewer.",
            extensions={"custom_field": "value"},
        )

        assert identity.prompt == "You are a code reviewer."
        assert identity.extensions == {"custom_field": "value"}


class TestDirectiveTrait:
    """Tests for DirectiveTrait."""

    def test_init_with_identity_object(self, mock_agent):
        identity = Directive(prompt="Test identity")
        trait = DirectiveTrait(mock_agent, identity)

        assert trait.directive == identity

    def test_init_with_string(self, mock_agent):
        """DirectiveTrait can be initialized with a string."""
        trait = DirectiveTrait(mock_agent, "You are a code reviewer.")

        assert trait.directive.prompt == "You are a code reviewer."

    def test_agent_assigned(self, mock_agent):
        """Trait receives agent on construction."""
        identity = Directive(prompt="Test identity")
        trait = DirectiveTrait(mock_agent, identity)

        assert trait.agent == mock_agent

    def test_build_prompt(self, mock_agent):
        identity = Directive(prompt="You are a code reviewer. Be critical.")
        trait = DirectiveTrait(mock_agent, identity)

        result = trait.build_prompt("Base system prompt.")

        # Directive is prepended
        assert result.startswith("You are a code reviewer.")
        assert "Base system prompt." in result


class TestMethodTrait:
    """Tests for MethodTrait."""

    def test_init(self, mock_agent):
        trait = MethodTrait(mock_agent, "- Step 1\n- Step 2")

        assert trait.method == "- Step 1\n- Step 2"

    def test_build_prompt(self, mock_agent):
        trait = MethodTrait(mock_agent, "- Step 1\n- Step 2")

        result = trait.build_prompt("Base prompt.")

        assert "Base prompt." in result
        assert "## Method" in result
        assert "- Step 1\n- Step 2" in result

    def test_update(self, mock_agent):
        trait = MethodTrait(mock_agent, "Original method")

        trait.update("Updated method")

        assert trait.method == "Updated method"


class TestAgentTraits:
    """Tests for Agent trait support."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    @pytest.fixture
    def agent(self, mock_logger):
        """Create a test agent."""
        from appinfra import DotDict

        from llm_gent.agents.default import Agent as DefaultAgent

        config = DotDict(identity={"name": "test"}, default_prompt="")
        return DefaultAgent(lg=mock_logger, config=config, session_storage=None)

    def test_add_trait(self, agent):
        identity = Directive(prompt="Test identity")
        trait = DirectiveTrait(agent, identity)
        agent.add_trait(trait)

        assert agent.has_trait(DirectiveTrait)
        assert agent.get_trait(DirectiveTrait) == trait

    def test_add_trait_attaches(self, agent):
        identity = Directive(prompt="Test identity")
        trait = DirectiveTrait(agent, identity)
        agent.add_trait(trait)

        assert trait.agent == agent

    def test_add_duplicate_trait_raises(self, agent):
        from llm_gent.core.errors import DuplicateTraitError

        identity = Directive(prompt="Test identity")
        agent.add_trait(DirectiveTrait(agent, identity))

        with pytest.raises(DuplicateTraitError, match="already registered"):
            agent.add_trait(DirectiveTrait(agent, identity))

    def test_get_trait_returns_none_if_not_added(self, agent):
        assert agent.get_trait(DirectiveTrait) is None

    def test_has_trait_returns_false_if_not_added(self, agent):
        assert not agent.has_trait(DirectiveTrait)
