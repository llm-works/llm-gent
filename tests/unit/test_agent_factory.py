# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for AgentFactory tutorial-shape construction paths."""

from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_gent.core.agent import Agent, AgentFactory
from llm_gent.core.platform import PlatformContext
from llm_gent.core.traits.builtin.directive import DirectiveTrait


pytestmark = pytest.mark.unit


class TestAgentFactoryConstruction:
    """AgentFactory accepts either a bare Logger or a PlatformContext."""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock(spec=Logger)

    def test_bare_logger_synthesizes_platform(self, mock_logger):
        factory = AgentFactory(mock_logger)

        assert isinstance(factory._platform, PlatformContext)
        assert factory._lg is mock_logger
        assert factory._platform.llm_config() == {}
        assert factory._platform.learn_config() is None

    def test_platform_context_used_as_is(self, mock_logger):
        platform = PlatformContext(lg=mock_logger, config={"llm": {"default": "x"}})

        factory = AgentFactory(platform)

        assert factory._platform is platform
        assert factory._lg is mock_logger

    def test_default_agent_class_is_bare_agent(self):
        assert AgentFactory.agent_class is Agent


class TestFromConfig:
    """AgentFactory.from_config wires a bare Agent from a dict, no subclass."""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock(spec=Logger)

    def test_returns_bare_agent_without_traits(self, mock_logger):
        factory = AgentFactory(mock_logger)

        agent = factory.from_config({"identity": {"name": "hello"}})

        assert isinstance(agent, Agent)
        assert agent.name == "hello"
        assert agent.traits.count() == 0

    def test_attaches_directive_via_required_list(self, mock_logger):
        factory = AgentFactory(mock_logger)

        agent = factory.from_config(
            {
                "identity": {"name": "hello"},
                "directive": "You are a helpful assistant.",
                "traits": {"required": ["directive"]},
            }
        )

        directive = agent.get_trait(DirectiveTrait)
        assert directive is not None
        assert directive.directive.prompt == "You are a helpful assistant."

    def test_accepts_dotdict_config(self, mock_logger):
        from appinfra import DotDict

        factory = AgentFactory(mock_logger)

        agent = factory.from_config(DotDict(identity={"name": "hello"}))

        assert agent.name == "hello"

    def test_missing_identity_raises(self, mock_logger):
        from llm_gent.core.errors import ConfigError

        factory = AgentFactory(mock_logger)

        with pytest.raises(ConfigError, match="identity.name is required"):
            factory.from_config({"directive": "hi"})

    def test_lifecycle_starts_attached_traits(self, mock_logger):
        factory = AgentFactory(mock_logger)

        agent = factory.from_config(
            {
                "identity": {"name": "hello"},
                "directive": "You are helpful.",
                "traits": {"required": ["directive"]},
            }
        )
        agent.start()
        try:
            assert agent._started is True
        finally:
            agent.stop()
        assert agent._started is False


class TestSubclassPathPreserved:
    """Existing subclass-based factory usage still works."""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock(spec=Logger)

    def test_subclass_overrides_agent_class(self, mock_logger):
        class Custom(AgentFactory):
            pass

        class CustomAgent(Agent):
            pass

        Custom.agent_class = CustomAgent

        agent = Custom(mock_logger).from_config({"identity": {"name": "sub"}})
        assert isinstance(agent, CustomAgent)
