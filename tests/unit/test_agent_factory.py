# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for AgentFactory construction on a platform."""

from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_gent.core.agent import Agent
from llm_gent.core.agent import Factory as AgentFactory
from llm_gent.core.errors import ConfigError
from llm_gent.core.traits.builtin.directive import DirectiveTrait
from tests.helpers import make_platform


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def platform(mock_logger):
    return make_platform(mock_logger)


class TestAgentFactoryConstruction:
    """AgentFactory builds agents on the platform it is given."""

    def test_uses_platform_and_its_logger(self, platform, mock_logger):
        factory = AgentFactory(platform)

        assert factory._platform is platform
        assert factory._lg is mock_logger

    def test_default_agent_class_is_bare_agent(self):
        assert AgentFactory.agent_class is Agent


class TestModelAccess:
    """Models are the platform's: an agent config configures none."""

    def test_llm_key_raises(self, platform):
        config = {"identity": {"name": "a"}, "llm": "anthropic"}

        with pytest.raises(ConfigError, match="'llm' key"):
            AgentFactory(platform).from_config(config)

    def test_llm_trait_name_is_unknown(self, platform):
        config = {"identity": {"name": "a"}, "traits": {"required": ["llm"]}}

        with pytest.raises(ConfigError):
            AgentFactory(platform).from_config(config)

    def test_agent_does_not_close_the_shared_router(self, platform):
        agent = AgentFactory(platform).from_config(
            {"identity": {"name": "a"}, "directive": "d", "traits": {"required": ["directive"]}}
        )

        agent.start()
        agent.stop()

        platform.llm.close.assert_not_called()


@pytest.fixture
def memory():
    """The platform's shared memory trait, on a patched kelt client."""
    from unittest.mock import patch

    from appinfra import DotDict

    from llm_gent.core.traits.builtin.memory import MemoryTrait

    with patch("llm_gent.core.traits.builtin.memory.KeltClient"):
        yield MemoryTrait(
            MagicMock(), DotDict(), database=MagicMock(schema=None), chat_client=MagicMock()
        )


def _memory_agent(name: str, *, tools: dict) -> dict:
    """An agent config that uses the platform's memory and the given tools."""
    return {"identity": {"name": name}, "traits": {"required": ["memory"]}, "tools": tools}


class TestPlatformTraits:
    """A required trait the platform has is bound to the agent; the factory knows no trait."""

    def test_the_agent_holds_its_binding_of_the_platform_memory(self, mock_logger, memory):
        from llm_gent.core.traits.builtin.memory import MemoryBinding, MemoryTrait

        platform = make_platform(mock_logger, memory=memory)
        config = {
            "identity": {"name": "a", "context_key": "jokes"},
            "traits": {"required": ["memory"]},
        }

        agent = AgentFactory(platform).from_config(config)

        binding = agent.require(MemoryTrait)
        assert isinstance(binding, MemoryBinding)
        assert binding.context_key == "jokes"
        assert binding.trait is memory

    def test_the_agent_holds_its_binding_of_the_platform_rating(self, mock_logger, memory):
        from llm_gent.core.traits.builtin.rating import RatingBinding, RatingTrait

        platform = make_platform(mock_logger, memory=memory)
        config = {
            "identity": {"name": "a", "context_key": "jokes"},
            "traits": {"required": ["rating"]},
            "rating": {"batch_size": 3},
        }

        agent = AgentFactory(platform).from_config(config)

        binding = agent.require(RatingTrait)
        assert isinstance(binding, RatingBinding)
        assert binding.context_key == "jokes"
        assert binding.trait is platform.traits.require(RatingTrait)
        assert binding.config["batch_size"] == 3

    def test_rating_needs_the_platforms_memory(self, platform):
        config = {"identity": {"name": "a"}, "traits": {"required": ["rating"]}}

        with pytest.raises(ConfigError, match="'rating' is not on this platform.*memory section"):
            AgentFactory(platform).from_config(config)

    def test_agents_share_the_platform_training_trait(self, platform):
        from llm_gent.core.traits.builtin.training import TrainingTrait

        factory = AgentFactory(platform)
        config = {"identity": {"name": "a"}, "traits": {"required": ["training"]}}

        first = factory.from_config(config)
        second = factory.from_config({**config, "identity": {"name": "b"}})

        shared = platform.traits.require(TrainingTrait)
        assert first.require(TrainingTrait) is shared
        assert second.require(TrainingTrait) is shared
        assert first.traits.count() == 0

    def test_a_required_platform_trait_that_is_not_scoped_is_the_platforms(self, mock_logger):
        from llm_gent.core.traits import Registry as TraitRegistry
        from llm_gent.core.traits import TraitName

        class SharedRating:
            """A platform trait under a required name, with no per-agent data."""

            trait_name = TraitName.RATING

            def on_start(self) -> None:
                pass

            def on_stop(self) -> None:
                pass

        traits = TraitRegistry(mock_logger)
        shared = SharedRating()
        traits.register(shared)
        platform = make_platform(mock_logger, traits=traits)
        config = {"identity": {"name": "a"}, "traits": {"required": ["rating"]}}

        agent = AgentFactory(platform).from_config(config)

        assert agent.require(SharedRating) is shared

    def test_memory_tools_use_their_own_agents_binding(self, mock_logger, memory):
        """An agent built while another's tools are being made leaves it its own memory."""
        from llm_gent.core.tools.base import BaseTool
        from llm_gent.core.traits.builtin.memory import MemoryTrait
        from llm_gent.core.traits.builtin.tools import ToolsTrait

        platform = make_platform(mock_logger, memory=memory)
        factory = AgentFactory(platform)
        built: list[Agent] = []

        def build_another(config):
            built.append(factory.from_config(_memory_agent("b", tools={"remember": {}})))
            tool = MagicMock(spec=BaseTool)
            tool.name = "interleave"
            return tool

        platform.tool_factory.register("interleave", build_another)
        first = factory.from_config(_memory_agent("a", tools={"interleave": {}, "remember": {}}))

        remember = first.require(ToolsTrait).registry.get("remember")
        assert remember is not None
        assert remember._memory_trait is first.require(MemoryTrait)
        assert built[0].require(MemoryTrait) is not first.require(MemoryTrait)

    def test_a_trait_neither_built_nor_on_the_platform_raises(self, platform):
        config = {"identity": {"name": "a"}, "traits": {"required": ["memory"]}}

        with pytest.raises(ConfigError, match="'memory' is not on this platform"):
            AgentFactory(platform).from_config(config)

    def test_requiring_storage_says_where_it_went(self, platform):
        config = {"identity": {"name": "a"}, "traits": {"required": ["storage"]}}

        with pytest.raises(ConfigError, match=r"part of memory.*MemoryTrait\)\.storage"):
            AgentFactory(platform).from_config(config)

    def test_an_agent_kelt_schema_raises(self, platform):
        config = {"identity": {"name": "a"}, "kelt": {"schema": {"name": "custom"}}}

        with pytest.raises(ConfigError, match="kelt.schema"):
            AgentFactory(platform).from_config(config)


class TestFromConfig:
    """AgentFactory.from_config wires a bare Agent from a dict, no subclass."""

    def test_returns_bare_agent_without_traits(self, platform):
        factory = AgentFactory(platform)

        agent = factory.from_config({"identity": {"name": "hello"}})

        assert isinstance(agent, Agent)
        assert agent.name == "hello"
        assert agent.traits.count() == 0

    def test_attaches_directive_via_required_list(self, platform):
        factory = AgentFactory(platform)

        agent = factory.from_config(
            {
                "identity": {"name": "hello"},
                "directive": "You are a helpful assistant.",
                "traits": {"required": ["directive"]},
            }
        )

        directive = agent.get(DirectiveTrait)
        assert directive is not None
        assert directive.directive.prompt == "You are a helpful assistant."

    def test_accepts_dotdict_config(self, platform):
        from appinfra import DotDict

        factory = AgentFactory(platform)

        agent = factory.from_config(DotDict(identity={"name": "hello"}))

        assert agent.name == "hello"

    def test_missing_identity_raises(self, platform):
        from llm_gent.core.errors import ConfigError

        factory = AgentFactory(platform)

        with pytest.raises(ConfigError, match="identity.name is required"):
            factory.from_config({"directive": "hi"})

    def test_lifecycle_starts_attached_traits(self, platform):
        factory = AgentFactory(platform)

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

    def test_subclass_overrides_agent_class(self, platform):
        class Custom(AgentFactory):
            pass

        class CustomAgent(Agent):
            pass

        Custom.agent_class = CustomAgent

        agent = Custom(platform).from_config({"identity": {"name": "sub"}})
        assert isinstance(agent, CustomAgent)
