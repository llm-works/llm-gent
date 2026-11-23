# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for trait factory."""

from unittest.mock import MagicMock, patch

import pytest
from appinfra import DotDict

from llm_gent.core.errors import ConfigError
from llm_gent.core.traits import TraitName
from llm_gent.core.traits.builtin.directive import Directive, DirectiveTrait, MethodTrait
from llm_gent.core.traits.builtin.memory import MemoryTrait
from llm_gent.core.traits.builtin.training import TrainingTrait
from llm_gent.core.traits.factory import Factory as TraitFactory
from llm_gent.core.traits.factory import MemoryResources


pytestmark = pytest.mark.unit


@pytest.fixture
def router():
    """The platform's shared router."""
    return MagicMock()


@pytest.fixture
def factory(router):
    """A TraitFactory on a shared router, without memory."""
    return TraitFactory(MagicMock(), router)


@pytest.fixture
def mock_agent():
    """Create a mock Agent."""
    agent = MagicMock()
    agent.config = DotDict({})
    agent.identity = MagicMock()
    agent.lg = MagicMock()
    return agent


class TestCreate:
    """Tests for TraitFactory.create() dispatch."""

    def test_unknown_trait_raises_config_error(self, factory, mock_agent):
        fake_trait = MagicMock()
        fake_trait.__str__ = lambda _self: "unknown_trait"

        with pytest.raises(ConfigError, match="Unknown trait type"):
            factory.create(fake_trait, mock_agent)

    def test_routes_to_directive_creator(self, factory, mock_agent):
        mock_agent.config = DotDict({"directive": "You are helpful."})
        trait = factory.create(TraitName.DIRECTIVE, mock_agent)
        assert isinstance(trait, DirectiveTrait)

    def test_llm_is_not_a_trait(self):
        assert "llm" not in {name.value for name in TraitName}

    def test_routes_to_method_creator(self, factory, mock_agent):
        mock_agent.config = DotDict({"method": "Step 1, Step 2"})
        trait = factory.create(TraitName.METHOD, mock_agent)
        assert isinstance(trait, MethodTrait)


class TestCreateDirectiveTrait:
    """Tests for TraitFactory.create_directive_trait()."""

    def test_string_config(self, factory, mock_agent):
        trait = factory.create_directive_trait(mock_agent, "You are a reviewer.")
        assert isinstance(trait, DirectiveTrait)
        assert trait.directive.prompt == "You are a reviewer."

    def test_dict_config(self, factory, mock_agent):
        config = {"prompt": "You are helpful.", "extensions": {"tone": "formal"}}
        trait = factory.create_directive_trait(mock_agent, config)
        assert trait.directive.prompt == "You are helpful."
        assert trait.directive.extensions == {"tone": "formal"}

    def test_directive_object(self, factory, mock_agent):
        directive = Directive(prompt="Already built.")
        trait = factory.create_directive_trait(mock_agent, directive)
        assert trait.directive is directive

    def test_none_config_raises(self, factory, mock_agent):
        with pytest.raises(ConfigError, match="Directive configuration required"):
            factory.create_directive_trait(mock_agent, None)

    def test_invalid_type_raises(self, factory, mock_agent):
        with pytest.raises(ConfigError, match="must be str, dict, or Directive"):
            factory.create_directive_trait(mock_agent, 42)


@pytest.fixture
def memory():
    """The platform's shared memory clients."""
    return MemoryResources(database=MagicMock(), chat_client=MagicMock(), embedder=MagicMock())


@pytest.fixture
def mock_kelt():
    """Patch the kelt client MemoryTrait composes; nothing touches a database."""
    with (
        patch("llm_gent.core.traits.builtin.memory.KeltClient") as kelt,
        patch("llm_gent.core.traits.builtin.memory.ContextBuilder"),
    ):
        # MagicMock intercepts `.assertions` for its assertion helpers; assign explicitly.
        instance = MagicMock()
        instance.atomic = MagicMock()
        instance.atomic.assertions = MagicMock()
        kelt.return_value = instance
        yield kelt


class TestCreateMemoryTrait:
    """Tests for TraitFactory.create_memory_trait(): the platform's memory clients, not owned."""

    def test_none_config_raises(self, factory, mock_agent):
        with pytest.raises(ConfigError, match="Memory configuration required"):
            factory.create_memory_trait(mock_agent, mock_agent.identity, None)

    def test_missing_db_raises(self, factory, mock_agent):
        config = DotDict({"embedder_url": "http://localhost:9000"})
        with pytest.raises(ConfigError, match="missing required 'db' field"):
            factory.create_memory_trait(mock_agent, mock_agent.identity, config)

    def test_platform_without_memory_raises(self, factory, mock_agent):
        config = DotDict({"db": {"url": "postgresql://localhost/test"}})
        with pytest.raises(ConfigError, match="platform built with memory"):
            factory.create_memory_trait(mock_agent, mock_agent.identity, config)

    def test_uses_shared_clients(self, router, memory, mock_agent, mock_kelt):
        factory = TraitFactory(MagicMock(), router, memory=memory)
        config = DotDict({"db": {"url": "postgresql://localhost/test"}})

        trait = factory.create_memory_trait(mock_agent, mock_agent.identity, config)

        assert isinstance(trait, MemoryTrait)
        assert trait._database is memory.database
        assert trait._client is memory.chat_client
        assert trait._embedder is memory.embedder
        assert trait._owns_chat_client is False
        assert trait._owns_embedder is False

    def test_defaults_applied(self, router, memory, mock_agent, mock_kelt):
        factory = TraitFactory(MagicMock(), router, memory=memory)
        config = DotDict({"db": {"url": "postgresql://localhost/test"}})

        trait = factory.create_memory_trait(mock_agent, mock_agent.identity, config)

        assert trait.config.embedder_model == "default"
        assert trait.config.embedder_timeout == 30.0


class TestCreateTrainingTrait:
    """Tests for TraitFactory.create_training_trait()."""

    def test_empty_source_yields_empty_config(self, factory, mock_agent):
        trait = factory.create_training_trait(mock_agent, None)
        assert isinstance(trait, TrainingTrait)
        assert trait.default_schema == "public"

    def test_reads_schema_and_adapters(self, factory, mock_agent):
        source = {
            "schema": {"name": "custom", "enforce": True},
            "adapters": {"lora": {"base_path": "/tmp/adapters"}},
        }
        trait = factory.create_training_trait(mock_agent, source)
        assert trait.default_schema == "custom"
        assert trait.config.get("schema", {}).get("enforce") is True
        assert trait.config.get("adapters", {}).get("lora", {}).get("base_path") == "/tmp/adapters"


class TestCreateMethodTrait:
    """Tests for TraitFactory.create_method_trait()."""

    def test_none_raises(self, factory, mock_agent):
        with pytest.raises(ConfigError, match="Method configuration required"):
            factory.create_method_trait(mock_agent, None)

    def test_empty_string_raises(self, factory, mock_agent):
        with pytest.raises(ConfigError, match="Method configuration required"):
            factory.create_method_trait(mock_agent, "")

    def test_valid_string(self, factory, mock_agent):
        trait = factory.create_method_trait(mock_agent, "Step 1: Analyze\nStep 2: Synthesize")
        assert isinstance(trait, MethodTrait)
        assert trait.method == "Step 1: Analyze\nStep 2: Synthesize"


class TestCreateRating:
    """Tests for _create_rating routing."""

    def test_creates_rating_trait(self, factory, mock_agent):
        from llm_gent.core.traits.builtin.rating import RatingTrait

        mock_agent.config = DotDict({"rating": {"scale": 5}})
        trait = factory._create_rating(mock_agent)
        assert isinstance(trait, RatingTrait)

    def test_rating_calls_go_through_the_shared_router(self, factory, router, mock_agent):
        trait = factory._create_rating(mock_agent)

        assert trait._router is router

    def test_rating_with_none_config(self, factory, mock_agent):
        from llm_gent.core.traits.builtin.rating import RatingTrait

        mock_agent.config = DotDict({})
        trait = factory._create_rating(mock_agent)
        assert isinstance(trait, RatingTrait)


class TestCreateStorage:
    """Tests for _create_storage routing."""

    def test_creates_storage_trait(self, factory, mock_agent):
        from llm_gent.core.traits.builtin.storage import StorageTrait

        trait = factory._create_storage(mock_agent)
        assert isinstance(trait, StorageTrait)


class TestCreateMemoryRouting:
    """Tests for _create_memory with the platform's memory settings."""

    def test_no_memory_config_raises(self, factory, mock_agent):
        mock_agent.config = DotDict({})
        with pytest.raises(ConfigError, match="Memory configuration required"):
            factory._create_memory(mock_agent)

    def test_merges_agent_kelt_schema(self, router):
        memory_config = {
            "db": {"url": "postgresql://localhost/test"},
            "schema": {"name": "default"},
        }
        factory_inst = TraitFactory(MagicMock(), router, memory_config=memory_config)

        mock_agent = MagicMock()
        mock_agent.config = DotDict({"kelt": {"schema": {"name": "custom", "enforce": True}}})
        mock_agent.identity = MagicMock()

        with patch.object(factory_inst, "create_memory_trait") as mock_create:
            mock_create.return_value = MagicMock()
            factory_inst._create_memory(mock_agent)

            config_passed = mock_create.call_args[0][2]
            assert config_passed["schema"] == {"name": "custom", "enforce": True}
        assert memory_config["schema"] == {"name": "default"}


class TestCreateTrainingRouting:
    """Tests for _create_training with the platform's memory settings."""

    def test_no_memory_config_creates_default_trait(self, factory):
        mock_agent = MagicMock()
        mock_agent.config = DotDict({})

        trait = factory._create_training(mock_agent)
        assert isinstance(trait, TrainingTrait)
        assert trait.default_schema == "public"

    def test_reads_platform_adapters(self, router):
        memory_config = {
            "schema": {"name": "platform"},
            "adapters": {"lora": {"base_path": "/data/adapters"}},
        }
        factory_inst = TraitFactory(MagicMock(), router, memory_config=memory_config)

        mock_agent = MagicMock()
        mock_agent.config = DotDict({})

        trait = factory_inst._create_training(mock_agent)
        assert trait.default_schema == "platform"
        assert trait.config.get("adapters", {}).get("lora", {}).get("base_path") == "/data/adapters"

    def test_agent_kelt_schema_overrides_platform(self, router):
        memory_config = {
            "schema": {"name": "platform"},
            "adapters": {"lora": {"base_path": "/data/adapters"}},
        }
        factory_inst = TraitFactory(MagicMock(), router, memory_config=memory_config)

        mock_agent = MagicMock()
        mock_agent.config = DotDict({"kelt": {"schema": {"name": "agent"}}})

        trait = factory_inst._create_training(mock_agent)
        assert trait.default_schema == "agent"
