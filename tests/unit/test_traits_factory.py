# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for trait factory."""

from unittest.mock import MagicMock, patch

import pytest
from appinfra import DotDict

from llm_gent.core.errors import ConfigError
from llm_gent.core.traits import TraitName
from llm_gent.core.traits.builtin.directive import Directive, DirectiveTrait, MethodTrait
from llm_gent.core.traits.builtin.llm import LLMTrait
from llm_gent.core.traits.builtin.memory import MemoryTrait
from llm_gent.core.traits.builtin.training import TrainingTrait
from llm_gent.core.traits.factory import TraitFactory


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_platform():
    """Create a mock PlatformContext."""
    platform = MagicMock()
    platform.logger = MagicMock()
    platform.llm_config.return_value = {
        "default": "anthropic",
        "backends": {
            "anthropic": {"adapter": "openai", "base_url": "https://api.anthropic.com"},
            "local": {"adapter": "openai", "base_url": "http://localhost:8000"},
        },
    }
    platform.learn_config.return_value = None
    return platform


@pytest.fixture
def factory(mock_platform):
    """Create a TraitFactory instance."""
    return TraitFactory(mock_platform)


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

    def test_routes_to_llm_creator(self, factory, mock_agent):
        trait = factory.create(TraitName.LLM, mock_agent)
        assert isinstance(trait, LLMTrait)

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


class TestCreateLLMTrait:
    """Tests for TraitFactory.create_llm_trait()."""

    def test_none_config_raises(self, factory, mock_agent):
        with pytest.raises(ConfigError, match="LLM configuration required"):
            factory.create_llm_trait(mock_agent, None)

    def test_builds_router_and_injects(self, factory, mock_agent):
        config = DotDict({"default": "anthropic", "backends": {"anthropic": {"adapter": "openai"}}})

        with patch("llm_infer.client.Factory") as mock_factory:
            mock_router = MagicMock()
            mock_factory.return_value.from_config.return_value = mock_router

            trait = factory.create_llm_trait(mock_agent, config)

            mock_factory.assert_called_once_with(mock_agent.lg)
            mock_factory.return_value.from_config.assert_called_once_with(config)
            assert isinstance(trait, LLMTrait)
            assert trait.router is mock_router
            assert trait._owns_router is True


class TestCreateMemoryTrait:
    """Tests for TraitFactory.create_memory_trait()."""

    def test_none_config_raises(self, factory, mock_agent):
        with pytest.raises(ConfigError, match="Memory configuration required"):
            factory.create_memory_trait(mock_agent, mock_agent.identity, None)

    def test_missing_db_raises(self, factory, mock_agent):
        config = DotDict({"embedder_url": "http://localhost:9000"})
        with pytest.raises(ConfigError, match="missing required 'db' field"):
            factory.create_memory_trait(mock_agent, mock_agent.identity, config)

    def test_valid_config(self, factory, mock_agent):
        config = DotDict(
            {
                "db": {"url": "postgresql://localhost/test"},
                "embedder_url": "http://localhost:9000",
            }
        )
        trait = factory.create_memory_trait(mock_agent, mock_agent.identity, config)
        assert isinstance(trait, MemoryTrait)

    def test_defaults_applied(self, factory, mock_agent):
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


class TestCreateLLMRouting:
    """Tests for _create_llm with config merging."""

    def test_no_agent_llm_config_uses_platform(self, factory, mock_agent):
        mock_agent.config = DotDict({})
        trait = factory.create(TraitName.LLM, mock_agent)
        assert isinstance(trait, LLMTrait)

    def test_string_shorthand_selects_backend(self, factory, mock_agent):
        mock_agent.config = DotDict({"llm": "local"})
        trait = factory.create(TraitName.LLM, mock_agent)
        assert isinstance(trait, LLMTrait)

    def test_string_shorthand_overrides_default(self, factory):
        mock_agent = MagicMock()
        mock_agent.config = DotDict({"llm": "local"})

        with patch.object(factory, "create_llm_trait") as mock_create:
            mock_create.return_value = MagicMock()
            factory._create_llm(mock_agent)

            config_passed = mock_create.call_args[0][1]
            assert config_passed["default"] == "local"
            assert "anthropic" in config_passed["backends"]

    def test_dict_config_merges(self, factory):
        mock_agent = MagicMock()
        mock_agent.config = DotDict(
            {"llm": {"default": "local", "backends": {"local": {"adapter": "custom"}}}}
        )

        with patch.object(factory, "create_llm_trait") as mock_create:
            mock_create.return_value = MagicMock()
            factory._create_llm(mock_agent)

            config_passed = mock_create.call_args[0][1]
            assert config_passed["default"] == "local"
            assert config_passed["backends"]["local"]["adapter"] == "custom"


class TestMergeLLMConfig:
    """Tests for TraitFactory._merge_llm_config()."""

    def test_override_existing_backend(self, factory):
        base = DotDict(
            {
                "default": "anthropic",
                "backends": {
                    "anthropic": {"adapter": "openai", "base_url": "https://api.anthropic.com"}
                },
            }
        )
        override = {"backends": {"anthropic": {"base_url": "https://custom.endpoint.com"}}}

        result = factory._merge_llm_config(base, override)

        assert result["backends"]["anthropic"]["adapter"] == "openai"
        assert result["backends"]["anthropic"]["base_url"] == "https://custom.endpoint.com"

    def test_add_new_backend(self, factory):
        base = DotDict(
            {
                "default": "anthropic",
                "backends": {"anthropic": {"adapter": "openai"}},
            }
        )
        override = {"backends": {"custom": {"adapter": "vllm", "base_url": "http://custom:8000"}}}

        result = factory._merge_llm_config(base, override)

        assert "custom" in result["backends"]
        assert result["backends"]["custom"]["adapter"] == "vllm"
        assert result["backends"]["anthropic"]["adapter"] == "openai"

    def test_top_level_keys_merged(self, factory):
        base = DotDict(
            {
                "default": "anthropic",
                "backends": {"anthropic": {"adapter": "openai"}},
            }
        )
        override = {"default": "local", "temperature": 0.7}

        result = factory._merge_llm_config(base, override)

        assert result["default"] == "local"
        assert result["temperature"] == 0.7

    def test_override_backends_only_in_override(self, factory):
        base = DotDict({"default": "anthropic"})
        override = {"backends": {"new": {"adapter": "test"}}}

        result = factory._merge_llm_config(base, override)

        assert result["backends"]["new"]["adapter"] == "test"

    def test_none_backend_skipped(self, factory):
        base = DotDict(
            {
                "default": "anthropic",
                "backends": {"anthropic": {"adapter": "openai", "model": "claude-3"}},
            }
        )
        override = {"backends": {"anthropic": None}}

        result = factory._merge_llm_config(base, override)

        assert result["backends"]["anthropic"]["adapter"] == "openai"

    def test_does_not_mutate_base(self, factory):
        base = DotDict(
            {
                "default": "anthropic",
                "backends": {"anthropic": {"adapter": "openai"}},
            }
        )
        original_adapter = base["backends"]["anthropic"]["adapter"]
        override = {"backends": {"anthropic": {"adapter": "custom"}}}

        factory._merge_llm_config(base, override)

        assert base["backends"]["anthropic"]["adapter"] == original_adapter


class TestCreateRating:
    """Tests for _create_rating routing."""

    def test_creates_rating_trait(self, factory, mock_agent):
        from llm_gent.core.traits.builtin.rating import RatingTrait

        mock_agent.config = DotDict({"rating": {"scale": 5}})
        trait = factory._create_rating(mock_agent)
        assert isinstance(trait, RatingTrait)

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
    """Tests for _create_memory with platform config merging."""

    def test_no_learn_config_raises(self, factory, mock_agent):
        mock_agent.config = DotDict({})
        with pytest.raises(ConfigError, match="Memory configuration required"):
            factory._create_memory(mock_agent)

    def test_merges_agent_kelt_schema(self, mock_platform):
        mock_platform.learn_config.return_value = {
            "db": {"url": "postgresql://localhost/test"},
            "schema": {"name": "default"},
        }
        factory_inst = TraitFactory(mock_platform)

        mock_agent = MagicMock()
        mock_agent.config = DotDict({"kelt": {"schema": {"name": "custom", "enforce": True}}})
        mock_agent.identity = MagicMock()

        with patch.object(factory_inst, "create_memory_trait") as mock_create:
            mock_create.return_value = MagicMock()
            factory_inst._create_memory(mock_agent)

            config_passed = mock_create.call_args[0][2]
            assert config_passed["schema"] == {"name": "custom", "enforce": True}


class TestCreateTrainingRouting:
    """Tests for _create_training with platform config merging."""

    def test_empty_platform_config_creates_default_trait(self, mock_platform):
        mock_platform.learn_config.return_value = None
        factory_inst = TraitFactory(mock_platform)

        mock_agent = MagicMock()
        mock_agent.config = DotDict({})

        trait = factory_inst._create_training(mock_agent)
        assert isinstance(trait, TrainingTrait)
        assert trait.default_schema == "public"

    def test_reads_platform_adapters(self, mock_platform):
        mock_platform.learn_config.return_value = {
            "schema": {"name": "platform"},
            "adapters": {"lora": {"base_path": "/data/adapters"}},
        }
        factory_inst = TraitFactory(mock_platform)

        mock_agent = MagicMock()
        mock_agent.config = DotDict({})

        trait = factory_inst._create_training(mock_agent)
        assert trait.default_schema == "platform"
        assert trait.config.get("adapters", {}).get("lora", {}).get("base_path") == "/data/adapters"

    def test_agent_kelt_schema_overrides_platform(self, mock_platform):
        mock_platform.learn_config.return_value = {
            "schema": {"name": "platform"},
            "adapters": {"lora": {"base_path": "/data/adapters"}},
        }
        factory_inst = TraitFactory(mock_platform)

        mock_agent = MagicMock()
        mock_agent.config = DotDict({"kelt": {"schema": {"name": "agent"}}})

        trait = factory_inst._create_training(mock_agent)
        assert trait.default_schema == "agent"
