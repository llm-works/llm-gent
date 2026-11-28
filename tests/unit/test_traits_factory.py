# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for trait factory."""

from unittest.mock import MagicMock

import pytest
from appinfra import DotDict

from llm_gent.core.errors import ConfigError
from llm_gent.core.traits import TraitName
from llm_gent.core.traits.builtin.directive import Directive, DirectiveTrait, MethodTrait
from llm_gent.core.traits.factory import Factory as TraitFactory


pytestmark = pytest.mark.unit


@pytest.fixture
def factory():
    """A TraitFactory."""
    return TraitFactory(MagicMock())


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


class TestPlatformTraitsAreNotBuiltHere:
    """Memory, rating and training are the platform's: the factory builds none of them."""

    @pytest.mark.parametrize("name", [TraitName.MEMORY, TraitName.RATING, TraitName.TRAINING])
    def test_does_not_create(self, factory, name):
        assert not factory.creates(name)


class TestCreateStorage:
    """Tests for _create_storage routing."""

    def test_creates_storage_trait(self, factory, mock_agent):
        from llm_gent.core.traits.builtin.storage import StorageTrait

        trait = factory._create_storage(mock_agent)
        assert isinstance(trait, StorageTrait)
