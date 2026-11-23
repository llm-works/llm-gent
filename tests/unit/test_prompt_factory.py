# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The prompt-agent factory's trait selection: sections, class requirements, config override."""

from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

from llm_gent.agents.default import Factory
from llm_gent.core.traits import TraitName
from llm_gent.core.traits.builtin.directive import DirectiveTrait, MethodTrait
from llm_gent.core.traits.builtin.storage import StorageTrait
from tests.helpers import make_platform


pytestmark = pytest.mark.unit


class _StorageFactory(Factory):
    """A prompt factory whose agents always get storage."""

    required_traits: ClassVar[list[TraitName]] = [TraitName.STORAGE]


def _config(**sections: Any) -> dict[str, Any]:
    return {"identity": {"name": "a"}, "tools": {"complete_task": {}}, **sections}


class TestTraitSelection:
    def test_sections_attach_their_traits(self) -> None:
        agent = Factory(make_platform(MagicMock())).create(_config(directive="d", method="m"))

        assert agent.has_trait(DirectiveTrait)
        assert agent.has_trait(MethodTrait)

    def test_class_requirements_add_to_the_sections(self) -> None:
        agent = _StorageFactory(make_platform(MagicMock())).create(_config(directive="d"))

        assert agent.has_trait(StorageTrait)
        assert agent.has_trait(DirectiveTrait)

    def test_class_requirement_also_declared_by_a_section_attaches_once(self) -> None:
        class _DirectiveFactory(Factory):
            required_traits: ClassVar[list[TraitName]] = [TraitName.DIRECTIVE]

        agent = _DirectiveFactory(make_platform(MagicMock())).create(_config(directive="d"))

        assert agent.has_trait(DirectiveTrait)

    def test_config_traits_required_overrides_both(self) -> None:
        config = _config(directive="d", traits={"required": ["method"]}, method="m")

        agent = _StorageFactory(make_platform(MagicMock())).create(config)

        assert agent.has_trait(MethodTrait)
        assert not agent.has_trait(DirectiveTrait)
        assert not agent.has_trait(StorageTrait)
