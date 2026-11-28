# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The prompt-agent factory's trait selection: sections, class requirements, config override."""

from collections.abc import Iterator
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
from appinfra import DotDict

from llm_gent.agents.default import Factory
from llm_gent.core.platform import Platform
from llm_gent.core.traits import TraitName
from llm_gent.core.traits.builtin.directive import DirectiveTrait, MethodTrait
from llm_gent.core.traits.builtin.memory import MemoryTrait
from tests.helpers import make_platform


pytestmark = pytest.mark.unit


class _MemoryFactory(Factory):
    """A prompt factory whose agents always get memory."""

    required_traits: ClassVar[list[TraitName]] = [TraitName.MEMORY]


@pytest.fixture
def with_memory() -> Iterator[Platform]:
    """A platform holding a shared memory trait, on a patched kelt client."""
    with patch("llm_gent.core.traits.builtin.memory.KeltClient"):
        memory = MemoryTrait(
            MagicMock(), DotDict(), database=MagicMock(schema=None), chat_client=MagicMock()
        )
        yield make_platform(MagicMock(), memory=memory)


def _config(**sections: Any) -> dict[str, Any]:
    return {"identity": {"name": "a"}, "tools": {"complete_task": {}}, **sections}


class TestTraitSelection:
    def test_sections_attach_their_traits(self) -> None:
        agent = Factory(make_platform(MagicMock())).create(_config(directive="d", method="m"))

        assert agent.has(DirectiveTrait)
        assert agent.has(MethodTrait)

    def test_class_requirements_add_to_the_sections(self, with_memory: Platform) -> None:
        agent = _MemoryFactory(with_memory).create(_config(directive="d"))

        assert agent.has(MemoryTrait)
        assert agent.has(DirectiveTrait)

    def test_class_requirement_also_declared_by_a_section_attaches_once(self) -> None:
        class _DirectiveFactory(Factory):
            required_traits: ClassVar[list[TraitName]] = [TraitName.DIRECTIVE]

        agent = _DirectiveFactory(make_platform(MagicMock())).create(_config(directive="d"))

        assert agent.has(DirectiveTrait)

    def test_config_traits_required_overrides_both(self, with_memory: Platform) -> None:
        config = _config(directive="d", traits={"required": ["method"]}, method="m")

        agent = _MemoryFactory(with_memory).create(config)

        assert agent.has(MethodTrait)
        assert not agent.has(DirectiveTrait)
        assert not agent.has(MemoryTrait)
