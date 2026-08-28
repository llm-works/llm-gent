# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Agent traits for composable capabilities."""

from enum import StrEnum

from .base import BaseTrait, Trait
from .builtin import (
    Directive,
    DirectiveTrait,
    HTTPConfig,
    HTTPTrait,
    LearnConfig,
    LearnTrait,
    LLMConfig,
    LLMTrait,
    MethodTrait,
    RatingConfig,
    RatingTrait,
    SAIAConfig,
    SAIATrait,
    StorageTrait,
    ToolsTrait,
)
from .factory import Factory
from .registry import Registry


class TraitName(StrEnum):
    """Trait name identifiers.

    Using str Enum so values work in YAML configs and as strings.
    Agents can declare required/optional traits using these enum values.
    """

    DIRECTIVE = "directive"
    LLM = "llm"
    LEARN = "learn"
    HTTP = "http"
    RATING = "rating"
    SAIA = "saia"
    STORAGE = "storage"
    TOOLS = "tools"
    METHOD = "method"


# All trait types available in the platform
ALL_TRAITS: list[TraitName] = [
    TraitName.DIRECTIVE,
    TraitName.LLM,
    TraitName.LEARN,
    TraitName.HTTP,
    TraitName.RATING,
    TraitName.SAIA,
    TraitName.STORAGE,
    TraitName.TOOLS,
    TraitName.METHOD,
]


__all__ = [
    # Base
    "BaseTrait",
    "Trait",
    # Names & Catalogs
    "TraitName",
    "ALL_TRAITS",
    # Factory & Registry
    "Factory",
    "Registry",
    # Directive/Method
    "Directive",
    "DirectiveTrait",
    "MethodTrait",
    # HTTP
    "HTTPConfig",
    "HTTPTrait",
    # Learn
    "LearnConfig",
    "LearnTrait",
    # LLM
    "LLMConfig",
    "LLMTrait",
    # Rating
    "RatingConfig",
    "RatingTrait",
    # SAIA
    "SAIAConfig",
    "SAIATrait",
    # Storage
    "StorageTrait",
    # Tools
    "ToolsTrait",
]
