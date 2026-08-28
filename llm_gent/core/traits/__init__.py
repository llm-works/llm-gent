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
    LLMConfig,
    LLMTrait,
    ManifestNotFoundError,
    MemoryConfig,
    MemoryTrait,
    MethodTrait,
    RatingConfig,
    RatingTrait,
    SAIAConfig,
    SAIATrait,
    StorageTrait,
    ToolsTrait,
    TrainingConfig,
    TrainingTrait,
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
    MEMORY = "memory"
    HTTP = "http"
    RATING = "rating"
    SAIA = "saia"
    STORAGE = "storage"
    TOOLS = "tools"
    METHOD = "method"
    TRAINING = "training"


# All trait types available in the platform
ALL_TRAITS: list[TraitName] = [
    TraitName.DIRECTIVE,
    TraitName.LLM,
    TraitName.MEMORY,
    TraitName.HTTP,
    TraitName.RATING,
    TraitName.SAIA,
    TraitName.STORAGE,
    TraitName.TOOLS,
    TraitName.METHOD,
    TraitName.TRAINING,
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
    # LLM
    "LLMConfig",
    "LLMTrait",
    # Memory
    "MemoryConfig",
    "MemoryTrait",
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
    # Training (adapter manifest / schema resolution)
    "ManifestNotFoundError",
    "TrainingConfig",
    "TrainingTrait",
]
