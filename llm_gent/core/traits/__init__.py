# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Agent traits for composable capabilities."""

from enum import StrEnum

from .base import BaseTrait, ScopedTrait, Trait, TraitBinding
from .builtin import (
    CostTrait,
    Directive,
    DirectiveTrait,
    HTTPConfig,
    HTTPTrait,
    ManifestNotFoundError,
    MemoryBinding,
    MemoryConfig,
    MemoryTrait,
    MethodTrait,
    RatingBinding,
    RatingConfig,
    RatingTrait,
    SAIAConfig,
    SAIATrait,
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

    COST = "cost"
    DIRECTIVE = "directive"
    MEMORY = "memory"
    HTTP = "http"
    RATING = "rating"
    SAIA = "saia"
    TOOLS = "tools"
    METHOD = "method"
    TRAINING = "training"


# All trait types available in the platform
ALL_TRAITS: list[TraitName] = [
    TraitName.COST,
    TraitName.DIRECTIVE,
    TraitName.MEMORY,
    TraitName.HTTP,
    TraitName.RATING,
    TraitName.SAIA,
    TraitName.TOOLS,
    TraitName.METHOD,
    TraitName.TRAINING,
]


__all__ = [
    # Base
    "BaseTrait",
    "ScopedTrait",
    "Trait",
    "TraitBinding",
    # Names & Catalogs
    "TraitName",
    "ALL_TRAITS",
    # Factory & Registry
    "Factory",
    "Registry",
    # Cost
    "CostTrait",  # cq: keep re-exports .builtin's list
    # Directive/Method
    "Directive",
    "DirectiveTrait",
    "MethodTrait",
    # HTTP
    "HTTPConfig",
    "HTTPTrait",
    # Memory
    "MemoryBinding",
    "MemoryConfig",
    "MemoryTrait",
    # Rating
    "RatingBinding",
    "RatingConfig",
    "RatingTrait",
    # SAIA
    "SAIAConfig",
    "SAIATrait",
    # Tools
    "ToolsTrait",
    # Training (adapter manifest / schema resolution)
    "ManifestNotFoundError",
    "TrainingConfig",
    "TrainingTrait",
]
