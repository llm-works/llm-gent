# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Built-in traits for composable agent capabilities."""

from .conversation import ConversationTrait, ConversationTraitConfig
from .directive import Directive, DirectiveTrait, MethodTrait
from .http import HTTPConfig, HTTPTrait
from .learn import LearnConfig, LearnTrait
from .llm import LLMConfig, LLMTrait
from .rating import RatingConfig, RatingTrait
from .saia import SAIAConfig, SAIATrait
from .storage import StorageTrait
from .tools import ToolsTrait


__all__ = [
    # Conversation
    "ConversationTrait",
    "ConversationTraitConfig",
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
