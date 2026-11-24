# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Core exceptions for llm-gent."""


class AgentError(Exception):
    """Base exception for all llm-gent errors."""


class ConfigError(AgentError):
    """Configuration is missing or invalid."""


class SessionNotFoundError(AgentError):
    """The agent has no session with the given id: it never issued it, or has forgotten it."""


class TraitError(AgentError):
    """Base for trait-related errors."""


class TraitNotFoundError(TraitError):
    """Required trait not found in registry."""


class DuplicateTraitError(TraitError):
    """Trait of this type already added to agent."""
