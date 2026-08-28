# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Directive and method traits for agents.

Directive: "Why I exist" - the agent's core purpose and instructions.
Method: "How I operate" - the agent's operational approach, which can evolve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..base import BaseTrait


if TYPE_CHECKING:
    from ...agent import Agent


class Directive(BaseModel):
    """Agent's directive - why it exists and what it's trying to accomplish.

    A directive defines the agent's core purpose and instructions, injected into
    every prompt. The prompt is the source of truth for agent behavior.

    Future: structured constraints and success signals will be added
    for enforcement and measurement. Design TBD based on concrete use cases.

    Example YAML config:
        directive: |
            You are a code reviewer. Your goal is to catch all issues in one pass.

            Be critical and evidence-based. Verify before flagging.
            No false positives. No nitpicks that waste developer time.

    Or with the full object syntax:
        directive:
          prompt: |
            You are a code reviewer...
    """

    prompt: str
    """The directive prompt - injected into system prompt."""

    # Placeholder for future structured behavior (constraints, signals, etc.)
    # Design TBD based on concrete use cases like code review agent.
    extensions: dict[str, Any] = Field(default_factory=dict)
    """Reserved for future structured constraints/signals."""


class DirectiveTrait(BaseTrait):
    """Trait for agents that have a directive.

    Holds the agent's directive and injects it into prompts.
    Directive is immutable - it defines why the agent exists.

    Usage:
        directive = Directive(
            prompt="You are a code reviewer. Be critical and evidence-based."
        )
        trait = DirectiveTrait(agent, directive)
        agent.add_trait(trait)

    Or with a simple string:
        trait = DirectiveTrait(agent, "You are a code reviewer.")
        agent.add_trait(trait)
    """

    def __init__(self, agent: Agent, directive: Directive | str) -> None:
        """Initialize directive trait.

        Args:
            agent: The agent this trait belongs to.
            directive: The agent's directive (Directive object or prompt string).
        """
        super().__init__(agent)
        if isinstance(directive, str):
            directive = Directive(prompt=directive)
        self._directive = directive

    @property
    def directive(self) -> Directive:
        """The agent's directive."""
        return self._directive

    def build_prompt(self, base_prompt: str) -> str:
        """Build system prompt with directive injected.

        Args:
            base_prompt: The base system prompt.

        Returns:
            System prompt with directive prepended.
        """
        return f"{self._directive.prompt}\n\n{base_prompt}"


class MethodTrait(BaseTrait):
    """Trait for agents that have an operational method.

    Holds the agent's method - how it operates. Unlike directive, method
    can evolve over time based on learning and feedback.

    Usage:
        trait = MethodTrait(
            agent,
            "- Read the full diff before commenting\\n"
            "- Prioritize security over style\\n"
            "- Be concise but thorough"
        )
        agent.add_trait(trait)

        # Later, update based on learning
        trait.update("- Read the full diff before commenting\\n"
                     "- Prioritize security over style\\n"
                     "- Be concise but thorough\\n"
                     "- Always check for SQL injection")
    """

    def __init__(self, agent: Agent, method: str) -> None:
        """Initialize method trait.

        Args:
            agent: The agent this trait belongs to.
            method: The agent's operational method.
        """
        super().__init__(agent)
        self._method = method

    @property
    def method(self) -> str:
        """The agent's current method."""
        return self._method

    def build_prompt(self, base_prompt: str) -> str:
        """Build system prompt with method injected.

        Args:
            base_prompt: The base system prompt.

        Returns:
            System prompt with method appended.
        """
        return f"{base_prompt}\n\n## Method\n{self._method}"

    def update(self, new_method: str) -> None:
        """Update method based on learning.

        Args:
            new_method: The new operational method.
        """
        self._method = new_method
