# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tools trait for agent tool capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...tools.base import Tool
from ...tools.registry import Registry
from ..base import BaseTrait


if TYPE_CHECKING:
    from ...agent import Agent


class ToolsTrait(BaseTrait):
    """Tool capability trait.

    Manages a registry of tools available to the agent. Tools can be
    registered after creation.

    Example:
        from llm_gent.core.traits import ToolsTrait
        from llm_gent.core.tools import ShellTool, FileReadTool

        agent = Agent(lg, config)
        tools_trait = ToolsTrait(agent)
        tools_trait.register(ShellTool())
        tools_trait.register(FileReadTool())

        agent.add_trait(tools_trait)
        agent.start()

        # Agent can now use tools during execute()
    """

    def __init__(self, agent: Agent) -> None:
        """Initialize tools trait.

        Args:
            agent: The agent this trait belongs to.
        """
        super().__init__(agent)
        self._registry = Registry()

    def register(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        self._registry.register(tool)

    def unregister(self, name: str) -> None:
        """Unregister a tool by name.

        Args:
            name: Name of the tool to unregister.

        Raises:
            KeyError: If no tool with that name is registered.
        """
        self._registry.unregister(name)

    @property
    def registry(self) -> Registry:
        """Access the tool registry."""
        return self._registry

    def has_tools(self) -> bool:
        """Check if any tools are registered.

        Returns:
            True if at least one tool is registered.
        """
        return len(self._registry) > 0
