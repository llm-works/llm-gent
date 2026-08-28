# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tool registry for managing available tools."""

from __future__ import annotations

from typing import Any

from .base import Tool


class Registry:
    """Registry of tools available to an agent.

    Manages tool registration and provides tools in formats needed
    by LLM backends (e.g., OpenAI function calling format).

    Example:
        registry = Registry()
        registry.register(shell_tool)
        registry.register(file_tool)

        # Get tools for LLM
        tools_for_llm = registry.to_openai_tools()

        # Execute a tool by name
        tool = registry.get("shell")
        result = tool.execute(command="ls -la")
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name.

        Args:
            name: Name of the tool to unregister.

        Raises:
            KeyError: If no tool with that name is registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
        del self._tools[name]

    def get(self, name: str) -> Tool | None:
        """Get a tool by name.

        Args:
            name: Name of the tool.

        Returns:
            Tool instance or None if not found.
        """
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names.
        """
        return list(self._tools.keys())

    def list_tools(self) -> list[Tool]:
        """List all registered tools.

        Returns:
            List of tool instances.
        """
        return list(self._tools.values())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Convert all tools to OpenAI function calling format.

        Returns:
            List of tool definitions in OpenAI format.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
